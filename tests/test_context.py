import json

import pytest

from pipeline.annotate import enumerate_units, run_pass
from pipeline.errors import SchemaInvalidError
from pipeline.providers.fake import FakeProvider
from pipeline.registry import PREPARED_IDX, EpisodeRow, WorkUnit, make_unit_id
from pipeline.stage_b.context import build_context, format_turn
from tests.conftest import needs_ffmpeg
from tests.stage_b_helpers import PASSES, VIDEO, make_project, segmentation_payload

# The real segmentation/episode templates arrive in later tasks; these stand-ins let the
# config load. They are never run here — build_context is exercised directly. The
# transcript template no longer exists (Revision R1), so stand-ins reuse stage_b_timeline.j2,
# and ConsistencyRecord was deleted (Task 14 deferred), so stand-ins reuse SegmentationRecord.
SEG = {**PASSES["segmentation"], "prompt": "stage_b_timeline.j2"}
EPI = {"name": "episode", "unit": "EPISODE", "prompt": "stage_b_timeline.j2",
       "schema": "pipeline.schemas.stage_b.SegmentationRecord", "depends_on": "segmentation",
       "context": ["roster", "timeline_slice", "pitch_summary"]}
CON = {"name": "consistency", "unit": "PITCH", "media": "none", "prompt": "stage_b_timeline.j2",
       "schema": "pipeline.schemas.stage_b.SegmentationRecord", "depends_on": "segmentation",
       "context": ["timeline", "episode_table"]}

ROWS = [
    EpisodeRow(VIDEO, PREPARED_IDX, 0.0, 4.0, kind="prepared", start_turn=0, end_turn=1),
    EpisodeRow(VIDEO, 0, 4.0, 7.0, probing_speaker="speaker-2", start_turn=2, end_turn=4,
               cut=False, topic="gross margins", resolution="resolved"),
    EpisodeRow(VIDEO, 1, 7.0, 12.0, probing_speaker="speaker-3", start_turn=5, end_turn=7,
               cut=False, topic="valuation", resolution="conceded"),
]


def fake_done(registry, run, tmp_path, pass_name, payload):
    """Register a finished upstream unit without running a pass."""
    uid = make_unit_id(video_id=VIDEO, pass_name=pass_name, prompt_hash="p", model="m",
                       media="full", k=0)
    registry.enqueue(WorkUnit(unit_id=uid, run_id=run.run_id, video_id=VIDEO, pass_name=pass_name,
                              prompt_hash="p", model="m", media="full", k=0))
    parsed = tmp_path / f"{pass_name}.json"
    parsed.write_text(json.dumps({"record": payload}))
    registry.mark_done(uid, "raw", str(parsed), 1, 1)
    return uid


@pytest.fixture
def staged(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", SEG, EPI, CON])
    run_pass(cfg, cfg.get_pass("timeline"), registry, run, FakeProvider(fixtures), progress=False)
    fake_done(registry, run, tmp_path, "segmentation", segmentation_payload())
    registry.replace_unfrozen_episodes(VIDEO, ROWS)
    registry.set_frozen(VIDEO, True)
    yield cfg, registry, run
    registry.close()


def unit_for(episode_idx=None):
    return WorkUnit(unit_id="u", run_id="r", video_id=VIDEO, pass_name="x", prompt_hash="p",
                    model="m", media="full", k=0, episode_idx=episode_idx)


def test_turn_format_is_fixed():
    turn = {"turn_idx": 12, "speaker": "speaker-2", "start_s": 204.0, "end_s": 211.0,
            "tag": "asks about the patent"}
    assert format_turn(turn, {"speaker-2": "investor"}) == \
        "[t12 3:24–3:31] speaker-2 (investor): asks about the patent"
    assert format_turn(turn, {}, offset=3.0).startswith("[t12 3:21–3:28] speaker-2 (other)")


@needs_ffmpeg
def test_roster_and_full_timeline(staged):
    cfg, registry, _ = staged
    ctx = build_context(cfg, cfg.get_pass("consistency"), registry, unit_for())
    assert set(ctx) == {"timeline", "episode_table"}
    lines = ctx["timeline"].splitlines()
    assert len(lines) == 8
    assert lines[2] == "[t2 0:04–0:05] speaker-2 (investor): asks about margins"
    assert "prepared pitch: turns 0–1" in ctx["episode_table"]
    assert "episode 1: turns 5–7, probed by speaker-3, topic: valuation" in ctx["episode_table"]


@needs_ffmpeg
def test_slice_is_limited_to_the_episode_and_rebased_to_the_clip(staged):
    cfg, registry, _ = staged
    ctx = build_context(cfg, cfg.get_pass("episode"), registry, unit_for(episode_idx=1))
    lines = ctx["timeline_slice"].splitlines()
    assert lines[0] == "[t5 0:00–0:02] speaker-3 (investor): challenges the valuation"
    assert len(lines) == 3 and lines[-1].startswith("[t7 0:03–0:05]")  # global idx, clip-relative time
    assert "speaker-1 — founder — standing centre" in ctx["roster"]
    assert "folding kettle" in ctx["pitch_summary"] and "valuation" in ctx["pitch_summary"]


@needs_ffmpeg
def test_missing_upstream_names_the_pass_to_run(tmp_path):
    cfg, registry, _, _ = make_project(tmp_path, ["timeline", SEG, EPI, CON])
    with pytest.raises(SchemaInvalidError, match="--pass timeline"):
        build_context(cfg, cfg.get_pass("segmentation"), registry, unit_for())
    registry.close()


@needs_ffmpeg
def test_null_turn_episode_row_makes_build_context_raise(staged):
    cfg, registry, _ = staged
    registry.set_frozen(VIDEO, False)
    bad_rows = [
        ROWS[0], ROWS[1],
        EpisodeRow(VIDEO, 1, 7.0, 12.0, probing_speaker="speaker-3", start_turn=None,
                   end_turn=None, cut=False, topic="valuation", resolution="conceded"),
    ]
    registry.replace_unfrozen_episodes(VIDEO, bad_rows)
    registry.set_frozen(VIDEO, True)
    with pytest.raises(SchemaInvalidError, match=f"{VIDEO}.*episode 1"):
        build_context(cfg, cfg.get_pass("episode"), registry, unit_for(episode_idx=1))
    registry.close()


@needs_ffmpeg
def test_pitch_summary_reports_the_ep_cut_flag_when_known(staged):
    cfg, registry, _ = staged
    ctx = build_context(cfg, cfg.get_pass("episode"), registry, unit_for(episode_idx=1))
    assert "Editing cut inside this episode (EP-CUT): no" in ctx["pitch_summary"]

    registry.set_frozen(VIDEO, False)
    cut_rows = [
        ROWS[0], ROWS[1],
        EpisodeRow(VIDEO, 1, 7.0, 12.0, probing_speaker="speaker-3", start_turn=5, end_turn=7,
                   cut=True, topic="valuation", resolution="conceded"),
    ]
    registry.replace_unfrozen_episodes(VIDEO, cut_rows)
    registry.set_frozen(VIDEO, True)
    ctx = build_context(cfg, cfg.get_pass("episode"), registry, unit_for(episode_idx=1))
    assert "Editing cut inside this episode (EP-CUT): yes" in ctx["pitch_summary"]
    registry.close()


@needs_ffmpeg
def test_units_appear_only_when_upstream_is_done_and_track_its_identity(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", SEG, EPI, CON])
    seg = cfg.get_pass("segmentation")
    assert enumerate_units(cfg, seg, registry, run, "p") == []
    run_pass(cfg, cfg.get_pass("timeline"), registry, run, FakeProvider(fixtures), progress=False)
    first = enumerate_units(cfg, seg, registry, run, "p")
    assert len(first) == 1

    # A newer timeline (say, after a prompt edit) must re-enqueue what was built on it.
    fake_done(registry, run, tmp_path, "timeline", {"video_length": "0:12", "roster": [], "turns": []})
    assert enumerate_units(cfg, seg, registry, run, "p")[0].unit_id != first[0].unit_id

    # Episode units need frozen boundaries; the consistency pass needs them for its table.
    fake_done(registry, run, tmp_path, "segmentation", segmentation_payload())
    registry.replace_unfrozen_episodes(VIDEO, ROWS)
    assert enumerate_units(cfg, cfg.get_pass("episode"), registry, run, "p") == []
    assert enumerate_units(cfg, cfg.get_pass("consistency"), registry, run, "p") == []
    registry.set_frozen(VIDEO, True)
    episode_units = enumerate_units(cfg, cfg.get_pass("episode"), registry, run, "p")
    assert [(u.episode_idx, u.span) for u in episode_units] == [(0, (4.0, 7.0)), (1, (7.0, 12.0))]
    assert len(enumerate_units(cfg, cfg.get_pass("consistency"), registry, run, "p")) == 1
    registry.close()
