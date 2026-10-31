import csv
import json
from pathlib import Path

import pytest

from pipeline.config import CorpusConfig, LimitsConfig, ModelConfig, PassConfig, PipelineConfig
from pipeline.registry import PREPARED_IDX, EpisodeRow, Registry, VideoRow, WorkUnit, make_unit_id
from pipeline.stage_b.freeze import COLUMNS, export_review, freeze, import_review, unfreeze
from tests.stage_b_helpers import timeline_payload


def fake_timeline(registry, tmp_path, video_id):
    """Register a finished timeline unit without running a pass (copied from test_context.fake_done)."""
    uid = make_unit_id(video_id=video_id, pass_name="timeline", prompt_hash="p", model="m",
                        media="full", k=0)
    registry.enqueue(WorkUnit(unit_id=uid, run_id="r", video_id=video_id, pass_name="timeline",
                              prompt_hash="p", model="m", media="full", k=0))
    parsed = tmp_path / f"timeline_{video_id}.json"
    parsed.write_text(json.dumps({"record": timeline_payload()}))
    registry.mark_done(uid, "raw", str(parsed), 1, 1)


@pytest.fixture
def reg(tmp_path):
    registry = Registry(tmp_path / "r.sqlite")
    for vid in ("v1", "v2"):
        registry.upsert_video(VideoRow(vid, "st", f"/{vid}.mp4", probe_ok=True,
                                       meta_json='{"title": "Kettle | Shark Tank"}'))
        registry.replace_unfrozen_episodes(vid, [
            EpisodeRow(vid, PREPARED_IDX, 0.0, 4.0, kind="prepared", start_turn=0, end_turn=1),
            EpisodeRow(vid, 0, 4.0, 7.0, probing_speaker="speaker-2", start_turn=2, end_turn=4,
                       cut=False, topic="margins", resolution="resolved"),
            EpisodeRow(vid, 1, 7.0, 12.0, probing_speaker="speaker-3", start_turn=5, end_turn=7,
                       cut=True, topic="valuation", resolution="conceded"),
        ])
        fake_timeline(registry, tmp_path, vid)
    yield registry
    registry.close()


@pytest.fixture
def cfg():
    return PipelineConfig(
        run_name="t",
        corpus=CorpusConfig(name="st", video_dir=Path("."), metadata_csv=Path(".")),
        model=ModelConfig(name="m"),
        limits=LimitsConfig(),
        passes=(
            PassConfig(name="timeline", unit="PITCH", prompt="stage_b_timeline.j2",
                       schema="pipeline.schemas.stage_b_timeline.TimelineRecord",
                       produces="timeline"),
        ),
        codebook=None, source_path=Path("cfg.yaml"), prompt_dir=Path("."), raw_yaml="",
    )


def read(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_export_lists_every_proposed_row(reg, tmp_path):
    out = tmp_path / "review.csv"
    assert export_review(reg, "st", out) == 6
    rows = read(out)
    assert list(rows[0]) == COLUMNS
    assert rows[0]["kind"] == "prepared" and rows[0]["title"] == "Kettle | Shark Tank"
    assert (rows[2]["topic"], rows[2]["cut"], rows[2]["frozen"]) == ("valuation", "True", "False")


def test_freeze_without_edits_freezes_everything(reg):
    assert freeze(reg, "st") == 2
    assert len(reg.episodes_for("v1", kind=None)) == 3
    assert freeze(reg, "st") == 0  # idempotent


def test_human_edits_are_imported_before_freezing(reg, tmp_path, cfg):
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    rows = [r for r in read(out) if not (r["video_id"] == "v1" and r["episode_idx"] == "1")]
    edited = next(r for r in rows if r["video_id"] == "v1" and r["episode_idx"] == "0")
    edited.update(end_s="12.0", end_turn="7", topic="margins and valuation")
    write(out, rows)

    assert import_review(reg, out, cfg) == (2, 0)
    assert freeze(reg, "st", video_ids=["v1"]) == 1
    episodes = reg.episodes_for("v1")
    assert [(e.episode_idx, e.end_s, e.end_turn, e.topic) for e in episodes] == [
        (0, 12.0, 7, "margins and valuation")]
    assert reg.episodes_for("v2") == []  # not frozen: not selected


def test_edited_turn_range_rederives_seconds_from_the_timeline(reg, tmp_path, cfg):
    """A corrected turn range must re-derive start_s/end_s; stale CSV seconds are ignored."""
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    rows = read(out)
    for row in rows:
        if row["video_id"] == "v1" and row["episode_idx"] == "0":
            # Deliberately wrong seconds in the CSV: they must be ignored on import.
            row.update(start_turn="1", end_turn="3", start_s="999", end_s="999")
    write(out, rows)

    assert import_review(reg, out, cfg) == (2, 0)
    ep = reg.episodes_for("v1", frozen_only=False, kind="episode")[0]
    assert (ep.start_turn, ep.end_turn) == (1, 3)
    # turn 1 starts at 2.0s; turns 1-3 end at 4.0, 5.0, 6.0 -> max 6.0
    assert (ep.start_s, ep.end_s) == (2.0, 6.0)


def test_import_never_touches_frozen_videos(reg, tmp_path, cfg):
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    freeze(reg, "st", video_ids=["v1"])
    assert import_review(reg, out, cfg) == (1, 1)
    assert unfreeze(reg, ["v1"]) == 1
    assert import_review(reg, out, cfg) == (2, 0)


@pytest.mark.parametrize("change, message", [
    ({"kind": "scene"}, "line 3.*kind"),
    ({"episode_idx": "x"}, "line 3"),
    ({"start_turn": ""}, "line 3.*start_turn"),
    ({"end_turn": ""}, "line 3.*end_turn"),
    ({"start_turn": "5", "end_turn": "2"}, "line 3.*start_turn.*end_turn"),
    ({"probing_speaker": "Deborah"}, "line 3.*probing_speaker"),
])
def test_bad_rows_name_their_line(reg, tmp_path, cfg, change, message):
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    rows = read(out)
    rows[1].update(change)  # file line 3: header is line 1
    write(out, rows)
    with pytest.raises(ValueError, match=message):
        import_review(reg, out, cfg)
    assert reg.episodes_for("v1", frozen_only=False)[0].end_s == 7.0  # nothing was applied


def test_turn_past_the_timeline_is_rejected(reg, tmp_path, cfg):
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    rows = read(out)
    rows[1]["end_turn"] = "40"  # v1 episode 0: file line 3
    write(out, rows)
    with pytest.raises(ValueError, match=r"line 3.*40.*8 turns"):
        import_review(reg, out, cfg)
    assert reg.episodes_for("v1", frozen_only=False)[0].end_s == 7.0  # nothing was applied


def test_duplicate_episode_idx_names_the_csv_lines(reg, tmp_path, cfg):
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    rows = read(out)
    dup = dict(rows[1])  # v1 episode_idx 0 already at line 3; duplicate it
    write(out, rows + [dup])
    with pytest.raises(ValueError, match=r"v1.*duplicate episode_idx 0.*line"):
        import_review(reg, out, cfg)


def test_import_without_a_timeline_producing_pass_is_refused(reg, tmp_path):
    bad_cfg = PipelineConfig(
        run_name="t",
        corpus=CorpusConfig(name="st", video_dir=Path("."), metadata_csv=Path(".")),
        model=ModelConfig(name="m"), limits=LimitsConfig(), passes=(
            PassConfig(name="segmentation", unit="PITCH", prompt="stage_b_timeline.j2",
                       schema="pipeline.schemas.stage_b.SegmentationRecord"),
        ),
        codebook=None, source_path=Path("cfg.yaml"), prompt_dir=Path("."), raw_yaml="",
    )
    out = tmp_path / "review.csv"
    export_review(reg, "st", out)
    with pytest.raises(ValueError, match="produces: timeline"):
        import_review(reg, out, bad_cfg)
