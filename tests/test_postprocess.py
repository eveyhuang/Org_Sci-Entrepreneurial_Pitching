import json

from pipeline.annotate import run_pass
from pipeline.providers.fake import FakeProvider
from pipeline.registry import PREPARED_IDX, WorkUnit
from pipeline.runs import timeline_path
from pipeline.schemas.stage_b import SegmentationRecord
from pipeline.stage_b.postprocess import run_postprocessor
from tests.conftest import needs_ffmpeg
from tests.stage_b_helpers import VIDEO, make_project, segmentation_payload, timeline_payload


@needs_ffmpeg
def test_timeline_pass_writes_the_k0_timeline_file(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline"], k={"timeline": 2})
    # Make replicate 1 distinguishable: only k=0 may be written as THE timeline.
    other = timeline_payload()
    other["turns"][0]["tag"] = "REPLICATE ONE"
    (fixtures / f"timeline__{VIDEO}__k1.json").write_text(json.dumps(other))

    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)
    assert (summary.done, summary.failed) == (2, 0)
    written = json.loads(timeline_path(run, VIDEO).read_text())
    assert written["turns"][0]["tag"] == "greets the panel, names the product"
    assert [r["speaker"] for r in written["roster"]] == ["speaker-1", "speaker-2", "speaker-3"]
    registry.close()


def test_timeline_prompt_forbids_names_and_fixes_the_label_scheme():
    from pipeline.codebook import Codebook
    from pipeline.prompts import render_prompt
    from tests.stage_b_helpers import ARCHIVE_B1

    text = render_prompt("stage_b_timeline.j2", codebook=Codebook(entries=[]), prompt_dir=ARCHIVE_B1)
    assert "speaker-1" in text and "Never" in text and "names" in text
    assert "turn_idx" in text
    assert "Do not transcribe or quote any dialogue." in text


def run_two(tmp_path, k=None):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation"], k=k)
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        summary = run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    return cfg, registry, run, fixtures, provider, summary


@needs_ffmpeg
def test_segmentation_writes_unfrozen_rows_with_turn_derived_times(tmp_path):
    _, registry, _, _, _, summary = run_two(tmp_path)
    assert (summary.done, summary.failed) == (1, 0)
    assert registry.episodes_for(VIDEO, kind=None) == []  # proposed, not frozen
    rows = registry.episodes_for(VIDEO, frozen_only=False, kind=None)
    assert [(r.episode_idx, r.kind, r.start_s, r.end_s, r.start_turn, r.end_turn) for r in rows] == [
        (PREPARED_IDX, "prepared", 0.0, 4.0, 0, 1),   # the model said "0:04"; turn 1 ends at 4.0
        (0, "episode", 4.0, 7.0, 2, 4),                # the model said "0:05"-"0:06"
        (1, "episode", 7.0, 12.0, 5, 7),
    ]
    assert (rows[1].probing_speaker, rows[1].topic, rows[1].resolution, rows[1].cut) == (
        "speaker-2", "gross margins", "resolved", False)
    assert rows[1].source_unit_id is not None
    registry.close()


@needs_ffmpeg
def test_context_reaches_the_model_but_not_the_prompt_hash(tmp_path):
    cfg, registry, run, _, provider, _ = run_two(tmp_path)
    sent = next(r.prompt for r in provider.requests if r.unit_key.startswith("segmentation"))
    assert "[t2 0:04–0:05] speaker-2 (investor): asks about margins" in sent
    assert "speaker-1 — founder" in sent
    archived = next((run.root / "prompts").glob("segmentation__*.txt")).read_text()
    assert "asks about margins" not in archived and "EP-BOUND" in archived
    registry.close()


@needs_ffmpeg
def test_only_replicate_zero_writes_episodes(tmp_path):
    import json as _json
    cfg, registry, run, fixtures = make_project(
        tmp_path, ["timeline", "segmentation"], k={"segmentation": 2})
    other = segmentation_payload()
    other["episodes"] = other["episodes"][:1]
    (fixtures / f"segmentation__{VIDEO}__k1.json").write_text(_json.dumps(other))
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    assert len(registry.episodes_for(VIDEO, frozen_only=False)) == 2  # k=0's two, not k=1's one
    registry.close()


@needs_ffmpeg
def test_episode_pointing_past_the_timeline_fails_the_unit(tmp_path):
    import json as _json
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation"])
    bad = segmentation_payload()
    bad["episodes"][1]["end_turn"] = 40
    (fixtures / f"segmentation__{VIDEO}.json").write_text(_json.dumps(bad))
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.get_pass("timeline"), registry, run, provider, progress=False)
    summary = run_pass(cfg, cfg.get_pass("segmentation"), registry, run, provider, progress=False)
    assert (summary.done, summary.failed) == (0, 1)
    assert registry.episodes_for(VIDEO, frozen_only=False, kind=None) == []
    registry.close()


@needs_ffmpeg
def test_zero_pitches_writes_no_rows_and_sets_skip_reason(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation"])
    empty = {"n_pitches": 0, "prepared_end": "0:00", "summary": "no pitch here", "episodes": [],
             "decisions": [], "deal": False}
    (fixtures / f"segmentation__{VIDEO}.json").write_text(json.dumps(empty))
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.get_pass("timeline"), registry, run, provider, progress=False)
    summary = run_pass(cfg, cfg.get_pass("segmentation"), registry, run, provider, progress=False)
    assert (summary.done, summary.failed) == (1, 0)
    assert registry.episodes_for(VIDEO, frozen_only=False, kind=None) == []
    assert registry.get_video(VIDEO).skip_reason == "no_pitch"
    registry.close()


@needs_ffmpeg
def test_nonzero_pitches_clears_a_previous_no_pitch_skip_reason(tmp_path):
    # Exercises the post-processor directly: enumerate_units excludes skipped videos from
    # every pass (including segmentation itself), so reclaiming one is a manual re-run —
    # this is the unit that reclaim depends on.
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation"])
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.get_pass("timeline"), registry, run, provider, progress=False)
    registry.set_skip_reason(VIDEO, "no_pitch")

    record = SegmentationRecord.model_validate(segmentation_payload())
    unit = WorkUnit(unit_id="seg-manual", run_id=run.run_id, video_id=VIDEO,
                     pass_name="segmentation", prompt_hash="p", model="m", media="full", k=0)
    run_postprocessor(cfg, cfg.get_pass("segmentation"), run, registry, unit, record)

    assert registry.get_video(VIDEO).skip_reason is None
    assert len(registry.episodes_for(VIDEO, frozen_only=False, kind=None)) == 3
    registry.close()


@needs_ffmpeg
def test_segmentation_writes_decisions_and_deal(tmp_path):
    _, registry, _, _, _, summary = run_two(tmp_path)
    assert (summary.done, summary.failed) == (1, 0)
    decisions = registry.decisions_for(VIDEO)
    assert [(d.speaker, d.decision, d.turn, d.terms) for d in decisions] == [
        ("speaker-2", "offer", 4, "100k for ten percent"),
        ("speaker-3", "out", 7, None),
    ]
    assert decisions[0].source_unit_id is not None
    assert registry.get_video(VIDEO).deal is True
    registry.close()


@needs_ffmpeg
def test_only_replicate_zero_writes_decisions(tmp_path):
    import json as _json
    cfg, registry, run, fixtures = make_project(
        tmp_path, ["timeline", "segmentation"], k={"segmentation": 2})
    other = segmentation_payload()
    other["decisions"] = []
    other["deal"] = False
    (fixtures / f"segmentation__{VIDEO}__k1.json").write_text(_json.dumps(other))
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    # k=0's two decisions and deal=True survive; k=1's empty/false payload must not win.
    assert len(registry.decisions_for(VIDEO)) == 2
    assert registry.get_video(VIDEO).deal is True
    registry.close()


@needs_ffmpeg
def test_zero_pitches_also_clears_decisions_and_deal(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation"])
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    assert len(registry.decisions_for(VIDEO)) == 2  # sanity: the fixture wrote some

    empty = {"n_pitches": 0, "prepared_end": "0:00", "summary": "no pitch here", "episodes": [],
             "decisions": [], "deal": False}
    (fixtures / f"segmentation__{VIDEO}.json").write_text(json.dumps(empty))
    # Re-run through the post-processor directly (segmentation is already 'done'; the
    # unit itself would need a config edit to re-enqueue, which is out of scope here).
    record = SegmentationRecord.model_validate(empty)
    unit = WorkUnit(unit_id="seg-manual", run_id=run.run_id, video_id=VIDEO,
                     pass_name="segmentation", prompt_hash="p", model="m", media="full", k=0)
    run_postprocessor(cfg, cfg.get_pass("segmentation"), run, registry, unit, record)

    assert registry.decisions_for(VIDEO) == []
    assert registry.get_video(VIDEO).deal is None
    registry.close()


@needs_ffmpeg
def test_frozen_video_refuses_resegmentation_before_calling_the_model(tmp_path):
    cfg, registry, run, fixtures, _, _ = run_two(tmp_path)
    registry.set_frozen(VIDEO, True)
    # A prompt edit would enqueue a new segmentation unit; simulate with --retry on a new k.
    cfg2, registry2, run2, _ = make_project(tmp_path, ["timeline", "segmentation"], k={"segmentation": 2})
    registry2.close()  # same database file; keep using the first connection
    provider = FakeProvider(fixtures)
    summary = run_pass(cfg2, cfg2.get_pass("segmentation"), registry, run2, provider, progress=False)
    assert (summary.skipped, summary.failed) == (1, 1)
    assert provider.calls == []  # refused before spending anything
    unit = next(u for u in registry.pending_units(run2.run_id, include_failed=True))
    assert "frozen" in unit.error_detail and "--unfreeze" in unit.error_detail
    registry.close()
