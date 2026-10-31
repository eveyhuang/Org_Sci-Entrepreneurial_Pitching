import json
import textwrap
from pathlib import Path

import pytest

from pipeline.annotate import enumerate_units, run_pass
from pipeline.config import load_config
from pipeline.errors import ProviderError, RunFatalError
from pipeline.ingest import ingest_corpus
from pipeline.providers.fake import FakeProvider
from pipeline.registry import Registry
from pipeline.runs import create_or_resume_run
from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES
from tests.conftest import needs_ffmpeg

CFG_BODY = """
    run_name: t
    corpus: {{name: hu-ma, video_dir: videos, metadata_csv: meta.csv}}
    model: {{name: gemini-x}}
    limits: {{max_concurrency: 2, requests_per_minute: 6000, max_attempts: 3, backoff_base_s: 0.001, backoff_cap_s: 0.002}}
    passes:
      - {{name: holistic, unit: PITCH, prompt: stage_a_perceived_delivery.j2,
          schema: pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord, k: {k}, media: full}}
    codebook: {codebook}
"""
BOOK = Path("pipeline/codebooks/stage_a_perceived_delivery.yaml")


def rating():
    return {"value": 4, "uncodable": False, "evidence": "Steady, warm delivery at 0:01."}


@pytest.fixture
def project(tmp_path, video_factory):
    video_factory("abc.mp4", seconds=2.0)
    video_factory("def.mp4", seconds=2.0)
    (tmp_path / "meta.csv").write_text(
        "video_id,platform,url\nabc,youtube,https://a\ndef,youtube,https://d\n"
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    payload = {name: rating() for name in STAGE_A_DELIVERY_CODES} | {"overall_notes": "n"}
    for vid in ("abc", "def"):
        (fixtures / f"holistic__{vid}.json").write_text(json.dumps(payload))
    return tmp_path, fixtures


def build(tmp_path, k=1):
    codebook_abs = BOOK.resolve()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent(CFG_BODY).format(k=k, codebook=codebook_abs))
    cfg = load_config(cfg_path)
    registry = Registry(tmp_path / "registry.sqlite")
    ingest_corpus(cfg, registry)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs")
    return cfg, registry, run


@needs_ffmpeg
def test_enumerates_one_unit_per_video_per_replicate(project):
    tmp_path, _ = project
    cfg, registry, run = build(tmp_path, k=3)
    units = enumerate_units(cfg, cfg.passes[0], registry, run, prompt_hash="p1")
    assert len(units) == 6  # 2 videos x k=3
    assert {u.k for u in units} == {0, 1, 2}
    registry.close()


@needs_ffmpeg
def test_run_pass_writes_raw_and_parsed_and_marks_done(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    assert (summary.done, summary.failed) == (2, 0)
    raw = next((run.root / "raw" / "holistic").glob("abc__k0__*.json"))
    parsed = next((run.root / "parsed" / "holistic").glob("abc__k0__*.json"))
    assert raw.exists() and parsed.exists()
    record = json.loads(parsed.read_text())
    assert record["record"]["pitch_positivity"]["value"] == 4
    assert record["video_id"] == "abc"
    assert record["model_version"] == "fake-model-v1"
    assert registry.status_counts(run.run_id)["done"] == 2
    registry.close()


@needs_ffmpeg
def test_rerun_skips_completed_work(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    second = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    assert (second.done, second.skipped) == (0, 2)
    assert len(provider.calls) == 2  # no new API calls on the second pass
    registry.close()


@needs_ffmpeg
def test_prompt_change_invalidates_units(project, monkeypatch):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)

    import pipeline.annotate as annotate_module

    monkeypatch.setattr(
        annotate_module, "render_pass_prompt", lambda *a, **k: ("EDITED PROMPT", "hash2")
    )
    third = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    assert third.done == 2  # new prompt hash => new units
    registry.close()


@needs_ffmpeg
def test_item_failure_is_recorded_and_run_continues(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    (fixtures / "holistic__def.json").unlink()
    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)
    assert (summary.done, summary.failed) == (1, 1)
    assert sum(registry.failure_counts(run.run_id).values()) == 1
    registry.close()


@needs_ffmpeg
def test_retry_failed_reprocesses_only_failures(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    payload = (fixtures / "holistic__abc.json").read_text()
    (fixtures / "holistic__def.json").unlink()
    run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    (fixtures / "holistic__def.json").write_text(payload)
    provider = FakeProvider(fixtures)
    summary = run_pass(
        cfg, cfg.passes[0], registry, run, provider, retry_failed=True, progress=False
    )
    assert summary.done == 1
    assert provider.calls == ["holistic__def__k0"]
    registry.close()


@needs_ffmpeg
def test_transient_error_is_retried_then_succeeds(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(
        fixtures, error_plan={"holistic__abc__k0": [ProviderError("503", status=503)]}
    )
    summary = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    assert (summary.done, summary.failed) == (2, 0)
    registry.close()


@needs_ffmpeg
def test_run_fatal_stops_the_pass(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(
        fixtures, error_plan={k: [RunFatalError("403")] for k in ("holistic__abc__k0", "holistic__def__k0")}
    )
    with pytest.raises(RunFatalError):
        run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    registry.close()


@needs_ffmpeg
def test_video_ids_and_limit_subset_the_work(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(fixtures)
    summary = run_pass(
        cfg, cfg.passes[0], registry, run, provider, video_ids=["abc"], progress=False
    )
    assert summary.done == 1
    assert provider.calls == ["holistic__abc__k0"]
    registry.close()


@needs_ffmpeg
def test_failed_unit_is_skipped_on_a_later_plain_run(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    payload = (fixtures / "holistic__abc.json").read_text()
    (fixtures / "holistic__def.json").unlink()

    first = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)
    assert (first.done, first.failed) == (1, 1)

    # Restore the fixture, but run WITHOUT retry_failed: the failed unit must be
    # skipped, not silently re-attempted, and the done unit must stay skipped too.
    (fixtures / "holistic__def.json").write_text(payload)
    provider = FakeProvider(fixtures)
    second = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)

    assert (second.done, second.failed, second.skipped) == (0, 0, 2)
    assert provider.calls == []          # no API call for either unit
    assert registry.status_counts(run.run_id) == {"done": 1, "failed": 1}
    registry.close()


@needs_ffmpeg
def test_schema_invalid_response_persists_its_raw_body(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    (fixtures / "holistic__def.json").write_text('{"pitch_positivity": {"value": 99}}')
    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    assert summary.failed == 1
    unit = next(u for u in registry.pending_units(run.run_id, include_failed=True)
                if u.video_id == "def")
    assert unit.error_class == "ITEM_FATAL"
    assert unit.raw_path and Path(unit.raw_path).exists()
    assert "pitch_positivity" in Path(unit.raw_path).read_text()
    registry.close()


@needs_ffmpeg
def test_prompt_change_preserves_earlier_raw_and_parsed_files(project, monkeypatch):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    provider = FakeProvider(fixtures)
    run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    before = sorted(p.name for p in (run.root / "parsed" / "holistic").iterdir())
    assert before

    import pipeline.annotate as annotate_module
    monkeypatch.setattr(
        annotate_module, "render_pass_prompt", lambda *a, **k: ("EDITED PROMPT", "hash2")
    )
    run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    after = sorted(p.name for p in (run.root / "parsed" / "holistic").iterdir())

    # Old records must still be on disk alongside the new ones, not overwritten.
    assert set(before).issubset(set(after))
    assert len(after) == 2 * len(before)
    registry.close()


@needs_ffmpeg
def test_temperature_change_reenqueues_units(project):
    """Temperature never reaches the prompt, but it changes what the model is asked."""
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    hot_path = tmp_path / "cfg_hot.yaml"
    hot_path.write_text(
        textwrap.dedent(CFG_BODY).format(
            k=1, codebook=BOOK.resolve()
        ).replace("name: gemini-x", "name: gemini-x, temperature: 0.7")
    )
    hot = load_config(hot_path)
    assert hot.model.temperature == 0.7
    hot_run = create_or_resume_run(hot, runs_root=tmp_path / "runs")
    provider = FakeProvider(fixtures)
    summary = run_pass(hot, hot.passes[0], registry, hot_run, provider, progress=False)

    assert (summary.done, summary.skipped) == (2, 0)
    assert len(provider.calls) == 2
    registry.close()


@needs_ffmpeg
def test_episode_pass_writes_the_episode_index(project):
    tmp_path, fixtures = project
    episode_cfg = (
        textwrap.dedent(CFG_BODY)
        .format(k=1, codebook=BOOK.resolve())
        .replace("unit: PITCH", "unit: EPISODE")
    )
    cfg_path = tmp_path / "cfg_ep.yaml"
    cfg_path.write_text(episode_cfg)
    cfg = load_config(cfg_path)
    registry = Registry(tmp_path / "registry.sqlite")
    ingest_corpus(cfg, registry)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs")

    registry.upsert_episode("abc", 0, 0.0, 1.0, frozen=True)
    registry.upsert_episode("abc", 1, 1.0, 2.0, frozen=True)

    summary = run_pass(
        cfg, cfg.passes[0], registry, run, FakeProvider(fixtures),
        video_ids=["abc"], progress=False,
    )
    assert summary.done == 2

    records = [
        json.loads(p.read_text()) for p in sorted((run.root / "parsed" / "holistic").iterdir())
    ]
    assert sorted(r["episode_idx"] for r in records) == [0, 1]
    # Each episode must be a distinct unit with its own span, not the same clip twice.
    assert sorted(tuple(r["span"]) for r in records) == [(0.0, 1.0), (1.0, 2.0)]
    registry.close()


@needs_ffmpeg
def test_missing_video_row_fails_the_unit_with_a_clear_message(project, monkeypatch):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    monkeypatch.setattr(registry, "get_video", lambda video_id: None)
    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    assert (summary.done, summary.failed) == (0, 2)
    unit = registry.pending_units(run.run_id, include_failed=True)[0]
    assert unit.error_class == "ITEM_FATAL"
    assert "no such video" in unit.error_detail
    registry.close()


@needs_ffmpeg
def test_unparsed_response_fails_the_unit_and_keeps_its_body(project):
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)

    class NoParse(FakeProvider):
        def generate(self, request):
            response = super().generate(request)
            return type(response)(
                raw_text="{'almost': 'json'}", parsed=None,
                model_version=response.model_version,
            )

    summary = run_pass(cfg, cfg.passes[0], registry, run, NoParse(fixtures), progress=False)
    assert (summary.done, summary.failed) == (0, 2)
    unit = registry.pending_units(run.run_id, include_failed=True)[0]
    assert "no parsed record" in unit.error_detail
    assert unit.raw_path and "almost" in Path(unit.raw_path).read_text()
    registry.close()


@needs_ffmpeg
def test_run_archives_the_exact_prompt_it_sent(project):
    """A run must be self-describing: the manifest names the hash, this file is the text."""
    tmp_path, fixtures = project
    cfg, registry, run = build(tmp_path)
    summary = run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)

    archived = run.root / "prompts" / f"holistic__{summary.prompt_hash}.txt"
    assert archived.exists(), "the rendered prompt was not archived"

    from pipeline.codebook import load_codebook
    from pipeline.prompts import prompt_hash, render_pass_prompt

    expected, _ = render_pass_prompt(cfg, cfg.passes[0], load_codebook(cfg.codebook))
    assert archived.read_text() == expected
    assert prompt_hash(archived.read_text()) == summary.prompt_hash
    registry.close()
