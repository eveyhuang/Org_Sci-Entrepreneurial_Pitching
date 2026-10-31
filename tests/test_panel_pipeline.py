import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from pipeline.errors import ConfigError, ProviderError, RunFatalError, SchemaInvalidError
from pipeline.instrument import load_release
from pipeline.providers.fake import FakeProvider
from pipeline.schemas.stage_b_panel import PanelRecord, PanelWire
from pipeline.stage_b.panel_books import PANEL_TEMPLATE, load_panel_books
from pipeline.stage_b import panel_pipeline
from pipeline.stage_b.panel_pipeline import (
    derive, load_panel_config, plan_calls, prepared_window, revalidate, run_agreement, run_video, unit_key,
)
from tests.panel_fixtures import DURATION_S, synthetic_record, synthetic_wire, to_wire

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = REPO / "configs" / "stage_b_panel.yaml"
KEY_P = "panel__vid1__primary"
KEY_R = "panel__vid1__replicate"
NOSLEEP = lambda s: None  # noqa: E731


def _fixtures(tmp_path, primary=None, replicate=None):
    fx = tmp_path / "fx"
    fx.mkdir(exist_ok=True)
    # the provider returns wire form (clock strings); tests describe passes as normalized records
    (fx / f"{KEY_P}.json").write_text(json.dumps(to_wire(primary or synthetic_record())))
    (fx / f"{KEY_R}.json").write_text(json.dumps(to_wire(replicate or synthetic_record())))
    return fx


def _video(tmp_path):
    path = tmp_path / "vid1.mp4"
    path.write_bytes(b"x")
    return {"video_id": "vid1", "path": str(path), "duration_s": DURATION_S}


def _setup(tmp_path, error_plan=None, primary=None, replicate=None):
    release = load_release()
    books = load_panel_books(release)
    cfg = load_panel_config(CONFIG)
    provider = FakeProvider(_fixtures(tmp_path, primary, replicate), error_plan=error_plan)
    return provider, release, books, cfg, _video(tmp_path), tmp_path / "run"


def _go(tmp_path, **kw):
    provider, release, books, cfg, video, root = _setup(tmp_path, **kw)
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    return provider, result, root / "vid1"


def test_exactly_two_calls_one_upload_identical_prompts(tmp_path):
    provider, result, vdir = _go(tmp_path)
    assert provider.calls == [KEY_P, KEY_R]
    assert len(provider.uploads) == 1
    a, b = provider.requests
    assert a.prompt == b.prompt and a.media == b.media and a.model == b.model
    assert a.schema is PanelWire and b.schema is PanelWire and a.model.temperature == 0.7
    assert result["status"] == "complete" and result["generation_calls"] == 2
    for name in ("results/primary.json", "results/replicate.json", "agreement.json", "experiment.json",
                 "completion.json", "prompts/panel_full.txt", "prompts/panel_full.schema.json"):
        assert (vdir / name).exists(), name
    assert json.loads((vdir / "experiment.json").read_text())["annotation_context"] == "descriptive_full_context"
    lines = [json.loads(x) for x in (vdir / "calls.jsonl").read_text().splitlines()]
    assert len(lines) == 2 and all(x["ok"] and x["error_class"] is None for x in lines)
    stored = json.loads((vdir / "results/primary.json").read_text())
    assert stored["annotation_context"] == "descriptive_full_context"
    assert stored["time_format"] == "clock_on_wire_seconds_in_record"
    assert PanelRecord.model_validate(stored["record"]) == PanelRecord.model_validate(synthetic_record())
    assert json.loads((vdir / "raw/primary.json").read_text()) == synthetic_wire()
    schema = json.loads((vdir / "prompts/panel_full.schema.json").read_text())
    assert "start" in schema["properties"]["turns"]["items"]["properties"]
    assert "start_s" not in schema["properties"]["turns"]["items"]["properties"]
    assert "exchanges" in schema["properties"] and "events" not in schema["properties"]


def test_results_carry_the_prepared_window_and_derived_fields_outside_the_record(tmp_path):
    _, _, vdir = _go(tmp_path)
    for name in ("primary", "replicate"):
        stored = json.loads((vdir / f"results/{name}.json").read_text())
        assert stored["prepared_window"] == ["0:05-1:00"]
        assert stored["derived"] == EXPECTED_DERIVED
        assert "derived" not in stored["record"] and "prepared_window" not in stored["record"]
        assert list(stored) == ["pass", "unit_key", "annotation_context", "prompt_hash", "model_version",
                                "time_format", "prepared_window", "derived", "normalizations", "record"]
        assert stored["normalizations"] == []


EXPECTED_DERIVED = {
    "turn_segment": {"1": "presentation", "2": "qa", "3": "qa", "4": "qa", "5": "qa", "6": "qa", "7": "qa",
                     "8": "social", "9": "social", "10": "none", "11": "qa", "12": "qa", "13": "qa", "14": "qa",
                     "15": "qa", "16": "qa", "17": "negotiation", "18": "negotiation", "19": "negotiation"},
    "segment_seconds": {"presentation": 55.0, "qa": 80.0, "social": 5.0, "demonstration": 0.0, "negotiation": 12.0,
                        "none": 3.0},
    "exchange_counts": {"substantive": 2, "social": 1, "demonstration": 0, "negotiation": 1},
    "turn_frame": {"2": "prevention", "3": "prevention", "4": "prevention", "5": "prevention", "11": "prevention",
                   "12": "prevention", "14": "prevention", "15": "promotion"},
    "frame_match": {"3": "match", "5": "match", "12": "match", "15": "switch_to_promotion"},
    "other_concepts": [],
}


def test_derive_on_the_synthetic_record():
    books = load_panel_books(load_release())
    assert derive(PanelRecord.model_validate(synthetic_record()), books) == EXPECTED_DERIVED


def test_derive_uses_canonical_labels():
    data = synthetic_record()
    data["turns"][0]["function"] = "Presents"
    data["exchanges"][0]["kind"] = "Substantive"
    data["exchanges"][1]["kind"] = "SOCIAL"
    assert derive(PanelRecord.model_validate(data), load_panel_books(load_release())) == EXPECTED_DERIVED


def test_derive_segments_follow_function_then_exchange_kind():
    data = synthetic_record()
    turns = {t["turn_id"]: t for t in data["turns"]}
    turns[2]["function"] = "presents"                 # presents wins over the exchange it sits in
    turns[8]["exchange_id"] = 9                       # an exchange that is not recorded
    data["exchanges"][1]["kind"] = "demonstration"    # exchange 2: turn 9 only now
    data["exchanges"][3]["kind"] = "haggling"         # not a kind: its turns fall to none
    out = derive(PanelRecord.model_validate(data), load_panel_books(load_release()))
    seg = out["turn_segment"]
    assert seg["2"] == "presentation" and seg["8"] == "none" and seg["9"] == "demonstration"
    assert seg["17"] == seg["18"] == seg["19"] == "none"
    assert "2" not in out["turn_frame"] and "3" not in out["frame_match"]      # turn 2 is no longer a question
    assert out["segment_seconds"] == {"presentation": 59.0, "qa": 76.0, "social": 0.0, "demonstration": 2.0,
                                      "negotiation": 0.0, "none": 18.0}
    assert out["exchange_counts"] == {"substantive": 2, "social": 0, "demonstration": 1, "negotiation": 0,
                                      "haggling": 1}


def test_prepared_window_is_the_clock_form_of_the_prepared_spans():
    data = synthetic_record()
    data["prepared_spans"] = [{"start_s": 15.0, "end_s": 103.0}, {"start_s": 110.0, "end_s": 3725.0}]
    assert prepared_window(PanelRecord.model_validate(data)) == ["0:15-1:43", "1:50-1:02:05"]
    data["prepared_spans"] = []
    assert prepared_window(PanelRecord.model_validate(data)) == []


def test_disagreeing_passes_do_not_trigger_another_call(tmp_path):
    rep = synthetic_record()
    rep["turns"] = [t for t in rep["turns"] if t["turn_id"] != 13]
    provider, result, vdir = _go(tmp_path, replicate=rep)
    assert len(provider.calls) == 2 and result["status"] == "complete"
    assert json.loads((vdir / "agreement.json").read_text())["flags"]


def test_invalid_output_is_retried_once_then_succeeds(tmp_path):
    plan = {KEY_P: [SchemaInvalidError("MAX_TOKENS", raw_text="{")]}
    provider, result, vdir = _go(tmp_path, error_plan=plan)
    assert provider.calls == [KEY_P, KEY_P, KEY_R]
    assert (vdir / "raw/primary__attempt1_invalid.txt").read_text() == "{"
    assert result["status"] == "complete" and result["generation_calls"] == 3
    assert result["generation_calls_total"] == 3
    assert json.loads((vdir / "completion.json").read_text())["generation_calls_total"] == 3
    lines = [json.loads(x) for x in (vdir / "calls.jsonl").read_text().splitlines()]
    assert lines[0]["ok"] is False and lines[0]["error_class"] == "SchemaInvalidError"
    assert lines[0]["input_tokens"] == 0


def test_invalid_output_twice_fails_the_pass_and_continues(tmp_path):
    plan = {KEY_P: [SchemaInvalidError("bad", raw_text="{"), SchemaInvalidError("bad", raw_text="[")]}
    provider, result, vdir = _go(tmp_path, error_plan=plan)
    assert provider.calls == [KEY_P, KEY_P, KEY_R]
    assert result["status"] == "partial" and result["generation_calls"] == 3
    assert result["passes"] == {"primary": "failed", "replicate": "ok"}
    assert (vdir / "results/primary_failed.json").exists()
    assert not (vdir / "agreement.json").exists()


def test_retryable_provider_error_backs_off_and_succeeds(tmp_path):
    plan = {KEY_P: [ProviderError("busy", status=503)]}
    sleeps = []
    provider, release, books, cfg, video, root = _setup(tmp_path, error_plan=plan)
    result = run_video(provider, release, books, cfg, video, root, sleep=sleeps.append)
    assert provider.calls == [KEY_P, KEY_P, KEY_R]
    assert len(sleeps) == 1 and result["status"] == "complete"


def test_fatal_error_propagates(tmp_path):
    provider, release, books, cfg, video, root = _setup(
        tmp_path, error_plan={KEY_P: [RunFatalError("auth")]})
    with pytest.raises(RunFatalError):
        run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)


def test_resume_reuses_cached_passes(tmp_path):
    _go(tmp_path)
    provider2, release, books, cfg, video, root = _setup(tmp_path)
    result = run_video(provider2, release, books, cfg, video, root, sleep=NOSLEEP)
    assert provider2.calls == [] and result["generation_calls"] == 0
    assert result["generation_calls_total"] == 2
    assert provider2.uploads == []
    assert result["status"] == "complete"
    rows = plan_calls([video], root)
    assert [r["cached"] for r in rows] == [True, True]


def _snapshot_files(vdir):
    return {str(p.relative_to(vdir)): p.read_bytes()
            for sub in ("bundle", "prompts") for p in sorted((vdir / sub).iterdir())}


@pytest.mark.parametrize("key", ["prompt_hash", "bundle_sha256", "source_sha256", "model", "schema_sha256"])
def test_changed_instrument_refuses_to_resume(tmp_path, key):
    _go(tmp_path)
    vdir = tmp_path / "run/vid1"
    exp = vdir / "experiment.json"
    data = json.loads(exp.read_text())
    data[key] = {"x": 1} if key in ("bundle_sha256", "model") else "deadbeef"
    exp.write_text(json.dumps(data))
    before = _snapshot_files(vdir)
    provider, release, books, cfg, video, root = _setup(tmp_path)
    with pytest.raises(ConfigError, match="new run name"):
        run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    assert provider.calls == [] and provider.uploads == []
    assert _snapshot_files(vdir) == before


def test_refused_resume_does_not_overwrite_original_snapshot(tmp_path):
    _go(tmp_path)
    vdir = tmp_path / "run/vid1"
    (vdir / "bundle" / PANEL_TEMPLATE).write_text("ORIGINAL")
    (vdir / "prompts/panel_full.txt").write_text("ORIGINAL PROMPT")
    data = json.loads((vdir / "experiment.json").read_text())
    data["bundle_sha256"][PANEL_TEMPLATE] = "changed"
    (vdir / "experiment.json").write_text(json.dumps(data))
    provider, release, books, cfg, video, root = _setup(tmp_path)
    with pytest.raises(ConfigError):
        run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    assert (vdir / "bundle" / PANEL_TEMPLATE).read_text() == "ORIGINAL"
    assert (vdir / "prompts/panel_full.txt").read_text() == "ORIGINAL PROMPT"


def test_upload_error_is_retried(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    real = provider.upload_media
    state = {"n": 0}

    def flaky(path, file_sha256=None):
        state["n"] += 1
        if state["n"] == 1:
            raise ProviderError("busy", status=503)
        return real(path, file_sha256=file_sha256)

    provider.upload_media = flaky
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    assert state["n"] == 2 and result["status"] == "complete"


def test_upload_failure_records_both_passes_failed(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)

    def broken(path, file_sha256=None):
        raise ProviderError("bad file", status=400)

    provider.upload_media = broken
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    assert result["status"] == "failed" and result["generation_calls"] == 0
    assert provider.calls == []
    failed = json.loads((root / "vid1/results/primary_failed.json").read_text())
    assert "bad file" in failed["message"]


def test_upload_fatal_error_propagates(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)

    def fatal(path, file_sha256=None):
        raise RunFatalError("auth")

    provider.upload_media = fatal
    with pytest.raises(RunFatalError):
        run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)


def test_non_retryable_provider_error_fails_pass_and_keeps_body(tmp_path):
    plan = {KEY_P: [ProviderError("blocked", status=400, raw_text="BODY")]}
    provider, result, vdir = _go(tmp_path, error_plan=plan)
    assert provider.calls == [KEY_P, KEY_R]
    assert result["status"] == "partial" and result["generation_calls"] == 2
    assert (vdir / "raw/primary__attempt1_failed.txt").read_text() == "BODY"


def test_validation_errors_are_recorded_not_retried(tmp_path):
    bad = synthetic_record()
    bad["turns"][1]["function"] = "not_a_function"
    provider, result, vdir = _go(tmp_path, primary=bad)
    assert len(provider.calls) == 2
    errs = json.loads((vdir / "results/primary_validation.json").read_text())["errors"]
    assert any("turn.function" in e for e in errs)
    assert result["validation_errors"]["primary"] == len(errs) >= 1
    assert result["validation_errors"]["replicate"] == 0


def test_bundle_snapshot_contains_books_templates_their_includes_and_manifest(tmp_path):
    _go(tmp_path)
    release = load_release()
    bundle = tmp_path / "run/vid1/bundle"
    names = {p.name for p in bundle.iterdir()}
    assert names == {release.manifest_path.name, PANEL_TEMPLATE, "stage_a_perceived_delivery_protocol.md"} | {
        n for b in release.books.values() for n in (b.markdown.name, b.yaml.name, b.template)}
    assert len(names) == 10
    # the protocol that panel_full.j2 includes travels with it, byte for byte, and is hashed like the rest
    protocol = "stage_a_perceived_delivery_protocol.md"
    assert (bundle / protocol).read_bytes() == (release.templates_dir / protocol).read_bytes()
    exp = json.loads((tmp_path / "run/vid1/experiment.json").read_text())
    assert set(exp["bundle_sha256"]) == names
    assert exp["release_id"] == "v3-dev9" == yaml.safe_load((bundle / release.manifest_path.name).read_text())["release_id"]
    assert exp["bundle_sha256"][protocol] == hashlib.sha256((release.templates_dir / protocol).read_bytes()).hexdigest()


def test_bundle_sources_follow_includes_found_in_the_template_text(tmp_path):
    from pipeline.stage_b.panel_pipeline import _bundle_sources
    release = load_release()
    templates = tmp_path / "prompts"
    shutil.copytree(release.templates_dir, templates)
    (templates / "outer.md").write_text("{%- include \"inner.md\" -%}")
    (templates / "inner.md").write_text("nested")
    with open(templates / PANEL_TEMPLATE, "a") as fh:
        fh.write("{% include 'outer.md' %}\n")
    sources = _bundle_sources(dataclasses.replace(release, templates_dir=templates))
    assert sources["outer.md"] == templates / "outer.md" and sources["inner.md"] == templates / "inner.md"


def test_run_agreement_recomputes_offline(tmp_path):
    _, result, vdir = _go(tmp_path)
    (vdir / "agreement.json").unlink()
    release, cfg = load_release(), load_panel_config(CONFIG)
    report = run_agreement(release, load_panel_books(release), cfg, "vid1", tmp_path / "run")
    assert (vdir / "agreement.json").exists() and report["video_id"] == "vid1"


def test_unit_key():
    assert unit_key("v", "primary") == "panel__v__primary"


def test_load_panel_config_reads_the_shipped_yaml():
    cfg = load_panel_config(CONFIG)
    assert cfg.model.temperature == 0.7 and cfg.model.max_output_tokens == 65536
    assert cfg.invalid_output_retries == 1 and cfg.alignment.decision_tolerance_s == 30.0
    assert cfg.run_name == "full-corpus-100"


def test_the_shipped_run_matches_the_reference_manifest():
    """The descriptive run name matches the released corpus; the instrument keeps its own version."""
    reference = json.loads((REPO / "examples/demo_release.json").read_text())
    run_name = load_panel_config(CONFIG).run_name
    assert load_release().release_id == "v3-dev9"
    assert run_name == Path(reference["run"]).name
    assert (REPO / reference["run"] / "selection_all100.json").is_file()


def _selection(tmp_path):
    video = _video(tmp_path)
    sel = tmp_path / "sel.json"
    sel.write_text(json.dumps({"videos": [video]}))
    return sel


def _cli(*args):
    return subprocess.run([PY, str(REPO / "scripts/experiments/run_panel_pilot.py"), *args],
                          capture_output=True, text=True)


def test_cli_preflight_makes_no_calls_and_lists_two_per_video(tmp_path):
    out = _cli("--phase", "preflight", "--selection", str(_selection(tmp_path)),
               "--runs-root", str(tmp_path / "runs"))
    assert out.returncode == 0, out.stderr
    assert out.stdout.count("billable") == 3  # two lines plus the total line
    assert "vid1 primary billable" in out.stdout and "vid1 replicate billable" in out.stdout
    assert "total billable: 2" in out.stdout and "prompt_hash" in out.stdout
    assert not (tmp_path / "runs").exists()


def test_cli_annotate_with_fake_provider_end_to_end(tmp_path):
    fx = _fixtures(tmp_path)
    args = ["--phase", "annotate", "--selection", str(_selection(tmp_path)), "--provider", "fake",
            "--fixtures", str(fx), "--run-name", "t", "--runs-root", str(tmp_path / "runs")]
    dry = _cli(*args)
    assert dry.returncode == 0 and "without --execute" in dry.stdout
    assert not (tmp_path / "runs").exists()
    out = _cli(*args, "--execute")
    assert out.returncode == 0, out.stderr
    assert "VIDEO vid1 complete" in out.stdout
    assert (tmp_path / "runs/t/vid1/agreement.json").exists()
    (tmp_path / "runs/t/vid1/agreement.json").unlink()
    results = tmp_path / "runs/t/vid1/results"
    (results / "primary_validation.json").write_text(json.dumps({"errors": ["stale"], "flags": []}))
    (results / "replicate_validation.json").unlink()
    agree_args = ["--phase", "agreement", "--selection", str(_selection(tmp_path)),
                  "--run-name", "t", "--runs-root", str(tmp_path / "runs")]
    again = _cli(*agree_args)
    assert again.returncode == 0 and (tmp_path / "runs/t/vid1/agreement.json").exists()
    for name in ("primary", "replicate"):
        assert json.loads((results / f"{name}_validation.json").read_text()) == {"errors": [], "flags": []}
    # a video that lacks a pass: its present pass is still revalidated, no agreement, exit 1
    (tmp_path / "runs/t/vid1/agreement.json").unlink()
    (results / "replicate.json").unlink()
    (results / "replicate_validation.json").unlink()
    (results / "primary_validation.json").write_text(json.dumps({"errors": ["stale"], "flags": []}))
    lacking = _cli(*agree_args)
    assert lacking.returncode == 1 and "needs both passes" in lacking.stdout
    assert not (tmp_path / "runs/t/vid1/agreement.json").exists()
    assert json.loads((results / "primary_validation.json").read_text())["errors"] == []
    assert not (results / "replicate_validation.json").exists()


def test_experiment_records_schema_hash_and_git_commit(tmp_path):
    import hashlib
    _, _, vdir = _go(tmp_path)
    exp = json.loads((vdir / "experiment.json").read_text())
    assert exp["schema_sha256"] == hashlib.sha256((vdir / "prompts/panel_full.schema.json").read_bytes()).hexdigest()
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    assert exp["git_commit"] == head and len(head) == 40


def test_git_commit_is_null_when_git_is_unavailable(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise OSError("no git")
    monkeypatch.setattr(panel_pipeline.subprocess, "run", boom)
    _, result, vdir = _go(tmp_path)
    exp = json.loads((vdir / "experiment.json").read_text())
    assert "git_commit" in exp and exp["git_commit"] is None and result["status"] == "complete"


def test_git_commit_is_not_compared_on_resume(tmp_path):
    _go(tmp_path)
    exp = tmp_path / "run/vid1/experiment.json"
    data = json.loads(exp.read_text())
    data["git_commit"] = "0" * 40
    exp.write_text(json.dumps(data))
    provider, release, books, cfg, video, root = _setup(tmp_path)
    assert run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)["status"] == "complete"
    assert json.loads(exp.read_text())["git_commit"] == "0" * 40


def test_failed_attempt_logs_usage_carried_by_the_error(tmp_path):
    exc = SchemaInvalidError("MAX_TOKENS", raw_text="{")
    exc.usage = {"input_tokens": 9000, "output_tokens": 65536, "thinking_tokens": 1200}
    exc.model_version = "model-2026-09"
    blocked = ProviderError("blocked", status=400, raw_text="BODY")
    blocked.usage = {"input_tokens": 9000, "output_tokens": 0, "thinking_tokens": 0}
    blocked.model_version = "model-2026-09"
    provider, result, vdir = _go(tmp_path, error_plan={KEY_P: [exc], KEY_R: [blocked]})
    lines = [json.loads(x) for x in (vdir / "calls.jsonl").read_text().splitlines()]
    assert [x["ok"] for x in lines] == [False, True, False]
    assert (lines[0]["input_tokens"], lines[0]["output_tokens"], lines[0]["thinking_tokens"]) == (9000, 65536, 1200)
    assert lines[0]["model_version"] == "model-2026-09" and lines[0]["error_class"] == "SchemaInvalidError"
    assert (lines[2]["input_tokens"], lines[2]["output_tokens"], lines[2]["thinking_tokens"]) == (9000, 0, 0)
    assert lines[2]["model_version"] == "model-2026-09" and lines[2]["error_class"] == "ProviderError"
    assert result["generation_calls_total"] == 3


def test_failed_attempt_without_usage_logs_zeros(tmp_path):
    plan = {KEY_P: [ProviderError("busy", status=503)]}
    _, _, vdir = _go(tmp_path, error_plan=plan)
    first = json.loads((vdir / "calls.jsonl").read_text().splitlines()[0])
    assert (first["input_tokens"], first["output_tokens"], first["thinking_tokens"]) == (0, 0, 0)
    assert first["model_version"] is None


def test_revalidate_rewrites_validation_for_present_passes(tmp_path):
    _, _, vdir = _go(tmp_path)
    (vdir / "results/primary_validation.json").write_text(json.dumps({"errors": ["stale"], "flags": []}))
    (vdir / "results/replicate.json").unlink()
    (vdir / "results/replicate_validation.json").unlink()
    books = load_panel_books(load_release())
    assert revalidate(books, "vid1", DURATION_S, tmp_path / "run") == ["primary"]
    assert json.loads((vdir / "results/primary_validation.json").read_text()) == {"errors": [], "flags": []}
    assert not (vdir / "results/replicate_validation.json").exists()
    # the duration is the one passed in: a shorter video puts the last turns beyond its end
    revalidate(books, "vid1", 100.0, tmp_path / "run")
    assert any("turn.time" in e for e in json.loads((vdir / "results/primary_validation.json").read_text())["errors"])


def _malformed_wire():
    bad = synthetic_wire()
    bad["turns"][3]["start"] = "120"          # 1:20 written without its colon
    return bad


def test_malformed_clock_time_is_an_invalid_output_and_is_retried(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    fixture = tmp_path / "fx" / f"{KEY_P}.json"
    fixture.write_text(json.dumps(_malformed_wire()))
    real = provider.generate

    def first_bad_then_good(request):
        try:
            return real(request)
        finally:
            if request.unit_key == KEY_P:
                fixture.write_text(json.dumps(synthetic_wire()))

    provider.generate = first_bad_then_good
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    vdir = root / "vid1"
    assert provider.calls == [KEY_P, KEY_P, KEY_R]
    assert result["status"] == "complete" and result["generation_calls"] == 3
    assert json.loads((vdir / "raw/primary__attempt1_invalid.txt").read_text()) == _malformed_wire()
    message = json.loads((vdir / "raw/primary__attempt1_error.json").read_text())["message"]
    assert "turns[3].start" in message and "'120'" in message
    lines = [json.loads(x) for x in (vdir / "calls.jsonl").read_text().splitlines()]
    assert [x["ok"] for x in lines] == [False, True, True]
    assert lines[0]["error_class"] == "SchemaInvalidError" and lines[0]["input_tokens"] > 0
    assert lines[0]["model_version"] == "fake-model-v1"
    stored = json.loads((vdir / "results/primary.json").read_text())["record"]
    assert stored["turns"][3]["start_s"] == 80.0
    assert not (vdir / "results/primary_failed.json").exists()


def test_malformed_clock_time_on_every_attempt_fails_the_pass_and_the_other_runs(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    (tmp_path / "fx" / f"{KEY_P}.json").write_text(json.dumps(_malformed_wire()))
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    vdir = root / "vid1"
    assert provider.calls == [KEY_P, KEY_P, KEY_R]            # one retry, as for an unparseable body
    assert result["status"] == "partial" and result["passes"] == {"primary": "failed", "replicate": "ok"}
    assert (vdir / "raw/primary__attempt1_invalid.txt").exists() and (vdir / "raw/primary__attempt2_invalid.txt").exists()
    assert "turns[3].start" in json.loads((vdir / "results/primary_failed.json").read_text())["message"]
    assert not (vdir / "results/primary.json").exists() and not (vdir / "raw/primary.json").exists()
    assert (vdir / "results/replicate.json").exists() and not (vdir / "agreement.json").exists()


def test_alignment_settings_have_no_turn_iou_and_old_configs_still_load(tmp_path):
    cfg = load_panel_config(CONFIG)
    assert not hasattr(cfg.alignment, "turn_iou") and "turn_iou" not in CONFIG.read_text()
    old = tmp_path / "old.yaml"
    old.write_text(CONFIG.read_text() + "  turn_iou: 0.5\n")
    loaded = load_panel_config(old)
    assert loaded.alignment == cfg.alignment and loaded.alignment.event_tolerance_s == 10.0


# ---- v3-dev6: a value given together with a missing reason ---------------------------------

def _wire_with_value_and_missing():
    wire = synthetic_wire()
    wire["turns"][2]["codes"][5].update(value="4", missing="not_applicable")      # turn 3, PREP-COH
    wire["stage_a"][2].update(value=4, uncodable=True)
    return wire


NOTES = ["stage_a[2] interpersonal_warmth: value 4 dropped because uncodable is true",
         "turns[2].codes PREP-COH: value '4' dropped because missing is 'not_applicable'"]


def test_value_with_a_missing_reason_is_normalized_not_retried(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    (tmp_path / "fx" / f"{KEY_P}.json").write_text(json.dumps(_wire_with_value_and_missing()))
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    vdir = root / "vid1"
    assert provider.calls == [KEY_P, KEY_R] and result["status"] == "complete"      # no paid retry
    assert not list((vdir / "raw").glob("*invalid*"))
    stored = json.loads((vdir / "results/primary.json").read_text())
    assert stored["normalizations"] == NOTES
    record = PanelRecord.model_validate(stored["record"])                           # strict invariants hold
    coh = next(c for c in record.turns[2].codes if c.code == "PREP-COH")
    assert coh.value is None and coh.missing == "not_applicable"
    assert record.stage_a[2].value is None and record.stage_a[2].uncodable is True
    assert json.loads((vdir / "raw/primary.json").read_text()) == _wire_with_value_and_missing()   # raw kept as returned
    validation = json.loads((vdir / "results/primary_validation.json").read_text())
    assert validation["errors"] == []
    assert validation["flags"] == [f"[flag.value_with_missing] {n}" for n in NOTES]
    assert json.loads((vdir / "results/replicate.json").read_text())["normalizations"] == []
    assert json.loads((vdir / "results/replicate_validation.json").read_text())["flags"] == []


def test_revalidation_keeps_the_stored_normalization_notes(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    (tmp_path / "fx" / f"{KEY_P}.json").write_text(json.dumps(_wire_with_value_and_missing()))
    run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    vdir = root / "vid1"
    before = (vdir / "results/primary.json").read_bytes()
    (vdir / "results/primary_validation.json").write_text(json.dumps({"errors": ["stale"], "flags": []}))
    assert revalidate(books, "vid1", DURATION_S, root) == ["primary", "replicate"]
    validation = json.loads((vdir / "results/primary_validation.json").read_text())
    assert validation == {"errors": [], "flags": [f"[flag.value_with_missing] {n}" for n in NOTES]}
    assert (vdir / "results/primary.json").read_bytes() == before
    # a result written before v3-dev6 has no notes and revalidates as before
    path = vdir / "results/replicate.json"
    data = json.loads(path.read_text())
    del data["normalizations"]
    path.write_text(json.dumps(data))
    revalidate(books, "vid1", DURATION_S, root)
    assert json.loads((vdir / "results/replicate_validation.json").read_text()) == {"errors": [], "flags": []}


def test_null_value_with_missing_none_is_still_an_invalid_output(tmp_path):
    for mutate in (lambda w: w["turns"][2]["codes"][5].update(value=None, missing="none"),
                   lambda w: w["stage_a"][2].update(value=None, uncodable=False)):
        sub = tmp_path / str(id(mutate))
        sub.mkdir()
        provider, release, books, cfg, video, root = _setup(sub)
        bad = synthetic_wire()
        mutate(bad)
        (sub / "fx" / f"{KEY_P}.json").write_text(json.dumps(bad))
        result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
        assert provider.calls == [KEY_P, KEY_P, KEY_R]
        assert result["passes"] == {"primary": "failed", "replicate": "ok"}


# ---- v3-dev7: frames derived from concepts ------------------------------------------------

def _derive_with(mutate):
    data = synthetic_record()
    mutate({t["turn_id"]: t for t in data["turns"]})
    return derive(PanelRecord.model_validate(data), load_panel_books(load_release()))


def _set_concept(turn, value, missing="none"):
    next(c for c in turn["codes"] if c["code"].endswith("-CONCEPT")).update(value=value, missing=missing)


def test_frame_match_values():
    def m(t):
        _set_concept(t[2], "sales")                 # promotion question, prevention answer
        _set_concept(t[4], "NONE")                  # neutral question
        _set_concept(t[12], "NONE")                 # neutral answer
        _set_concept(t[14], "Vision ")              # promotion question, promotion answer; case and space repaired
    out = _derive_with(m)
    assert out["turn_frame"] == {"2": "promotion", "3": "prevention", "4": "NEUTRAL", "5": "prevention",
                                 "11": "prevention", "12": "NEUTRAL", "14": "promotion", "15": "promotion"}
    assert out["frame_match"] == {"3": "switch_to_prevention", "5": "neutral", "12": "neutral", "15": "match"}


def test_frames_are_omitted_when_the_concept_is_missing_or_there_is_no_question():
    def m(t):
        _set_concept(t[2], None, "ambiguous")       # question without a concept: answer 3 has a frame but no match
        _set_concept(t[5], None, "too_short")       # answer without a concept: no frame and no match
        t[12]["responds_to"] = [7]                  # answer 12 now responds to an accepts turn only
        t[15]["responds_to"] = [13, 2, 14]          # first question with a concept is 14 (2 has none; 13 is no question)
    out = _derive_with(m)
    assert set(out["turn_frame"]) == {"3", "4", "11", "12", "14", "15"}
    assert out["frame_match"] == {"15": "switch_to_promotion"}


def test_frames_are_only_for_question_and_answer_turns():
    def m(t):
        t[13]["codes"] = [{"code": "PR-CONCEPT", "value": "sales", "missing": "none"}]     # an agrees turn
        t[4].update(function="challenges")          # keeps its PR codes, but is not a question any more
    out = _derive_with(m)
    assert "13" not in out["turn_frame"] and "4" not in out["turn_frame"]
    assert "5" not in out["frame_match"]            # answer 5 responds to a challenges turn, not a question


# ---- v3-dev8: other concepts ----------------------------------------------------------------

def test_derived_lists_other_concepts_and_marks_unclassified_frames():
    from tests.panel_fixtures import set_other
    data = synthetic_record()
    turns = {t["turn_id"]: t for t in data["turns"]}
    set_other(turns[2], "shelf life")                          # unclassified question, prevention answer
    set_other(turns[12], "money the founder put in")           # prevention question, unclassified answer
    next(c for c in turns[14]["codes"] if c["code"] == "PR-CONCEPT")["value"] = "NONE"
    set_other(turns[15], "a patent")                           # neutral question, unclassified answer
    data["exchanges"][0].update(topic="other", label="shelf life of the lids")
    out = derive(PanelRecord.model_validate(data), load_panel_books(load_release()))
    assert out["turn_frame"]["2"] == "unclassified" and out["turn_frame"]["12"] == "unclassified"
    assert out["frame_match"] == {"3": "unclassified", "5": "match", "12": "unclassified", "15": "unclassified"}
    assert out["other_concepts"] == [
        {"where": "turn", "id": 2, "code": "PR-CONCEPT", "name": "shelf life", "clock": "1:02"},
        {"where": "turn", "id": 12, "code": "RS-CONCEPT", "name": "money the founder put in", "clock": "1:59"},
        {"where": "turn", "id": 15, "code": "RS-CONCEPT", "name": "a patent", "clock": "2:18"},
        {"where": "exchange", "id": 1, "code": "EX-TOPIC", "name": "shelf life of the lids", "clock": "1:02"},
    ]


# ---- a code value written as the word "missing" ---------------------------------------------

def _wire_with_missing_words():
    wire = synthetic_wire()
    wire["turns"][2]["codes"][4].update(value="missing")      # PREP-SUBS, ordinal
    wire["turns"][2]["codes"][5].update(value="None")         # PREP-COH, ordinal
    wire["turns"][2]["codes"][1].update(value="NONE")         # RS-CONCEPT: the codebook's residual value
    return wire


WORD_NOTES = ["turns[2].codes PREP-SUBS: value 'missing' read as missing, reason set to 'not_applicable'",
              "turns[2].codes PREP-COH: value 'None' read as missing, reason set to 'not_applicable'"]


def test_books_name_the_categorical_codes():
    books = load_panel_books(load_release())
    ids = books.categorical_code_ids()
    assert {"PR-CONCEPT", "RS-CONCEPT", "RS-RESP", "TURN-FUNC"} <= ids
    assert not ids & {"PR-SEV", "PREP-SUBS", "RS-EVID"}


def test_the_word_missing_as_a_value_is_normalized_by_the_runner(tmp_path):
    provider, release, books, cfg, video, root = _setup(tmp_path)
    (tmp_path / "fx" / f"{KEY_P}.json").write_text(json.dumps(_wire_with_missing_words()))
    result = run_video(provider, release, books, cfg, video, root, sleep=NOSLEEP)
    vdir = root / "vid1"
    assert provider.calls == [KEY_P, KEY_R] and result["status"] == "complete"
    stored = json.loads((vdir / "results/primary.json").read_text())
    assert stored["normalizations"] == WORD_NOTES
    codes = {c["code"]: (c["value"], c["missing"]) for c in stored["record"]["turns"][2]["codes"]}
    assert codes["PREP-SUBS"] == (None, "not_applicable") and codes["PREP-COH"] == (None, "not_applicable")
    assert codes["RS-CONCEPT"] == ("NONE", "none")
    validation = json.loads((vdir / "results/primary_validation.json").read_text())
    assert not any("PREP-SUBS" in e or "PREP-COH" in e for e in validation["errors"])
    assert [f for f in validation["flags"] if "value_with_missing" in f] == \
        [f"[flag.value_with_missing] {n}" for n in WORD_NOTES]
    assert json.loads((vdir / "raw/primary.json").read_text()) == _wire_with_missing_words()


def test_revalidate_normalizes_a_stored_record_that_holds_the_word(tmp_path):
    _, _, vdir = _go(tmp_path)
    books = load_panel_books(load_release())
    path = vdir / "results/primary.json"
    stored = json.loads(path.read_text())
    stored["record"]["turns"][2]["codes"][4].update(value="missing")       # as an older run stored it
    stored["record"]["turns"][2]["codes"][5].update(value="None")
    path.write_text(json.dumps(stored))
    untouched = (vdir / "results/replicate.json").read_bytes()
    revalidate(books, "vid1", DURATION_S, vdir.parent)
    validation = json.loads((vdir / "results/primary_validation.json").read_text())
    assert validation == {"errors": [], "flags": [f"[flag.value_with_missing] {n}" for n in WORD_NOTES]}
    after = json.loads(path.read_text())
    assert after["normalizations"] == WORD_NOTES
    assert after["record"]["turns"][2]["codes"][4] == {"code": "PREP-SUBS", "value": None, "missing": "not_applicable"}
    assert after["derived"] == stored["derived"] and after["prepared_window"] == stored["prepared_window"]
    assert (vdir / "results/replicate.json").read_bytes() == untouched
    # a second revalidation changes nothing and keeps the notes
    once = path.read_bytes()
    revalidate(books, "vid1", DURATION_S, vdir.parent)
    assert path.read_bytes() == once
    assert json.loads((vdir / "results/primary_validation.json").read_text()) == validation
