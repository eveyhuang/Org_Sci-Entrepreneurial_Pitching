from datetime import datetime, timezone

from pipeline.config import load_config
from pipeline.runs import (
    create_or_resume_run,
    parsed_path,
    raw_path,
    read_manifest,
    update_manifest,
    write_manifest,
)
from tests.test_config import BASE, write_config

NOW = datetime(2026, 9, 2, 14, 30, tzinfo=timezone.utc)


def make_cfg(tmp_path, body=BASE):
    return load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_create_run_makes_directory_and_freezes_config(tmp_path):
    cfg = make_cfg(tmp_path)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    assert run.run_id == "stage_a_v1_20260902_143000"
    assert (run.root / "config.yaml").read_text() == cfg.raw_yaml
    assert (run.root / "raw").is_dir()
    assert (run.root / "parsed").is_dir()


def test_resumes_existing_run_with_identical_config(tmp_path):
    cfg = make_cfg(tmp_path)
    first = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    later = NOW.replace(hour=18)
    second = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=later)
    assert second.run_id == first.run_id


def test_changed_config_starts_a_new_run(tmp_path):
    cfg = make_cfg(tmp_path)
    first = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    changed = make_cfg(tmp_path, BASE.replace("temperature: 0.0", "temperature: 0.3"))
    # Same `now`, so a differing run_id can only come from the config comparison
    # refusing to reuse the existing directory.
    second = create_or_resume_run(changed, runs_root=tmp_path / "runs", now=NOW)
    assert second.run_id != first.run_id
    assert second.root != first.root
    assert (second.root / "config.yaml").read_text() == changed.raw_yaml
    assert (first.root / "config.yaml").read_text() == cfg.raw_yaml


def test_force_new_run(tmp_path):
    cfg = make_cfg(tmp_path)
    first = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    second = create_or_resume_run(
        cfg, runs_root=tmp_path / "runs", now=NOW.replace(hour=18), force_new=True
    )
    assert second.run_id != first.run_id


def test_output_paths_encode_span_replicate_and_prompt(tmp_path):
    cfg = make_cfg(tmp_path)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    whole = raw_path(run, "holistic", "vidA", k=2, span=None, prompt_hash="abcdef0123456789")
    assert whole.name == "vidA__k2__abcdef01.json"
    assert whole.parent == run.root / "raw" / "holistic"
    episode = parsed_path(
        run, "episode", "vidA", k=0, span=(12.0, 48.5), prompt_hash="abcdef0123456789"
    )
    assert episode.name == "vidA__12.000_48.500__k0__abcdef01.json"
    assert episode.parent == run.root / "parsed" / "episode"


def test_manifest_write_and_patch(tmp_path):
    cfg = make_cfg(tmp_path)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs", now=NOW)
    write_manifest(run, {"model_requested": "gemini-x", "passes": {}})
    update_manifest(run, {"passes": {"holistic": {"prompt_hash": "abc"}}})
    manifest = read_manifest(run)
    assert manifest["model_requested"] == "gemini-x"
    assert manifest["passes"]["holistic"]["prompt_hash"] == "abc"
    assert manifest["run_id"] == run.run_id
