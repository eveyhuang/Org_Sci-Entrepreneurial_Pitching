"""Stage B through the real CLIs: ingest -> annotate -> export -> freeze -> annotate -> aggregate."""
import csv
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from tests.conftest import make_test_video, needs_ffmpeg
from tests.stage_b_helpers import PASSES, VIDEO, make_prompt_dir, write_fixtures

REPO = Path(__file__).resolve().parents[1]
ORDER = ["timeline", "segmentation", "prepared", "episode"]


def cli(script, *args, cwd):
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args], cwd=cwd, capture_output=True,
        text=True, env={"PYTHONPATH": str(REPO), "PATH": os.environ["PATH"]})
    assert result.returncode == 0, result.stderr
    return result.stdout


@needs_ffmpeg
def test_stage_b_end_to_end(tmp_path):
    make_test_video(tmp_path / "videos" / f"{VIDEO}.mp4", seconds=12.0)
    (tmp_path / "meta.csv").write_text(f"video_id\n{VIDEO}\n")
    fixtures = write_fixtures(tmp_path / "fixtures")
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({
        "run_name": "stb", "historical": True, "prompt_dir": str(make_prompt_dir(tmp_path / "prompts")),
        "model": {"name": "gemini-pro-x"},
        "corpus": {"name": "shark-tank", "video_dir": "videos", "metadata_csv": "meta.csv"},
        "limits": {"requests_per_minute": 6000, "max_concurrency": 2},
        "passes": [PASSES[name] for name in ORDER],
    }))
    cfg = ["--config", str(tmp_path / "cfg.yaml")]
    fake = ["--provider", "fake", "--fixtures", str(fixtures)]

    cli("ingest.py", *cfg, cwd=tmp_path)
    first = cli("annotate.py", *cfg, *fake, cwd=tmp_path)
    assert "timeline [gemini-pro-big]: done=1" in first and "segmentation [gemini-pro-x]: done=1" in first
    assert "episode [gemini-pro-x]: done=0" in first          # nothing frozen yet: nothing enumerated
    run_dir = next((tmp_path / "runs").glob("stb_*"))
    assert json.loads((run_dir / "timelines" / f"{VIDEO}.json").read_text())["turns"][7]["tag"] == "declines to invest"

    review = tmp_path / "review.csv"
    assert "exported 3 rows" in cli("freeze_episodes.py", *cfg, "--export", str(review), cwd=tmp_path)
    with open(review, newline="") as handle:
        assert [r["frozen"] for r in csv.DictReader(handle)] == ["False"] * 3
    assert "froze 1 videos" in cli("freeze_episodes.py", *cfg, "--freeze", "--from", str(review), cwd=tmp_path)

    second = cli("annotate.py", *cfg, *fake, cwd=tmp_path)
    assert "prepared [gemini-pro-x]: done=1" in second and "episode [gemini-pro-x]: done=2" in second
    assert "timeline [gemini-pro-big]: done=0" in second

    cli("aggregate.py", *cfg, cwd=tmp_path)
    tables = {p.stem: pd.read_parquet(p) for p in (run_dir / "tables").glob("*.parquet")}
    assert set(tables) == {"annotations", "videos", "turns", "episodes", "decisions"}
    assert len(tables["annotations"]) == 11 and bool(tables["episodes"].frozen.all())
    assert len(tables["decisions"]) == 2 and bool(tables["videos"].iloc[0]["deal"])

    # Stability report runs on what exists (one replicate: nothing to compare, no crash),
    # and it is strictly read-only: it must not create a new run directory.
    run_dirs_before = {p.name for p in (tmp_path / "runs").iterdir()}
    assert VIDEO in cli("boundary_report.py", *cfg, cwd=tmp_path)
    run_dirs_after = {p.name for p in (tmp_path / "runs").iterdir()}
    assert run_dirs_after == run_dirs_before

    # A config with no matching run directory is refused, not silently given a fresh one.
    (tmp_path / "cfg2.yaml").write_text(yaml.safe_dump({
        "run_name": "stb2", "historical": True, "prompt_dir": str(tmp_path / "prompts"), "model": {"name": "gemini-pro-x"},
        "corpus": {"name": "shark-tank", "video_dir": "videos", "metadata_csv": "meta.csv"},
        "limits": {"requests_per_minute": 6000, "max_concurrency": 2},
        "passes": [PASSES[name] for name in ORDER],
    }))
    no_run_result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "boundary_report.py"),
         "--config", str(tmp_path / "cfg2.yaml")],
        cwd=tmp_path, capture_output=True, text=True,
        env={"PYTHONPATH": str(REPO), "PATH": os.environ["PATH"]},
    )
    assert no_run_result.returncode != 0
    run_dirs_final = {p.name for p in (tmp_path / "runs").iterdir()}
    assert run_dirs_final == run_dirs_before  # still no new directory

    # A third run is a no-op, and the manifest records each pass's own model.
    third = cli("annotate.py", *cfg, *fake, cwd=tmp_path)
    for pass_name in ORDER:
        assert f"{pass_name} [" in third
    for line in third.splitlines():
        if any(f"{pass_name} [" in line for pass_name in ORDER):
            assert "done=0 failed=0" in line
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["passes"]["timeline"]["model"] == "gemini-pro-big"
    assert manifest["passes"]["episode"]["model"] == "gemini-pro-x"
    for pass_name in ORDER:
        assert "thinking_tokens" in manifest["passes"][pass_name]

    with sqlite3.connect(tmp_path / "registry.sqlite") as conn:
        assert conn.execute("SELECT COUNT(*) FROM work_units WHERE status != 'done'").fetchone()[0] == 0


def test_dev_config_differs_from_the_main_config_only_in_name_and_k():
    def normalised(path):
        data = yaml.safe_load((REPO / "configs" / path).read_text())
        data.pop("run_name")
        for entry in data["passes"]:
            entry.pop("k", None)
        return data

    assert normalised("archived_stage_b_v1_dev.yaml") == normalised("archived_stage_b_v1.yaml")
