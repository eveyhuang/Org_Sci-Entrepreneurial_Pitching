"""Whole pipeline, no network: ingest -> annotate (fake provider) -> aggregate."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES
from tests.conftest import make_test_video, needs_ffmpeg

REPO = Path(__file__).resolve().parents[1]


def build_project(tmp_path: Path) -> tuple[Path, Path]:
    videos = tmp_path / "videos"
    for vid in ("abc", "def"):
        make_test_video(videos / f"{vid}.mp4", seconds=2.0)
    (tmp_path / "meta.csv").write_text(
        "video_id,platform,url\nabc,youtube,https://a\ndef,youtube,https://d\n"
    )
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    rating = {"value": 4, "uncodable": False, "evidence": "Steady, warm delivery at 0:01."}
    payload = {name: dict(rating) for name in STAGE_A_DELIVERY_CODES} | {"overall_notes": ""}
    for vid in ("abc", "def"):
        (fixtures / f"holistic__{vid}.json").write_text(json.dumps(payload))

    codebook = REPO / "pipeline" / "codebooks" / "stage_a_perceived_delivery.yaml"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent(f"""
        run_name: e2e
        corpus:
          name: hu-ma
          video_dir: videos
          metadata_csv: meta.csv
        model:
          name: gemini-test-model
        limits:
          requests_per_minute: 6000
          max_concurrency: 2
        passes:
          - name: holistic
            unit: PITCH
            prompt: stage_a_perceived_delivery.j2
            schema: pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord
            k: 2
            media: full
        codebook: {codebook}
    """))
    return cfg, fixtures


def run_cli(script: str, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        cwd=cwd, capture_output=True, text=True, env={"PYTHONPATH": str(REPO), "PATH": __import__("os").environ["PATH"]},
    )
    assert result.returncode == 0, result.stderr
    return result


@needs_ffmpeg
def test_full_pipeline_ingest_annotate_aggregate(tmp_path):
    cfg, fixtures = build_project(tmp_path)

    run_cli("ingest.py", "--config", str(cfg), cwd=tmp_path)
    assert (tmp_path / "registry.sqlite").exists()

    run_cli(
        "annotate.py", "--config", str(cfg),
        "--provider", "fake", "--fixtures", str(fixtures), cwd=tmp_path,
    )
    run_cli("aggregate.py", "--config", str(cfg), cwd=tmp_path)

    run_dir = next((tmp_path / "runs").glob("e2e_*"))
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["passes"]["holistic"]["done"] == 4  # 2 videos x k=2
    assert manifest["passes"]["holistic"]["prompt_hash"]
    assert manifest["status_counts"]["done"] == 4

    annotations = pd.read_parquet(run_dir / "tables" / "annotations.parquet")
    assert len(annotations) == 12  # 2 videos x k=2 x 3 codes
    videos = pd.read_parquet(run_dir / "tables" / "videos.parquet")
    assert len(videos) == 2


@needs_ffmpeg
def test_second_annotate_run_is_a_no_op(tmp_path):
    cfg, fixtures = build_project(tmp_path)
    run_cli("ingest.py", "--config", str(cfg), cwd=tmp_path)
    args = ("annotate.py", "--config", str(cfg), "--provider", "fake", "--fixtures", str(fixtures))
    run_cli(*args, cwd=tmp_path)
    second = run_cli(*args, cwd=tmp_path)
    assert "skipped=4" in second.stdout


@needs_ffmpeg
def test_aggregate_before_annotate_warns_instead_of_silently_empty(tmp_path):
    cfg, _fixtures = build_project(tmp_path)
    run_cli("ingest.py", "--config", str(cfg), cwd=tmp_path)
    result = run_cli("aggregate.py", "--config", str(cfg), cwd=tmp_path)
    assert "no completed annotation units" in result.stderr
