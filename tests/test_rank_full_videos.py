"""Ranking whole videos: a corpus of standalone pitch videos (Hu & Ma) has no panel run and no
questioning, so each video's prepared window is the whole file and the source is uploaded as the clip."""

import csv
import json
import subprocess
import sys
from pathlib import Path

from pipeline.errors import ConfigError
from pipeline.ingest import ProbeResult
from pipeline.instrument import load_release
from pipeline.providers.fake import FakeProvider
from pipeline.stage_a.rank_pipeline import (
    full_video_windows, plan_ranking, plan_ranking_from_windows, run_ranking, whole_source,
)
from pipeline.stage_b.panel_pipeline import load_panel_config
from tests.conftest import make_test_video, needs_ffmpeg
from tests.test_rank_pipeline import CODES, CONFIG, NOSLEEP, _add_video, _payload

PY = sys.executable
CLI = Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "run_stage_a_ranking.py"


def _probe(seconds=60.0):
    def probe(path):
        name = Path(path).name
        if name.startswith("broken"):
            raise RuntimeError("ffprobe failed")
        if name.startswith("nodur"):
            return ProbeResult(duration_s=None, width=None, height=None, has_audio=True)
        return ProbeResult(duration_s=seconds, width=640, height=360, has_audio=True)
    return probe


def _corpus(tmp_path, names=("vid1.mp4", "vid2.mkv", "vid3.webm")):
    vdir = tmp_path / "videos"
    vdir.mkdir(exist_ok=True)
    for name in names:
        (vdir / name).write_bytes(b"video " + name.encode())
    return vdir


def _cli(*args):
    return subprocess.run([PY, str(CLI), *args], capture_output=True, text=True)


# --- windows -------------------------------------------------------------------------------

def test_whole_video_is_the_window_and_the_file_is_the_source(tmp_path):
    vdir = _corpus(tmp_path)
    windows, skipped = full_video_windows(vdir, ["vid2", "vid1", "vid3"], probe=_probe(58.5))
    assert skipped == {}
    assert list(windows) == ["vid1", "vid2", "vid3"]  # sorted, input order irrelevant
    assert windows["vid2"] == {"start_s": 0.0, "end_s": 58.5, "n_spans": 1, "source": str(vdir / "vid2.mkv")}


def test_videos_that_cannot_be_ranked_are_skipped_with_a_reason(tmp_path):
    vdir = _corpus(tmp_path, ("vid1.mp4", "dup.mp4", "dup.mkv", "broken.mp4", "nodur.mp4",
                              "partial.f137.mp4.part"))
    windows, skipped = full_video_windows(vdir, ["vid1", "dup", "broken", "nodur", "partial", "absent"],
                                          probe=_probe())
    assert list(windows) == ["vid1"]
    assert "no video file" in skipped["absent"] and "no video file" in skipped["partial"]
    assert "more than one" in skipped["dup"]
    assert "ffprobe failed" in skipped["broken"]
    assert "no duration" in skipped["nodur"]


def test_plan_from_windows_is_the_same_design_as_a_panel_plan(tmp_path):
    vdir = _corpus(tmp_path, tuple(f"vid{i}.mp4" for i in range(1, 7)))
    windows, skipped = full_video_windows(vdir, [f"vid{i}" for i in range(1, 7)], probe=_probe())
    plan = plan_ranking_from_windows(windows, skipped, 3, 2, 0, windows_from="whole_videos")
    root = tmp_path / "panel_run"
    for vid in windows:
        _add_video(root, vid)
    panel = plan_ranking(root, list(windows), 3, 2, 0)
    assert plan["design"] == panel["design"] and plan["summary"] == panel["summary"]
    assert plan["settings"] == panel["settings"] == {"batch_size": 3, "appearances": 2, "seed": 0}
    assert plan["windows_from"] == "whole_videos" and panel["windows_from"] == "panel_primary_pass"
    assert plan["skipped"] == {}


def test_plan_from_windows_needs_two_videos(tmp_path):
    try:
        plan_ranking_from_windows({"only": {"start_s": 0.0, "end_s": 1.0, "n_spans": 1, "source": "x"}},
                                  {"gone": "no video file"}, 5, 2, 0, windows_from="whole_videos")
    except ConfigError as exc:
        assert "at least 2" in str(exc) and "gone" in str(exc)
    else:
        raise AssertionError("expected ConfigError")


# --- the source as the clip ------------------------------------------------------------------

def test_whole_source_returns_the_file_itself_and_writes_nothing(tmp_path):
    src = tmp_path / "vid1.mp4"
    src.write_bytes(b"x")
    assert whole_source(src, 0.0, 60.0, tmp_path / "clips") == src
    assert not (tmp_path / "clips").exists()


def test_ranking_whole_videos_uploads_the_sources_and_cuts_no_clip(tmp_path):
    vdir = _corpus(tmp_path, tuple(f"vid{i}.mp4" for i in range(1, 7)))
    windows, skipped = full_video_windows(vdir, [f"vid{i}" for i in range(1, 7)], probe=_probe())
    plan = plan_ranking_from_windows(windows, skipped, 3, 2, 0, windows_from="whole_videos")
    fx = tmp_path / "fx"
    fx.mkdir()
    for batch in plan["design"]:
        (fx / f"rank__{batch['batch_id']}.json").write_text(json.dumps(_payload(batch)))
    provider = FakeProvider(fx)
    root = tmp_path / "rank_run"
    result = run_ranking(provider, load_release(), load_panel_config(CONFIG), root, plan,
                         sleep=NOSLEEP, cut=whole_source)
    assert result["status"] == "complete" and result["batches_ok"] == 4
    assert not (root / "ranking" / "clips").exists()
    for request, batch in zip(provider.requests, plan["design"]):
        assert [h.local_path for _, h in request.media_parts] == [vdir / f"{v}.mp4" for v in batch["video_ids"]]
    design = json.loads((root / "ranking" / "design.json").read_text())
    assert design["windows_from"] == "whole_videos"
    assert design["windows"]["vid1"]["source"] == str(vdir / "vid1.mp4")
    rows = list(csv.DictReader((root / "ranking" / "scores.csv").open()))
    assert {r["video_id"] for r in rows} == set(windows) and len(rows) == 6 * len(CODES)


# --- CLI ---------------------------------------------------------------------------------------

def _real_corpus(tmp_path, n=6):
    vdir = tmp_path / "videos"
    for i in range(1, n + 1):
        make_test_video(vdir / f"vid{i}.mp4", seconds=1.0)
    return vdir


@needs_ffmpeg
def test_cli_video_dir_dry_run_plans_whole_videos_and_writes_nothing(tmp_path):
    vdir = _real_corpus(tmp_path)
    root = tmp_path / "rank_run"
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    out = _cli("--run", str(root), "--video-dir", str(vdir), "--batch-size", "3")
    assert out.returncode == 0, out.stderr
    assert "whole videos" in out.stdout and "total billable calls: 4" in out.stdout
    assert "vid6" in out.stdout and "no calls made" in out.stdout
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before and not root.exists()


@needs_ffmpeg
def test_cli_video_ids_from_csv_restricts_the_corpus(tmp_path):
    vdir = _real_corpus(tmp_path)
    listing = tmp_path / "outcomes.csv"
    listing.write_text("video_id,Accepted?\nvid1,0\nvid2,1\nvid3,0\nvid9,0\n")
    out = _cli("--run", str(tmp_path / "rank_run"), "--video-dir", str(vdir),
               "--video-ids-from", str(listing))
    assert out.returncode == 0, out.stderr
    assert "total billable calls: 2" in out.stdout
    assert "skipped vid9: no video file" in out.stdout
    assert "vid4" not in out.stdout
    both = _cli("--run", str(tmp_path / "rank_run"), "--video-dir", str(vdir),
                "--video-ids-from", str(listing), "--video-ids", "vid1")
    assert both.returncode == 1 and "mutually exclusive" in both.stderr


@needs_ffmpeg
def test_cli_whole_videos_with_the_fake_provider_end_to_end(tmp_path):
    vdir = tmp_path / "videos"
    ids = ["vid1", "vid2", "vid3"]
    for vid in ids:
        make_test_video(vdir / f"{vid}.mp4", seconds=2.0)
    windows, skipped = full_video_windows(vdir, ids)
    assert skipped == {} and all(1.6 < w["end_s"] < 2.4 for w in windows.values())
    plan = plan_ranking_from_windows(windows, skipped, 5, 2, 0, windows_from="whole_videos")
    fx = tmp_path / "fx"
    fx.mkdir()
    for batch in plan["design"]:
        (fx / f"rank__{batch['batch_id']}.json").write_text(json.dumps(_payload(batch)))
    config = tmp_path / "panel.yaml"
    config.write_text(CONFIG.read_text().replace("requests_per_minute: 4", "requests_per_minute: 6000"))
    root = tmp_path / "rank_run"
    args = ["--run", str(root), "--video-dir", str(vdir), "--config", str(config),
            "--provider", "fake", "--fixtures", str(fx)]
    out = _cli(*args, "--execute")
    assert out.returncode == 0, out.stderr
    assert "RANKING complete" in out.stdout and f"{CODES[0]}: vid1 > vid2 > vid3" in out.stdout
    rdir = root / "ranking"
    assert json.loads((rdir / "scores.json").read_text())["batches_ok"] == 2
    assert not (rdir / "clips").exists()
    again = _cli(*args)
    assert "total billable calls: 0" in again.stdout
