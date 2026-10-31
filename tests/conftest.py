"""Shared fixtures. Video fixtures are synthesized with ffmpeg so no media
binaries are committed (the repo gitignores *.mp4)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def make_test_video(path: Path, seconds: float = 2.0, with_audio: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=128x96:rate=10",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture
def video_factory(tmp_path):
    def _factory(name: str = "vidA.mp4", seconds: float = 2.0, with_audio: bool = True) -> Path:
        return make_test_video(tmp_path / "videos" / name, seconds, with_audio)

    return _factory
