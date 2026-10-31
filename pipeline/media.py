"""ffmpeg-backed episode clipping, cached on disk.

Clips are re-encoded rather than stream-copied so the cut lands on the requested
timestamps instead of snapping to the nearest keyframe: an episode boundary that
drifts by a second corrupts the evidence spans the codebook requires, and these
are short files where the extra encode cost is irrelevant.

There is deliberately no media-derivation function here. Every pass sends the
complete audio-video stream; degraded-input conditions are not part of the design.
"""

from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path

DEFAULT_CLIP_DIR = Path("data/clips")

# Two threads cutting the same [src, start_s, end_s) race on the shared `.tmp.mp4` ->
# `.mp4` rename: the loser's `tmp.replace(out)` (or its `tmp.unlink` on failure) can hit a
# file the winner already moved/removed, raising FileNotFoundError. One lock per output
# path serialises same-clip callers; the dict itself is guarded by a single lock.
_locks_guard = threading.Lock()
_locks: dict[Path, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        lock = _locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _locks[path] = lock
        return lock


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])}…\n{result.stderr.strip()[:500]}")


def _tag(src: Path) -> str:
    """Short digest of path + mtime + size, so edited sources invalidate caches."""
    stat = src.stat()
    payload = f"{src.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def clip(
    src: Path, start_s: float, end_s: float, cache_dir: Path | str = DEFAULT_CLIP_DIR
) -> Path:
    """Cut [start_s, end_s) from `src`, returning a cached mp4."""
    if end_s <= start_s:
        raise ValueError(f"end_s ({end_s}) must be greater than start_s ({start_s})")
    if start_s < 0:
        raise ValueError(f"start_s must be non-negative, got {start_s}")
    out_dir = Path(cache_dir) / src.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{_tag(src)}__{start_s:.3f}_{end_s:.3f}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    with _lock_for(out):
        if out.exists() and out.stat().st_size > 0:
            return out
        tmp = out.with_suffix(".tmp.mp4")
        try:
            _run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(src),
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-movflags", "+faststart", str(tmp),
            ])
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(out)
        return out
