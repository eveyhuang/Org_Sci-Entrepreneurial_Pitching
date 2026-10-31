import threading

import pytest

from pipeline.ingest import ffprobe
from pipeline.media import clip
from tests.conftest import needs_ffmpeg


@needs_ffmpeg
def test_clip_extracts_requested_span(tmp_path, video_factory):
    src = video_factory("vidA.mp4", seconds=4.0)
    out = clip(src, 1.0, 3.0, cache_dir=tmp_path / "clips")
    probe = ffprobe(out)
    assert 1.6 < probe.duration_s < 2.4


@needs_ffmpeg
def test_clip_is_cached_and_not_recut(tmp_path, video_factory):
    src = video_factory("vidA.mp4", seconds=4.0)
    first = clip(src, 1.0, 3.0, cache_dir=tmp_path / "clips")
    stamp = first.stat().st_mtime_ns
    second = clip(src, 1.0, 3.0, cache_dir=tmp_path / "clips")
    assert second == first
    assert second.stat().st_mtime_ns == stamp


@needs_ffmpeg
def test_clip_rejects_inverted_span(tmp_path, video_factory):
    src = video_factory("vidA.mp4", seconds=4.0)
    with pytest.raises(ValueError, match="end_s"):
        clip(src, 3.0, 1.0, cache_dir=tmp_path / "clips")


@needs_ffmpeg
def test_clip_rejects_negative_start(tmp_path, video_factory):
    src = video_factory("vidA.mp4", seconds=4.0)
    with pytest.raises(ValueError, match="start_s"):
        clip(src, -1.0, 2.0, cache_dir=tmp_path / "clips")


@needs_ffmpeg
def test_clip_reflects_source_edits_via_cache_tag(tmp_path, video_factory):
    src = video_factory("vidA.mp4", seconds=4.0)
    first = clip(src, 0.0, 2.0, cache_dir=tmp_path / "clips")
    video_factory("vidA.mp4", seconds=6.0)  # rewrite the source
    second = clip(src, 0.0, 2.0, cache_dir=tmp_path / "clips")
    assert second != first  # cache tag includes mtime+size, so the stale clip is not reused


@needs_ffmpeg
def test_concurrent_clip_of_the_same_range_does_not_race(tmp_path, video_factory):
    """Two threads cutting the identical [start, end) must not FileNotFoundError on the
    shared tmp -> final rename; both should end up with the one cached file."""
    src = video_factory("vidA.mp4", seconds=4.0)
    cache = tmp_path / "clips"
    results: list[object] = [None, None]
    errors: list[BaseException] = []

    def run(idx: int) -> None:
        try:
            results[idx] = clip(src, 1.0, 3.0, cache_dir=cache)
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert results[0] is not None and results[0] == results[1]
    assert results[0].exists() and results[0].stat().st_size > 0
    assert list(cache.rglob("*.tmp.mp4")) == []


@needs_ffmpeg
def test_failed_clip_leaves_no_temp_file(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    cache = tmp_path / "clips"
    with pytest.raises(RuntimeError):
        clip(broken, 0.0, 1.0, cache_dir=cache)
    leftovers = list(cache.rglob("*.tmp.mp4"))
    assert leftovers == [], f"orphaned temp files: {leftovers}"
