"""Corpus ingest: discover videos, probe them, register them.

Identity always comes from the metadata CSV's `video_id`, never from a sanitized
display name — the prior pipeline's habit of keying on mangled filenames is what
made its state unrecoverable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.registry import Registry, VideoRow

VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi")
SKIP_SUFFIXES = (".part", ".ytdl", ".tmp")


def _nfc(text: str) -> str:
    # macOS may hand back decomposed (NFD) names; a CSV saved by an editor is usually NFC.
    return unicodedata.normalize("NFC", text)


def video_id_for_filename(filename: str) -> str:
    """Stable, shell-safe id for a file whose name is a raw video title."""
    return "st_" + hashlib.sha1(_nfc(filename).encode("utf-8")).hexdigest()[:10]


def is_video_file(path: Path) -> bool:
    """A complete video file: a known video suffix and not an unfinished download."""
    return (
        path.is_file()
        and path.suffix.lower() not in SKIP_SUFFIXES
        and path.suffix.lower() in VIDEO_SUFFIXES
    )


def build_metadata_rows(video_dir: Path) -> list[dict[str, str]]:
    return [
        {"video_id": video_id_for_filename(p.name), "filename": p.name, "title": p.stem}
        for p in sorted(Path(video_dir).iterdir())
        if is_video_file(p)
    ]


def write_metadata_csv(video_dir: Path, out_csv: Path) -> int:
    rows = build_metadata_rows(video_dir)
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "filename", "title"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


@dataclass(frozen=True)
class ProbeResult:
    duration_s: float | None
    width: int | None
    height: int | None
    has_audio: bool


@dataclass
class IngestSummary:
    discovered: int = 0
    registered: int = 0
    skipped: int = 0
    failed: int = 0
    too_long: int = 0


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> ProbeResult:
    """Read duration/dimensions/audio presence. Raises on unreadable media."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:300]}")
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = data.get("format", {}).get("duration")
    return ProbeResult(
        duration_s=float(duration) if duration else None,
        width=int(video["width"]) if video and video.get("width") else None,
        height=int(video["height"]) if video and video.get("height") else None,
        has_audio=has_audio,
    )


def read_metadata(csv_path: Path) -> dict[str, dict[str, str]]:
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "video_id" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must have a video_id column")
        return {row["video_id"]: dict(row) for row in reader if row.get("video_id")}


def discover_videos(
    video_dir: Path, metadata: dict[str, dict[str, str]]
) -> list[tuple[str, Path]]:
    """Match files on disk to metadata ids. Unfinished downloads are skipped.

    Two modes. When the metadata has a `filename` column, files are matched by exact
    filename and the id comes from the CSV (files whose names are raw titles are never
    renamed). Otherwise the id is the filename stem, as in the Hu & Ma corpus.
    """
    by_filename = {
        _nfc(row["filename"]): video_id for video_id, row in metadata.items() if row.get("filename")
    }
    found: dict[str, Path] = {}
    for path in sorted(video_dir.rglob("*")):
        if not is_video_file(path):
            continue
        if by_filename:
            video_id = by_filename.get(_nfc(path.name))
            if video_id is None:
                continue
        else:
            video_id = path.name.split(".")[0]
            if video_id not in metadata:
                continue
        # Prefer .mp4 when several containers exist for one id.
        existing = found.get(video_id)
        if existing is None or (existing.suffix.lower() != ".mp4" and path.suffix.lower() == ".mp4"):
            found[video_id] = path
    return sorted(found.items())


def _skip_reason(cfg: PipelineConfig, duration_s: float | None) -> str | None:
    limit = cfg.corpus.max_duration_s
    # A mechanical guard, not curation: a clip this long will not fit one model call.
    return "too_long" if limit is not None and duration_s is not None and duration_s > limit else None


def ingest_corpus(
    cfg: PipelineConfig,
    registry: Registry,
    video_ids: list[str] | None = None,
    limit: int | None = None,
) -> IngestSummary:
    metadata = read_metadata(cfg.corpus.metadata_csv)
    discovered = discover_videos(cfg.corpus.video_dir, metadata)
    if video_ids:
        wanted = set(video_ids)
        discovered = [(vid, path) for vid, path in discovered if vid in wanted]
    if limit is not None:
        discovered = discovered[:limit]

    summary = IngestSummary(discovered=len(discovered))
    for video_id, path in discovered:
        existing = registry.get_video(video_id)
        if existing is not None and existing.probe_ok and existing.path == str(path):
            reason = _skip_reason(cfg, existing.duration_s)
            if reason != existing.skip_reason:
                registry.set_skip_reason(video_id, reason)
            summary.too_long += reason is not None
            summary.skipped += 1
            continue
        try:
            probe = ffprobe(path)
        except (RuntimeError, OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            # unreadable media / ffprobe unusable is data, not a crash; a real
            # programming bug (KeyError, AttributeError, ...) still propagates
            registry.upsert_video(
                VideoRow(
                    video_id=video_id, corpus=cfg.corpus.name, path=str(path),
                    probe_ok=False, meta_json=json.dumps({"probe_error": str(exc)[:500]}),
                )
            )
            summary.failed += 1
            print(f"[FAIL] {video_id}: {exc}")
            continue
        registry.upsert_video(
            VideoRow(
                video_id=video_id, corpus=cfg.corpus.name, path=str(path),
                sha256=sha256_file(path), duration_s=probe.duration_s, width=probe.width,
                height=probe.height, has_audio=probe.has_audio, probe_ok=True,
                meta_json=json.dumps(metadata.get(video_id, {})),
                skip_reason=_skip_reason(cfg, probe.duration_s),
            )
        )
        summary.registered += 1
        summary.too_long += _skip_reason(cfg, probe.duration_s) is not None
        print(f"[OK] {video_id} ({probe.duration_s:.1f}s)" if probe.duration_s else f"[OK] {video_id}")
    return summary
