"""Run directories, output paths, and the reproducibility manifest.

A run directory is reused when its frozen config.yaml is byte-identical to the
config being loaded, so `annotate.py` resumes into the same directory instead of
scattering partial output across timestamped folders.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import PipelineConfig, config_hash

MANIFEST_NAME = "manifest.json"
CONFIG_NAME = "config.yaml"


@dataclass(frozen=True)
class Run:
    run_id: str
    root: Path
    config_hash: str


def create_or_resume_run(
    cfg: PipelineConfig,
    runs_root: Path | str = Path("runs"),
    now: datetime | None = None,
    force_new: bool = False,
) -> Run:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    digest = config_hash(cfg)

    if not force_new:
        candidates = sorted(
            (p for p in runs_root.glob(f"{cfg.run_name}_*") if p.is_dir()), reverse=True
        )
        for candidate in candidates:
            frozen = candidate / CONFIG_NAME
            if frozen.exists() and frozen.read_text() == cfg.raw_yaml:
                return Run(run_id=candidate.name, root=candidate, config_hash=digest)

    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    root = runs_root / f"{cfg.run_name}_{stamp}"
    suffix = 1
    while root.exists():
        root = runs_root / f"{cfg.run_name}_{stamp}_{suffix}"
        suffix += 1
    for sub in ("raw", "parsed", "prompts", "timelines", "tables"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / CONFIG_NAME).write_text(cfg.raw_yaml)
    run = Run(run_id=root.name, root=root, config_hash=digest)
    write_manifest(run, {"config_hash": digest, "config_source": str(cfg.source_path)})
    return run


def _unit_filename(
    video_id: str, k: int, span: tuple[float, float] | None, prompt_hash: str
) -> str:
    """Filename for one unit's output.

    The prompt hash is part of the name because a run directory is reused whenever
    config.yaml is byte-identical, and the codebook/template are NOT part of that
    text. Without it, editing the codebook would overwrite the previous run's
    records in place and destroy the audit trail.
    """
    stem = video_id if span is None else f"{video_id}__{span[0]:.3f}_{span[1]:.3f}"
    return f"{stem}__k{k}__{prompt_hash[:8]}.json"


def raw_path(
    run: Run,
    pass_name: str,
    video_id: str,
    k: int,
    span: tuple[float, float] | None = None,
    *,
    prompt_hash: str,
) -> Path:
    directory = run.root / "raw" / pass_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _unit_filename(video_id, k, span, prompt_hash)


def parsed_path(
    run: Run,
    pass_name: str,
    video_id: str,
    k: int,
    span: tuple[float, float] | None = None,
    *,
    prompt_hash: str,
) -> Path:
    directory = run.root / "parsed" / pass_name
    directory.mkdir(parents=True, exist_ok=True)
    return directory / _unit_filename(video_id, k, span, prompt_hash)


def prompt_path(run: Run, pass_name: str, prompt_hash: str) -> Path:
    """Where a run archives the exact prompt text it sent.

    The prompt is rendered from the template plus the codebook, so reconstructing
    it later would mean checking out the right commit. Archiving it per run makes
    each run self-describing: the manifest names the hash, this file is the text.
    """
    directory = run.root / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{pass_name}__{prompt_hash}.txt"


def schema_path(run: Run, pass_name: str, prompt_hash: str) -> Path:
    """Where a run archives the exact Gemini response schema it sent.

    Kept alongside the prompt text it was sent with, under the same hash: together they
    fully describe the wire contract for the pass, without a checkout of the codebook that
    generated the schema.
    """
    directory = run.root / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{pass_name}__{prompt_hash}.schema.json"


def timeline_path(run: Run, video_id: str) -> Path:
    directory = run.root / "timelines"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{video_id}.json"


def tables_dir(run: Run) -> Path:
    directory = run.root / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_manifest(run: Run) -> dict:
    path = run.root / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_manifest(run: Run, data: dict) -> None:
    payload = {**data, "run_id": run.run_id, "git_sha": git_sha()}
    (run.root / MANIFEST_NAME).write_text(json.dumps(payload, indent=2, sort_keys=True))


def update_manifest(run: Run, patch: dict) -> dict:
    manifest = read_manifest(run)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(manifest.get(key), dict):
            manifest[key].update(value)
        else:
            manifest[key] = value
    write_manifest(run, {k: v for k, v in manifest.items() if k not in ("run_id", "git_sha")})
    return read_manifest(run)


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout.strip() or None) if result.returncode == 0 else None
