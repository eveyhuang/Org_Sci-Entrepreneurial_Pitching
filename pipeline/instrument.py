"""The live instrument release: which codebooks, templates and schemas are current.

`pipeline/codebooks/current_instrument.yaml` is the single statement of what the current
experiment uses. `check_release` is enforced by the test suite and is called by the
panel runner before any API call: a canonical directory with an extra file,
a stale generated YAML, a missing template or an unimportable schema is a mixed bundle, and a
mixed bundle must not run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from pipeline.codebook_md import status
from pipeline.config import GENERATED_SCHEMA, import_schema
from pipeline.errors import ConfigError

CANONICAL_DIR = Path(__file__).resolve().parent / "codebooks"
DEFAULT_MANIFEST = CANONICAL_DIR / "current_instrument.yaml"
VALID_STATUS = ("development", "frozen")
REVIEW_LABELS_FILE = "review_labels.yaml"                # found beside the manifest by name; the manifest is
VALIDATION_THRESHOLDS_FILE = "validation_thresholds.yaml"  # hashed into every run and is not edited for them
VALIDATION_REPORTING_FILE = "validation_reporting.yaml"    # the thresholds file is kept as the October 8 record
OPTIONAL_FILES = {"review_labels": REVIEW_LABELS_FILE, "validation_thresholds": VALIDATION_THRESHOLDS_FILE,
                  "validation_reporting": VALIDATION_REPORTING_FILE}


@dataclass(frozen=True)
class BookRef:
    key: str
    markdown: Path
    yaml: Path
    template: str
    schema: str


@dataclass(frozen=True)
class InstrumentRelease:
    release_id: str
    status: str
    canonical_dir: Path
    templates_dir: Path
    manifest_path: Path
    books: dict[str, BookRef]
    review_labels: Path | None = None            # review_labels.yaml beside the manifest, when present
    validation_thresholds: Path | None = None    # validation_thresholds.yaml beside the manifest, when present
    validation_reporting: Path | None = None     # validation_reporting.yaml beside the manifest, when present


def load_release(manifest: Path | None = None) -> InstrumentRelease:
    path = Path(manifest) if manifest else DEFAULT_MANIFEST
    if not path.exists():
        raise ConfigError(f"instrument manifest not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"instrument manifest must be a mapping: {path}")
    base = path.parent
    release_id = data.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise ConfigError(f"{path}: release_id must be a non-empty string")
    state = data.get("status")
    if state not in VALID_STATUS:
        raise ConfigError(f"{path}: status must be one of {VALID_STATUS}, got {state!r}")
    templates_dir = (base / str(data.get("templates_dir", "../prompts"))).resolve()
    raw_books = data.get("books")
    if not isinstance(raw_books, dict) or not raw_books:
        raise ConfigError(f"{path}: books must be a non-empty mapping")
    books: dict[str, BookRef] = {}
    for key, entry in raw_books.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: book {key!r} must be a mapping")
        for field in ("markdown", "yaml", "template", "schema"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ConfigError(f"{path}: book {key!r} needs a string {field!r}")
        books[key] = BookRef(
            key=key,
            markdown=(base / entry["markdown"]).resolve(),
            yaml=(base / entry["yaml"]).resolve(),
            template=entry["template"],
            schema=entry["schema"],
        )
    optional = {key: (base / name).resolve() if (base / name).exists() else None
                for key, name in OPTIONAL_FILES.items()}
    return InstrumentRelease(
        release_id=release_id, status=state, canonical_dir=base.resolve(),
        templates_dir=templates_dir, manifest_path=path.resolve(), books=books, **optional,
    )


def canonical_files(release: InstrumentRelease, optional: bool = True) -> set[str]:
    """File names of the bundle; `optional=False` leaves out the page wording and the validation files, which no model
    pass reads."""
    names = {release.manifest_path.name}
    for book in release.books.values():
        names.add(book.markdown.name)
        names.add(book.yaml.name)
    for key in OPTIONAL_FILES if optional else ():
        if getattr(release, key) is not None:
            names.add(getattr(release, key).name)
    return names


def check_release(release: InstrumentRelease) -> list[str]:
    """Problems that make the bundle unfit to run. Empty list means ready."""
    problems: list[str] = []
    allowed = canonical_files(release)
    for entry in sorted(release.canonical_dir.iterdir()):
        if entry.name == "__pycache__":
            continue
        if entry.name not in allowed:
            problems.append(f"unexpected file in canonical directory: {entry.name}")
    for book in release.books.values():
        if not book.markdown.exists():
            problems.append(f"{book.key}: markdown missing: {book.markdown}")
        if not book.yaml.exists():
            problems.append(f"{book.key}: yaml missing: {book.yaml}")
        elif book.markdown.exists():
            state = status(book.yaml)
            if state != "in-sync":
                problems.append(f"{book.key}: generated yaml is {state}: {book.yaml.name}")
        if not (release.templates_dir / book.template).exists():
            problems.append(f"{book.key}: template not found: {book.template}")
        if book.schema != GENERATED_SCHEMA:
            try:
                import_schema(book.schema)
            except ConfigError as exc:
                problems.append(f"{book.key}: schema unimportable: {exc}")
    return problems
