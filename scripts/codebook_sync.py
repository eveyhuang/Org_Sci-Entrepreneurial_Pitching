#!/usr/bin/env python3
"""Keep generated codebook YAML in step with its Markdown master.

You rarely need this: every codebook load (so every annotate run) syncs automatically.
Use it to see what state things are in, to regenerate on demand, or to create a
Markdown master from an existing hand-written YAML.

  python scripts/codebook_sync.py                         # sync every managed codebook
  python scripts/codebook_sync.py --check                 # report only, exit 1 on problems
  python scripts/codebook_sync.py path/to/book.md         # sync one
  python scripts/codebook_sync.py path/to/book.md --force # overwrite a hand-edited YAML
  python scripts/codebook_sync.py --from-yaml path/to/book.yaml   # one-time bootstrap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from pipeline.codebook_md import entries_to_md, illustrative_examples, status, write_yaml  # noqa: E402
from pipeline.errors import ConfigError  # noqa: E402

from pipeline.codebook_md import is_archived  # noqa: E402

WRITABLE_DIRS = (REPO / "pipeline" / "codebooks", REPO / "docs" / "codebooks" / "drafts")
ARCHIVE_DIR = REPO / "docs" / "codebooks" / "archive"


def _is_master(md: Path) -> bool:
    text = md.read_text()
    return "\n## " in text and "**Definition.**" in text


def writable_markdown() -> list[Path]:
    """Masters that ordinary sync may regenerate: canonical and drafts."""
    return [md for base in WRITABLE_DIRS if base.exists() for md in sorted(base.rglob("*.md")) if _is_master(md)]


def archived_markdown() -> list[Path]:
    """Frozen masters: checked, never written."""
    return [md for md in sorted(ARCHIVE_DIR.rglob("*.md")) if _is_master(md)] if ARCHIVE_DIR.exists() else []


def managed_markdown() -> list[Path]:
    return writable_markdown() + archived_markdown()


def bootstrap(yaml_path: Path) -> Path:
    md_path = yaml_path.with_suffix(".md")
    if md_path.exists():
        raise SystemExit(f"{md_path} already exists — refusing to overwrite a Markdown master")
    raw = yaml_path.read_text()
    comments = [line[1:].strip() for line in raw.splitlines() if line.startswith("#")]
    title = yaml_path.stem.replace("_", " ").replace(".", " ")
    preamble = f"# {title}\n\n" + "\n".join(comments).strip()
    md_path.write_text(entries_to_md(yaml.safe_load(raw), preamble))
    write_yaml(md_path)
    return md_path


def process(md: Path, *, check: bool, force: bool, no_illustrative: bool) -> int:
    """Report or regenerate one master. Returns the number of problems found."""
    from pipeline.codebook_md import render_yaml

    yaml_path = md.with_suffix(".yaml")
    state = status(yaml_path)
    label = md.relative_to(REPO) if md.is_relative_to(REPO) else md
    frozen = is_archived(md) or (ARCHIVE_DIR in md.resolve().parents)
    try:
        if check or frozen:
            render_yaml(md)  # parses + validates without writing
            tag = f"{state} (frozen)" if frozen else state
            print(f"{tag:22s} {label}")
            problems = int(state != "in-sync")
            if frozen and state != "in-sync":
                print("  archive records are never regenerated; restore the file from git")
            if check and no_illustrative:
                for code_id, text in illustrative_examples(md):
                    print(f"  illustrative  [{code_id}] {text[:70]}")
                    problems += 1
            return problems
        if state == "in-sync":
            print(f"{'in-sync':22s} {label}")
            return 0
        if state in ("hand-edited", "unmanaged") and not force:
            print(f"{state:22s} {label}  — re-run with --force to regenerate from the Markdown")
            return 1
        write_yaml(md)
        print(f"{'regenerated':22s} {label}")
        return 0
    except ConfigError as exc:
        print(f"{'INVALID':22s} {label}\n  {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("markdown", nargs="*", type=Path)
    parser.add_argument("--check", action="store_true", help="report only; exit 1 on any problem")
    parser.add_argument("--no-illustrative", action="store_true",
                        help="with --check: fail while any [illustrative] example remains "
                             "(run before a codebook freeze)")
    parser.add_argument("--force", action="store_true", help="overwrite YAML that was edited by hand")
    parser.add_argument("--from-yaml", type=Path, help="create a Markdown master from this YAML")
    args = parser.parse_args()

    if args.from_yaml:
        print(f"created {bootstrap(args.from_yaml.resolve())}")
        return 0

    problems = 0
    for md in [m.resolve() for m in args.markdown] or managed_markdown():
        problems += process(md, check=args.check, force=args.force, no_illustrative=args.no_illustrative)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
