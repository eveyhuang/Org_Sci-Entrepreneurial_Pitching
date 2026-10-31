#!/usr/bin/env python3
"""Promote codes from a reviewed draft codebook into a working codebook.

Copies the chosen `## CODE-ID` blocks verbatim (parse -> render is lossless) into a new
Markdown master and generates its YAML. Wording is never altered.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.codebook_md import entries_to_md, md_to_entries, write_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--codes", nargs="+", default=None, help="default: every code in the draft")
    args = parser.parse_args()

    source, out = Path(args.source), Path(args.out)
    entries = md_to_entries(source.read_text(), source=str(source))
    if args.codes:
        by_id = {e["id"]: e for e in entries}
        missing = [c for c in args.codes if c not in by_id]
        if missing:
            raise SystemExit(f"not in {source}: {', '.join(missing)}")
        entries = [by_id[c] for c in args.codes]
    preamble = (
        f"# {args.title}\n\n"
        f"> **This Markdown file is the master copy**; the `.yaml` beside it is generated and is\n"
        f"> re-synced automatically whenever the codebook is loaded. Promoted from\n"
        f"> `{source}`. **Provenance** is documentation only and is never sent to the model.\n"
    )
    out.write_text(entries_to_md(entries, preamble))
    print(f"wrote {out} ({len(entries)} codes) and {write_yaml(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
