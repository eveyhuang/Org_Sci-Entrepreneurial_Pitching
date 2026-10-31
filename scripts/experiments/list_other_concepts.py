#!/usr/bin/env python3
"""List every concept or exchange topic coded `other` in a panel run, with the name the model gave it.

Reads <run>/<video>/results/primary.json and replicate.json of every video in the run and writes
one Markdown table (video, pass, time, where, code, name, and the turn summary or exchange label),
then a tally of the names grouped without regard to case. The list is for a person looking for
recurring matters that the concept list does not cover. Offline; reads results only and writes
nothing but the --out file. Every name in it is model-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.stage_b.panel_books import other_concepts  # noqa: E402

PASSES = ("primary", "replicate")
UNNAMED = "(unnamed)"
HEADER = ["video", "pass", "time", "where", "code", "name", "turn summary or exchange label"]
DASH = "—"


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def collect(run: Path) -> list[dict]:
    """One row per `other` item, by video id, then pass, then turns before exchanges."""
    rows = []
    for vdir in sorted(p for p in Path(run).iterdir() if p.is_dir()):
        for name in PASSES:
            result = _read(vdir / "results" / f"{name}.json")
            if not isinstance(result, dict) or not isinstance(result.get("record"), dict):
                continue
            record = result["record"]
            summary = {}
            for t in record["turns"]:
                summary.setdefault(t["turn_id"], t["summary"])
            label = {}
            for e in record["exchanges"]:
                label.setdefault(e["exchange_id"], e["label"])
            for item in other_concepts(record):
                context = summary[item["id"]] if item["where"] == "turn" else label[item["id"]]
                rows.append({"video_id": vdir.name, "pass": name, **item, "context": context})
    return rows


def tally(rows: list[dict]) -> list[tuple[str, int]]:
    """(name as first written, count), names grouped case-insensitively, most frequent first."""
    shown: dict[str, str] = {}
    counts: dict[str, int] = {}
    for r in rows:
        name = r["name"] or UNNAMED
        key = name.casefold()
        shown.setdefault(key, name)
        counts[key] = counts.get(key, 0) + 1
    return [(shown[k], n) for k, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list]) -> list[str]:
    def line(cells):
        return "| " + " | ".join(_cell(c) for c in cells) + " |"
    return [line(header), "|" + "|".join("---" for _ in header) + "|"] + [line(r) for r in rows]


def render(run: Path, rows: list[dict]) -> str:
    L = [f"# Other concepts named: {Path(run).name}", "",
         "Every name below is model-coded and not human-validated.", ""]
    if not rows:
        return "\n".join(L + ["No other concepts in this run."]) + "\n"
    videos = len({r["video_id"] for r in rows})
    L += [f"{len(rows)} item{'s' if len(rows) != 1 else ''} in {videos} video{'s' if videos != 1 else ''}.", ""]
    L += _table(HEADER, [[r["video_id"], r["pass"], r["clock"] or DASH, r["where"], r["code"], r["name"] or UNNAMED,
                          r["context"]] for r in rows])
    L += ["", "## Names", "", "Grouped without regard to case; each pass of a video counts separately.", ""]
    L += _table(["name", "count"], [[name, n] for name, n in tally(rows)])
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True, help="a panel run directory (one folder per video)")
    ap.add_argument("--out", type=Path, help="the Markdown file to write; printed when omitted")
    args = ap.parse_args(argv)
    if not args.run.is_dir():
        ap.error(f"{args.run} is not a run directory")
    text = render(args.run, collect(args.run))
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
