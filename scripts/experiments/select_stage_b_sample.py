#!/usr/bin/env python3
"""Choose the Stage B videos for a batch run and write runs/<run>/selection.json.

Random draw with a fixed seed from registered Shark Tank videos, excluding:
  - videos flagged by ingest (too_long, no_pitch);
  - videos already coded with the final pipeline (--exclude, default: the five pilot pitches);
  - titles that are plainly not a single pitch (compilations, quizzes), decided by title before
    any outcome is seen.
The rest of the shuffled order is kept as `reserve`. If a drawn video turns out not to be a single
pitch (the front end stops on it), replace it with the next reserve video: `--n` one higher, same seed.
Videos passed with --include (e.g. Copy Keyboard, to re-code it with the final pipeline) are added first.
    .venv/bin/python scripts/experiments/select_stage_b_sample.py --run-name stage_b_sample50_frontend --n 44 --include st_1f094ea0ad
"""
from __future__ import annotations
import argparse, json, random, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PILOT = ["st_8642720802", "st_d218d68cff", "st_38f8bf18e0", "st_aebb7e3764", "st_ea076612ce"]
SHOW = "Shark Tank US"
NOT_A_PITCH = ["Everyday Things That Got Their Start", "Guess Who Said It", "Funny Moments From The Sharks"]
# Drawn videos that the front end's segmentation found are not one pitch. Like other shows, they are dropped
# after the shuffle, so the next reserve video takes the slot and every other video keeps its place.
FOUND_NOT_A_PITCH = {"st_d5931ea7f5": "segmentation found 13 pitches (compilation: 'And For Those Reasons I\u2019m Out')"}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--n", type=int, required=True, help="number of new videos to draw")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--include", nargs="*", default=[], help="video ids to add regardless of the draw")
    ap.add_argument("--exclude", nargs="*", default=PILOT, help="video ids already coded (default: the five pilot pitches)")
    ap.add_argument("--registry", default=str(ROOT / "registry.sqlite"))
    a = ap.parse_args()

    db = sqlite3.connect(f"file:{a.registry}?mode=ro", uri=True); db.row_factory = sqlite3.Row
    rows = db.execute("SELECT video_id, path, duration_s, meta_json, skip_reason FROM videos "
                      "WHERE corpus='shark-tank' AND probe_ok=1 ORDER BY video_id").fetchall()
    title = lambda r: json.loads(r["meta_json"]).get("title", "")
    excluded = [{"video_id": r["video_id"], "reason": f"ingest: {r['skip_reason']}"} for r in rows if r["skip_reason"]]
    excluded += [{"video_id": r["video_id"], "reason": "already coded with the final pipeline"} for r in rows if r["video_id"] in a.exclude]
    excluded += [{"video_id": r["video_id"], "reason": f"not a single pitch (title: {title(r)[:60]})"}
                 for r in rows if any(p in title(r) for p in NOT_A_PITCH)]
    out_ids = {e["video_id"] for e in excluded} | set(a.include)
    eligible = [r for r in rows if r["video_id"] not in out_ids]
    order = eligible[:]; random.Random(a.seed).shuffle(order)
    # Other shows (e.g. one Dragons' Den upload) are dropped AFTER the shuffle, so every other video keeps its
    # place in the draw; the first reserve video takes the dropped one's slot (2026-09-28).
    show = lambda r: (title(r).split("|") + ["", ""])[1].strip()   # "<pitch> | <show> | <channel>"; a Dragons' Den upload also carries "Shark Tank US"
    other_show = [r for r in order if show(r) != SHOW]
    excluded += [{"video_id": r["video_id"], "reason": f"other show: {show(r)} (title: {title(r)[:50]})"} for r in other_show]
    order = [r for r in order if show(r) == SHOW]
    excluded += [{"video_id": r["video_id"], "reason": "not a single pitch, found after the draw: " + FOUND_NOT_A_PITCH[r["video_id"]]} for r in order if r["video_id"] in FOUND_NOT_A_PITCH]
    order = [r for r in order if r["video_id"] not in FOUND_NOT_A_PITCH]
    if a.n > len(order): sys.exit(f"only {len(order)} eligible videos; asked for {a.n}")
    as_row = lambda r: {"video_id": r["video_id"], "path": r["path"], "duration_s": r["duration_s"], "meta_json": r["meta_json"]}
    by_id = {r["video_id"]: r for r in rows}
    missing = [v for v in a.include if v not in by_id]
    if missing: sys.exit(f"--include ids not in the registry: {missing}")
    videos = [as_row(by_id[v]) for v in a.include] + [as_row(r) for r in order[:a.n]]
    run = ROOT / "runs" / a.run_name; run.mkdir(parents=True, exist_ok=True)
    (run / "selection.json").write_text(json.dumps({
        "selection_seed": a.seed,
        "selection": f"Random draw of {a.n} from {len(eligible)} eligible registered Shark Tank videos; excluded before outcomes: "
                     "ingest flags, videos already coded with the final pipeline, titles that are not a single pitch; "
                     f"videos from another show are dropped after the shuffle (their slot goes to the next video). "
                     f"Added regardless of the draw: {a.include or 'none'}.",
        "excluded": excluded,
        "videos": videos,
        "reserve": [r["video_id"] for r in order[a.n:]],
    }, indent=1) + "\n")
    print(f"{len(videos)} videos -> {run/'selection.json'} ({len(eligible)} eligible, {len(order) - a.n} in reserve, {len(excluded)} excluded)")
    for v in videos: print(" ", v["video_id"], json.loads(v["meta_json"]).get("title", "")[:70])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
