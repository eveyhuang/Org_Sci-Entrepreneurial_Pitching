#!/usr/bin/env python3
"""List candidate codebook examples from the dev set (run aggregate.py first)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import load_config  # noqa: E402
from pipeline.runs import create_or_resume_run, tables_dir  # noqa: E402
from pipeline.stage_b.examples import candidates_markdown, find_candidates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="example_candidates.md")
    parser.add_argument("--per-answer", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tables = tables_dir(create_or_resume_run(cfg, runs_root=args.runs_root))
    frame = find_candidates(
        pd.read_parquet(tables / "annotations.parquet"), pd.read_parquet(tables / "turns.parquet"),
        dev_ids=cfg.dev_video_ids(), per_answer=args.per_answer)
    Path(args.out).write_text(candidates_markdown(frame))
    print(f"{len(frame)} candidates across {frame.code_id.nunique() if len(frame) else 0} codes -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
