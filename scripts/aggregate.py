#!/usr/bin/env python3
"""Build tidy parquet tables from a run's parsed records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.aggregate import build_tables  # noqa: E402
from pipeline.config import load_config  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.runs import create_or_resume_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default="registry.sqlite")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = Registry(args.registry)
    try:
        run = create_or_resume_run(cfg, runs_root=args.runs_root)
        completed = registry.done_units(run.run_id)
        tables = build_tables(cfg, registry, run)
    finally:
        registry.close()

    if not completed:
        print(
            f"warning: run {run.run_id} has no completed annotation units, so the tables "
            f"below are empty. Run scripts/annotate.py --config {args.config} first.",
            file=sys.stderr,
        )
    for name, path in tables.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
