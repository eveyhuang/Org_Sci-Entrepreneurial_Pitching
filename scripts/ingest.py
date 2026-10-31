#!/usr/bin/env python3
"""Discover, probe, and register a corpus's videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import load_config  # noqa: E402
from pipeline.ingest import ingest_corpus  # noqa: E402
from pipeline.registry import Registry  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a run config YAML")
    parser.add_argument("--registry", default="registry.sqlite")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--video-ids", nargs="+", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = Registry(args.registry)
    try:
        summary = ingest_corpus(cfg, registry, video_ids=args.video_ids, limit=args.limit)
    finally:
        registry.close()
    print(
        f"ingest: discovered={summary.discovered} registered={summary.registered} "
        f"skipped={summary.skipped} failed={summary.failed} too_long={summary.too_long}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
