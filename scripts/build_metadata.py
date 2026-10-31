#!/usr/bin/env python3
"""Write a metadata CSV (video_id, filename, title) for a folder of raw-titled videos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.ingest import write_metadata_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    count = write_metadata_csv(Path(args.video_dir), Path(args.out))
    print(f"wrote {count} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
