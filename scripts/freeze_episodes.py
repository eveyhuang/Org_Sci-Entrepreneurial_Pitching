#!/usr/bin/env python3
"""Review and freeze proposed episode boundaries.

  --export review.csv            write every proposed row for human review
  --freeze [--from review.csv]   freeze (importing the reviewer's edits first). On import,
                                  start_s/end_s are always recomputed from the start_turn/
                                  end_turn you give — the CSV's own seconds are ignored.
  --unfreeze VIDEO_ID ...        reopen videos so they can be re-segmented or re-edited
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import load_config  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.stage_b.freeze import export_review, freeze, import_review, unfreeze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default="registry.sqlite")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--export", metavar="CSV")
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--unfreeze", nargs="+", metavar="VIDEO_ID")
    parser.add_argument("--from", dest="from_csv", metavar="CSV", help="with --freeze: import edits first")
    parser.add_argument("--video-ids", nargs="+", default=None, help="with --freeze: only these videos")
    args = parser.parse_args()

    cfg = load_config(args.config)
    registry = Registry(args.registry)
    try:
        if args.export:
            print(f"exported {export_review(registry, cfg.corpus.name, args.export)} rows to {args.export}")
        elif args.unfreeze:
            print(f"unfroze {unfreeze(registry, args.unfreeze)} videos")
        else:
            if args.from_csv:
                replaced, skipped = import_review(registry, args.from_csv, cfg)
                print(f"imported edits for {replaced} videos ({skipped} already frozen, left alone)")
            print(f"froze {freeze(registry, cfg.corpus.name, args.video_ids)} videos")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        registry.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
