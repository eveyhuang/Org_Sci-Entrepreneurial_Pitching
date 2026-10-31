#!/usr/bin/env python3
"""Print the exact prompt a config's pass sends to the model.

The prompt is composed from two editable files — the Jinja template
(pipeline/prompts/*.j2, which holds the framing and rules) and the codebook YAML
(pipeline/codebooks/*.yaml, which holds the constructs). This renders them the way
a run does, so what you see is what the model gets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import load_config  # noqa: E402
from pipeline.prompts import render_pass_prompt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--pass", dest="pass_name", default=None,
                        help="pass to render (default: the config's first)")
    parser.add_argument("--hash-only", action="store_true",
                        help="print just the prompt hash, for scripting or diffing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pass_cfg = cfg.get_pass(args.pass_name) if args.pass_name else cfg.passes[0]
    text, digest = render_pass_prompt(cfg, pass_cfg)

    if args.hash_only:
        print(digest)
        return 0
    print(text)
    print(f"\n---\npass: {pass_cfg.name}  |  prompt hash: {digest}  |  {len(text)} chars",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
