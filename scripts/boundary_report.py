#!/usr/bin/env python3
"""Episode-boundary agreement for a run: replicate stability, and model vs human.

Stability compares the k replicates of the segmentation pass. With --human, the model's
k=0 segmentation is compared with a reviewer-edited CSV from freeze_episodes.py --export.

This is a read-only report: it never creates or resumes a run directory. It finds the
newest existing run whose config.yaml matches the config being loaded (or, with --run-id,
the named run) and reads from it; it never writes anything under runs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PipelineConfig, load_config  # noqa: E402
from pipeline.registry import Registry  # noqa: E402
from pipeline.reliability.boundaries import (  # noqa: E402
    boundary_agreement, episode_jaccard, pairwise_agreement,
)
from pipeline.runs import CONFIG_NAME, Run  # noqa: E402


def _find_run(cfg: PipelineConfig, runs_root: Path | str, run_id: str | None) -> Run | None:
    """Newest existing run directory whose frozen config.yaml matches this config.

    Same matching rule as `pipeline.runs.create_or_resume_run`, but never creates a new
    directory: a report has no business writing to paid-for run state.
    """
    runs_root = Path(runs_root)
    if run_id:
        candidate = runs_root / run_id
        if candidate.is_dir() and (candidate / CONFIG_NAME).exists():
            return Run(run_id=candidate.name, root=candidate, config_hash="")
        return None
    candidates = sorted(
        (p for p in runs_root.glob(f"{cfg.run_name}_*") if p.is_dir()), reverse=True
    )
    for candidate in candidates:
        frozen = candidate / CONFIG_NAME
        if frozen.exists() and frozen.read_text() == cfg.raw_yaml:
            return Run(run_id=candidate.name, root=candidate, config_hash="")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--registry", default="registry.sqlite")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--run-id", default=None, help="name a run directory explicitly")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--human", default=None, help="reviewer-edited CSV to compare k=0 against")
    args = parser.parse_args()

    cfg = load_config(args.config)
    producer = cfg.pass_producing("episodes")
    if producer is None:
        raise SystemExit("this config has no pass with `produces: episodes`")

    run = _find_run(cfg, args.runs_root, args.run_id)
    if run is None:
        where = f"--run-id {args.run_id}" if args.run_id else f"matching {args.config}"
        print(
            f"ERROR: no run directory found under {args.runs_root} ({where}). "
            f"Run scripts/annotate.py --config {args.config} first.",
            file=sys.stderr,
        )
        return 1

    registry = Registry(args.registry)
    try:
        spans: dict[str, dict[int, list]] = defaultdict(dict)
        for unit in registry.done_units(run.run_id, pass_name=producer.name):
            record = json.loads(Path(unit.parsed_path).read_text())["record"]
            spans[unit.video_id][unit.k] = [(e["start_turn"], e["end_turn"]) for e in record["episodes"]]
    finally:
        registry.close()

    if not spans:
        print(
            f"warning: run {run.run_id} has no completed segmentation units, so there is "
            f"nothing to report. Run scripts/annotate.py --config {args.config} first.",
            file=sys.stderr,
        )
        return 0

    print(f"{'video':14s} {'k':>2s} {'episodes':>10s} {'jaccard':>8s} {'f1':>6s}")
    rows = []
    for video_id, by_k in sorted(spans.items()):
        result = pairwise_agreement(by_k, args.tolerance)
        counts = "/".join(str(len(by_k[k])) for k in sorted(by_k))
        if result["pairs"]:
            rows.append(result)
            print(f"{video_id:14s} {len(by_k):>2d} {counts:>10s} {result['jaccard']:8.3f} {result['f1']:6.3f}")
        else:
            print(f"{video_id:14s} {len(by_k):>2d} {counts:>10s} {'—':>8s} {'—':>6s}")
    if rows:
        print(f"\nstability over {len(rows)} videos (±{args.tolerance} turns): "
              f"jaccard={sum(r['jaccard'] for r in rows) / len(rows):.3f}  "
              f"f1={sum(r['f1'] for r in rows) / len(rows):.3f}")

    if args.human:
        human: dict[str, list] = defaultdict(list)
        with open(args.human, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["kind"] == "episode":
                    human[row["video_id"]].append((int(row["start_turn"]), int(row["end_turn"])))
        shared = [v for v in human if 0 in spans.get(v, {})]
        if shared:
            jac = [episode_jaccard(spans[v][0], sorted(human[v])) for v in shared]
            f1 = [boundary_agreement(spans[v][0], sorted(human[v]), args.tolerance)["f1"] for v in shared]
            print(f"model vs human over {len(shared)} videos: "
                  f"jaccard={sum(jac) / len(jac):.3f}  f1={sum(f1) / len(f1):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
