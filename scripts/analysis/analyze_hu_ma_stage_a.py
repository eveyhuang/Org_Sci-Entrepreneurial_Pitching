#!/usr/bin/env python3
"""Stage A ratings and forced ranking on the Hu & Ma videos against the accelerator's decision.

Reads a Stage A run's exported annotations (run scripts/aggregate.py first), a ranking run's
scores.csv, and data/hu-ma/funding_outcomes.csv; writes the Pitch Factors, the Table III and
Table IV analogues, repeatability and method-convergence tables, and a summary under --out
(default: <stage-a run>/analysis/hu_ma_funding/).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.hu_ma_funding import run_analysis, run_provenance  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-a-run", required=True, help="runs/<stage_a run> with tables/annotations.parquet")
    p.add_argument("--rank-run", help="runs/<ranking run> with ranking/scores.csv (optional)")
    p.add_argument("--outcomes", default=str(ROOT / "data" / "hu-ma" / "funding_outcomes.csv"))
    p.add_argument("--out", help="output directory (default <stage-a run>/analysis/hu_ma_funding)")
    p.add_argument("--hu-ma-pitch-factor", default=str(ROOT / "data" / "hu-ma" / "pitch_factor.csv"),
                   help="Hu & Ma's Pitch Factor and features from their code (video_id, pitch_factor, ten features); "
                        "'none' to skip; skipped when the default file is absent")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stage_a_run = Path(args.stage_a_run)
    annotations = stage_a_run / "tables" / "annotations.parquet"
    if not annotations.exists():
        print(f"error: {annotations} not found; run scripts/aggregate.py for that run first", file=sys.stderr)
        return 1
    rank_run = Path(args.rank_run) if args.rank_run else None
    scores = rank_run / "ranking" / "scores.csv" if rank_run else None
    if scores is not None and not scores.exists():
        print(f"error: {scores} not found", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else stage_a_run / "analysis" / "hu_ma_funding"
    hu_ma = None if args.hu_ma_pitch_factor.lower() == "none" else Path(args.hu_ma_pitch_factor)
    if hu_ma is not None and not hu_ma.exists():
        if args.hu_ma_pitch_factor != str(ROOT / "data" / "hu-ma" / "pitch_factor.csv"):
            print(f"error: {hu_ma} not found", file=sys.stderr)
            return 1
        hu_ma = None
    result = run_analysis(annotations, scores, args.outcomes, out, provenance=run_provenance(stage_a_run, rank_run),
                          hu_ma_csv=hu_ma)
    print(f"videos labelled {result['videos_with_outcome_label']}, accepted {result['acceptances']}, "
          f"rated {result['videos_rated']}, ranked {result['videos_ranked']}")
    for row in result["table3"]:
        print(f"{row['predictor']}: ML AME {row['ml_ame']:.3f} (SE {row['ml_ame_se']:.3f}, p {row['ml_p']:.3f}); "
              f"Firth coef {row['firth_coef']:.3f} (p {row['firth_p']:.3f}); AUC {row['auc']:.3f} (exact p {row['auc_p_exact']:.3f})"
              + (f"; {row['ml_note']}" if row.get("ml_note") else ""))
    for row in result.get("hu_ma_pitch_factor", {}).get("agreement", []):
        print(f"Pitch Factor ({row['method']}) vs Hu & Ma: Pearson {row['pearson']:.3f} "
              f"({row['pearson_low']:.3f} to {row['pearson_high']:.3f}), Spearman {row['spearman']:.3f}, n {row['videos']}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
