#!/usr/bin/env python3
"""Each shark's offer against a baseline of who the shark is, with every shark of a pitch cut at the same moment.

    python scripts/experiments/identity_baseline_test.py --run runs/<run> --selection selection_all100.json \
        --analysis-id identity_baseline [--n-boot 1999] [--seed 42] [--force]

Exploratory, NOT pre-registered: it follows the reaction-trajectory test (analysis reaction_test), which found
that a shark's identity ranks the sharks of a pitch as well as anything said, and that "how much was said
before this shark decided" mostly encodes when the shark decided. Two changes to the per-shark (O2) rows:

- Common cut. Every shark of a pitch is cut at the pitch's first decision of any kind (the O1 cut), not at the
  shark's own decision, so history length cannot differ with the outcome. Nothing any shark decided is read.
- Identity. `investor_prior`, the shark's offer rate over the OTHER pitches (leave-own-pitch-out; missing with
  fewer than three other pitches, with an indicator), enters every model of the I ladder.

Ladders (cumulative): I0 identity; I1 + controls and exposure; I2 + Stage A; I3 + own questions and answers;
I4a + other sharks' questions and answers; I4 + reactions (existing); IR + every reaction (rx_own, rx_pan);
I5 + rapport and repair. The same steps without identity (N1..N5, NR) show what identity changes. Read the
within-pitch AUC: controls, exposure and Stage A are (nearly) the same for every shark of a pitch, so only the
shark-specific families can rank sharks within it.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.panel_bottleneck import N_BOOT, SEED  # noqa: E402
from analysis.archive.shark_level_models import evaluate  # noqa: E402
# the common-cut rows and the I/N ladders live in the archive's shark-level run (plan B task 6); imported back here
from analysis.archive.shark_level import ID, STEPS, common_cut_rows, identity_models as models  # noqa: E402,F401
from analysis.panel_dataset import load_run  # noqa: E402
from analysis.panel_explorer import resolve_video_ids  # noqa: E402
from analysis.archive.within_pitch_eval import within_pitch_deltas, within_pitch_table  # noqa: E402

_spec = importlib.util.spec_from_file_location("rtt", REPO / "scripts" / "experiments" / "reaction_trajectory_test.py")
rtt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtt)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--analysis-id", required=True)
    p.add_argument("--selection", default=None)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    run_root = a.run.resolve()
    out = run_root / "analysis" / a.analysis_id
    if out.exists() and not a.force:
        print(f"identity_baseline_test: {out} already exists; pass --force", file=sys.stderr)
        return 2
    started = time.time()
    data = load_run(run_root, resolve_video_ids(run_root, a.selection))
    rows = common_cut_rows(data)
    spec, pairs = models(rows)
    with rtt.prior_declared():
        result = evaluate(rows, "ever_offer", models=spec, comparisons=pairs, n_boot=a.n_boot, seed=a.seed)
    pred = result["predictions"]
    within = within_pitch_table(pred, list(spec), n_boot=a.n_boot, seed=a.seed)
    metrics = result["metrics"][["model", "families", "auc", "auc_low", "auc_high", "skill_pct", "skill_low",
                                 "skill_high", "pred_spread"]].merge(within, on="model")
    deltas = pd.concat([result["deltas"], within_pitch_deltas(pred, pairs, n_boot=a.n_boot, seed=a.seed)],
                       ignore_index=True)

    # what the common cut does to the rows, against each shark's own cut (the reaction_test table)
    own_cut = pd.read_csv(run_root / "analysis" / "reaction_test" / "features_o2.csv") \
        if (run_root / "analysis" / "reaction_test" / "features_o2.csv").exists() else None
    known = rows[rows.ever_offer.notna()]
    spread = known.groupby("video_id").hist_turns.agg(lambda s: s.max() - s.min())
    coverage = {"rows_known": int(len(known)), "offers": int(known.ever_offer.sum()),
                "median_history_turns": float(known.hist_turns.median()),
                "max_within_pitch_history_difference_turns": int(spread.max()),
                "share_with_own_questions": float((known.own_questions > 0).mean()),
                "share_with_own_answers_reacted": float((known.rx_own_answers > 0).mean()),
                "identity_missing_share": float(known[f"{rtt.PRIOR}_missing"].mean())}
    if own_cut is not None:
        k2 = own_cut[own_cut.ever_offer.notna()]
        coverage.update({"own_cut_median_history_turns": float(k2.hist_turns.median()),
                         "own_cut_share_with_own_questions": float((k2.own_questions > 0).mean())})
        both = known.merge(k2[["video_id", "investor", "own_questions"]], on=["video_id", "investor"], suffixes=("", "_own_cut"))
        coverage["own_questions_by_outcome"] = {
            "common_cut": both.groupby("ever_offer").own_questions.mean().round(2).to_dict(),
            "own_cut": both.groupby("ever_offer").own_questions_own_cut.mean().round(2).to_dict()}

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows.to_csv(out / "features_o2_common_cut.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    deltas.to_csv(out / "deltas.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    (out / "manifest.json").write_text(json.dumps(rtt._plain({
        "analysis_id": a.analysis_id, "run": str(run_root), "selection": a.selection, "n_boot": a.n_boot, "seed": a.seed,
        "git_commit": commit, "status": "exploratory, not pre-registered", "cut": "pitch's first decision, every shark",
        "identity": "leave-own-pitch-out offer rate of the same shark; missing with fewer than 3 other pitches",
        "models": {k: list(v) for k, v in spec.items()}, "coverage": coverage, "coding": result["coding"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1)}), indent=2) + "\n")
    print(metrics[["model", "auc", "within_auc", "within_low", "within_high", "between_auc"]].round(3).to_string())
    print(deltas[deltas.metric.isin(["within_auc", "auc"])][["comparison", "metric", "delta", "ci_low", "ci_high", "p_value"]]
          .round(3).to_string())
    print(json.dumps(rtt._plain(coverage), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
