#!/usr/bin/env python3
"""The reaction-trajectory test: do every reaction to the founders' answers, the asker's apart from the
panel's, predict offers beyond the existing reaction counts and closing states?

    python scripts/experiments/reaction_trajectory_test.py --run runs/<run> --selection selection_all100.json \
        --analysis-id reaction_test [--n-boot 1999] [--seed 42] [--guard-turns 1] [--penalty X] [--force]

The test and its decision rule are fixed in docs/plans/design docs/2026-10-19-reaction-trajectory-test.md,
section 0, which report.md quotes verbatim before any result. Two outcomes are read: each shark's offer
(O2, pitch x investor rows, `ever_offer`) on the ladder B1, B3, R1, B4a, R2, B4, R3, B5, R4, and whether
any shark offers (O1, pitch rows, `any_offer`) on C1, C3, R-C3, C4, R-C4. Every model is evaluated by
leave-one-pitch-out with the penalty chosen inside each training fold (analysis.archive.shark_level_models.evaluate);
for O2 the within-pitch AUC (analysis.archive.within_pitch_eval) says whether a model ranks the sharks of one
pitch, not only the pitches.

Sensitivities (sensitivity.csv): the guard band (`--guard-turns`, default 1, drops the focal shark's own
last turn(s) before the cut), the investor prior (the offer rate of the same shark over the other pitches,
which reads other pitches' outcomes and is never a primary model), and the counts-only variant (no column
on the course of the reactions: no last-answer and no trend columns). Offline and read-only towards the run: it writes nowhere but <run>/analysis/<analysis_id>/.
"""
from __future__ import annotations

import argparse
import json
import re
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
from analysis.archive.shark_level_models import (  # noqa: E402
    MODEL_FAMILY, evaluate, o1_reaction_comparisons, o1_reaction_models, reaction_comparisons, reaction_models,
)
from analysis.archive.shark_level_dataset import investor_cuts  # noqa: E402
from analysis.archive.shark_level_features import o2_features  # noqa: E402
# the investor prior lives in the archive's shark-level run (plan B task 6); imported back under its old names
from analysis.archive.shark_level import (  # noqa: E402,F401
    PRIOR, PRIOR_FAMILY, PRIOR_MIN_PITCHES, investor_prior, prior_declared,
)
from analysis.panel_dataset import load_run  # noqa: E402
from analysis.panel_explorer import resolve_video_ids  # noqa: E402
from analysis.panel_features import pitch_features  # noqa: E402
from analysis.reaction_events import REACTION_KINDS, reaction_events  # noqa: E402
from analysis.archive.within_pitch_eval import conditional_logit_check, within_pitch_deltas, within_pitch_table  # noqa: E402
from pipeline.errors import ConfigError  # noqa: E402

PLAN = REPO / "docs" / "plans" / "design docs" / "2026-10-19-reaction-trajectory-test.md"
ANALYSIS_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MIN_GAIN = 0.03                      # the decision rule's smallest gain in AUC that counts
PRIMARY = ("B4", "R3")
SUPPORT = (("B3", "R1"), ("B4a", "R2"))
O1_PAIR = ("C4", "R-C4")
JUSTIFIED, NOT_JUSTIFIED, MIXED = "justified", "not justified", "mixed"
FRAGILE_GUARD = "fragile: possibly a decision tell"
FRAGILE_PRIOR = "fragile: possibly shark disposition"
# the conditional-logit check: (a) the existing closing-state and first-move columns, (b) reaction trajectories
CLOGIT_EXISTING = ("own_satisfied", "own_unsatisfied", "own_receipt_share", "own_challenge_share", "oth_agrees",
                   "oth_challenges", "oth_satisfied", "oth_unsatisfied")
CLOGIT_REACTIONS = ("rx_own_receipt_share", "rx_own_followup_share", "rx_own_challenge_share", "rx_own_last_receipt",
                    "rx_own_last_challenge", "rx_own_trend_receipt", "rx_own_trend_challenge", "rx_own_other_first_share",
                    "rx_pan_reaction_rate", "rx_pan_receipt_share", "rx_pan_challenge_share", "rx_pan_last_receipt",
                    "rx_pan_last_challenge")
VARIANT_FULL, VARIANT_COUNTS = "full", "counts_only"


# ---- the decision rule, pure ----------------------------------------------------------------------

def _row(deltas: pd.DataFrame, earlier: str, later: str, metric: str) -> pd.Series | None:
    if deltas.empty:
        return None
    rows = deltas[(deltas.comparison == f"{later} minus {earlier}") & (deltas.metric == metric)]
    return rows.iloc[0] if len(rows) else None


def holds(row, min_gain: float = MIN_GAIN) -> bool:
    """A gain of at least `min_gain` whose 95% interval excludes 0 (a missing value never holds)."""
    if row is None:
        return False
    delta, low = float(row["delta"]), float(row["ci_low"])
    return bool(np.isfinite(delta) and np.isfinite(low) and delta >= min_gain and low > 0)


def verdict(deltas_o2: pd.DataFrame, deltas_o1: pd.DataFrame, min_gain: float = MIN_GAIN,
            primary: tuple[str, str] = PRIMARY) -> str:
    """justified | not justified | mixed, from section 0's rule: `deltas_o2` carries the within-pitch AUC
    deltas (metric `within_auc`), `deltas_o1` the pooled AUC deltas (metric `auc`)."""
    main = holds(_row(deltas_o2, *primary, "within_auc"), min_gain)
    support = any(holds(_row(deltas_o2, *pair, "within_auc"), min_gain) for pair in SUPPORT)
    o1 = holds(_row(deltas_o1, *O1_PAIR, "auc"), min_gain)
    if main and support:
        return JUSTIFIED
    if not main and not o1:
        return NOT_JUSTIFIED
    return MIXED


def final_verdict(base: str, guard: str | None, prior: str | None) -> str:
    """The verdict with the fragility flags of section 0: a verdict that flips under the guard band, or when
    the investor prior is added, is reported as fragile."""
    flags = [flag for other, flag in ((guard, FRAGILE_GUARD), (prior, FRAGILE_PRIOR)) if other is not None and other != base]
    return f"{base} ({'; '.join(flags)})" if flags else base


# ---- descriptives (whole encounter; never predictors) ---------------------------------------------

def _shares(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    return {str(k): {"n": int(v), "share": float(v / len(series))} for k, v in counts.items()} if len(series) else {}


def descriptives(data, o2: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    events = reaction_events(data)
    o2_cut = investor_cuts(data)
    cut = {}
    for r in o2_cut.itertuples():
        cut.setdefault(r.video_id, {})[r.investor] = int(r.cut_pos)
    before = reaction_events(data, cut=cut)
    answers = events.drop_duplicates(["video_id", "answer_turn"])

    def block(ev: pd.DataFrame) -> dict:
        any_pan = {kind: int((ev[col] > 0).sum()) for kind, col in
                   (("receipt", "pan_receipts"), ("follow_up", "pan_follow_ups"),
                    ("challenge", "pan_challenges"), ("other", "pan_other"))}
        return {"rows": int(len(ev)),
                "asker_reaction": _shares(ev.asker_reaction),
                "panel_reaction_any": {"n": int((ev.pan_reactions > 0).sum()),
                                       "share": float((ev.pan_reactions > 0).mean()) if len(ev) else np.nan},
                "panel_reaction_by_kind": {k: {"n": v, "share": v / len(ev) if len(ev) else np.nan} for k, v in any_pan.items()},
                "first_reactor": _shares(ev.first_reactor),
                "asker_followup_severity": _shares(ev.asker_followup_sev.dropna().astype(int)),
                "panel_first_kind": _shares(ev.pan_first_kind.dropna())}

    known = before.merge(o2[["video_id", "investor", "ever_offer"]].rename(columns={"investor": "asker"}),
                         on=["video_id", "asker"], how="left")
    reacted_before = (known.pan_reactions > 0) | known.asker_reaction.isin(REACTION_KINDS)
    out = {"scope": "whole encounter as annotated; descriptive only, never a predictor",
           "answers": int(len(answers)), "answer_asker_rows": int(len(events)),
           "all": block(events),
           "any_reaction_before_the_askers_cut": {"n": int(reacted_before.sum()),
                                                  "share": float(reacted_before.mean()) if len(known) else np.nan},
           "by_asker_offer": {label: block(events.merge(known.loc[known.ever_offer == value, ["video_id", "answer_turn", "asker"]],
                                                        on=["video_id", "answer_turn", "asker"]))
                              for label, value in (("asker_offered", 1.0), ("asker_did_not_offer", 0.0))}}
    return events, out


# ---- one evaluation of both ladders ---------------------------------------------------------------

def _ladders(features_o2: pd.DataFrame, features_o1: pd.DataFrame, n_boot: int, seed: int, penalty) -> dict:
    o2_models = {**reaction_models(features_o2), **{k: v for k, v in reaction_models(features_o2, True).items()
                                                    if k not in MODEL_FAMILY}}
    o2_pairs = reaction_comparisons() + reaction_comparisons(True)
    o2 = evaluate(features_o2, "ever_offer", models=o2_models, comparisons=o2_pairs, n_boot=n_boot, seed=seed,
                  penalty=penalty)
    o1_models = {**o1_reaction_models(features_o1), **{k: v for k, v in o1_reaction_models(features_o1, True).items()
                                                       if k.startswith("R-")}}
    o1_pairs = o1_reaction_comparisons() + o1_reaction_comparisons(True)
    o1 = evaluate(features_o1, "any_offer", models=o1_models, comparisons=o1_pairs, n_boot=n_boot, seed=seed,
                  penalty=penalty)
    within = within_pitch_table(o2["predictions"], list(o2_models), n_boot=n_boot, seed=seed)
    within_d = within_pitch_deltas(o2["predictions"], o2_pairs, n_boot=n_boot, seed=seed)
    return {"o2": o2, "o1": o1, "o2_models": o2_models, "o1_models": o1_models, "within": within,
            "within_deltas": within_d, "o2_pairs": o2_pairs, "o1_pairs": o1_pairs}


def _variant(model: str) -> str:
    return VARIANT_COUNTS if model.endswith("c") and model[0] == "R" else VARIANT_FULL


def _with_variant(table: pd.DataFrame, column: str) -> pd.DataFrame:
    table = table.copy()
    names = table[column].map(lambda c: c.split(" minus ")[0])
    table.insert(0, "variant", names.map(_variant))
    return table


def clogit_table(features_o2: pd.DataFrame) -> pd.DataFrame:
    """Conditional logit (strata: pitches) on (a) existing columns, (b) reaction trajectories, (c) both, on the
    same rows; a missing value is filled with the column median (a share without observations, mostly)."""
    columns = [c for c in (*CLOGIT_EXISTING, *CLOGIT_REACTIONS) if c in features_o2.columns]
    rows = features_o2[["video_id", "investor", "ever_offer", *columns]].copy()
    for c in columns:
        rows[c] = rows[c].astype(float).fillna(rows[c].astype(float).median())
    sets = {"a_existing": [c for c in CLOGIT_EXISTING if c in columns],
            "b_reactions": [c for c in CLOGIT_REACTIONS if c in columns]}
    sets["c_both"] = sets["a_existing"] + sets["b_reactions"]
    fits = {name: conditional_logit_check(rows, "ever_offer", cols) for name, cols in sets.items()}
    out = []
    for name, fit in fits.items():
        for column, beta in fit["beta"].items():
            out.append({"set": name, "column": column, "beta": beta, "loglik": fit["loglik"], "loglik0": fit["loglik0"],
                        "lr_vs_zero": fit["lr"], "lr_p_vs_zero": fit["lr_p"], "pitches": fit["pitches"], "rows": fit["rows"]})
    from scipy.stats import chi2
    lr = max(0.0, 2.0 * (fits["c_both"]["loglik"] - fits["a_existing"]["loglik"]))
    out.append({"set": "c_vs_a", "column": "", "beta": np.nan, "loglik": fits["c_both"]["loglik"],
                "loglik0": fits["a_existing"]["loglik"], "lr_vs_zero": lr,
                "lr_p_vs_zero": float(chi2.sf(lr, df=len(sets["b_reactions"]))), "pitches": fits["c_both"]["pitches"],
                "rows": fits["c_both"]["rows"]})
    return pd.DataFrame(out)


# ---- the figure -------------------------------------------------------------------------------------

def figure(metrics: pd.DataFrame, path_stem: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis import panel_figures as pf

    rows = metrics[metrics.variant == VARIANT_FULL].reset_index(drop=True)
    with pf._context():
        fig, axes = plt.subplots(1, 2, figsize=(pf.width_inches("double"), 0.32 * len(rows) + 1.2), dpi=pf.SCREEN_DPI,
                                 layout="constrained", sharey=True)
        for i, (ax, (col, low, high, title)) in enumerate(zip(axes, (
                ("within_auc", "within_low", "within_high", "Within-pitch AUC (sharks of one pitch)"),
                ("auc", "auc_low", "auc_high", "Pooled AUC (all rows)")))):
            for y, r in enumerate(rows.itertuples()):
                filled = r.model.startswith("R")
                colour = pf.BLUE if filled else pf.GREY
                ax.plot([getattr(r, low), getattr(r, high)], [y, y], color=colour, linewidth=1.2)
                pf._point(ax, getattr(r, col), y, colour, "o", filled=filled)
            ax.axvline(0.5, color=pf.INK, linestyle="--", linewidth=0.8)
            ax.set_yticks(range(len(rows)), list(rows.model))
            ax.invert_yaxis() if i == 0 else None
            ax.set_xlabel("AUC (95% pitch-bootstrap interval)")
            pf._letter(ax, i, title)
        axes[1].legend(handles=[pf._handle(pf.GREY, "o", False, "existing model (B)"),
                                pf._handle(pf.BLUE, "o", True, "with reaction trajectories (R)")],
                       loc="lower right", frameon=False)
        files = []
        for suffix in ("png", "pdf"):
            fig.savefig(path_stem.with_suffix(f".{suffix}"), dpi=pf.SAVE_DPI)
            files.append(path_stem.with_suffix(f".{suffix}").name)
        plt.close(fig)
    return files


# ---- the report ---------------------------------------------------------------------------------------

def section_zero() -> str:
    text = PLAN.read_text()
    start = text.index("## 0. The pre-registered test and its decision rule")
    end = text.index("\n---", start)
    return text[start:end].strip() + "\n"


def _fmt(v, digits=3) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{digits}f}"


def _ci(r, col, low, high) -> str:
    return f"{_fmt(r[col])} [{_fmt(r[low])}, {_fmt(r[high])}]"


def report(results: dict, verdicts: dict, descr: dict, sensitivity: pd.DataFrame, clogit: pd.DataFrame,
           meta: dict) -> str:
    m2, d2, m1, d1 = results["metrics_o2"], results["deltas_o2"], results["metrics_o1"], results["deltas_o1"]
    lines = [section_zero(), "## 1. Results", "",
             f"{meta['pitches']} pitches, {meta['o2_rows']} shark rows with a known outcome "
             f"({meta['o2_offers']} offers); {meta['informative']} pitches have both an offering and a non-offering "
             f"shark ({meta['pairs']} within-pitch pairs). Coding: {meta['coding']}. Bootstrap draws: {meta['n_boot']}.", "",
             "### Each shark's offer (O2)", "",
             "| Model | variant | pooled AUC [95%] | skill % [95%] | within-pitch AUC [95%] | between-pitch AUC | spread | note |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in m2.iterrows():
        lines.append(f"| {r.model} | {r.variant} | {_ci(r, 'auc', 'auc_low', 'auc_high')} | "
                     f"{_ci(r, 'skill_pct', 'skill_low', 'skill_high')} | {_ci(r, 'within_auc', 'within_low', 'within_high')} | "
                     f"{_fmt(r.between_auc)} | {_fmt(r.pred_spread)} | {r.note_within or ''} |")
    lines += ["", "Paired comparisons, later minus earlier (95% pitch-bootstrap interval, two-sided bootstrap p):", "",
              "| Comparison | variant | metric | delta [95%] | p |", "|---|---|---|---|---|"]
    for _, r in d2[d2.metric.isin(["within_auc", "auc", "log_loss"])].iterrows():
        lines.append(f"| {r.comparison} | {r.variant} | {r.metric} | {_ci(r, 'delta', 'ci_low', 'ci_high')} | {_fmt(r.p_value)} |")
    lines += ["", "### Any shark's offer (O1)", "", "| Model | variant | pooled AUC [95%] | skill % [95%] | spread |",
              "|---|---|---|---|---|"]
    for _, r in m1.iterrows():
        lines.append(f"| {r.model} | {r.variant} | {_ci(r, 'auc', 'auc_low', 'auc_high')} | "
                     f"{_ci(r, 'skill_pct', 'skill_low', 'skill_high')} | {_fmt(r.pred_spread)} |")
    lines += ["", "| Comparison | variant | metric | delta [95%] | p |", "|---|---|---|---|---|"]
    for _, r in d1[d1.metric.isin(["auc", "log_loss"])].iterrows():
        lines.append(f"| {r.comparison} | {r.variant} | {r.metric} | {_ci(r, 'delta', 'ci_low', 'ci_high')} | {_fmt(r.p_value)} |")
    lines += ["", "### Sensitivities", "", "| Setting | comparison | metric | delta [95%] | holds | verdict under it |",
              "|---|---|---|---|---|---|"]
    for _, r in sensitivity.iterrows():
        lines.append(f"| {r.setting} | {r.comparison} | {r.metric} | {_ci(r, 'delta', 'ci_low', 'ci_high')} | "
                     f"{'yes' if r.holds else 'no'} | {r.verdict} |")
    lines += ["", "The investor prior reads other pitches' outcomes (the same shark's offer rate elsewhere); it is a "
              "sensitivity, never a primary model.", "",
              "### Conditional logit (strata: pitches; informative pitches only)", ""]
    cva = clogit[clogit.set == "c_vs_a"].iloc[0]
    for name in ("a_existing", "b_reactions", "c_both"):
        part = clogit[clogit.set == name]
        if len(part):
            lines.append(f"- {name}: LR against all-zero {_fmt(part.lr_vs_zero.iloc[0], 2)}, p {_fmt(part.lr_p_vs_zero.iloc[0])}"
                         f" ({part.pitches.iloc[0]} pitches)")
    lines.append(f"- reaction trajectories beyond the existing columns (c against a): LR {_fmt(cva.lr_vs_zero, 2)}, "
                 f"p {_fmt(cva.lr_p_vs_zero)}")
    first = descr["all"]["first_reactor"]
    lines += ["", "### Descriptives (whole encounter, not predictors)", "",
              f"{descr['answers']} answers to an investor question ({descr['answer_asker_rows']} answer-asker rows). "
              "Who reacted first: " + ", ".join(f"{k} {v['share']:.0%}" for k, v in first.items()) + ". "
              f"Some reaction before the asker's cut: {descr['any_reaction_before_the_askers_cut']['share']:.0%}.", "",
              "## 2. Verdict", "",
              f"Base verdict {verdicts['base']}; under the guard band {verdicts['guard']}; with the investor prior "
              f"{verdicts['prior']}.", "", f"Verdict: {verdicts['final']}"]
    return "\n".join(lines) + "\n"


# ---- the experiment -------------------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def compute(run_root: Path, video_ids, n_boot: int = N_BOOT, seed: int = SEED, guard_turns: int = 1,
            penalty: float | None = None) -> dict:
    data = load_run(run_root, resolve_video_ids(run_root, video_ids))
    if data.videos.empty:
        raise ConfigError(f"no video with a result under {run_root}")
    features_o2, features_o1 = o2_features(data), pitch_features(data)
    main = _ladders(features_o2, features_o1, n_boot, seed, penalty)

    metrics_o2 = main["o2"]["metrics"].merge(main["within"].rename(columns={"note": "note_within"}), on="model")
    metrics_o2.insert(1, "variant", metrics_o2.model.map(_variant))
    metrics_o1 = main["o1"]["metrics"].copy()
    metrics_o1.insert(1, "variant", metrics_o1.model.map(lambda m: VARIANT_COUNTS if m.startswith("R-") and m.endswith("c") else VARIANT_FULL))
    deltas_o2 = _with_variant(pd.concat([main["o2"]["deltas"], main["within_deltas"]], ignore_index=True), "comparison")
    deltas_o1 = main["o1"]["deltas"].copy()
    deltas_o1.insert(0, "variant", deltas_o1.comparison.map(lambda c: VARIANT_COUNTS if c.split(" minus ")[0].endswith("c") else VARIANT_FULL))
    base = verdict(deltas_o2, deltas_o1)

    # sensitivity 1: the guard band
    # the pitch rows have no guard band since plan B task 2: the O1 half of this sensitivity is the main O1 ladder
    guard_o2, guard_o1 = o2_features(data, guard_turns=guard_turns), features_o1
    guard = _ladders(guard_o2, guard_o1, n_boot, seed, penalty)
    guard_deltas_o2 = pd.concat([guard["o2"]["deltas"], guard["within_deltas"]], ignore_index=True)
    guard_verdict = verdict(guard_deltas_o2, guard["o1"]["deltas"])

    # sensitivity 2: the investor prior, added to B4 and R3
    prior_rows = features_o2.copy()
    prior_rows[PRIOR] = investor_prior(prior_rows)
    prior_rows[f"{PRIOR}_missing"] = prior_rows[PRIOR].isna().astype(int)
    models = reaction_models(features_o2)
    with prior_declared():
        prior_models = {"B4i": models["B4"] + (PRIOR, f"{PRIOR}_missing"), "R3i": models["R3"] + (PRIOR, f"{PRIOR}_missing")}
        prior = evaluate(prior_rows, "ever_offer", models=prior_models, comparisons=[("B4i", "R3i")], n_boot=n_boot,
                         seed=seed, penalty=penalty)
    prior_within = within_pitch_deltas(prior["predictions"], [("B4i", "R3i")], n_boot=n_boot, seed=seed)
    prior_verdict = verdict(pd.concat([deltas_o2[~deltas_o2.comparison.eq("R3 minus B4")], prior_within]), deltas_o1,
                            primary=("B4i", "R3i"))

    sensitivity = []

    def sens(setting, table, earlier, later, metric, verdict_text):
        r = _row(table, earlier, later, metric)
        sensitivity.append({"setting": setting, "comparison": f"{later} minus {earlier}", "metric": metric,
                            "delta": r["delta"] if r is not None else np.nan,
                            "ci_low": r["ci_low"] if r is not None else np.nan,
                            "ci_high": r["ci_high"] if r is not None else np.nan,
                            "p_value": r["p_value"] if r is not None else np.nan,
                            "holds": holds(r), "verdict": verdict_text})

    sens("main", deltas_o2, "B4", "R3", "within_auc", base)
    sens(f"guard_turns={guard_turns}", guard_deltas_o2, "B4", "R3", "within_auc", guard_verdict)
    sens("investor_prior", prior_within, "B4i", "R3i", "within_auc", prior_verdict)
    sens("counts_only", deltas_o2, "B4", "R3c", "within_auc", "")
    sens(f"guard_turns={guard_turns}, any offer (O1)", guard["o1"]["deltas"], *O1_PAIR, "auc", guard_verdict)

    events, descr = descriptives(data, features_o2)
    informative = int(main["within"].pitches.iloc[0]) if len(main["within"]) else 0
    known = features_o2[features_o2.ever_offer.notna()]
    meta = {"pitches": int(data.videos.video_id.nunique()), "o2_rows": int(len(known)),
            "o2_offers": int(known.ever_offer.sum()), "informative": informative,
            "pairs": int(main["within"].pairs.iloc[0]) if len(main["within"]) else 0,
            "coding": main["o2"]["coding"], "n_boot": n_boot}
    verdicts = {"base": base, "guard": guard_verdict, "prior": prior_verdict,
                "final": final_verdict(base, guard_verdict, prior_verdict)}
    return {"data": data, "features_o2": features_o2, "features_o1": features_o1, "events": events, "descriptives": descr,
            "metrics_o2": metrics_o2, "deltas_o2": deltas_o2, "metrics_o1": metrics_o1, "deltas_o1": deltas_o1,
            "clogit": clogit_table(features_o2), "sensitivity": pd.DataFrame(sensitivity), "verdicts": verdicts,
            "meta": meta, "penalties": {"o2": main["o2"]["penalty"], "o1": main["o1"]["penalty"]}}


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def write(results: dict, out: Path, manifest: dict) -> None:
    out.mkdir(parents=True)
    results["events"].to_csv(out / "reaction_events.csv", index=False)
    (out / "reaction_descriptives.json").write_text(json.dumps(_plain(results["descriptives"]), indent=2) + "\n")
    results["features_o2"].to_csv(out / "features_o2.csv", index=False)
    results["features_o1"].to_csv(out / "features_o1.csv", index=False)
    for name in ("metrics_o2", "deltas_o2", "metrics_o1", "deltas_o1"):
        results[name].to_csv(out / f"{name}.csv", index=False)
    results["clogit"].to_csv(out / "clogit_o2.csv", index=False)
    results["sensitivity"].to_csv(out / "sensitivity.csv", index=False)
    figures = figure(results["metrics_o2"], out / "figure_within_pitch_ladder")
    (out / "report.md").write_text(report(results, results["verdicts"], results["descriptives"], results["sensitivity"],
                                          results["clogit"], results["meta"]))
    manifest = {**manifest, "figures": figures, "verdicts": results["verdicts"], "coding": results["meta"]["coding"],
                "penalties": results["penalties"], "outputs": sorted(p.name for p in out.iterdir()) + ["manifest.json"]}
    (out / "manifest.json").write_text(json.dumps(_plain(manifest), indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, type=Path, help="run folder, e.g. runs/<run_name>")
    parser.add_argument("--analysis-id", required=True, help="name of the folder under <run>/analysis/")
    parser.add_argument("--selection", default=None, help="a selection file inside the run folder (or a path)")
    parser.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap draws for every interval")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--guard-turns", type=int, default=1, help="the guard band of the sensitivity (default 1)")
    parser.add_argument("--penalty", type=float, default=None,
                        help="fix the ridge penalty (a labelled extra table only; by default chosen inside each fold)")
    parser.add_argument("--force", action="store_true", help="replace an existing analysis of this id")
    args = parser.parse_args(argv)

    def refuse(message: str) -> int:
        print(f"reaction_trajectory_test: {message}", file=sys.stderr)
        return 2

    if not ANALYSIS_ID.fullmatch(args.analysis_id):
        return refuse(f"analysis id {args.analysis_id!r} must be a plain folder name")
    run_root = args.run.resolve()
    if not run_root.is_dir():
        return refuse(f"run folder not found: {args.run}")
    out = run_root / "analysis" / args.analysis_id
    if out.exists() and not args.force:
        return refuse(f"{out} already exists; pass --force to replace it")
    started = time.time()
    try:
        results = compute(run_root, args.selection, args.n_boot, args.seed, args.guard_turns, args.penalty)
    except ConfigError as exc:
        return refuse(str(exc))
    if out.exists():
        shutil.rmtree(out)
    manifest = {"analysis_id": args.analysis_id, "run": str(run_root), "selection": args.selection,
                "video_ids": sorted(results["data"].videos.video_id.unique()), "seed": args.seed, "n_boot": args.n_boot,
                "guard_turns": args.guard_turns, "penalty": args.penalty, "git_commit": _git_commit(),
                "plan": str(PLAN.relative_to(REPO)),
                "decision_rule": {"min_gain": MIN_GAIN, "primary": "R3 minus B4, within-pitch AUC",
                                  "support": ["R1 minus B3", "R2 minus B4a"], "o1": "R-C4 minus C4, pooled AUC",
                                  "interval": "95% pitch bootstrap, must exclude 0"},
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.time() - started, 1)}
    write(results, out, manifest)
    print(f"reaction_trajectory_test: wrote {out}\nVerdict: {results['verdicts']['final']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
