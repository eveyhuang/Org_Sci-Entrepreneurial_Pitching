"""The shark-level run behind `analyze_panel.py --with-shark-level`: what the F1 paragraph cites, recomputed on the
records of the run through the archived code (refocus spec, decision 7). Reported, not tested; no decision rule.

`run(data, run_root, n_boot, seed, n_perm)` returns, as tables (written under `<analysis>/shark_level/`):

- `metrics`, `deltas`, `predictions`: the archived B-family B0-B5d on the shark rows (pitch x investor, each
  investor cut at its own first decision; `analysis.archive.shark_level_models`), every metrics row with its
  within- and between-pitch AUC (`analysis.archive.within_pitch_eval.within_pitch_table`), the deltas with the
  within-pitch differences beside the pooled ones;
- `permutation`: the pitch-level permutation test of the B models (empty when `n_perm` is 0);
- `identity_metrics`, `identity_deltas`: the identity baseline (formerly
  `scripts/experiments/identity_baseline_test.py`): every shark of a pitch cut at the pitch's first decision
  (`common_cut_rows`), the I ladder with the shark's leave-own-pitch-out offer rate (`investor_prior`) and the N
  ladder without it (`identity_models`);
- `f1`: the numbers of the F1 paragraph: identity alone ranking the sharks within a pitch (I0's within-pitch AUC),
  and what content adds beside identity within the pitch (own questions and answers, I3 - I2; the reaction
  features, IR - I4; all content, I5 - I2) and pooled (all content, I5 - I2).

The investor prior reads other pitches' outcomes, so it is a sensitivity of the shark-level design and never a
feature of the pitch-level ladder.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.archive import shark_level_models as shark
from analysis.archive.shark_level_dataset import investor_outcomes
from analysis.archive.shark_level_features import (
    ARCHIVE_FEATURES, _declare, _o2_row, _Pitch, _with_ranks, declare_stage_a, o2_features,
)
from analysis.panel_bottleneck import N_BOOT, PERMUTATION_COLUMNS, SEED
from analysis.panel_dataset import PanelData, pitch_cut
from analysis.archive.within_pitch_eval import TABLE_COLUMNS as WITHIN_COLUMNS, within_pitch_deltas, within_pitch_table

OUTCOME = "ever_offer"
PRIOR_MIN_PITCHES = 3
PRIOR, PRIOR_FAMILY = "investor_prior", "identity_prior"
ID = (PRIOR, f"{PRIOR}_missing")
# the I and N ladders: (step, families it adds), cumulative
STEPS = (("1", ("control", "exposure")), ("2", ("stage_a",)), ("3", ("own",)), ("4a", ("panel_content",)),
         ("4", ("panel_reactions",)), ("5", ("rapport_repair",)))
# the F1 numbers: (what, comparison or model, metric)
F1_ROWS = (("identity alone ranks the sharks of a pitch", "I0", "within_auc"),
           ("own questions and answers beside identity", "I3 minus I2", "within_auc"),
           ("the reaction features beside identity", "IR minus I4", "within_auc"),
           ("all content beside identity", "I5 minus I2", "within_auc"),
           ("all content beside identity, pooled over pitches", "I5 minus I2", "auc"))
F1_COLUMNS = ["finding", "model_or_comparison", "metric", "value", "ci_low", "ci_high", "n_pitches_informative"]
TABLES = ("metrics", "deltas", "predictions", "permutation", "identity_metrics", "identity_deltas", "f1")


# ---- the investor prior (a sensitivity: it reads other pitches' outcomes) --------------------------

def investor_prior(rows: pd.DataFrame, outcome: str = OUTCOME, min_pitches: int = PRIOR_MIN_PITCHES) -> pd.Series:
    """Per row, the offer rate of the same `investor_label` over the OTHER pitches of the table (rows with a
    known outcome); NaN when that label has a known outcome in fewer than `min_pitches` other pitches."""
    known = rows[rows[outcome].notna()]
    per = known.groupby(["investor_label", "video_id"])[outcome].mean().reset_index()
    out = []
    for r in rows.itertuples():
        other = per[(per.investor_label == r.investor_label) & (per.video_id != r.video_id)]
        out.append(float(other[outcome].mean()) if len(other) >= min_pitches else np.nan)
    return pd.Series(out, index=rows.index, dtype=float)


@contextlib.contextmanager
def prior_declared():
    """The prior's two columns in the shark-level feature dictionary, under a family no standard model uses, only
    while the sensitivity is evaluated."""
    _declare(PRIOR, PRIOR_FAMILY, "the shark's offer rate on other pitches", "o2", "concept",
             "Offer rate of the same investor over the other pitches (leave-own-pitch-out); a sensitivity only.")
    _declare(f"{PRIOR}_missing", PRIOR_FAMILY, "the shark's offer rate: too few other pitches", "o2", "indicator",
             f"1 when the investor has a known outcome in fewer than {PRIOR_MIN_PITCHES} other pitches.")
    try:
        yield
    finally:
        ARCHIVE_FEATURES.pop(PRIOR, None)
        ARCHIVE_FEATURES.pop(f"{PRIOR}_missing", None)


# ---- the identity baseline ------------------------------------------------------------------------

def common_cut_rows(data: PanelData) -> pd.DataFrame:
    """O2 rows (pitch x shark) whose history stops at the pitch's first decision, for every shark alike, with the
    shark's leave-own-pitch-out offer rate (`investor_prior`) and its missing flag."""
    declare_stage_a(data.stage_a_codes)
    o2 = investor_outcomes(data)
    o1_cut = pitch_cut(data)
    cut_of = dict(zip(o1_cut.video_id, o1_cut.cut_pos.astype(int)))
    with_ranks = _with_ranks(data)
    pitches, rows = {}, []
    for r in o2.itertuples():
        pitch = pitches.setdefault(r.video_id, _Pitch(data, r.video_id))
        cut = cut_of[r.video_id]
        rows.append({"video_id": r.video_id, "investor": r.investor, "source": pitch.video["source"], "cut_pos": cut,
                     OUTCOME: r.ever_offer, "investor_label": r.label,
                     **_o2_row(data, pitch, r.investor, cut, with_ranks)})
    table = pd.DataFrame(rows)
    if not len(table):
        return table
    table[PRIOR] = investor_prior(table)
    table[f"{PRIOR}_missing"] = table[PRIOR].isna().astype(int)
    return table


def identity_models(rows: pd.DataFrame) -> tuple[dict, list]:
    """(the I and N ladders, the pairs they are compared on): I0 identity; each step with and without identity."""
    rx = shark.reaction_columns(rows, ("rx_own_", "rx_pan_"))
    out, families = {"I0": ID}, ()
    for step, added in STEPS:
        families += added
        out[f"N{step}"] = families
        out[f"I{step}"] = families + ID
        if step == "4":
            out["NR"], out["IR"] = families + rx, families + rx + ID
    pairs = [("N1", "I1"), ("N3", "I3"), ("N4", "I4"), ("NR", "IR"),            # what identity adds
             ("I2", "I3"), ("I3", "I4a"), ("I4a", "I4"), ("I4", "IR"), ("I4", "I5"),   # each content step, with identity
             ("I2", "I4"), ("I2", "I5"), ("I0", "I5"),                          # content together, beyond identity
             ("N2", "N3"), ("N2", "N4"), ("N4", "NR")]                         # the same steps without identity
    return out, pairs


# ---- the run --------------------------------------------------------------------------------------

def _with_within(result: dict, pairs: list, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(metrics with the within- and between-pitch AUC of each model, deltas with the within-pitch differences)."""
    metrics, deltas = result["metrics"], result["deltas"]
    if result.get("status") != "estimated" or not len(metrics):
        return (metrics if len(metrics.columns) else pd.DataFrame(columns=["model", *WITHIN_COLUMNS[1:]]),
                deltas)
    pred = result["predictions"]
    within = within_pitch_table(pred, list(metrics.model), n_boot=n_boot, seed=seed)
    metrics = metrics.merge(within, on="model", how="left", suffixes=("", "_within"))
    deltas = pd.concat([deltas, within_pitch_deltas(pred, pairs, n_boot=n_boot, seed=seed)], ignore_index=True)
    return metrics, deltas


def f1_table(identity_metrics: pd.DataFrame, identity_deltas: pd.DataFrame) -> pd.DataFrame:
    """The F1 numbers (F1_ROWS) read from the identity baseline's tables; NaN where the run cannot give one."""
    rows = []
    by_model = identity_metrics.set_index("model") if "model" in identity_metrics and len(identity_metrics) else None
    within = by_model is not None and "within_auc" in by_model.columns
    # every model of the baseline reads the same rows, so its informative pitches are I0's
    pitches = float(by_model.pitches.iloc[0]) if within else np.nan
    for finding, key, metric in F1_ROWS:
        value = low = high = np.nan
        if " minus " not in key:
            if within and key in by_model.index:
                row = by_model.loc[key]
                value, low, high = row.within_auc, row.within_low, row.within_high
        elif len(identity_deltas):
            found = identity_deltas[(identity_deltas.comparison == key) & (identity_deltas.metric == metric)]
            if len(found):
                value, low, high = found.iloc[0][["delta", "ci_low", "ci_high"]]
        rows.append({"finding": finding, "model_or_comparison": key, "metric": metric, "value": value,
                     "ci_low": low, "ci_high": high, "n_pitches_informative": pitches})
    return pd.DataFrame(rows, columns=F1_COLUMNS)


def run(data: PanelData, run_root: Path | str, n_boot: int = N_BOOT, seed: int = SEED, n_perm: int = 0,
        penalty: float | None = None) -> dict[str, pd.DataFrame]:
    """The shark-level tables of a run (TABLES, see the module docstring). Reads only; `run_root` is where the run
    lives (kept for the provenance of the caller; nothing beyond `data` is read from it)."""
    features = o2_features(data)
    pairs = shark._comparisons(list(shark.MODEL_FAMILY))
    result = shark.evaluate(features, OUTCOME, penalty=penalty, models=dict(shark.MODEL_FAMILY), n_boot=n_boot,
                            seed=seed, comparisons=pairs)
    metrics, deltas = _with_within(result, pairs, n_boot, seed)
    if n_perm and result.get("status") == "estimated":
        permutation = shark.permutation_test(features, OUTCOME, result, n_perm=n_perm, seed=seed)
    else:
        permutation = pd.DataFrame(columns=PERMUTATION_COLUMNS)

    rows = common_cut_rows(data)
    if len(rows):
        spec, id_pairs = identity_models(rows)
        with prior_declared():
            identity = shark.evaluate(rows, OUTCOME, penalty=penalty, models=spec, comparisons=id_pairs,
                                      n_boot=n_boot, seed=seed)
        identity_metrics, identity_deltas = _with_within(identity, id_pairs, n_boot, seed)
    else:
        identity_metrics, identity_deltas = pd.DataFrame(columns=["model"]), pd.DataFrame(columns=["comparison"])
    label = {"label": result["label"], "n_pitches": result.get("n_groups", 0), "n_rows": result.get("n_rows", 0)}
    return {"metrics": metrics.assign(**label), "deltas": deltas.assign(**label),
            "predictions": result["predictions"], "permutation": permutation,
            "identity_metrics": identity_metrics, "identity_deltas": identity_deltas,
            "f1": f1_table(identity_metrics, identity_deltas)}
