"""Does a model rank the investors of one pitch, or only the pitches? Within- and between-pitch AUC.

A pooled AUC over pitch x investor rows mixes two things: telling a pitch that draws offers from one
that does not (between pitches) and telling, inside one pitch, the investor who offers from the one
who does not (within pitches). Every feature fixed for a pitch (its controls, its Stage A ratings, the
whole panel's reactions) can help only the first. `within_pitch_auc` is the second alone: pooled over
every (offer, non-offer) pair of investors of the same pitch, ties counting one half, exactly
`analysis.within_pitch.stratified_auc`; pitches whose investors all offered or all did not give no pair
and are left out. `between_pitch_auc` is the first alone: the pitch's mean prediction against whether
any of its investors offered.

Intervals resample informative pitches with replacement (`panel_bottleneck._bootstrap_counts`); paired
deltas use the same draws for both models, as `panel_bottleneck.paired_deltas` does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from analysis.panel_bottleneck import CONDITIONAL, DELTA_COLUMNS, N_BOOT, SEED, _bootstrap_counts, boot_p
from analysis.within_pitch import clogit

GROUP = "video_id"
UNINFORMATIVE = "ranking uninformative"
TABLE_COLUMNS = ["model", "within_auc", "within_low", "within_high", "pairs", "pitches", "between_auc",
                 "constant_within", "note"]


def informative_pitches(pred: pd.DataFrame, outcome: str = "y") -> list[str]:
    """Pitches with at least one offer and one non-offer, in sorted order."""
    known = pred[pred[outcome].notna()]
    per = known.groupby(GROUP)[outcome].agg(["min", "max"])
    return sorted(str(v) for v in per.index[(per["min"] == 0) & (per["max"] == 1)])


def _per_pitch(pred: pd.DataFrame, column: str, pitches: list[str], outcome: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """(concordant pairs + half the ties, pairs) per pitch, in the order of `pitches`."""
    num, pairs = np.zeros(len(pitches)), np.zeros(len(pitches))
    groups = {str(k): g for k, g in pred[pred[outcome].notna()].groupby(GROUP)}
    for i, pitch in enumerate(pitches):
        g = groups[pitch]
        pos = g.loc[g[outcome] == 1, column].to_numpy(dtype=float)
        neg = g.loc[g[outcome] == 0, column].to_numpy(dtype=float)
        diff = pos[:, None] - neg[None, :]
        num[i] = float((diff > 0).sum() + 0.5 * (diff == 0).sum())
        pairs[i] = float(diff.size)
    return num, pairs


def within_pitch_auc(pred: pd.DataFrame, column: str, pitches: list[str] | None = None,
                     outcome: str = "y") -> tuple[float, int]:
    """(AUC pooled over within-pitch offer/non-offer pairs, number of pairs); NaN with no pair."""
    pitches = informative_pitches(pred, outcome) if pitches is None else [str(p) for p in pitches]
    num, pairs = _per_pitch(pred, column, pitches, outcome)
    total = pairs.sum()
    return (float(num.sum() / total) if total else np.nan), int(total)


def between_pitch_auc(pred: pd.DataFrame, column: str, outcome: str = "y") -> float:
    """AUC of the pitch's mean prediction for whether any of its investors offered; NaN with one class."""
    known = pred[pred[outcome].notna()]
    per = known.groupby(GROUP).agg(score=(column, "mean"), any_offer=(outcome, "max"))
    pos = per.score[per.any_offer == 1].to_numpy(dtype=float)
    neg = per.score[per.any_offer == 0].to_numpy(dtype=float)
    if not len(pos) or not len(neg):
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def _constant_within(pred: pd.DataFrame, column: str, pitches: list[str]) -> bool:
    inside = pred[pred[GROUP].astype(str).isin(pitches)]
    return bool(len(pitches)) and bool((inside.groupby(GROUP)[column].nunique() <= 1).all())


def _draws(num: np.ndarray, pairs: np.ndarray, counts: np.ndarray) -> np.ndarray:
    total = counts @ pairs
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (counts @ num) / total, np.nan)


def within_pitch_table(pred: pd.DataFrame, models: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """One row per model (prediction column p_<model>): within-pitch AUC with a pitch-bootstrap interval,
    its pairs and pitches, the between-pitch AUC, and whether every informative pitch got a constant
    prediction (then the within-pitch AUC is 0.5 by construction: "ranking uninformative")."""
    pitches = informative_pitches(pred)
    counts = _bootstrap_counts(len(pitches), n_boot, seed)
    rows = []
    for model in models:
        column = f"p_{model}"
        num, pairs = _per_pitch(pred, column, pitches)
        auc = float(num.sum() / pairs.sum()) if pairs.sum() else np.nan
        draws = _draws(num, pairs, counts)
        draws = draws[np.isfinite(draws)]
        low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
        constant = _constant_within(pred, column, pitches)
        rows.append({"model": model, "within_auc": auc, "within_low": float(low), "within_high": float(high),
                     "pairs": int(pairs.sum()), "pitches": len(pitches),
                     "between_auc": between_pitch_auc(pred, column), "constant_within": constant,
                     "note": UNINFORMATIVE if constant else ""})
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def within_pitch_deltas(pred: pd.DataFrame, pairs: list[tuple[str, str]], n_boot: int = N_BOOT,
                        seed: int = SEED) -> pd.DataFrame:
    """`later minus earlier` within-pitch AUC for each pair of models, with a paired pitch-bootstrap interval."""
    pitches = informative_pitches(pred)
    counts = _bootstrap_counts(len(pitches), n_boot, seed)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for earlier, later in pairs:
        for name in (earlier, later):
            if name not in cache:
                cache[name] = _per_pitch(pred, f"p_{name}", pitches)
        (n0, p0), (n1, p1) = cache[earlier], cache[later]
        delta = float(n1.sum() / p1.sum() - n0.sum() / p0.sum()) if p0.sum() else np.nan
        draws = _draws(n1, p1, counts) - _draws(n0, p0, counts)
        draws = draws[np.isfinite(draws)]
        low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
        rows.append({"comparison": f"{later} minus {earlier}", "metric": "within_auc", "delta": delta,
                     "ci_low": float(low), "ci_high": float(high), "p_value": boot_p(draws),
                     "n_boot": int(len(draws)), "better": "higher", "uncertainty": CONDITIONAL})
    return pd.DataFrame(rows, columns=DELTA_COLUMNS)


def conditional_logit_check(rows: pd.DataFrame, outcome: str, columns: list[str], ridge: float = 1.0) -> dict:
    """Conditional logit (strata: pitches) of `outcome` on standardised `columns`, over informative pitches
    and rows observed on every column: beta per column, log-likelihoods, and the likelihood-ratio p
    (chi-square, df = number of columns) against all betas zero."""
    known = rows.dropna(subset=[outcome, *columns]).copy()
    pitches = informative_pitches(known, outcome)
    known = known[known[GROUP].astype(str).isin(pitches)].copy()
    for c in columns:
        sd = float(known[c].std(ddof=0))
        known[c] = (known[c] - known[c].mean()) / sd if sd > 1e-12 else 0.0
    groups = [g.assign(offer=g[outcome].astype(int)) for _, g in known.groupby(GROUP)]
    if not groups:
        return {"beta": dict.fromkeys(columns, np.nan), "loglik": np.nan, "loglik0": np.nan, "lr": np.nan,
                "lr_p": np.nan, "pitches": 0, "rows": 0}
    beta, loglik, loglik0 = clogit(groups, list(columns), ridge=ridge)
    lr = max(0.0, 2.0 * float(loglik - loglik0))
    return {"beta": {c: float(b) for c, b in zip(columns, beta)}, "loglik": float(loglik), "loglik0": float(loglik0),
            "lr": lr, "lr_p": float(chi2.sf(lr, df=len(columns))), "pitches": len(groups), "rows": int(len(known))}
