"""The pre-registered decision rules of the pitch-level test (refocus spec, `docs/plans/design docs/
2026-10-19-pitch-level-refocus.md`, section 6), applied to the paired differences of the ladder.

- H1' is supported when C3 - C2p has an AUC gain of at least +0.05 with an interval excluding 0, or a skill gain
  whose interval excludes 0.
- H8 is supported when C3 - C3q has an AUC gain of at least +0.03 with an interval excluding 0.
- Each is reported as "not distinguishable from zero" otherwise, with the power statement beside every verdict.

A gain is a positive difference, so "an interval excluding 0" is an interval wholly above 0. The AUC threshold is
met within a tolerance of 1e-12 (a difference of 0.05 computed as 0.0499999... meets +0.05). A comparison the deltas
do not hold, or one without an interval, is never supported.

The AUC route is withheld only by the backstop of plan B task 7 review, round 2: when either model of the pair has
an AUC interval wholly below 0.5 (`auc_high` < 0.5). The ladder's AUC ranks the base-rate-adjusted held-out score,
under which a model rarely ranks the pitches backwards, and then by chance or through a residual leave-one-out bias.
The AUC route then cannot make a verdict "supported"; H1''s skill route is unaffected, and the thresholds are
unchanged.

The spread rule (`auc_meaningful` False: no features, or held-out predictions within about 3 points of the offer
rate) no longer withholds the route (ruling I3 of plan B's final review, made on the argument before any verdict of
the real run was read): it guarded the raw-probability AUC's leave-one-pitch-out artefact, which the adjusted score
reduces (a residual bias against weak models remains, report section 8), and under that score a near-constant model's ranking is real signal whose noise the bootstrap interval
carries. `auc_meaningful` and `pred_spread` stay in the metrics as diagnostic columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULES = {"H1'": {"pair": ("C2p", "C3"), "auc_min": 0.05, "skill_alternative": True},
         "H8": {"pair": ("C3q", "C3"), "auc_min": 0.03, "skill_alternative": False}}
SUPPORTED = "supported"
NOT_DISTINGUISHABLE = "not distinguishable from zero"
AUC_USED = "used"
AUC_TOLERANCE = 1e-12
VERDICT_COLUMNS = ["hypothesis", "comparison", "auc_delta", "auc_low", "auc_high", "skill_delta", "skill_low",
                   "skill_high", "auc_route", "verdict", "power"]


def power_sentence(deltas: pd.DataFrame, n_pitches: int, n_no_offer: int) -> str:
    """What difference the data can resolve (half the mean width of the AUC intervals in `deltas`), set against the
    rules' AUC thresholds (RULES): a threshold below the resolution means a pass rests on the interval excluding
    zero, not on the size of the gain (fix round 1 of plan B task 9)."""
    auc = deltas[deltas.metric == "auc"] if len(deltas) else deltas
    widths = (auc.ci_high - auc.ci_low).to_numpy(dtype=float) if len(auc) else np.array([])
    widths = widths[np.isfinite(widths)]
    thresholds = [rule["auc_min"] for rule in RULES.values()]
    named = ", ".join(f"+{t:.2f}" for t in thresholds)
    head = f"With {n_no_offer} {'pitch' if n_no_offer == 1 else 'pitches'} without an offer among {n_pitches}, "
    if not len(widths):
        return (head + "the interval widths could not be estimated, so the smallest AUC difference that can be told "
                f"from zero is unknown; the rule's thresholds ({named}) are passed only with an interval excluding zero.")
    resolution = widths.mean() / 2
    head += f"AUC differences smaller than about ±{resolution:.2f} cannot be told from zero; "
    below = [t for t in thresholds if t < resolution - AUC_TOLERANCE]
    if len(below) == len(thresholds):
        return head + f"the rule's thresholds ({named}) are below that resolution, so a pass rests on the interval excluding zero."
    if not below:
        return head + f"the rule's thresholds ({named}) are at or above that resolution."
    words = ", ".join(f"+{t:.2f}" for t in below)
    rest = ", ".join(f"+{t:.2f}" for t in thresholds if t not in below)
    return (head + f"the thresholds {words} are below that resolution (a pass on them rests on the interval excluding "
            f"zero) and {rest} at or above it.")


def _numbers(deltas: pd.DataFrame, comparison: str, metric: str) -> tuple[float, float, float]:
    found = deltas[(deltas.comparison == comparison) & (deltas.metric == metric)] if len(deltas) else deltas
    if not len(found):
        return np.nan, np.nan, np.nan
    row = found.iloc[0]
    return float(row.delta), float(row.ci_low), float(row.ci_high)


def _gain(delta: float, low: float, minimum: float = 0.0) -> bool:
    """A gain of at least `minimum` (within AUC_TOLERANCE, and above zero) whose interval lies wholly above 0;
    False when anything is missing."""
    return bool(delta == delta and low == low and delta > 0 and delta >= minimum - AUC_TOLERANCE and low > 0)


def auc_route(metrics: pd.DataFrame | None, pair: tuple[str, str]) -> str:
    """AUC_USED, or "withheld: AUC interval below 0.5 for <model>" when a model of `pair` has `auc_high` below 0.5
    in `metrics` (no metrics, or no `auc_high`: used). `auc_meaningful` is not read (diagnostic only; ruling I3)."""
    if metrics is None or not len(metrics) or "auc_high" not in metrics:
        return AUC_USED
    m = metrics.set_index("model")
    below = [model for model in pair if model in m.index
             and m.auc_high[model] == m.auc_high[model] and m.auc_high[model] < 0.5]
    return f"withheld: AUC interval below 0.5 for {', '.join(below)}" if below else AUC_USED


def verdicts(deltas: pd.DataFrame, n_pitches: int, n_no_offer: int, metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per hypothesis of RULES: the comparison it reads (later minus earlier), its AUC and skill differences
    with their intervals, whether the AUC route was used (`auc_route`), the verdict ("supported" or "not
    distinguishable from zero") and the power sentence. `deltas` and `metrics` are those of
    `analysis.panel_bottleneck.evaluate` of the ladder (deltas with metrics auc and skill). `metrics` is required:
    without it the AUC route could not be withheld for a model whose AUC interval lies below 0.5; None is refused."""
    if metrics is None:
        raise ValueError("verdicts needs the ladder's metrics frame (auc_high decides the AUC route)")
    power = power_sentence(deltas, n_pitches, n_no_offer)
    out = []
    for hypothesis, rule in RULES.items():
        earlier, later = rule["pair"]
        comparison = f"{later} minus {earlier}"
        auc, skill = _numbers(deltas, comparison, "auc"), _numbers(deltas, comparison, "skill")
        route = auc_route(metrics, rule["pair"])
        supported = ((route == AUC_USED and _gain(auc[0], auc[1], rule["auc_min"]))
                     or (rule["skill_alternative"] and _gain(skill[0], skill[1])))
        out.append({"hypothesis": hypothesis, "comparison": comparison,
                    "auc_delta": auc[0], "auc_low": auc[1], "auc_high": auc[2],
                    "skill_delta": skill[0], "skill_low": skill[1], "skill_high": skill[2], "auc_route": route,
                    "verdict": SUPPORTED if supported else NOT_DISTINGUISHABLE, "power": power})
    return pd.DataFrame(out, columns=VERDICT_COLUMNS)
