"""Concept bottleneck heads for the pitch outcome (does any investor offer?), one row per pitch, evaluated
pitch-held-out: the pre-registered ladder C0-C5 with C2p and C3q (refocus spec, section 6).

LADDER adds one step at a time: the venture, the ask and exposure (C1); the primary Stage A rank score (C2); the
prepared presentation (C2p); the questions alone (C3q); the answers (C3); the reactions (C4); rapport and repair
(C5). Each step is a whole feature family of `analysis.panel_features.pitch_features`, except C2, which adds the
rank score of the book's primary Stage A item only (the other items enter in a sensitivity). Three further models
(spec amendment 6.9, 2026-10-25) add one family alone to the baseline C2p: the answers (C3a), the reactions (C4r),
rapport and repair (C5r); with C3q (C2p + the questions) they are ONE_FAMILY, so that every family is compared with
the baseline and not only with the step before it (BASELINE_COMPARISONS, `baseline_gains`). LADDER_COMPARISONS are
the paired differences the spec names, later minus earlier; `analysis.verdicts` reads two of them. A run without a
ranking has no rank score: C2 then reads the rating of the same item and every metrics row says so
(STAGE_A_FALLBACK_NOTE). SENSITIVITY_LADDERS are the named variants run with the same ladder.

A head sees only the named columns of the feature dictionary that belong to its feature families: no text, label,
name, summary or embedding can reach it, because columns are selected by that dictionary and nothing else. The
ladder and its sensitivities fit fewer columns than their families hold: LADDER_NOT_FITTED (every `_n` count behind
a mean or share, the H5 frame-match share, the exposure columns beyond the spec's C1 list, and `rx_answers`, a
denominator of the reaction shares) stays in the feature table as descriptive columns and is never fitted (`ladder_columns`; ruling I4 of plan B's final review). The
retired shark-level model sets (one row per investor) and the functions that read them are in
`analysis.archive.shark_level_models`, which runs this machinery with the shark-level dictionary.

Every number here is out of fold. All rows of a pitch are held out together, and everything that is
learned (imputation medians, scaling, the pooling of rare categories, which columns are constant, the
ridge penalty, the coefficients) is learned from the training pitches of that fold only.

The penalty is chosen inside each training fold by cross-validation over its pitches (leave one pitch
out, or five grouped folds with more than 25 training pitches), by log loss with every pitch weighing
the same, from PENALTY_GRID; equal losses go to the larger penalty. The inner folds reuse the
preprocessing of the outer training fold, which never sees the held-out pitch. A fixed penalty can be
given instead and is then recorded as such.

Each pitch has the same total weight in the headline metrics (a row weighs 1 / rows of its pitch);
the row-weighted metrics are reported beside them. Intervals come from a paired bootstrap over
pitches of the fixed out-of-fold predictions. They are conditional: training is not refit, so they
understate the uncertainty of the fitted heads. With fewer than 20 pitches every result carries the
label `demonstration only`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from analysis.panel_dataset import (
    AFTER_CUT, ALL_QUESTIONS, HARD_QUESTIONS, PANEL_CHALLENGE, PANEL_QUESTION, PANEL_RECEIPT, panel_next_move,
    support_level,
)
from analysis.panel_features import FAMILIES, FEATURES, STAGE_A_CODES

RANK_PREFIX = "sa_rank_"
RATING_PREFIX = "sa_"
# the book's primary Stage A item: the first id of `books.stage_a_ids()` (pitch_positivity in the live book)
PRIMARY_STAGE_A = STAGE_A_CODES[0]


def ladder(stage_a: tuple[str, ...] = (f"{RANK_PREFIX}{PRIMARY_STAGE_A}",)) -> dict[str, tuple[str, ...]]:
    """The pitch-level ladder C0-C5, each model the one before plus one step; `stage_a` is what C2 adds (the
    primary item's rank score; the all-Stage-A sensitivity gives every item's)."""
    c1 = ("control", "exposure")
    c2 = c1 + tuple(stage_a)
    c2p = c2 + ("prepared",)
    c3q = c2p + ("questions",)
    c3 = c3q + ("answers",)
    c4 = c3 + ("reactions",)
    # the one-family models (spec amendment 6.9, 2026-10-25): the baseline C2p plus one family alone, so that every
    # family is compared with the baseline and not only with the step before it; C3q is already C2p + questions
    return {"C0": (), "C1": c1, "C2": c2, "C2p": c2p, "C3q": c3q, "C3": c3, "C4": c4, "C5": c4 + ("rapport_repair",),
            "C3a": c2p + ("answers",), "C4r": c2p + ("reactions",), "C5r": c2p + ("rapport_repair",)}


LADDER: dict[str, tuple[str, ...]] = ladder()
# the baseline every family is compared with: the venture, the ask, exposure, delivery and the prepared presentation
BASELINE = "C2p"
# the cumulative ladder, each model the one before plus one family (the figures' curves and the verdicts read these)
CUMULATIVE: tuple[str, ...] = ("C0", "C1", "C2", "C2p", "C3q", "C3", "C4", "C5")
# the baseline plus one family alone: {model: the family it adds to BASELINE}
ONE_FAMILY: dict[str, str] = {"C3q": "questions", "C3a": "answers", "C4r": "reactions", "C5r": "rapport_repair"}
# the cumulative model in which the same family enters the ladder
CUMULATIVE_OF: dict[str, str] = {"C3q": "C3q", "C3a": "C3", "C4r": "C4", "C5r": "C5"}
# the paired differences of the pre-registered test, (earlier, later): read as later minus earlier; the last four
# are each family against the baseline (amendment 6.9)
LADDER_COMPARISONS = [("C1", "C2"), ("C2", "C2p"), ("C2p", "C3"), ("C3q", "C3"), ("C3", "C4"), ("C4", "C5"),
                      ("C2p", "C3q"), ("C2p", "C3a"), ("C2p", "C4r"), ("C2p", "C5r"), ("C2p", "C4"), ("C2p", "C5")]
# every family against the baseline: alone (the one-family model) and cumulative (the ladder's model)
BASELINE_COMPARISONS = [(BASELINE, name) for name in ONE_FAMILY] + [(BASELINE, "C3"), (BASELINE, "C4"), (BASELINE, "C5")]
assert all(pair in LADDER_COMPARISONS for pair in BASELINE_COMPARISONS)


CHANCE_MODELS = {
    'C2p': 'Preparedness and delivery of the pitch\n+ control variables (venture, ask,\namount of interaction)',
    'C3q': "+ the sharks' questions",
    'C3a': "+ the entrepreneurs' answers",
    'C4r': "+ the sharks' reactions to the answers",
    'C5r': '+ rapport and repair',
    'C5': '+ all four families',
}


def chance_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Proposal family comparisons against chance, using the existing held-out model results.

    Each one-family model includes C2p's controls, delivery and prepared pitch. The excess is
    AUC minus .5; intervals are the saved pitch-bootstrap AUC intervals. P-values are the
    existing label-permutation tests (perm_p_auc), NOT the paired C2p-difference tests or
    the bootstrap auc_p_value. Missing permutation results stay missing.
    """
    columns = ['model', 'family', 'auc', 'auc_low', 'auc_high', 'above_chance', 'perm_p_auc']
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    models = [m for m in CHANCE_MODELS if m in set(metrics['model'])]
    selected = metrics.set_index('model').loc[models].copy()
    selected['family'] = [CHANCE_MODELS[m] for m in selected.index]
    selected['above_chance'] = selected['auc'] - 0.5
    selected['perm_p_auc'] = selected.get('perm_p_auc', np.nan)
    return selected.reset_index()[columns]


def baseline_gains(deltas: pd.DataFrame, metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per family and metric (auc, skill): the family's gain over BASELINE when it enters alone (the
    one-family model) and when it enters the cumulative ladder, each with its interval and bootstrap p from
    `deltas` (paired_deltas of LADDER_COMPARISONS, which pairs every one of these with the baseline). Columns:
    family, metric, alone_model, alone_delta, alone_low, alone_high, alone_p, cumulative_model, cumulative_delta,
    cumulative_low, cumulative_high, cumulative_p."""
    rows = []
    if deltas is None or not len(deltas):
        return pd.DataFrame(columns=BASELINE_GAIN_COLUMNS)
    index = deltas.set_index(["comparison", "metric"])

    def read(later: str, metric: str) -> tuple[float, float, float, float]:
        key = (f"{later} minus {BASELINE}", metric)
        if key in index.index:
            r = index.loc[key]
            return float(r.delta), float(r.ci_low), float(r.ci_high), float(r.p_value)
        return np.nan, np.nan, np.nan, np.nan

    for alone, family in ONE_FAMILY.items():
        cumulative = CUMULATIVE_OF[alone]
        for metric in ("auc", "skill"):
            a_delta, a_low, a_high, a_p = read(alone, metric)
            c_delta, c_low, c_high, c_p = read(cumulative, metric)
            rows.append({"family": family, "metric": metric, "alone_model": alone, "alone_delta": a_delta,
                         "alone_low": a_low, "alone_high": a_high, "alone_p": a_p, "cumulative_model": cumulative,
                         "cumulative_delta": c_delta, "cumulative_low": c_low, "cumulative_high": c_high,
                         "cumulative_p": c_p})
    return pd.DataFrame(rows, columns=BASELINE_GAIN_COLUMNS)


BASELINE_GAIN_COLUMNS = ["family", "metric", "alone_model", "alone_delta", "alone_low", "alone_high", "alone_p",
                         "cumulative_model", "cumulative_delta", "cumulative_low", "cumulative_high", "cumulative_p"]
# without any rank score in the table (a run without a ranking), each rank score a model names is replaced by the
# rating of the same item, and every metrics row carries the note
STAGE_A_FALLBACK = {f"{RANK_PREFIX}{code}": f"{RATING_PREFIX}{code}" for code in STAGE_A_CODES}
STAGE_A_FALLBACK_NOTE = "rank score absent; rating used"
# the named variants of the ladder (spec section 6.4), each run with the same models' names and comparisons:
# `models` and `penalty` go to `evaluate`; `features` is how the feature table is built (pitch_features(data, **...))
SENSITIVITY_LADDERS = {
    "rates": {"models": LADDER, "penalty": None, "features": {"exposure": "rates"},
              "note": "exposure as rates per minute, not counts"},
    "all_stage_a": {"models": ladder(tuple(f"{RANK_PREFIX}{code}" for code in STAGE_A_CODES)), "penalty": None,
                    "features": {}, "note": "C2 adds the rank score of every Stage A item, not the primary one only"},
    "fixed_penalty_10": {"models": LADDER, "penalty": 10.0, "features": {},
                         "note": "ridge penalty fixed at 10 in every fold, not chosen inside it"},
}


# Ruling I4 (plan B final review), per spec section 6's ladder table, which names means and shares: the ladder's
# designs leave out every `_n` count twin of a mean or share (how many observations it rests on), the H5 frame-match
# share with its twins (H5 is descriptive), and the exposure columns the spec's C1 does not list. They stay in the
# feature table as descriptive columns. `_missing` indicators stay (dropped in a fold where they are constant).
C1_COLUMNS = ("founder_gender", "primary_industry", "product_form", "customer_type", "ask_log_amount",
              "ask_equity_pct", "ask_missing", "hist_turns", "hist_seconds", "hist_questions")
FRAME_MATCH_COLUMNS = ("a_frame_match_share", "a_frame_match_share_n", "a_frame_match_share_missing")
# counts that are a denominator, not a share (ruling of plan B's final fix wave, applied before the v3-dev9 run):
# rx_answers counts the answers with a reaction window; the spec's C4 lists shares
COUNT_COLUMNS = ("rx_answers",)
LADDER_NOT_FITTED: tuple[str, ...] = tuple(dict.fromkeys([
    *(name for name in FEATURES if name.endswith("_n") and name[:-2] in FEATURES),
    *FRAME_MATCH_COLUMNS,
    *(name for name, spec in FEATURES.items() if spec["family"] in LADDER["C1"] and name not in C1_COLUMNS),
    *COUNT_COLUMNS]))
LADDER_NOT_FITTED_NOTE = ("not fitted by any ladder model (descriptive columns of features.csv): every `_n` count "
                          "behind a mean or share, the H5 frame-match share (a_frame_match_share), the exposure "
                          "columns beyond the spec's C1 list (" + ", ".join(
                              n for n in LADDER_NOT_FITTED if FEATURES[n]["family"] in LADDER["C1"]) + ") and the "
                          "count of answers with a reaction window (" + ", ".join(COUNT_COLUMNS) + "), a denominator "
                          "of the reaction shares")


def ladder_columns(frame: pd.DataFrame, spec: tuple[str, ...],
                   not_fitted: tuple[str, ...] = LADDER_NOT_FITTED) -> list[str]:
    """The columns of `frame` a model with this specification fits, in dictionary order, without `not_fitted`
    (LADDER_NOT_FITTED: what the ladder leaves out). Columns that are constant in a training fold are dropped
    there as well (`fit_design`), so a fold may fit fewer."""
    chosen = {c for columns in family_columns(frame, tuple(spec), not_fitted).values() for c in columns}
    return [name for name in FEATURES if name in chosen]


def stage_a_fallback(models: dict, frame: pd.DataFrame | None) -> tuple[dict, str]:
    """(`models`, note): when `frame` has no rank score at all, every rank score a model names is replaced by the
    rating of the same item (STAGE_A_FALLBACK) and the note is STAGE_A_FALLBACK_NOTE; otherwise `models`
    unchanged and no note."""
    if frame is None or any(str(c).startswith(RANK_PREFIX) for c in frame.columns):
        return models, ""
    replaced, out = False, {}
    for name, spec in models.items():
        if isinstance(spec, str):            # refused by check_models with its own message
            out[name] = spec
            continue
        new = tuple(STAGE_A_FALLBACK.get(f, f) for f in spec)
        replaced |= new != tuple(spec)
        out[name] = new
    return (out, STAGE_A_FALLBACK_NOTE) if replaced else (models, "")


STATUS_PRIMARY, STATUS_DESCRIPTIVE, STATUS_EXPLORATORY = "primary", "descriptive", "exploratory"


def feature_status(name: str, ladder_exclusion: bool = True) -> str | None:
    """`primary` for a column the pre-registered ladder reads (a column of a family of its full model C5, the
    primary Stage A rank score, or the rating that stands in for it), `descriptive` for a column it never fits
    (LADDER_NOT_FITTED), `exploratory` for every other declared feature (the other Stage A items, read by the
    all-Stage-A sensitivity only); None for a key column, which is never a predictor. `ladder_exclusion=False` gives
    the status from the families alone, as before ruling I4 (the archived shark-level dictionary reads it so)."""
    family = FEATURES.get(name, {}).get("family")
    if family is None:
        return None
    if ladder_exclusion and name in LADDER_NOT_FITTED:
        return STATUS_DESCRIPTIVE
    full = LADDER["C5"]
    read = family in full or name in full or name in {STAGE_A_FALLBACK[f] for f in full if f in STAGE_A_FALLBACK}
    return STATUS_PRIMARY if read else STATUS_EXPLORATORY


PENALTY_GRID = (1.0, 10.0, 100.0, 1000.0)
INNER_LOO_MAX = 25      # training pitches up to which the inner folds leave one pitch out
INNER_FOLDS = 5
INNER_CV, FIXED = "inner_cv", "fixed"
GROUP = "video_id"
NOT_PREDICTORS: tuple[str, ...] = ()     # declared columns that are never a predictor (none on pitch rows)
POOLED = "other_pooled"
UNKNOWN = "NONE"
DEMONSTRATION = "demonstration only"
HELD_OUT = "pitch-held-out estimate"
MIN_PITCHES = 20
N_BOOT = 1999
SEED = 42
CLIP = 1e-10
METRICS = ("log_loss", "brier", "auc")
SKILL = "skill"         # the difference in skill over the base rate, beside METRICS in `evaluate`'s deltas
BETTER = {"log_loss": "negative", "brier": "negative", "auc": "positive", SKILL: "positive"}
CONDITIONAL = ("conditional: paired bootstrap over pitches of fixed out-of-fold predictions; "
               "training is not refit")
KEY_COLUMNS: tuple[str, ...] = ()        # row keys beside the group, when the table has them (none on pitch rows)
QUESTION_COUNT = "hist_questions"        # the questions asked before the cut: what single_features adjusts for


def label_for(n_groups: int) -> str:
    return DEMONSTRATION if n_groups < MIN_PITCHES else HELD_OUT


# ---- design matrix --------------------------------------------------------------------------

def predictor_names(frame: pd.DataFrame | None = None) -> list[str]:
    """Every feature a head may be given by name: the declared features that belong to a family, minus
    identity; with `frame`, only those the table carries."""
    return [name for name, spec in FEATURES.items()
            if spec["family"] is not None and name not in NOT_PREDICTORS
            and (frame is None or name in frame.columns)]


def check_models(models: dict, frame: pd.DataFrame | None = None) -> dict[str, tuple[str, ...]]:
    """`models` with every specification as a tuple, after checking each name in it.

    A specification is a list of feature families (FAMILIES of analysis.panel_features), of feature
    column names, or of both; an empty list is the base rate. A name that is neither a family nor a
    predictor column (of `frame`, when given) raises a ValueError that lists the valid names.
    """
    valid = predictor_names(frame)
    checked: dict[str, tuple[str, ...]] = {}
    for model, spec in models.items():
        if isinstance(spec, str):
            raise ValueError(f"model {model!r}: give a list of names, e.g. [{spec!r}], not the bare text {spec!r}")
        spec = tuple(spec)
        unknown = [name for name in spec if name not in FAMILIES and name not in valid]
        if unknown:
            raise ValueError(
                f"model {model!r}: unknown feature or family name {', '.join(repr(u) for u in unknown)}.\n"
                f"Valid families: {', '.join(FAMILIES)}.\n"
                f"Valid feature names{' in this table' if frame is not None else ''}: {', '.join(valid)}.")
        checked[str(model)] = spec
    return checked


def family_columns(frame: pd.DataFrame, families: tuple[str, ...],
                   not_fitted: tuple[str, ...] = ()) -> dict[str, list[str]]:
    """The columns of `frame` a head with this specification may read, by how they are encoded.

    Each name in `families` is a feature family (all its columns) or one feature column. A column of `not_fitted`
    is left out however it is named (the ladder passes LADDER_NOT_FITTED).
    """
    out: dict[str, list[str]] = {"categorical": [], "indicator": [], "numeric": []}
    for name, spec in FEATURES.items():
        chosen = spec["family"] in families or (spec["family"] is not None and name in families)
        if not chosen or name not in frame.columns or name in NOT_PREDICTORS or name in not_fitted:
            continue
        kind = {"control": "categorical", "indicator": "indicator"}.get(spec["kind"], "numeric")
        out[kind].append(name)
    return out


def _level(value) -> str:
    return UNKNOWN if value is None or value != value else str(value)


@dataclass
class Design:
    """The preprocessing of one training fold; `transform` applies it to any rows."""
    names: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    numeric: list[tuple[str, float, float, float]] = field(default_factory=list)   # name, median, mean, sd
    indicators: list[str] = field(default_factory=list)
    levels: dict[str, list[str]] = field(default_factory=dict)   # control -> levels kept as columns
    pooled: dict[str, list[str]] = field(default_factory=dict)   # control -> training levels pooled

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = []
        for name, median, mean, sd in self.numeric:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
            columns.append((np.where(np.isfinite(values), values, median) - mean) / sd)
        for name in self.indicators:
            columns.append(pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy(dtype=float))
        for name, kept in self.levels.items():
            known = set(kept) | {UNKNOWN}
            # a level the training pitches did not keep as a column, seen or not, is the pooled level
            values = [v if v in known else POOLED for v in frame[name].map(_level)]
            for level in kept:
                columns.append(np.array([float(v == level) for v in values]))
        return np.column_stack(columns) if columns else np.zeros((len(frame), 0))


def fit_design(train: pd.DataFrame, families: tuple[str, ...], min_category: int = 5,
               group: str = GROUP, not_fitted: tuple[str, ...] = ()) -> Design:
    """Learn a fold's preprocessing from its training rows only, for the columns of `families` without
    `not_fitted` (family_columns).

    A categorical level is rare, and pooled, when fewer than `min_category` distinct training pitches
    have it (rows of one pitch share their controls, so rows would overstate it); a table without the
    `group` column counts rows.
    """
    design = Design()
    columns = family_columns(train, families, tuple(not_fitted))
    for name in columns["numeric"]:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=float)
        observed = values[np.isfinite(values)]
        if name.startswith(RANK_PREFIX) and len(observed) < len(values):
            design.dropped.append(name)          # rank scores are used only when every training pitch has one
            continue
        if not len(observed):
            design.dropped.append(name)
            continue
        median = float(np.median(observed))
        filled = np.where(np.isfinite(values), values, median)
        sd = float(filled.std())
        if sd <= 1e-12:
            design.dropped.append(name)
            continue
        design.numeric.append((name, median, float(filled.mean()), sd))
        design.names.append(name)
    for name in columns["indicator"]:
        values = pd.to_numeric(train[name], errors="coerce").fillna(0).to_numpy(dtype=float)
        if len(values) == 0 or np.all(values == values[0]):
            design.dropped.append(name)
            continue
        design.indicators.append(name)
        design.names.append(name)
    for name in columns["categorical"]:
        values = train[name].map(_level)
        counts = train[group].groupby(values).nunique() if group in train.columns else values.value_counts()
        rare = sorted(level for level, n in counts.items() if n < min_category and level != UNKNOWN)
        recoded = values.where(~values.isin(rare), POOLED)
        kept = []
        for level in sorted(recoded.unique()):
            if recoded.eq(level).all():
                design.dropped.append(f"{name}={level}")
            else:
                kept.append(level)
        design.levels[name] = kept
        design.pooled[name] = rare
        design.names.extend(f"{name}={level}" for level in kept)
    return design


def design_matrix(train: pd.DataFrame, test: pd.DataFrame, families: tuple[str, ...],
                  min_category: int = 5, group: str = GROUP,
                  not_fitted: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(training matrix, test matrix, column names), preprocessing fitted on `train` only.

    Numeric features are median-imputed and standardised with training statistics; `_missing`
    indicators stay 0/1; categorical controls are one-hot, with levels seen in fewer than `min_category`
    training pitches pooled into `other_pooled`, the unknown level NONE kept as its own, and
    a test level without a training column mapped to `other_pooled`. Columns that are constant or
    entirely missing in training are dropped; `fit_design(...).dropped` lists them.
    """
    design = fit_design(train, families, min_category, group, not_fitted)
    return design.transform(train), design.transform(test), list(design.names)


# ---- the head -------------------------------------------------------------------------------

def ridge_logistic(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    """Coefficients (intercept first, unpenalised) of a ridge logistic fit.

    Same objective and Newton iteration as `analysis.auc.fit_predict`, on a matrix that is already
    preprocessed (that function would re-scale the 0/1 columns).
    """
    x = np.column_stack([np.ones(len(x)), x])
    y = np.asarray(y, dtype=float)
    beta = np.zeros(x.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    regular = np.full(len(beta), float(penalty))
    regular[0] = 0.0

    def loss(b):
        logits = x @ b
        return np.logaddexp(0, logits).sum() - y @ logits + 0.5 * np.sum(regular * b * b)

    for _ in range(100):
        prob = 1 / (1 + np.exp(-np.clip(x @ beta, -35, 35)))
        grad = x.T @ (prob - y) + regular * beta
        hessian = (x.T * (prob * (1 - prob))) @ x + np.diag(regular) + np.eye(len(beta)) * 1e-9
        step = np.linalg.solve(hessian, grad)
        rate, old = 1.0, loss(beta)
        while rate > 1e-8 and loss(beta - rate * step) > old:
            rate *= 0.5
        beta = beta - rate * step
        if np.max(np.abs(rate * step)) < 1e-8:
            break
    return beta


def _probabilities(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    z = np.column_stack([np.ones(len(x)), x])
    return 1 / (1 + np.exp(-np.clip(z @ beta, -35, 35)))


def inner_folds(groups: np.ndarray) -> list[np.ndarray]:
    """Boolean masks of the rows each inner fold holds out: one training pitch each up to INNER_LOO_MAX
    pitches, otherwise INNER_FOLDS folds of whole pitches (pitches dealt out in id order)."""
    names = sorted(set(groups))
    if len(names) <= INNER_LOO_MAX:
        return [groups == name for name in names]
    fold_of = {name: i % INNER_FOLDS for i, name in enumerate(names)}
    index = np.array([fold_of[g] for g in groups])
    return [index == k for k in range(INNER_FOLDS)]


def choose_penalty(x: np.ndarray, y: np.ndarray, groups, grid: tuple[float, ...] = PENALTY_GRID) -> tuple[float, dict]:
    """(penalty, log loss per candidate) by cross-validation over the pitches of one training fold.

    Each candidate is scored by the log loss of its inner out-of-fold predictions, every pitch
    weighing the same; the smallest wins and equal losses go to the larger penalty. An inner fold
    whose training rows hold one class predicts their base rate. With fewer than two training
    pitches nothing can be held out and the largest penalty is taken.
    """
    grid = tuple(sorted(float(g) for g in grid))
    groups = np.asarray(groups, dtype=object)
    y = np.asarray(y, dtype=float)
    if len(set(groups)) < 2:
        return grid[-1], {}
    counts = pd.Series(groups).map(pd.Series(groups).value_counts()).to_numpy(dtype=float)
    weights = 1.0 / counts
    losses: dict[float, float] = {}
    folds = inner_folds(groups)
    predictions = {penalty: np.full(len(y), np.nan) for penalty in grid}
    for held in folds:
        y_in = y[~held]
        if len(np.unique(y_in)) < 2:
            for penalty in grid:
                predictions[penalty][held] = y_in.mean()
            continue
        for penalty in grid:
            predictions[penalty][held] = _probabilities(x[held], ridge_logistic(x[~held], y_in, penalty))
    for penalty in grid:
        p = np.clip(predictions[penalty], CLIP, 1 - CLIP)
        losses[penalty] = float(-(weights * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / weights.sum())
    best = grid[-1]
    for penalty in reversed(grid[:-1]):           # from the larger to the smaller: a tie keeps the larger
        if losses[penalty] < losses[best] - 1e-12:
            best = penalty
    return best, losses


def fit_head(train: pd.DataFrame, test: pd.DataFrame, outcome: str, families: tuple[str, ...],
             penalty: float | None = None, min_category: int = 5, group: str = GROUP,
             penalty_grid: tuple[float, ...] = PENALTY_GRID,
             not_fitted: tuple[str, ...] = ()) -> tuple[np.ndarray, dict]:
    """Out-of-fold probabilities for `test` and the fitted schema of the fold.

    `penalty=None` chooses the penalty by cross-validation inside `train` (see choose_penalty); a
    number fixes it. The schema records the penalty used and how it came about. `not_fitted` columns are
    left out of the design (family_columns).
    """
    y = train[outcome].to_numpy(dtype=float)
    base_rate = float(y.mean())
    mode = INNER_CV if penalty is None else FIXED
    if not families:
        return np.full(len(test), base_rate), {"columns": [], "dropped": [], "single_class_training": False,
                                               "intercept": base_rate, "coefficients": {}, "penalty": None,
                                               "penalty_mode": None, "note": "training base rate"}
    design = fit_design(train, families, min_category, group, not_fitted)
    detail = {"columns": list(design.names), "dropped": list(design.dropped),
              "medians": {name: median for name, median, _, _ in design.numeric},
              "means": {name: mean for name, _, mean, _ in design.numeric},
              "sds": {name: sd for name, _, _, sd in design.numeric},
              "pooled_levels": {k: v for k, v in design.pooled.items() if v}}
    if len(np.unique(y)) < 2:
        return np.full(len(test), base_rate), {**detail, "single_class_training": True, "intercept": base_rate,
                                               "coefficients": {}, "penalty": None, "penalty_mode": None,
                                               "note": "training base rate"}
    x = design.transform(train)
    cv: dict = {}
    if penalty is None:
        if group not in train.columns:
            raise ValueError(f"choosing the penalty needs the pitch of every training row (column {group!r})")
        penalty, cv = choose_penalty(x, y, train[group].to_numpy(), penalty_grid)
    beta = ridge_logistic(x, y, penalty)
    prob = _probabilities(design.transform(test), beta)
    return prob, {**detail, "single_class_training": False, "intercept": float(beta[0]),
                  "coefficients": dict(zip(design.names, (float(b) for b in beta[1:]))),
                  "penalty": float(penalty), "penalty_mode": mode,
                  "penalty_cv_log_loss": {f"{k:g}": v for k, v in cv.items()}}


# ---- metrics --------------------------------------------------------------------------------

def weighted_auc(y, p, w) -> float:
    y, p, w = (np.asarray(v, dtype=float) for v in (y, p, w))
    pos, neg = y == 1, y == 0
    if not pos.any() or not neg.any():
        return np.nan
    wins = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    return float(w[pos] @ wins @ w[neg] / (w[pos].sum() * w[neg].sum()))


# What the AUC of an evaluation ranks (ruled before any v3-dev9 analysis, plan B task 7 review, round 2). Under
# leave-one-pitch-out the training base rate of a fold is (S - y_i) / (N - 1): it moves against the held-out pitch's
# own outcome, so a model that stays near the base rate ranks the pitches backwards (AUC near 0, an artefact).
# BASE_RATE_ADJUSTED ranks logit(p_heldout) - logit(b_fold) instead, the held-out prediction measured from the
# training base rate of its fold: a pure intercept model scores 0 everywhere (AUC exactly 0.5). RAW ranks the
# probabilities themselves, as before (the archived shark-level path). Log loss, Brier score and skill always
# read the probabilities.
BASE_RATE_ADJUSTED, RAW = "base_rate_adjusted", "raw"
AUC_SCORES = (BASE_RATE_ADJUSTED, RAW)
AUC_SCORE_NOTE = ("AUC is computed on the base-rate-adjusted held-out score logit(p) - logit(b_fold), b_fold the "
                  "training base rate of the pitch's fold, which removes the shift of the base rate and reduces the "
                  "leave-one-pitch-out artefact (a residual bias against weak models remains); "
                  "log loss, Brier score and skill read the probabilities")


def adjusted_score(p, base_p) -> np.ndarray:
    """logit(p) - logit(base_p), both clipped to [CLIP, 1 - CLIP]: a held-out prediction measured from the training
    base rate of its fold (BASE_RATE_ADJUSTED)."""
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    base_p = np.clip(np.asarray(base_p, dtype=float), CLIP, 1 - CLIP)
    return np.log(p / (1 - p)) - np.log(base_p / (1 - base_p))


def metrics_of(y, p, w=None) -> dict:
    """Log loss, Brier score, AUC (NaN when one class is absent), mean prediction and base rate."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), CLIP, 1 - CLIP)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    total = w.sum()
    return {"log_loss": float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / total),
            "brier": float((w * (p - y) ** 2).sum() / total),
            "auc": weighted_auc(y, p, w),
            "mean_pred": float((w * p).sum() / total),
            "base_rate": float((w * y).sum() / total)}


def skill_over_base(log_loss: float, base_log_loss: float) -> float:
    """Skill over the base rate in percent: 100 x (1 - log loss / base-rate log loss). Higher is better;
    0 is no better than always predicting the training offer rate, negative is worse, 100 is perfect."""
    if not base_log_loss or base_log_loss != base_log_loss or log_loss != log_loss:
        return np.nan
    return float(100.0 * (1.0 - log_loss / base_log_loss))


def base_reference(groups: pd.Series, y: np.ndarray, base_p: np.ndarray, w: np.ndarray) -> dict:
    """The bar every model is read against, from the held-out base-rate predictions `base_p` (for each row
    the offer rate of the other pitches): its log loss (pitches weighted equally), the mean rate it
    predicts, the log loss of always saying 50% (ln 2), and the correlation over pitches between the rate
    predicted for a held-out pitch and that pitch's own offer rate. That correlation is strongly negative
    by construction (taking a pitch with offers out lowers the rate of the rest), which is why a model
    that predicts almost a constant shows an AUC far below 0.5 under leave-one-pitch-out."""
    y, base_p, w = (np.asarray(v, dtype=float) for v in (y, base_p, w))
    per = pd.DataFrame({"g": np.asarray(groups), "y": y, "p": base_p}).groupby("g").mean()
    varies = per.y.std() > 1e-12 and per.p.std() > 1e-12
    return {"log_loss": metrics_of(y, base_p, w)["log_loss"], "predicted_rate": float((w * base_p).sum() / w.sum()),
            "coin_log_loss": float(np.log(2.0)),
            "corr_with_pitch_rate": float(np.corrcoef(per.p, per.y)[0, 1]) if varies else np.nan}


NEAR_CONSTANT_SPREAD = 0.03     # a rule of thumb: predictions within about 3 points of the offer rate


def metric_intervals(groups: pd.Series, y: np.ndarray, p: np.ndarray, base_p: np.ndarray, w: np.ndarray,
                     n_boot: int = N_BOOT, seed: int = SEED, ranks: bool = True, score: np.ndarray | None = None) -> dict:
    """95% pitch-bootstrap intervals of one model's AUC and of its skill over the base rate (fixed held-out
    predictions, whole pitches resampled, the same draws as `paired_deltas`), and whether its AUC means
    anything.

    `pred_spread` is the standard deviation (pitches weighted equally) of the model's prediction minus the
    base-rate prediction of the same fold: how far the model moves away from the offer rate at all. Below
    NEAR_CONSTANT_SPREAD, or for a model without features (`ranks=False`), the model is near-constant:
    its ordering of the rows is mostly the ordering of the leave-one-out base rates, which runs against
    the outcome by construction, so `auc_meaningful` is False: a warning for the raw-probability AUC (RAW, the
    archived path). The base-rate-adjusted score reduces that artefact (a residual bias against weak models remains,
    so their AUCs can sit a little below 0.5), and `auc_meaningful` and
    `pred_spread` are diagnostic columns only: `analysis.verdicts` does not read them (ruling I3, plan B final
    review).

    With `score` (the base-rate-adjusted score, BASE_RATE_ADJUSTED) the AUC interval and its p-value rank the score,
    and they are given for a model without features too (all ties: AUC 0.5); otherwise they rank `p` and are
    missing for a model without features.

    `auc_base_rate_corr` is a diagnostic only: the Spearman correlation between each held-out prediction and that
    fold's training base rate (NaN when either side is constant). It does not decide `auc_meaningful`.
    """
    order = {g: i for i, g in enumerate(sorted(pd.unique(groups)))}
    group_index = pd.Series(np.asarray(groups)).map(order).to_numpy()
    counts = _bootstrap_counts(len(order), n_boot, seed)
    model, base = _boot_metrics(y, p, w, group_index, counts, score), _boot_metrics(y, base_p, w, group_index, counts)
    has_auc = ranks or score is not None
    with np.errstate(invalid="ignore", divide="ignore"):
        skill = 100.0 * (1.0 - model["log_loss"] / base["log_loss"])
    moved = p - base_p
    centre = float((w * moved).sum() / w.sum())
    spread = float(np.sqrt((w * (moved - centre) ** 2).sum() / w.sum()))
    auc_low, auc_high = _interval(model["auc"]) if has_auc else (np.nan, np.nan)
    skill_low, skill_high = _interval(skill)
    return {"auc_low": auc_low, "auc_high": auc_high, "skill_low": skill_low, "skill_high": skill_high,
            "skill_p_value": boot_p(skill) if ranks else np.nan,              # null: skill 0, the offer rate
            "auc_p_value": boot_p(model["auc"], 0.5) if has_auc else np.nan,  # null: AUC 0.5, chance
            "pred_spread": spread, "auc_base_rate_corr": base_rate_corr(p, base_p),
            "auc_meaningful": bool(ranks and spread >= NEAR_CONSTANT_SPREAD)}


def base_rate_corr(p, base_p) -> float:
    """Spearman correlation between held-out predictions and the training base rates of their folds; NaN when
    either is constant."""
    p, base_p = pd.Series(np.asarray(p, dtype=float)), pd.Series(np.asarray(base_p, dtype=float))
    if p.nunique() < 2 or base_p.nunique() < 2:
        return np.nan
    return float(p.rank().corr(base_p.rank()))


def _interval(draws: np.ndarray) -> tuple[float, float]:
    draws = draws[np.isfinite(draws)]
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975])) if len(draws) else (np.nan, np.nan)


BOOT_P_NOTE = ("two-sided bootstrap p-value from the same pitch-resampling draws as the interval: 2 x the smaller of "
               "the share of draws at or below the null value and the share at or above it, never below "
               "1 / (draws + 1)")
FDR_LEVEL = 0.05
FDR_NOTE = ("p_value: two-sided pitch-level permutation p-value of the correlation (each pitch's set of outcomes is "
            "handed to another pitch with the same number of rows; p = (1 + permutations with a correlation at least "
            "as large in size) / (1 + permutations)); p_boot: the bootstrap p-value from the interval's draws; "
            "q_value: Benjamini-Hochberg false-discovery-rate adjustment of p_value within this table; "
            f"significant_fdr: q < {FDR_LEVEL:g}")


def permuted_outcomes(groups, y: np.ndarray, n_perm: int, seed: int = SEED) -> np.ndarray:
    """`n_perm` x rows: the outcome column under pitch-level permutations (`permutation_orders`)."""
    return np.asarray(y, dtype=float)[permutation_orders(groups, n_perm, seed)]


def permutation_p_corr(x: np.ndarray, y: np.ndarray, permuted: np.ndarray) -> float:
    """The two-sided permutation p-value of the correlation of `x` with `y`: the share of the permuted
    outcome columns (`permuted`, permutations x rows) whose correlation with `x` is at least as large in
    size, counting the observed one. A permutation in which the outcome does not vary counts as 0."""
    if not len(permuted) or len(x) < 3:
        return np.nan
    xc = x - x.mean()
    norm = np.sqrt((xc ** 2).sum())
    if norm <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    observed = abs(float(xc @ (y - y.mean())) / (norm * np.sqrt(((y - y.mean()) ** 2).sum())))
    centred = permuted - permuted.mean(axis=1, keepdims=True)
    spread = np.sqrt((centred ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(spread > 1e-12, np.abs(centred @ xc) / (norm * spread), 0.0)
    return float((1 + (r >= observed - 1e-12).sum()) / (1 + len(permuted)))


def boot_p(draws, null: float = 0.0) -> float:
    """A two-sided p-value from bootstrap draws of an estimate (see BOOT_P_NOTE): how often resampling
    pitches puts the estimate on the other side of `null`, doubled. NaN without finite draws."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if not len(draws):
        return np.nan
    tail = min(float((draws <= null).mean()), float((draws >= null).mean()))
    return float(min(1.0, max(2.0 * tail, 1.0 / (len(draws) + 1))))


def bh_q(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values of a set of p-values (missing ones stay missing and do not count as tests):
    q of the i-th smallest p is the smallest p_j x m / j over j >= i, capped at 1."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    known = np.flatnonzero(np.isfinite(p))
    if len(known):
        order = known[np.argsort(p[known], kind="stable")]
        scaled = p[order] * len(order) / np.arange(1, len(order) + 1)
        q[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return q


def roc_points(y, p, w=None) -> tuple[np.ndarray, np.ndarray]:
    """(false-positive rate, true-positive rate) of the ROC curve of predictions `p`, rows weighted by `w`,
    from (0, 0) to (1, 1). Tied predictions move both rates at once (a diagonal step), so the area under
    the curve equals `weighted_auc`. Empty arrays when one class is absent."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    pos, neg = w * (y == 1), w * (y == 0)
    if pos.sum() <= 0 or neg.sum() <= 0:
        return np.array([]), np.array([])
    order = np.argsort(-p, kind="stable")
    ranked = p[order]
    last = np.r_[np.nonzero(np.diff(ranked))[0], len(ranked) - 1]      # the last row of every distinct prediction
    hits, alarms = np.cumsum(pos[order]), np.cumsum(neg[order])
    return np.r_[0.0, alarms[last] / alarms[-1]], np.r_[0.0, hits[last] / hits[-1]]


def pitch_weights(groups: pd.Series) -> np.ndarray:
    """1 / rows of the row's pitch: every pitch then has total weight 1."""
    return (1.0 / groups.map(groups.value_counts())).to_numpy(dtype=float)


def _bootstrap_counts(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = rng.integers(n_groups, size=(n_boot, n_groups))
    counts = np.zeros((n_boot, n_groups))
    np.add.at(counts, (np.arange(n_boot)[:, None], draws), 1)
    return counts


def _boot_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray, group_index: np.ndarray,
                  counts: np.ndarray, score: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Each metric in every bootstrap draw of pitches, pitches weighted equally; the AUC ranks `score` when given."""
    p = np.clip(p, CLIP, 1 - CLIP)
    multiplicity = counts[:, group_index] * w                      # draws x rows
    total = multiplicity.sum(axis=1)
    out = {"log_loss": -(multiplicity @ (y * np.log(p) + (1 - y) * np.log(1 - p))) / total,
           "brier": (multiplicity @ (p - y) ** 2) / total}
    pos, neg = y == 1, y == 0
    s = p if score is None else np.asarray(score, dtype=float)
    wins = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
    wp, wn = multiplicity[:, pos], multiplicity[:, neg]
    denominator = wp.sum(axis=1) * wn.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["auc"] = np.where(denominator > 0, np.einsum("bi,ij,bj->b", wp, wins, wn) / denominator, np.nan)
    return out


DELTA_COLUMNS = ["comparison", "metric", "delta", "ci_low", "ci_high", "p_value", "n_boot", "better", "uncertainty"]


def paired_deltas(pred: pd.DataFrame, pairs: list[tuple[str, str]], group: str = "video_id",
                  n_boot: int = N_BOOT, seed: int = SEED, base: np.ndarray | None = None,
                  auc_score: str = RAW) -> pd.DataFrame:
    """`later minus earlier` for each pair of prediction columns, with a paired pitch-bootstrap interval.

    `auc_score` RAW ranks the `p_<model>` columns for the AUC; BASE_RATE_ADJUSTED ranks the `score_<model>` columns
    (`evaluate` writes them). The other metrics always read `p_<model>`.

    With `base`, the base-rate prediction of every row, each pair also gets a row of metric SKILL: the difference
    in skill over the base rate (percent, `skill_over_base`), the base log loss recomputed in every draw. It has
    the sign of the log-loss improvement in every draw, scaled to the base rate's loss."""
    groups = pred[group]
    order = {g: i for i, g in enumerate(sorted(groups.unique()))}
    group_index = groups.map(order).to_numpy()
    y = pred["y"].to_numpy(dtype=float)
    w = pitch_weights(groups)
    counts = _bootstrap_counts(len(order), n_boot, seed)
    cache: dict[str, dict] = {}
    if base is not None:
        base = np.asarray(base, dtype=float)
        base_loss = (metrics_of(y, base, w)["log_loss"], _boot_metrics(y, base, w, group_index, counts)["log_loss"])
    rows = []
    for earlier, later in pairs:
        for name in (earlier, later):
            if name not in cache:
                p = pred[f"p_{name}"].to_numpy(dtype=float)
                score = pred[f"score_{name}"].to_numpy(dtype=float) if auc_score == BASE_RATE_ADJUSTED else None
                observed = metrics_of(y, p, w)
                if score is not None:
                    observed["auc"] = weighted_auc(y, score, w)
                cache[name] = {"observed": observed, "boot": _boot_metrics(y, p, w, group_index, counts, score)}
        for metric in METRICS:
            delta = cache[later]["observed"][metric] - cache[earlier]["observed"][metric]
            draws = cache[later]["boot"][metric] - cache[earlier]["boot"][metric]
            draws = draws[np.isfinite(draws)]
            low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
            rows.append({"comparison": f"{later} minus {earlier}", "metric": metric, "delta": delta,
                         "ci_low": float(low), "ci_high": float(high), "p_value": boot_p(draws),
                         "n_boot": int(len(draws)), "better": BETTER[metric], "uncertainty": CONDITIONAL})
        if base is not None:
            observed, boot = base_loss
            delta = (skill_over_base(cache[later]["observed"]["log_loss"], observed)
                     - skill_over_base(cache[earlier]["observed"]["log_loss"], observed))
            with np.errstate(invalid="ignore", divide="ignore"):
                draws = 100.0 * (cache[earlier]["boot"]["log_loss"] - cache[later]["boot"]["log_loss"]) / boot
            draws = draws[np.isfinite(draws)]
            low, high = np.quantile(draws, [0.025, 0.975]) if len(draws) else (np.nan, np.nan)
            rows.append({"comparison": f"{later} minus {earlier}", "metric": SKILL, "delta": delta,
                         "ci_low": float(low), "ci_high": float(high), "p_value": boot_p(draws),
                         "n_boot": int(len(draws)), "better": BETTER[SKILL], "uncertainty": CONDITIONAL})
    return pd.DataFrame(rows, columns=DELTA_COLUMNS)


# ---- evaluation -----------------------------------------------------------------------------

def _comparisons(names: list[str]) -> list[tuple[str, str]]:
    """The default pairs of `evaluate`: LADDER_COMPARISONS for the ladder, consecutive models otherwise."""
    return list(LADDER_COMPARISONS) if names == list(LADDER) else list(zip(names[:-1], names[1:]))


def _coding(known: pd.DataFrame, group: str) -> dict:
    if "source" not in known.columns:
        return {}
    per_pitch = known.groupby(group)["source"].first()
    return {str(k): int(v) for k, v in per_pitch.value_counts().sort_index().items()}


def _empty(known: pd.DataFrame, group: str, reason: str) -> dict:
    n_groups = int(known[group].nunique())
    return {"n_rows": int(len(known)), "n_groups": n_groups, "n_events": int(known["y"].sum()) if len(known) else 0,
            "label": label_for(n_groups), "coding": _coding(known, group), "status": reason, "penalty": {}, "base": {},
            "predictions": pd.DataFrame(), "metrics": pd.DataFrame(), "deltas": pd.DataFrame(), "folds": []}


def penalty_setting(penalty: float | None, grid: tuple[float, ...] = PENALTY_GRID) -> dict:
    """How the penalty of a run was set, for a manifest: chosen by inner cross-validation, or fixed."""
    if penalty is None:
        return {"mode": INNER_CV, "grid": [float(g) for g in grid], "text": f"penalty: {INNER_CV}",
                "rule": (f"per outer training fold and model: leave-one-pitch-out up to {INNER_LOO_MAX} training "
                         f"pitches, else {INNER_FOLDS} grouped folds; log loss with equal weight per pitch; "
                         "ties to the larger penalty")}
    return {"mode": FIXED, "value": float(penalty), "text": f"penalty: {FIXED} {float(penalty):g}"}


def _penalty_text(details: list[dict], penalty: float | None) -> str:
    """The penalties one model used over the folds, e.g. "inner_cv: 10 x3, 100 x2" or "fixed 10"."""
    used = [d["penalty"] for d in details if d.get("penalty") is not None]
    if not used:
        return "none fitted"
    if penalty is not None:
        return f"{FIXED} {float(penalty):g}"
    counts = pd.Series(used).value_counts().sort_index()
    return f"{INNER_CV}: " + ", ".join(f"{value:g} x{n}" for value, n in counts.items())


def evaluate(rows: pd.DataFrame, outcome: str, group: str = GROUP, penalty: float | None = None,
             models: dict | None = None, min_category: int = 5, n_boot: int = N_BOOT, seed: int = SEED,
             penalty_grid: tuple[float, ...] = PENALTY_GRID,
             comparisons: list[tuple[str, str]] | None = None, auc_score: str = BASE_RATE_ADJUSTED,
             not_fitted: tuple[str, ...] | None = None) -> dict:
    """Leave-one-pitch-out evaluation of every model on the rows whose outcome is known.

    `models` maps a model name to its specification: feature families, feature column names, or both
    (see check_models; an unknown name is an error that lists the valid ones). The default is LADDER.
    When the table has no Stage A rank score at all, every rank score a model names is replaced by the
    rating of the same item and every metrics row carries STAGE_A_FALLBACK_NOTE in `note`.

    `comparisons` replaces the default pairs of `deltas` (LADDER_COMPARISONS for the ladder,
    consecutive models otherwise) with the given (earlier, later) pairs.

    `not_fitted` are columns no model fits, however its specification names them (family_columns). None (the
    default) is LADDER_NOT_FITTED for the default models (the ladder) and nothing for models given by the caller:
    the pitch-level analysis passes LADDER_NOT_FITTED with its ladder and sensitivities, the archived and legacy
    paths pass nothing. The result records it (`not_fitted`) and each model's design columns on the rows with a
    known outcome (`design_columns`, `ladder_columns`), before a fold drops a constant one.

    All models see the same rows and the same folds. `predictions` holds one row per model row
    (group, row key, y, weight, p_<model>...); `metrics` weights pitches equally and repeats each
    metric row-weighted as `<metric>_rows`; `deltas` compares the pairs (later minus earlier) in log
    loss, Brier score, AUC and skill over the base rate, with conditional bootstrap intervals; `folds`
    records what each fold fitted, the penalty each model used there included.

    `penalty=None` (the default) chooses the penalty inside every training fold, per model, by
    cross-validation over that fold's pitches; a number fixes it for every fold and model.

    `auc_score` (AUC_SCORES) says what the AUC ranks. BASE_RATE_ADJUSTED (the default, the pitch-level ladder):
    the score logit(p) - logit(b_fold), b_fold the training base rate of the row's fold (a row mean, pitch-weighted
    only because the ladder has one row per pitch; what the intercept-only
    head predicts there), written beside each `p_<model>` as `score_<model>`; the metrics' `auc`, its interval and
    p-value, the AUC deltas and (through `auc_score` in the result) the permutation test's AUC all read it, and a
    model without features scores 0 everywhere (AUC 0.5). RAW (the archived shark-level path): the probabilities,
    and a model without features gets a missing AUC, as does any AUC difference against it, rather than the
    artefact of leave-one-out base rates.
    """
    if auc_score not in AUC_SCORES:
        raise ValueError(f"auc_score must be one of {', '.join(AUC_SCORES)}, not {auc_score!r}")
    adjusted = auc_score == BASE_RATE_ADJUSTED
    if not_fitted is None:
        not_fitted = LADDER_NOT_FITTED if models is None else ()
    not_fitted = tuple(not_fitted)
    models, note = stage_a_fallback(LADDER if models is None else models, rows)
    models = check_models(models, rows)
    known = rows[rows[outcome].notna()].reset_index(drop=True).copy()
    known["y"] = known[outcome].astype(float)
    groups = sorted(known[group].unique())
    if len(groups) < 2:
        return _empty(known, group, "not estimable: fewer than two pitches with a known outcome")
    keys = [group] + [c for c in KEY_COLUMNS if c in known.columns] + (["source"] if "source" in known else [])
    pred = known[keys + ["y"]].copy()
    pred["weight"] = pitch_weights(known[group])
    for name in models:
        pred[f"p_{name}"] = np.nan
    folds = []
    # the training offer rate of each row's fold, the bar to beat (b_fold): a mean over the fold's ROWS, so it is
    # pitch-weighted only because the pitch-level ladder has one row per pitch (the archived shark rows are not)
    base_p = np.empty(len(known))
    for held_out in groups:
        test_mask = (known[group] == held_out).to_numpy()
        train, test = known[~test_mask], known[test_mask]
        base_p[test_mask] = float(train["y"].mean())
        fold = {"held_out": held_out, "n_train": int(len(train)), "n_test": int(len(test)),
                "train_groups": sorted(train[group].unique()), "models": {}}
        for name, families in models.items():
            prob, detail = fit_head(train, test, "y", tuple(families), penalty, min_category, group, penalty_grid,
                                    not_fitted)
            pred.loc[test_mask, f"p_{name}"] = prob
            fold["models"][name] = detail
        folds.append(fold)
    columns = [f"p_{name}" for name in models]
    assert pred[columns].notna().all().all(), "every model must predict every row exactly once"
    if adjusted:
        for name in models:
            pred.insert(pred.columns.get_loc(f"p_{name}") + 1, f"score_{name}",
                        adjusted_score(pred[f"p_{name}"].to_numpy(dtype=float), base_p))

    y, w = pred["y"].to_numpy(), pred["weight"].to_numpy()
    constant = {name for name, families in models.items() if not families}
    base = base_reference(pred[group], y, base_p, w)
    table = []
    for name in models:
        p = pred[f"p_{name}"].to_numpy()
        score = pred[f"score_{name}"].to_numpy() if adjusted else None
        by_pitch, by_row = metrics_of(y, p, w), metrics_of(y, p)
        if adjusted:
            by_pitch["auc"], by_row["auc"] = weighted_auc(y, score, w), weighted_auc(y, score, np.ones(len(y)))
        elif name in constant:      # a constant has no ranking; leave-one-out base rates would fake one
            by_pitch["auc"] = by_row["auc"] = np.nan
        by_pitch = {**by_pitch, "base_log_loss": base["log_loss"],
                    "skill_pct": skill_over_base(by_pitch["log_loss"], base["log_loss"]),
                    **metric_intervals(known[group], y, p, base_p, w, n_boot, seed, ranks=name not in constant,
                                       score=score)}
        table.append({"model": name, "families": "+".join(models[name]) or "(base rate)", **by_pitch,
                      **{f"{k}_rows": v for k, v in by_row.items()},
                      "single_class_folds": sum(f["models"][name]["single_class_training"] for f in folds),
                      "penalty": _penalty_text([f["models"][name] for f in folds], penalty),
                      "role": "main", "note": note})
    pairs = _comparisons(list(models)) if comparisons is None else [tuple(pair) for pair in comparisons]
    deltas = paired_deltas(pred, pairs, group, n_boot, seed, base=base_p, auc_score=auc_score)
    if len(deltas) and constant and not adjusted:
        against_constant = deltas.comparison.map(lambda c: bool(set(c.split(" minus ")) & constant))
        undefined = against_constant & (deltas.metric == "auc")
        deltas.loc[undefined, ["delta", "ci_low", "ci_high", "p_value"]] = np.nan
        deltas.loc[undefined, "n_boot"] = 0
    n_groups = len(groups)
    return {"n_rows": int(len(known)), "n_groups": n_groups, "n_events": int(y.sum()),
            "label": label_for(n_groups), "coding": _coding(known, group), "status": "estimated",
            "penalty": penalty_setting(penalty, penalty_grid), "base": base, "auc_score": auc_score,
            "not_fitted": list(not_fitted),
            "design_columns": {name: ladder_columns(known, spec, not_fitted) for name, spec in models.items()},
            "predictions": pred, "metrics": pd.DataFrame(table), "deltas": deltas, "folds": folds}


def count_behind_a_mean(name: str) -> bool:
    """True for the `<statistic>_n` companion columns: how many observations a mean or share rests on.
    For the own, other and pooled statistics they all count (nearly) the same thing, the questions and
    answers there were, so they are one engagement signal under many names."""
    return name.endswith("_n") and name[:-2] in FEATURES and f"{name[:-2]}_missing" in FEATURES


# ---- answer quality by whether an offer came (descriptive) ------------------------------------

BY_OFFER_COLUMNS = ["questions", "whose", "measure", "feature", "n", "n_offered", "n_not_offered", "mean_offered",
                    "mean_not_offered", "difference", "correlation", "corr_low", "corr_high", "p_value", "p_boot",
                    "q_value", "significant_fdr", "n_boot", "n_perm"]
BY_OFFER_QUESTIONS = (("all questions", ""), ("hard questions only (severity 2-3)", "_hard"))
# whose questions: the pitch rows pool every investor's (the shark rows' own and other blocks are in the archive)
BY_OFFER_WHOSE = (("all sharks' questions (pooled)", "a_"),)
BY_OFFER_MEASURES = (("substance of the answers, mean (1-5)", "subs{}_mean"),
                     ("share of answers with checkable support", "evid{}_share"))


def _with_fdr(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`table` with the Benjamini-Hochberg q-value of its `p_value` column and the flag q < FDR_LEVEL."""
    q = bh_q(table.p_value) if len(table) else np.array([])
    return table.assign(q_value=q, significant_fdr=[bool(v < FDR_LEVEL) for v in q])[columns]


def answer_quality_by_offer(rows: pd.DataFrame, outcome: str, group: str = GROUP, n_boot: int = N_BOOT,
                            seed: int = SEED, whose: tuple[tuple[str, str], ...] = BY_OFFER_WHOSE) -> pd.DataFrame:
    """Answer quality by whether an offer came: one row per kind of question (all, hard only), whose
    questions (`whose`: (words, column prefix) pairs; pooled on pitch rows) and measure (substance,
    checkable support).

    On the rows with a known outcome and a value: the mean among rows with an offer, the mean among rows
    without, their difference, and the correlation of the feature with the outcome with a 95% percentile
    interval from resampling whole pitches, its two-sided pitch-level permutation p-value (`p_value`, as many
    permutations as draws; `p_boot` is the bootstrap p-value) and the Benjamini-Hochberg q-value over the
    rows of this table (`q_value`, `significant_fdr`): see FDR_NOTE. `n` is the number
    of rows with a value (an investor who asked no hard question has no value for the hard-question rows). The features are the history-cut ones of
    the feature table, so nothing after the decision enters. Descriptive, not a test.
    """
    known = rows[rows[outcome].notna()].reset_index(drop=True)
    if known.empty:
        return pd.DataFrame(columns=BY_OFFER_COLUMNS)
    y = known[outcome].to_numpy(dtype=float)
    names = sorted(known[group].unique())
    group_index = known[group].map({g: i for i, g in enumerate(names)}).to_numpy()
    multiplicity = _bootstrap_counts(len(names), n_boot, seed)[:, group_index]
    permuted = permuted_outcomes(known[group].to_numpy(), y, n_boot, seed)
    out = []
    for questions, hard in BY_OFFER_QUESTIONS:
        for words, prefix in whose:
            for measure, pattern in BY_OFFER_MEASURES:
                feature = prefix + pattern.format(hard)
                if feature not in known.columns:
                    continue
                x = pd.to_numeric(known[feature], errors="coerce").to_numpy(dtype=float)
                seen = np.isfinite(x)
                xs, ys = x[seen], y[seen]
                yes, no = xs[ys == 1], xs[ys == 0]
                mean_yes, mean_no = (float(v.mean()) if len(v) else np.nan for v in (yes, no))
                r = float(_weighted_corr(np.ones((1, len(xs))), xs, ys)[0]) if len(xs) > 2 else np.nan
                draws = _weighted_corr(multiplicity[:, seen], xs, ys) if r == r else np.array([])
                low, high = _interval(draws)
                out.append({"questions": questions, "whose": words, "measure": measure, "feature": feature,
                            "n": int(seen.sum()), "n_offered": len(yes), "n_not_offered": len(no),
                            "mean_offered": mean_yes, "mean_not_offered": mean_no, "difference": mean_yes - mean_no,
                            "correlation": r, "corr_low": low, "corr_high": high,
                            "p_value": permutation_p_corr(xs, ys, permuted[:, seen]) if r == r else np.nan,
                            "p_boot": boot_p(draws), "n_boot": int(n_boot), "n_perm": int(n_boot)})
    return _with_fdr(pd.DataFrame(out, columns=[c for c in BY_OFFER_COLUMNS if c not in ("q_value", "significant_fdr")]),
                     BY_OFFER_COLUMNS)


# ---- H4: the panel's first reaction after an answer, by checkable support ----------------------

H4_SHARE_MOVES = (PANEL_RECEIPT, PANEL_QUESTION, PANEL_CHALLENGE)
H4_SHARE_COLUMNS = ["questions", "move", "n_yes", "count_yes", "share_yes", "yes_low", "yes_high", "n_no", "count_no",
                    "share_no", "no_low", "no_high", "difference", "diff_low", "diff_high", "p_value", "n_not_rated", "n_pitches",
                    "n_boot"]


def h4_support_shares(followups: pd.DataFrame, moves: tuple[str, ...] = H4_SHARE_MOVES, group: str = GROUP,
                      n_boot: int = N_BOOT, seed: int = SEED) -> pd.DataFrame:
    """The share of answers followed by each first reaction of the panel, by whether the answer gave
    checkable support, with intervals from resampling whole pitches.

    Input: `answer_followups` of the whole encounter (no cut; one row = an answer and an investor question
    it responds to). One output row per set of questions (all, hard = severity 2 or 3) and move
    (`panel_next_move`: receipt, question, challenge). `share_yes` = count_yes / n_yes among answers with
    checkable support, `share_no` likewise among answers without, `difference` = share_yes - share_no.
    The counts are those of `h4_tables(...)["panel_by_support"]`. Each 95% percentile interval resamples
    pitches with replacement and recomputes the pooled share (a draw without any such answer is left out);
    `p_value` is the two-sided bootstrap p-value of the difference (BOOT_P_NOTE).
    Answers whose support was not rated are in neither group (`n_not_rated`). Descriptive: the shares do
    not say that support caused the reaction.
    """
    if followups.empty:
        return pd.DataFrame(columns=H4_SHARE_COLUMNS)
    if (followups.next_move == AFTER_CUT).any():
        raise ValueError("h4_support_shares describes the whole encounter: call answer_followups without a cut")
    rows = followups.assign(move=panel_next_move(followups), support=followups.evid.map(support_level))
    names = sorted(rows[group].unique())
    index = rows[group].map({g: i for i, g in enumerate(names)}).to_numpy()
    counts = _bootstrap_counts(len(names), n_boot, seed)

    def per_pitch(keep: np.ndarray) -> np.ndarray:
        return np.bincount(index[keep], minlength=len(names)).astype(float)

    def share(hits: np.ndarray, total: np.ndarray) -> tuple[float, np.ndarray]:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (float(hits.sum() / total.sum()) if total.sum() else np.nan), (counts @ hits) / (counts @ total)

    out = []
    for questions, in_set in ((ALL_QUESTIONS, np.ones(len(rows), dtype=bool)),
                              (HARD_QUESTIONS, rows.hard_question.to_numpy(dtype=bool))):
        yes, no = in_set & (rows.support == "yes").to_numpy(), in_set & (rows.support == "no").to_numpy()
        n_yes, n_no = per_pitch(yes), per_pitch(no)
        for move in moves:
            is_move = (rows.move == move).to_numpy()
            hit_yes, hit_no = per_pitch(yes & is_move), per_pitch(no & is_move)
            (s_yes, d_yes), (s_no, d_no) = share(hit_yes, n_yes), share(hit_no, n_no)
            out.append({"questions": questions, "move": move, "n_yes": int(n_yes.sum()), "count_yes": int(hit_yes.sum()),
                        "share_yes": s_yes, **dict(zip(("yes_low", "yes_high"), _interval(d_yes))),
                        "n_no": int(n_no.sum()), "count_no": int(hit_no.sum()), "share_no": s_no,
                        **dict(zip(("no_low", "no_high"), _interval(d_no))), "difference": s_yes - s_no,
                        **dict(zip(("diff_low", "diff_high"), _interval(d_yes - d_no))),
                        "p_value": boot_p(d_yes - d_no),
                        "n_not_rated": int((in_set & ~yes & ~no).sum()), "n_pitches": len(names), "n_boot": int(n_boot)})
    return pd.DataFrame(out, columns=H4_SHARE_COLUMNS)


# ---- each feature on its own -----------------------------------------------------------------
SINGLE_MIN_COVERAGE = 0.30
SINGLE_PENALTY = 1.0
NOTE_COUNT = "count behind a mean"
NOTE_DECISION_ORDER = "order of deciding, not a concept"
SINGLE_COLUMNS = ["feature", "label", "family", "status", "note", "n", "coverage", "correlation", "corr_low",
                  "corr_high", "p_value", "p_boot", "q_value", "significant_fdr", "log_loss_gain", "gain_low",
                  "gain_high", "gain_p_value", "coefficient", "adjusted_correlation", "adj_low", "adj_high",
                  "adj_p_value", "adjusted_for", "n_boot", "n_perm"]
SINGLE_DRAWS = 9999     # draws behind the single-feature and by-offer p-values: fine enough for FDR over ~100 tests


def _weighted_corr(m: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The correlation of a and b under each row of weights m (draws x rows); NaN where a or b has no spread."""
    total = m.sum(axis=1)
    sa, sb = m @ a, m @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = m @ (a * b) - sa * sb / total
        va, vb = m @ (a * a) - sa * sa / total, m @ (b * b) - sb * sb / total
        return np.where((va > 1e-12) & (vb > 1e-12), cov / np.sqrt(va * vb), np.nan)


def _partial(r_xy, r_xq, r_yq):
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = np.sqrt((1 - r_xq ** 2) * (1 - r_yq ** 2))
        return np.where(denominator > 1e-9, (r_xy - r_xq * r_yq) / denominator, np.nan)


def single_features(rows: pd.DataFrame, outcome: str, group: str = GROUP, n_boot: int = N_BOOT, seed: int = SEED,
                    min_coverage: float = SINGLE_MIN_COVERAGE, adjust_for: str | None = None) -> pd.DataFrame:
    """Each numeric feature on its own against the outcome: how strong the association is and which way it points.

    One row per numeric feature column of the dictionary that `rows` carries (categorical controls and
    the 0/1 `_missing` indicators are not ranked), on the rows with a known outcome. A feature is
    skipped when it is observed on less than `min_coverage` of those rows or does not vary. Per feature:

    - `n`, `coverage`: rows with a value, and their share;
    - `correlation` with the outcome over the rows with a value (Pearson; with a 0/1 outcome this is the
      point-biserial correlation), with a 95% percentile interval from resampling whole pitches;
    - `log_loss_gain`: a one-feature ridge logistic head (penalty 1, intercept unpenalised), fitted
      leaving one pitch out, missing values replaced by the training median and the feature standardised
      on the training pitches. The gain is the log loss of that fold's base rate minus the log loss of
      the head, averaged with equal weight per pitch: positive when the feature predicts better than
      the offer rate alone. Its interval is a paired bootstrap over pitches of the fixed predictions;
    - `coefficient`: the head's weight fitted on all rows, per one standard deviation of the feature
      (log-odds; the sign is the direction);
    - `adjusted_correlation`: the partial correlation with the outcome once `adjust_for` (by default
      the number of questions asked before the cut: QUESTION_COUNT) has been taken out of both, with its interval;
    - `p_value`: the two-sided pitch-level permutation p-value of the correlation (FDR_NOTE; `n_boot`
      permutations, the scheme of `permutation_orders`). It is preferred to the bootstrap p (`p_boot`,
      kept beside it) because a feature that is non-zero in a handful of pitches has a bootstrap
      distribution that rarely crosses zero whatever its size; `q_value` is the Benjamini-Hochberg
      adjustment of `p_value` over all ranked features of the table and `significant_fdr` says
      q < FDR_LEVEL. Give `n_boot` at least SINGLE_DRAWS: with about a hundred tests the smallest p must
      be able to fall below 0.05 / 100;
    - `gain_p_value`, `adj_p_value`: two-sided bootstrap p-values of the gain and of the adjusted
      correlation, from the same draws as their intervals (BOOT_P_NOTE), not adjusted.

    `note` marks a count behind a mean (NOTE_COUNT) and the decision-history features
    (NOTE_DECISION_ORDER). Sorted by the size of the correlation. This is a descriptive ranking of many
    features, not a test: no interval here is corrected for the number of features looked at.
    """
    known = rows[rows[outcome].notna()].reset_index(drop=True)
    if known[group].nunique() < 2 or known[outcome].nunique() < 2:      # nothing to rank against
        return pd.DataFrame(columns=SINGLE_COLUMNS)
    y = known[outcome].to_numpy(dtype=float)
    groups = known[group]
    names = sorted(groups.unique())
    group_index = groups.map({g: i for i, g in enumerate(names)}).to_numpy()
    weights = pitch_weights(groups)
    counts = _bootstrap_counts(len(names), n_boot, seed)
    multiplicity = counts[:, group_index]                              # draws x rows: how often each row is drawn
    permuted = permuted_outcomes(groups.to_numpy(), y, n_boot, seed)   # the outcome under pitch-level permutations
    adjust_for = QUESTION_COUNT if adjust_for is None else adjust_for
    q = pd.to_numeric(known[adjust_for], errors="coerce").to_numpy(dtype=float) if adjust_for in known else None
    masks = [group_index == i for i in range(len(names))]

    def loss(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, CLIP, 1 - CLIP)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    base = np.empty(len(y))
    for held in masks:
        base[held] = y[~held].mean()
    out = []
    for name, spec in FEATURES.items():
        if spec["family"] is None or spec["kind"] in ("control", "indicator") or name not in known.columns:
            continue
        x = pd.to_numeric(known[name], errors="coerce").to_numpy(dtype=float)
        seen = np.isfinite(x)
        if seen.mean() < min_coverage or seen.sum() < 3 or np.nanstd(x) <= 1e-12:
            continue
        xs, ys, ms = x[seen], y[seen], multiplicity[:, seen]
        one = np.ones((1, int(seen.sum())))
        r = float(_weighted_corr(one, xs, ys)[0])
        r_draws = _weighted_corr(ms, xs, ys)
        adjusted, adjusted_draws = np.nan, np.array([])
        if q is not None and name != adjust_for and np.isfinite(q[seen]).all():
            qs = q[seen]
            adjusted = float(_partial(r, _weighted_corr(one, xs, qs)[0], _weighted_corr(one, ys, qs)[0]))
            adjusted_draws = _partial(r_draws, _weighted_corr(ms, xs, qs), _weighted_corr(ms, ys, qs))
        predicted = base.copy()
        for held in masks:
            train = x[~held]
            observed = train[np.isfinite(train)]
            if not len(observed) or len(np.unique(y[~held])) < 2:
                continue
            median = float(np.median(observed))
            filled = np.where(np.isfinite(train), train, median)
            mean, sd = float(filled.mean()), float(filled.std())
            if sd <= 1e-12:
                continue
            beta = ridge_logistic(((filled - mean) / sd)[:, None], y[~held], SINGLE_PENALTY)
            test = (np.where(np.isfinite(x[held]), x[held], median) - mean) / sd
            predicted[held] = _probabilities(test[:, None], beta)
        gain = (loss(base) - loss(predicted)) * weights
        gain_draws = (multiplicity @ gain) / (multiplicity @ weights)
        filled = np.where(seen, x, np.median(xs))
        coefficient = np.nan if len(np.unique(y)) < 2 else float(
            ridge_logistic(((filled - filled.mean()) / filled.std())[:, None], y, SINGLE_PENALTY)[1])
        note = (NOTE_COUNT if count_behind_a_mean(name) else
                NOTE_DECISION_ORDER if spec["family"] == "decision_history" else "")
        out.append({"feature": name, "label": spec["label"], "family": spec["family"], "status": feature_status(name),
                    "note": note, "n": int(seen.sum()), "coverage": float(seen.mean()), "correlation": r,
                    **dict(zip(("corr_low", "corr_high"), _interval(r_draws))),
                    "p_value": permutation_p_corr(xs, ys, permuted[:, seen]), "p_boot": boot_p(r_draws),
                    "log_loss_gain": float(gain.sum() / weights.sum()),
                    **dict(zip(("gain_low", "gain_high"), _interval(gain_draws))),
                    "gain_p_value": boot_p(gain_draws), "coefficient": coefficient,
                    "adjusted_correlation": adjusted,
                    **dict(zip(("adj_low", "adj_high"), _interval(adjusted_draws))),
                    "adj_p_value": boot_p(adjusted_draws),
                    "adjusted_for": adjust_for if adjusted == adjusted else "", "n_boot": int(n_boot),
                    "n_perm": int(n_boot)})
    table = _with_fdr(pd.DataFrame(out, columns=[c for c in SINGLE_COLUMNS if c not in ("q_value", "significant_fdr")]),
                      SINGLE_COLUMNS)
    order = table.correlation.abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].reset_index(drop=True)


FAMILY_SUMMARY_COLUMNS = ["family", "features", "p_below_05", "survive_fdr", "median_abs_correlation",
                          "largest_feature", "largest_label", "largest_correlation", "largest_q_value"]


def family_summary(table: pd.DataFrame, counts: bool = False) -> pd.DataFrame:
    """A single-feature table summed up per feature family: how many features were ranked, how many have an
    unadjusted bootstrap p below 0.05, how many survive the false-discovery adjustment (q < FDR_LEVEL, the
    q-values of the whole table), the median size of the correlation, and the feature with the largest.
    The counts behind a mean are left out unless `counts`: they repeat the number of questions asked."""
    kept = table if counts else table[table.note != NOTE_COUNT]
    out = []
    for family in [f for f in FAMILIES if f in set(kept.family)]:
        part = kept[kept.family == family]
        top = part.loc[part.correlation.abs().idxmax()]
        out.append({"family": family, "features": int(len(part)), "p_below_05": int((part.p_value < 0.05).sum()),
                    "survive_fdr": int(part.significant_fdr.astype(bool).sum()),
                    "median_abs_correlation": float(part.correlation.abs().median()), "largest_feature": top.feature,
                    "largest_label": top.label, "largest_correlation": float(top.correlation),
                    "largest_q_value": float(top.q_value)})
    return pd.DataFrame(out, columns=FAMILY_SUMMARY_COLUMNS)


# ---- permutation test: does a model beat the offer rate? ---------------------------------------

PERMUTATION_COLUMNS = ["model", "families", "penalty", "n_perm", "log_loss", "log_loss_inner_cv", "perm_log_loss_mean",
                       "p_log_loss", "auc", "perm_auc_mean", "p_auc", "n_rows", "n_groups", "seed"]
PERMUTATION_NOTE = (
    "permutation test of 'this model predicts held-out pitches no better than the offer rate'. Null: no association "
    "between a pitch's pre-decision features and its sharks' outcomes, keeping the dependence of the outcomes within "
    "a pitch. In each permutation every pitch's whole set of known outcomes is handed to another pitch with the same "
    "number of sharks with a known outcome (on pitch rows that is a plain shuffle of the pitch outcomes), in random "
    "order within the pitch, since sharks of different pitches do not correspond; the feature rows stay where they "
    "are. The full leave-one-pitch-out fit is then repeated (preprocessing fitted on the training pitches of each "
    "fold), with the ridge penalty of each model fixed at the value its inner cross-validation chose most often in "
    "the real run, so the penalty search is not repeated; the observed statistic is computed in the same way "
    "(log_loss, auc), and the real run's inner-CV log loss is given beside it (log_loss_inner_cv). "
    "p = (1 + number of permutations at least as good as observed) / (1 + n_perm), for log loss (lower is better) "
    "and for AUC (higher is better), pitches weighted equally")


def modal_penalty(result: dict, model: str) -> float | None:
    """The ridge penalty `model` used most often over the folds of an evaluation (ties: the larger); None
    when it never fitted (a base-rate model)."""
    used = [fold["models"][model].get("penalty") for fold in result.get("folds", []) if model in fold["models"]]
    used = [float(u) for u in used if u is not None]
    if not used:
        return None
    counts = pd.Series(used).value_counts()
    return float(max(counts.index[counts == counts.max()]))


def permutation_orders(groups, n_perm: int, seed: int = SEED) -> np.ndarray:
    """Row orders for the permutation test: `n_perm` x rows integer indices, so that y[order] is the outcome
    column of one permutation. Whole pitches trade their outcome sets with pitches of the same number of
    rows, and the order within a set is random (see PERMUTATION_NOTE)."""
    groups = np.asarray(groups, dtype=object)
    rows_of = {g: np.flatnonzero(groups == g) for g in sorted(set(groups))}
    strata: dict[int, list] = {}
    for g, index in rows_of.items():
        strata.setdefault(len(index), []).append(g)
    rng = np.random.default_rng(seed)
    n_perm = int(n_perm)
    orders = np.empty((n_perm, len(groups)), dtype=int)
    for size, members in strata.items():
        rows = np.array([rows_of[g] for g in members])                                # pitches x rows of a pitch
        source = np.argsort(rng.random((n_perm, len(members))), axis=1)               # which pitch each one draws from
        inner = np.argsort(rng.random((n_perm, len(members), size)), axis=2)          # the order within the set
        orders[:, rows.ravel()] = np.take_along_axis(rows[source], inner, axis=2).reshape(n_perm, -1)
    return orders


def _ridge_logistic_many(x: np.ndarray, targets: np.ndarray, penalty: float) -> np.ndarray:
    """Ridge logistic coefficients (intercept first, unpenalised) for many outcome vectors on one matrix:
    `targets` is fits x rows, the result fits x (1 + columns). Same objective as `ridge_logistic`; solved
    by the fixed-curvature iteration (the Hessian is bounded by X'X / 4 + penalty, factorised once), which
    lets all fits advance together. A target with one class only gets no fit (NaN)."""
    z = np.column_stack([np.ones(len(x)), x])
    regular = np.full(z.shape[1], float(penalty))
    regular[0] = 0.0
    inverse = np.linalg.inv(z.T @ z / 4.0 + np.diag(regular) + np.eye(z.shape[1]) * 1e-9)
    mean = targets.mean(axis=1)
    fitted = (mean > 0) & (mean < 1)
    beta = np.zeros((len(targets), z.shape[1]))
    beta[fitted, 0] = np.log(mean[fitted] / (1 - mean[fitted]))
    live = np.flatnonzero(fitted)
    for _ in range(2000):
        if not len(live):
            break
        prob = 1 / (1 + np.exp(-np.clip(beta[live] @ z.T, -35, 35)))
        step = ((prob - targets[live]) @ z + beta[live] * regular) @ inverse
        beta[live] -= step
        live = live[np.abs(step).max(axis=1) >= 1e-8]
    beta[~fitted] = np.nan
    return beta


def _fold_designs(known: pd.DataFrame, models: dict, group: str, min_category: int,
                  not_fitted: tuple[str, ...] = ()) -> list[dict]:
    """Per held-out pitch: the training and test matrices of every model. Preprocessing is fitted on the
    training pitches once for all columns any model uses (each column is prepared on its own, so a
    model's matrix is a selection of those columns), exactly as `fit_head` does per model."""
    union = tuple(dict.fromkeys(name for spec in models.values() for name in spec))
    folds = []
    for held_out in sorted(known[group].unique()):
        test_mask = (known[group] == held_out).to_numpy()
        train, test = known[~test_mask], known[test_mask]
        design = fit_design(train, union, min_category, group, not_fitted)
        x_train, x_test = design.transform(train), design.transform(test)
        source = [name.split("=", 1)[0] if name not in train.columns else name for name in design.names]
        columns = {}
        for name, spec in models.items():
            wanted = {c for kind in family_columns(train, tuple(spec), not_fitted).values() for c in kind}
            columns[name] = [i for i, column in enumerate(source) if column in wanted]
        folds.append({"test": test_mask, "x_train": x_train, "x_test": x_test, "columns": columns})
    return folds


def permutation_test(rows: pd.DataFrame, outcome: str, result: dict, n_perm: int = 500, seed: int = SEED,
                     group: str = GROUP, min_category: int = 5) -> pd.DataFrame:
    """Does each model of an evaluation beat the offer rate? A pitch-level permutation test (PERMUTATION_NOTE).

    `result` is `evaluate(rows, outcome, ...)`: its models, the columns it did not fit (`not_fitted`), and the
    penalty each model chose most often, are reused. One row per model with features: the observed log loss and AUC under the fixed penalty, the
    real run's inner-CV log loss, the mean over permutations, and the two p-values. Every model sees the
    same permutations. Empty when nothing was estimated or `n_perm` is 0.
    """
    if result.get("status") != "estimated" or int(n_perm) <= 0:
        return pd.DataFrame(columns=PERMUTATION_COLUMNS)
    known = rows[rows[outcome].notna()].reset_index(drop=True)
    y = known[outcome].to_numpy(dtype=float)
    groups = known[group].to_numpy()
    w = pitch_weights(known[group])
    penalties = {name: modal_penalty(result, name) for name in result["metrics"].model}
    models = {name: tuple(spec) for name, spec in zip(result["metrics"].model, result["metrics"].families.str.split("+"))
              if penalties[name] is not None}
    if "models" in result:                                     # the question-control evaluations carry their columns
        models = {name: tuple(result["models"][name]) for name in models}
    if not models:
        return pd.DataFrame(columns=PERMUTATION_COLUMNS)
    targets = np.vstack([y, y[permutation_orders(groups, n_perm, seed)]])          # row 0 is the observed outcome
    folds = _fold_designs(known, models, group, min_category, tuple(result.get("not_fitted", ())))
    real = result["metrics"].set_index("model")
    adjusted = result.get("auc_score", RAW) == BASE_RATE_ADJUSTED     # rank what the evaluation ranked
    out = []
    for name, spec in models.items():
        prob, base_all = np.empty_like(targets), np.empty_like(targets)
        for fold in folds:
            train_y = targets[:, ~fold["test"]]
            base = train_y.mean(axis=1, keepdims=True)
            base_all[:, fold["test"]] = base
            columns = fold["columns"][name]
            beta = _ridge_logistic_many(fold["x_train"][:, columns], train_y, penalties[name])
            z = np.column_stack([np.ones(int(fold["test"].sum())), fold["x_test"][:, columns]])
            fitted = 1 / (1 + np.exp(-np.clip(np.nan_to_num(beta) @ z.T, -35, 35)))
            prob[:, fold["test"]] = np.where(np.isnan(beta[:, :1]), base, fitted)      # one class in training: base rate
        clipped = np.clip(prob, CLIP, 1 - CLIP)
        loss = -((targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)) * w).sum(axis=1) / w.sum()
        ranked = adjusted_score(prob, base_all) if adjusted else prob
        auc = np.array([weighted_auc(t, q, w) for t, q in zip(targets, ranked)])
        out.append({"model": name, "families": real.families[name], "penalty": penalties[name], "n_perm": int(n_perm),
                    "log_loss": float(loss[0]), "log_loss_inner_cv": float(real.log_loss[name]),
                    "perm_log_loss_mean": float(loss[1:].mean()),
                    "p_log_loss": float((1 + (loss[1:] <= loss[0] + 1e-12).sum()) / (1 + int(n_perm))),
                    "auc": float(auc[0]), "perm_auc_mean": float(np.nanmean(auc[1:])),
                    "p_auc": float((1 + (auc[1:] >= auc[0] - 1e-12).sum()) / (1 + int(n_perm))),
                    "n_rows": int(len(known)), "n_groups": int(known[group].nunique()), "seed": int(seed)})
    return pd.DataFrame(out, columns=PERMUTATION_COLUMNS)


def add_permutation_p(result: dict, table: pd.DataFrame) -> dict:
    """Write the permutation p-values of `table` into the metrics of an evaluation (`perm_p_log_loss`,
    `perm_p_auc`, `n_perm`; missing for a model without features or when no permutation was run)."""
    if result.get("status") != "estimated":
        return result
    by_model = table.set_index("model") if len(table) else table
    metrics = result["metrics"]
    for column, source in (("perm_p_log_loss", "p_log_loss"), ("perm_p_auc", "p_auc"), ("n_perm", "n_perm")):
        metrics[column] = [by_model[source].get(m, np.nan) if len(table) else np.nan for m in metrics.model]
    return result


# ---- the full model's coefficients -------------------------------------------------------------

COEFFICIENT_COLUMNS = ["model", "feature", "label", "family", "coefficient", "ci_low", "ci_high", "p_value", "penalty",
                       "n_boot", "n_rows", "n_groups"]
COEFFICIENT_NOTE = ("standardised coefficients (log-odds per 1 SD of a numeric feature, per level of a category) of "
                    "the full exploratory model refitted on all rows with a known outcome, the ridge penalty fixed at "
                    "the value the leave-one-pitch-out run chose most often. Intervals and p-values resample whole "
                    "pitches and refit the model with the preprocessing held fixed. These are conditional associations "
                    "inside one penalised model: each depends on which other features are in it, the penalty shrinks "
                    "all of them towards zero, and correlated features share their weight. Not independent effects")


def _weighted_ridge(z: np.ndarray, y: np.ndarray, weights: np.ndarray, regular: np.ndarray, start: np.ndarray) -> np.ndarray:
    """Newton iterations of the ridge logistic objective with row weights, from `start` (intercept in column 0)."""
    beta = start.copy()

    def loss(b):
        logits = z @ b
        return weights @ (np.logaddexp(0, logits) - y * logits) + 0.5 * np.sum(regular * b * b)
    for _ in range(50):
        prob = 1 / (1 + np.exp(-np.clip(z @ beta, -35, 35)))
        grad = z.T @ (weights * (prob - y)) + regular * beta
        hessian = (z.T * (weights * prob * (1 - prob))) @ z + np.diag(regular) + np.eye(len(beta)) * 1e-9
        step = np.linalg.solve(hessian, grad)
        rate, old = 1.0, loss(beta)
        while rate > 1e-6 and loss(beta - rate * step) > old:
            rate *= 0.5
        beta = beta - rate * step
        if np.max(np.abs(rate * step)) < 1e-7:
            break
    return beta


def model_coefficients(rows: pd.DataFrame, outcome: str, spec: tuple[str, ...], penalty: float, model: str = "",
                       group: str = GROUP, min_category: int = 5, n_boot: int = N_BOOT, seed: int = SEED,
                       not_fitted: tuple[str, ...] = ()) -> pd.DataFrame:
    """The coefficients of one model refitted on all rows with a known outcome (COEFFICIENT_NOTE).

    One row per column of the model's design (a numeric feature, a missing-value indicator, or a level of
    a categorical control), largest coefficient first: the coefficient, a 95% percentile interval and a
    two-sided bootstrap p-value from `n_boot` resamples of whole pitches (each a weighted refit, the
    imputation and scaling of the full sample held fixed). Empty when the outcome does not vary. `not_fitted`
    columns are left out of the design (the ladder's C5 passes LADDER_NOT_FITTED).
    """
    known = rows[rows[outcome].notna()].reset_index(drop=True)
    y = known[outcome].to_numpy(dtype=float) if len(known) else np.array([])
    if known[group].nunique() < 2 or len(np.unique(y)) < 2 or not spec or penalty is None:
        return pd.DataFrame(columns=COEFFICIENT_COLUMNS)
    design = fit_design(known, tuple(spec), min_category, group, not_fitted)
    x = design.transform(known)
    beta = ridge_logistic(x, y, penalty)
    z = np.column_stack([np.ones(len(x)), x])
    regular = np.full(z.shape[1], float(penalty))
    regular[0] = 0.0
    names = sorted(known[group].unique())
    index = known[group].map({g: i for i, g in enumerate(names)}).to_numpy()
    counts = _bootstrap_counts(len(names), n_boot, seed)[:, index]
    draws = np.array([_weighted_ridge(z, y, m, regular, beta) for m in counts if len(np.unique(y[m > 0])) > 1])
    out = []
    for i, name in enumerate(design.names, start=1):
        raw = name if name in FEATURES else name.split("=", 1)[0]
        feature = FEATURES.get(raw, {})
        label = feature.get("label", raw) + (f" = {name.split('=', 1)[1]}" if name not in FEATURES and "=" in name else "")
        column = draws[:, i] if len(draws) else np.array([])
        out.append({"model": model, "feature": name, "label": label, "family": feature.get("family"),
                    "coefficient": float(beta[i]), **dict(zip(("ci_low", "ci_high"), _interval(column))),
                    "p_value": boot_p(column), "penalty": float(penalty), "n_boot": int(len(draws)),
                    "n_rows": int(len(known)), "n_groups": len(names)})
    table = pd.DataFrame(out, columns=COEFFICIENT_COLUMNS)
    return table.reindex(table.coefficient.abs().sort_values(ascending=False, kind="stable").index).reset_index(drop=True)


# ---- one held-out prediction, taken apart and edited ----------------------------------------

SENSITIVITY_NOTE = ("model sensitivity, not a causal effect: this is how the fitted head's number moves when one input "
                    "is changed and everything else is held fixed; it says nothing about what would have happened "
                    "in the room")


def fold_detail(result: dict, model: str, video_id: str) -> dict:
    """What `model` fitted in the fold that held `video_id` out."""
    if result.get("status") != "estimated":
        raise ValueError(f"no fitted fold to inspect: {result.get('status')}")
    for fold in result["folds"]:
        if fold["held_out"] == video_id:
            if model not in fold["models"]:
                raise ValueError(f"unknown model {model!r}. Models of this evaluation: {', '.join(fold['models'])}.")
            return fold["models"][model]
    raise ValueError(f"pitch {video_id!r} was not held out in this evaluation (its outcome may be unknown). "
                     f"Held-out pitches: {', '.join(f['held_out'] for f in result['folds'])}.")


def fold_design(detail: dict) -> Design:
    """The preprocessing of a fold rebuilt from its stored schema, so it can be applied to other rows."""
    design = Design(names=list(detail["columns"]), dropped=list(detail.get("dropped", [])))
    numeric = detail.get("medians", {})
    for name in detail["columns"]:
        if name in numeric:
            design.numeric.append((name, numeric[name], detail["means"][name], detail["sds"][name]))
        elif "=" in name:
            control, level = name.split("=", 1)
            design.levels.setdefault(control, []).append(level)
        else:
            design.indicators.append(name)
    design.pooled = {k: list(v) for k, v in detail.get("pooled_levels", {}).items()}
    return design


def _fold_probabilities(detail: dict, frame: pd.DataFrame) -> np.ndarray:
    if not detail["coefficients"]:           # the fold predicted its training base rate
        return np.full(len(frame), float(detail["intercept"]))
    beta = np.array([detail["intercept"], *(detail["coefficients"][name] for name in detail["columns"])])
    return _probabilities(fold_design(detail).transform(frame), beta)


def _one_row(rows: pd.DataFrame, video_id: str, investor: str | None, group: str = GROUP) -> pd.DataFrame:
    keep = rows[group] == video_id
    if investor is not None:
        if "investor" not in rows.columns:
            raise ValueError("this table has one row per pitch; leave the investor out")
        keep &= rows["investor"] == investor
    found = rows[keep]
    if len(found) != 1:
        what = f"pitch {video_id!r}" + (f", investor {investor!r}" if investor is not None else "")
        raise ValueError(f"{what}: expected exactly one row, found {len(found)}"
                         + (" (name the investor)" if investor is None and len(found) > 1 else ""))
    return found


def editable_features(result: dict, model: str, video_id: str) -> list[str]:
    """The feature columns that move `model`'s prediction for the held-out pitch: those its fold kept."""
    design = fold_design(fold_detail(result, model, video_id))
    return [name for name, *_ in design.numeric] + list(design.indicators) + list(design.levels)


def prediction_breakdown(result: dict, rows: pd.DataFrame, model: str, video_id: str,
                         investor: str | None = None) -> pd.DataFrame:
    """The held-out prediction of one row, taken apart: one line per column the fold's head used.

    `value` is the row's own value (`filled` when it was missing and the training median stood in),
    `standardised` what the head saw, `coefficient` the fold's weight, and `contribution` their
    product, on the log-odds scale: positive pushes the prediction towards an offer. The intercept
    is the first line. Sorted by the size of the contribution. Empty when the fold predicted its
    training base rate.
    """
    detail = fold_detail(result, model, video_id)
    row = _one_row(rows, video_id, investor)
    columns = ["feature", "label", "family", "value", "filled", "standardised", "coefficient", "contribution"]
    if not detail["coefficients"]:
        return pd.DataFrame(columns=columns)
    design = fold_design(detail)
    x = design.transform(row)[0]
    median = {name: m for name, m, _, _ in design.numeric}
    out = []
    for name, seen in zip(design.names, x):
        raw_name = name.split("=", 1)[0]
        spec = FEATURES.get(raw_name, {})
        raw = row[raw_name].iloc[0]
        missing = raw is None or raw != raw
        coefficient = detail["coefficients"][name]
        out.append({"feature": name, "label": spec.get("label", raw_name), "family": spec.get("family"),
                    "value": raw, "filled": median.get(name) if missing else None, "standardised": float(seen),
                    "coefficient": coefficient, "contribution": float(seen) * coefficient})
    table = pd.DataFrame(out, columns=columns)
    table = table.reindex(table.contribution.abs().sort_values(ascending=False, kind="stable").index)
    first = pd.DataFrame([{"feature": "(intercept)", "label": "starting point: the training pitches' offer rate",
                           "family": None, "value": None, "filled": None, "standardised": 1.0,
                           "coefficient": detail["intercept"], "contribution": detail["intercept"]}], columns=columns)
    return pd.concat([first.astype(object), table.astype(object)], ignore_index=True)


def edited_prediction(result: dict, rows: pd.DataFrame, model: str, video_id: str, investor: str | None = None,
                      edits: dict | None = None) -> dict:
    """The held-out prediction of one row, and what it becomes when feature values are replaced.

    The fold that held the row's pitch out is reused as stored (its imputation medians, scaling,
    category levels, penalty and coefficients): nothing is refit. `edits` maps a feature column to
    a new value. A column that is not a feature of the model is an error listing the model's
    features; one the fold dropped (constant in its training pitches) changes nothing and the
    `note` says so. See SENSITIVITY_NOTE for what the change means.
    """
    edits = dict(edits or {})
    detail = fold_detail(result, model, video_id)
    row = _one_row(rows, video_id, investor)
    usable = editable_features(result, model, video_id)
    dropped = sorted({name.split("=", 1)[0] for name in detail.get("dropped", [])} - set(usable))
    unknown = [name for name in edits if name not in usable and name not in dropped]
    if unknown:
        raise ValueError(f"{', '.join(repr(u) for u in unknown)}: not a feature of model {model!r} for this pitch. "
                         f"Features it uses: {', '.join(usable) or '(none: it predicts the base rate)'}."
                         + (f" Dropped in this fold (constant in training): {', '.join(dropped)}." if dropped else ""))
    edited = row.copy()
    for name, value in edits.items():
        edited[name] = [value]
    before = float(_fold_probabilities(detail, row)[0])
    after = float(_fold_probabilities(detail, edited)[0])
    notes = [SENSITIVITY_NOTE]
    if not detail["coefficients"]:
        notes.insert(0, "this fold predicted the training base rate, so no edit can change it")
    notes[:0] = [f"{name} was dropped in this fold (constant in its training pitches), so editing it changes nothing"
                 for name in edits if name in dropped]
    return {"model": model, "video_id": video_id, "investor": investor, "edits": edits,
            "original_values": {name: row[name].iloc[0] for name in edits},
            "original": before, "edited": after, "change": after - before, "note": "; ".join(notes)}
