"""H9, the answer-level test: does a more prepared answer change what the panel does next?

The unit is an answer and an investor question it responds to, before the pitch's cut (`answer_table`,
built on `analysis.panel_dataset.answer_followups` and its panel next move). The outcomes (refocus spec,
section 6.7):

- whether the panel's next turn accepts, challenges or asks a further question, and whether the next
  investor question keeps the promotion/prevention frame of the question answered (BINARY_OUTCOMES):
  conditional logit stratified by pitch (`conditional_logit`, statsmodels' ConditionalLogit), so that only
  within-pitch differences between answers count;
- the next question's severity and the number of further investor questions, in turn order, that keep
  that frame (CONTINUOUS_OUTCOMES): linear models with a dummy per pitch (`fixed_effects`).

Every outcome is read on turn order; none reads the model's exchange grouping (plan B2, 2026-10-21). The two
same-frame outcomes have a base rate set by the frame answered (most questions are promotion-framed), so
for them every nested model also enters `q_promotion` (OUTCOME_COVARIATES), the answered question's frame as
0/1: a covariate, not a predictor of interest, so `h9_summary` states no direction for it. A question
whose frame is neither promotion nor prevention has no `q_promotion` and so leaves those two models. The
exchange-based outcomes of the earlier plan (SENSITIVITY_OUTCOMES: the next question in the answer's
exchange, further questions in it) are a sensitivity only (`h9_sensitivity`), computed on request and never
reported as a result.

The predictors enter in nested models (MODELS), each fitted on the answers rated on all of its own
predictors, so each estimates for its own population (POPULATIONS) rather than one model losing every
answer that misses any rating:

- `support`: checkable support and admitted uncertainty (`evid`, `uncert`), on all answers;
- `substance`: support, uncertainty and substance (`subs`), on the answers with a substance rating;
- `full`: support, uncertainty, substance, coherence and context, on the answers rated on all five;
- `responsiveness`: admitted uncertainty and a founder `repair` turn before the panel's next turn, on
  all answers: the behavioural form of Pollack et al.'s negative responsiveness result. Repair enters
  this model only (plan v4).

Intervals come from resampling pitches with replacement and refitting (N_BOOT draws, SEED). `h9` runs
every model and outcome for all questions and for hard ones (severity 2 or 3). With `workers` > 1 the
draws of each model (all made up front from the seed) are split into contiguous blocks fitted in a
process pool and put back in order, so the result is the same as with one worker. There is no decision
rule: `h9_summary` claims a direction only where the intervals for all questions and for hard questions
both exclude zero on the same side, and never for a predictor that separates the outcome.
"""
from __future__ import annotations

import multiprocessing
import warnings
from concurrent.futures import Executor, ProcessPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.base.model import LikelihoodModel
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.sm_exceptions import PerfectSeparationError

from analysis.panel_bottleneck import N_BOOT, SEED
from analysis.panel_dataset import (
    ALL_QUESTIONS, AFTER_CUT, HARD_QUESTIONS, PANEL_CHALLENGE, PANEL_QUESTION, PANEL_RECEIPT, PanelData,
    _next_investor_turn, answer_followups, panel_next_move,
)
from pipeline.reliability.agreement import MIN_DEFINED_SHARE

SUPPORT_MODEL, SUBSTANCE_MODEL, FULL_MODEL, RESPONSIVENESS_MODEL = "support", "substance", "full", "responsiveness"
MODELS = {
    SUPPORT_MODEL: ("evid", "uncert"),
    SUBSTANCE_MODEL: ("evid", "uncert", "subs"),
    FULL_MODEL: ("evid", "uncert", "subs", "coh", "ctx"),
    RESPONSIVENESS_MODEL: ("uncert", "repair_before_next"),
}
POPULATIONS = {
    SUPPORT_MODEL: "all answers (rated on checkable support and admitted uncertainty)",
    SUBSTANCE_MODEL: "answers with a substance rating",
    FULL_MODEL: "answers rated on support, uncertainty, substance, coherence and context",
    RESPONSIVENESS_MODEL: "all answers (rated on admitted uncertainty)",
}
FRAME_COVARIATE = "q_promotion"      # the answered question's frame: 1 promotion, 0 prevention, NaN otherwise
BINARY_PREDICTORS = ("evid", "uncert", "repair_before_next", FRAME_COVARIATE)   # entered as 0/1, never standardised
BINARY_OUTCOMES = ("next_accept", "next_challenge", "next_further_question", "next_question_same_frame")
CONTINUOUS_OUTCOMES = ("next_question_sev", "further_questions_same_frame")
OUTCOMES = (*BINARY_OUTCOMES, *CONTINUOUS_OUTCOMES)
# the exchange-based outcomes, binary first and continuous second: a sensitivity, never reported (h9_sensitivity)
SENSITIVITY_OUTCOMES = ("next_question_same_exchange", "further_questions_in_exchange")
_BINARY = frozenset((*BINARY_OUTCOMES, SENSITIVITY_OUTCOMES[0]))     # fitted by conditional logit; the rest by OLS
# what every nested model also enters for an outcome, beside its own predictors
OUTCOME_COVARIATES = {"next_question_same_frame": (FRAME_COVARIATE,),
                      "further_questions_same_frame": (FRAME_COVARIATE,)}
COVARIATES = frozenset(c for names in OUTCOME_COVARIATES.values() for c in names)
FRAMES = {"promotion": 1.0, "prevention": 0.0}
REPAIR_FUNCTION = "repair"
CLOGIT, FIXED_EFFECTS = "conditional_logit", "fixed_effects"
MIN_CONVERGED_SHARE = MIN_DEFINED_SHARE      # an interval is given when at least this share of the draws estimate it
MAX_ABS_LOGIT = 5.0          # a bootstrap log-odds beyond this (per unit or per SD) is a separated draw, not an estimate
QUANTILES = (2.5, 97.5)
ANSWER_TABLE_COLUMNS = (
    "video_id", "answer_turn", "question_turn", "hard_question", "resp", "evid", "uncert", "subs", "coh", "ctx",
    "repair_before_next", FRAME_COVARIATE, "next_accept", "next_challenge", "next_further_question",
    "next_question_sev", "next_question_same_frame", "further_questions_same_frame", "next_question_same_concept",
    "next_question_same_exchange", "further_questions_in_exchange")
MODEL_COLUMNS = ("outcome", "predictor", "coef", "low", "high", "n_answers", "n_pitches_informative",
                 "n_pitches_dropped", "n_failed", "separated", "main_failed", "estimable")
H9_COLUMNS = ("model", "population", "method", *MODEL_COLUMNS)
SUMMARY_COLUMNS = ("model", "outcome", "predictor", "direction", "hard_and_all_agree")
POSITIVE, NEGATIVE, NOT_DISTINGUISHABLE = "positive", "negative", "not distinguishable"
# why a model's intervals are (not) given: resampling one pitch only redraws that pitch (a zero-width interval), and
# a design whose rank is below its column count does not identify its coefficients
ESTIMABLE, TOO_FEW_PITCHES, RANK_DEFICIENT = "yes", "too few informative pitches", "rank-deficient design"
MIN_INFORMATIVE_PITCHES = 2
_WITHIN_TOL = 1e-12
_FIT_ERRORS = (np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError, ZeroDivisionError,
               PerfectSeparationError)


# ---- the answer table ----------------------------------------------------------------------------------

def _repair_before_next(turns: pd.DataFrame, answer_pos: int, next_pos: float, cut_pos: int) -> bool:
    """A founder repair turn after the answer and before `next_pos` (the panel's next turn; NaN: the
    next investor turn of any kind anywhere in the pitch), never at or after the cut."""
    after = turns[turns.pos > answer_pos]
    if next_pos == next_pos:
        bound = next_pos
    else:
        investor = after[after.role == "investor"]
        bound = investor.pos.min() if len(investor) else np.inf
    bound = min(bound, cut_pos)
    between = after[after.pos < bound]
    return bool(((between.role == "founder") & (between.function == REPAIR_FUNCTION)).any())


def answer_table(data: PanelData, cut: pd.DataFrame) -> pd.DataFrame:
    """One row per answer and investor question it responds to, both before the cut of their pitch.

    An answer that responds to several investor questions appears once per question, so it weighs as
    many rows as the questions it answers. `cut` is the pitch cut table (`analysis.panel_dataset.pitch_cut`:
    video_id, cut_pos); every asker of a pitch gets that pitch's cut. Columns ANSWER_TABLE_COLUMNS:

    - the answer's ratings `resp`, `evid`, `uncert`, `subs`, `coh`, `ctx` and `hard_question`, as in
      `answer_followups`;
    - `repair_before_next` (bool): a founder `repair` turn between the answer and the panel's next turn;
    - `next_accept`, `next_challenge`, `next_further_question`: 1.0 / 0.0 from the panel's next move
      (`panel_next_move`: the first investor reaction by anyone, in turn order; an answer with no
      reaction, or whose next investor turn is decision talk, is 0.0 on all three). NaN when the next
      investor turn lies at or after the cut, whatever it is: a decision stated at the cut is the cut;
    - `q_promotion`: the answered question's frame, 1.0 promotion, 0.0 prevention, NaN when unclassified
      (deal_terms, other) or NEUTRAL (no concept);
    - `next_question_sev`: the severity of the panel's next turn when it is a question before the cut,
      else NaN;
    - `next_question_same_frame` (1.0 / 0.0 / NaN), `further_questions_same_frame` and
      `next_question_same_concept` (1.0 / 0.0 / NaN), in turn order, and the exchange-based sensitivity
      columns `next_question_same_exchange` (1.0 / 0.0 / NaN) and `further_questions_in_exchange`: all from
      `answer_followups` with the pitch cut, so only questions before the cut are read.
    """
    cut_pos = dict(zip(cut.video_id, cut.cut_pos.astype(int)))
    investors = data.actors[data.actors.role == "investor"]
    per_asker = {v: {a: cut_pos[v] for a in investors[investors.video_id == v].actor_id} for v in cut_pos}
    followups = answer_followups(data, cut=per_asker, video_ids=list(cut_pos))
    if len(followups):
        before = [cut_pos[v] for v in followups.video_id]
        followups = followups[(followups.answer_pos < before) & (followups.question_pos < before)]
    severity = dict(zip(zip(data.turns.video_id, data.turns.turn_id), data.turns["PR-SEV"]))
    turns = {v: g for v, g in data.turns.groupby("video_id")}
    ordered = {v: g.sort_values("pos").to_dict("records") for v, g in turns.items()}
    # the next investor turn, re-found: a decision at the cut must leave the row not knowing its move
    # (answer_followups already tests the cut first, so this re-check is a guard, not a correction)
    found = [_next_investor_turn(ordered[v], ordered[v][int(p)])
             for v, p in zip(followups.video_id, followups.answer_pos)]
    found_pos = pd.Series([np.nan if f is None else float(f["pos"]) for f in found], index=followups.index)
    move = panel_next_move(followups)
    cut_off = (move == AFTER_CUT) | (found_pos >= pd.Series([cut_pos[v] for v in followups.video_id],
                                                             index=followups.index, dtype=float))
    rows = pd.DataFrame({
        "video_id": followups.video_id.astype(object), "answer_turn": followups.answer_turn.astype(int),
        "question_turn": followups.question_turn.astype(int), "hard_question": followups.hard_question.astype(bool),
        **{name: followups[name].astype(float) for name in ("resp", "evid", "uncert", "subs", "coh", "ctx")}})
    rows["repair_before_next"] = pd.Series(
        [_repair_before_next(turns[r.video_id], r.answer_pos, p, cut_pos[r.video_id])
         for r, p in zip(followups.itertuples(), found_pos)], index=followups.index, dtype=bool)
    for name, kind in (("next_accept", PANEL_RECEIPT), ("next_challenge", PANEL_CHALLENGE),
                       ("next_further_question", PANEL_QUESTION)):
        rows[name] = (move == kind).astype(float).where(~cut_off)
    rows["next_question_sev"] = pd.Series(
        [severity.get((v, int(t)), np.nan) if m == PANEL_QUESTION else np.nan
         for v, t, m in zip(followups.video_id, followups.next_turn, move)],
        index=followups.index, dtype=float).where(~cut_off)
    frame = dict(zip(zip(data.turns.video_id, data.turns.turn_id), data.turns.frame))
    rows[FRAME_COVARIATE] = pd.Series([FRAMES.get(frame.get((v, int(t))), np.nan)
                                       for v, t in zip(followups.video_id, followups.question_turn)],
                                      index=followups.index, dtype=float)
    for name in ("next_question_same_frame", "next_question_same_concept", "next_question_same_exchange"):
        rows[name] = followups[name].astype(float)
    for name in ("further_questions_same_frame", "further_questions_in_exchange"):
        rows[name] = followups[name].astype(np.int64)
    return rows[list(ANSWER_TABLE_COLUMNS)].reset_index(drop=True)


# ---- the models ----------------------------------------------------------------------------------------

def _within_variation(x: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per column of `x`, the sum of squares left after removing each group's mean."""
    if not len(x):
        return np.zeros(x.shape[1])
    frame = pd.DataFrame(x)
    demeaned = frame - frame.groupby(groups).transform("mean")
    return (demeaned ** 2).sum().to_numpy()


def _fit_clogit(y: np.ndarray, x: np.ndarray, groups: np.ndarray, start=None) -> np.ndarray | None:
    """The coefficients of a conditional logit of `y` on `x` stratified by `groups`; None when the
    optimiser does not converge, the fit fails numerically, or a coefficient is not finite."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ConditionalLogit(y, x, groups=groups)
            # LikelihoodModel.fit, which ConditionalLogit.fit wraps, keeps the optimiser's convergence flag
            result = LikelihoodModel.fit(model, start_params=start, method="bfgs", maxiter=200, disp=False,
                                         skip_hessian=True)
    except _FIT_ERRORS:
        return None
    params = np.asarray(result.params, dtype=float)
    if not result.mle_retvals.get("converged", False) or not np.all(np.isfinite(params)):
        return None
    return params


def _fit_fixed_effects(y: np.ndarray, x: np.ndarray, groups: np.ndarray, start=None) -> np.ndarray | None:
    """The coefficients of `x` in an OLS of `y` on `x` and a dummy per group; None when the fit fails
    numerically or a coefficient is not finite."""
    try:
        dummies = pd.get_dummies(pd.Series(groups), dtype=float).to_numpy()
        result = sm.OLS(y, np.column_stack([x, dummies])).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None
    params = np.asarray(result.params, dtype=float)[: x.shape[1]]
    return params if np.all(np.isfinite(params)) else None


def _draw(groups: np.ndarray, pitches: np.ndarray, draw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The rows of a bootstrap sample of pitches (`draw` indexes `pitches`) and their strata: the draw's
    slot, so a pitch drawn twice is two strata and is never merged into one."""
    parts = [np.flatnonzero(groups == pitches[d]) for d in draw]
    rows = np.concatenate(parts) if parts else np.array([], dtype=int)
    strata = np.repeat(np.arange(len(draw)), [len(p) for p in parts])
    return rows, strata


def _draw_matrix(n_pitches: int, n_boot: int, seed: int) -> np.ndarray:
    """The bootstrap draws of a model, all made before any fit: row b indexes the pitches of draw b. Fixed
    by the seed alone, so the draws can be split across workers without changing the result."""
    return np.random.default_rng(seed).integers(0, n_pitches, size=(n_boot, n_pitches))


def _bootstrap(y: np.ndarray, x: np.ndarray, groups: np.ndarray, pitches: np.ndarray, matrix: np.ndarray,
               method: str) -> np.ndarray:
    """One row of coefficients per draw of `matrix`, NaN where the draw does not estimate a predictor.

    Each draw is fitted from zero (a cold start). A predictor without within-stratum variation in the
    draw is left out of that draw's fit and is NaN there; a draw whose fit fails, or whose design is
    rank-deficient (`design_rank_ok`), is NaN throughout. In a
    conditional logit a draw that did not move from its start is NaN throughout, and a coefficient
    beyond MAX_ABS_LOGIT is NaN: that draw separates the outcome on that predictor.
    """
    fit = globals()["_fit_clogit" if method == CLOGIT else "_fit_fixed_effects"]   # at call time: a test may stand in
    out = np.full((len(matrix), x.shape[1]), np.nan)
    for b, draw in enumerate(matrix):
        sample, strata = _draw(groups, pitches, draw)
        xb = x[sample]
        varies = _within_variation(xb, strata) > _WITHIN_TOL
        if not varies.any() or not design_rank_ok(xb[:, varies], strata, method):
            continue                    # a rank-deficient draw identifies nothing: a failed draw, not an estimate
        params = fit(y[sample], xb[:, varies], strata, None)
        if params is None:
            continue
        if method == CLOGIT:
            if np.allclose(params, 0.0, rtol=0.0, atol=1e-12):
                continue
            params = np.where(np.abs(params) > MAX_ABS_LOGIT, np.nan, params)
        out[b, np.flatnonzero(varies)] = params
    return out


def _bootstrap_split(y: np.ndarray, x: np.ndarray, groups: np.ndarray, pitches: np.ndarray, matrix: np.ndarray,
                     method: str, executor: Executor | None = None, workers: int = 1) -> np.ndarray:
    """`_bootstrap` of `matrix`, its rows split into `workers` contiguous blocks fitted on `executor` and stacked
    back in order: each draw is fitted from its own row of the pre-drawn matrix and from nothing else, so the
    result equals `_bootstrap` of the whole matrix in one process. Without an executor (or one worker), that call."""
    workers = max(1, int(workers)) if executor is not None else 1
    if workers <= 1 or len(matrix) < 2:
        return _bootstrap(y, x, groups, pitches, matrix, method)
    blocks = [block for block in np.array_split(matrix, min(workers, len(matrix))) if len(block)]
    futures = [executor.submit(_bootstrap, y, x, groups, pitches, block, method) for block in blocks]
    return np.vstack([future.result() for future in futures])


def worker_pool(workers: int) -> Executor | None:
    """A process pool of `workers` processes for the bootstrap draws (spawned, never forked), or None for one."""
    workers = max(1, int(workers))
    if workers == 1:
        return None
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _interval(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, high, n_failed) per column of `draws`: the 2.5th and 97.5th percentiles of the draws that
    estimate the predictor, NaN when fewer than MIN_CONVERGED_SHARE of all draws do."""
    n_boot, k = draws.shape
    ok = np.isfinite(draws)
    low, high = np.full(k, np.nan), np.full(k, np.nan)
    for j in range(k):
        if n_boot and ok[:, j].sum() >= MIN_CONVERGED_SHARE * n_boot:
            low[j], high[j] = np.percentile(draws[ok[:, j], j], QUANTILES)
    return low, high, n_boot - ok.sum(axis=0)


def _separated(x: np.ndarray, y: np.ndarray, predictors: tuple) -> np.ndarray:
    """Per predictor: a binary predictor for which one of the four outcome × predictor cells is empty."""
    flags = np.zeros(len(predictors), dtype=bool)
    for j, name in enumerate(predictors):
        if name in BINARY_PREDICTORS:
            flags[j] = min(int(((x[:, j] == a) & (y == b)).sum()) for a in (0.0, 1.0) for b in (0.0, 1.0)) == 0
    return flags


def design_rank_ok(x: np.ndarray, groups: np.ndarray, method: str) -> bool:
    """Whether the fitted design has full column rank: for FIXED_EFFECTS `x` beside a dummy per group (the OLS
    design itself), for CLOGIT `x` demeaned within groups (what the conditional likelihood can identify)."""
    if not len(x):
        return False
    if method == FIXED_EFFECTS:
        design = np.column_stack([x, pd.get_dummies(pd.Series(groups), dtype=float).to_numpy()])
    else:
        frame = pd.DataFrame(x)
        design = (frame - frame.groupby(groups).transform("mean")).to_numpy()
    return int(np.linalg.matrix_rank(design)) == design.shape[1]


def _model(table: pd.DataFrame, outcome: str, predictors: tuple, n_boot: int, seed: int, method: str,
           executor: Executor | None = None, workers: int = 1) -> pd.DataFrame:
    """Fit `method` (CLOGIT or FIXED_EFFECTS) on the informative strata and bootstrap it over pitches.

    The intervals need at least MIN_INFORMATIVE_PITCHES informative pitches (resampling one pitch redraws it every
    time, a zero-width interval) and a full-rank design (`design_rank_ok`); `estimable` says which held. With too
    few pitches the coefficient of the fit is kept and the interval is NaN; with a rank-deficient design both are
    NaN, since the coefficients are not identified. A conditional logit whose fit on the data gives a coefficient
    beyond MAX_ABS_LOGIT in absolute value (the bound the bootstrap draws obey) takes the separated path, as a
    binary predictor with an empty cell does (plan B Task 9, pre-run): that predictor is reported `separated`, with
    no coefficient or interval, and the model is refitted without it, then bootstrapped; this repeats until no
    coefficient of the fit is beyond the bound. A model without a usable predictor counts as too few pitches when
    it has fewer than two, and as estimable otherwise (its rows are NaN for their own reasons)."""
    predictors = tuple(predictors)
    k = len(predictors)
    n_present = table.video_id.nunique()
    rows = table[["video_id", outcome, *predictors]].dropna()
    size = rows.groupby("video_id")[outcome].transform("size")
    keep = size >= 2                                         # a single answer says nothing within its pitch
    if method == CLOGIT:
        keep &= rows.groupby("video_id")[outcome].transform("nunique") > 1   # nor does an outcome that never varies
    rows = rows[keep.to_numpy()]
    groups = rows.video_id.to_numpy()
    n_informative = int(pd.Series(groups).nunique())
    x = np.array(rows[list(predictors)], dtype=float, copy=True).reshape(len(rows), k)
    y = np.array(rows[outcome], dtype=float, copy=True)
    for j, name in enumerate(predictors):                    # coefficients per SD, binary ones per 0 -> 1
        if name not in BINARY_PREDICTORS and len(x):
            sd = x[:, j].std()
            x[:, j] = (x[:, j] - x[:, j].mean()) / sd if sd > 0 else 0.0
    varies = _within_variation(x, groups) > _WITHIN_TOL
    separated = varies & _separated(x, y, predictors) if method == CLOGIT else np.zeros(k, dtype=bool)
    usable = varies & ~separated
    coef, low, high = np.full(k, np.nan), np.full(k, np.nan), np.full(k, np.nan)
    n_failed = np.zeros(k, dtype=int)
    main_failed = False
    estimable = ESTIMABLE if n_informative >= MIN_INFORMATIVE_PITCHES else TOO_FEW_PITCHES
    fit = globals()["_fit_clogit" if method == CLOGIT else "_fit_fixed_effects"]
    while n_informative and usable.any():
        cols = np.flatnonzero(usable)
        if not design_rank_ok(x[:, cols], groups, method):
            estimable = RANK_DEFICIENT
            break
        main = fit(y, x[:, cols], groups, None)
        if main is None:
            main_failed = True
            break
        beyond = np.abs(main) > MAX_ABS_LOGIT if method == CLOGIT else np.zeros(len(cols), dtype=bool)
        if beyond.any():
            # separated in all but name: the separated path (no coefficient, refit without it)
            separated[cols[beyond]] = True
            usable[cols[beyond]] = False
            continue
        coef[cols] = main
        if estimable == ESTIMABLE:
            pitches = np.array(sorted(set(groups)), dtype=object)
            draws = _bootstrap_split(y, x[:, cols], groups, pitches, _draw_matrix(len(pitches), n_boot, seed),
                                     method, executor, workers)
            low[cols], high[cols], n_failed[cols] = _interval(draws)
        break
    return pd.DataFrame({
        "outcome": [outcome] * k, "predictor": list(predictors), "coef": coef, "low": low, "high": high,
        "n_answers": len(rows), "n_pitches_informative": n_informative,
        "n_pitches_dropped": n_present - n_informative, "n_failed": n_failed,
        "separated": separated, "main_failed": main_failed, "estimable": estimable}, columns=list(MODEL_COLUMNS))


def conditional_logit(table: pd.DataFrame, outcome: str, predictors: tuple, n_boot: int = N_BOOT,
                      seed: int = SEED, executor: Executor | None = None, workers: int = 1) -> pd.DataFrame:
    """A conditional logit of the binary `outcome` (1.0 / 0.0) on `predictors`, stratified by pitch.

    statsmodels' ConditionalLogit with `groups=video_id`: each pitch is a stratum, so the coefficients
    compare answers within a pitch and nothing that is constant within it can reach them. Rows missing
    the outcome or a predictor are left out of this model only. Strata with a single answer or an
    outcome that never varies carry no information and are dropped before the fit: `n_answers` and
    `n_pitches_informative` count what is left, `n_pitches_dropped` the pitches of `table` that are not
    informative (including those lost entirely to missing values).

    Coefficients are log-odds per one standard deviation of the predictor among the rows fitted; the
    binary predictors (BINARY_PREDICTORS: evid, uncert, repair_before_next, q_promotion) stay 0/1, so theirs are per
    0 -> 1. A predictor without variation within the strata has no coefficient (NaN). A binary predictor
    with an empty outcome × predictor cell in the informative strata separates the outcome: it is
    reported with `separated` true and no coefficient or interval, and the model is fitted without it.
    `main_failed` is true when the fit on the data itself fails (every coefficient is then NaN).

    `low`, `high`: per predictor, the 2.5th and 97.5th percentiles over `n_boot` bootstrap samples of the
    informative pitches, drawn with replacement (all draws made up front from `seed`) and refitted from
    zero, a pitch drawn twice entering as two strata. A draw does not estimate a predictor when the fit
    fails or does not move from zero, when the predictor lacks within-stratum variation in the draw, or
    when its coefficient exceeds MAX_ABS_LOGIT in absolute value; `n_failed` counts those draws per
    predictor, and with fewer than MIN_CONVERGED_SHARE of the draws estimating it the interval is NaN.
    The fit on the data itself obeys the same bound: a predictor whose coefficient is beyond it is reported
    `separated` and the model is refitted without it before the bootstrap (see `_model`).
    """
    return _model(table, outcome, predictors, n_boot, seed, CLOGIT, executor, workers)


def fixed_effects(table: pd.DataFrame, outcome: str, predictors: tuple, n_boot: int = N_BOOT,
                  seed: int = SEED, executor: Executor | None = None, workers: int = 1) -> pd.DataFrame:
    """A linear model of the continuous `outcome` on `predictors` with a dummy per pitch (statsmodels OLS).

    The dummies absorb everything constant within a pitch, so the coefficients compare answers within a
    pitch. Rows missing the outcome or a predictor are left out of this model only; pitches with a single
    answer carry no information and are dropped (a pitch whose outcome never varies still informs a
    linear model and is kept). Coefficients are outcome units per one standard deviation of the
    predictor among the rows fitted, per 0 -> 1 for the binary predictors. `separated` is always false.
    Intervals, `n_failed`, `main_failed` and the counts as in `conditional_logit`, without the
    MAX_ABS_LOGIT bound.
    """
    return _model(table, outcome, predictors, n_boot, seed, FIXED_EFFECTS, executor, workers)


# ---- H9 ------------------------------------------------------------------------------------------------

def _all_outcomes(table: pd.DataFrame, model: str, n_boot: int, seed: int,
                  executor: Executor | None = None, workers: int = 1,
                  outcomes: tuple[str, ...] = OUTCOMES) -> list[pd.DataFrame]:
    """Every outcome of `outcomes` under one nested model, in order: a conditional logit for a binary outcome
    (membership of BINARY_OUTCOMES or the binary sensitivity outcome), pitch fixed effects otherwise. The model's
    predictors come first, then the outcome's OUTCOME_COVARIATES."""
    frames = []
    for outcome in outcomes:
        predictors = MODELS[model] + OUTCOME_COVARIATES.get(outcome, ())
        if outcome in _BINARY:
            frames.append(conditional_logit(table, outcome, predictors, n_boot, seed, executor, workers)
                          .assign(method=CLOGIT))
        else:
            frames.append(fixed_effects(table, outcome, predictors, n_boot, seed, executor, workers)
                          .assign(method=FIXED_EFFECTS))
    return [f.assign(model=model, population=POPULATIONS[model]) for f in frames]


def _run(data: PanelData, cut: pd.DataFrame, outcomes: tuple[str, ...], n_boot: int, seed: int,
         workers: int) -> dict[str, pd.DataFrame]:
    table = answer_table(data, cut)
    out = {}
    executor = worker_pool(workers)
    try:
        for name, part in ((ALL_QUESTIONS, table), (HARD_QUESTIONS, table[table.hard_question])):
            frames = [f for model in MODELS
                      for f in _all_outcomes(part, model, n_boot, seed, executor, workers, outcomes)]
            out[name] = pd.concat(frames, ignore_index=True)[list(H9_COLUMNS)]
    finally:
        if executor is not None:
            executor.shutdown()
    return out


def h9(data: PanelData, cut: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED,
       workers: int = 1) -> dict[str, pd.DataFrame]:
    """H9 for all questions (`all`) and for hard ones (`hard`, severity 2 or 3).

    Each frame (H9_COLUMNS) stacks, for every model of MODELS in turn (`model`, with its `population`),
    the conditional logits of BINARY_OUTCOMES and the pitch fixed-effect models of CONTINUOUS_OUTCOMES
    (`method`). For the two same-frame outcomes each model also enters `q_promotion` (OUTCOME_COVARIATES),
    whose row follows the model's predictors. Coefficients are per one standard deviation, per 0 -> 1 for
    binary predictors (see `conditional_logit`); `n_answers` says how many answers each model rests on.
    Reported with intervals and no decision rule; see `h9_summary`.

    `workers` > 1 fits each model's bootstrap draws in a process pool of that many processes (`worker_pool`,
    `_bootstrap_split`); the numbers are those of one worker.
    """
    return _run(data, cut, OUTCOMES, n_boot, seed, workers)


def h9_sensitivity(data: PanelData, cut: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED,
                   workers: int = 1) -> dict[str, pd.DataFrame]:
    """The exchange-based outcomes (SENSITIVITY_OUTCOMES) under the same nested models, for all and hard
    questions, in the frame shape of `h9`: the conditional logit of `next_question_same_exchange` and the
    pitch fixed-effect model of `further_questions_in_exchange`, with no covariate. A sensitivity of the
    exchange grouping, which is unstable between model passes: computed on request, never a result and never
    summarised into a direction."""
    return _run(data, cut, SENSITIVITY_OUTCOMES, n_boot, seed, workers)


def _side(low, high) -> str | None:
    if low > 0:
        return POSITIVE
    if high < 0:
        return NEGATIVE
    return None                  # the interval holds zero, or there is none (NaN compares false)


def h9_summary(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per model, outcome and predictor, the direction H9 may claim.

    `hard_and_all_agree` is true when the intervals for all questions and for hard questions both
    exclude zero on the same side and the predictor separates the outcome in neither; `direction` is
    then "positive" or "negative", and "not distinguishable" otherwise (either interval holds zero or is
    missing, they exclude it on opposite sides, the predictor is separated, or either model is not
    `estimable`: too few informative pitches or a rank-deficient design). Rows follow those of
    `result["all"]`, without the covariates (COVARIATES: `q_promotion`), which are not predictors of interest
    and get no direction.
    """
    keys = ["model", "outcome", "predictor"]
    result = {name: frame[~frame.predictor.isin(COVARIATES)] for name, frame in result.items()}

    def part(frame: pd.DataFrame) -> pd.DataFrame:
        sep = frame["separated"] if "separated" in frame else pd.Series(False, index=frame.index)
        fine = frame["estimable"].eq(ESTIMABLE) if "estimable" in frame else pd.Series(True, index=frame.index)
        # a predictor H9 must not read: separated, or in a model that is not estimable
        return frame[keys + ["low", "high"]].assign(separated=sep.astype(bool) | ~fine.astype(bool))
    both = part(result[ALL_QUESTIONS]).merge(part(result[HARD_QUESTIONS]), on=keys, how="left",
                                             suffixes=("_all", "_hard"))
    sides_all = [_side(lo, hi) for lo, hi in zip(both.low_all, both.high_all)]
    sides_hard = [_side(lo, hi) for lo, hi in zip(both.low_hard, both.high_hard)]
    separated = (both.separated_all.astype(bool) | both.separated_hard.fillna(False).astype(bool)).tolist()
    agree = [a is not None and a == h and not s for a, h, s in zip(sides_all, sides_hard, separated)]
    return pd.DataFrame({
        "model": both.model, "outcome": both.outcome, "predictor": both.predictor,
        "direction": [a if ok else NOT_DISTINGUISHABLE for a, ok in zip(sides_all, agree)],
        "hard_and_all_agree": pd.Series(agree, dtype=bool)}, columns=list(SUMMARY_COLUMNS))
