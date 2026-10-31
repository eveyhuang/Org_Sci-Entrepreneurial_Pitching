"""The shark-level model sets (the retired O2 design: one row per pitch and investor) and the functions that
read them, moved here from `analysis.panel_bottleneck` when the pitch-level ladder replaced them (plan B task 3).

Kept importable and runnable, with the numbers they gave before the move: the exploratory B-family B0-B5d, the
draft primary set P0-P5d, the reaction-trajectory ladders, questions asked as a control (Q, and QC on pitch
rows), answer quality beyond it, the removal ranking and the pooled-versus-attributed comparison. The pitch-level
sets they were read beside are frozen here too (`ARCHIVED_C_FAMILY`, the C0-C5 of v3-dev8 before the ladder).

The machinery itself (folds, preprocessing, the ridge head, bootstrap, permutations) stays in
`analysis.panel_bottleneck`, which reads only the pitch-level dictionary. A shark row carries columns that
dictionary does not declare, so every function here runs that machinery inside `shark_level_rows()`, which
lets it read the pitch and the shark-level dictionaries together (`COMBINED_FEATURES`), keep the investor as a
row key and leave `investor_label` out of the predictors. The generic functions that read a dictionary are
re-exported here in that form (`evaluate`, `check_models`, `family_columns`, `fit_design`, ...), so code that
evaluates shark rows imports them from this module.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager

import numpy as np
import pandas as pd

from analysis import panel_bottleneck as pb
from analysis.panel_bottleneck import (
    CONDITIONAL, GROUP, LADDER, LADDER_COMPARISONS, N_BOOT, PENALTY_GRID, RANK_PREFIX, SEED, STATUS_DESCRIPTIVE,
    STATUS_EXPLORATORY, STATUS_PRIMARY, label_for, metrics_of, paired_deltas, penalty_setting,
)
from analysis.archive.shark_level_features import ARCHIVE_FEATURES, COMBINED_FAMILIES, COMBINED_FEATURES
from analysis.panel_features import FEATURES as PITCH_FEATURES, STAGE_A_CODES

# ---- the shark-level reading of the generic machinery ----------------------------------------

# the pitch-level status from the families alone, as before plan B's ruling I4 (the archive is unaffected by it)
_pitch_feature_status = functools.partial(pb.feature_status, ladder_exclusion=False)

# ---- the pre-specified model set (draft) ----------------------------------------------------
PRIMARY_STATUS = "draft — to be confirmed by the researcher before the scale run"
# The features each step of the primary set adds, by the step that adds them. Deliberately absent:
# answer directness (`*_resp_mean`), uncertainty, coherence and context, which barely vary in the pilot
# and are reported descriptively only (DESCRIPTIVE_STATS).
PRIMARY_FEATURES: dict[str, tuple[str, ...]] = {
    "control": ("founder_gender", "product_form", "primary_industry"),
    "exposure": ("hist_turns", "hist_questions_all"),
    # every item of the live Stage A book, read from the books
    "stage_a": tuple(f"{RANK_PREFIX}{code}" for code in STAGE_A_CODES),
    "own": ("own_questions", "own_subs_mean", "own_evid_share", "own_subs_hard_mean", "own_receipt_share",
            "own_satisfied", "own_unsatisfied"),
    "panel_content": ("oth_questions", "oth_subs_mean", "oth_evid_share"),
    "panel_reactions": ("oth_agrees", "oth_challenges", "oth_unsatisfied"),
    # n_social_exchanges is declared in the family `exposure`; the primary set adds it at the rapport step
    "rapport_repair": ("own_social", "n_social_exchanges"),
}
# what stands in for the rank scores of `stage_a` when a run has no ranking
PRIMARY_STAGE_A_RATINGS = tuple(f"sa_{code}" for code in STAGE_A_CODES)
PRIMARY_SENSITIVITY_FEATURES = ("oth_exits_before", "oth_offers_before")     # P5d only
PRIMARY_STEPS = (("P0", ()), ("P1", ("control", "exposure")), ("P2", ("stage_a",)), ("P3", ("own",)),
                 ("P4", ("panel_content", "panel_reactions")), ("P5", ("rapport_repair",)))
DESCRIPTIVE_STATS = ("resp_mean", "uncert_share", "coh_mean", "ctx_mean")    # reported, never in the primary set


def feature_status(name: str) -> str | None:
    """The status of a column on shark rows: for a shark-level column, `primary` for a feature of the draft primary
    set (PRIMARY_FEATURES, and the Stage A ratings that stand in for its rank scores), `descriptive` for the answer
    statistics kept out of it on purpose (directness, uncertainty, coherence, context, with their companion
    columns), `exploratory` for every other feature; a pitch column has its pitch-level status
    (`analysis.panel_bottleneck.feature_status`); None for a key column, which is never a predictor."""
    if name in PITCH_FEATURES:
        return _pitch_feature_status(name)
    if ARCHIVE_FEATURES.get(name, {}).get("family") is None:
        return None
    if name in PRIMARY_STAGE_A_RATINGS or any(name in names for names in PRIMARY_FEATURES.values()):
        return STATUS_PRIMARY
    stem = name.removesuffix("_n").removesuffix("_missing")
    if stem.startswith(("own_", "oth_")) and stem.endswith(DESCRIPTIVE_STATS):
        return STATUS_DESCRIPTIVE
    return STATUS_EXPLORATORY


# what `analysis.panel_bottleneck` reads while a shark row passes through it: both dictionaries, the investor as a
# row key, and the investor's label (identity, kept for later shrinkage only) never a predictor
SHARK_LEVEL = {"FEATURES": COMBINED_FEATURES, "FAMILIES": COMBINED_FAMILIES, "KEY_COLUMNS": ("investor",),
               "NOT_PREDICTORS": ("investor_label",), "feature_status": feature_status}


@contextmanager
def shark_level_rows():
    """While open, the generic machinery of `analysis.panel_bottleneck` reads shark rows: the names of SHARK_LEVEL
    replace its pitch-level ones, and are put back on leaving (also on an error, and when nested). Not thread-safe;
    never call pitch-level functions inside this context."""
    saved = {name: getattr(pb, name) for name in SHARK_LEVEL}
    for name, value in SHARK_LEVEL.items():
        setattr(pb, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(pb, name, value)


def _on_shark_rows(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with shark_level_rows():
            return function(*args, **kwargs)
    return wrapped


# the generic functions that read a dictionary, in their shark-level form
predictor_names = _on_shark_rows(pb.predictor_names)
check_models = _on_shark_rows(pb.check_models)
family_columns = _on_shark_rows(pb.family_columns)
fit_design = _on_shark_rows(pb.fit_design)
design_matrix = _on_shark_rows(pb.design_matrix)
fit_head = _on_shark_rows(pb.fit_head)
count_behind_a_mean = _on_shark_rows(pb.count_behind_a_mean)
family_summary = _on_shark_rows(pb.family_summary)
permutation_test = _on_shark_rows(pb.permutation_test)
model_coefficients = _on_shark_rows(pb.model_coefficients)
prediction_breakdown = _on_shark_rows(pb.prediction_breakdown)
edited_prediction = _on_shark_rows(pb.edited_prediction)


# ---- the model sets -------------------------------------------------------------------------

MODEL_FAMILY: dict[str, tuple[str, ...]] = {   # cumulative feature families per model
    "B0": (),
    "B1": ("control", "exposure"),
    "B2": ("control", "exposure", "stage_a"),
    "B3": ("control", "exposure", "stage_a", "own"),
    "B4a": ("control", "exposure", "stage_a", "own", "panel_content"),
    "B4": ("control", "exposure", "stage_a", "own", "panel_content", "panel_reactions"),
    "B5": ("control", "exposure", "stage_a", "own", "panel_content", "panel_reactions", "rapport_repair"),
    "B5d": ("control", "exposure", "stage_a", "own", "panel_content", "panel_reactions", "rapport_repair",
            "decision_history"),
}
# The pitch-level family C0-C5 as Task 2 of plan B left it (whole families; no prepared step, no C3q), frozen for
# the comparisons that were read against it: the collective arm of pooled_vs_attributed and the O1 reaction ladder.
ARCHIVED_C_FAMILY: dict[str, tuple[str, ...]] = {
    "C0": (),
    "C1": ("control", "exposure"),
    "C2": ("control", "exposure", "stage_a"),
    "C3": ("control", "exposure", "stage_a", "questions", "answers"),
    "C4": ("control", "exposure", "stage_a", "questions", "answers", "reactions"),
    "C5": ("control", "exposure", "stage_a", "questions", "answers", "reactions", "rapport_repair"),
}

# models reported beside the main family, each with what it adds and why it is read apart
_DECISION_ORDER = ("sensitivity: adds the order in which other investors decided; in this format exits "
                   "usually precede offers")
SENSITIVITY = {"B5d": _DECISION_ORDER, "P5d": _DECISION_ORDER}


def primary_models(frame: pd.DataFrame | None = None) -> dict[str, tuple[str, ...]]:
    """The primary set P0-P5 and its sensitivity P5d, each model the one before plus a few named features.

    With `frame`, the Stage A features are the rank scores when the table carries them and the
    ratings of the same items otherwise (a run without a ranking has no rank columns).
    """
    features = dict(PRIMARY_FEATURES)
    if frame is not None and not all(name in frame.columns for name in features["stage_a"]):
        features["stage_a"] = PRIMARY_STAGE_A_RATINGS
    models: dict[str, tuple[str, ...]] = {}
    so_far: tuple[str, ...] = ()
    for name, added in PRIMARY_STEPS:
        so_far = so_far + tuple(f for step in added for f in features[step])
        models[name] = so_far
    models["P5d"] = so_far + PRIMARY_SENSITIVITY_FEATURES
    return models


PRIMARY_MODELS = primary_models()

# ---- deprecated aliases of the pitch-level ladder (not shark-level) --------------------------
# The draft pitch-level short list (CP0-CP5) and its question-control variant (QCP0-QCP5) were dropped (refocus spec:
# "No short list is kept"). The report, the figures and the notebook still render a pitch-level "primary" set, so
# under its old name it is now the ladder itself, its models renamed C* -> CP*. Removed in plan B task 7/9.
O1_PRIMARY_MODELS: dict[str, tuple[str, ...]] = {f"CP{name[1:]}": spec for name, spec in LADDER.items()}
O1_PRIMARY_COMPARISONS = [(f"CP{a[1:]}", f"CP{b[1:]}") for a, b in LADDER_COMPARISONS]


def o1_primary_models(frame: pd.DataFrame | None = None) -> dict[str, tuple[str, ...]]:
    """Deprecated: O1_PRIMARY_MODELS (the ladder under the names CP*), whatever `frame`. Removed in plan B task 7/9."""
    return dict(O1_PRIMARY_MODELS)


def _comparisons(names: list[str]) -> list[tuple[str, str]]:
    pairs = list(zip(names[:-1], names[1:]))
    for extra in (("B3", "B4"), ("B2", "B5"), ("P2", "P5"), ("C2", "C5"), ("CP2", "CP5")):
        if set(extra) <= set(names) and extra not in pairs:
            pairs.append(extra)
    return pairs


def evaluate(rows: pd.DataFrame, outcome: str, group: str = GROUP, penalty: float | None = None,
             models: dict | None = None, min_category: int = 5, n_boot: int = N_BOOT, seed: int = SEED,
             penalty_grid: tuple[float, ...] = PENALTY_GRID,
             comparisons: list[tuple[str, str]] | None = None) -> dict:
    """`analysis.panel_bottleneck.evaluate` of shark rows (or of pitch rows next to them), as it was before plan B:
    MODEL_FAMILY by default; by default the consecutive pairs plus (B3, B4), (B2, B5), (P2, P5), (C2, C5) and
    (CP2, CP5) when both models are there; a model named in SENSITIVITY marked so in `metrics` (`role`, `note`);
    the deltas in log loss, Brier score and AUC (the skill rows the pitch ladder adds are left out)."""
    with shark_level_rows():
        checked = pb.check_models(MODEL_FAMILY if models is None else models, rows)
        pairs = _comparisons(list(checked)) if comparisons is None else comparisons
        result = pb.evaluate(rows, outcome, group, penalty, checked, min_category, n_boot, seed, penalty_grid, pairs,
                             auc_score=pb.RAW)      # the archived path ranks the probabilities, as before
    metrics, deltas = result["metrics"], result["deltas"]
    if len(metrics):
        metrics["role"] = ["sensitivity" if name in SENSITIVITY else "main" for name in metrics.model]
        metrics["note"] = [SENSITIVITY.get(name, "") for name in metrics.model]
    if len(deltas):
        result["deltas"] = deltas[deltas.metric != "skill"].reset_index(drop=True)
    return result


# ---- the reaction-trajectory ladders (docs/plans/design docs/2026-10-19-reaction-trajectory-test.md) ----
# Each R model is a B (or C) model plus columns of the reaction_events family, so its paired comparison
# with that B model asks one question: do every reaction to the answers, the asker's apart from the
# panel's, add to what the first next move and the closing states already carry?
RX_OWN, RX_PAN, RX_ALL = "rx_own_", "rx_pan_", "rx_"     # RX_ALL: the pitch row's reaction shares (family reactions)
RX_FAMILIES = ("reaction_events", "reactions")
_RX_SHAPE_WORDS = ("_last_", "_trend_")     # the course of the reactions: left out by the counts-only variant


def reaction_columns(frame: pd.DataFrame, prefixes: tuple[str, ...], counts_only: bool = False) -> tuple[str, ...]:
    """Every reaction column of `frame` starting with one of `prefixes`, in dictionary order: on shark rows the
    `rx_own_*` and `rx_pan_*` columns of family reaction_events, on pitch rows the `rx_*` columns of family reactions
    (since plan B task 2 the pitch row's reaction shares, read on turn order since plan B2; `rx_answers` among them,
    which the live ladder no longer fits). The `_n` and `_missing` twins are listed by name, so the model reads them
    (family_columns reads a twin only when it is named). `counts_only` leaves out the columns on the course of the
    reactions (last answer, trend), which only shark rows have."""
    return tuple(name for name, spec in COMBINED_FEATURES.items()
                 if spec["family"] in RX_FAMILIES and name in frame.columns and name.startswith(prefixes)
                 and not (counts_only and any(word in name for word in _RX_SHAPE_WORDS)))


def _r(name: str, counts_only: bool) -> str:
    return f"{name}c" if counts_only else name


def reaction_models(frame: pd.DataFrame, counts_only: bool = False) -> dict[str, tuple[str, ...]]:
    """The O2 ladder: B1, B3, R1 (B3 + own reactions), B4a, R2 (B4a + own and panel reactions), B4, R3
    (B4 + own and panel reactions; the primary comparison), B5, R4 (B5 + every reaction column). With
    `counts_only` the R models are named R1c..R4c and carry no last-answer or trend column."""
    own = reaction_columns(frame, (RX_OWN,), counts_only)
    both = reaction_columns(frame, (RX_OWN, RX_PAN), counts_only)
    every = reaction_columns(frame, (RX_OWN, RX_PAN, RX_ALL), counts_only)
    return {"B1": MODEL_FAMILY["B1"], "B3": MODEL_FAMILY["B3"], _r("R1", counts_only): MODEL_FAMILY["B3"] + own,
            "B4a": MODEL_FAMILY["B4a"], _r("R2", counts_only): MODEL_FAMILY["B4a"] + both,
            "B4": MODEL_FAMILY["B4"], _r("R3", counts_only): MODEL_FAMILY["B4"] + both,
            "B5": MODEL_FAMILY["B5"], _r("R4", counts_only): MODEL_FAMILY["B5"] + every}


def reaction_comparisons(counts_only: bool = False) -> list[tuple[str, str]]:
    return [("B3", _r("R1", counts_only)), ("B4a", _r("R2", counts_only)), ("B4", _r("R3", counts_only)),
            ("B5", _r("R4", counts_only))]


def o1_reaction_models(frame: pd.DataFrame, counts_only: bool = False) -> dict[str, tuple[str, ...]]:
    """The O1 ladder on ARCHIVED_C_FAMILY: C1, C3, R-C3 (C3 + every investor's reactions), C4, R-C4 (C4 + the
    same). Since the pitch rows were rewritten (plan B task 2) C4 already holds the reaction shares, so R-C4 adds
    nothing to it."""
    every = reaction_columns(frame, (RX_ALL,), counts_only)
    c = ARCHIVED_C_FAMILY
    return {"C1": c["C1"], "C3": c["C3"], _r("R-C3", counts_only): c["C3"] + every,
            "C4": c["C4"], _r("R-C4", counts_only): c["C4"] + every}


def o1_reaction_comparisons(counts_only: bool = False) -> list[tuple[str, str]]:
    return [("C3", _r("R-C3", counts_only)), ("C4", _r("R-C4", counts_only))]


# ---- the removal ranking ----------------------------------------------------------------------
NEVER_REMOVED = ("control", "exposure")
REMOVAL_CAVEAT = ("a small delta is not evidence of irrelevance: families overlap, a removed family can be "
                  "stood in for by the others, and the sample is small")


def removal_ranking(rows: pd.DataFrame, outcome: str, base_model: str = "B5", **kw) -> pd.DataFrame:
    """The base model refit with each concept family removed in turn, on the same folds.

    `delta_vs_full` is the log loss without the family minus the log loss of the full model: positive
    when the family carried something the others did not. Controls and exposure are never removed;
    every other family of the base model is, `decision_history` included when the base model has it
    (base_model="B5d"). Sorted by that delta, largest first. A small delta is not evidence of
    irrelevance.
    """
    full = tuple(MODEL_FAMILY[base_model])
    removable = [f for f in full if f not in NEVER_REMOVED]
    models = {"full": full, **{f"without_{f}": tuple(x for x in full if x != f) for f in removable}}
    result = evaluate(rows, outcome, models=models, **kw)
    columns = ["family", "log_loss_full", "log_loss_without", "delta_vs_full", "ci_low", "ci_high", "p_value", "n_boot",
               "base_model", "uncertainty", "caveat"]
    if result["status"] != "estimated":
        return pd.DataFrame(columns=columns)
    pred = result["predictions"]
    metrics = result["metrics"].set_index("model")
    deltas = paired_deltas(pred, [("full", f"without_{f}") for f in removable], kw.get("group", GROUP),
                           kw.get("n_boot", N_BOOT), kw.get("seed", SEED))
    deltas = deltas[deltas.metric == "log_loss"].set_index("comparison")
    out = []
    for family in removable:
        d = deltas.loc[f"without_{family} minus full"]
        out.append({"family": family, "log_loss_full": metrics.loc["full", "log_loss"],
                    "log_loss_without": metrics.loc[f"without_{family}", "log_loss"],
                    "delta_vs_full": d["delta"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                    "p_value": d["p_value"], "n_boot": d["n_boot"], "base_model": base_model, "uncertainty": CONDITIONAL,
                    "caveat": REMOVAL_CAVEAT})
    return (pd.DataFrame(out, columns=columns).sort_values("delta_vs_full", ascending=False, kind="stable")
            .reset_index(drop=True))


# ---- questions asked as a control ------------------------------------------------------------
# How many questions were put before the cut is the most consistent single signal (see single_features).
# These families move it into the control step and ask what the later steps add beyond it. On pitch rows the
# ladder's C3q (questions without answers) and C3 now ask this; the QC family is kept beside the investor rows.
QUESTIONS_ASKED = {"o2": "own_questions", "o1": pb.QUESTION_COUNT}
# the same count under another name on pitch rows (every investor question before the cut)
QUESTIONS_ASKED_ALIASES = {"o2": (), "o1": ()}
Q_STEPS = {"o2": (("stage_a",), ("own",), ("panel_content",), ("panel_reactions",), ("rapport_repair",)),
           "o1": (("stage_a",), ("questions", "answers"), ("reactions",), ("rapport_repair",))}
Q_PRIMARY_STEPS = {"o2": (("stage_a",), ("own",), ("panel_content", "panel_reactions"), ("rapport_repair",))}
Q_PREFIX = {"o2": "Q", "o1": "QC"}
Q_NOTE = {
    "o2": "questions asked = own_questions, the questions this investor put before deciding",
    "o1": ("questions asked = hist_questions, every investor question before the first decision; it is already "
           "one of the exposure controls of C1 and CP1, so the first model here "
           "equals C1 (CP1) and the later steps differ from the C family only in leaving the counts out"),
}


def _unit(frame: pd.DataFrame) -> str:
    return "o2" if "investor" in frame.columns else "o1"


def _step_columns(frame: pd.DataFrame, families: tuple[str, ...], unit: str) -> tuple[str, ...]:
    """The columns of `families` in `frame`, without the question count, its aliases and every count behind a mean."""
    skip = {QUESTIONS_ASKED[unit], *QUESTIONS_ASKED_ALIASES[unit]}
    return tuple(name for columns in family_columns(frame, families).values() for name in columns
                 if name not in skip and not count_behind_a_mean(name))


def question_control_models(frame: pd.DataFrame, primary: bool = False) -> dict[str, tuple[str, ...]]:
    """An ordered family in which the number of questions asked is a control.

    Investor rows (O2): Q0 the base rate; Q1 controls, exposure and `own_questions`; then Stage A, the
    answers to the investor's own questions, the other investors' questions and answers, their
    reactions, and rapport, one step each (Q2-Q6). Pitch rows (O1): QC0-QC5 with `hist_questions`,
    which the exposure controls already hold (see Q_NOTE). No later step re-enters the question
    count: the count itself, its aliases and the counts behind a mean (`*_n`, which follow it closely)
    are left out of the steps, so a step can only add what its ratings, shares and reactions carry.

    `primary=True` gives the draft named-feature variant on investor rows (QP0-QP5): the primary features of
    each step, with the question count in the control step and nowhere else. Its pitch-row variant (QCP) was
    dropped with the draft pitch-level list, so asking for it there is an error.
    """
    unit = _unit(frame)
    question = QUESTIONS_ASKED[unit]
    prefix = Q_PREFIX[unit] + ("P" if primary else "")
    if primary:
        if unit != "o2":
            raise ValueError("the draft pitch-level list (QCP0-QCP5) was dropped in plan B: there is no primary "
                             "question-control set on pitch rows; the ladder's C3q and C3 ask its question")
        named = dict(PRIMARY_FEATURES)
        if not all(name in frame.columns for name in named["stage_a"]):
            named["stage_a"] = PRIMARY_STAGE_A_RATINGS
        skip = {question, *QUESTIONS_ASKED_ALIASES[unit]}
        first = tuple(dict.fromkeys((*named["control"], *named["exposure"], question)))
        steps = [tuple(f for family in step for f in named[family] if f not in skip and f not in first)
                 for step in Q_PRIMARY_STEPS[unit]]
    else:
        first = (*(c for columns in family_columns(frame, ("control", "exposure")).values() for c in columns),)
        first = tuple(dict.fromkeys((*first, question)))
        steps = [tuple(c for c in _step_columns(frame, step, unit) if c not in first) for step in Q_STEPS[unit]]
    models: dict[str, tuple[str, ...]] = {f"{prefix}0": (), f"{prefix}1": first}
    so_far = first
    for i, added in enumerate(steps, start=2):
        so_far = so_far + added
        models[f"{prefix}{i}"] = so_far
    return models


def question_control_comparisons(models: dict) -> list[tuple[str, str]]:
    """The pairs to compare in a question-control family: every model with the one before it, and every
    later model with the controls-plus-questions model (the second of the family)."""
    names = list(models)
    pairs = list(zip(names[:-1], names[1:]))
    return pairs + [(names[1], later) for later in names[3:]]


def evaluate_question_control(rows: pd.DataFrame, outcome: str, primary: bool = False, **kw) -> dict:
    """`evaluate` of the question-control family of `rows`; the result also carries its `models`."""
    models = question_control_models(rows, primary)
    result = evaluate(rows, outcome, models=models, comparisons=question_control_comparisons(models), **kw)
    return {**result, "models": models}


# ---- answer quality beyond questions asked: own questions versus the other investors' ----------
ANSWER_QUALITY_STATS = ("subs_mean", "evid_share", "subs_hard_mean", "evid_hard_share")
ANSWER_QUALITY_BLOCKS = {"o2": (("own", "own_"), ("other", "oth_")), "o1": (("all", "a_"),)}
ANSWER_QUALITY_WORDS = {"own": "answers to this investor's own questions",
                        "other": "answers to the other investors' questions",
                        "both": "answers to this investor's own and to the other investors' questions",
                        "all": "answers to all investors' questions, pooled (pitch rows have no own and other)"}


def answer_quality_columns(frame: pd.DataFrame, prefix: str) -> tuple[str, ...]:
    """The four answer-quality statistics of one block (substance and checkable support, on all and on
    hard questions) that `frame` carries, each with its none-observed indicator; never a count."""
    return tuple(c for stat in ANSWER_QUALITY_STATS for c in (f"{prefix}{stat}", f"{prefix}{stat}_missing")
                 if c in frame.columns)


def answer_quality_models(frame: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """Does answer quality add to controls plus questions asked, and whose questions were being answered?

    The first two models are the question-control base rate and controls-plus-questions model (Q0, Q1
    on investor rows; QC0, QC1 on pitch rows), unchanged. Then, each built on that second model alone
    (not nested in one another): `<Q1>_own_answers` adds the quality of the answers to the investor's
    own questions, `<Q1>_other_answers` that of the answers to the other investors' questions, and
    `<Q1>_both_answers` both. Pitch rows have only the pooled statistics: `<QC1>_all_answers`.
    """
    unit = _unit(frame)
    base = question_control_models(frame)
    zero, control = list(base)[:2]
    models = {zero: base[zero], control: base[control]}
    blocks = [(word, answer_quality_columns(frame, prefix)) for word, prefix in ANSWER_QUALITY_BLOCKS[unit]]
    blocks = [(word, columns) for word, columns in blocks if columns]
    for word, columns in blocks:
        models[f"{control}_{word}_answers"] = base[control] + columns
    if len(blocks) > 1:
        models[f"{control}_both_answers"] = base[control] + tuple(c for _, columns in blocks for c in columns)
    return models


def answer_quality_comparisons(models: dict) -> list[tuple[str, str]]:
    """Controls-plus-questions against the base rate; every answer-quality model against
    controls-plus-questions; and the model with both blocks against each model with one."""
    names = list(models)
    control, later = names[1], names[2:]
    pairs = [(names[0], control), *((control, name) for name in later)]
    if len(later) > 1:
        pairs += [(name, later[-1]) for name in later[:-1]]
    return pairs


def evaluate_answer_quality(rows: pd.DataFrame, outcome: str, **kw) -> dict:
    """`evaluate` of `answer_quality_models(rows)`; the result also carries its `models`."""
    models = answer_quality_models(rows)
    result = evaluate(rows, outcome, models=models, comparisons=answer_quality_comparisons(models), **kw)
    return {**result, "models": models}


BY_OFFER_WHOSE = {"o2": (("the shark's own questions", "own_"), ("other sharks' questions", "oth_")),
                  "o1": pb.BY_OFFER_WHOSE}


def answer_quality_by_offer(rows: pd.DataFrame, outcome: str, **kw) -> pd.DataFrame:
    """`analysis.panel_bottleneck.answer_quality_by_offer` with the blocks of the row's unit: on shark rows the
    shark's own questions and the other sharks', on pitch rows the pooled block."""
    return pb.answer_quality_by_offer(rows, outcome, whose=BY_OFFER_WHOSE[_unit(rows)], **kw)


def single_features(rows: pd.DataFrame, outcome: str, adjust_for: str | None = None, **kw) -> pd.DataFrame:
    """`analysis.panel_bottleneck.single_features` of shark rows, adjusted by default for the questions asked of the
    row's unit (QUESTIONS_ASKED: own_questions on shark rows)."""
    with shark_level_rows():
        return pb.single_features(rows, outcome, adjust_for=QUESTIONS_ASKED[_unit(rows)] if adjust_for is None
                                  else adjust_for, **kw)


# ---- the pitch outcome predicted two ways ----------------------------------------------------
PRIMARY_ATTRIBUTED = "noisy_or"
ATTRIBUTED_NOTE = ("attributed is the noisy-or of the investors' probabilities, 1 - prod(1 - p): the chance that at "
                   "least one investor offers if they decided independently; it is the primary attributed estimate. "
                   "attributed_max is the largest single probability, which ignores how many investors there are.")


@_on_shark_rows
def pooled_vs_attributed(o1_rows: pd.DataFrame, o2_rows: pd.DataFrame, outcome: str = "any_offer",
                         o2_outcome: str = "ever_offer", group: str = GROUP, penalty: float | None = None,
                         min_category: int = 5, n_boot: int = N_BOOT, seed: int = SEED,
                         penalty_grid: tuple[float, ...] = PENALTY_GRID) -> dict:
    """Predict the pitch outcome O1 two ways on the same pitches and folds.

    `collective`: a head on the pitch rows (ARCHIVED_C_FAMILY's C5: controls, exposure, Stage A and every pooled
    feature). `attributed`: from the B5 probabilities of the held-out pitch's investors, the O2 head being
    trained on the investor rows with a known outcome of the other pitches. It is reported two ways:
    `attributed` is the noisy-or 1 - prod(1 - p), the primary attributed estimate, and
    `attributed_max` the largest investor probability. `difference` is attributed minus collective,
    `difference_max` the same for the maximum. A pitch with a known outcome but no investor row has
    nothing to attribute: it is left out of both arms and named in `pitches_without_investor_rows`.
    """
    o1 = o1_rows[o1_rows[outcome].notna()]
    pitches = sorted(set(o1[group]) & set(o2_rows[group]))
    without = sorted(set(o1[group]) - set(o2_rows[group]))
    o1 = o1[o1[group].isin(pitches)].reset_index(drop=True)
    o2 = o2_rows[o2_rows[group].isin(pitches)].reset_index(drop=True)
    n_groups = len(pitches)
    base = {"n_groups": n_groups, "n_events": int(o1[outcome].sum()), "label": label_for(n_groups),
            "pitches": pitches, "uncertainty": CONDITIONAL, "attributed_primary": PRIMARY_ATTRIBUTED,
            "attributed_note": ATTRIBUTED_NOTE, "penalty": penalty_setting(penalty, penalty_grid),
            "pitches_without_investor_rows": without}
    if n_groups < 2:
        return {**base, "status": "not estimable: fewer than two pitches with a known outcome in both tables",
                "collective": {}, "attributed": {}, "attributed_max": {}, "difference": {}, "difference_max": {},
                "predictions": pd.DataFrame()}
    families, collective_families = MODEL_FAMILY["B5"], ARCHIVED_C_FAMILY["C5"]
    o1 = o1.assign(y=o1[outcome].astype(float))
    o2 = o2.assign(y=o2[o2_outcome].astype(float))
    rows = []
    for held_out in pitches:
        train1, test1 = o1[o1[group] != held_out], o1[o1[group] == held_out]
        collective, _ = pb.fit_head(train1, test1, "y", collective_families, penalty, min_category, group, penalty_grid)
        train2 = o2[(o2[group] != held_out) & o2.y.notna()]
        test2 = o2[o2[group] == held_out]
        if len(train2) and len(test2):
            prob = pb.fit_head(train2, test2, "y", families, penalty, min_category, group, penalty_grid)[0]
            noisy_or, largest = float(1.0 - np.prod(1.0 - prob)), float(np.max(prob))
        else:       # nothing to learn from (or, defensively, no row to attribute to): the pitch base rate
            noisy_or = largest = float(train1.y.mean())
        rows.append({group: held_out, "y": float(test1.y.iloc[0]), "n_investors": int(len(test2)),
                     "p_collective": float(collective[0]), "p_attributed": noisy_or, "p_attributed_max": largest})
    pred = pd.DataFrame(rows)
    deltas = paired_deltas(pred, [("collective", "attributed"), ("collective", "attributed_max")], group, n_boot, seed)

    def difference(comparison: str) -> dict:
        return {r.metric: {"delta": r.delta, "ci_low": r.ci_low, "ci_high": r.ci_high, "p_value": r.p_value,
                           "n_boot": r.n_boot,
                           "better": r.better} for r in deltas[deltas.comparison == comparison].itertuples()}

    return {**base, "status": "estimated",
            "collective": metrics_of(pred.y, pred.p_collective), "attributed": metrics_of(pred.y, pred.p_attributed),
            "attributed_max": metrics_of(pred.y, pred.p_attributed_max),
            "difference": difference("attributed minus collective"),
            "difference_max": difference("attributed_max minus collective"),
            "predictions": pred}
