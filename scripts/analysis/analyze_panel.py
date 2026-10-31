#!/usr/bin/env python3
"""Write the pitch-level analysis of a panel run under <run>/analysis/<analysis_id>/, with its report.md.

    python scripts/analysis/analyze_panel.py --run runs/<run_name> --analysis-id pitch_level \
        [--video-ids ... | --selection selection_all100.json] [--penalty X] [--n-boot 1999] [--n-perm 500] \
        [--workers N] [--skip-h9] [--h9-exchange-outcomes [--h9-sensitivity-draws 499]] [--with-shark-level] [--force]

Offline and read-only towards the run: no provider is called and nothing is written outside the analysis folder.
Each video is read from its reviewed record when there is one, else its primary pass (or its replicate pass, when
the primary has validation errors and the replicate has fewer). Reviewed records with something still open are
read all the same, printed and listed; nothing is excluded.

The pre-registered test (refocus spec, `docs/plans/design docs/2026-10-19-pitch-level-refocus.md`, section 6):
one row per pitch, outcome `any_offer`, the ladder C0-C5 with C2p and C3q (analysis.panel_bottleneck.LADDER),
leave-one-pitch-out with the ridge penalty chosen inside each training fold, pitch-bootstrap intervals, the
pitch-level permutation test, and the verdicts of H1' (C3 - C2p) and H8 (C3 - C3q) from the rules in
analysis.verdicts, each with the power sentence. The same ladder is evaluated under five sensitivities (both
passes averaged, exposure as rates, without the development videos, the penalty fixed at 10, every Stage A rank
score in C2), each with its own verdicts. H9, the answer-level test (analysis.answer_level), runs every model
with intervals and no decision rule; `--workers` splits its bootstrap draws over processes without changing a
number, `--skip-h9` leaves it out of a quick run. `--h9-exchange-outcomes` also computes H9's exchange-based
outcomes (analysis.answer_level.h9_sensitivity) as a sensitivity at `--h9-sensitivity-draws` draws, written to
h9_exchange_sensitivity.csv: never a result, and report.md only points at the file.

Leakage rule. The model annotated each whole video, decisions included, so a pitch row's features use only turns
strictly before the pitch's cut: the first decision of any kind by any investor, so that the cut does not depend
on the outcome. Never a `decides` or `negotiates` turn, never a turn of a negotiation exchange. The descriptive
counts (descriptives.json, H4, H5) cover the whole encounter and are not predictors.

The shark-level design (one row per investor) is archived (analysis/archive). `--with-shark-level` reruns it on
the same records for the F1 paragraph (analysis.archive.shark_level.run) and writes it under shark_level/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis import pitch_report  # noqa: E402
from analysis.answer_level import (  # noqa: E402
    H9_COLUMNS, MODELS as H9_MODELS, POPULATIONS as H9_POPULATIONS, RESPONSIVENESS_MODEL, SENSITIVITY_OUTCOMES,
    SUMMARY_COLUMNS, h9, h9_sensitivity, h9_summary,
)
from analysis.panel_bottleneck import (  # noqa: E402
    BOOT_P_NOTE, COEFFICIENT_COLUMNS, COEFFICIENT_NOTE, DELTA_COLUMNS, FDR_NOTE, LADDER, LADDER_COMPARISONS,
    LADDER_NOT_FITTED, LADDER_NOT_FITTED_NOTE, N_BOOT, PENALTY_GRID, PERMUTATION_COLUMNS, PERMUTATION_NOTE, PRIMARY_STAGE_A, SEED, SENSITIVITY_LADDERS,
    SINGLE_DRAWS, SINGLE_MIN_COVERAGE, SINGLE_PENALTY, add_permutation_p, answer_quality_by_offer, check_models,
    evaluate, chance_comparisons,
    family_summary, h4_support_shares, modal_penalty, model_coefficients, penalty_setting, permutation_test,
    single_features, stage_a_fallback,
)
from analysis.panel_dataset import (  # noqa: E402
    BOOKS_BUNDLE, BOOKS_LIVE, MODEL_SOURCES, OPEN_ISSUE_COLUMNS, PASSES_BEST, PASSES_BOTH, RANKING_SCORES,
    REVIEWED_SOURCES, SINGLE_PASS_VIDEOS, SOURCE_OPEN, PanelData, answer_followups, development_videos, h4_tables,
    load_run, panel_next_move, pitch_cut, pitch_outcome, video_dirs,
)
from analysis.panel_explorer import (  # noqa: E402
    agreement_reports, agreement_table, function_agreement_table, instrument, ranking_batches, ranking_stability,
    resolve_video_ids,
)
from analysis.panel_features import EXPOSURE_RATES, feature_dictionary, pitch_features  # noqa: E402
from analysis.verdicts import RULES, VERDICT_COLUMNS, verdicts  # noqa: E402
from pipeline.errors import ConfigError  # noqa: E402

OUTCOME = "any_offer"
ANNOTATION_CONTEXT = "descriptive_full_context"
ANALYSIS_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
N_PERM = 500
SELF_CONSISTENCY_LABEL = "model self-consistency under sampling; not validity"
SCOPE = ("descriptive counts over the whole encounter as annotated with full context, decisions included; "
         "they are not predictors, which stop at the pitch's cut")
LEAKAGE_RULE = ("features of a pitch row use only turns strictly before the pitch's cut, the first decision of any "
                "kind by any investor (so the cut does not depend on the outcome); never decides or negotiates "
                "turns, never turns of a negotiation exchange")
OPEN_ISSUES_NOTE = ("reviewed records read although something is still open: validation errors of the reviewed "
                    "record, or a structural (wrong, missing, absent or present) review judgment on a decision that "
                    "no correction could apply. Nothing is excluded automatically.")
UNKNOWN_RULE = ("a pitch's outcome is unknown when no investor offered and at least one investor has neither an offer "
                "nor an exit on record, or when a reviewer judged it unknown")
H4_SCOPE = ("descriptive counts over the whole encounter as annotated, with no history cut; not predictors. One "
            "counted unit is an answer and an investor question it responds to. The next move is read from turn "
            "order, whatever exchange the model put each turn in: the first investor turn after the answer, passing "
            "over founder turns and turns that are no reaction")
H4_DEFINITIONS = {
    "asker": "what the investor who asked does next: positive_receipt (accepts), "
             "follow_up_question, challenge, other_by_asker, other_investor (another investor speaks first), "
             "no_reaction (no further investor turn, or the next one is decision talk)",
    "panel": "the first investor reaction by anyone, the asker or another investor: receipt (accepts), question, "
             "challenge, other, none (no investor reacts)",
    "hard": "questions of severity 2 (asks for substantiation) or 3 (contests a claim)",
    "substance": "the answer's substance rating, 1 (low) to 5 (high)",
    "support": "whether the answer gives checkable support",
}
H5_NOTE = ("question frame by answer frame, one count per answer turn, whole encounter (H5, descriptive): the question "
           "is the first question the answer responds to that has a frame; `no_frame` when it responds only to "
           "questions without one, `no_linked_question` when it responds to none. `matches` is true when both frames "
           "are set and equal; deal-terms and other questions have no frame")
FIGURES_FOLDER = "figures"
SHARK_LEVEL_FOLDER = "shark_level"
CONTROL_COLUMNS = ("founder_gender", "primary_industry", "product_form", "customer_type")
FOLD_COLUMNS = ["held_out", "model", "n_train", "n_test", "penalty", "penalty_mode", "single_class_training",
                "n_columns"]
SELF_CONSISTENCY_TABLES = ("self_consistency.csv", "self_consistency_turn_functions.csv")
RANKING_AGREEMENT_TABLE = "stage_a_ranking_agreement.csv"
RANKING_AGREEMENT_NOTE = ("computed over all ranked videos of the run, not only the analysed ones: a video's "
                          "standing is relative to the other clips of its batch")
SINGLE_NOTE = (f"each numeric feature on its own: correlation with the outcome (interval: bootstrap over pitches), "
               f"leave-one-pitch-out log-loss gain of a one-feature ridge logistic head (penalty {SINGLE_PENALTY:g}) "
               f"over the training base rate, the head's standardised coefficient on all rows, and the partial "
               f"correlation given the number of questions asked. Features observed on less than "
               f"{SINGLE_MIN_COVERAGE:.0%} of the rows, or constant, are left out. Descriptive ranking, not a test")
MAIN = "main"
BOTH_PASSES, RATES, NON_DEVELOPMENT, FIXED_PENALTY, ALL_STAGE_A = (
    "both_passes", "rates", "non_development", "fixed_penalty_10", "all_stage_a")
SENSITIVITIES = (BOTH_PASSES, RATES, NON_DEVELOPMENT, FIXED_PENALTY, ALL_STAGE_A)
SENSITIVITY_NOTES = {
    BOTH_PASSES: "both annotation passes averaged for every numeric feature (spec 6.4 a)",
    RATES: SENSITIVITY_LADDERS["rates"]["note"] + " (spec 6.4 b)",
    NON_DEVELOPMENT: ("without the pilot development videos named by the selection file (`--selection`, else the "
                      "run's own; spec 6.4 c); every video was looked at while the instrument was shaped (plan v4 "
                      "section 11), so this removes the pilot videos, not the looking"),
    FIXED_PENALTY: SENSITIVITY_LADDERS["fixed_penalty_10"]["note"] + " (spec 6.4 d)",
    ALL_STAGE_A: SENSITIVITY_LADDERS["all_stage_a"]["note"] + " (spec decision 9)",
}
RANKING_COMPLETE, RANKING_ABSENT = "complete", "absent: rating used"
# a sensitivity whose videos are the main analysis's is not run (final review I2): its rows would be main's
NOT_RUN_IDENTICAL = "not run: identical to main (no development video among the analysed videos)"
H9_RUN, H9_SKIPPED = "run", "skipped"
H9_TABLES = ("h9_all.csv", "h9_hard.csv", "h9_responsiveness.csv", "h9_summary.csv")
H9_SENSITIVITY_TABLE = "h9_exchange_sensitivity.csv"
H9_SENSITIVITY_DRAWS = 499       # fewer than the main draws: a sensitivity, not a result (ruling of plan B2)
_METRIC_COLUMNS = ["model", "families", "log_loss", "brier", "auc", "mean_pred", "base_rate", "base_log_loss",
                   "skill_pct", "auc_low", "auc_high", "skill_low", "skill_high", "skill_p_value", "auc_p_value",
                   "pred_spread", "auc_base_rate_corr", "auc_meaningful", "penalty", "role", "note", "perm_p_log_loss",
                   "perm_p_auc", "n_perm"]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_workers() -> int:
    """Every core but one, at least one."""
    return max(1, (os.cpu_count() or 1) - 1)


def _plain(obj):
    """JSON-ready: numpy scalars as Python numbers, NaN as null, tables as lists of records."""
    if isinstance(obj, pd.DataFrame):
        return [_plain(r) for r in obj.to_dict("records")]
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_plain(v) for v in (sorted(obj) if isinstance(obj, set) else obj)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if obj != obj else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _counts(series: pd.Series) -> dict:
    return {str(k): int(v) for k, v in series.value_counts().items()}


def coding_text(coding: dict) -> str:
    def pitches(n: int) -> str:
        return f"{n} pitch" if n == 1 else f"{n} pitches"
    reviewed = sum(coding.get(source, 0) for source in REVIEWED_SOURCES)
    still_open = coding.get(SOURCE_OPEN, 0)
    return (f"model-coded: {pitches(sum(coding.get(s, 0) for s in MODEL_SOURCES))}; human-reviewed: {pitches(reviewed)}"
            + (f" ({still_open} with open issues)" if still_open else ""))


def input_files(run_root: Path, video_ids: list[str] | None, selection_files: list[Path] = ()) -> list[Path]:
    """Every file this analysis reads, in a stable order: the selection files read (which videos, which of them are
    development videos; they may lie outside the run), then the run's: the record of the main analysis, both model
    passes (the both-passes sensitivity reads the other one), the reports beside them and the ranking."""
    from pipeline.stage_b.panel_pipeline import load_best_record       # deferred, as in analysis.panel_dataset

    files = [Path(p) for p in selection_files]
    for vdir in video_dirs(run_root, video_ids):
        results = vdir / "results"
        read = results / f"{load_best_record(vdir)[1]}.json"
        files.append(read)
        passes = [results / "primary.json", results / "replicate.json"]
        optional = [p for p in passes if p != read] + [vdir / "experiment.json", vdir / "agreement.json",
                                                       vdir / "review" / "human_agreement.json"]
        if (results / "reviewed.json").exists():
            optional += [results / "reviewed_validation.json", vdir / "review" / "review_log.jsonl"]
        optional += [results / "primary_validation.json", results / "replicate_validation.json"]
        files.extend(p for p in optional if p.exists())
    if (run_root / RANKING_SCORES).exists():
        files.append(run_root / RANKING_SCORES)
    batches = run_root / "ranking" / "batches"
    if batches.is_dir():
        files.extend(sorted(batches.glob("*/result.json")))
    return list(dict.fromkeys(files))


def input_key(path: Path, run_root: Path) -> str:
    """How manifest["inputs"] names a file read: relative to the run, or its absolute path outside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path(run_root).resolve()))
    except ValueError:
        return str(path)


def selection_file(run_root: Path, video_ids) -> Path | None:
    """The selection file `video_ids` names (a file name inside the run folder, or a path), as
    `analysis.panel_explorer.resolve_video_ids` reads it; None for a list of ids or every video."""
    if not isinstance(video_ids, (str, Path)):
        return None
    path = Path(video_ids)
    return path if path.is_absolute() else Path(run_root) / path


def self_consistency(run_root: Path, video_ids: list[str] | None) -> dict:
    """The headline numbers of each video's agreement.json (two identical passes compared), and their means."""
    videos: dict[str, dict] = {}
    label = SELF_CONSISTENCY_LABEL
    for vdir in video_dirs(run_root, video_ids):
        path = vdir / "agreement.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        label = report.get("label") or label
        videos[vdir.name] = {f"{section}.{key}": float(value)
                             for section, block in report.items() if isinstance(block, dict)
                             for key, value in block.items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)}
    keys = sorted({k for numbers in videos.values() for k in numbers})
    mean = {k: float(np.mean([v[k] for v in videos.values() if k in v])) for k in keys}
    return {"label": label, "n_videos": len(videos), "mean": mean, "videos": videos}


def human_agreement(run_root: Path, video_ids: list[str] | None) -> dict:
    out = {}
    for vdir in video_dirs(run_root, video_ids):
        path = vdir / "review" / "human_agreement.json"
        if path.exists():
            out[vdir.name] = json.loads(path.read_text())
    return out


def _summary(values: pd.Series) -> dict:
    observed = values.dropna()
    return {"n": int(len(observed)), "mean": float(observed.mean()) if len(observed) else None,
            "sd": float(observed.std(ddof=0)) if len(observed) else None,
            "min": float(observed.min()) if len(observed) else None,
            "max": float(observed.max()) if len(observed) else None}


def _frame_table(turns: pd.DataFrame) -> dict:
    """Question frame by answer frame, one count per answer turn.

    The question is the one `derive` matches the answer against: the first question the answer
    responds to that has a frame. An answer linked only to questions without a frame counts under
    `no_frame`, one linked to no question under `no_linked_question`.
    """
    table: dict[str, dict[str, int]] = {}
    for video_id, rows in turns.groupby("video_id"):
        frame_of = {t.turn_id: t.frame for t in rows.itertuples() if t.function == "question"}
        for t in rows[rows.function == "answer"].itertuples():
            linked = [frame_of[q] for q in t.responds_to if q in frame_of]
            framed = [frame for frame in linked if isinstance(frame, str)]
            question = framed[0] if framed else ("no_frame" if linked else "no_linked_question")
            answer = t.frame if isinstance(t.frame, str) else "no_frame"
            table.setdefault(question, {}).setdefault(answer, 0)
            table[question][answer] += 1
    return table


UNFRAMED = ("no_frame", "no_linked_question")


def h5_frame_match(turns: pd.DataFrame) -> pd.DataFrame:
    """h5_frame_match.csv: the frame table in long form, with whether the frames match and each cell's share of the
    answers to that question frame (H5_NOTE)."""
    rows = [{"question_frame": q, "answer_frame": a, "n": n,
             "matches": bool(q not in UNFRAMED and a not in UNFRAMED and q == a)}
            for q, answers in sorted(_frame_table(turns).items()) for a, n in sorted(answers.items())]
    table = pd.DataFrame(rows, columns=["question_frame", "answer_frame", "n", "matches"])
    totals = table.groupby("question_frame").n.transform("sum") if len(table) else table.n
    return table.assign(share_of_question_frame=(table.n / totals) if len(table) else table.n.astype(float),
                        scope=H5_NOTE)


def open_issues(videos: pd.DataFrame) -> dict:
    """The reviewed videos that still have something open, with both counts; see OPEN_ISSUES_NOTE."""
    flagged = videos[videos.source == SOURCE_OPEN]
    return {"note": OPEN_ISSUES_NOTE, "n": int(len(flagged)),
            "videos": {r.video_id: {name: int(getattr(r, name)) for name in OPEN_ISSUE_COLUMNS}
                       for r in flagged.itertuples()}}


def books_counts(books_source: dict) -> dict:
    """Videos per books source ("bundle" or "live"); the both-passes key `single_pass_videos` is no video."""
    return _counts(pd.Series([v for k, v in books_source.items() if k != SINGLE_PASS_VIDEOS], dtype=object))


def descriptives(data: PanelData, outcome: pd.DataFrame, run_root: Path, video_ids: list[str] | None,
                 label: str) -> dict:
    coding = _counts(data.videos.source)
    turns = data.turns
    questions = turns[turns.function == "question"]
    substantive = data.exchanges[data.exchanges.kind == "substantive"]
    ratings = data.stage_a[data.stage_a.source == "rating"]
    ranks = data.stage_a[data.stage_a.source == "rank_pl"]
    unknown = outcome[outcome.any_offer.isna()]
    return {
        "annotation_context": ANNOTATION_CONTEXT, "scope": SCOPE, "label": label,
        "coding": coding, "coding_text": coding_text(coding),
        "source_by_video": dict(zip(data.videos.video_id, data.videos.source)),
        "reviewed_with_open_issues": open_issues(data.videos),
        "outcome_unknown": {"rule": UNKNOWN_RULE, "n": int(len(unknown)), "videos": list(unknown.video_id)},
        "outcome_source": _counts(outcome.any_offer_source),
        "counts": {"pitches": int(len(data.videos)), "pitches_with_offer": int((outcome.any_offer == 1).sum()),
                   "pitches_without_offer": int((outcome.any_offer == 0).sum()),
                   "pitches_unknown": int(outcome.any_offer.isna().sum()),
                   "investor_seats": int(len(data.actors[data.actors.role == "investor"]))
                   if "role" in data.actors else None},
        "exchanges_by_kind": _counts(data.exchanges.kind),
        "exchanges_by_topic": _counts(substantive.topic),
        "question_concepts": _counts(questions.concept),
        "question_frames": _counts(questions.frame),
        "frame_match": _frame_table(turns),
        "frame_match_states": _counts(turns.frame_match[turns.function == "answer"]),
        **{name: _counts(data.videos[name]) for name in CONTROL_COLUMNS},
        "self_consistency": self_consistency(run_root, video_ids),
        "human_agreement": human_agreement(run_root, video_ids),
        "stage_a": {
            "ratings": {code: {**_summary(rows.value), "uncodable": int(rows.value.isna().sum())}
                        for code, rows in ratings.groupby("code")},
            "ranks": {code: _summary(rows.value) for code, rows in ranks.groupby("code")}},
    }


def _flat(table: pd.DataFrame) -> pd.DataFrame:
    """Tuple-valued columns as space-separated text, for CSV."""
    out = table.copy()
    for column in out.columns:
        if len(out) and out[column].map(lambda v: isinstance(v, tuple)).any():
            out[column] = out[column].map(lambda v: " ".join(str(x) for x in v) if isinstance(v, tuple) else v)
    return out


def _folds(result: dict) -> pd.DataFrame:
    """One row per outer fold and model: what was held out and the penalty that model used there."""
    rows = [{"held_out": fold["held_out"], "model": name, "n_train": fold["n_train"], "n_test": fold["n_test"],
             "penalty": detail.get("penalty"), "penalty_mode": detail.get("penalty_mode"),
             "single_class_training": detail["single_class_training"], "n_columns": len(detail["columns"])}
            for fold in result["folds"] for name, detail in fold["models"].items()]
    return pd.DataFrame(rows, columns=FOLD_COLUMNS)


def stage_a_ranking(data: PanelData) -> str:
    """Whether every analysed pitch has the rank score C2 reads: RANKING_COMPLETE, "partial: n of N", or
    RANKING_ABSENT (no pitch has one, so C2 reads the rating of the same item)."""
    ranked = data.stage_a[(data.stage_a.source == "rank_pl") & (data.stage_a.code == PRIMARY_STAGE_A)
                          & data.stage_a.value.notna()]
    n, total = int(ranked.video_id.nunique()), int(len(data.videos))
    if n == 0:
        return RANKING_ABSENT
    return RANKING_COMPLETE if n >= total else f"partial: {n} of {total}"


def self_consistency_tables(run_root: Path, video_ids: list[str]) -> dict[str, pd.DataFrame]:
    """The pass-to-pass agreement of the analysed videos as two export tables (analysis.panel_explorer's
    agreement_table and function_agreement_table, with plain column names, the number of videos compared
    and the label in every row). Counts are summed over videos before a share is taken."""
    reports = agreement_reports(run_root, list(video_ids))
    books = instrument(Path(run_root) / next(iter(reports))).books if reports else None
    out = {}
    for name, table in zip(SELF_CONSISTENCY_TABLES, (agreement_table(reports, books), function_agreement_table(reports))):
        table = table.rename(columns={"Within one (count)": "within_one", "Within one point": "share_within_one"})
        out[name] = table.rename(columns=_snake).assign(n_videos=len(reports), label=SELF_CONSISTENCY_LABEL)
    return out


def _snake(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass
class Analysis:
    """Everything the pitch-level analysis computes, in memory: `assemble` turns it into files, `report_markdown`
    into report.md."""
    run_root: Path
    video_ids: list[str] | None
    penalty: float | None
    n_boot: int
    seed: int
    n_perm: int
    single_draws: int
    workers: int
    with_shark_level: bool
    data: PanelData
    outcome: pd.DataFrame                 # pitch_outcome with pitch_cut: one row per pitch
    features: pd.DataFrame                # pitch_features (counts, best record)
    models: dict                          # the ladder as evaluated (the rating in C2 when the run has no rank)
    result: dict                          # evaluate() of the ladder with LADDER_COMPARISONS
    permutation: pd.DataFrame
    verdicts: pd.DataFrame                # the main verdicts (VERDICT_COLUMNS)
    sensitivities: dict = field(default_factory=dict)        # name -> evaluate() of the ladder, with `note`
    sensitivity_verdicts: dict = field(default_factory=dict)  # name -> verdicts frame
    sensitivity_info: dict = field(default_factory=dict)      # name -> {note, excluded_videos, ...} for the manifest
    selection_files: list = field(default_factory=list)       # every selection file read (hashed in the manifest)
    single: pd.DataFrame = None           # single_features of the pitch rows
    families: pd.DataFrame = None         # family_summary of `single`
    by_offer: pd.DataFrame = None         # answer_quality_by_offer of the pitch rows (figure 5)
    coefficients: pd.DataFrame = None     # model_coefficients of C5
    h9: dict | None = None                # answer_level.h9 (all, hard), None when skipped
    h9_summary: pd.DataFrame = None
    h9_sensitivity: dict | None = None    # answer_level.h9_sensitivity (all, hard), with --h9-exchange-outcomes only
    h9_sensitivity_draws: int | None = None
    followups: pd.DataFrame = None        # answer_followups of the whole encounter (H4)
    h4_shares: pd.DataFrame = None
    h5: pd.DataFrame = None               # h5_frame_match
    self_consistency: dict = None         # the two self-consistency tables
    ranking_agreement: pd.DataFrame = None
    stage_a_ranking: str = RANKING_COMPLETE
    single_pass_videos: dict = field(default_factory=dict)    # from the both-passes read: {video: why}
    shark_level: dict | None = None       # analysis.archive.shark_level.run, with --with-shark-level
    seconds: dict = field(default_factory=dict)               # how long each step took

    @property
    def h9_status(self) -> str:
        return H9_SKIPPED if self.h9 is None else H9_RUN

    @property
    def label(self) -> str:
        return self.result["label"]


def _evaluate(features: pd.DataFrame, models: dict, penalty: float | None, n_boot: int, seed: int,
              note: str = "") -> dict:
    """`evaluate` of the ladder `models` with LADDER_COMPARISONS, the columns of LADDER_NOT_FITTED left out of every
    design (ruling I4). `note` is the Stage A fallback note of `stage_a_fallback`, applied before this call (so
    `evaluate` sees ratings and finds nothing to replace): it is stamped on every metrics row (final review I1)."""
    result = evaluate(features, OUTCOME, penalty=penalty, models=dict(models), n_boot=n_boot, seed=seed,
                      comparisons=LADDER_COMPARISONS, not_fitted=LADDER_NOT_FITTED)
    if note and len(result["metrics"]):
        result["metrics"]["note"] = note     # evaluate found nothing to replace, so its own note is empty
    return result


def _not_run(name: str, note: str, main: dict) -> tuple[dict, pd.DataFrame]:
    """(the evaluation, the verdicts) of a sensitivity that was not run (NOT_RUN_IDENTICAL): no metrics or deltas,
    the main analysis's sample size, and one verdict row per hypothesis that says so."""
    evaluated = {"status": NOT_RUN_IDENTICAL, "sensitivity": name, "sensitivity_note": note,
                 "n_rows": main.get("n_rows", 0), "n_groups": main.get("n_groups", 0),
                 "n_events": main.get("n_events", 0), "label": main.get("label"), "metrics": pd.DataFrame(),
                 "deltas": pd.DataFrame(), "design_columns": {}, "folds": []}
    rows = [{"hypothesis": hypothesis, "comparison": f"{rule['pair'][1]} minus {rule['pair'][0]}",
             "auc_route": NOT_RUN_IDENTICAL, "verdict": NOT_RUN_IDENTICAL, "power": ""}
            for hypothesis, rule in RULES.items()]
    return evaluated, pd.DataFrame(rows).reindex(columns=VERDICT_COLUMNS)


def _verdicts(result: dict, features: pd.DataFrame) -> pd.DataFrame:
    known = features[features[OUTCOME].notna()]
    return verdicts(result["deltas"], int(len(known)), int((known[OUTCOME] == 0).sum()), metrics=result["metrics"])


def compute(run_root: Path | str, video_ids: list[str] | None = None, penalty: float | None = None,
            n_boot: int = N_BOOT, seed: int = SEED, n_perm: int = N_PERM, with_shark_level: bool = False,
            workers: int = 1, skip_h9: bool = False, single_draws: int = SINGLE_DRAWS,
            h9_exchange_outcomes: bool = False, h9_sensitivity_draws: int = H9_SENSITIVITY_DRAWS) -> Analysis:
    """Read a run and compute the pitch-level analysis. Reads only; writes nothing.

    Steps, in order: `load_run` (best record) -> `pitch_outcome`, `pitch_cut` -> `pitch_features` -> `evaluate` of
    LADDER with LADDER_COMPARISONS -> `permutation_test` (skipped with `n_perm` 0) -> `verdicts` (with the ladder's
    metrics) -> the sensitivities (both passes, rates, non-development, fixed penalty 10, all Stage A; each a full
    `evaluate` of the ladder with the same `n_boot` and `seed`, and its verdicts) -> `single_features` and
    `answer_quality_by_offer` (figure 5) -> `model_coefficients` of C5 -> `h9` (skipped with `skip_h9`; `workers` processes) -> H4 (`answer_followups`,
    `h4_support_shares`) -> H5 frame-match table -> self-consistency tables -> `ranking_stability`; with
    `with_shark_level`, `analysis.archive.shark_level.run` on the same records.

    `video_ids` is None (every video with a result), a list of ids, or the name of a selection file in the run
    folder (analysis.panel_explorer.resolve_video_ids). `penalty` None chooses the ridge penalty inside each
    training fold; a number fixes it (the sensitivities other than the fixed-penalty one use the same setting).
    `single_draws` is the number of bootstrap draws behind the single-feature table's FDR-adjusted p-values (at
    least `n_boot`). `h9_exchange_outcomes` also runs `h9_sensitivity` (the exchange-based outcomes) with
    `h9_sensitivity_draws` draws; it needs H9, so it is refused with `skip_h9`.
    """
    if h9_exchange_outcomes and skip_h9:
        raise ConfigError("--h9-exchange-outcomes needs H9: it cannot be combined with --skip-h9")
    if int(h9_sensitivity_draws) < 1:
        raise ConfigError(f"--h9-sensitivity-draws must be at least 1, not {h9_sensitivity_draws}")
    seconds: dict[str, float] = {}

    def timed(name):
        started = time.perf_counter()

        def stop():
            seconds[name] = round(time.perf_counter() - started, 2)
        return stop

    run_root = Path(run_root)
    selection = selection_file(run_root, video_ids)
    video_ids = resolve_video_ids(run_root, video_ids)
    stop = timed("load")
    data = load_run(run_root, video_ids, passes=PASSES_BEST)
    if data.videos.empty:
        raise ConfigError(f"no video with a primary or reviewed result under {run_root}")
    outcome = pitch_outcome(data).merge(pitch_cut(data), on="video_id", validate="one_to_one")
    features = pitch_features(data)
    stop()
    models, fallback_note = stage_a_fallback(dict(LADDER), features)
    try:
        models = check_models(models, features)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    stop = timed("ladder")
    result = _evaluate(features, models, penalty, n_boot, seed, fallback_note)
    stop()
    stop = timed("permutation")
    if n_perm and result["status"] == "estimated":
        permutation = permutation_test(features, OUTCOME, result, n_perm=n_perm, seed=seed)
        add_permutation_p(result, permutation)
    else:
        permutation = pd.DataFrame(columns=PERMUTATION_COLUMNS)
    stop()
    main_verdicts = _verdicts(result, features)

    stop = timed("sensitivities")
    both = load_run(run_root, video_ids, passes=PASSES_BOTH)
    development, selection_files = development_videos(run_root, selection)
    if selection is not None and selection not in selection_files:      # read for the videos, not the key
        selection_files.insert(0, selection)
    excluded = sorted(set(features.video_id) & development)
    variants = {
        BOTH_PASSES: (pitch_features(both), LADDER, penalty),
        RATES: (pitch_features(data, exposure=EXPOSURE_RATES), SENSITIVITY_LADDERS["rates"]["models"], penalty),
        NON_DEVELOPMENT: (features[~features.video_id.isin(development)].reset_index(drop=True), LADDER, penalty),
        FIXED_PENALTY: (features, SENSITIVITY_LADDERS["fixed_penalty_10"]["models"],
                        SENSITIVITY_LADDERS["fixed_penalty_10"]["penalty"]),
        ALL_STAGE_A: (features, SENSITIVITY_LADDERS["all_stage_a"]["models"], penalty),
    }
    notes = dict(SENSITIVITY_NOTES)
    sensitivities, sensitivity_verdicts, info = {}, {}, {}
    for name, (rows, spec, setting) in variants.items():
        if name == NON_DEVELOPMENT and set(rows.video_id) == set(features.video_id):
            sensitivities[name], sensitivity_verdicts[name] = _not_run(name, notes[name], result)
            info[name] = {"note": notes[name], "status": NOT_RUN_IDENTICAL, "penalty": penalty_setting(setting)["text"],
                          "n_pitches": int(rows[OUTCOME].notna().sum()), "models": {}, "design_columns": {}}
            continue
        spec, note = stage_a_fallback(dict(spec), rows)
        evaluated = _evaluate(rows, spec, setting, n_boot, seed, note)
        sensitivities[name] = {**evaluated, "sensitivity": name, "sensitivity_note": notes[name]}
        sensitivity_verdicts[name] = _verdicts(evaluated, rows)
        info[name] = {"note": notes[name], "status": evaluated["status"], "penalty": penalty_setting(setting)["text"],
                      "n_pitches": int(rows[OUTCOME].notna().sum()),
                      "models": {m: list(s) for m, s in spec.items()},
                      "design_columns": evaluated.get("design_columns", {})}
    info[NON_DEVELOPMENT]["excluded_videos"] = excluded
    info[NON_DEVELOPMENT]["selection_files"] = [str(p) for p in selection_files]
    info[BOTH_PASSES]["single_pass_videos"] = dict(both.books_source.get(SINGLE_PASS_VIDEOS, {}))
    stop()

    stop = timed("single_features")
    draws = max(int(single_draws), int(n_boot))
    single = single_features(features, OUTCOME, n_boot=draws, seed=seed)
    families = family_summary(single)
    by_offer = answer_quality_by_offer(features, OUTCOME, n_boot=draws, seed=seed)
    stop()
    if result["status"] == "estimated" and "C5" in result["metrics"].model.values:
        coefficients = model_coefficients(features, OUTCOME, tuple(models["C5"]), modal_penalty(result, "C5"),
                                          model="C5", n_boot=n_boot, seed=seed, not_fitted=LADDER_NOT_FITTED)
    else:
        coefficients = pd.DataFrame(columns=COEFFICIENT_COLUMNS)

    stop = timed("h9")
    h9_result = None if skip_h9 else h9(data, pitch_cut(data), n_boot=n_boot, seed=seed, workers=workers)
    summary = h9_summary(h9_result) if h9_result is not None else pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    stop()
    sensitivity = None
    if h9_exchange_outcomes:
        stop = timed("h9_exchange_sensitivity")
        sensitivity = h9_sensitivity(data, pitch_cut(data), n_boot=h9_sensitivity_draws, seed=seed, workers=workers)
        stop()

    stop = timed("descriptives")
    followups = answer_followups(data)
    shares = h4_support_shares(followups, n_boot=n_boot, seed=seed)
    consistency = self_consistency_tables(run_root, list(data.videos.video_id))
    agreement = ranking_stability(run_root, n_boot=n_boot, seed=seed)
    stop()

    shark = None
    if with_shark_level:
        from analysis.archive import shark_level       # deferred: the archive is read only behind the flag

        stop = timed("shark_level")
        shark = shark_level.run(data, run_root, n_boot=n_boot, seed=seed, n_perm=n_perm, penalty=penalty)
        stop()
    return Analysis(run_root=run_root, video_ids=video_ids, penalty=penalty, n_boot=int(n_boot), seed=int(seed),
                    n_perm=int(n_perm), single_draws=draws, workers=int(workers), with_shark_level=bool(with_shark_level),
                    data=data, outcome=outcome, features=features, models=models, result=result,
                    permutation=permutation, verdicts=main_verdicts, sensitivities=sensitivities,
                    sensitivity_verdicts=sensitivity_verdicts, sensitivity_info=info,
                    selection_files=list(selection_files), single=single,
                    families=families, by_offer=by_offer, coefficients=coefficients, h9=h9_result, h9_summary=summary,
                    h9_sensitivity=sensitivity,
                    h9_sensitivity_draws=int(h9_sensitivity_draws) if sensitivity is not None else None,
                    followups=followups, h4_shares=shares, h5=h5_frame_match(data.turns), self_consistency=consistency,
                    ranking_agreement=agreement, stage_a_ranking=stage_a_ranking(data),
                    single_pass_videos=info[BOTH_PASSES]["single_pass_videos"], shark_level=shark, seconds=seconds)


def h4_document(followups: pd.DataFrame, label: str, coding: dict) -> dict:
    """h4_tables.json: the next-move counts with what they count and over what."""
    return {"annotation_context": ANNOTATION_CONTEXT, "scope": H4_SCOPE, "label": label, "coding": coding,
            "coding_text": coding_text(coding), "definitions": H4_DEFINITIONS,
            "n_pitches": int(followups.video_id.nunique()), "n_rows": int(len(followups)),
            "n_rows_hard": int(followups.hard_question.sum()), "tables": h4_tables(followups)}


def _labelled(table: pd.DataFrame, result: dict, coding: dict, columns: list[str] | None = None) -> pd.DataFrame:
    """A model table with the sample behind it in every row."""
    if not len(table.columns):
        table = pd.DataFrame(columns=columns or _METRIC_COLUMNS)
    return table.assign(label=result["label"], coding=coding_text(coding), n_pitches=result["n_groups"],
                        n_events=result["n_events"])


def h9_tables(a: Analysis) -> dict[str, pd.DataFrame]:
    """h9_all.csv and h9_hard.csv (the prepared-answer models), h9_responsiveness.csv (its own model, for all and
    hard questions, column `questions`) and h9_summary.csv; header-only tables when H9 was skipped."""
    if a.h9 is None:
        empty = pd.DataFrame(columns=list(H9_COLUMNS))
        return {"h9_all.csv": empty, "h9_hard.csv": empty,
                "h9_responsiveness.csv": pd.DataFrame(columns=["questions", *H9_COLUMNS]),
                "h9_summary.csv": pd.DataFrame(columns=list(SUMMARY_COLUMNS))}
    prepared = {name: frame[frame.model != RESPONSIVENESS_MODEL].reset_index(drop=True) for name, frame in a.h9.items()}
    responsiveness = pd.concat([frame[frame.model == RESPONSIVENESS_MODEL].assign(questions=name)
                                for name, frame in a.h9.items()], ignore_index=True)
    tables = {"h9_all.csv": prepared["all"], "h9_hard.csv": prepared["hard"],
              "h9_responsiveness.csv": responsiveness[["questions", *H9_COLUMNS]], "h9_summary.csv": a.h9_summary}
    if a.h9_sensitivity is not None:
        tables[H9_SENSITIVITY_TABLE] = h9_sensitivity_table(a)
    return tables


def h9_sensitivity_table(a: Analysis) -> pd.DataFrame:
    """h9_exchange_sensitivity.csv: the exchange-based outcomes (answer_level.h9_sensitivity) for all and hard
    questions, column `questions`, every model in one table. Written only with --h9-exchange-outcomes."""
    return pd.concat([frame.assign(questions=name) for name, frame in a.h9_sensitivity.items()],
                     ignore_index=True)[["questions", *H9_COLUMNS]]


def h9_sensitivity_facts(a: Analysis) -> dict | None:
    """What the manifest records of the exchange-based sensitivity; None when it was not run."""
    if a.h9_sensitivity is None:
        return None
    return {"table": H9_SENSITIVITY_TABLE, "draws": a.h9_sensitivity_draws, "outcomes": list(SENSITIVITY_OUTCOMES),
            "note": "outcomes read on the model's exchange grouping, unstable between passes: computed as a "
                    "sensitivity, not reported, no direction stated"}


def sensitivity_tables(a: Analysis) -> dict[str, pd.DataFrame]:
    coding = _counts(a.data.videos.source)
    metrics, deltas = [], []
    for name, evaluated in a.sensitivities.items():
        extra = {"sensitivity": name, "sensitivity_note": evaluated["sensitivity_note"]}
        if str(evaluated.get("status", "")).startswith("not run"):          # one row that says so (I2)
            row = pd.DataFrame([{"note": evaluated["status"]}])
            metrics.append(_labelled(row, evaluated, coding).assign(**extra))
            deltas.append(_labelled(row, evaluated, coding).assign(**extra))
            continue
        metrics.append(_labelled(evaluated["metrics"], evaluated, coding).assign(**extra))
        deltas.append(_labelled(evaluated["deltas"], evaluated, coding, list(DELTA_COLUMNS)).assign(**extra))
    first = ["sensitivity", "sensitivity_note"]

    def ordered(frames):
        table = pd.concat(frames, ignore_index=True)
        return table[first + [c for c in table.columns if c not in first]]
    return {"sensitivity_metrics.csv": ordered(metrics), "sensitivity_deltas.csv": ordered(deltas)}


def verdict_table(a: Analysis) -> pd.DataFrame:
    """verdicts.csv: the main verdicts first, then each sensitivity's, column `setting`."""
    parts = [a.verdicts.assign(setting=MAIN)] + [table.assign(setting=name)
                                                 for name, table in a.sensitivity_verdicts.items()]
    return pd.concat(parts, ignore_index=True)[["setting", *VERDICT_COLUMNS]]


def assemble(a: Analysis) -> tuple[dict, dict, dict]:
    """(csv tables, json documents, facts for the manifest) of a computed analysis, all in memory."""
    data, result = a.data, a.result
    source = data.videos[["video_id", "source"]]
    coding = _counts(data.videos.source)
    keys = ["video_id", "source", "y", "weight"]
    tables = {
        "videos.csv": data.videos,
        "turns.csv": _flat(data.turns).merge(source, on="video_id"),
        "exchanges.csv": data.exchanges.merge(source, on="video_id"),
        "decisions.csv": data.decisions.merge(source, on="video_id"),
        "prepared.csv": data.prepared.merge(source, on="video_id"),
        "outcomes.csv": a.outcome.merge(source, on="video_id"),
        "features.csv": a.features,
        "feature_dictionary.csv": feature_dictionary(),
        "metrics.csv": _labelled(result["metrics"], result, coding),
        "chance_comparisons.csv": _labelled(chance_comparisons(result["metrics"]), result, coding),
        "deltas.csv": _labelled(result["deltas"], result, coding, list(DELTA_COLUMNS)),
        "predictions.csv": _labelled(result["predictions"], result, coding,
                                     [*keys, *(c for m in a.models for c in (f"p_{m}", f"score_{m}"))]),
        "folds.csv": _folds(result),
        "permutation.csv": a.permutation.assign(label=result["label"], coding=coding_text(coding)),
        "verdicts.csv": verdict_table(a),
        **sensitivity_tables(a),
        "single_feature.csv": a.single.assign(sample_label=result["label"], coding=coding_text(coding),
                                              n_pitches=result["n_groups"], n_events=result["n_events"]),
        "single_feature_families.csv": a.families,
        "answer_quality_by_offer.csv": a.by_offer.assign(sample_label=result["label"], coding=coding_text(coding),
                                                         n_pitches=result["n_groups"], n_events=result["n_events"]),
        "coefficients.csv": a.coefficients.assign(sample_label=result["label"], coding=coding_text(coding)),
        **h9_tables(a),
        "h4_followups.csv": a.followups.assign(panel_next_move=panel_next_move(a.followups)).merge(source, on="video_id"),
        "h4_support_shares.csv": a.h4_shares.assign(label=result["label"], coding=coding_text(coding)),
        "h5_frame_match.csv": a.h5,
        **a.self_consistency,
        RANKING_AGREEMENT_TABLE: a.ranking_agreement.rename(
            columns={"95% interval, low": "ci_low", "95% interval, high": "ci_high", "p-value": "p_value"})
        .rename(columns=_snake).assign(scope=RANKING_AGREEMENT_NOTE, label=SELF_CONSISTENCY_LABEL),
    }
    described = descriptives(data, a.outcome, a.run_root, a.video_ids, result["label"])
    documents = {"descriptives.json": described, "h4_tables.json": h4_document(a.followups, result["label"], coding)}
    setting = penalty_setting(a.penalty)
    facts = {
        "release_ids": sorted({r for r in data.videos.release_id if isinstance(r, str)}),
        "label": result["label"], "coding": coding, "counts": described["counts"], "model_status": result["status"],
        "penalty": setting["text"].split(": ", 1)[1], "penalty_grid": setting.get("grid"),
        "penalty_rule": setting.get("rule"),
        "models": {name: list(spec) for name, spec in a.models.items()},
        "design_columns": result.get("design_columns", {}),
        "not_fitted": {"columns": list(LADDER_NOT_FITTED), "note": LADDER_NOT_FITTED_NOTE},
        "comparisons": [list(pair) for pair in LADDER_COMPARISONS],
        "stage_a_ranking": a.stage_a_ranking,
        "verdicts": _plain(a.verdicts[["hypothesis", "comparison", "auc_route", "verdict"]]),
        "passes": {MAIN: PASSES_BEST, BOTH_PASSES: PASSES_BOTH},
        "single_pass_videos": a.single_pass_videos,
        "sensitivities": a.sensitivity_info,
        "h9": a.h9_status, "h9_models": {name: list(p) for name, p in H9_MODELS.items()},
        "h9_populations": dict(H9_POPULATIONS), "workers": a.workers,
        "h9_exchange_sensitivity": h9_sensitivity_facts(a),
        "single_feature_note": SINGLE_NOTE,
        "significance": {"bootstrap_p": BOOT_P_NOTE, "permutation": {"note": PERMUTATION_NOTE, "n_perm": a.n_perm},
                         "fdr": {"note": FDR_NOTE, "draws": a.single_draws}, "coefficients": COEFFICIENT_NOTE},
        "ranking_agreement": {"table": RANKING_AGREEMENT_TABLE, "scope": RANKING_AGREEMENT_NOTE,
                              "n_batches": len(ranking_batches(a.run_root))},
        "books": {"note": (f"codebook spellings and the Stage A item list come from each video's own bundle folder "
                           f"({BOOKS_BUNDLE}) when it holds the instrument, otherwise from the live release "
                           f"({BOOKS_LIVE})"),
                  "videos_by_source": books_counts(data.books_source),
                  "source_by_video": {k: v for k, v in sorted(data.books_source.items()) if k != SINGLE_PASS_VIDEOS},
                  "stage_a_codes": list(data.stage_a_codes)},
        "reviewed_with_open_issues": described["reviewed_with_open_issues"]["videos"],
        "seconds": a.seconds,
    }
    return tables, documents, facts


def provenance(a: Analysis) -> dict:
    """What report.md section 9 and the manifest both state: commit, texts quoted, settings."""
    return {"git_commit": pitch_report.git_commit(REPO),
            "release_ids": sorted({r for r in a.data.videos.release_id if isinstance(r, str)}),
            "seed": a.seed, "n_boot": a.n_boot, "n_perm": a.n_perm, "workers": a.workers,
            "penalty": penalty_setting(a.penalty)["text"].split(": ", 1)[1], "h9": a.h9_status, "with_shark_level": a.with_shark_level,
            "spec": {"path": str(pitch_report.SPEC.relative_to(REPO)), "sha256": sha256(pitch_report.SPEC),
                     "section": "6"},
            "limits": {"path": str(pitch_report.PLAN_V4.relative_to(REPO)), "sha256": sha256(pitch_report.PLAN_V4),
                       "section": "11"}}


def write_analysis(a: Analysis, analysis_id: str, force: bool = False, figure_width="double") -> Path:
    """Write the files of a computed analysis under <run>/analysis/<analysis_id>/ and return that folder.

    Refuses an id that is not a plain folder name, and an existing analysis unless `force` (which replaces the
    folder). The figures (analysis.pitch_figures) go to `figures/` at the width preset `figure_width`; a figure
    that cannot be drawn (S1 without the shark-level run, figure 9 with --skip-h9) or whose builder fails is left
    out and listed with the reason in the manifest (`figures_skipped`). With the shark-level run, its tables go to
    `shark_level/`.
    """
    from analysis import pitch_figures      # deferred: matplotlib is needed only when files are written

    if not ANALYSIS_ID.fullmatch(str(analysis_id)):
        raise ConfigError(f"analysis id {analysis_id!r} must be a plain folder name (letters, digits, . _ -)")
    run_root = Path(a.run_root).resolve()
    out = run_root / "analysis" / analysis_id
    if out.exists() and not force:
        raise ConfigError(f"{out} already exists; pass --force to replace it")
    try:
        pitch_figures.width_inches(figure_width)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    inputs = {input_key(p, run_root): sha256(p) for p in input_files(run_root, a.video_ids, a.selection_files)}
    tables, documents, facts = assemble(a)
    stated = provenance(a)
    report = pitch_report.report_markdown(a)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(out / name, index=False)
    for name, document in documents.items():
        (out / name).write_text(json.dumps(_plain(document), indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(report)
    shark_files = []
    if a.shark_level is not None:
        (out / SHARK_LEVEL_FOLDER).mkdir()
        for name, table in a.shark_level.items():
            table.to_csv(out / SHARK_LEVEL_FOLDER / f"{name}.csv", index=False)
            shark_files.append(f"{name}.csv")
    skipped: dict[str, str] = {}
    figures = pitch_figures.write_figures(a, out / FIGURES_FOLDER, figure_width, skipped=skipped)
    outputs = sorted([*tables, *documents, "report.md", "manifest.json",
                      *(f"{FIGURES_FOLDER}/{name}" for name in figures),
                      *(f"{SHARK_LEVEL_FOLDER}/{name}" for name in shark_files)])
    manifest = {
        "analysis_id": analysis_id, "run": str(run_root),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "annotation_context": ANNOTATION_CONTEXT, "leakage_rule": LEAKAGE_RULE,
        "video_ids": sorted(set(a.video_ids)) if a.video_ids else None,
        "n_boot": a.n_boot, "seed": a.seed, "n_perm": a.n_perm, "single_draws": a.single_draws, **facts, **stated,
        "shark_level": ({"folder": SHARK_LEVEL_FOLDER, "tables": shark_files,
                         "note": "the archived shark-level design (analysis.archive.shark_level.run), for the F1 "
                                 "paragraph; reported, not tested"} if a.shark_level is not None else None),
        "figures": {"folder": FIGURES_FOLDER, "width": figure_width,
                    "width_inches": pitch_figures.width_inches(figure_width), "png_dpi": pitch_figures.SAVE_DPI,
                    "files": figures},
        "figures_skipped": skipped,
        "inputs": inputs, "rows": {name: int(len(table)) for name, table in tables.items()}, "outputs": outputs,
    }
    (out / "manifest.json").write_text(json.dumps(_plain(manifest), indent=2, allow_nan=False) + "\n")
    return out


def headline(analysis_id: str, facts: dict) -> str:
    c = facts["counts"]
    return (f"pitch-level analysis {analysis_id}: {c['pitches']} pitches ({coding_text(facts['coding'])}); with an "
            f"offer {c['pitches_with_offer']}, without {c['pitches_without_offer']}, unknown {c['pitches_unknown']} "
            f"[{facts['label']}]")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, type=Path, help="run folder, e.g. runs/<run_name>")
    p.add_argument("--analysis-id", required=True, help="name of the folder under <run>/analysis/")
    chosen = p.add_mutually_exclusive_group()
    chosen.add_argument("--video-ids", nargs="+", default=None, help="analyse only these videos")
    chosen.add_argument("--selection", default=None,
                        help="analyse only the videos of this selection file (a name inside the run folder, e.g. "
                             "selection_all100.json, or a path)")
    p.add_argument("--penalty", type=float, default=None,
                   help="fix the ridge penalty of every head; by default it is chosen inside each training fold by "
                        f"cross-validation over its pitches, from {', '.join(f'{g:g}' for g in PENALTY_GRID)}")
    p.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap draws for every interval")
    p.add_argument("--n-perm", type=int, default=N_PERM, help="permutations of the pitch-level test (0 = skip it)")
    p.add_argument("--single-draws", type=int, default=SINGLE_DRAWS,
                   help="bootstrap draws behind the single-feature p-values (FDR-adjusted)")
    p.add_argument("--workers", type=int, default=default_workers(),
                   help="processes for H9's bootstrap draws (default: every core but one); the numbers do not depend "
                        "on it")
    p.add_argument("--skip-h9", action="store_true", help="leave H9 out (a quick run); the report says so")
    p.add_argument("--h9-exchange-outcomes", action="store_true",
                   help="also compute H9's exchange-based outcomes as a sensitivity (h9_exchange_sensitivity.csv); "
                        "not reported, report.md only points at the file")
    p.add_argument("--h9-sensitivity-draws", type=int, default=H9_SENSITIVITY_DRAWS,
                   help=f"bootstrap draws of that sensitivity (default {H9_SENSITIVITY_DRAWS})")
    p.add_argument("--with-shark-level", action="store_true",
                   help="also rerun the archived shark-level design for the F1 paragraph (shark_level/)")
    p.add_argument("--force", action="store_true", help="replace an existing analysis of this id")
    p.add_argument("--figure-width", choices=("single", "double"), default="double",
                   help="width preset of the figures: single = 3.5 in, double = 7 in")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)

    def refuse(message: str) -> int:
        print(f"analyze_panel: {message}", file=sys.stderr)
        return 2

    if not ANALYSIS_ID.fullmatch(args.analysis_id):
        return refuse(f"analysis id {args.analysis_id!r} must be a plain folder name (letters, digits, . _ -)")
    run_root = args.run.resolve()
    if not run_root.is_dir():
        return refuse(f"run folder not found: {args.run}")
    out = run_root / "analysis" / args.analysis_id
    if out.exists() and not args.force:
        return refuse(f"{out} already exists; pass --force to replace it")
    started = time.perf_counter()
    try:
        import matplotlib
        matplotlib.use("Agg")       # files only: no window is ever opened
        analysis = compute(run_root, args.selection or args.video_ids, args.penalty, args.n_boot, n_perm=args.n_perm,
                           with_shark_level=args.with_shark_level, workers=max(1, args.workers),
                           skip_h9=args.skip_h9, single_draws=args.single_draws,
                           h9_exchange_outcomes=args.h9_exchange_outcomes,
                           h9_sensitivity_draws=args.h9_sensitivity_draws)
        out = write_analysis(analysis, args.analysis_id, force=args.force, figure_width=args.figure_width)
    except ConfigError as exc:
        return refuse(str(exc))
    facts = json.loads((out / "manifest.json").read_text())
    print(headline(args.analysis_id, facts))
    for video_id, counts in facts["reviewed_with_open_issues"].items():
        print(f"reviewed with open issues: {video_id} ({counts['reviewed_validation_errors']} validation errors, "
              f"{counts['review_structural_on_decisions']} structural judgments on decisions); read as reviewed, "
              "not excluded")
    for row in facts["verdicts"]:
        print(f"{row['hypothesis']} ({row['comparison']}): {row['verdict']} [AUC route: {row['auc_route']}]")
    if facts["h9"] == H9_SKIPPED:
        print("H9 skipped (--skip-h9)")
    print(f"wrote {len(facts['outputs'])} files to {out} in {time.perf_counter() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
