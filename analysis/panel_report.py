"""Readable tables, captions and plain-language readings for the pitch-level analysis notebook
(`notebooks/stage_B/project_v3_panel_analysis.ipynb`).

The numbers come from `scripts/analysis/analyze_panel.py` (`compute` reads the run and computes everything,
`write_analysis` writes the export the command line writes), which this module calls and never re-implements. It
only arranges the result for a reader: readable headers, people by name with their id, times as M:SS, a caption
that states the sample behind every table, and one-sentence readings generated from the numbers. report.md and the
helpers it shares with the figures are in `analysis.pitch_report`; the figures in `analysis.pitch_figures`. Reads
only; `export` and `save_figures` write the analysis folder and nothing else. Nothing here reads the archived
shark-level design (analysis/archive).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import panel_dataset
from analysis.answer_level import COVARIATES, MODELS as H9_MODELS, POPULATIONS as H9_POPULATIONS
from analysis.panel_bottleneck import (
    DEMONSTRATION, FDR_LEVEL, LADDER_NOT_FITTED, LADDER_NOT_FITTED_NOTE, N_BOOT, NOTE_COUNT, SEED, SINGLE_DRAWS,
    STATUS_DESCRIPTIVE, STATUS_EXPLORATORY, STATUS_PRIMARY, editable_features, fold_detail, prediction_breakdown,
)
from analysis.panel_dataset import (
    ALL_QUESTIONS, BOOKS_BUNDLE, HARD_QUESTIONS, MODEL_SOURCES, NEXT_MOVES, PANEL_MOVES, REVIEWED_SOURCES, SOURCE_OPEN,
    SOURCE_PRIMARY, SOURCE_REPLICATE, SUBSTANCE_LEVELS, SUPPORT_LEVELS, PanelData, panel_next_move, substance_level,
    support_level,
)
from analysis.panel_explorer import (
    ASKER_MOVE_WORDS, PANEL_MOVE_WORDS, REPO, agreement_reports, agreement_table, clock, clock_range, followup_rows,
    function_agreement_table, instrument, ranking_stability_reading, review_summary,
)
from analysis.panel_features import FEATURES, describe_row, feature_dictionary, history_table
from analysis.pitch_report import (  # noqa: F401 (re-exported: the notebook reads them here)
    BOOT_SKILL_HEADER, METHOD_LIMITS, P_HEADER, PERM_AUC_HEADER, PERM_LOSS_HEADER, Q_HEADER, STARS_NOTE,
    CONTEXT_SENTENCE, H9_EXPLORATORY_TEXT, _limit_bullets, _sensitivity_rows, _verdict_lines, analyze_panel,
    h9_claim_lines, h9_unclaimed_lines, labelled, p_phrase, p_text, plan_limits, stars,
)

SOURCE_WORDS = {SOURCE_PRIMARY: "model-coded (primary pass, no human review)",
                SOURCE_REPLICATE: "model-coded (replicate pass: the primary failed validation; no human review)",
                "reviewed": "human-reviewed", SOURCE_OPEN: "human-reviewed, with issues still open"}
RESPONSIVENESS_WORDS = {1.0: "addresses the question", 0.5: "partly addresses it", 0.0: "deflects"}
YES_NO = {1.0: "yes", 0.0: "no"}
SKILL = "Skill over base rate (%) (higher is better; 0 = offer rate)"
METRIC_WORDS = {"log_loss": "log loss", "brier": "Brier score", "auc": "AUC", "skill": "skill over the base rate"}
BETTER_WHEN_LOWER = ("log_loss", "brier")
EXPORT_NOTE = "EXPORT is False: nothing was written."


def label(a) -> str:
    """The label every result of this analysis carries: DEMONSTRATION ONLY under 20 pitches."""
    return "DEMONSTRATION ONLY" if a.result["label"] == DEMONSTRATION else a.result["label"]


# ---- computing and exporting: analyze_panel's own functions ------------------------------------

def compute(run_root, video_ids=None, penalty="inner_cv", bootstrap_reps: int = N_BOOT, seed: int = SEED,
            permutations: int = 500, single_draws: int = SINGLE_DRAWS, workers: int | None = 1, run_h9: bool = True,
            h9_exchange_outcomes: bool = False, h9_sensitivity_draws: int | None = None,
            with_shark_level: bool = False):
    """Everything the notebook shows: `analyze_panel.compute` with the notebook's settings (the function the command
    line runs). `penalty` "inner_cv" (or None) chooses the ridge penalty inside each training fold; a positive number
    fixes it. `workers` None is the command line's default (every core but one); no number depends on it. `run_h9`
    False is the command line's `--skip-h9` (and then no exchange-based sensitivity either);
    `h9_sensitivity_draws` None is the command line's default."""
    if penalty is None or penalty == "inner_cv":
        chosen = None
    elif isinstance(penalty, (int, float)) and not isinstance(penalty, bool) and penalty > 0:
        chosen = float(penalty)
    else:
        raise ValueError(f'PENALTY must be "inner_cv" or a positive number, not {penalty!r}')
    ap = analyze_panel()
    workers = ap.default_workers() if workers is None else max(1, int(workers))
    draws = ap.H9_SENSITIVITY_DRAWS if h9_sensitivity_draws is None else int(h9_sensitivity_draws)
    return ap.compute(run_root, video_ids, chosen, int(bootstrap_reps), int(seed), n_perm=int(permutations),
                      with_shark_level=bool(with_shark_level), workers=workers, skip_h9=not run_h9,
                      single_draws=int(single_draws), h9_exchange_outcomes=bool(h9_exchange_outcomes) and bool(run_h9),
                      h9_sensitivity_draws=draws)


def export(a, analysis_id: str, figure_width="double") -> Path:
    """The command line's export of this analysis (`analyze_panel.write_analysis`, replacing an earlier export of
    the same id), figures included."""
    return analyze_panel().write_analysis(a, analysis_id, force=True, figure_width=figure_width)


def export_note(out: Path) -> str:
    return f"Saved {sum(1 for p in Path(out).rglob('*') if p.is_file())} files to {Path(out).resolve()}"


# ---- names ----------------------------------------------------------------------------------

def names(data: PanelData) -> dict[tuple[str, str], str]:
    """(video id, actor id) -> "Name (id)" for everyone in the run."""
    return {(r.video_id, r.actor_id): f"{r.label} ({r.actor_id})" if isinstance(r.label, str) and r.label else r.actor_id
            for r in data.actors.itertuples()}


def default_video(a, video_id: str | None = None) -> str:
    """The pitch to inspect: the one given, else the first pitch with a known outcome."""
    videos = list(a.outcome.video_id)
    if video_id is None:
        known = a.outcome[a.outcome.any_offer.notna()]
        return str((known if len(known) else a.outcome).video_id.iloc[0])
    if video_id not in videos:
        raise ValueError(f"INSPECT_VIDEO {video_id!r} is not in this analysis. Videos: {', '.join(sorted(videos))}.")
    return str(video_id)


def turn_summaries(run_root, video_id: str) -> dict[int, str]:
    """turn id -> the annotator's one-line summary, from the record the analysis reads (for display only; no
    feature ever reads this text)."""
    from pipeline.stage_b.panel_pipeline import load_best_record

    result, _ = load_best_record(Path(run_root) / video_id)
    return {t["turn_id"]: t["summary"] for t in result["record"]["turns"]}


# ---- 1. provenance --------------------------------------------------------------------------

def provenance_table(a) -> pd.DataFrame:
    """Per video: who coded what is analysed, with which instrument, and what a review left open."""
    data = a.data
    return pd.DataFrame([{
        "Video": v.video_id, "Coded by": SOURCE_WORDS.get(v.source, v.source), "Instrument release": v.release_id,
        "Codebooks read from": ("the run's own copy" if data.books_source.get(v.video_id) == BOOKS_BUNDLE
                                else "the live release (the run kept no copy)"),
        "Turns": int(v.n_turns), "Open validation errors after review": int(v.reviewed_validation_errors),
        "Open review judgments on decisions": int(v.review_structural_on_decisions)} for v in data.videos.itertuples()])


def status_text(a) -> str:
    """Model-coded versus human-reviewed, and the videos whose review left something open, in words."""
    videos = a.data.videos
    reviewed = int(videos.source.isin(REVIEWED_SOURCES).sum())
    still_open = list(videos.video_id[videos.source == SOURCE_OPEN])
    text = (f"{len(videos)} videos: {int(videos.source.isin(MODEL_SOURCES).sum())} model-coded only, "
            f"{reviewed} human-reviewed.")
    from_replicate = int((videos.source == SOURCE_REPLICATE).sum())
    if from_replicate:
        text += (f" For {from_replicate} of them the replicate pass is read, because the primary pass has "
                 "validation errors and the replicate has fewer.")
    if not reviewed:
        text += " No video has been human-reviewed, so every number below describes what the model coded."
    text += (f" Reviewed with issues still open: {', '.join(still_open)} (read all the same, not excluded)."
             if still_open else " No reviewed video has open issues.")
    return text


def settings_table(a) -> pd.DataFrame:
    """The settings of this analysis and where its pre-registered texts come from (report.md section 9)."""
    stated = analyze_panel().provenance(a)
    rows = [("Instrument release(s)", ", ".join(stated["release_ids"]) or "unknown"),
            ("Code (git commit)", stated["git_commit"]),
            ("Outcome", "any_offer: did any investor make or join an offer"),
            ("Penalty", stated["penalty"]), ("Bootstrap draws", a.n_boot), ("Permutations", a.n_perm),
            ("Draws behind the single-feature p-values", a.single_draws), ("Random seed", a.seed),
            ("Processes for H9's draws", f"{a.workers} (changes no number)"), ("H9", stated["h9"]),
            ("H9 exchange-based sensitivity",
             "computed, not reported" if a.h9_sensitivity is not None else "not computed"),
            ("Shark-level archive run (F1)", "yes" if a.with_shark_level else "no"),
            ("Stage A ranking (the rank score C2 reads)", a.stage_a_ranking),
            ("Pre-registered test quoted from", f"{stated['spec']['path']}, section 6"),
            ("Limits quoted from", f"{stated['limits']['path']}, section 11")]
    return pd.DataFrame(rows, columns=["Setting", "Value"]).astype({"Value": str})


# ---- 2. sample and outcomes -----------------------------------------------------------------

def sample_caption(a) -> str:
    """The sample of the whole analysis in one line."""
    o = a.outcome.any_offer
    return (f"n pitches = {len(a.outcome)} ({int((o == 1).sum())} with an offer, {int((o == 0).sum())} without, "
            f"{int(o.isna().sum())} unknown)  [{label(a)}]")


def outcome_table(a) -> pd.DataFrame:
    """One row per pitch: the decisions in time order, the outcome read from them, why, and where the cut falls."""
    data, who = a.data, names(a.data)
    decisions = data.decisions[data.decisions.role == "investor"].sort_values(["video_id", "pos", "decision_id"],
                                                                            kind="stable")
    rows = []
    for r in a.outcome.itertuples():
        own = decisions[decisions.video_id == r.video_id]
        if r.any_offer_source != "record":
            why = "set by a human reviewer"
        elif r.any_offer == 1:
            why = "an investor made, joined, revised or withdrew an offer"
        elif r.any_offer == 0:
            why = "every investor declared out and none offered"
        else:
            why = "no offer, and at least one investor has neither an offer nor an exit on record"
        first = own[own.pos == r.cut_pos]
        cut = (f"{first.kind.iloc[0]} by {who.get((r.video_id, first.actor.iloc[0]), first.actor.iloc[0])} at "
               f"{clock(first.time_s.iloc[0])}" if len(first) else "no decision: the whole record")
        rows.append({"Video": r.video_id,
                     "Decisions, in order": "; ".join(f"{who.get((r.video_id, d.actor), d.actor)} {d.kind} at "
                                                      f"{clock(d.time_s)}" for d in own.itertuples()) or "none",
                     "Any offer": {1.0: "yes", 0.0: "no"}.get(r.any_offer, "unknown"), "Why": why,
                     "History cut at (the first decision of anyone)": cut})
    return pd.DataFrame(rows)


def unknown_outcomes(a) -> str:
    """Which outcomes are unknown and why, in words."""
    unknown = a.outcome[a.outcome.any_offer.isna()]
    if unknown.empty:
        return "No unknown outcomes: every pitch either got an offer or every investor declared out."
    return (f"{len(unknown)} pitches have an unknown outcome and are left out of the models (never counted as 'no "
            f"offer'): {', '.join(unknown.video_id)}. Rule: {analyze_panel().UNKNOWN_RULE}.")


# ---- 3. the cut -----------------------------------------------------------------------------

def history_view(a, video_id: str) -> pd.DataFrame:
    """Every turn of the pitch with whether its features read it, and why not."""
    who = names(a.data)
    table = history_table(a.data, video_id)
    summaries = turn_summaries(a.run_root, video_id)
    return pd.DataFrame([{
        "Position": int(t.pos), "Turn": int(t.turn_id), "Clock": clock_range(t.start_s, t.end_s),
        "Speaker": who.get((video_id, t.speaker), t.speaker), "Function": t.function,
        "Exchange kind": t.exchange_kind or "", "Feeds the features": "yes" if t.used else "NO",
        "Why excluded": t.why_excluded, "Summary": summaries.get(t.turn_id, "")} for t in table.itertuples()])


def history_text(a, video_id: str) -> str:
    """The cut in words: where it falls, what decision sets it, how many turns are read and left out."""
    who = names(a.data)
    table = history_table(a.data, video_id)
    cut = int(table.cut_pos.iloc[0])
    decisions = a.data.decisions[(a.data.decisions.video_id == video_id) & (a.data.decisions.role == "investor")]
    at_cut = decisions[decisions.pos == cut].sort_values("decision_id")
    if len(at_cut):
        d = at_cut.iloc[0]
        where = (f"the first decision of any investor ({d.kind} by {who.get((video_id, d.actor), d.actor)}, at "
                 f"{clock(d.time_s)}), which is turn position {cut} of {len(table)}")
    else:
        where = f"the end of the record (no decision found), turn position {cut} of {len(table)}"
    left_out = table[~table.used].why_excluded.value_counts()
    lines = [f"History cut for the pitch {video_id}: {where}.",
             f"Turns that feed the features: {int(table.used.sum())}. Turns left out: {int((~table.used).sum())}."]
    lines += [f"  - {n} × {reason}" for reason, n in left_out.items()]
    lines.append("Check: no turn at or after the cut, and no turn that states a decision, is marked 'yes'.")
    return "\n".join(lines)


# ---- 4. features ----------------------------------------------------------------------------

STATUSES = (STATUS_PRIMARY, STATUS_DESCRIPTIVE, STATUS_EXPLORATORY)
STATUS_NOTE = (f"Status: '{STATUS_PRIMARY}' = fitted by the pre-registered ladder (C1 to C5 with C2p and C3q); "
               f"'{STATUS_DESCRIPTIVE}' = in the feature table but never fitted ({LADDER_NOT_FITTED_NOTE}); "
               f"'{STATUS_EXPLORATORY}' = the Stage A items other than the primary one, read by the all-Stage-A "
               "sensitivity only.")


def dictionary_table(status: str | None = None) -> pd.DataFrame:
    """The pitch feature dictionary with readable headers; key columns are listed last. `status` keeps only the
    features with that status ("primary", "descriptive" or "exploratory")."""
    if status is not None and status not in STATUSES:
        raise ValueError(f"status must be None or one of {', '.join(repr(s) for s in STATUSES)}, not {status!r}")
    table = feature_dictionary().rename(columns={
        "name": "Feature (column name)", "family": "Family", "label": "Plain-language label", "unit": "Unit",
        "kind": "Kind", "status": "Status", "description": "Description"})
    if status is not None:
        table = table[table["Status"] == status]
    table["Family"] = table["Family"].fillna("(key, never a predictor)")
    table["Status"] = table["Status"].fillna("(key)")
    return (table.drop(columns=["Unit"]).sort_values("Family", key=lambda s: s.str.startswith("("), kind="stable")
            .reset_index(drop=True))


def status_counts() -> str:
    """How many features carry each status, with the note that explains the three words."""
    counts = feature_dictionary().status.value_counts()
    return (", ".join(f"{int(counts.get(s, 0))} {s}" for s in STATUSES) + " columns (companion columns counted one "
            "by one).\n" + STATUS_NOTE)


def feature_row(a, video_id: str) -> pd.DataFrame:
    """One pitch row of the feature table as "feature — label — value — denominator"."""
    table = describe_row(a.features, video_id)
    table["value"] = [("missing (nothing observed)" if not isinstance(v, str) and (v is None or v != v)
                       else round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for v in table["value"]]
    table["denominator"] = [("" if d is None else f"of {d} observed") for d in table["denominator"]]
    return table.rename(columns={"feature": "Feature", "family": "Family", "label": "Label", "value": "Value",
                                 "denominator": "Denominator", "description": "Description"})[
        ["Feature", "Label", "Value", "Denominator", "Family", "Description"]]


def readable_features(a) -> pd.DataFrame:
    """The feature table with the feature labels as headers."""
    return a.features.rename(columns={c: FEATURES[c]["label"] for c in a.features.columns if c in FEATURES})


# ---- 5. descriptives ------------------------------------------------------------------------

def descriptive_caption(a) -> str:
    """What the descriptive tables count over."""
    turns = a.data.turns
    return (f"Whole encounter as annotated. n pitches = {len(a.data.videos)}, n turns = {len(turns)}, "
            f"n questions = {int((turns.function == 'question').sum())}, n answers = "
            f"{int((turns.function == 'answer').sum())}, n exchanges = {len(a.data.exchanges)}  [{label(a)}; "
            "model-coded unless section 1 says otherwise; descriptive, not predictors]")


def _count_table(values: pd.Series, header: str, order: list | None = None) -> pd.DataFrame:
    counts = values.dropna().value_counts()
    if order:
        counts = counts.reindex([v for v in order if v in counts.index] + [v for v in counts.index if v not in order])
    total = int(counts.sum())
    return pd.DataFrame({header: counts.index, "Count": counts.to_numpy(dtype=int),
                         "Share": [round(c / total, 2) if total else None for c in counts]})


def _r(value, digits: int = 2):
    return None if value is None or value != value else round(float(value), digits)


def _rating(values: pd.Series) -> pd.Series:
    return values.map(lambda v: None if v != v else int(v))


def _summary_rows(frame: pd.DataFrame, header: str, counted: str) -> pd.DataFrame:
    """One row per code of a long table (code, value): how many pitches have a value, and its spread."""
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        header: code, counted: int(rows.value.notna().sum()), "Missing": int(rows.value.isna().sum()),
        "Mean": _r(rows.value.mean()), "SD": _r(rows.value.std(ddof=0)), "Min": _r(rows.value.min()),
        "Max": _r(rows.value.max())} for code, rows in frame.groupby("code", sort=False)])


def descriptive_tables(a) -> dict[str, pd.DataFrame]:
    """The measurement descriptives of the analysed videos, each as a small table. They cover the whole encounter
    as annotated (decisions and negotiation included) and are not the ladder's predictors."""
    data = a.data
    turns = data.turns
    questions, answers = turns[turns.function == "question"], turns[turns.function == "answer"]
    substantive = data.exchanges[data.exchanges.kind == "substantive"]
    out: dict[str, pd.DataFrame] = {}
    out["question_concepts"] = _count_table(questions.concept, "Question concept")
    out["question_frames"] = _count_table(questions.frame, "Question frame (derived)")
    out["answer_concepts"] = _count_table(answers.concept, "Answer concept")
    out["answer_frames"] = _count_table(answers.frame, "Answer frame (derived)")
    h5 = a.h5
    out["frame_match"] = (h5.pivot_table(index="question_frame", columns="answer_frame", values="n", aggfunc="sum",
                                         fill_value=0).rename_axis("question frame ↓ / answer frame →")
                          if len(h5) else pd.DataFrame())
    out["frame_match_states"] = _count_table(answers.frame_match, "Does the answer's frame match the question's?")
    out["answer_directness"] = _count_table(answers["RS-RESP"].map(RESPONSIVENESS_WORDS), "Answer directness",
                                            list(RESPONSIVENESS_WORDS.values()))
    out["answer_support"] = _count_table(answers["RS-EVID"].map(YES_NO), "Answer gives checkable support", ["yes", "no"])
    out["answer_uncertainty"] = _count_table(answers["RS-UNCERT"].map(YES_NO), "Answer admits uncertainty", ["yes", "no"])
    for code, key, words in (("PREP-SUBS", "answer_substance", "Answer substance"),
                             ("PREP-COH", "answer_coherence", "Answer coherence"),
                             ("PREP-CTX", "answer_context", "Answer context (Gemini only, not human-coded)")):
        out[key] = _count_table(_rating(answers[code]), f"{words} (1 low to 5 high)", [1, 2, 3, 4, 5])
    out["question_severity"] = _count_table(_rating(questions["PR-SEV"]), "Question severity (1 to 3)", [1, 2, 3])
    out["exchanges_by_kind"] = _count_table(data.exchanges.kind, "Exchange kind")
    out["exchanges_by_topic"] = _count_table(substantive.topic, "Topic of substantive exchanges")
    out["turn_functions"] = _count_table(turns.function, "Turn function")
    per_pitch = []
    for v in data.videos.itertuples():
        t = turns[turns.video_id == v.video_id]
        demo = t[t.function == "demonstrates"]
        per_pitch.append({"Video": v.video_id, "Social turns": int((t.function == "social").sum()),
                          "Demonstration turns": len(demo),
                          "Demonstration seconds": round(float((demo.end_s - demo.start_s).clip(lower=0).sum())),
                          "Prepared presentation seconds": round(float(v.prepared_seconds))})
    out["social_and_demonstration"] = pd.DataFrame(per_pitch)
    out["prepared_ratings"] = _summary_rows(data.prepared, "Prepared presentation code", "Pitches rated")
    videos = data.videos
    out["ask"] = pd.DataFrame([{
        "Pitches": len(videos), "Ask stated": int(videos.ask_amount_usd.notna().sum()),
        "Median amount (USD)": _r(videos.ask_amount_usd.median(), 0), "Smallest": _r(videos.ask_amount_usd.min(), 0),
        "Largest": _r(videos.ask_amount_usd.max(), 0), "Median equity (%)": _r(videos.ask_equity_pct.median(), 1)}])
    out["stage_a_ratings"] = _summary_rows(data.stage_a[data.stage_a.source == "rating"], "Stage A item",
                                           "Pitches rated")
    out["stage_a_ranks"] = _summary_rows(data.stage_a[data.stage_a.source == "rank_pl"], "Stage A item",
                                         "Pitches ranked")
    out["team_gender"] = _count_table(videos.founder_gender, "Presenting team gender")
    for column, header in (("primary_industry", "Primary industry"), ("product_form", "Product form"),
                           ("customer_type", "Customer type")):
        out[column] = _count_table(videos[column], header)
    out["concepts_by_team_gender"] = concepts_by_gender(data)
    return out


def concepts_by_gender(data: PanelData) -> pd.DataFrame:
    """Question concepts (rows) by the presenting team's gender (columns): counts, with the number of pitches and
    questions behind each column in its header. Descriptive only."""
    questions = data.turns[data.turns.function == "question"].merge(
        data.videos[["video_id", "founder_gender"]], on="video_id")
    if questions.empty:
        return pd.DataFrame()
    table = pd.crosstab(questions.concept.fillna("(none)"), questions.founder_gender.fillna("(unknown)"))
    pitches = data.videos.founder_gender.fillna("(unknown)").value_counts()
    table.columns = [f"{g} (n = {int(pitches.get(g, 0))} pitches, {int(table[g].sum())} questions)" for g in table.columns]
    table.index.name = "Question concept"
    return table.loc[table.sum(axis=1).sort_values(ascending=False, kind="stable").index]


def ranking_agreement_caption(a) -> str:
    table = a.ranking_agreement
    n = int(table["Videos ranked in two batches"].max()) if len(table) else 0
    return (f"n videos ranked in two batches = {n}: ALL ranked videos of the run, not only the {len(a.data.videos)} "
            f"analysed here, because a standing is relative to the other clips of a batch. 95% intervals resample "
            f"videos ({a.n_boot} draws). Label: {analyze_panel().SELF_CONSISTENCY_LABEL}.")


def ranking_agreement_reading(a) -> str:
    return ranking_stability_reading(a.ranking_agreement)


# ---- H4: what the panel does after an answer -------------------------------------------------

H4_WHO = {"asker": ("The asker's next move", ASKER_MOVE_WORDS, NEXT_MOVES),
          "panel": ("The panel's next move (first investor reaction by anyone)", PANEL_MOVE_WORDS, PANEL_MOVES)}
H4_BY = {"substance": "Answer substance (1 low to 5 high)", "support": "Answer gives checkable support",
         "video": "Video"}
H4_QUESTIONS = {ALL_QUESTIONS: "all questions", HARD_QUESTIONS: "hard questions (severity 2 or 3)"}
H4_EMPTY = "No answer to an investor question falls in this selection."


def h4_tables(a) -> dict[str, pd.DataFrame]:
    """The H4 count tables of this analysis (analysis.panel_dataset.h4_tables on the whole encounter)."""
    return panel_dataset.h4_tables(a.followups)


def h4_caption(a) -> str:
    rows = a.followups
    return (f"Whole encounter as annotated, no history cut. n pitches = {len(a.data.videos)}, n answers to an investor "
            f"question = {len(rows)} ({int(rows.hard_question.sum())} of them to a hard question, severity 2 or 3)  "
            f"[{label(a)}; descriptive counts, model-coded unless section 1 says otherwise]")


def _h4_name(name: str) -> tuple[str, str | None]:
    who, _, by = name.partition("_by_")
    if who not in H4_WHO or (by and by not in H4_BY):
        raise ValueError(f"unknown H4 table {name!r}. Tables: {', '.join(panel_dataset.H4_TABLES)}.")
    return who, by or None


def _share(count, total):
    return round(count / total, 2) if total else None


def h4_view(tables: dict[str, pd.DataFrame], name: str, questions: str = ALL_QUESTIONS) -> pd.DataFrame:
    """One H4 table with readable headers: "asker" and "panel" one row per next move, for all and for hard questions
    with a total row; the `_by_` tables one row per level (substance, support or video) for `questions`."""
    who, by = _h4_name(name)
    header, words, moves = H4_WHO[who]
    table = tables[name]
    if by is None:
        every, hard = int(table[ALL_QUESTIONS].sum()), int(table[HARD_QUESTIONS].sum())
        out = pd.DataFrame({
            header: [words[m] for m in table.next_move], "All questions": table[ALL_QUESTIONS].astype(int),
            "Share": [_share(c, every) for c in table[ALL_QUESTIONS]],
            "Hard questions (severity 2 or 3)": table[HARD_QUESTIONS].astype(int),
            "Share of hard": [_share(c, hard) for c in table[HARD_QUESTIONS]]})
        out.loc[len(out)] = ["Total", every, 1.0 if every else None, hard, 1.0 if hard else None]
        return out
    if questions not in H4_QUESTIONS:
        raise ValueError(f"questions must be one of {', '.join(repr(q) for q in H4_QUESTIONS)}, not {questions!r}")
    column = "video_id" if by == "video" else by
    part = table[(table.questions == questions) & (table.n > 0)]
    if part.empty:
        return pd.DataFrame()
    out = pd.DataFrame({H4_BY[by]: [str(v) if by == "video" else str(v).replace("_", " ") for v in part[column]],
                        **{words[m]: part[m].astype(int).to_numpy() for m in moves},
                        "Answers": part.n.astype(int).to_numpy()})
    out.loc[len(out)] = ["Total", *(int(part[m].sum()) for m in moves), int(part.n.sum())]
    return out


def h4_heading(name: str, questions: str = ALL_QUESTIONS) -> str:
    who, by = _h4_name(name)
    split = f" by {H4_BY[by][0].lower() + H4_BY[by][1:]}" if by else ""
    return (f"{H4_WHO[who][0]}{split}, after answers to {H4_QUESTIONS[questions]}. Unit: one answer and an investor "
            "question it responds to. Counts.")


def h4_reading(tables: dict[str, pd.DataFrame]) -> str:
    """The asker's and the panel's next move in a sentence for all and for hard questions, from the counts."""
    asker, panel = tables["asker"].set_index("next_move"), tables["panel"].set_index("next_move")

    def sentence(column: str, what: str) -> str:
        a, p = asker[column], panel[column]
        total = int(a.sum())
        if not total:
            return f"No answer to {what} in this selection."
        own = int(a.positive_receipt + a.follow_up_question + a.challenge + a.other_by_asker)
        return (f"After the {total} answers to {what}: the asker reacted first {own} times (accepted "
                f"{int(a.positive_receipt)}, asked again {int(a.follow_up_question)}, challenged {int(a.challenge)}, "
                f"something else {int(a.other_by_asker)}); another investor spoke first {int(a.other_investor)} times; "
                f"{int(a.no_reaction)} times no investor reacted (none spoke again, or the next one stated or "
                "negotiated a decision). Whoever spoke, the "
                f"panel's first reaction: receipt {int(p.receipt)}, question {int(p.question)}, challenge "
                f"{int(p.challenge)}, something else {int(p.other)}.")
    return "\n".join([sentence(ALL_QUESTIONS, "an investor question"), sentence(HARD_QUESTIONS, "a hard question"),
                      "These are counts of what was annotated. They do not say an answer caused the reaction."])


def h4_shares_view(a) -> pd.DataFrame:
    """The table behind figure 7 with readable headers (shares in percent, differences in points)."""
    t = a.h4_shares
    if t is None or t.empty:
        return pd.DataFrame()

    def pct(share, low, high):
        return ["" if s != s else f"{100 * s:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]" for s, lo, hi in zip(share, low, high)]
    return pd.DataFrame({
        "Questions": t.questions.map(H4_QUESTIONS), "Panel's first reaction": t.move,
        "Answers with support": t.n_yes, "of them followed by it": t.count_yes,
        "Share, with support [95% interval]": pct(t.share_yes, t.yes_low, t.yes_high),
        "Answers without support": t.n_no, "of them followed by it ": t.count_no,
        "Share, without support [95% interval]": pct(t.share_no, t.no_low, t.no_high),
        "Difference in points, with minus without [95% interval]": [
            "" if d != d else f"{100 * d:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]"
            for d, lo, hi in zip(t.difference, t.diff_low, t.diff_high)],
        "p (difference, bootstrap)": [p_text(p) for p in t.p_value]})


def _h4_choice(setting: str, value, valid) -> None:
    if value is not None and value not in valid:
        raise ValueError(f"{setting} must be None or one of {', '.join(repr(v) for v in valid)}, not {value!r}")


def h4_turns(a, who: str = "asker", move: str | None = None, hard_only: bool = False, substance=None,
             support: str | None = None, video_id: str | None = None) -> pd.DataFrame:
    """The actual turns behind one cell of an H4 table: question, answer and the next investor turn. `who` is
    "asker" or "panel" and decides which next move `move` names (None keeps every move); `hard_only` keeps answers
    to hard questions; `substance` is 1 to 5 or "not_rated"; `support` is "yes", "no" or "not_rated"; `video_id`
    keeps one video. Every setting left at None keeps all rows."""
    _h4_choice("H4_WHO", who, tuple(H4_WHO))
    _h4_choice("H4_MOVE", move, H4_WHO[who or "asker"][2])
    substance = None if substance is None else str(substance)
    _h4_choice("H4_SUBSTANCE", substance, SUBSTANCE_LEVELS)
    _h4_choice("H4_SUPPORT", support, SUPPORT_LEVELS)
    rows = a.followups
    _h4_choice("H4_VIDEO", video_id, tuple(sorted(set(a.data.videos.video_id))))
    keep = pd.Series(True, index=rows.index)
    if move is not None:
        keep &= (panel_next_move(rows) if who == "panel" else rows.next_move) == move
    if hard_only:
        keep &= rows.hard_question
    if substance is not None:
        keep &= rows.subs.map(substance_level) == substance
    if support is not None:
        keep &= rows.evid.map(support_level) == support
    if video_id is not None:
        keep &= rows.video_id == video_id
    rows = rows[keep]
    summaries = {v: turn_summaries(a.run_root, v) for v in sorted(set(rows.video_id))}
    return followup_rows(rows, summaries, names(a.data))


def h4_selection(a, who: str = "asker", move: str | None = None, hard_only: bool = False, substance=None,
                 support: str | None = None, video_id: str | None = None) -> str:
    """The selection of `h4_turns` in words, with how many rows it holds."""
    n = len(h4_turns(a, who, move, hard_only, substance, support, video_id))
    parts = [f"{'the panel' if who == 'panel' else 'the asker'}'s next move: "
             + (H4_WHO[who][1][move] if move else "any")]
    parts.append("hard questions only" if hard_only else "all questions")
    if substance is not None:
        parts.append(f"answer substance {str(substance).replace('_', ' ')}")
    if support is not None:
        parts.append(f"checkable support: {support.replace('_', ' ')}")
    parts.append(video_id if video_id else "all videos")
    return f"{n} answers match ({'; '.join(parts)})."


# ---- 6. self-consistency and human agreement --------------------------------------------------

def self_consistency_caption(a) -> str:
    n = len(agreement_reports(a.run_root, list(a.data.videos.video_id)))
    return (f"n videos with two passes compared = {n} of {len(a.data.videos)}. "
            f"Label: {analyze_panel().SELF_CONSISTENCY_LABEL}.")


def self_consistency_table(a) -> pd.DataFrame:
    """The two model passes compared, pooled over the videos of this analysis."""
    reports = agreement_reports(a.run_root, list(a.data.videos.video_id))
    if not reports:
        return pd.DataFrame()
    return agreement_table(reports, instrument(Path(a.run_root) / next(iter(reports))).books)


def self_consistency_functions(a) -> pd.DataFrame:
    """Per turn function: turns per pass, turns paired with the same function, and 2 x matched / (primary +
    replicate)."""
    return function_agreement_table(agreement_reports(a.run_root, list(a.data.videos.video_id)))


def human_agreement_text(a) -> str:
    """The pooled human-review summary, or a plain statement that there is none yet."""
    summary = review_summary(a.run_root, list(a.data.videos.video_id))
    if summary is None:
        return ("Human agreement has not been collected for this run: no filled review sheet has been imported. Until "
                "it has, nothing here says the codes are valid; the human validation sample (ten videos) is coded and "
                "reported separately (review_panel.py summary --paper-table; the run explorer shows its table).")
    return summary


# ---- 7. the ladder --------------------------------------------------------------------------

def ladder_caption(a) -> str:
    r = a.result
    return (f"n pitches = {r['n_groups']}, n pitches with an offer = {r['n_events']}  [{label(a)}; one row per pitch, "
            "leave-one-pitch-out]")


def ladder_guide(a) -> pd.DataFrame:
    """What each model of the ladder adds, the families it reads and the columns it fits."""
    from analysis.pitch_figures import LADDER_WORDS, ladder_label, rating_fallback

    design = a.result.get("design_columns") or {}
    fallback = rating_fallback(a)
    return pd.DataFrame([{
        "Model": name, "Adds": ladder_label(name, fallback=fallback).split("  ", 1)[-1] if name in LADDER_WORDS else "",
        "Reads (families or columns)": ", ".join(spec) or "nothing: the offer rate of the training pitches",
        "Columns fitted": len(design.get(name, [])),
        "Design columns": ", ".join(map(labelled, design.get(name, []))) or "(base rate)"}
        for name, spec in a.models.items()])


def not_fitted_note() -> str:
    return (f"{CONTEXT_SENTENCE} Never fitted ({len(LADDER_NOT_FITTED)} columns of the feature table): {LADDER_NOT_FITTED_NOTE}. A column "
            "that is constant within a training fold is dropped in that fold.")


def metrics_table(result: dict) -> pd.DataFrame:
    """The headline metrics per model (pitches weighted equally), with the penalties each used."""
    if result["status"] != "estimated":
        return pd.DataFrame()
    m = result["metrics"]

    def column(name):
        return m[name] if name in m else pd.Series(np.nan, index=m.index)

    def interval(low, high, digits):
        return ["" if lo != lo else f"[{lo:+.{digits}f}, {hi:+.{digits}f}]" for lo, hi in zip(low, high)]
    return pd.DataFrame({
        "Model": m.model, SKILL: m.skill_pct.round(1), "Skill 95% interval": interval(m.skill_low, m.skill_high, 1),
        BOOT_SKILL_HEADER: [p_text(p) for p in column("skill_p_value")],
        PERM_LOSS_HEADER: [p_text(p) for p in column("perm_p_log_loss")],
        "AUC (higher is better)": m.auc.round(3), "AUC 95% interval": interval(m.auc_low, m.auc_high, 3),
        PERM_AUC_HEADER: [p_text(p) for p in column("perm_p_auc")],
        "Near-constant predictions (diagnostic)": ["" if ok else "yes" for ok in m.auc_meaningful.astype(bool)],
        "Log loss (lower is better)": m.log_loss.round(3), "Brier (lower is better)": m.brier.round(3),
        "Mean predicted offer rate": m.mean_pred.round(2), "Actual offer rate": m.base_rate.round(2),
        "Penalty chosen (value × folds)": m.penalty})


def how_to_read(a) -> str:
    """How to read the ladder's metrics, with this evaluation's own numbers."""
    result = a.result
    if result["status"] != "estimated":
        return f"Nothing to read: {result['status']}."
    base, m = result["base"], result["metrics"]
    rate, loss, coin = base["predicted_rate"], base["log_loss"], base["coin_log_loss"]
    near = list(m.model[~m.auc_meaningful.astype(bool) & (m.model != "C0")])
    below = ", ".join(f"{r.model} {r.auc:.3f}" for r in m.itertuples() if r.model != "C0" and r.auc < 0.5)
    return "\n".join([
        "HOW TO READ THESE NUMBERS",
        f"1. C0 uses no features. For every held-out pitch it predicts the offer rate of the other pitches, about "
        f"{rate:.0%} here. Its log loss is {loss:.3f} (a perfect forecast scores 0; always saying 50% scores "
        f"{coin:.3f}). It is the bar to beat.",
        f"2. '{SKILL}' puts log loss on that scale: 100 x (1 - log loss / {loss:.3f}). 0 = no better than the offer "
        "rate; negative = worse; 100 = perfect. Read it first.",
        "3. AUC ranks the base-rate-adjusted held-out score (the prediction measured from the offer rate of its own "
        "training fold), which reduces the leave-one-pitch-out artefact that made near-constant models rank pitches "
        "backwards but does not remove it: a residual bias against weak models remains, so a weak model can still "
        "score below 0.5" + (f" (here {below})" if below else "") + ". C0 scores 0 everywhere and its AUC is 0.5 by "
        "construction. These AUCs are not comparable with those of the v3-dev8 analyses.",
        "4. 'Near-constant predictions' marks a model whose held-out predictions stay within about 3 points of the "
        "offer rate (" + (", ".join(near) or "none here") + "): a diagnostic, not a filter; its AUC still counts.",
        "5. The penalty is chosen inside each training fold from 1, 10, 100, 1000 (larger = predictions held closer "
        "to the offer rate). The largest value in most folds means trusting the features made held-out predictions "
        "worse: no signal detectable at this sample size, which is not the same as no signal.",
        "6. The permutation p asks whether a model beats the offer rate when whole pitches' outcomes are shuffled "
        "(the one to read); the bootstrap p beside the skill interval describes the same estimate under resampling.",
    ])


def best_reading(a) -> str:
    """Which model has the lowest log loss and how it stands against C0, in a sentence."""
    result = a.result
    if result["status"] != "estimated":
        return f"Models were not estimated: {result['status']}."
    m = result["metrics"].set_index("model")
    best = m.log_loss.idxmin()
    text = f"Lowest log loss: {best} ({m.log_loss[best]:.3f}, skill {m.skill_pct[best]:+.1f}%)."
    if best != "C0" and "C0" in m.index:
        text += f" C0 (the offer rate) has {m.log_loss['C0']:.3f}."
    elif best == "C0":
        text += " No model with features does better than the training offer rate."
    clear = list(m.index[m.skill_low > 0])
    return text + f" Models whose skill interval lies wholly above 0: {', '.join(clear) or 'none'}. {ladder_caption(a)}"


def delta_table(result: dict, metric: str | None = None) -> pd.DataFrame:
    """Paired differences between models (later minus earlier) with their 95% intervals; `metric` keeps one."""
    if result["status"] != "estimated" or result["deltas"].empty:
        return pd.DataFrame()
    d = result["deltas"] if metric is None else result["deltas"][result["deltas"].metric == metric]
    return pd.DataFrame({"Comparison": d.comparison, "Metric": d.metric.map(METRIC_WORDS),
                         "Difference": d.delta.round(3), "95% interval, low": d.ci_low.round(3),
                         "95% interval, high": d.ci_high.round(3), P_HEADER: [p_text(p) for p in d.p_value],
                         "Bootstrap draws": d.n_boot}).reset_index(drop=True)


def delta_reading(comparison: str, metric: str, delta: float, low: float, high: float, p=None) -> str:
    """One sentence for one paired difference; with `p`, the bootstrap p-value closes it."""
    later, _, earlier = comparison.partition(" minus ")
    name = METRIC_WORDS.get(metric, metric)
    if delta != delta:
        return f"{later} against {earlier}: {name} is not defined for this pair."
    improves = (delta < 0) if metric in BETTER_WHEN_LOWER else (delta > 0)
    direction = "an improvement" if improves else ("no change" if delta == 0 else "a worsening")
    head = f"{later} changes {name} by {delta:+.3f} relative to {earlier} ({direction}); "
    if low != low or high != high:
        tail = "no interval could be computed."
    elif low <= 0 <= high:
        tail = f"the 95% interval [{low:+.3f}, {high:+.3f}] includes zero, so this sample does not show a difference."
    else:
        tail = f"the 95% interval [{low:+.3f}, {high:+.3f}] excludes zero."
    if p is not None and p == p:
        tail = tail[:-1] + f" (bootstrap {p_phrase(p)})."
    return head + tail


def delta_readings(result: dict, metric: str = "skill") -> list[str]:
    if result["status"] != "estimated" or result["deltas"].empty:
        return [f"No comparison to read: {result['status']}."]
    d = result["deltas"][result["deltas"].metric == metric]
    return [delta_reading(r.comparison, r.metric, r.delta, r.ci_low, r.ci_high, r.p_value) for r in d.itertuples()]


def baseline_gain_view(result: dict) -> pd.DataFrame:
    """Each family's gain over the baseline C2p, alone (the one-family model) and in the cumulative ladder, in AUC
    and in skill, with 95% intervals and bootstrap p (analysis.panel_bottleneck.baseline_gains)."""
    from analysis.panel_bottleneck import baseline_gains
    from analysis.pitch_figures import GAIN_FAMILY_WORDS

    if result["status"] != "estimated" or result["deltas"].empty:
        return pd.DataFrame()
    t = baseline_gains(result["deltas"], result["metrics"])
    digits = {"auc": 3, "skill": 1}
    return pd.DataFrame({
        "Family": t.family.map(GAIN_FAMILY_WORDS).fillna(t.family), "Metric": t.metric.map(METRIC_WORDS),
        "Alone (model)": t.alone_model,
        "Gain over C2p, alone": [_with_ci(v, lo, hi, digits[m]) for v, lo, hi, m in
                                 zip(t.alone_delta, t.alone_low, t.alone_high, t.metric)],
        "Bootstrap p, alone": [p_text(p) for p in t.alone_p],
        "Cumulative (model)": t.cumulative_model,
        "Gain over C2p, cumulative": [_with_ci(v, lo, hi, digits[m]) for v, lo, hi, m in
                                      zip(t.cumulative_delta, t.cumulative_low, t.cumulative_high, t.metric)],
        "Bootstrap p, cumulative": [p_text(p) for p in t.cumulative_p]})


def baseline_gain_reading(result: dict) -> str:
    """One sentence per family on its AUC gain over the baseline, alone and cumulative."""
    from analysis.panel_bottleneck import baseline_gains
    from analysis.pitch_figures import GAIN_FAMILY_WORDS

    if result["status"] != "estimated" or result["deltas"].empty:
        return f"No baseline comparison to read: {result['status']}."
    t = baseline_gains(result["deltas"], result["metrics"])
    lines = []
    for r in t[t.metric == "auc"].itertuples():
        def words(value, low, high, p):
            if value != value:
                return "not computed"
            told = "not told from zero" if low <= 0 <= high else "told from zero"
            return f"{value:+.3f} [{low:+.3f}, {high:+.3f}], {told} ({p_phrase(p)})"
        name = GAIN_FAMILY_WORDS.get(r.family, r.family)
        alone = words(r.alone_delta, r.alone_low, r.alone_high, r.alone_p)
        if r.alone_model == r.cumulative_model:
            lines.append(f"{name} ({r.alone_model}) over C2p: AUC {alone}.")
        else:
            cumulative = words(r.cumulative_delta, r.cumulative_low, r.cumulative_high, r.cumulative_p)
            lines.append(f"{name}: alone ({r.alone_model}) over C2p, AUC {alone}; cumulative ({r.cumulative_model}) "
                         f"over C2p, AUC {cumulative}.")
    return "\n".join(lines)


def penalty_table(a) -> pd.DataFrame:
    """One row per held-out pitch: the penalty each model used there."""
    folds = analyze_panel()._folds(a.result)
    if folds.empty:
        return folds
    table = folds.pivot(index="held_out", columns="model", values="penalty")     # every fold, a fold without a fit as NaN
    return table[[m for m in a.models if m in table.columns]].rename_axis(index="Held-out pitch", columns=None)


# ---- 8. verdicts ----------------------------------------------------------------------------

def rules_table() -> pd.DataFrame:
    """The two pre-registered rules in words (analysis.verdicts.RULES)."""
    from analysis.verdicts import RULES

    rows = []
    for hypothesis, rule in RULES.items():
        earlier, later = rule["pair"]
        words = (f"supported when the AUC gain is at least +{rule['auc_min']:.2f} with a 95% interval excluding 0"
                 + (", or the skill gain's interval excludes 0" if rule["skill_alternative"] else "")
                 + "; otherwise not distinguishable from zero")
        rows.append({"Hypothesis": hypothesis, "Comparison": f"{later} minus {earlier}", "Rule": words})
    return pd.DataFrame(rows)


def _with_ci(value, low, high, digits: int) -> str:
    if value != value:
        return "—"
    return f"{value:+.{digits}f}" + ("" if low != low else f" [{low:+.{digits}f}, {high:+.{digits}f}]")


def verdict_view(a) -> pd.DataFrame:
    """The verdicts of the pre-registered analysis, one row per hypothesis."""
    v = a.verdicts
    return pd.DataFrame({
        "Hypothesis": v.hypothesis, "Comparison": v.comparison,
        "AUC gain [95% interval]": [_with_ci(*x, 3) for x in zip(v.auc_delta, v.auc_low, v.auc_high)],
        "Skill gain, points [95% interval]": [_with_ci(*x, 2) for x in zip(v.skill_delta, v.skill_low, v.skill_high)],
        "AUC route": v.auc_route, "Verdict": v.verdict})


def verdict_text(a) -> str:
    """Each verdict with its numbers and the power sentence, as report.md section 3 states them."""
    return "\n".join(line for line in _verdict_lines(a.verdicts, a.result["metrics"]) if not line.startswith("### ")).strip()


# ---- 9. sensitivities -----------------------------------------------------------------------

def sensitivity_view(a) -> pd.DataFrame:
    """report.md section 4: one row per setting (the pre-registered analysis first), with each hypothesis's AUC and
    skill gains and its verdict."""
    return pd.DataFrame(_sensitivity_rows(a))


def sensitivity_reading(a) -> str:
    """Whether the sensitivities give the same verdicts as the pre-registered analysis, in words."""
    from analysis.verdicts import RULES

    main = a.verdicts.set_index("hypothesis").verdict
    lines = []
    for hypothesis in RULES:
        same, differ, not_run = [], [], []
        for name, table in a.sensitivity_verdicts.items():
            verdict = table.set_index("hypothesis").verdict[hypothesis]
            if str(verdict).startswith("not run"):
                not_run.append(name)
            else:
                (same if verdict == main[hypothesis] else differ).append(name)
        lines.append(f"{hypothesis}: main verdict '{main[hypothesis]}'; the same in {len(same)} of "
                     f"{len(same) + len(differ)} sensitivities run"
                     + (f"; different in {', '.join(differ)}" if differ else "")
                     + (f"; not run: {', '.join(not_run)}" if not_run else "") + ".")
    lines.append("The pre-registered verdicts are those of the main analysis; a sensitivity shows how much a verdict "
                 "depends on one choice, not a second test.")
    return "\n".join(lines)


def single_pass_text(a) -> str:
    single = a.single_pass_videos
    return (f"Read from one pass only in both_passes: {', '.join(f'{v} ({why})' for v, why in single.items())}."
            if single else "Every video has both passes for both_passes.")


# ---- 10. each concept on its own --------------------------------------------------------------

def single_feature_selection(table: pd.DataFrame, top_n: int = 25, show_all: bool = False) -> pd.DataFrame:
    """The rows shown: the primary features and the `top_n` others with the largest correlation, without the counts
    behind a mean; everything when `show_all`. Largest correlation first."""
    if show_all or table.empty:
        return table.reset_index(drop=True)
    kept = table[table.note != NOTE_COUNT]
    primary = kept.status == STATUS_PRIMARY
    others = kept[~primary].head(max(0, int(top_n)))
    return kept[primary | kept.index.isin(others.index)].reset_index(drop=True)


def single_feature_view(a, top_n: int = 25, show_all: bool = False) -> pd.DataFrame:
    t = single_feature_selection(a.single, top_n, show_all)
    if t.empty:
        return pd.DataFrame()

    def interval(low, high) -> list[str]:
        return ["" if lo != lo else f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(low, high)]
    return pd.DataFrame({
        "Feature": t.label, "Column name": t.feature, "Family": t.family, "Status": t.status, "Note": t.note,
        "Pitches with a value": t.n, "Correlation with any offer": t.correlation.round(2),
        "95% interval": interval(t.corr_low, t.corr_high), P_HEADER: [p_text(p) for p in t.p_value],
        Q_HEADER: [p_text(q) for q in t.q_value], "Survives FDR 5%": ["yes" if s else "" for s in t.significant_fdr],
        "Log-loss gain over the offer rate": t.log_loss_gain.round(3),
        "95% interval of the gain": ["" if lo != lo else f"{lo:+.3f} to {hi:+.3f}"
                                     for lo, hi in zip(t.gain_low, t.gain_high)],
        "Correlation after adjusting for questions asked": t.adjusted_correlation.round(2)})


def single_feature_caption(a, top_n: int = 25, show_all: bool = False) -> str:
    table, result = a.single, a.result
    shown = single_feature_selection(table, top_n, show_all)
    what = ("all of them" if show_all else
            f"{len(shown)}: the {int((shown.status == STATUS_PRIMARY).sum())} primary features and the "
            f"{int((shown.status != STATUS_PRIMARY).sum())} others with the largest correlation, counts behind a mean "
            "left out")
    return (f"Pitch rows: n pitches = {result['n_groups']}, n pitches with an offer = {result['n_events']}. Features "
            f"ranked: {len(table)}; shown: {what}. Intervals resample pitches ({a.single_draws} draws); p permutes "
            f"pitch outcomes ({a.single_draws} permutations); q is the p adjusted for the {len(table)} features "
            f"ranked.  [{label(a)}; descriptive ranking, model-coded, exploratory]")


def fdr_sentence(table: pd.DataFrame, what: str = "features") -> str:
    tested = table[table.p_value.notna()]
    k, raw = int(tested.significant_fdr.astype(bool).sum()), int((tested.p_value < 0.05).sum())
    return (f"{k} of {len(tested)} {what} survive FDR 5% (Benjamini-Hochberg q < {FDR_LEVEL:g} over the "
            f"{len(tested)} tested here); {raw} have an unadjusted permutation p below 0.05.")


def single_feature_reading(a) -> str:
    """The cautions that belong to a ranking of many features, with this table's own numbers."""
    table, result = a.single, a.result
    if table.empty:
        return f"No feature could be ranked: {result['status']}."
    clear = table[(table.corr_low > 0) | (table.corr_high < 0)]
    top = table.iloc[0]
    lines = [
        f"{len(table)} features were ranked. For {len(clear)} of them the 95% interval of the correlation excludes "
        f"zero; with {len(table)} features about {len(table) * 0.05:.0f} would do so by chance alone. This is a "
        "descriptive ranking, not a test.",
        fdr_sentence(table, "features"),
        f"Largest correlation: '{top.label}' ({top.feature}), {top.correlation:+.2f} [{top.corr_low:+.2f}, "
        f"{top.corr_high:+.2f}]" + (f". Note: {top.note}." if top.note else "."),
        "Each row looks at one feature alone; features overlap, so the rows do not add up, and a feature that ranks "
        "high here may add nothing once others are known (section 7).",
        f"There are only {result['n_groups']} pitch rows ({result['n_events']} with an offer), so every interval is "
        "wide and a few pitches can move a correlation a lot.",
    ]
    return "\n".join(f"- {line}" for line in lines)


def family_summary_view(a) -> pd.DataFrame:
    t = a.families
    if t is None or t.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Feature family": t.family, "Features ranked": t.features, "Unadjusted p < 0.05": t.p_below_05,
        "Survive FDR 5%": t.survive_fdr, "Median size of the correlation": t.median_abs_correlation.round(2),
        "Largest": [f"{lab} ({r:+.2f}, q {p_text(q)})" for lab, r, q in zip(
            t.largest_label, t.largest_correlation, t.largest_q_value)]})


def survivors_text(a, top: int = 15) -> str:
    table = a.single
    kept = table[table.significant_fdr.astype(bool)] if len(table) else table
    if not len(kept):
        return "No feature survives FDR 5%."
    listed = "; ".join(f"{r.label} ({r.correlation:+.2f}, q {p_text(r.q_value)}" + (f"; {r.note}" if r.note else "")
                       + ")" for r in kept.head(top).itertuples())
    more = f"; and {len(kept) - top} more" if len(kept) > top else ""
    return f"Surviving FDR 5% ({len(kept)}), largest correlation first: {listed}{more}."


def coefficient_view(a, top: int = 20) -> pd.DataFrame:
    t = a.coefficients.head(int(top)) if a.coefficients is not None else pd.DataFrame()
    if t.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "Feature": t.label, "Column": t.feature, "Family": t.family,
        "Coefficient (log-odds per 1 SD)": t.coefficient.round(3),
        "95% interval": [f"{lo:+.3f} to {hi:+.3f}" if lo == lo else "" for lo, hi in zip(t.ci_low, t.ci_high)],
        P_HEADER: [p_text(p) for p in t.p_value]})


def coefficient_caption(a, top: int = 20) -> str:
    t = a.coefficients
    if t is None or t.empty:
        return f"The full model C5 was not fitted: {a.result['status']}."
    return (f"Model C5 refitted on all {int(t.n_rows.iloc[0])} pitches with a known outcome, penalty "
            f"{t.penalty.iloc[0]:g} (the value chosen most often in the held-out run). {len(t)} columns in the model; "
            f"the {min(int(top), len(t))} largest are shown; {int((t.p_value < 0.05).sum())} have an unadjusted "
            f"bootstrap p below 0.05 ({int(t.n_boot.iloc[0])} resamples of pitches, model refitted in each).  "
            f"[{label(a)}; conditional associations within a penalised model, not independent effects; exploratory]")


def by_offer_view(a) -> pd.DataFrame:
    t = a.by_offer
    if t is None or t.empty:
        return pd.DataFrame()

    def measure(value, kind):
        return "" if value != value else f"{value:.0%}" if kind.startswith("share") else f"{value:.2f}"
    return pd.DataFrame({
        "Questions": t.questions, "Measure": t.measure,
        "Mean, pitches with an offer": [measure(v, m) for v, m in zip(t.mean_offered, t.measure)],
        "Mean, pitches without": [measure(v, m) for v, m in zip(t.mean_not_offered, t.measure)],
        "Correlation with any offer": t.correlation.round(2),
        "95% interval": ["" if lo != lo else f"{lo:+.2f} to {hi:+.2f}" for lo, hi in zip(t.corr_low, t.corr_high)],
        P_HEADER: [p_text(p) for p in t.p_value], Q_HEADER: [p_text(q) for q in t.q_value],
        "Pitches with data": t.n, "of them with an offer": t.n_offered, "Column name": t.feature})


def by_offer_caption(a) -> str:
    r = a.result
    return (f"Pitch rows with a known outcome: n pitches = {r['n_groups']}, n pitches with an offer = {r['n_events']}. "
            f"Answers before the cut, pooled over all sharks. Intervals resample pitches ({a.single_draws} draws); p "
            f"permutes pitch outcomes; q adjusts it for the rows of the table.  [{label(a)}; descriptive, not a test]")


def by_offer_reading(a) -> str:
    t = a.by_offer
    if t is None or t.empty or t.correlation.isna().all():
        return "Nothing to compare: no row has both a value and a varying outcome."
    lines = []
    for questions, part in t.groupby("questions", sort=False):
        clear = part[(part.corr_low > 0) | (part.corr_high < 0)]
        values = "; ".join(f"{'checkable support' if r.measure.startswith('share') else 'substance'}: "
                           f"{r.correlation:+.2f} [{r.corr_low:+.2f}, {r.corr_high:+.2f}], n = {r.n}"
                           for r in part.itertuples())
        lines.append(f"{questions[0].upper() + questions[1:]}: {values}. Intervals that exclude zero: {len(clear)} of "
                     f"{len(part)}.")
    lines.append(fdr_sentence(t, "comparisons"))
    lines.append("Descriptive and looked at beside many other features: a pattern, not a finding.")
    return "\n".join(f"- {line}" for line in lines)


# ---- 11. H9 ---------------------------------------------------------------------------------

H9_PREDICTOR_WORDS = {"evid": "checkable support", "uncert": "admitted uncertainty", "subs": "substance",
                      "coh": "coherence", "ctx": "context (Gemini only, not human-coded)",
                      "repair_before_next": "a repair before the next turn",
                      "q_promotion": "the question's frame (covariate)"}


def h9_caption(a) -> str:
    if a.h9 is None:
        return "H9 was not run in this execution (RUN_H9 = False)."
    every = a.h9["all"]
    return (f"Answers to investor questions before the cut: up to {int(every.n_answers.max())} answers in "
            f"{len(a.data.videos)} pitches, compared within each pitch. {a.n_boot} pitch-bootstrap draws.  "
            f"[{label(a)}; exploratory pattern, no decision rule, no multiple-test correction]")


def h9_models_table() -> pd.DataFrame:
    """The nested H9 models: their predictors and the answers each rests on."""
    return pd.DataFrame([{"Model": model, "Predictors": ", ".join(H9_PREDICTOR_WORDS.get(p, p) for p in predictors),
                          "Answers it rests on": H9_POPULATIONS[model]} for model, predictors in H9_MODELS.items()])


def h9_view(a, claimed_only: bool = False) -> pd.DataFrame:
    """The H9 coefficients for all and for hard questions side by side, with the direction claimed (only where both
    intervals exclude zero on the same side); the frame covariate's rows say covariate. `claimed_only` keeps the
    rows with a direction."""
    if a.h9 is None:
        return pd.DataFrame()
    keys = ["model", "outcome", "predictor"]
    both = a.h9["all"].merge(a.h9["hard"], on=keys, how="left", suffixes=("_all", "_hard")).merge(
        a.h9_summary[keys + ["direction"]], on=keys, how="left")

    def estimate(coef, low, high, separated, estimable) -> str:
        if separated == separated and bool(separated):
            return "separated"
        why = "" if not isinstance(estimable, str) or estimable == "yes" else f" ({estimable})"
        if coef != coef:
            return "—" + why
        return f"{coef:+.2f}" + ("" if low != low else f" [{low:+.2f}, {high:+.2f}]") + why
    table = pd.DataFrame({
        "Model": both.model, "Outcome": both.outcome, "Predictor": [labelled(p) for p in both.predictor],
        "All questions": [estimate(*x) for x in zip(both.coef_all, both.low_all, both.high_all, both.separated_all,
                                                     both.estimable_all)],
        "n (all)": both.n_answers_all.astype(int),
        "Hard questions": [estimate(*x) for x in zip(both.coef_hard, both.low_hard, both.high_hard,
                                                      both.separated_hard, both.estimable_hard)],
        "n (hard)": ["—" if n != n else int(n) for n in both.n_answers_hard],
        "Direction": ["covariate, no direction" if p in COVARIATES else d
                      for p, d in zip(both.predictor, both.direction)]})
    if claimed_only:
        table = table[table.Direction.isin(("positive", "negative"))]
    return table.reset_index(drop=True)


def h9_directions(a) -> str:
    """Every direction H9 claims, and how many predictor rows are not distinguishable, in words."""
    if a.h9 is None:
        return h9_caption(a)
    s = a.h9_summary
    claimed = s[s.hard_and_all_agree.astype(bool)]
    lines = [f"{len(claimed)} of {len(s)} predictor rows (model × outcome × predictor, the covariate left out) have "
             "a direction: the intervals for all and for hard questions both exclude zero on the same side. "
             f"{int((s.direction == 'not distinguishable').sum())} are not distinguishable."]
    lines += [f"  - {line}." for line in h9_claim_lines(a, H9_PREDICTOR_WORDS)]
    unclaimed = h9_unclaimed_lines(a, H9_PREDICTOR_WORDS)
    lines.append(f"Not claimed: in {len(unclaimed)} rows the all-questions interval excludes zero but the "
                 "hard-questions interval does not (or there is none)" + (":" if unclaimed else "."))
    lines += [f"  - {line}." for line in unclaimed]
    lines.append(H9_EXPLORATORY_TEXT)
    return "\n".join(lines)


def h9_sensitivity_note(a) -> str:
    """One sentence pointing at the exchange-based sensitivity, with no number or direction."""
    if a.h9_sensitivity is None:
        return "The exchange-based outcomes were not computed in this execution (H9_EXCHANGE_OUTCOMES = False)."
    return (f"The exchange-based outcomes were computed as an unreported sensitivity ({a.h9_sensitivity_draws} draws) "
            "and are exported to h9_exchange_sensitivity.csv; they are not shown here.")


# ---- 12. one prediction ---------------------------------------------------------------------

def default_model(a, model: str | None = None) -> str:
    """The model to inspect: the one named, else C3 (the model both verdicts read), else the last model."""
    if model is not None:
        if model not in a.models:
            raise ValueError(f"INSPECT_MODEL {model!r} is not a model of the ladder: {', '.join(a.models)}.")
        return model
    return "C3" if "C3" in a.models else list(a.models)[-1]


def _held_out(a, video_id: str) -> bool:
    return a.result["status"] == "estimated" and (a.result["predictions"].video_id == video_id).any()


def prediction_text(a, model: str, video_id: str) -> str:
    if a.result["status"] != "estimated":
        return f"No held-out prediction to inspect: {a.result['status']}."
    if not _held_out(a, video_id):
        return f"{video_id} has no known outcome, so no held-out prediction exists."
    row = a.result["predictions"][a.result["predictions"].video_id == video_id].iloc[0]
    p, y = float(row[f"p_{model}"]), float(row.y)
    penalty = fold_detail(a.result, model, video_id).get("penalty")
    return (f"Model {model}, trained without {video_id}, gives it a {p:.1%} chance of an offer. What happened: "
            f"{'an offer' if y == 1 else 'no offer'}. This fold's model used "
            f"{len(editable_features(a.result, model, video_id))} feature columns (a column that is the same for every "
            f"training pitch is dropped), penalty {penalty:g}; the intercept and the largest pushes come first below."
            if penalty is not None else
            f"Model {model}, trained without {video_id}, gives it a {p:.1%} chance of an offer (the training offer "
            f"rate: no features). What happened: {'an offer' if y == 1 else 'no offer'}.")


def prediction_view(a, model: str, video_id: str) -> pd.DataFrame:
    """The features behind one held-out prediction, largest push first (the intercept first)."""
    if not _held_out(a, video_id):
        return pd.DataFrame()
    b = prediction_breakdown(a.result, a.features, model, video_id)
    if b.empty:
        return b
    return pd.DataFrame({
        "Feature": b.feature, "Label": b.label, "This pitch's value": b.value, "Filled in (was missing)": b.filled,
        "Weight in this fold": [round(float(c), 4) for c in b.coefficient],
        "Push on the prediction (log-odds)": [round(float(c), 4) for c in b.contribution]})


# ---- 13. limits -----------------------------------------------------------------------------

def limits_table(a) -> pd.DataFrame:
    """Plan v4 section 11 (read from the file, with this run's count of pitches without an offer), then the limits
    of the method that plan B's reviews named."""
    rows = [("plan v4, section 11", bullet) for bullet in _limit_bullets(a)]
    rows += [("the method", bullet) for bullet in METHOD_LIMITS]
    return pd.DataFrame(rows, columns=["From", "Limit"])


# ---- 15. figures ----------------------------------------------------------------------------

def figures_dir(a, analysis_id: str) -> Path:
    return Path(a.run_root).resolve() / "analysis" / analysis_id / analyze_panel().FIGURES_FOLDER


def save_figures(a, analysis_id: str, width="double") -> Path:
    """Write the figures of an analysis under <run>/analysis/<analysis_id>/figures/ and return that folder."""
    from analysis import pitch_figures

    folder = figures_dir(a, analysis_id)
    pitch_figures.write_figures(a, folder, width, skipped={})
    return folder


def figure_files(a, analysis_id: str) -> pd.DataFrame:
    """Every figure file under <run>/analysis/<analysis_id>/figures/, one row per figure, with its caption."""
    from analysis.pitch_figures import FIGURES, file_names

    folder = figures_dir(a, analysis_id)
    rows = []
    for number, (_, title) in FIGURES.items():
        files = file_names(number)
        found = [files[kind] for kind in ("png", "pdf") if (folder / files[kind]).exists()]
        if not found:
            continue
        note = folder / files["caption"]
        rows.append({"Figure": number, "What it shows": title, "Files": ", ".join(found),
                     "Caption file": note.name if note.exists() else "",
                     "Caption": note.read_text().strip() if note.exists() else ""})
    return pd.DataFrame(rows)


def _shown_path(folder: Path) -> str:
    """A folder as runs/<run>/... when it lies under a runs/ folder (no absolute path of this machine)."""
    text = str(folder)
    return re.sub(r"^.*?/runs/", "runs/", text) if "/runs/" in text else text


def files_note(a, analysis_id: str, written: bool) -> str:
    """Where the figure files are, and whether this execution wrote them."""
    from analysis.pitch_figures import SAVE_DPI

    folder = _shown_path(figures_dir(a, analysis_id))
    n = len(figure_files(a, analysis_id))
    if written:
        return f"Saved {n} figures (PNG at {SAVE_DPI} dpi, vector PDF and a caption file each) to {folder}/"
    if n:
        return (f"EXPORT and SAVE_FIGURES are False: nothing was written. The {n} figures listed below are already in "
                f"{folder}/ from an earlier export (they may come from other settings than this execution's).")
    return (f"EXPORT and SAVE_FIGURES are False: nothing was written, and {folder}/ holds no figure yet. Set "
            "SAVE_FIGURES = True in the configuration cell to write them.")
