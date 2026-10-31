#!/usr/bin/env python3
"""Human review of a panel run: export review sheets, build coding pages, import what coders and adjudicators filled.

export  per video, a CSV of review items, a Markdown page of instructions and a manifest of what was selected and
        why, under <video>/review/. The default preset (exchange) is a seeded random sample of rows; flagged,
        quick and full are flag-driven sheets for adjudication. --blind hides the model's values and reasons.
        --sample-videos N draws the videos (seeded, excluding --exclude-from's pilot videos) into
        <run>/review/video_sample.json and .csv; with it (or with --rows-per-field, default 60) the number of
        question-answer pairs per video is planned so every sampled field reaches that many rows
        over the sampled videos, and the plan is recorded in each manifest (sample.planned). --stratify any_offer
        (2026-10-21) draws half of the videos from the pitches with an offer and half from those without and records
        the strata. The pairs are drawn only before the pitch's first decision (the answers the analysis reads).
page    one self-contained HTML coding page per coder for the sampled (or named) videos, built from the blind
        manifests, the records and the codebooks -> <run>/review/pages/<coder>.html. The coder opens it in Chrome,
        picks the video folder and saves sheets to a shared folder.
copy-videos  copies the full video files of the sampled videos and of the clip session to a folder to hand to the
        coders (--training: the training videos). The clip session plays these same full videos at their prepared
        windows, so no clips are cut (--clips is refused since 2026-10-21).
clips-export  the clip session, outcome-blind: a balanced design of 25 pairs of the ten videos (each in five), the
        session's sheet (each video's seven cards: the Stage A items, the prepared codes but PRES-CTX, PRES-EVID in
        bands, the ask; then three left/right cards per pair), a manifest holding the model's answers (a pair's from
        <run>/ranking/scores.json, refused without it) and one page per coder in a seeded video order, each video
        played in full with its prepared window -> <run>/review/clips/.
session-export  one page per coder, as a task list (2026-10-21, the researcher's "order A"): in one step the clip
        session of clips-export (manifest and sheet, no clip-session pages) and each video's blind full-video sheet
        (as export --blind; an export already there is kept, or with --force drawn again with its own seed), then
        <run>/review/pages/<coder>__tasks.html:
        each video's prepared presentation and Q&A as one task, each comparison as its own task as soon as both its
        videos are done. The page writes clips__<coder>.csv and <video>__<coder>.csv as the separate pages did. The
        comparisons are then made after the coder has seen the questioning, so the pair agreement is not
        outcome-blind.
clips-import  a coder's session sheet: each video's cards into its review log as coder lines, the pair
        judgements into <run>/review/clips/pairs_log.jsonl (an unanswered pair counted and skipped), and the
        coder's Bradley-Terry scores per Stage A item -> <run>/review/clips/<coder>_scores.json. --sheet may be
        the coder's zip of sheets (<coder>__sheets.zip, the page's "Download all sheets") or a folder holding it.
discuss  a discussion page for the researcher and one coder (outside the validation): each video with a question-answer
        turn the coder flagged as structurally wrong or a question concept the coder put in another frame than
        Gemini's (every pair holding one, all six codes) and --pairs
        comparisons drawn (--seed) from those the coder chose against the model, each card showing the coder's answer
        and Gemini's -> <run>/review/pages/<coder>__discussion.html and <run>/review/discussion/<coder>/manifest.json.
        Its sheet, <coder>__discussion.csv, is never imported.
training  blind sheets for a few development videos (never the validation sample) and one training page per
        coder, with the whole codebook laid out on the page -> <run>/review/training/ (pages/<coder>.html, and
        model_answers.html, the model's answers and recorded reasons for the joint walkthrough, and
        training_videos.csv, the list to send). --clips adds the training clip session (clips/: the training
        videos at their prepared windows with their seven cards, then their one pair three times, from the run's
        rank scores; the full-video sheets then leave the Stage A items to it, each coder's page is then the task
        page of session-export, and model_answers.html opens with the clip session's cards and the pairs). Nothing
        it writes is part of the validation.
calibrate  imports the trainees' sheets (the full-video sheets and the clip-session sheets, clips__<coder>.csv) into
        a throwaway copy of the run and reports how far the coders agree with each other per code, as the paper
        table's coder-against-coder table (raw agreement, alpha in parentheses, band, flags), with alpha against a bar
        (default 0.61, the lower bound of Landis & Koch's "substantial" band, read from validation_reporting.yaml),
        and the pair judgements per Stage A item against 0.5 (Clopper-Pearson interval and p)
        -> <run>/review/training/calibration.md and .json; the run's own review logs are never touched.
import  checks a filled sheet against the codebooks. --role coder (default) logs and measures it and never writes
        results/reviewed.json; --role adjudicator also writes the corrected copy of the primary record,
        results/reviewed.json, with a correction log and validation report. Any problem is printed and nothing is written.
import-folder  imports every <video>__<coder>.csv of a folder as role coder (an adjudicator's only with --role adjudicator);
        a clip-session sheet (clips__<coder>.csv) is skipped: clips-import reads it. --dir may be a coder's zip of
        sheets (<coder>__sheets.zip); a member with an absolute path or .. refuses the zip.
status  per video: whether a sheet was exported, rows reviewed per reviewer, the agreement headline.
summary pools the imported videos into one agreement table per selection (again without the rows a coder ticked "not
        sure", [unsure]; for a stratified sample also per stratum and reweighted to the run's mix); validity figures use coder lines only
        (a video with only adjudicator lines is not a validity video), and the agreement of each reported code with
        the statistics of its kind (the instrument's validation_reporting.yaml). --paper-table OUT.md writes the
        paper's validation table (OUT.md, OUT.json, OUT_appendix.md) from coder-coded videos: raw agreement with
        Krippendorff's alpha in parentheses, alpha's Landis & Koch band and flags in words, coder against coder and
        pass against pass, the appendix of every other figure and a generated note on the statistics; the clip
        session's comparisons give one row per Stage A item, read against 0.5. Nothing is passed or failed.

Role rule: a blind sheet is a coder's judgment and is measured, never applied to the record; only an adjudicator's
import corrects it. Outputs go under <run>/review/ and <run>/<video>/review/.

Offline. The model's passes (results/primary.json, results/replicate.json) are never modified.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import json
import os
import random
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.errors import ConfigError  # noqa: E402
from pipeline.ingest import sha256_file  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.schemas.stage_b_panel import (  # noqa: E402
    ANNOTATION_CONTEXT, MISSING_REASONS, PanelRecord, seconds_to_clock,
)
from pipeline.stage_b.panel_books import PanelBooks, load_panel_books  # noqa: E402
from pipeline.stage_b.panel_pipeline import _dump, _json_text, _read, derive, model_pass, prepared_window  # noqa: E402
from pipeline.stage_b.panel_validate import validate_record  # noqa: E402
from pipeline.stage_a import rank_pipeline  # noqa: E402
from pipeline.stage_b import review_clips, review_discuss  # noqa: E402
from pipeline.stage_b.review_labels import PAIR_VALUES, load_labels  # noqa: E402
from pipeline.stage_b.review_report import (  # noqa: E402
    ALPHA_KINDS, HEADLINE, REPORTING_BAND_FLOORS, Reporting, clock_differences, methods_note, note_of,
    paper_appendix_markdown, paper_rows, paper_table_markdown, strata_markdown, stratum_rows, summary_rows,
    table_lines)
from pipeline.stage_b.review_report import load_reporting as _reporting_of_release  # noqa: E402
from pipeline.stage_b.review_codebook import codebook_model  # noqa: E402
from pipeline.stage_b.review_page import (  # noqa: E402
    TRAINING, WALKTHROUGH, PageError, clips_walkthrough, on_sheet, page_model, render_page, tasks_page_model)
from pipeline.stage_b.review_apply import (  # noqa: E402
    MIN_CLUSTER_VIDEOS, OUTCOME_OVERRIDE, WILSON_METHOD, Correction, ReviewError, Tolerances, agrees,
    apply_corrections, counted_not_banded, human_agreement, human_human, level_of, parse_sheet, pooled_summary,
    unnamed_others, validity_lines, decision_turn_context,
)
from pipeline.stage_b.review_items import (  # noqa: E402
    ACTOR_ROLES, ASK, CUT, CUT_PATH, EVID_BANDS, EVID_PATH, NOT_AN_EXCHANGE, OUTCOME, OUTCOME_PATH,
    OUTCOME_VALUES, PAIR, REPLICATE_ONLY, SHEET_CLASSES, SHEET_COLUMNS, STAGE_A_CHANGED, UNIT_PRESENCE, VENTURE_CODES,
    allowed_values, any_offer, canonical_full, field_class, locator, locator_span, normalize_path, presence_ref, read_value,
)
from pipeline.stage_b.review_select import (  # noqa: E402
    ALL, EXCHANGE_DEFAULTS, FLAGGED, QA_CODES, SAMPLED, select_exchange_sheet, select_sheet,
)

COLUMNS = SHEET_COLUMNS
MODEL_COLUMNS = ("priority", "primary", "replicate", "reasons", "selected_as")     # empty on a blind sheet
EXCHANGE = "exchange"                                       # the default preset: a random sample, not flag-driven
# the flag-driven presets: name -> (max flagged rows or no cap, sampled unflagged rows)
PRESETS = {"flagged": (60, 20), "quick": (25, 15), "full": (None, 30)}
PRESET_PURPOSE = {
    EXCHANGE: "the full-video sheet, a random sample of rows per video: the pitch's first decision (when is it "
              "spoken) and a number of question-answer pairs (the question's concept and severity; the answer's "
              "responsiveness, evidence, substance and coherence), set per video or planned from --rows-per-field; "
              "exchange-topic, Stage A and venture rows only when --exchanges-per-video, --stage-a-per-video or --venture-per-video ask for "
              "them (the exchange topics are not drawn by default since 2026-10-21), and the outcome (did any "
              "investor make an offer) and the check of the prepared window only when --field-classes names "
              "`outcome` or `prepared_window` (neither is drawn by default since 2026-10-21); every row "
              "an unbiased estimate; for measuring agreement over many videos",
    "full": "the complete sheet: every flagged row of every field class plus 30 sampled ones; for adjudicating "
            "one video in depth",
    "flagged": "the flagged rows up to a cap of 60 plus 20 sampled ones; for correcting what looks wrong",
    "quick": "as flagged with a cap of 25 and 15 sampled ones; for a fast look at what looks wrong",
}
VIDEO_SAMPLE = "video_sample.json"                          # under <run>/review/
MANIFEST = "export_manifest.json"
REVIEW_DIR = "review"
LOG = "review_log.jsonl"
AGREEMENT = "human_agreement.json"
ADDED = "added"         # selected_as of a row the coder added to the sheet
REVIEWER_NAME = re.compile(r"[A-Za-z0-9_-]+")
NOT_A_NAME = "is not a coder name (letters, digits, - and _ only)"
DUPLICATE_HINT = "a browser's duplicate download adds ' (1)': delete the duplicate"
HUMAN_REVIEWED = "partial"
SHEET_ENCODING = "utf-8-sig"    # UTF-8 with the mark spreadsheets need to read it as UTF-8; also reads plain UTF-8
DELIMITERS = (",", ";", "\t")   # what a spreadsheet may save a sheet with, depending on its locale
TEMP_SUFFIX, BACKUP_SUFFIX = ".tmp", ".bak"


def _one_line(value) -> str:
    return "" if value is None else " ".join(str(value).split())


def reviewer_name(text) -> str:
    """A coder's or adjudicator's name as it is logged and as it names a sheet and a page: letters, digits, `-`
    and `_` only. Anything else is a ConfigError (a name with a blank or a bracket is how a duplicated download
    becomes a third coder)."""
    name = "" if text is None else str(text)
    if not REVIEWER_NAME.fullmatch(name):
        raise ConfigError(f"{name!r} {NOT_A_NAME}")
    return name


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _cell(value) -> str:
    return _one_line(value).replace("|", "\\|")


def _video_dirs(run: Path, video_ids: list[str] | None) -> list[Path]:
    if not run.is_dir():
        raise ConfigError(f"run directory not found: {run}")
    found = sorted(p for p in run.iterdir() if (p / "results" / "primary.json").exists())
    if video_ids:
        missing = sorted(set(video_ids) - {p.name for p in found})
        if missing:
            raise ConfigError(f"no primary result in {run} for: {', '.join(missing)}")
        found = [p for p in found if p.name in set(video_ids)]
    return found


def _optional(path: Path):
    return _read(path) if path.exists() else None


def _record(result: dict, path: Path) -> PanelRecord:
    """The panel record of a result file (results/primary.json, replicate.json or reviewed.json of a video). A record
    that fails the schema is refused in one line: with the release its video's experiment.json names when that is
    not the live release (one annotated up to v3-dev8, or with no release recorded), otherwise as a record of the
    live release that fails its schema, with where the first problem is."""
    if not isinstance(result, dict) or "record" not in result:
        raise ConfigError(f"{path} does not hold a panel record (KeyError)")
    try:
        return PanelRecord.model_validate(result["record"])
    except ValueError as exc:
        experiment = _optional(path.parent.parent / "experiment.json")
        release_id = (experiment or {}).get("release_id") if isinstance(experiment, dict) else None
        live = load_release().release_id
        if release_id == live:
            errors = exc.errors() if hasattr(exc, "errors") else []
            where = (f"; {len(errors)} problem{'s' if len(errors) != 1 else ''}, the first at "
                     f"{'.'.join(str(part) for part in errors[0]['loc'])}: {errors[0]['msg']}" if errors else "")
            raise ConfigError(f"{path} fails the {live} panel record schema ({type(exc).__name__}{where})") from exc
        release = f"release {release_id}" if release_id else "an earlier release"
        raise ConfigError(f"{path} was annotated with {release} and is not a {live} panel record "
                          f"({type(exc).__name__}); the review tooling reads {live} records only") from exc


def _manifest_entry(directory: Path, video_id: str) -> dict | None:
    manifest = _optional(directory / MANIFEST)
    return ((manifest or {}).get("videos") or {}).get(video_id)


# ---- export -------------------------------------------------------------------------------

def _code_lines(books: PanelBooks) -> list[str]:
    lines = []
    for function in ("question", "answer"):
        for entry in books.codes_for(function):
            if entry.type == "categorical":
                values = ", ".join(entry.values or [])
            elif entry.type == "binary":
                values = "true, false"
            else:
                values = f"an integer {int(entry.scale[0])}-{int(entry.scale[1])}"
            lines.append(f"  - `{entry.id}` ({function} turns): {values}")
    return lines


NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
EXCHANGE_CLASSES = ("turn_exchange", "exchange_kind", "exchange_topic")


def _units_of(classes: set[str]) -> str:
    """Which units the sheet's rows belong to, for the `context` sentence: turn, exchange, decision, as present."""
    units = [unit for unit, present in (
        ("turn", classes & {"actor", "turn_function", "turn_code", "turn_exchange", "turn_link"}),
        ("exchange", classes & {"exchange_kind", "exchange_topic"}),
        ("decision", classes & {"decision"})) if present]
    return " or ".join([", ".join(units[:-1]), units[-1]] if len(units) > 1 else units)


OTHER_NEEDS_A_NAME = ("Choosing `other` for a concept needs a note: name the matter in a few words (the coding page's "
                      "note box asks for it, as it does when you tick a box).")


def _qa_sentence(qa_codes) -> str:
    """What each drawn question-answer pair carries, from the codes the export drew."""
    qa_codes = [tuple(code) for code in qa_codes]
    count = NUMBER_WORDS[len(qa_codes)] if len(qa_codes) < len(NUMBER_WORDS) else str(len(qa_codes))
    if qa_codes == list(QA_CODES):
        return (f"Each question-answer pair carries {count} codes: the question's concept and severity, the "
                "answer's responsiveness, evidence, substance and coherence.")
    parts = [f"the {role}'s " + ", ".join(f"`{code}`" for r, code in qa_codes if r == role)
             for role in ("question", "answer") if any(r == role for r, _ in qa_codes)]
    return f"Each question-answer pair carries {count} code{'s' if len(qa_codes) != 1 else ''}: {'; '.join(parts)}."


def instructions(video_id: str, experiment: dict, record: PanelRecord, books: PanelBooks, entry: dict,
                 run: Path, decision_tolerance_s: float | None = None) -> str:
    """The page a coder reads before filling the sheet. Its sentences describe the rows the sheet carries: the
    pitch rows, the window check, the question-answer codes and the exchange wording only when such rows are on it.
    `decision_tolerance_s` is the pass-to-pass decision tolerance the rows were flagged by (the run's agreement
    settings; the default when None)."""
    blind = entry["blind"]
    classes = {row["field_class"] for row in entry["rows"]}
    exchanges = bool(classes & set(EXCHANGE_CLASSES))
    flag_s = Tolerances().decision_s if decision_tolerance_s is None else decision_tolerance_s
    qa_codes = ((entry.get("sample") or {}).get("requested") or {}).get("qa_codes")
    units = _units_of(classes)
    mode = ("blind: the model's values and the reasons for selecting a row are hidden; code each row from the footage"
            if blind else
            "adjudication: both model passes are shown; decide what the footage supports")
    L = [f"# Review sheet: {video_id}", "",
         f"- Video file: `{experiment.get('source') or 'not recorded in experiment.json'}`",
         f"- Run: `{run}` (instrument release {entry['release_id']})",
         f"- Sheet: `{entry['sheet']}`, {len(entry['rows'])} rows, preset `{entry['preset']}`",
         f"- Mode: {mode}",
         *([] if blind or not entry["flagged"] or not entry["flagged"]["left_out"]["n"] else [
             f"- Flagged rows: {entry['flagged']['on_sheet']['n']} of {entry['flagged']['found']['n']} are on this "
             f"sheet; the cap of {entry['max_flagged']} left out {entry['flagged']['left_out']['n']} "
             "(counted in the manifest)"]),
         # the window is itself a judged value (and Stage A is judged over it), so a blind page does not print it
         ("- Prepared presentation: judge it as you see it (this sheet does not show the window the model recorded; the "
          "coding page, built with the `page` subcommand, does show it as an aid; the window is checked only by the "
          "`prepared_window` row's structural tick, and the ratings of the prepared presentation are therefore "
          "conditional on it)" if blind and "prepared_window" in classes else
          "- Prepared presentation: this sheet does not show the window the model recorded (the coding page, built "
          "with the `page` subcommand, shows it as an aid), and no row of it checks the window" if blind
          else f"- Prepared presentation: {', '.join(prepared_window(record)) or 'none recorded'}"),
         "", "## Actors", "", "| id | role | label |", "|---|---|---|"]
    L += [f"| {_cell(a.actor_id)} | {_cell(a.role)} | {_cell(a.label)} |" for a in record.actors]
    pitch = classes & {OUTCOME, CUT}
    pitch_context = "the question it asks" if blind else "the decisions the record holds"
    context = ", or for a pitch row ".join(filter(None, (f"which {units} it belongs to" if units else "",
                                                        pitch_context if pitch else "")))
    context = context if units or not pitch else f"for a pitch row {pitch_context}"
    fill_how = [
        *([(f"The `{OUTCOME_PATH}` and `{CUT_PATH}` rows belong to the whole video: watch it to the end and answer "
            "whether any investor made an offer and at what clock the first decision is spoken."
            if pitch == {OUTCOME, CUT} else
            f"The `{OUTCOME_PATH}` row belongs to the whole video: watch it to the end and answer whether any "
            "investor made an offer." if OUTCOME in pitch else
            f"The `{CUT_PATH}` row belongs to the whole video: watch it to the end and give the clock at which the "
            "first decision is spoken, even if other questions follow it.")] if pitch else []),
        *(["The `prepared_window` row asks only whether the prepared presentation is where "
           f"{'the coding page shows it' if blind else 'the `primary` column says'}: leave `human_value` empty and, "
           "when it is not, set `resolution` to `structural` and write in `note` where it starts and ends."
           + ("" if blind else " To correct the record instead, write the window in `human_value`.")]
          if "prepared_window" in classes else []),
        *([_qa_sentence(qa_codes)] if qa_codes and "turn_code" in classes else []),
        # a concept of `other` needs the matter named, which the coding page asks for in the note
        *([OTHER_NEEDS_A_NAME] if classes & {"turn_code", "exchange_topic"} else [])]
    L += ["", "## How to fill the sheet", "",
          "Each row is one field of the record. `path` says which field, `clock_start`-`clock_end` where to look in "
          f"the video{', `context` ' + context if context else ''}. Watch that part of the video and fill "
          "only the last three columns; leave every other column as it is.", "",
          *([" ".join(fill_how), ""] if fill_how else []),
          *((["Stage A rows carry no clocks: judge the prepared presentation as you see it, from where the founders "
              "start their prepared pitch to where the investors take over.", ""] if "stage_a" in classes else []) +
            ["## What blind mode still reveals", "",
             "Blind mode hides the model's values, the reasons a row was selected and the prepared window. It cannot "
             "hide everything, and you should know what you are still shown: the model's turn segmentation (where "
             f"each turn starts and ends{', and which turns form an exchange' if exchanges else ''}); who speaks each "
             "turn; and, through the "
             "path names, which turns the model took for questions or answers (a row `turn:<id>/code:PR-...` is a "
             "turn coded as a question, `turn:<id>/code:RS-...` or `PREP-...` one coded as an answer). A row that "
             "exists only because the other pass lacks a unit (`unit_presence`) is left off a blind sheet for that "
             "reason. Agreement measured on a blind sheet is therefore agreement given the model's segmentation, not "
             "independent coding from the footage alone.", ""] if blind else [
              "`priority` orders the work when time is short: 1 = a sampled row, 2 = a turn, exchange or decision "
              "only one pass recorded (`unit_presence`), 3 = a row a validator error points at, 4 = any other "
              "flagged row. Sort by it and work upward from 1. On a flag-driven sheet (`flagged`, `quick`, `full`) "
              "the sampled rows are drawn in proportion to each field class from the fields that nothing flagged "
              "and whose unit both passes recorded, so they estimate human-model agreement on such fields only; on "
              "the `exchange` sheet every row is a random draw. The flagged rows were chosen because something "
              "looked wrong.", ""]),
          "- `human_value`: what the footage supports, written as in `allowed_values` when that column lists values.",
          "- `resolution`: optional when `human_value` is filled. One of:",
          *([] if blind else [
              "  - `confirm_primary`: the primary value is right (also assumed when `human_value` equals it).",
              "  - `confirm_replicate`: the replicate value is right."]),
          ("  - `correct`: `human_value` is what the footage shows (assumed whenever `human_value` is filled). With "
           "`human_value` left empty it means the field should be empty." if blind else
           "  - `correct`: `human_value` is right (assumed when it differs from the primary). With `human_value` left "
           "empty it means the field should be empty."),
          "  - `unclear`: the footage does not let you decide. Nothing changes; say why in `note`.",
          "  - `structural`: the problem is not this value but the unit itself (a missing turn, a wrong boundary, "
          "two turns in one). Nothing changes; describe it in `note`.",
          "- `note`: anything a later reader needs.", "",
          "A row counts as reviewed only when `human_value` or `resolution` is filled; rows left blank are not "
          "treated as confirmed. Times are clock strings (`1:48`). A code that cannot be given a value takes "
          f"`missing:<reason>` with one of: {', '.join(r for r in MISSING_REASONS if r != 'none')}.", "",
          *(["Changing a turn's `function` to `question` or `answer` needs that function's codes in the same sheet: "
             "add a row per code with `path` set to `turn:<id>/code:<CODE>` and the value in `human_value`. Codes the "
             "new function does not take are removed.", ""] if "turn_function" in classes else []),
          "## Presets", "",
          *[f"- `{name}`{' (this sheet)' if name == entry['preset'] else ''}: {purpose}."
            for name, purpose in PRESET_PURPOSE.items()], "",
          "## Allowed values", "",
          "For the kinds of row on this sheet:", ""]

    def allowed(field_class: str) -> bool:
        return field_class in classes

    if allowed(OUTCOME):
        L.append(f"- `{OUTCOME}` (`{OUTCOME_PATH}`): {', '.join(OUTCOME_VALUES)}. Did any investor make or join an "
                 "offer at any point (`yes`), did every investor leave without one (`no`), or does the footage not "
                 "show the panel decide (`unknown`)? " + (
                     "Your answer is compared with the model's; the record is not edited." if blind else
                     "An answer that differs from the record is stored as an override; the record's decisions are "
                     "not edited."))
    if allowed(CUT):
        L.append(f"- `{CUT}` (`{CUT_PATH}`): the clock (`M:SS`) of the first decision any investor speaks, an offer "
                 "or an exit, or empty when there is none. " + (
                     "A clock within five seconds of the model's is the same moment." if blind else
                     "A clock within five seconds of the record confirms it; another clock is stored beside the "
                     "record, whose decisions are not edited. The row is flagged `passes_differ` only when the two "
                     f"passes' first decisions are more than {flag_s:g} seconds apart (the pass-to-pass decision "
                     "tolerance), a coarser check than the five seconds an answer is matched within."))
    if allowed(UNIT_PRESENCE) and not blind:
        L.append(f"- `{UNIT_PRESENCE}`: `present` or `absent`. The row asks whether the footage shows this turn, "
                 "exchange or decision at all, not what its values are. Nothing is added to or removed from the "
                 f"record; the judgment is logged. A path starting `{REPLICATE_ONLY}:` is a unit only the replicate "
                 "recorded: `present` says the primary missed it.")
    if allowed("actor"):
        L.append(f"- `actor`: `label` is free text; `role` is one of {', '.join(ACTOR_ROLES)}")
    if allowed("turn_function"):
        L.append(f"- `turn_function` (TURN-FUNC): {', '.join(books.turn_functions())}")
    # a changed function needs the new function's codes, so the codes are listed beside the functions
    if allowed("turn_code") or allowed("turn_function"):
        L.append("- `turn_code`: `other_concept` is a few words naming the matter when the concept is `other`; the "
                 "codes:")
        L += _code_lines(books)
    # the exchange bullets appear only when the sheet carries rows of those classes (the full-video sheet draws no
    # exchange-topic cards by default since 2026-10-21; the flag-driven presets still do)
    if allowed("turn_exchange"):
        L.append("- `turn_exchange`: the id of an exchange of this record, or empty when the turn belongs to none")
    if allowed("turn_link"):
        L.append("- `turn_link`: ids of the earlier turns this one responds to, separated by commas, or empty")
    if allowed("exchange_kind"):
        L.append(f"- `exchange_kind` (EX-KIND): {', '.join(books.exchange_kinds())}")
    if allowed("exchange_topic"):
        L.append(f"- `exchange_topic` (EX-TOPIC): {', '.join(books.exchange_topics())}; `label` is a few words of free "
                 f"text. Answer `{NOT_AN_EXCHANGE}` when the turns are not one line of questioning at all (logged, "
                 "nothing changes)")
    if allowed("decision"):
        L.append(f"- `decision`: `kind` (DEC-KIND) is one of {', '.join(books.decision_kinds())}; `actor` is an actor "
                 "id (a founder for a `founder_...` kind, an investor otherwise); `time` is a clock time inside the "
                 "video, written `M:SS` (a spreadsheet may turn `1:48` into `1:48:00`, which is refused on a video "
                 "shorter than an hour); `offer_id` is the label shared by the moves of one offer, or empty. A kind "
                 "that moves an offer needs an offer id: correct `kind`, `actor` and `offer_id` of one decision in "
                 "the same sheet when they depend on each other")
    if allowed("stage_a"):
        L.append(f"- `stage_a`: an integer 1-5, or empty when the item cannot be rated. A changed value adds "
                 f"`{STAGE_A_CHANGED}` to the item's note, which was written for the model's value")
    if allowed("venture"):
        L.append("- `venture`:")
        L += [f"  - `{name}` ({code_id}): {', '.join(books.venture_values(code_id))}"
              for name, code_id in VENTURE_CODES.items() if name != ASK]
        L.append(f"  - `{ASK}` ({VENTURE_CODES[ASK]}): whole US dollars and per cent of equity, `<usd>;<pct>` (for "
                 "example `200000;10`; `$200,000; 10%` is read the same), or `missing:<reason>`")
    if allowed("prep_pres"):
        L.append("- `prep_pres`: a prepared presentation code, `prep_pres:<CODE>`: a rated code takes its scale, a "
                 "count a whole number of 0 or more; either may be `missing:<reason>`"
                 + (f". On this blind sheet `{EVID_PATH}` takes a band, {', '.join(f'`{b}`' for b in EVID_BANDS)}"
                    if blind else ""))
    if allowed("prepared_window"):
        L.append("- `prepared_window`: no value on a blind sheet: `structural` when the prepared presentation is not "
                 "where the coding page shows it, with where it starts and ends in `note` (for example `0:15-1:43`)"
                 if blind else
                 "- `prepared_window`: the corrected window as clock ranges separated by `;`, for example `0:15-1:43`")
    L.append("")
    return "\n".join(L)


def export_video(vdir: Path, out_dir: Path, books: PanelBooks, args) -> dict:
    video_id = vdir.name
    results = vdir / "results"
    sheet = out_dir / f"{video_id}__review.csv"
    if sheet.exists() and not args.force:
        raise ConfigError(f"{sheet} exists and may hold a coder's work; pass --force to replace it")
    # the sheet is drawn from the pass the analysis reads (final review I1): the primary, or the replicate when the
    # primary fails validation and the replicate fails less; the rows' `primary` holds that pass's value and
    # `replicate` the other pass's, and the manifest records which pass (`pass`)
    drawn = model_pass(vdir)
    other = "replicate" if drawn == "primary" else "primary"
    record = _record(_read(results / f"{drawn}.json"), results / f"{drawn}.json")
    replicate_path = results / f"{other}.json"
    replicate = _record(_read(replicate_path), replicate_path) if replicate_path.exists() else None
    experiment = _optional(vdir / "experiment.json") or {}
    agreement = _optional(vdir / "agreement.json") if replicate is not None else None
    validation = _optional(results / f"{drawn}_validation.json") or {}
    asked = tuple(args.field_classes) if args.field_classes else None
    try:
        if args.preset == EXCHANGE:
            selection = select_exchange_sheet(
                record, replicate, agreement, validation, books, seed=args.seed, key=video_id,
                stage_a=args.stage_a_per_video, exchanges=args.exchanges_per_video, qa=args.qa_per_video,
                venture=args.venture_per_video, qa_codes=args.qa_codes,
                field_classes=asked)
        else:
            selection = select_sheet(
                record, replicate, agreement, validation, books,
                sample_unflagged=args.sample_unflagged, seed=args.seed, max_flagged=args.max_flagged,
                # the complete sheet takes every class whole, with the fields that are otherwise left off
                field_classes=asked or (SHEET_CLASSES if args.preset == "full" else None),
                # a unit_presence row exists only because the other pass lacks the unit (or has one the primary
                # lacks): either way it shows what the other pass holds, so a blind sheet has none
                replicate_only=not args.blind, presence=not args.blind)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    if args.planned and selection.get("sample") is not None:
        selection["sample"] = {**selection["sample"], "planned": args.planned}
    rows = selection["rows"]
    canonical = canonical_full(record, books)

    by_class = {c: sum(r["field_class"] == c for r in rows) for c in SHEET_CLASSES}
    by_reason: dict[str, int] = {}
    for r in rows:
        for reason in r["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    entry = {
        "video_id": video_id,
        "sheet": sheet.name,
        "release_id": experiment.get("release_id"),
        "annotation_context": ANNOTATION_CONTEXT,
        "seed": args.seed,
        "sample_unflagged": args.sample_unflagged,
        "max_flagged": args.max_flagged,
        "preset": args.preset,
        "field_classes": list(args.field_classes) if args.field_classes else None,
        "blind": bool(args.blind),
        # the pass the rows were drawn from (and the cut read from); `primary` in each row is its value
        "pass": drawn,
        "primary_sha256": sha256_file(results / "primary.json"),
        "replicate_sha256": sha256_file(results / "replicate.json") if (results / "replicate.json").exists() else None,
        "counts": {"by_field_class": {c: n for c, n in by_class.items() if n},
                   "by_selection": {s: sum(r["selected_as"] == s for r in rows) for s in (FLAGGED, SAMPLED)},
                   "by_priority": {str(n): sum(r["priority"] == n for r in rows) for n in (1, 2, 3, 4)},
                   "by_reason": dict(sorted(by_reason.items()))},
        # what was flagged, what the sheet took and what --max-flagged left out, to reweight the flagged rows
        # (None for the exchange preset, which selects nothing by flag)
        "flagged": selection.get("flagged"),
        # the exchange preset's draw: asked for, drawn, and per field class the rows and the population behind them
        "sample": selection.get("sample"),
        # the flag-driven presets' unflagged pool: per field class the population the sampled rows were drawn from
        # (in proportion), the rows drawn, and the fields left out because the other pass lacks their unit
        "unflagged": selection.get("unflagged"),
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(sheet, "w", newline="", encoding=SHEET_ENCODING) as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for number, row in enumerate(rows, 1):
            allowed = allowed_values(row["path"], canonical, books, coder=args.blind)     # a blind sheet is a coder's
            if allowed and row["field_class"] == "exchange_topic" and row["path"].endswith("/topic"):
                allowed = (*allowed, NOT_AN_EXCHANGE)
            cells = {"row": number, **row, "reasons": "; ".join(row["reasons"]),
                     "allowed_values": " | ".join(allowed) if allowed else "",
                     "human_value": "", "resolution": "", "note": ""}
            if args.blind:
                start, end = locator_span(canonical, row["path"])
                cells.update(dict.fromkeys(MODEL_COLUMNS, ""),
                             context=row.get("blind_context") or locator(canonical, row["path"]),
                             clock_start=start, clock_end=end)
            writer.writerow([_one_line(cells[c]) for c in COLUMNS])
    (out_dir / f"{video_id}__review.md").write_text(
        instructions(video_id, experiment, canonical, books, entry, vdir.parent,
                     decision_tolerance_s=Tolerances.of(agreement).decision_s), encoding="utf-8")
    manifest = _optional(out_dir / MANIFEST) or {}
    videos = dict(manifest.get("videos") or {})
    videos[video_id] = entry
    _dump(out_dir / MANIFEST, {"videos": dict(sorted(videos.items()))})
    return entry


SAMPLE_SHEET = "video_sample.csv"
SAMPLE_COLUMNS = ["video_id", "title", "file_name", "duration_s", "duration_clock", "draw", "seed"]    # no path
ROWS_PER_FIELD = 60     # rows wanted per sampled field when --sample-videos draws the videos and no target is given


def plan_draw(rows_per_field: int, n_videos: int) -> dict[str, int]:
    """How many question-answer pairs to draw per video so that each sampled field reaches `rows_per_field` rows
    over `n_videos` videos: the ceiling of the ratio, at least one. Lines of questioning are planned the same way
    only when the sheet draws them by default (it does not since 2026-10-21): a plan never turns the exchange cards
    on; --exchanges-per-video does."""
    if rows_per_field < 1 or n_videos < 1:
        raise ConfigError("--rows-per-field and the number of videos must be at least 1")
    per_video = max(1, -(-rows_per_field // n_videos))
    return {"exchanges": per_video if EXCHANGE_DEFAULTS["exchanges"] else 0, "qa": per_video}


def write_video_sheet(path: Path, run: Path, groups: list[tuple[object, object, list[str]]]) -> Path:
    """A list of videos as a spreadsheet (opens in Excel), the columns of SAMPLE_COLUMNS: one row per video of each
    (draw, seed, video ids) group, in order. Title, file name and duration come from each video's experiment.json.
    It never carries a video's outcome stratum (final review C1, 2026-10-22): a stratified draw's strata are in
    video_sample.json only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding=SHEET_ENCODING) as fh:
        writer = csv.writer(fh)
        writer.writerow(SAMPLE_COLUMNS)
        for draw, seed, video_ids in groups:
            for video_id in video_ids:
                experiment = _optional(run / video_id / "experiment.json") or {}
                source = experiment.get("source") or ""
                duration = experiment.get("duration_s")
                writer.writerow([video_id, Path(source).stem if source else video_id, Path(source).name if source else "",
                                 duration if duration is not None else "",
                                 seconds_to_clock(float(duration)) if duration is not None else "", draw, seed])
    return path


def _stratum_of(sample: dict) -> dict[str, str] | None:
    """video id -> its stratum, over the sample's stratified draws; None when no draw is stratified."""
    draws = [d for d in sample.get("draws") or [] if d.get("stratify")]
    if not draws:
        return None
    return {v: name for d in draws for name, stratum in d["strata"].items() for v in stratum["chosen"]}


def write_sample_sheet(run: Path, sample: dict) -> Path:
    """The sampled videos as a spreadsheet (opens in Excel), the list that may be sent to the coders: ids, titles,
    source file names and durations, never the outcome stratum (final review C1; the strata stay in
    video_sample.json, which is not sent)."""
    return write_video_sheet(run / REVIEW_DIR / SAMPLE_SHEET, run,
                             [(number, draw["seed"], draw["chosen"]) for number, draw in enumerate(sample["draws"], 1)])


STRATIFY = ("any_offer",)       # what --stratify can split the sample by: the pitch's outcome as the record has it
STRATA = ("yes", "no")          # its strata, in the order they are drawn and reported (an unknown outcome is neither)


def video_outcome(run: Path, video_id: str) -> str:
    """Whether any investor of the video's analysed record (the pass panel_pipeline.model_pass names) made or joined an
    offer (review_items.any_offer: yes, no or unknown), as the record has it: a reviewed verdict does not move a video
    between strata."""
    path = run / video_id / "results" / f"{model_pass(run / video_id)}.json"
    try:
        record = _read(path)["record"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: cannot read the record's outcome for --stratify ({exc})") from exc
    return any_offer(record)


def _stratified_draw(run: Path, eligible: list[str], n: int, seed: int, stratify: str) -> dict:
    """The strata part of a stratified draw: n/2 videos drawn at random (seeded per stratum) from the eligible pitches
    with an offer and n/2 from those without; refused for an odd n or a stratum with too few eligible pitches."""
    if stratify not in STRATIFY:
        raise ConfigError(f"--stratify {stratify}: the sample can be stratified by {', '.join(STRATIFY)}")
    if n % 2:
        raise ConfigError(f"--sample-videos {n} --stratify {stratify}: a stratified sample takes an even number of "
                          "videos, half from the pitches with an offer and half from those without")
    outcome = {v: video_outcome(run, v) for v in eligible}
    strata = {}
    for name in STRATA:
        members = sorted(v for v in eligible if outcome[v] == name)
        if len(members) < n // 2:
            raise ConfigError(f"--sample-videos {n} --stratify {stratify}: {len(members)} eligible pitches "
                              f"{'with' if name == 'yes' else 'without'} an offer, fewer than the {n // 2} wanted")
        chosen = sorted(random.Random(f"{seed}:videos:{stratify}={name}").sample(members, n // 2))
        strata[name] = {"eligible": members, "n_eligible": len(members), "chosen": chosen}
    return {"stratify": stratify, "strata": strata,
            "unknown": sorted(v for v in eligible if outcome[v] not in STRATA)}


def sample_strata(run: Path) -> dict | None:
    """The run's validation sample as strata, for the agreement figures per stratum and reweighted to the run's mix:
    {"stratify", "of": {video: stratum}, "run": {stratum: eligible pitches in the run}}; None when the sample is not
    stratified. Every draw must be stratified by the same variable (a mixed sample is reported unstratified, with a
    warning). The run's sizes are those of the first draw, made before any video was taken."""
    sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE) or {}
    draws = sample.get("draws") or []
    kinds = {d.get("stratify") for d in draws}
    if not draws or kinds == {None}:
        return None
    if len(kinds) > 1:
        print(f"warning: {run / REVIEW_DIR / VIDEO_SAMPLE} mixes stratified and unstratified draws (or two "
              "stratifications); agreement is reported unstratified", file=sys.stderr)
        return None
    return {"stratify": draws[0]["stratify"], "of": _stratum_of(sample),
            "run": {name: draws[0]["strata"][name]["n_eligible"] for name in STRATA}}


def sample_videos(run: Path, n: int, seed: int, force: bool, exclude: set[str] = frozenset(),
                  append: bool = False, stratify: str | None = None) -> list[str]:
    """`n` videos of the run drawn at random (seeded) from those with both passes and not excluded; recorded
    in <run>/review/video_sample.json as one draw. `append` adds a draw to an existing sample (new videos
    only); without it a different sample is refused unless `force` replaces the file.

    `stratify` ("any_offer", plan C Task 7f, 2026-10-21) draws n/2 of the eligible pitches with an offer and n/2 of
    those without (the outcome as the analysed record has it; a pitch whose outcome is unknown is in neither), each
    stratum seeded on its own; the draw records `stratify`, per stratum its eligible pitches, their number and the
    videos chosen, and the eligible pitches of unknown outcome. An odd n or a stratum with too few pitches is refused.
    Without it the draw is the plain one, recorded as before.

    A call whose seed, n, excluded set and stratification are those of a recorded draw is that draw again: its
    videos are returned and nothing is drawn or written, so a re-run of an export never grows the sample. (`force`
    without `append` still replaces the sample.)"""
    path = run / REVIEW_DIR / VIDEO_SAMPLE
    earlier = _optional(path)
    if earlier is not None and (append or not force):
        for number, recorded in enumerate(earlier.get("draws") or [], 1):
            if (recorded.get("seed"), recorded.get("n"), sorted(recorded.get("excluded") or []),
                    recorded.get("stratify")) == (seed, n, sorted(exclude), stratify):
                print(f"video sample: draw {number} of {path} already records seed {seed}, n {n}, these "
                      f"exclusions and {'--stratify ' + stratify if stratify else 'no stratification'}; it is returned "
                      "unchanged and nothing new is drawn")
                return list(recorded["chosen"])
    taken = set(earlier.get("chosen") or []) if earlier and append else set()
    eligible = sorted(p.name for p in _video_dirs(run, None)
                      if (p / "results" / "replicate.json").exists() and p.name not in exclude and p.name not in taken)
    if not 0 < n <= len(eligible):
        raise ConfigError(f"--sample-videos {n}: {len(eligible)} videos are eligible (both passes, not excluded"
                          f"{', not drawn before' if append else ''})")
    if stratify is None:
        chosen = sorted(random.Random(f"{seed}:videos").sample(eligible, n))
        draw = {"seed": seed, "n": n, "excluded": sorted(exclude), "eligible": eligible, "chosen": chosen}
    else:
        strata = _stratified_draw(run, eligible, n, seed, stratify)
        chosen = sorted(v for stratum in strata["strata"].values() for v in stratum["chosen"])
        draw = {"seed": seed, "n": n, "excluded": sorted(exclude), "eligible": eligible, "chosen": chosen, **strata}
    if earlier is not None and not append and not force:
        if earlier.get("chosen") != chosen:
            recorded = (earlier.get("draws") or [{}])[0].get("stratify")
            raise ConfigError(f"{path} already records a sample of {len(earlier.get('chosen') or [])} videos "
                              f"({'stratified by ' + recorded if recorded else 'not stratified'}); the same draw needs "
                              "the same --sample-videos, --seed and --stratify; --append adds a draw; --force replaces")
    draws = (earlier.get("draws") or []) + [draw] if append and earlier else [draw]
    path.parent.mkdir(parents=True, exist_ok=True)
    sample = {"draws": draws, "chosen": [v for d in draws for v in d["chosen"]]}
    _dump(path, sample)
    write_sample_sheet(run, sample)
    return chosen


CLIPS_REFUSED = ("--clips: since 2026-10-21 the clip session plays each pitch's full video at its prepared window, "
                 "so no clips are cut; copy the full videos with `review_panel.py copy-videos --run {run} --to "
                 "<folder>` (add --training for the training session)")


def cmd_copy_videos(args, books: PanelBooks) -> int:
    """The full videos to hand to the coders: the validation sample's and the clip session's (the session plays these
    same files at their prepared windows), or with --training the training videos (its clip session's among them)."""
    run, target = Path(args.run), Path(args.to)
    if getattr(args, "clips", False):
        raise ConfigError(CLIPS_REFUSED.format(run=run))
    if args.training:
        manifest = _optional(run / REVIEW_DIR / TRAINING_DIR / MANIFEST)
        if not manifest or not manifest.get("videos"):
            raise ConfigError(f"no training export at {run / REVIEW_DIR / TRAINING_DIR / MANIFEST}; run "
                              "`review_panel.py training` first")
        video_ids = sorted(manifest["videos"])
    else:
        sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE)
        session = _optional(_clips_dir(run) / review_clips.MANIFEST)
        if not sample and not session:
            raise ConfigError(f"{run / REVIEW_DIR / VIDEO_SAMPLE} not found; draw the sample first (export --sample-videos)")
        # the clip session's videos are the sample's unless clips-export named others
        video_ids = list(dict.fromkeys([*((sample or {}).get("chosen") or []), *((session or {}).get("videos") or [])]))
    target.mkdir(parents=True, exist_ok=True)
    copied, size, missing, failed = 0, 0, [], []
    for video_id in video_ids:
        source = Path((_optional(run / video_id / "experiment.json") or {}).get("source") or "")
        if not source.is_file():
            missing.append(f"{video_id}: {source if source.name else 'no source in experiment.json'}")
            continue
        dest = target / source.name
        try:
            if dest.exists():
                if dest.stat().st_size != source.stat().st_size:
                    failed.append(f"{video_id}: collision: {dest} exists with another size "
                                  f"({dest.stat().st_size} bytes, the source has {source.stat().st_size}); "
                                  "it was left as it is")
                    continue
            else:
                (os.link if args.link else shutil.copy2)(source, dest)
            size += dest.stat().st_size
        except OSError as exc:
            failed.append(f"{video_id}: {source.name}: {exc}")
            continue
        copied += 1
    print(f"{copied} of {len(video_ids)} videos in {target} ({size / 1e9:.2f} GB)")
    for line in missing:
        print(f"missing: {line}", file=sys.stderr)
    for line in failed:
        print(f"failed: {line}", file=sys.stderr)
    return 1 if missing or failed else 0


def _qa_codes(items: list[str]) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in items:
        role, sep, code = item.partition(":")
        if not sep or not role or not code:
            raise ConfigError(f"--qa-codes takes ROLE:CODE items such as question:PR-SEV; got {item!r}")
        pairs.append((role, code))
    return tuple(pairs)


def cmd_export(args, books: PanelBooks) -> int:
    run = Path(args.run)
    release_id = load_release().release_id
    args.preset = args.preset or EXCHANGE
    flag_options = {"--max-flagged": args.max_flagged, "--sample-unflagged": args.sample_unflagged}
    draw_options = {"--stage-a-per-video": "stage_a", "--exchanges-per-video": "exchanges", "--qa-per-video": "qa",
                    "--venture-per-video": "venture"}
    if args.preset == EXCHANGE:
        given = [name for name, value in flag_options.items() if value is not None]
        if given:
            raise ConfigError(f"{', '.join(given)} belong to the flag-driven presets; the exchange preset selects "
                              f"nothing by flag. Add --preset {' | '.join(PRESETS)}")
        for name, part in draw_options.items():
            attr = name[2:].replace("-", "_")
            if getattr(args, attr) is None:
                if part not in ("exchanges", "qa"):
                    setattr(args, attr, EXCHANGE_DEFAULTS[part])
            elif getattr(args, attr) == ALL and attr == "stage_a_per_video":
                continue
            else:
                try:
                    setattr(args, attr, int(getattr(args, attr)))
                except (TypeError, ValueError):
                    raise ConfigError(f"{name} takes a whole number" + (" or 'all'" if attr == "stage_a_per_video" else ""))
                if getattr(args, attr) < 0:
                    raise ConfigError(f"{name} cannot be negative")
        args.qa_codes = _qa_codes(args.qa_codes) if args.qa_codes else QA_CODES
    else:
        given = [name for name in draw_options if getattr(args, name[2:].replace("-", "_")) is not None]
        if args.qa_codes is not None:
            given.append("--qa-codes")
        if given:
            raise ConfigError(f"{', '.join(given)} belong to --preset {EXCHANGE}, not to --preset {args.preset}")
        preset_cap, preset_sample = PRESETS[args.preset]
        if args.sample_unflagged is None:
            args.sample_unflagged = preset_sample
        if args.max_flagged is None:
            args.max_flagged = preset_cap
        elif args.max_flagged < 0:
            args.max_flagged = None
    if args.sample_videos is not None:
        if args.video_ids:
            raise ConfigError("--sample-videos draws the videos; it cannot be combined with --video-ids")
        exclude = set(args.exclude_video_ids or [])
        if args.exclude_from:
            selection = _read(run / args.exclude_from)
            if not isinstance(selection, dict) or args.exclude_key not in selection:
                keys = ", ".join(sorted(map(str, selection))) if isinstance(selection, dict) and selection else "none"
                raise ConfigError(f"{run / args.exclude_from} has no key {args.exclude_key!r} (--exclude-key), so "
                                  f"nothing would be excluded; its keys: {keys}")
            exclude |= set(selection[args.exclude_key] or [])
        args.video_ids = sample_videos(run, args.sample_videos, args.seed, args.force, exclude, args.append,
                                       args.stratify)
        split = ""
        if args.stratify:
            of = _stratum_of(_optional(run / REVIEW_DIR / VIDEO_SAMPLE) or {}) or {}
            yes = sum(of.get(v) == "yes" for v in args.video_ids)
            split = f": {yes} with an offer, {len(args.video_ids) - yes} without, stratified by {args.stratify}"
        print(f"video sample: {len(args.video_ids)} videos (seed {args.seed}){split} -> {run / REVIEW_DIR / VIDEO_SAMPLE}")
    elif args.stratify:
        raise ConfigError("--stratify belongs to --sample-videos: it says how the sample's videos are drawn")
    vdirs = _video_dirs(run, args.video_ids)
    args.planned = None
    if args.preset == EXCHANGE:
        # the target applies when it is asked for, or when this call draws the videos; a plain export keeps the defaults
        target = args.rows_per_field if args.rows_per_field is not None else (
            ROWS_PER_FIELD if args.sample_videos is not None else None)
        # the plan is recorded only when it sets the number of pairs: --qa-per-video overrides it (final review I2)
        if target is not None and args.qa_per_video is None:
            plan = plan_draw(target, len(vdirs))
            args.planned = {"rows_per_field": target, "n_videos": len(vdirs), **plan}
        for part in ("exchanges", "qa"):
            if getattr(args, f"{part}_per_video") is None:
                setattr(args, f"{part}_per_video", args.planned[part] if args.planned else EXCHANGE_DEFAULTS[part])
    for vdir in vdirs:
        out_dir = Path(args.out) if args.out else vdir / REVIEW_DIR
        entry = export_video(vdir, out_dir, books, args)
        if entry["release_id"] not in (None, release_id):
            print(f"warning: {vdir.name} was annotated with release {entry['release_id']}; values are checked "
                  f"against the current release {release_id}", file=sys.stderr)
        counts = entry["counts"]["by_selection"]
        left = entry["flagged"]["left_out"]["n"] if entry["flagged"] else 0
        print(f"{vdir.name}: {len(entry['rows'])} rows, preset {args.preset} ({counts[FLAGGED]} flagged"
              + (f" of {entry['flagged']['found']['n']}, {left} left out by --max-flagged {args.max_flagged}"
                 if left else "")
              + f", {counts[SAMPLED]} sampled, seed {args.seed}{', blind' if args.blind else ''}) -> "
              f"{out_dir / entry['sheet']}")
    return 0


# ---- import -------------------------------------------------------------------------------

def _log_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Older logs can recover context only from the export's pinned pass and hash; never guess a current best pass.
    if path.name == LOG and any(l.get('path') == CUT_PATH and 'decision_turn_context' not in l for l in lines):
        vdir = path.parent.parent
        for candidate in (path.parent / MANIFEST, vdir.parent / REVIEW_DIR / MANIFEST):
            entry = ((_optional(candidate) or {}).get('videos') or {}).get(vdir.name)
            if not entry:
                continue
            context = _decision_context(vdir, entry)
            if context:
                for line in lines:
                    if line.get('path') == CUT_PATH and 'decision_turn_context' not in line:
                        line['decision_turn_context'] = context
                break
    return lines


def _decision_context(vdir: Path, entry: dict) -> dict | None:
    drawn = entry.get('pass') or 'primary'
    path = vdir / 'results' / f'{drawn}.json'
    digest = entry.get(f'{drawn}_sha256')
    if not digest or not path.exists() or sha256_file(path) != digest:
        return None
    return decision_turn_context(_read(path)['record'], drawn, digest)


ROLES = ("coder", "adjudicator")


def _role(line: dict) -> str:
    return line.get("role", "adjudicator")      # a line written before roles existed was an adjudicator's


def _reviewed_lines(lines: list[dict], role: str | None = None) -> dict[str, dict]:
    """The last log line of every path reviewed by lines of `role` (every role when None); not the codes a
    function change removed."""
    last: dict[str, dict] = {}
    for line in lines:
        if not line.get("implied") and (role is None or _role(line) == role):
            last[line["path"]] = line
    return last


def _lines_by_reviewer(lines: list[dict]) -> dict[str, dict[str, dict]]:
    """coder -> the last log line per path that coder gave. An adjudicator is no coder, even under a coder's name."""
    out: dict[str, dict[str, dict]] = {}
    for line in lines:
        if not line.get("implied") and _role(line) == "coder":
            out.setdefault(line["reviewer"], {})[line["path"]] = line
    return out


def _too_many_coders(video_id: str) -> str:
    return f"more than two coders on {video_id}: coder-against-coder is not computed"


def _agreement_of(last: list[dict], left_out: dict | None = None, population: dict | None = None,
                  tolerances: Tolerances | None = None) -> dict:
    corrections = [Correction(e["path"], e["new"], e["resolution"], e.get("note") or "", e.get("reviewer") or "",
                              _role(e))
                   for e in last]
    return human_agreement(last, corrections, left_out, population, tolerances)


def _agreement_from_log(lines: list[dict], left_out: dict | None = None, population: dict | None = None,
                        tolerances: Tolerances | None = None) -> dict:
    return _agreement_of(validity_lines(lines), left_out, population, tolerances)


NO_FIRST_DECISION = "none"      # pitch_override["first_decision"] when the footage shows no decision at all


def _overrides(lines: list[dict], path: str) -> int:
    """How many log lines override the record at one pitch path (the outcome, or the first decision's clock)."""
    return sum(line["path"] == path and line["resolution"] == OUTCOME_OVERRIDE for line in lines)


def _pitch_override(lines: list[dict]) -> dict[str, str | None]:
    """{"any_offer": yes|no|unknown, "first_decision": clock}: the adjudicator's verdict on the pitch where its last
    adjudicated review differs from what the record's decisions give; None where it does not (no override). A
    verdict that the footage shows no first decision at all (an override with an empty value) is stored as
    "none" (NO_FIRST_DECISION), apart from None. A coder's judgment never corrects the record."""
    last = _reviewed_lines(lines, role="adjudicator")

    def verdict(path: str) -> str | None:
        line = last.get(path)
        if line is None or line["resolution"] != OUTCOME_OVERRIDE:
            return None
        return line["new"] if line["new"] is not None or path != CUT_PATH else NO_FIRST_DECISION
    return {"any_offer": verdict(OUTCOME_PATH), "first_decision": verdict(CUT_PATH)}


def _changed(line: dict, tolerances: Tolerances | None = None) -> bool:
    """Whether a log line is a correction: it wrote a value that is another value than the one before (a clock
    within tolerance of the old one is the same value; a structural or unclear line never writes)."""
    return (line["resolution"] in ("confirm_replicate", "correct")
            and not agrees(line["path"], line["old"], line["new"], tolerances))


def read_sheet(sheet: Path) -> list[dict]:
    """The rows of a filled sheet, however the coder's spreadsheet saved it: UTF-8 (with or without the mark),
    columns separated by a comma, a semicolon or a tab. Anything else is one clear ConfigError."""
    try:
        text = sheet.read_bytes().decode(SHEET_ENCODING)
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{sheet} is not UTF-8 (byte {exc.start}: {exc.reason}); save it as 'CSV UTF-8' and "
                          "import it again") from None
    header = text.splitlines()[0] if text.strip() else ""
    for delimiter in DELIMITERS:
        columns = [c.strip() for c in next(csv.reader([header], delimiter=delimiter), [])]
        if "path" in columns and "human_value" in columns:
            rows = list(csv.DictReader(text.splitlines(keepends=True), delimiter=delimiter))
            return [{(k or "").strip(): v for k, v in row.items()} for row in rows]
    raise ConfigError(f"{sheet} has no `path` column: its first line is not the sheet's header with a comma, "
                      "semicolon or tab between the columns; export it again or save it as CSV")


def write_group(files: dict[Path, str]) -> None:
    """Write several files together or not at all: each to a temporary file first, then all put in place.

    When anything fails, the files already put in place are restored from their backups (or removed
    when they did not exist before), every temporary file is removed, and the error is raised.
    """
    temps = {path: path.with_name(path.name + TEMP_SUFFIX) for path in files}
    backups = {path: path.with_name(path.name + BACKUP_SUFFIX) for path in files}
    placed: list[Path] = []
    try:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temps[path].write_text(text, encoding="utf-8")
        for path in files:
            if path.exists():
                os.replace(path, backups[path])
            placed.append(path)
            os.replace(temps[path], path)
    except BaseException:
        for path in placed:
            if path.exists() and backups[path].exists():
                path.unlink()
            if backups[path].exists():
                os.rename(backups[path], path)
            elif path.exists():
                path.unlink()                   # it was not there before this import
        for temp in temps.values():
            temp.unlink(missing_ok=True)
        raise
    for backup in backups.values():
        backup.unlink(missing_ok=True)


def _sheet_rows(sheet: Path, entry: dict, primary: dict) -> list[dict]:
    """The sheet's rows with the model's side taken from the manifest, whatever the sheet's own cells say.
    Paths are normalised, so `turn:0005/function` finds the exported row `turn:5/function`."""
    rows = read_sheet(sheet)
    exported = {normalize_path(row["path"]): row for row in entry["rows"]}
    out = []
    for number, row in enumerate(rows, 1):
        path = normalize_path(row.get("path"))
        merged = {"row": (row.get("row") or "").strip() or str(number), "path": path,
                  "human_value": row.get("human_value"), "resolution": row.get("resolution"), "note": row.get("note")}
        known = exported.get(path)
        if known is not None:
            merged.update({k: known[k] for k in ("field_class", "primary", "replicate", "selected_as")})
        else:       # a row the coder added, for instance a code that a new function needs
            presence = presence_ref(path)
            try:
                value = presence.primary_value if presence else read_value(primary, path)
                merged.update(field_class=field_class(path), primary=value)
            except ValueError:
                merged.update(field_class=None, primary=None)
            merged.update(replicate=presence.replicate_value if presence else None, selected_as=ADDED)
        out.append(merged)
    return out


def _import_files(vdir: Path, entry: dict, log: list[dict], corrections: list[Correction], by_path: dict[str, dict],
                  sheet_name: str, role: str, tolerances: Tolerances, books: PanelBooks
                  ) -> tuple[list[dict], list[dict], int, dict[Path, str]]:
    """The log lines one import adds to a video's review log, every line of the log after it, the import's number,
    and the two files that record it (the log and human_agreement.json), not yet written."""
    video_id = vdir.name
    log_path = vdir / REVIEW_DIR / LOG
    before = _log_lines(log_path)
    number = max((line.get("import", 0) for line in before), default=0) + 1
    asked = {c.path for c in corrections}
    imported_at = _now()
    fresh = []
    turn_context = _decision_context(vdir, entry) if any(e['path'] == CUT_PATH for e in log) else None
    for e in log:
        row = by_path.get(e["path"]) or {}
        fresh.append({**e, "implied": e["path"] not in asked,
                      **({'decision_turn_context': turn_context} if e['path'] == CUT_PATH else {}),
                      "field_class": row.get("field_class") or field_class(e["path"]),
                      "primary": row.get("primary"), "replicate": row.get("replicate"),
                      "selected_as": row.get("selected_as") or ADDED, "sheet": sheet_name, "import": number,
                      "role": role, "imported_at": imported_at,
                      # the measurement level alpha reads the line at, from the codebooks (None: no alpha)
                      **({} if e["path"] not in asked else {"typed": str(row.get("human_value") or "").strip() or None,
                                                            "level": level_of(e["path"], books)})})
    lines = before + fresh
    left_out = (entry.get("flagged") or {}).get("left_out")
    population = (entry.get("sample") or entry.get("unflagged") or {}).get("population")
    coders = _lines_by_reviewer(lines)
    agreement = {"video_id": video_id, "annotation_context": ANNOTATION_CONTEXT,
                 **_agreement_from_log(lines, left_out, population, tolerances),
                 "by_reviewer": {name: _agreement_of(list(last.values()), left_out, population, tolerances)
                                 for name, last in coders.items()}}
    if len(coders) > 2:
        agreement["human_human_note"] = _too_many_coders(video_id)
    elif len(coders) == 2:
        agreement["human_human"] = human_human(coders, tolerances)
    earlier_log = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if earlier_log and not earlier_log.endswith("\n"):
        earlier_log += "\n"
    files = {
        log_path: earlier_log + "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in fresh),
        vdir / REVIEW_DIR / AGREEMENT: _json_text(agreement),
    }
    return fresh, lines, number, files


def cmd_import(args, books: PanelBooks) -> int:
    run, sheet = Path(args.run), Path(args.sheet)
    _not_discussion(sheet)
    if not sheet.exists():
        raise ConfigError(f"sheet not found: {sheet}")
    manifest = _optional(sheet.parent / MANIFEST)
    if manifest is None:
        raise ConfigError(f"{sheet.parent / MANIFEST} not found; a sheet is imported from the directory it was "
                          "exported to, beside its manifest")
    videos = manifest.get("videos") or {}
    video_id = args.video_id or next((v for v, e in videos.items() if e.get("sheet") == sheet.name), None)
    if video_id is None or video_id not in videos:
        raise ConfigError(f"{sheet.parent / MANIFEST} lists no video "
                          + (f"{video_id!r}" if video_id else f"for the sheet {sheet.name}; pass --video-id"))
    entry = videos[video_id]
    vdir = run / video_id
    drawn = entry.get("pass") or "primary"          # the pass the sheet was drawn from (an older manifest: primary)
    primary_path = vdir / "results" / f"{drawn}.json"
    if not primary_path.exists():
        raise ConfigError(f"no {drawn} result at {primary_path}")
    primary_sha = sha256_file(primary_path)
    if primary_sha != entry[f"{drawn}_sha256"]:
        raise ConfigError(f"{primary_path} has sha256 {primary_sha}, but the sheet was exported from "
                          f"{entry[f'{drawn}_sha256']}; the {drawn} result changed, export a new sheet")
    primary_result = _read(primary_path)
    primary = canonical_full(_record(primary_result, primary_path), books)
    args.reviewer = reviewer_name(args.reviewer)
    role = getattr(args, "role", None) or ("coder" if entry.get("blind") else "adjudicator")
    if role == "adjudicator" and entry.get("blind"):
        print(f"warning: {sheet.name} was exported blind; importing it as an adjudicator writes its values into "
              "the reviewed record", file=sys.stderr)
    reviewed_path = vdir / "results" / "reviewed.json"
    # a coder's sheet is checked against the primary record and never touches the reviewed record
    earlier = _optional(reviewed_path) if role == "adjudicator" else None
    if earlier is not None and earlier.get("based_on_sha256") != primary_sha:
        raise ConfigError(f"{reviewed_path} was built on another {drawn} result; move it away before importing")
    base = _record(earlier, reviewed_path) if earlier is not None else _record(primary_result, primary_path)
    experiment = _optional(vdir / "experiment.json") or {}
    if "duration_s" not in experiment:
        raise ConfigError(f"{vdir / 'experiment.json'} does not give duration_s, which validation needs")

    tolerances = Tolerances.of(_optional(vdir / "agreement.json"))      # the tolerances the selection flagged by
    rows = _sheet_rows(sheet, entry, primary.model_dump(mode="json"))
    if role == "coder":
        # a note on a card that was never answered (the coding page keeps what was typed) is no review; an
        # adjudicator's sheet with one is still refused, by parse_sheet
        answered = [row for row in rows if not (str(row.get("note") or "").strip()
                                               and not str(row.get("human_value") or "").strip()
                                               and not str(row.get("resolution") or "").strip())]
        if len(answered) != len(rows):
            print(f"{video_id}: {len(rows) - len(answered)} rows with a note but no answer were skipped")
        rows = answered
    for warning in unnamed_others(rows):          # imported as it is, with a warning
        print(f"{video_id}: warning: {warning}")
    by_path: dict[str, dict] = {}
    for row in rows:
        by_path.setdefault(row["path"], row)        # a path given twice is refused by parse_sheet, naming the row
    corrections, problems = parse_sheet(rows, args.reviewer, tolerances, role)
    if earlier is not None:
        current = canonical_full(base, books).model_dump(mode="json")
        for c in corrections:
            if c.resolution != "confirm_primary":
                continue
            try:
                now = read_value(current, c.path)
            except ValueError:
                continue        # reported by apply_corrections
            if not agrees(c.path, now, by_path[c.path]["primary"], tolerances):
                problems.append(f"row {by_path[c.path]['row']}: {c.path} confirms the primary value "
                                f"{by_path[c.path]['primary']!r}, but an earlier import corrected it to {now!r}; "
                                "use resolution correct with the value you want")
    if not corrections and not problems:
        problems.append("no row of the sheet is reviewed (human_value and resolution are empty everywhere)")
    reviewed = log = None
    try:
        reviewed, log = apply_corrections(base, corrections, books, duration_s=float(experiment["duration_s"]))
    except ReviewError as exc:
        for problem in exc.problems:
            row = next((by_path[c.path]["row"] for c in corrections if problem.startswith(f"{c.path}:")), None)
            problems.append(f"row {row}: {problem}" if row is not None else problem)
    if problems:
        for problem in problems:
            print(f"problem: {problem}", file=sys.stderr)
        print(f"{len(problems)} problem(s) in {sheet}; nothing was written", file=sys.stderr)
        return 1

    record = PanelRecord.model_validate(reviewed)
    fresh, lines, number, files = _import_files(vdir, entry, log, corrections, by_path, sheet.name, role, tolerances,
                                                books)
    counts = (f"{video_id}: {len(corrections)} rows reviewed, {sum(_changed(e, tolerances) for e in fresh)} "
              f"values changed, {_overrides(fresh, OUTCOME_PATH)} outcome overrides, "
              f"{_overrides(fresh, CUT_PATH)} first-decision overrides (import {number}, {args.reviewer} as {role})")
    added = sum(not e["implied"] and e["selected_as"] == ADDED for e in fresh)
    if added:
        print(f"{video_id}: {added} rows are not on the exported sheet and were logged as added")
    if role == "coder":
        # the coder's values are logged and measured; the record is neither read back nor written
        write_group(files)
        print(f"{counts}; the reviewed record is not touched")
        if counted_not_banded(corrections):
            print(f"{video_id}: {EVID_PATH} answered as a count, not a band ({', '.join(EVID_BANDS)}); it is banded "
                  "for agreement, but the coder answered off the sheet's scale")
        return 0
    report = validate_record(record, books, float(experiment["duration_s"]),
                             normalizations=primary_result.get("normalizations") or [])
    adjudicated = [line for line in lines if _role(line) == "adjudicator"]
    reviewed_result = {
        "based_on": drawn, "based_on_sha256": primary_sha,
        "reviewer": ", ".join(dict.fromkeys(line["reviewer"] for line in adjudicated)),
        "annotation_context": ANNOTATION_CONTEXT, "human_reviewed": HUMAN_REVIEWED,
        "corrections": sum(_changed(line, tolerances) for line in adjudicated),
        "prepared_window": prepared_window(record), "derived": derive(record, books),
        # a human's verdict on the pitch where it differs from the record's decisions (None where it does not):
        # analysis.panel_dataset.pitch_outcome applies any_offer; the first decision's clock is for the paper table
        # only and never moves the cut; the decisions below are not edited
        "pitch_override": _pitch_override(lines),
        "record": record.model_dump(mode="json"),
    }
    # the four files describe one import: all of them are put in place, or none is touched
    write_group({
        reviewed_path: _json_text(reviewed_result),
        vdir / "results" / "reviewed_validation.json": _json_text({"errors": report.errors, "flags": report.flags}),
        **files,
    })
    print(f"{counts}; validation: {len(report.errors)} errors, {len(report.flags)} flags -> {reviewed_path}")
    if report.errors:
        print(f"warning: the reviewed record of {video_id} has {len(report.errors)} validation errors; analysis "
              "still reads it, and marks the video reviewed_with_open_issues until they are corrected",
              file=sys.stderr)
    return 0


CLIP_SHEET_PREFIX = "clips"     # a coder's clip-session sheet is clips__<coder>.csv (the page names it so)
MAC_ZIP_EXTRAS = "__MACOSX"     # what the Finder's "Compress" adds to a zip beside the files: never a sheet


def _unzip_sheets(archive: Path, dest: Path) -> Path:
    """Extract a coder's zip of sheets (the page's "Download all sheets", <coder>__sheets.zip) into `dest` and return
    the folder that holds the sheets: `dest`, or the one folder inside it when the zip holds a folder (a folder Safari
    unpacked, compressed again in the Finder). A member with an absolute path or `..` refuses the whole zip."""
    try:
        with zipfile.ZipFile(archive) as zf:
            members = []
            for info in zf.infolist():
                name = info.filename.replace("\\", "/")
                parts = PurePosixPath(name).parts
                if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or ".." in parts:
                    raise ConfigError(f"{archive}: the zip holds {info.filename!r}, a path outside its folder; "
                                      "nothing was extracted or imported")
                if parts and parts[0] != MAC_ZIP_EXTRAS:
                    members.append(info)
            zf.extractall(dest, members)
    except zipfile.BadZipFile as exc:
        raise ConfigError(f"{archive} is not a zip file ({exc})") from exc
    if not any(dest.glob("*__*.csv")):
        inner = [d for d in dest.iterdir() if d.is_dir()]
        if len(inner) == 1:
            return inner[0]
    return dest


@contextlib.contextmanager
def _sheets_at(path: Path):
    """The sheets a coder handed in, as given: a folder (also the one Safari unpacks a downloaded zip into) or a
    sheet as it is, or a .zip extracted into a temporary folder for the time of the import."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".zip":
        with tempfile.TemporaryDirectory() as tmp:
            yield _unzip_sheets(path, Path(tmp))
    else:
        yield path


def cmd_import_folder(args, books: PanelBooks) -> int:
    """Import every <video_id>__<reviewer>.csv of a folder, the reviewer taken from the file name. The sheets
    are coders' whatever the manifest says of blindness (a folder of downloads is what coders hand in); only
    `--role adjudicator` makes them an adjudicator's. `--dir` may be a coder's zip of sheets (<coder>__sheets.zip),
    read as the folder it holds."""
    with _sheets_at(Path(args.dir)) as folder:
        return _import_folder(args, books, folder)


def _import_folder(args, books: PanelBooks, folder: Path) -> int:
    role = args.role or "coder"
    sheets = sorted(folder.glob("*__*.csv"))
    if not sheets:
        raise ConfigError(f"no <video_id>__<reviewer>.csv in {args.dir}")
    failed = skipped = 0
    for sheet in sheets:
        video_id, reviewer = sheet.stem.rsplit("__", 1)
        if reviewer == review_discuss.DISCUSSION:
            print(f"skipped: {sheet.name} is a discussion sheet (review_panel.py discuss), which no import reads",
                  file=sys.stderr)
            skipped += 1
            continue
        if video_id == CLIP_SHEET_PREFIX:
            print(f"skipped: {sheet.name} is a clip-session sheet, which clips-import reads", file=sys.stderr)
            skipped += 1
            continue
        if not REVIEWER_NAME.fullmatch(reviewer):
            print(f"problem: {sheet.name}: {reviewer!r} {NOT_A_NAME}; {DUPLICATE_HINT}", file=sys.stderr)
            failed += 1
            continue
        review_dir = Path(args.export_dir) if args.export_dir else Path(args.run) / video_id / REVIEW_DIR
        manifest = _optional(review_dir / MANIFEST)
        entry = ((manifest or {}).get("videos") or {}).get(video_id)
        if entry is None:
            print(f"problem: {sheet.name}: {review_dir / MANIFEST} lists no video {video_id!r}; not imported",
                  file=sys.stderr)
            failed += 1
            continue
        if sheet.name == entry.get("sheet"):
            print(f"skipped: {sheet.name} is the exported sheet of {video_id}, not a coder's", file=sys.stderr)
            skipped += 1
            continue
        staged = review_dir / sheet.name        # beside the manifest, where import looks
        if staged.resolve() != sheet.resolve():
            if staged.name == entry.get("sheet") and staged.exists() and staged.read_bytes() != sheet.read_bytes():
                print(f"problem: {sheet.name}: would replace the exported sheet {staged}; rename the file after "
                      "its coder", file=sys.stderr)
                failed += 1
                continue
            staged.write_bytes(sheet.read_bytes())
        ns = argparse.Namespace(run=args.run, sheet=str(staged), reviewer=reviewer, video_id=video_id, role=role)
        try:
            failed += cmd_import(ns, books) != 0
        except ConfigError as exc:
            print(f"problem: {sheet.name}: {exc}", file=sys.stderr)
            failed += 1
    print(f"{len(sheets) - skipped - failed} of {len(sheets) - skipped} sheets imported from {args.dir} as {role}")
    return 1 if failed else 0


# ---- the clip session ----------------------------------------------------------------------

def _clips_dir(run: Path) -> Path:
    return run / REVIEW_DIR / review_clips.CLIPS_DIR


def _session_videos(run: Path, video_ids: list[str] | None) -> list[str]:
    """The session's videos: those named, else the validation sample's."""
    if video_ids:
        return sorted(video_ids)
    sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE)
    if not sample:
        raise ConfigError(f"{run / REVIEW_DIR / VIDEO_SAMPLE} not found; pass --video-ids or draw the sample first "
                          "(export --sample-videos)")
    return sorted(sample["chosen"])


def _clip_checks(run: Path, video_ids: list[str]) -> dict[str, str]:
    """{video_id: sha256 of its primary record} of a clip session's videos, refused (ConfigError) for a record older
    than the release or a video with no prepared window to play. The session's cards are drawn from the pass the
    analysis reads (review_clips records it and its sha256 in the manifest)."""
    shas = {}
    for vdir in _video_dirs(run, video_ids):
        primary = vdir / "results" / "primary.json"
        _record(_read(primary), primary)                # an older record is refused with its release
        drawn = vdir / "results" / f"{model_pass(vdir)}.json"
        _record(_read(drawn), drawn)
        shas[vdir.name] = sha256_file(primary)
        window, why = rank_pipeline.prepared_window_reason(run, vdir.name)
        if window is None:
            raise ConfigError(f"{vdir.name} has no prepared window to play in the clip session ({why})")
    return shas


def _earlier_session(run: Path, force: bool) -> dict | None:
    """The clip session's manifest already exported (None when there is none); refused without `force`."""
    out = _clips_dir(run)
    earlier = _optional(out / review_clips.MANIFEST)
    if earlier is not None and not force:
        raise ConfigError(f"{out / review_clips.MANIFEST} exists: the coders' sheets are numbered by it; pass --force "
                          "to export the session again")
    return earlier


def _write_session(run: Path, video_ids: list[str], coders: list[str], seed: int, books: PanelBooks,
                   earlier: dict | None, shas: dict[str, str], with_codebook: bool, pages: bool = True,
                   per_video: int | None = None) -> dict:
    """Write the clip session (review_clips.export_session) and say what an earlier design leaves behind: its pair
    answers stay in the log, stamped, and leave every figure; its coders' Bradley-Terry scores are set aside."""
    out = _clips_dir(run)
    try:
        manifest = review_clips.export_session(run, video_ids, coders, seed, books, load_labels(load_release()),
                                               out, with_codebook=with_codebook, primary_sha256=shas, pages=pages,
                                               per_video=per_video)
    except (review_clips.ClipsError, PageError) as exc:
        raise ConfigError(str(exc)) from exc
    old_design = (earlier or {}).get("design_id")
    if earlier is not None and old_design != manifest["design_id"]:
        # answers to the earlier pairs stay in the log, stamped with their design, and leave every figure
        logged = [line for line in _log_lines(out / review_clips.PAIRS_LOG)
                  if line.get("design_id") != manifest["design_id"]]
        if logged:
            print(f"{len(logged)} pair answers logged for an earlier design ({old_design or 'unstamped'}) are left out "
                  f"of the agreement and the Bradley-Terry scores; {out / review_clips.PAIRS_LOG} keeps them")
        stale = sorted(out.glob("*_scores.json"))
        if stale:
            aside = out / f"earlier_{old_design or 'unstamped'}"
            aside.mkdir(exist_ok=True)
            for path in stale:
                os.replace(path, aside / path.name)
            print(f"{len(stale)} coders' Bradley-Terry scores of the earlier design moved to {aside}")
    print(f"clip session of {len(video_ids)} videos, {len(manifest['pairs'])} pairs (each video in "
          f"{manifest['per_video']}, seed {seed}, design {manifest['design_id']}), {len(manifest['rows'])} rows "
          f"-> {out / review_clips.SHEET}")
    return manifest


def _coders(names: list[str]) -> list[str]:
    coders = [reviewer_name(c) for c in names]
    if len(set(coders)) != len(coders):
        raise ConfigError("a coder is named twice")
    return coders


def cmd_clips_export(args, books: PanelBooks) -> int:
    """The clip session: the pair design, the blank session sheet, the manifest (with the model's answers, from the
    run's rank scores) and one page per coder -> <run>/review/clips/. Refused, writing nothing, without the rank
    scores of every Stage A item for every session video."""
    run = Path(args.run)
    coders = _coders(args.coders)
    video_ids = _session_videos(run, args.video_ids)
    earlier = _earlier_session(run, args.force)
    shas = _clip_checks(run, video_ids)
    manifest = _write_session(run, video_ids, coders, args.seed, books, earlier, shas, not args.no_codebook)
    out = _clips_dir(run)
    for coder in coders:
        print(f"{coder}: video order {' '.join(manifest['order'][coder])} -> {out / review_clips.PAGES / (coder + '.html')}")
    print(f"the videos: review_panel.py copy-videos --run {run} --to <folder> (the full videos; the page plays each at "
          "its prepared window)")
    return 0


TASK_PAGE = "{coder}__tasks.html"      # a coder's task page, beside (never in place of) the full-video page <coder>.html


def _sample_seed(run: Path, video_id: str) -> int | None:
    """The seed of the sample draw that chose `video_id` (<run>/review/video_sample.json), None when none did."""
    for draw in (_optional(run / REVIEW_DIR / VIDEO_SAMPLE) or {}).get("draws") or []:
        if video_id in (draw.get("chosen") or []):
            return draw.get("seed")
    return None


def _sheet_plan(run: Path, video_ids: list[str], args, books: PanelBooks) -> dict[str, dict]:
    """What session-export does with each video's blind full-video sheet, decided before anything is written:
    {video_id: {"action": "keep" | "redraw" | "export", "seed", "qa", "exchanges", "planned", "note"}}.

    An existing export is kept as it is; with --force it is drawn again with its own recorded seed and plan (the same
    rows, plus any code QA_CODES has gained since), never with the session's pair seed, and without Stage A or venture
    rows, which the session codes. A new sheet is drawn with the seed of the sample draw that chose the video (as
    `export --sample-videos ... --blind` draws it), else --sheet-seed (default 0), and the planned number of
    question-answer pairs. Refused: an export that is not blind, a sheet without its manifest entry (or the
    reverse), and a kept export carrying a card of the session (a Stage A item, a prepared code, the ask)."""
    if args.qa_per_video is not None and args.rows_per_field is not None:
        raise ConfigError("give --qa-per-video or --rows-per-field, not both")
    planned, qa = None, args.qa_per_video
    if qa is None and (args.rows_per_field is not None or not args.video_ids):
        # the sample's videos: the draw `export --sample-videos` plans, over every video of the session
        target = args.rows_per_field or ROWS_PER_FIELD
        plan = plan_draw(target, len(video_ids))
        planned, qa = {"rows_per_field": target, "n_videos": len(video_ids), **plan}, plan["qa"]
    session = set(review_clips.session_paths(books))
    out = {}
    for video_id in video_ids:
        review_dir = run / video_id / REVIEW_DIR
        entry = ((_optional(review_dir / MANIFEST) or {}).get("videos") or {}).get(video_id)
        sheet = review_dir / f"{video_id}__review.csv"
        if (entry is None) != (not sheet.exists()) and not args.force:
            raise ConfigError(f"{video_id}: {review_dir} holds a sheet or a manifest entry without the other; pass "
                              "--force to export its blind sheet again")
        if entry is None:
            seed = _sample_seed(run, video_id)
            out[video_id] = {"action": "export", "seed": seed if seed is not None else args.sheet_seed or 0,
                             "qa": qa if qa is not None else EXCHANGE_DEFAULTS["qa"],
                             "exchanges": EXCHANGE_DEFAULTS["exchanges"], "planned": planned, "note": ""}
            continue
        requested = (entry.get("sample") or {}).get("requested") or {}
        codes = [tuple(code) for code in requested.get("qa_codes") or []]
        if args.force:
            redrawn = requested.get("qa", qa if qa is not None else EXCHANGE_DEFAULTS["qa"])
            recorded = (entry.get("sample") or {}).get("planned")
            out[video_id] = {"action": "redraw", "seed": entry.get("seed", 0), "qa": redrawn,
                             "exchanges": requested.get("exchanges", EXCHANGE_DEFAULTS["exchanges"]),
                             # a recorded plan is kept only when it gave the number of pairs drawn (I2)
                             "planned": recorded if (recorded or {}).get("qa") == redrawn else None,
                             "note": "" if codes == list(QA_CODES) else
                             f"; drawn again with today's {len(QA_CODES)} question-answer codes (it had {len(codes)})"}
            continue
        if not entry.get("blind"):
            raise ConfigError(f"{video_id}: the export at {review_dir} is not blind (an adjudication sheet); a coder's "
                              "page is built from a blind one: pass --force to export it blind with its own seed (the "
                              "sheet is replaced)")
        clash = sorted(session & {row["path"] for row in entry.get("rows") or []})
        if clash:
            raise ConfigError(f"{video_id}: its full-video sheet carries {', '.join(clash)}, which the session codes; "
                              "pass --force to draw it again with its own seed, without them")
        out[video_id] = {"action": "keep", "seed": entry.get("seed"), "qa": requested.get("qa"),
                         "exchanges": requested.get("exchanges"), "planned": None,
                         "note": "" if codes == list(QA_CODES) else
                         f"; warning: it was drawn with {len(codes)} question-answer codes, not today's "
                         f"{len(QA_CODES)} ({', '.join(c for r, c in QA_CODES if (r, c) not in codes) or 'others'} "
                         "missing); --force draws it again with its own seed and today's codes"}
    return out


def _export_blind(run: Path, video_id: str, plan: dict, books: PanelBooks) -> dict:
    """A video's blind full-video sheet drawn as `plan` says (session-export's own draw: no Stage A or venture rows)."""
    args = parse_args(["export", "--run", str(run), "--blind", "--force"])
    args.preset, args.seed, args.qa_codes, args.planned = EXCHANGE, plan["seed"], QA_CODES, plan["planned"]
    args.stage_a_per_video, args.venture_per_video = 0, 0
    args.exchanges_per_video, args.qa_per_video = plan["exchanges"], plan["qa"]
    return export_video(run / video_id, run / video_id / REVIEW_DIR, books, args)


def cmd_session_export(args, books: PanelBooks) -> int:
    """One page per coder, as a task list (2026-10-21): in one step what clips-export and export --blind do for the
    same videos, then each coder's task page -> <run>/review/pages/<coder>__tasks.html (beside the full-video page
    that `page` writes as <coder>.html). Checked before anything is written: the coders, the records and their
    prepared windows, the rank scores, the pair design, an earlier session (--force) and each video's full-video
    export (`_sheet_plan`: kept, or with --force drawn again with its own seed and plan; refused when it is not blind
    or carries a card of the session). A video's prepared presentation and Q&A are one task; a comparison follows as
    soon as both its videos are done, so a coder compares two pitches after seeing their questioning (the pair
    agreement is not outcome-blind)."""
    run = Path(args.run)
    coders = _coders(args.coders)
    video_ids = _session_videos(run, args.video_ids)
    earlier = _earlier_session(run, args.force)
    shas = _clip_checks(run, video_ids)
    try:
        per_video = review_clips.session_per_video(len(video_ids))
        review_clips.model_scores(run, video_ids, list(books.stage_a_ids()))
        review_clips.pair_design(video_ids, args.seed, per_video)
    except (review_clips.ClipsError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc
    plans = _sheet_plan(run, video_ids, args, books)
    for video_id, plan in plans.items():
        where = run / video_id / REVIEW_DIR
        if plan["action"] == "keep":
            print(f"{video_id}: its blind full-video sheet is kept (seed {plan['seed']}) -> {where}{plan['note']}")
            continue
        entry = _export_blind(run, video_id, plan, books)
        print(f"{video_id}: {len(entry['rows'])} rows, blind full-video sheet "
              f"{'drawn again with its own' if plan['action'] == 'redraw' else 'drawn with'} seed {plan['seed']}"
              f"{plan['note']} -> {where / entry['sheet']}")
    manifest = _write_session(run, video_ids, coders, args.seed, books, earlier, shas, with_codebook=False,
                              pages=False, per_video=per_video)
    labels = load_labels(load_release())
    out = run / REVIEW_DIR / "pages"
    out.mkdir(parents=True, exist_ok=True)
    for coder in coders:
        try:
            model = tasks_page_model(run, manifest, coder, books, labels)
        except PageError as exc:
            raise ConfigError(str(exc)) from exc
        if not args.no_codebook:
            model["codebook"] = codebook_model(load_release(), books, labels, on_sheet(model))
        target = out / TASK_PAGE.format(coder=coder)
        target.write_text(render_page(model), encoding="utf-8")
        cards = sum(len(s["cards"]) for v in model["videos"] for s in v["sections"])
        n_pairs = len(manifest["pairs"])
        print(f"{coder}: {len(model['videos'])} tasks ({len(video_ids)} videos, {n_pairs} comparison"
              f"{'' if n_pairs == 1 else 's'}), "
              f"{cards} cards, video order {' '.join(manifest['order'][coder])} -> {target}")
    print(f"the videos: review_panel.py copy-videos --run {run} --to <folder> (the full videos)")
    return 0


def _session_rows(sheet: Path, manifest: dict) -> tuple[list[dict], list[str]]:
    """The sheet's rows matched to the exported rows by their `row` number, with the problems: a row the export
    does not have, a path that is not the one exported at that row, a row given twice."""
    exported = {row["row"]: row for row in manifest["rows"]}
    rows, problems, seen = [], [], set()
    for number, row in enumerate(read_sheet(sheet), 1):
        cell = (row.get("row") or "").strip()
        path = normalize_path(row.get("path"))
        known = exported.get(int(cell)) if cell.isdigit() else None
        if known is None:
            problems.append(f"row {cell or number}: {path or '(no path)'} is not a row of the clip session's export")
            continue
        if path != known["path"]:
            problems.append(f"row {cell}: {path or '(no path)'} is not the exported row's path {known['path']}")
            continue
        if known["row"] in seen:
            problems.append(f"row {cell}: given twice")
            continue
        seen.add(known["row"])
        rows.append({**known, "human_value": row.get("human_value"), "resolution": row.get("resolution"),
                     "note": row.get("note"), "sheet_context": row.get("context") or ""})
    return rows, problems


def _sheet_design(rows: list[dict], manifest: dict) -> str | None:
    """The problem with a session sheet's design, or None: every row that names a design must name the live
    export's, and the sheet must name one (the page and clips.csv write it into the context cells)."""
    live = manifest.get("design_id")
    named = {m.group(1) for row in rows for m in [review_clips.DESIGN.search(row["sheet_context"])] if m}
    if not named:
        return ("the sheet carries no design id: it was not written by this clip session's page or clips.csv; "
                f"the live session is design {live}")
    other = sorted(named - {live})
    if other:
        return (f"the sheet was filled for design {', '.join(other)} (an earlier clips-export); the live session is "
                f"design {live}, whose pairs show other clips on the left and the right: the coder fills the live "
                "session's page again")
    return None


def cmd_clips_import(args, books: PanelBooks) -> int:
    """A coder's session sheet: each video's cards into its review log as coder lines (as a full-video
    sheet's are; the record is never touched), the pair judgements into <run>/review/clips/pairs_log.jsonl, and the
    coder's Bradley-Terry scores per Stage A item -> <run>/review/clips/<coder>_scores.json. An unanswered pair is
    counted and left out. Any problem is printed and nothing is written. `--sheet` may be the sheet, or a coder's
    zip of sheets or a folder of them, in which clips__<reviewer>.csv is read."""
    reviewer = reviewer_name(args.reviewer)
    _not_discussion(Path(args.sheet))
    with _sheets_at(Path(args.sheet)) as found:
        if not found.is_dir():
            return _clips_import(args, books, found, reviewer)
        sheet = found / f"{CLIP_SHEET_PREFIX}__{reviewer}.csv"
        if not sheet.exists():
            raise ConfigError(f"no {sheet.name} in {args.sheet}")
        return _clips_import(args, books, sheet, reviewer)


def _clips_import(args, books: PanelBooks, sheet: Path, reviewer: str) -> int:
    run = Path(args.run)
    out = _clips_dir(run)
    manifest = _optional(out / review_clips.MANIFEST)
    if manifest is None:
        raise ConfigError(f"{out / review_clips.MANIFEST} not found; export the clip session first (clips-export)")
    if not sheet.exists():
        raise ConfigError(f"sheet not found: {sheet}")
    coders = manifest.get("coders") or []
    if reviewer not in coders:
        raise ConfigError(f"{reviewer!r} is not a coder of this clip session ({', '.join(coders)}); a third name "
                          "would leave coder against coder unmeasured")
    design = manifest.get("design_id")
    rows, problems = _session_rows(sheet, manifest)
    wrong = _sheet_design(rows, manifest) if rows else None
    if wrong:
        problems.append(wrong)
        rows = []
    log_path = _clips_dir(run) / review_clips.PAIRS_LOG
    earlier = review_clips.of_design(_log_lines(log_path), design)
    standing = review_clips.last_pairs(earlier)
    imported_at = _now()
    pair_lines, unanswered, withdrawn = [], 0, 0
    by_video: dict[str, list[dict]] = {}
    for row in rows:
        if row["field_class"] != PAIR:
            by_video.setdefault(row["video_id"], []).append(row)
            continue
        value = str(row.get("human_value") or "").strip().lower()
        resolution = str(row.get("resolution") or "").strip()
        if not value or resolution in ("unclear", "structural"):
            unanswered += 1                             # no choice: no evidence, counted and left out
            if (reviewer, row["batch_id"], row["code"]) in standing:
                # answered on an earlier import and blank now: the coder took the answer back
                withdrawn += 1
                pair_lines.append(review_clips.pair_line(row, None, row.get("note") or "", reviewer, imported_at,
                                                         design, withdrawn=True))
            continue
        if value not in PAIR_VALUES:
            problems.append(f"row {row['row']}: {row['path']}: {row['human_value']!r} is not left or right")
            continue
        pair_lines.append(review_clips.pair_line(row, value, row.get("note") or "", reviewer, imported_at, design))
    files: dict[Path, str] = {}
    messages, reviewed = [], 0
    for video_id, video_rows in sorted(by_video.items()):
        vdir = run / video_id
        # the cards were drawn from the pass the manifest names (an older manifest: the primary)
        drawn = (manifest.get("passes") or {}).get(video_id) or "primary"
        primary_path = vdir / "results" / f"{drawn}.json"
        if not primary_path.exists():
            problems.append(f"{video_id}: no {drawn} result at {primary_path}")
            continue
        answered = [r for r in video_rows if str(r.get("human_value") or "").strip()
                    or str(r.get("resolution") or "").strip()]
        skipped = sum(1 for r in video_rows if r not in answered and str(r.get("note") or "").strip())
        if skipped:
            messages.append(f"{video_id}: {skipped} rows with a note but no answer were skipped")
        if not answered:
            continue
        sha = sha256_file(primary_path)
        if sha != (manifest.get("pass_sha256") or manifest.get("primary_sha256") or {}).get(video_id):
            problems.append(f"{video_id}: {primary_path} changed since the clip session was exported; export it again")
            continue
        experiment = _optional(vdir / "experiment.json") or {}
        if "duration_s" not in experiment:
            problems.append(f"{video_id}: {vdir / 'experiment.json'} does not give duration_s, which validation needs")
            continue
        base = _record(_read(primary_path), primary_path)
        tolerances = Tolerances.of(_optional(vdir / "agreement.json"))
        by_path = {r["path"]: r for r in answered}
        corrections, found = parse_sheet(answered, reviewer, tolerances, "coder")
        problems += [f"{video_id}: {p}" for p in found]
        try:
            _, log = apply_corrections(base, corrections, books, duration_s=float(experiment["duration_s"]))
        except ReviewError as exc:
            for problem in exc.problems:
                row = next((by_path[c.path]["row"] for c in corrections if problem.startswith(f"{c.path}:")), None)
                problems.append(f"{video_id}: row {row}: {problem}" if row is not None else f"{video_id}: {problem}")
            continue
        if found:
            continue
        entry = {"rows": video_rows, "blind": True}
        fresh, _, number, written = _import_files(vdir, entry, log, corrections, by_path, sheet.name, "coder",
                                                  tolerances, books)
        files.update(written)
        reviewed += len(corrections)
        messages.append(f"{video_id}: {len(corrections)} rows reviewed (import {number}, {reviewer} as coder)")
        off_scale = counted_not_banded(corrections)
        if off_scale:
            messages.append(f"{video_id}: {EVID_PATH} answered as a count, not a band ({', '.join(EVID_BANDS)}); "
                            "it is banded for agreement, but the coder answered off the sheet's scale")
    if not reviewed and not pair_lines and not problems:
        problems.append("no row of the sheet is reviewed (human_value and resolution are empty everywhere)")
    if problems:
        for problem in problems:
            print(f"problem: {problem}", file=sys.stderr)
        print(f"{len(problems)} problem(s) in {sheet}; nothing was written", file=sys.stderr)
        return 1
    scores_path = out / f"{reviewer}_scores.json"
    if pair_lines:
        text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        if text and not text.endswith("\n"):
            text += "\n"
        files[log_path] = text + "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in pair_lines)
        order = list(books.stage_a_ids())
        fitted = review_clips.bradley_terry(earlier + pair_lines, reviewer, design)
        files[scores_path] = _json_text({code: fitted[code] for code in [*order, *sorted(set(fitted) - set(order))]
                                         if code in fitted})
    write_group(files)
    for line in messages:
        print(line)
    answered = sum(line.get("value") in PAIR_VALUES for line in pair_lines)
    print(f"pairs: {answered} pairs answered, {unanswered} unanswered (skipped), {reviewer} -> {log_path}"
          + (f"; {withdrawn} earlier answers withdrawn (blank now)" if withdrawn else ""))
    if pair_lines:
        model = _optional(run / review_clips.SCORES_SOURCE) or {}
        print(f"Bradley-Terry scores of {reviewer} -> {scores_path}")
        for code, own in review_clips.bradley_terry(earlier + pair_lines, reviewer, design).items():
            rho = review_clips.spearman(own, (model.get(code) or {}).get("pl_score") or {})
            agree = [l["value"] == l["model"] for l in pair_lines
                     if l["code"] == code and l["value"] in PAIR_VALUES and l["model"] in PAIR_VALUES]
            print(f"  {code}: {sum(agree)} of {len(agree)} choices as the model's; Spearman with its rank scores over "
                  f"{len(own)} videos " + ("-" if rho is None else f"{rho:.2f}") + " (descriptive)")
    return 0


# ---- the discussion page -------------------------------------------------------------------

def _not_discussion(sheet: Path) -> None:
    """A discussion page's sheet (<coder>__discussion.csv) is refused by every import: it is not a coder's answer."""
    if sheet.name.endswith(f"__{review_discuss.DISCUSSION}.csv"):
        raise ConfigError(f"{sheet.name} is a discussion sheet (review_panel.py discuss): no import reads it")


def cmd_discuss(args, books: PanelBooks) -> int:
    """A discussion page for the researcher and one coder (2026-10-29): each video with a question-answer turn the coder
    flagged as structurally wrong (their latest answer) or a question concept the coder put in another frame than
    Gemini's, with every question-answer pair holding one, and
    --pairs comparisons drawn (--seed) from those the coder chose against the model -> <run>/review/pages/
    <coder>__discussion.html, and what was selected and why -> <run>/review/discussion/<coder>/manifest.json. Not part
    of the validation: the page writes <coder>__discussion.csv, which no import reads; nothing imported is touched."""
    run = Path(args.run)
    coder = reviewer_name(args.coder)
    if args.pairs < 0:
        raise ConfigError("--pairs must be 0 or more")
    session = _optional(_clips_dir(run) / review_clips.MANIFEST)
    if session is None:
        raise ConfigError(f"{_clips_dir(run) / review_clips.MANIFEST} not found; the discussion page is built from the "
                          "session (session-export) and the coder's imported answers")
    page = run / REVIEW_DIR / "pages" / review_discuss.PAGE.format(coder=coder)
    manifest_path = run / REVIEW_DIR / review_discuss.DIR / coder / "manifest.json"
    if (page.exists() or manifest_path.exists()) and not args.force:
        raise ConfigError(f"{page} or {manifest_path} exists (its answers are kept in the browser, under the run and "
                          "the coder, not in the page); pass --force to build it again")
    logs = {v: _log_lines(run / v / REVIEW_DIR / LOG) for v in session.get("videos") or []}
    pair_lines = _log_lines(_clips_dir(run) / review_clips.PAIRS_LOG)
    try:
        scores = review_clips.model_scores(run, list(session.get("videos") or []), list(books.stage_a_ids()))
    except review_clips.ClipsError:
        scores = {}             # the gap is in the pairs log; the two scores are shown when the ranking is at hand
    labels = load_labels(load_release())
    try:
        model, manifest = review_discuss.discussion_page_model(run, session, coder, books, labels, logs, pair_lines,
                                                               args.pairs, args.seed, scores)
    except PageError as exc:
        raise ConfigError(str(exc)) from exc
    if not model["videos"]:
        raise ConfigError(f"nothing to discuss for {coder}: no question-answer turn flagged as structurally wrong, no "
                          "question concept in another frame than Gemini's and no comparison chosen against the model "
                          "among the imported answers")
    model["codebook"] = codebook_model(load_release(), books, labels, on_sheet(model))
    page.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_group({page: render_page(model),
                 manifest_path: _json_text({**manifest, "created_at": _now(), "page": str(page.relative_to(run))})})
    videos = [t for t in model["videos"] if t["task"] == "video"]
    pairs = [p for v in manifest["videos"] for p in v["pairs"]]
    of = lambda reason: sum(reason in p["selected_for"] for p in pairs)     # noqa: E731
    framed = sum(len(v["frame_paths"]) for v in manifest["videos"])
    compares = len(model["videos"]) - len(videos)
    print(f"{coder}: {len(videos)} video tasks ({len(pairs)} question-answer pairs: "
          f"{of(review_discuss.STRUCTURAL_FLAG)} with a structural flag, {of(review_discuss.FRAME_DISAGREEMENT)} with "
          f"a frame disagreement ({framed} concept cards)), {compares} comparison tasks "
          f"(of {manifest['comparisons']['candidates']} chosen against the model; seed {args.seed}) -> {page}")
    print(f"what was selected and why -> {manifest_path}; the page's sheet, {manifest['sheet']}, is never imported")
    return 0


# ---- training and calibration --------------------------------------------------------------

TRAINING_DIR = "training"               # under <run>/review/
MODEL_ANSWERS = "model_answers.html"
TRAINING_SHEET = "training_videos.csv"   # the training videos as a spreadsheet, like video_sample.csv     # beside pages/, not in it: it is the researcher's, not a coder's
CALIBRATION = "calibration"
EARLIER_CLIPS = "earlier_clips_"         # <training>/earlier_clips_<design>/: a clip session a retraining set aside
TRAINING_DRAW = {"exchanges": EXCHANGE_DEFAULTS["exchanges"], "qa": 6}   # per video, as the validation sample's planned draw


def _training_videos(run: Path, args) -> list[str]:
    if args.video_ids:
        return list(args.video_ids)
    if not args.from_selection:
        raise ConfigError("give --video-ids, or --from <selection file> to draw from its --key list")
    selection = _read(run / args.from_selection)
    if not isinstance(selection, dict) or args.key not in selection:
        keys = ", ".join(sorted(map(str, selection))) if isinstance(selection, dict) and selection else "none"
        raise ConfigError(f"{run / args.from_selection} has no key {args.key!r} (--key); its keys: {keys}")
    pool = sorted(selection[args.key] or [])
    if not pool:
        raise ConfigError(f"{run / args.from_selection}: {args.key!r} lists no video")
    return sorted(random.Random(f"{args.seed}:training").sample(pool, min(args.videos, len(pool))))


def cmd_training(args, books: PanelBooks) -> int:
    """Blind training sheets for a few videos outside the validation sample, one training page per coder with the
    codebook on it, and a model-answers page for the joint walkthrough; all under <run>/review/training/.

    --clips adds the training clip session under <run>/review/training/clips/, as clips-export writes the real one
    (the same seven cards per video at its prepared window, the pairs from the run's rank scores, a design id on
    every row): each of the
    two training videos in TRAINING_PER_VIDEO comparisons, so their one pair three times with its sides seeded. The
    full-video training sheets then leave the Stage A items to the clip session, as the validation sheets do, and
    the model-answers page opens with the clip session's cards and the pairs. Refused, writing nothing, unless exactly two
    videos are given, and without the rank scores, a prepared window or a v3-dev9 record. Training again without
    --clips (with --force) sets an earlier clip session aside as earlier_clips_<design>/ (_2, _3, ... when that name is
    taken; nothing is deleted), so calibrate no longer
    imports the coders' old clip sheets over the new full-video answers."""
    run = Path(args.run)
    coders = [reviewer_name(c) for c in args.coders]
    if len(set(coders)) != len(coders):
        raise ConfigError("--coders names a coder twice")
    out = Path(args.out) if args.out else run / REVIEW_DIR / TRAINING_DIR
    video_ids = _training_videos(run, args)
    sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE) or {}
    overlap = sorted(set(video_ids) & set(sample.get("chosen") or []))
    if overlap:
        raise ConfigError(f"{', '.join(overlap)} {'is' if len(overlap) == 1 else 'are'} in the validation sample; "
                          "train on other videos (the development videos are meant for it)")
    clips_out = out / review_clips.CLIPS_DIR
    with_clips = getattr(args, "clips", False)
    earlier_session = _optional(clips_out / review_clips.MANIFEST)
    if earlier_session is not None and not args.force:
        raise ConfigError(f"{clips_out / review_clips.MANIFEST} exists: the coders' clip sheets are numbered by it; "
                          "pass --force to export the training again"
                          + ("" if with_clips else " (the earlier clip session is then set aside, not deleted)"))
    if with_clips:
        # everything the clip session needs is checked before anything is written: two videos, the records and
        # their prepared windows, the rank scores and the pair design itself
        if len(video_ids) != 2:
            raise ConfigError(f"the training clip session takes exactly two development videos (their one pair is "
                              f"shown {review_clips.TRAINING_PER_VIDEO} times); got {len(video_ids)}: "
                              f"{', '.join(sorted(video_ids))}. Give two with --video-ids, or --from with --videos 2")
        shas = _clip_checks(run, video_ids)
        try:
            scores = review_clips.model_scores(run, sorted(video_ids), list(books.stage_a_ids()))
            review_clips.pair_design(sorted(video_ids), args.seed, review_clips.TRAINING_PER_VIDEO,
                                     allow_repeats=True)
        except (review_clips.ClipsError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
    elif earlier_session is not None:
        # a retraining without the clip session: the earlier one would otherwise be imported by calibrate with the
        # coders' old clip sheets, over the new full-video answers; it is kept beside the training, not deleted
        # never deleted: a design set aside before (the same videos and seed give the same design) keeps its folder,
        # and this one takes the next free name, earlier_clips_<design>_2, _3, ...
        stem = f"{EARLIER_CLIPS}{earlier_session.get('design_id') or 'unstamped'}"
        aside, n = out / stem, 1
        while aside.exists():
            n += 1
            aside = out / f"{stem}_{n}"
        os.replace(clips_out, aside)
        print(f"the earlier training clip session (design {earlier_session.get('design_id')}) was set aside -> {aside}; "
              "calibrate no longer reads clip-session sheets for this training")
    export = parse_args(["export", "--run", str(run), "--video-ids", *video_ids, "--blind",
                         "--stage-a-per-video", "0" if with_clips else "all", "--venture-per-video", "0",
                         "--exchanges-per-video", str(args.exchanges_per_video), "--qa-per-video", str(args.qa_per_video),
                         "--seed", str(args.seed), "--out", str(out), *(["--force"] if args.force else [])])
    cmd_export(export, books)
    release = load_release()
    labels = load_labels(release)
    pages = out / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    session = None
    if with_clips:
        # the clip session's sheet and manifest, no page of its own: each coder's page is the task page, which writes
        # the clip session's sheet and the full-video sheets alike
        try:
            session = review_clips.export_session(run, video_ids, coders, args.seed, books, labels, clips_out,
                                                  primary_sha256=shas, training=True, pages=False)
        except (review_clips.ClipsError, PageError) as exc:
            raise ConfigError(str(exc)) from exc
        print(f"training clip session of {len(session['videos'])} videos, {len(session['pairs'])} pairs (each video in "
              f"{session['per_video']}, seed {args.seed}, design {session['design_id']}), {len(session['rows'])} rows "
              f"-> {clips_out / review_clips.SHEET}")
    for coder in coders:
        if session is not None:
            try:
                model = tasks_page_model(run, session, coder, books, labels, {v: out for v in video_ids},
                                         training=True)
            except PageError as exc:
                raise ConfigError(str(exc)) from exc
            model["codebook"] = codebook_model(release, books, labels, on_sheet(model))
            target = pages / f"{coder}.html"
            target.write_text(render_page(model), encoding="utf-8")
            cards = sum(len(s["cards"]) for v in model["videos"] for s in v["sections"])
            print(f"{coder}: {len(model['videos'])} training tasks, video order {' '.join(session['order'][coder])}, "
                  f"{cards} cards -> {target}")
            continue
        order = list(video_ids)
        random.Random(f"{args.seed}:{coder}").shuffle(order)
        try:
            model = page_model(run, order, coder, books, labels, {v: out for v in order}, order_seed=args.seed,
                               mode=TRAINING)
        except PageError as exc:
            raise ConfigError(str(exc)) from exc
        model["codebook"] = codebook_model(release, books, labels, on_sheet(model))
        target = pages / f"{coder}.html"
        target.write_text(render_page(model), encoding="utf-8")
        cards = sum(len(s["cards"]) for v in model["videos"] for s in v["sections"])
        print(f"{coder}: {len(order)} training videos, {cards} cards -> {target}")
    if with_clips:
        print(f"the videos: review_panel.py copy-videos --run {run} --training --to <folder> (the full videos; the "
              "page plays each at its prepared window)")
    # the same cards with the model's answers and what it recorded in support, for the walkthrough with the coders;
    # the clip session's cards and pairs first, as the coders meet them
    model = page_model(run, sorted(video_ids), "model-answers", books, labels, {v: out for v in video_ids},
                       mode=WALKTHROUGH)
    if session is not None:
        model["videos"] = clips_walkthrough(run, session, books, labels, scores) + model["videos"]
    model["codebook"] = codebook_model(release, books, labels, on_sheet(model))
    (out / MODEL_ANSWERS).write_text(render_page(model), encoding="utf-8")
    stale = out / "model_values.csv"            # what an earlier version wrote in its place
    if stale.exists():
        stale.unlink()
    print(f"model answers for the walkthrough -> {out / MODEL_ANSWERS} (do not give it to the coders before they "
          "have coded: it shows the model's answers)")
    sheet = write_video_sheet(out / TRAINING_SHEET, run, [("training", args.seed, sorted(video_ids))])
    print(f"training videos: {', '.join(video_ids)} -> {sheet}")
    return 0


def _copy_videos_for_calibration(run: Path, video_ids: list[str], into: Path) -> None:
    """What an import reads of each video (experiment, agreement settings, the model's passes), nothing else."""
    for video_id in video_ids:
        src, dst = run / video_id, into / video_id
        (dst / "results").mkdir(parents=True)
        for name in ("experiment.json", "agreement.json"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        for path in (src / "results").glob("*.json"):
            if not path.name.startswith("reviewed"):
                shutil.copy2(path, dst / "results" / path.name)


CALIBRATION_BAND = "substantial"      # the Landis & Koch band whose lower bound is calibrate's default bar


def calibration_table(pooled: dict, reporting: Reporting | None) -> list[dict] | None:
    """The coder-against-coder rows of the paper table (review_report.paper_rows of pooled["reporting"]
    ["human_human"]), the comparisons aside (the pair judgements have their own section); None when the pooled
    figures carry no reporting (a reporting file that was refused)."""
    if reporting is None or not pooled.get("reporting"):
        return None
    return [r for r in paper_rows(pooled, reporting, "human_human") if not r["missing"] and r["kind"] != "comparison"]


def _reaches(alpha, bar: float) -> bool | None:
    # at the two decimals the tables print and the band reads
    return None if alpha is None else round(alpha, 2) + 0.0 >= round(bar, 2)


def calibration_rows(pooled: dict, bar: float, reporting: Reporting | None = None) -> list[dict]:
    """With the reporting (validation_reporting.yaml), one row per reported code of the coders' agreement with each
    other (scope "code"), as the paper table's coder-against-coder table gives it: the kind, the units, the leading
    raw agreement (and within one for an ordinal code), alpha with its band, the flags, and alpha against `bar`
    ("reached"; None, and no bar, for a kind without alpha: the first decision, the ask). Without it, one row per
    field class, then per variable, of the pooled summary's coder-against-coder alpha against `bar`. Then, when the
    clip session was coded, one row per Stage A item of the pair judgements (scope "pair"): the coders' pooled
    agreement with the model ("agree", k of n) with its Clopper-Pearson interval and two-sided p against 0.5, each
    coder's share ("by_coder") and coder against coder beside it. A pair row has no bar ("reached" None): a
    comparison is read against 0.5. The binomial pools the coders: each coder's answer to the same pair is one of the
    n judgements, so with two coders the interval and p are optimistic (the per-coder shares sit beside them).

    A field reaches the bar when its alpha, at the two decimals the table prints and the band reads, does."""
    from pipeline.reliability.agreement import binomial_interval, binomial_p_vs_half
    rows = []
    table = calibration_table(pooled, reporting)
    for r in table or []:
        alpha = (r["alpha"] or {}).get("value")
        lead, near = (HEADLINE[r["kind"]] + (None,))[:2]
        rows.append({"scope": "code", "field": r["field"], "kind": r["kind"], "videos": r["videos"],
                     "units": r["units"], "n": r["n"], "agree": (r["figures"].get(lead) or {}).get("value"),
                     "within_one": (r["figures"].get(near) or {}).get("value") if r["kind"] == "ordinal" else None,
                     "alpha": alpha, "band": r["band"], "flags": r["flags"],
                     "bar": bar if r["kind"] in ALPHA_KINDS else None,
                     "reached": _reaches(alpha, bar) if r["kind"] in ALPHA_KINDS else None})
    hh = (pooled.get("human_human") or {}) if table is None else {}         # the older figures, without the reporting
    for scope, table_ in (("class", {k: v for k, v in hh.items() if k != "by_variable"}),
                          ("variable", hh.get("by_variable") or {})):
        for key, f in table_.items():
            alpha = f.get("alpha")
            rows.append({"scope": scope, "field": key, "videos": f.get("videos"), "n": f.get("n"),
                         "agree": f.get("agree"), "alpha": alpha, "alpha_units": f.get("alpha_units"),
                         "level": f.get("alpha_level"), "bar": bar, "reached": _reaches(alpha, bar)})
    for code, f in (pooled.get("pairs") or {}).items():
        share, n = f.get("coder_model"), f.get("n") or 0
        k = round(share * n) if share is not None and n else None
        interval = list(binomial_interval(k, n)) if k is not None else None
        note = "" if f.get("coder_coder") is not None else "coder against coder unmeasured: the pairs of one coder only"
        rows.append({"scope": PAIR, "field": f"{PAIR}:{code}", "videos": f.get("videos"), "n": f.get("n"),
                     "pairs": f.get("n_pairs"), "agree": share, "k": k, "interval": interval,
                     "p_vs_half": binomial_p_vs_half(k, n) if k is not None else None,
                     "by_coder": f.get("coder_model_by_coder"), "coder_coder": f.get("coder_coder"), "alpha": None,
                     "alpha_units": None, "level": None, "bar": None, "note": note, "reached": None})
    return rows


def calibration_markdown(rows: list[dict], coders: list[str], videos: int, bar: float,
                         table: list[dict] | None = None) -> str:
    """calibration.md: with `table` (calibration_table's rows), the paper table's coder-against-coder table with a
    last column comparing alpha with `bar`; without it, the pooled summary's alpha per field class and field; then the
    pair judgements."""
    def num(x):
        return "–" if x is None else f"{x:.2f}"

    def verdict(r):
        return "not estimable" if r["reached"] is None else ("yes" if r["reached"] else "not yet")
    plural = "s" if videos != 1 else ""
    lines = [f"# Calibration of {' and '.join(coders) or 'the coders'}", ""]
    if table is not None:
        reached = {r["field"]: r for r in rows if r["scope"] == "code"}

        def bar_cell(r):
            got = reached.get(r["field"])
            return "—" if got is None or got["bar"] is None else verdict(got)
        lines += [f"The coders' agreement with each other on {videos} training video{plural}, per code as the paper "
                  "table's coder-against-coder table gives it (validation_reporting.yaml): raw agreement with "
                  "Krippendorff's α in parentheses, α's Landis & Koch band and the flags. The last column compares α "
                  f"with a bar of {bar:.2f}, a guide for the training: \"not estimable\" means too few units or no "
                  "variation to compute α (look at the agreement and talk the units through); a code without α (the "
                  "first decision, the ask) shows —. Training sheets never enter the validation.", "",
                  "## Coder against coder", ""]
        lines += (table_lines(table, (f"α ≥ {bar:.2f}", bar_cell)) if table
                  else ["No training video was judged by two coders."])
        lines.append("")
    else:
        head = "| Field | Rows | α units | Agreement | Coder–coder α | Reached the bar |"
        lines += [f"The coders' agreement with each other on {videos} training video{plural}, against a bar of "
                  f"α ≥ {bar:.2f}. A field reaches the bar when its coder–coder alpha does; \"not estimable\" means "
                  "there were too few rows or no variation to compute alpha (look at the agreement share and talk the "
                  "rows through). Training sheets never enter the validation.", ""]
        for scope, title in (("class", "Field classes"), ("variable", "Single fields")):
            part = [r for r in rows if r["scope"] == scope]
            if not part:
                continue
            lines += [f"## {title}", "", head, "|---|---|---|---|---|---|"]
            lines += [f"| {r['field'].replace('|', chr(92) + '|')} | {r['n'] or 0} | "
                      f"{r['alpha_units'] if r['alpha_units'] is not None else '–'} "
                      f"| {num(r['agree'])} | {num(r['alpha'])} | {verdict(r)} |" for r in part]
            lines.append("")
    pairs = [r for r in rows if r["scope"] == PAIR]
    if pairs:
        def against(r):
            if r.get("interval") is None:
                return "–"
            p = r["p_vs_half"]
            return (f"{num(r['interval'][0])}–{num(r['interval'][1])}; p "
                    + ("< 0.001" if p < 0.001 else f"= {p:.3f}"))
        lines += ["## Pair judgements", "",
                  "Which of two prepared presentations shows more, per Stage A item: the coders' choices against the "
                  "model's rank scores (pooled over the coders, each coder's beside it) and against each other. "
                  "Agreement with the model is read against 0.5, a coin flip, with its exact (Clopper–Pearson) 95% "
                  "interval and two-sided p: the paper table's interval resamples videos and needs "
                  f"{MIN_CLUSTER_VIDEOS} videos, more than a training has. The binomial pools the coders: each "
                  "coder's answer to a pair counts as one judgement, and pairs share clips, so the interval and p are "
                  "optimistic. With one coder's pairs, coder against coder is unmeasured. On two training videos the "
                  "pairs are one pair shown three times: a coder who changes their mind shows here.", "",
                  "| Item | Pairs | Answers | Coders vs model | 95% interval; p against 0.5 | Each coder vs model "
                  "| Coder vs coder |",
                  "|---|---|---|---|---|---|---|"]
        for r in pairs:
            each = ", ".join(f"{c} {num(v)}" for c, v in sorted((r.get("by_coder") or {}).items())) or "–"
            lines.append(f"| {r['field'].partition(':')[2]} | {r.get('pairs') or 0} | {r['n'] or 0} | {num(r['agree'])} "
                         f"| {against(r)} | {each} | {num(r['coder_coder'])} |")
        lines.append("")
    return "\n".join(lines)


def training_imports(run: Path, sheets: Path, books: PanelBooks, training_dir: Path | None = None
                     ) -> tuple[dict[str, list[dict]], int, dict[str, dict | None], dict]:
    """The coders' training sheets imported as the validation imports them, per training video: ({video_id: its
    review-log lines}, the imports' exit status (1 when a sheet failed), {video_id: its agreement settings},
    the clip session's pair agreement). The full-video sheets (<video_id>__<coder>.csv) go through import-folder;
    the clip-session sheets (clips__<coder>.csv, when `training --clips` exported a session) through clips-import,
    whose per-clip cards land in the same videos' logs, so a path is pooled whichever sheet it was coded on, and
    whose pair judgements give review_clips.pair_agreement ({} without a clip sheet). Everything is imported into
    a throwaway copy of the run, so the run's own review logs and results are never touched. `calibrate` and the
    human-agreement notebook both read these."""
    tdir = Path(training_dir) if training_dir else run / REVIEW_DIR / TRAINING_DIR
    manifest = _optional(tdir / MANIFEST)
    if not manifest or not manifest.get("videos"):
        raise ConfigError(f"no training export at {tdir / MANIFEST}; run `review_panel.py training` first")
    session = _optional(tdir / review_clips.CLIPS_DIR / review_clips.MANIFEST)
    clip_sheets = sorted(Path(sheets).glob(f"{CLIP_SHEET_PREFIX}__*.csv"))
    video_ids = sorted(manifest["videos"])
    pairs: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        scratch_run, scratch_export = Path(tmp) / run.name, Path(tmp) / "export"
        shutil.copytree(tdir, scratch_export, ignore=shutil.ignore_patterns("pages", f"{CALIBRATION}*", MODEL_ANSWERS,
                                                                            TRAINING_SHEET, "training_results",
                                                                            review_clips.CLIPS_DIR,
                                                                            f"{EARLIER_CLIPS}*"))
        _copy_videos_for_calibration(run, video_ids, scratch_run)
        status = 0
        if not clip_sheets or any(p for p in Path(sheets).glob("*__*.csv") if p not in clip_sheets):
            status = cmd_import_folder(argparse.Namespace(run=str(scratch_run), dir=str(sheets),
                                                          export_dir=str(scratch_export), role="coder"), books)
        if clip_sheets and session is None:
            print(f"problem: {', '.join(p.name for p in clip_sheets)}: no training clip session at "
                  f"{tdir / review_clips.CLIPS_DIR / review_clips.MANIFEST}; run `review_panel.py training --clips`",
                  file=sys.stderr)
            status = 1
        elif clip_sheets and sorted(session.get("videos") or []) != sorted(manifest["videos"]):
            print(f"problem: {', '.join(p.name for p in clip_sheets)}: the training clip session is of "
                  f"{', '.join(session.get('videos') or [])}, the training of {', '.join(manifest['videos'])}; "
                  "not imported (export the training again with --clips --force)", file=sys.stderr)
            status = 1
        elif clip_sheets:
            scratch_clips = _clips_dir(scratch_run)
            scratch_clips.mkdir(parents=True)
            shutil.copy2(tdir / review_clips.CLIPS_DIR / review_clips.MANIFEST, scratch_clips / review_clips.MANIFEST)
            if (run / review_clips.SCORES_SOURCE).exists():          # for the import's Spearman line only
                (scratch_run / review_clips.SCORES_SOURCE).parent.mkdir(parents=True)
                shutil.copy2(run / review_clips.SCORES_SOURCE, scratch_run / review_clips.SCORES_SOURCE)
            for sheet in clip_sheets:
                reviewer = sheet.stem.split("__", 1)[1]
                ns = argparse.Namespace(run=str(scratch_run), sheet=str(sheet), reviewer=reviewer)
                try:
                    status |= cmd_clips_import(ns, books) != 0
                except ConfigError as exc:
                    print(f"problem: {sheet.name}: {exc}", file=sys.stderr)
                    status = 1
            pairs = session_agreement(scratch_run, books)
        logs = {v: _log_lines(scratch_run / v / REVIEW_DIR / LOG) for v in video_ids}
        agreements = {v: _optional(scratch_run / v / "agreement.json") for v in video_ids}
    return logs, int(status), agreements, pairs


def pool_training(logs: dict[str, list[dict]], agreements: dict[str, dict | None]) -> tuple[dict, dict, dict]:
    """(videos, tolerances, by_reviewer) of training logs, in the shapes `pooled_summary` takes."""
    videos, tolerances, by_reviewer = {}, {}, {}
    for video_id, log in sorted(logs.items()):
        coders = _lines_by_reviewer(log)
        if len(coders) > 2:
            print(_too_many_coders(video_id), file=sys.stderr)
        if coders:
            videos[video_id] = validity_lines(log)
            tolerances[video_id] = Tolerances.of(agreements.get(video_id))
        for name, last in coders.items():
            by_reviewer.setdefault(name, {})[video_id] = list(last.values())
    return videos, tolerances, by_reviewer


def cmd_calibrate(args, books: PanelBooks) -> int:
    """The trainees' agreement with each other, from their training sheets, without touching the run's logs."""
    run = Path(args.run)
    tdir = Path(args.training_dir) if args.training_dir else run / REVIEW_DIR / TRAINING_DIR
    try:
        reporting = load_reporting(books)
    except ConfigError as exc:
        # the per-code figures need the reporting; without it the pooled summary's alpha per field class is given
        reporting = None
        print(f"note: the agreement per code is left out of the calibration: {exc}", file=sys.stderr)
    bar = args.bar if args.bar is not None else (reporting.band_floor(CALIBRATION_BAND) if reporting is not None
                                                 else REPORTING_BAND_FLOORS["landis_koch"][CALIBRATION_BAND])
    logs, status, agreements, pairs = training_imports(run, Path(args.dir), books, tdir)
    video_ids = sorted(logs)
    videos, tolerances, by_reviewer = pool_training(logs, agreements)
    if not by_reviewer:
        raise ConfigError(f"no training sheet of {', '.join(video_ids)} was imported from {args.dir}")
    pooled = pooled_summary(videos, tolerances, by_reviewer=by_reviewer,
                            pairs=review_clips.pooled_pairs(pairs) if pairs else None, reporting=reporting)
    rows = calibration_rows(pooled, bar, reporting)
    coders = sorted(by_reviewer)
    text = calibration_markdown(rows, coders, len(videos), bar, calibration_table(pooled, reporting))
    (tdir / f"{CALIBRATION}.md").write_text(text, encoding="utf-8")
    (tdir / f"{CALIBRATION}.json").write_text(json.dumps({"coders": coders, "videos": sorted(videos), "bar": bar,
                                                           "rows": rows}, indent=2, ensure_ascii=False) + "\n",
                                               encoding="utf-8")
    print(text, end="")
    if len(coders) != 2:
        print(f"note: coder-against-coder needs exactly two coders per video; found {', '.join(coders)}",
              file=sys.stderr)
    missed = [r["field"] for r in rows if r["scope"] in ("code", "class") and r["reached"] is False]
    print(f"calibration -> {tdir / (CALIBRATION + '.md')}"
          + (f"; below the bar: {', '.join(missed)}" if missed else ""))
    return 1 if status else 0


# ---- status -------------------------------------------------------------------------------

def cmd_page(args, books: PanelBooks) -> int:
    run = Path(args.run)
    args.coder = reviewer_name(args.coder)
    if args.sample:
        sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE)
        if not sample:
            raise ConfigError(f"{run / REVIEW_DIR / VIDEO_SAMPLE} not found; export with --sample-videos first")
        video_ids = list(sample["chosen"])
    elif args.video_ids:
        video_ids = list(args.video_ids)
    else:
        raise ConfigError("give --video-ids or --sample")
    # each coder meets the videos in an order of their own, so that fatigue and practice do not fall on the same
    # videos for both; seeded by --seed and the coder's name, and recorded in the page
    random.Random(f"{args.seed}:{args.coder}").shuffle(video_ids)
    labels = load_labels(load_release())
    out_dirs = {v: Path(args.export_dir) for v in video_ids} if args.export_dir else None
    try:
        model = page_model(run, video_ids, args.coder, books, labels, out_dirs, order_seed=args.seed)
    except PageError as exc:
        raise ConfigError(str(exc)) from exc
    if not args.no_codebook:
        model["codebook"] = codebook_model(load_release(), books, labels, on_sheet(model))
    out = Path(args.out) if args.out else run / REVIEW_DIR / "pages"
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{args.coder}.html"
    target.write_text(render_page(model), encoding="utf-8")
    cards = sum(len(s["cards"]) for v in model["videos"] for s in v["sections"])
    print(f"{args.coder}: {len(model['videos'])} videos, {cards} cards -> {target}")
    print(f"video order (seed {args.seed}): {', '.join(video_ids)}")
    return 0


def _headline(agreement: dict | None) -> str:
    if agreement is None:
        return "no agreement yet"
    parts = []
    for selection in (SAMPLED, FLAGGED):
        figure = agreement["by_selection"][selection]["all"]["human_vs_primary"]
        if figure["n"]:
            agree = round(figure["agree"] * figure["n"])
            parts.append(f"{agree}/{figure['n']} {selection} rows ({figure['agree']:.2f})")
        else:
            parts.append(f"no {selection} row with a value")
    return "human = primary on " + ", ".join(parts)


def cmd_status(args, books: PanelBooks) -> int:
    run = Path(args.run)
    dirs = _video_dirs(run, args.video_ids)
    sample = _optional(run / REVIEW_DIR / VIDEO_SAMPLE)
    chosen = set((sample or {}).get("chosen") or [])
    if sample is not None:
        done = sorted(v for v in chosen if _log_lines(run / v / REVIEW_DIR / LOG))
        draws = sample.get("draws") or []
        print(f"video sample ({len(draws)} draw{'s' if len(draws) != 1 else ''}, seed "
              f"{', '.join(str(d['seed']) for d in draws)}): {len(chosen)} videos, {len(done)} reviewed: "
              f"{', '.join(sorted(chosen))}")
    for vdir in dirs:
        entry = _manifest_entry(Path(args.out) if args.out else vdir / REVIEW_DIR, vdir.name)
        sheet = f"sheet exported ({len(entry['rows'])} rows)" if entry else "no sheet exported"
        reviewed = _optional(vdir / "results" / "reviewed.json") or {}
        lines = _log_lines(vdir / REVIEW_DIR / LOG)
        rows = f"{len(_reviewed_lines(lines))} rows reviewed"
        if lines:
            rows += " (" + ", ".join(f"{r}: {len(paths)}" for r, paths in _lines_by_reviewer(lines).items()) + ")"
        print(f"{vdir.name}{' [in sample]' if vdir.name in chosen else ''}: {sheet}; {rows}; {reviewed.get('corrections', 0)} corrections applied; "
              f"{_headline(_optional(vdir / REVIEW_DIR / AGREEMENT))}")
    return 0


# ---- summary ------------------------------------------------------------------------------

def _share(figure: dict) -> str:
    return "-" if not figure["n"] else f"{figure['agree']:.2f} ({round(figure['agree'] * figure['n'])}/{figure['n']})"


def _summary_table(table: dict) -> list[str]:
    columns = ["field class", "videos", "rows", "human = primary", "95% interval", "human = replicate",
               "primary = replicate", "within one (human, primary)", "interval method"]
    lines = [f"| {' | '.join(columns)} |", "|" + "---|" * len(columns)]
    for cls, f in table.items():
        ci = f["human_vs_primary_ci95"]
        near = _share(f["within_one"]["human_vs_primary"]) if "within_one" in f else "-"
        lines.append(f"| {cls} | {f['videos']} | {f['n']} | {_share(f['human_vs_primary'])} | "
                     + ("-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}")
                     + f" | {_share(f['human_vs_replicate'])} | {_share(f['primary_vs_replicate'])} | {near} | "
                     + f"{f['ci_method']} |")
    return lines


def _plural_videos(n: int) -> str:
    return f"{n} video{'' if n == 1 else 's'}"


def summary_text(run: Path, pooled: dict, reporting: Reporting | None = None, table_pooled: dict | None = None,
                 adjudicator_only: int = 0) -> str:
    """summary.md. The per-kind table and the strata come from `table_pooled`, the coder-coded videos the paper table
    is built from (`pooled` itself when no video was reviewed by an adjudicator only); every other section from
    `pooled`."""
    table_pooled = pooled if table_pooled is None and not adjudicator_only else table_pooled
    L = [f"# Human review summary: {run.name}", "",
         f"{pooled['n_videos']} videos with an imported sheet, {pooled['n_rows']} reviewed rows; "
         f"{pooled['n_structural']} structural notes, {pooled['n_outcome_overrides']} outcome overrides, "
         f"{pooled.get('n_cut_overrides', 0)} first-decision overrides, "
         f"{pooled['n_without_a_value']} rows without a definite value.", "",
         "## Sampled rows (the estimate)", "",
         "Rows drawn at random, pooled over videos: the share on which the human gives the primary pass's value, "
         "with a 95% interval, beside the pass-to-pass figure on the rows that have a replicate value. The first "
         "decision agrees when both times map to the same turn of the sheet’s reference pass; unmappable times "
         "are excluded and counted separately. Other clock "
         "values count as the same within tolerance (a decision time within the agreement settings' tolerance, each "
         "edge of the prepared window within two seconds); ordinal fields (rated scales) are compared "
         "exactly, and the `within one` column gives the share within one scale point. Rows of one video are not "
         f"independent, so with at least {MIN_CLUSTER_VIDEOS} videos carrying rows of a class the interval resamples "
         f"whole videos (cluster bootstrap, seeded); with fewer it is the Wilson interval, labelled `{WILSON_METHOD}`.",
         ""]
    L += _summary_table(pooled["sampled"]) if pooled["sampled"] else ["No sampled row has been reviewed."]
    if reporting is not None:
        left_out = (f" The {_plural_videos(adjudicator_only)} only an adjudicator reviewed "
                    f"{'is' if adjudicator_only == 1 else 'are'} left out, as in the paper table."
                    if adjudicator_only else "")
        L += ["", "## Agreement per code, coder against model (the validation table)", "",
              "Each reported code with the statistics of its kind (validation_reporting.yaml), from the coder-coded "
              "videos: raw agreement with its 95% interval resampling videos, Krippendorff's alpha in parentheses "
              "with its interval, alpha's Landis & Koch band, and flags in words; the column without unsure units is "
              "a point. `--paper-table` adds coder against coder, pass against pass, the appendix and the note on the "
              f"statistics.{left_out}", ""]
        L += (table_lines(paper_rows(table_pooled, reporting)) if table_pooled is not None
              else ["No video has a coder's lines."])
    unsure = (pooled.get("without_unsure") or {}).get("n_unsure")
    if unsure:
        L += ["", "## Without the rows a coder was not sure of", "",
              f"{unsure} of {pooled['n_rows']} reviewed rows carry a coder's best guess ticked not sure ([unsure] in the "
              "note); the sampled figures without them:", ""]
        L += _summary_table(pooled["without_unsure"]["sampled"]) if pooled["without_unsure"]["sampled"] else [
            "No sampled row is left."]
    L += strata_markdown(table_pooled, reporting) if table_pooled is not None else []
    if pooled["flagged"]:
        L += ["", "## Flagged and added rows (not an estimate)", "",
              "Rows chosen because something looked wrong, or added by the coder; they understate agreement.", ""]
        L += _summary_table(pooled["flagged"])
    if pooled.get("by_reviewer"):
        L += ["", "## Coders", "",
              "Each coder's own agreement with the primary pass, pooled over the videos that coder judged."]
        for name, own in pooled["by_reviewer"].items():
            L += ["", f"### {name}", ""]
            L += _summary_table(own["sampled"]) if own["sampled"] else ["No sampled row has been reviewed."]
        L += ["", "## Coder against coder", "",
              "Rows both coders gave a definite value, on videos judged by exactly two coders; `unclear` and "
              "`structural` rows are dropped. The interval resamples whole videos from "
              f"{MIN_CLUSTER_VIDEOS} videos, else it is the Wilson interval treating rows as independent.", ""]
        hh = {cls: f for cls, f in pooled["human_human"].items() if cls != "by_variable"}
        if hh:
            columns = ["field class", "videos", "rows", "coder = coder", "95% interval", "within one",
                       "interval method"]
            L += [f"| {' | '.join(columns)} |", "|" + "---|" * len(columns)]
            for cls, f in hh.items():
                ci = f["ci95"]
                near = _share(f["within_one"]) if "within_one" in f else "-"
                L.append(f"| {cls} | {f['videos']} | {f['n']} | {_share(f)} | "
                         + ("-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}") + f" | {near} | {f['ci_method']} |")
        else:
            L.append("No video qualifies: it needs exactly two coders with a row both gave a definite value.")
    if pooled.get("pairs"):
        L += ["", "## Pair judgements (the clip session)", "",
              "Which of two prepared presentations shows more of a Stage A item: the share of the coders' answers that agree "
              "with the model's rank scores (pooled over coders), coder against coder, and the coders' mean share in the "
              "narrowest and widest third of the model's score gaps; chance is 0.5. The 95% interval resamples videos "
              "(a pair weighted by how often each of its two videos is drawn), the interval the validation table "
              "gives; the exact (Clopper–Pearson) one is in the table's appendix.", "",
              "| item | pairs | answers | coders = model | 95% interval | coder = coder | narrowest third | widest third |",
              "|---|---|---|---|---|---|---|---|"]
        num = lambda x: "-" if x is None else f"{x:.2f}"        # noqa: E731
        for code, f in pooled["pairs"].items():
            ci = f["ci95"]["coder_model"]
            ends = [f["by_gap_tercile"][i] if f["by_gap_tercile"] else None for i in (0, -1)]
            pooled_third = [None if t is None or not t["coder_model"] or None in t["coder_model"].values()
                            else sum(t["coder_model"].values()) / len(t["coder_model"]) for t in ends]
            L.append(f"| {code} | {f['n_pairs']} | {f['n']} | {num(f['coder_model'])} | "
                     + ("-" if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}")
                     + f" | {num(f['coder_coder'])} | {num(pooled_third[0])} | {num(pooled_third[1])} |")
    if pooled.get("structural"):
        L += ["", "## Structural", "",
              "Rows a reviewer marked structural (the unit itself is wrong), per field class, over the rows "
              "that measure validity."]
        L.append("")
        L += [f"- {cls}: {f['n_structural']} of {f['n_rows']} rows structural ({f['share']:.2f})"
              for cls, f in pooled["structural"].items()]
        if pooled.get("structural_by_reviewer"):
            L += ["", "By coder:", ""]
        for name, table in pooled.get("structural_by_reviewer", {}).items():
            L.append(f"- {name}: " + ", ".join(f"{cls} {f['n_structural']}/{f['n_rows']}"
                                                for cls, f in table.items()))
    if pooled["unit_presence"]:
        u = pooled["unit_presence"]
        L += ["", "## Units only one pass recorded", ""]
        for side, text in (("primary_only", "only the primary"), ("replicate_only", "only the replicate")):
            f = u[side]
            share = "-" if not f["n"] else f"{f['judged_present']:.2f}"
            L.append(f"- {text} recorded: {f['n']} judged, share judged present {share}")
    return "\n".join(L) + "\n"


def load_reporting(books: PanelBooks) -> Reporting:
    """The instrument's reporting configuration: pipeline/codebooks/validation_reporting.yaml beside the release
    manifest, the kind of each code and the statistics and reason of each kind, read when the table is written and
    refused (review_report.check_reporting) when the summary could not apply it or the release has none."""
    return _reporting_of_release(books, load_release())


def session_agreement(run: Path, books: PanelBooks) -> dict:
    """review_clips.pair_agreement of the run's clip session: the imported pair judgements of the live design per
    Stage A item, with each coder's ratings of the same clips for coherence ({} when no pair was imported)."""
    out = _clips_dir(run)
    lines = _log_lines(out / review_clips.PAIRS_LOG)
    manifest = _optional(out / review_clips.MANIFEST)
    if not lines or manifest is None:
        return {}
    videos = manifest["videos"]
    lines = review_clips.of_design(lines, manifest.get("design_id"))       # answers to earlier designs leave out
    if not any("batch_id" in line for line in lines):
        return {}
    return review_clips.pair_agreement(lines + review_clips.rating_lines(run, videos), videos,
                                       codes=list(books.stage_a_ids()))


def session_pairs(run: Path, books: PanelBooks) -> dict:
    """pooled_summary["pairs"] of the run: `session_agreement` pooled over the coders ({} when no pair was
    imported)."""
    agreement = session_agreement(run, books)
    return review_clips.pooled_pairs(agreement) if agreement else {}


def cmd_summary(args, books: PanelBooks) -> int:
    run = Path(args.run)
    videos, tolerances, by_reviewer = {}, {}, {}
    for vdir in _video_dirs(run, args.video_ids):
        log = _log_lines(vdir / REVIEW_DIR / LOG)
        lines = validity_lines(log)
        if lines:
            videos[vdir.name] = lines
            tolerances[vdir.name] = Tolerances.of(_optional(vdir / "agreement.json"))
        coders = _lines_by_reviewer(log)
        if len(coders) > 2:
            print(_too_many_coders(vdir.name), file=sys.stderr)
        for name, last in coders.items():
            by_reviewer.setdefault(name, {})[vdir.name] = list(last.values())
    # the statistics of each kind of code: a reporting file the table could not apply refuses the paper table; the
    # plain summary is then written without its per-kind table, and says why
    try:
        reporting = load_reporting(books)
    except ConfigError as exc:
        if args.paper_table:
            raise
        reporting = None
        print(f"note: the agreement per kind of code is left out of the summary: {exc}", file=sys.stderr)
    pairs = session_pairs(run, books)
    strata = sample_strata(run)         # a sample stratified by outcome: figures per stratum and reweighted
    pooled = pooled_summary(videos, tolerances, by_reviewer=by_reviewer, pairs=pairs, strata=strata,
                            reporting=reporting)
    # the table measures coders: a video only an adjudicator reviewed is left out and counted, in the paper table and
    # in the summary's per-kind table alike
    coded = {v: lines for v, lines in videos.items() if any(_role(l) == "coder" for l in lines)}
    if args.paper_table and not coded:
        raise ConfigError("no video has a coder's lines: the paper table is built from coder-coded videos only")
    adjudicator_only = len(videos) - len(coded)
    table_pooled = pooled
    if adjudicator_only or any(v not in coded for per in by_reviewer.values() for v in per):
        table_pooled = (pooled_summary(coded, {v: t for v, t in tolerances.items() if v in coded},
                                       by_reviewer={n: {v: l for v, l in per.items() if v in coded}
                                                    for n, per in by_reviewer.items()}, pairs=pairs, strata=strata,
                                       reporting=reporting) if coded else None)
    text = summary_text(run, pooled, reporting, table_pooled, adjudicator_only)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    if args.paper_table:
        write_paper_table(Path(args.paper_table), table_pooled, reporting, adjudicator_only, clock_differences(coded))
    return 0


def write_paper_table(out: Path, pooled: dict, reporting: Reporting, adjudicator_only: int = 0,
                      clocks: list[dict] | None = None) -> None:
    """OUT (Markdown: the table, its note, coder against coder, pass against pass, the strata and the note on the
    statistics), OUT.json (the same rows and texts) and OUT_appendix.md (every other figure and the definitions);
    prints where they went and each flagged row. No row is passed or failed."""
    note = note_of(pooled, adjudicator_only)
    rows = paper_rows(pooled, reporting)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(paper_table_markdown(pooled, reporting, note), encoding="utf-8")
    strata = pooled.get("strata")
    extra = ({"strata": {"stratify": strata["stratify"], "strata": strata["strata"]},
              "stratum_rows": stratum_rows(pooled, reporting)} if strata else {})
    data = {"summary_rows": summary_rows(pooled, reporting), "rows": rows, "human_human_rows": paper_rows(pooled, reporting, "human_human"),
            "pass_to_pass_rows": paper_rows(pooled, reporting, "pass_to_pass"), "clock_by_video": clocks or [],
            "reporting": {"source": reporting.source, "release_id": reporting.release_id,
                          "sha256": getattr(reporting, "sha256", None), "bands": reporting.bands,
                          "flags": reporting.flags, "comparisons": reporting.comparisons,
                          "not_reported": reporting.not_reported_reasons},
            "note": note, "methods": methods_note(pooled, reporting, note), **extra}
    out.with_suffix(".json").write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                        encoding="utf-8")
    out.with_name(out.stem + "_appendix.md").write_text(paper_appendix_markdown(pooled, reporting, clocks),
                                                        encoding="utf-8")
    print(f"paper table -> {out}")
    for r in rows:
        if r["flags"]:
            print(f"flagged: {r['field']} ({', '.join(f.replace('_', ' ') for f in r['flags'])})")


# ---- entry point --------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="write a review sheet per video")
    p.add_argument("--run", required=True, help="the run directory (runs/<run_name>)")
    p.add_argument("--video-ids", nargs="+")
    p.add_argument("--preset", choices=[EXCHANGE, *PRESETS],
                   help=f"what goes on the sheet (default {EXCHANGE}). "
                        + " ".join(f"{name}: {purpose}." for name, purpose in PRESET_PURPOSE.items())
                        + " An option given beside a preset wins")
    p.add_argument("--sample-videos", type=int, metavar="N",
                   help="review N videos drawn at random (by --seed) from those with both passes, instead of "
                        "--video-ids; the draw is recorded in <run>/review/video_sample.json and a different one is "
                        "refused without --force")
    p.add_argument("--exclude-video-ids", nargs="+", metavar="ID", help="--sample-videos: never draw these videos")
    p.add_argument("--exclude-from", metavar="SELECTION",
                   help="--sample-videos: a JSON file in the run folder; the videos its --exclude-key lists are not drawn")
    p.add_argument("--exclude-key", metavar="KEY", default="pilot_development_videos",
                   help="--exclude-from: the key of the file that lists the videos (default "
                        "pilot_development_videos); a file without it is refused")
    p.add_argument("--stratify", choices=STRATIFY,
                   help="--sample-videos: draw half of the videos from the pitches with an offer and half from those "
                        "without (any_offer, as the analysed record has it; a pitch of unknown outcome is in neither); "
                        "an even N, and enough pitches in each stratum; the strata are recorded in video_sample.json "
                        "and the agreement is then reported per stratum and reweighted to the run's mix")
    p.add_argument("--append", action="store_true",
                   help="--sample-videos: add a draw of new videos to the existing sample instead of replacing it; "
                        "a draw already recorded (same seed, N and exclusions) is returned, not drawn again")
    p.add_argument("--rows-per-field", type=int,
                   help="exchange preset: rows wanted per sampled field over all exported videos; sets "
                        "--qa-per-video unless it is given (never turns the exchange cards on) (applies when given, and "
                        f"with --sample-videos, which defaults it to {ROWS_PER_FIELD})")
    p.add_argument("--stage-a-per-video", metavar="N|all",
                   help=f"exchange preset: Stage A items drawn per video, or 'all' (default {EXCHANGE_DEFAULTS['stage_a']})")
    p.add_argument("--venture-per-video", type=int,
                   help="exchange preset: venture fields drawn per video, the founders' gender first; 0 for none "
                        f"(default {EXCHANGE_DEFAULTS['venture']})")
    p.add_argument("--qa-codes", nargs="+", metavar="ROLE:CODE",
                   help="exchange preset: the codes each drawn question-answer pair puts on the sheet "
                        "(default " + " ".join(f"{r}:{c}" for r, c in QA_CODES) + ")")
    p.add_argument("--exchanges-per-video", type=int,
                   help="exchange preset: substantive exchanges drawn per video, each with its topic "
                        f"(default {EXCHANGE_DEFAULTS['exchanges']}: no exchange cards since 2026-10-21)")
    p.add_argument("--qa-per-video", type=int,
                   help="exchange preset: question-answer pairs drawn per video, each with the codes of --qa-codes "
                        "(by default the question's concept and severity and the answer's responsiveness, evidence, "
                        f"substance and coherence); default {EXCHANGE_DEFAULTS['qa']}, or what --rows-per-field plans")
    p.add_argument("--sample-unflagged", type=int,
                   help="flag-driven presets: unflagged items drawn at random (flagged 20, quick 15, full 30)")
    p.add_argument("--max-flagged", type=int,
                   help="flag-driven presets: at most this many flagged rows (flagged 60, quick 25, full no cap; "
                        "negative for no cap): every unit_presence row and every row a validator error points at "
                        "is kept, the rest is a seeded draw across field classes, and the manifest counts what was "
                        "left out")
    p.add_argument("--field-classes", nargs="+", metavar="CLASS",
                   help=f"only these field classes ({', '.join(SHEET_CLASSES)}); in a flag-driven preset a class "
                        "named here comes whole, with the fields left off by default (actors, free text, the link "
                        "of an answer to the question right before it, outcomes)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blind", action="store_true", help="hide the model's values and the reasons")
    p.add_argument("--out", help="directory for the sheets (default <run>/<video_id>/review/)")
    p.add_argument("--force", action="store_true", help="replace a sheet that already exists")

    p = sub.add_parser("copy-videos", help="copy the full video files of the sample and the clip session into a folder")
    p.add_argument("--run", required=True)
    p.add_argument("--to", required=True, help="the folder to fill")
    p.add_argument("--link", action="store_true", help="hard-link instead of copying (same volume only)")
    p.add_argument("--training", action="store_true",
                   help="copy the training videos (those `training` exported) instead of the validation sample")
    p.add_argument("--clips", action="store_true",
                   help="refused since 2026-10-21: the clip session plays the full videos at their prepared windows, "
                        "so no clips are cut; copy-videos without it copies those full videos")

    p = sub.add_parser("clips-export", help="the clip session: pair design, session sheet and one page per coder")
    p.add_argument("--run", required=True)
    p.add_argument("--coders", nargs="+", required=True, help="the coders' names: letters, digits, - and _")
    p.add_argument("--video-ids", nargs="+", help="the session's videos (default the validation sample)")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the pair design and each coder's clip order (default 0)")
    p.add_argument("--no-codebook", action="store_true", help="leave the codebook off the pages")
    p.add_argument("--force", action="store_true", help="export the session again over an earlier export")

    p = sub.add_parser("session-export", help="one page per coder as a task list: the clip session and each video's "
                                              "blind full-video sheet in one step")
    p.add_argument("--run", required=True)
    p.add_argument("--coders", nargs="+", required=True, help="the coders' names: letters, digits, - and _")
    p.add_argument("--video-ids", nargs="+", help="the session's videos (default the validation sample)")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the pair design and each coder's video order (default 0); not the full-video draw")
    p.add_argument("--qa-per-video", type=int,
                   help="question-answer pairs drawn per video on the full-video sheets (default: planned from "
                        f"--rows-per-field {ROWS_PER_FIELD} for the sample, else the export's default)")
    p.add_argument("--rows-per-field", type=int,
                   help="rows wanted per sampled field over the session's videos; plans --qa-per-video")
    p.add_argument("--sheet-seed", type=int,
                   help="seed of a new full-video sheet's draw for a video no sample draw chose (default 0); a sample "
                        "video's sheet is drawn with its draw's seed, and an existing sheet keeps its own")
    p.add_argument("--no-codebook", action="store_true", help="leave the codebook off the pages")
    p.add_argument("--force", action="store_true",
                   help="export the session again over an earlier one, and draw each full-video sheet again with its "
                        "own recorded seed and plan")

    p = sub.add_parser("clips-import", help="import a coder's clip-session sheet (of the live design, by one of "
                                            "the session's coders); importing it again replaces earlier answers, "
                                            "and a pair left blank since is withdrawn")
    p.add_argument("--run", required=True)
    p.add_argument("--sheet", required=True, help="the coder's filled session sheet (clips__<coder>.csv), or the "
                                                  "coder's zip of sheets (<coder>__sheets.zip) or a folder holding it")
    p.add_argument("--reviewer", required=True, help="the coder's name: letters, digits, - and _")

    p = sub.add_parser("import", help="apply a filled sheet")
    p.add_argument("--run", required=True)
    p.add_argument("--sheet", required=True, help="the filled CSV, beside its export_manifest.json")
    p.add_argument("--reviewer", required=True, help="the coder's or adjudicator's name: letters, digits, - and _")
    p.add_argument("--video-id", help="when the manifest does not name the sheet")
    p.add_argument("--role", choices=ROLES,
                   help="coder: the sheet is logged and measured and never corrects the record; adjudicator: its "
                        "values are written into results/reviewed.json (default: coder for a sheet exported "
                        "--blind, adjudicator otherwise)")

    p = sub.add_parser("import-folder", help="import every <video_id>__<reviewer>.csv of a folder")
    p.add_argument("--run", required=True)
    p.add_argument("--dir", required=True, help="the folder of filled sheets, named <video_id>__<reviewer>.csv, "
                                                "or a coder's zip of them (<coder>__sheets.zip)")
    p.add_argument("--export-dir", help="where the manifest and the exported sheets are, when not the default")
    p.add_argument("--role", choices=ROLES,
                   help="default coder, whatever the export was: a folder of sheets is what coders hand in; "
                        "adjudicator writes their values into results/reviewed.json")

    p = sub.add_parser("page", help="write one self-contained coding page for a coder")
    p.add_argument("--run", required=True)
    p.add_argument("--coder", required=True, help="the coder's name: letters, digits, - and _")
    p.add_argument("--video-ids", nargs="+")
    p.add_argument("--sample", action="store_true", help="the videos of <run>/review/video_sample.json")
    p.add_argument("--seed", type=int, default=0,
                   help="the order of the videos on the page is shuffled per coder, seeded by this and the coder's "
                        "name (default 0); recorded in the page as order_seed")
    p.add_argument("--export-dir", help="where the blind sheets were exported, when not the default")
    p.add_argument("--out", help="directory for the page (default <run>/review/pages/)")
    p.add_argument("--no-codebook", action="store_true",
                   help="leave the codebook off the page (by default the whole codebook is laid out on it)")

    p = sub.add_parser("discuss", help="a discussion page for the researcher and one coder: where the coder and "
                                       "Gemini disagree (outside the validation; its sheet is never imported)")
    p.add_argument("--run", required=True)
    p.add_argument("--coder", required=True, help="the coder whose imported answers are discussed")
    p.add_argument("--pairs", type=int, default=8,
                   help="comparisons to draw from those the coder chose against the model (default 8)")
    p.add_argument("--seed", type=int, default=0, help="seeds the draw of the comparisons (default 0)")
    p.add_argument("--force", action="store_true", help="build the page again over an earlier one")

    p = sub.add_parser("training", help="training sheets and pages for the coders, outside the validation")
    p.add_argument("--run", required=True)
    p.add_argument("--coders", nargs="+", required=True, help="the coders' names: letters, digits, - and _")
    p.add_argument("--video-ids", nargs="+", help="the training videos (never one of the validation sample)")
    p.add_argument("--from", dest="from_selection", metavar="SELECTION",
                   help="draw the training videos from a JSON file in the run folder, from the list under --key")
    p.add_argument("--key", default="pilot_development_videos",
                   help="--from: the key that lists the videos (default pilot_development_videos)")
    p.add_argument("--videos", type=int, default=2, help="--from: how many videos to draw (default 2)")
    p.add_argument("--exchanges-per-video", type=int, default=TRAINING_DRAW["exchanges"])
    p.add_argument("--qa-per-video", type=int, default=TRAINING_DRAW["qa"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="directory for the training export and pages (default <run>/review/training/)")
    p.add_argument("--clips", action="store_true",
                   help="also the training clip session (exactly two videos; needs <run>/ranking/scores.json): each "
                        "training video at its prepared window with its seven cards, then their pair three times; the "
                        "full-video sheets then leave the Stage A items to it -> <out>/clips/")
    p.add_argument("--force", action="store_true", help="replace training sheets that already exist")

    p = sub.add_parser("calibrate", help="the trainees' agreement with each other, from their training sheets")
    p.add_argument("--run", required=True)
    p.add_argument("--dir", required=True, help="the folder of filled training sheets, <video_id>__<coder>.csv")
    p.add_argument("--training-dir", help="where `training` wrote its export (default <run>/review/training/)")
    p.add_argument("--bar", type=float,
                   help="the coder-against-coder alpha a field is compared with (default 0.61, the lower bound of "
                        "Landis & Koch's \"substantial\" band, from validation_reporting.yaml); the pair "
                        "judgements are read against 0.5")

    p = sub.add_parser("status", help="what has been exported, reviewed and corrected")
    p.add_argument("--run", required=True)
    p.add_argument("--video-ids", nargs="+")
    p.add_argument("--out", help="where the sheets were exported, when not the default")

    p = sub.add_parser("summary", help="agreement pooled over every imported video")
    p.add_argument("--run", required=True)
    p.add_argument("--video-ids", nargs="+")
    p.add_argument("--out", help="also write the summary to this Markdown file")
    p.add_argument("--paper-table", metavar="OUT.md",
                   help="also write the paper's validation table (OUT.md, OUT.json, OUT_appendix.md): each code "
                        "with the statistics of its kind (validation_reporting.yaml), no pass or fail")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        books = load_panel_books(load_release())
        commands = {"export": cmd_export, "copy-videos": cmd_copy_videos, "clips-export": cmd_clips_export, "session-export": cmd_session_export, "clips-import": cmd_clips_import, "import": cmd_import, "import-folder": cmd_import_folder, "page": cmd_page, "discuss": cmd_discuss, "training": cmd_training, "calibrate": cmd_calibrate, "status": cmd_status, "summary": cmd_summary}
        return commands[args.command](args, books)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
