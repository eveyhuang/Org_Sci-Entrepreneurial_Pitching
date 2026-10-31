"""Human agreement on the training videos: the tables and figures of notebooks/human agreement/.

It follows the design of the human validation (docs/plans/human_validation/2026-10-08-validation-specification.md,
§4 as rewritten on 2026-10-30, plan D): coder against coder is the ceiling, each coder against the model and the
model's two passes against each other sit beside it. Per code, where the pooled figures carry plan D's reporting
(`pooled["reporting"]`), each comparison shows the statistics of the code's kind as the paper table does: raw
agreement (the identical share; within one point on a rating scale; same turn for the first decision) with
Krippendorff's alpha beside it (none for the clock and the ask), Gwet's AC1 or AC2, alpha's Landis & Koch band and the
row's flags (prevalence, small n); the question concept also gets its frame's row. Without that section the per-code
figures are the pooled summary's alpha (stratified by variable within a field class) and plain agreement. The bar
(0.61, the lower bound of Landis & Koch's "substantial" band, from validation_reporting.yaml) is a guide for the
calibration discussion, not a gate. The coders' sheets are read the way `review_panel.py calibrate` reads them:
imported into a throwaway copy of the run, so nothing of the run is written.

Nothing is estimated here that the review pipeline does not already compute: `pooled_summary` gives every alpha and
share; this module only lays them out per code, adds the per-card table a calibration discussion needs, and draws.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis import panel_figures as pf
from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import PanelBooks, load_panel_books
from pipeline.stage_b.review_apply import (CUT_NOTE, MISSED_NOTE, NO_EXCHANGE_NOTE, NOT_PRESENT_NOTE, OVERRIDE_NOTE,
                                           UNSURE, Tolerances, agrees, is_unsure, variable_of, within_one,
                                           decision_turn_comparison)
from pipeline.stage_b.review_items import OUTCOME, banded, is_outcome
from pipeline.stage_b.review_labels import entry_for
from pipeline.errors import ConfigError
from pipeline.stage_b.review_report import (ALPHA_KINDS, REPORTING_BAND_FLOORS, paper_rows, stratum_rows,
                                            stratum_word)

ROOT = Path(__file__).resolve().parents[1]
UNDECIDED = ("unclear", "structural")
# the notes the import writes itself when a coder left none: not the coder's words, so never shown as theirs
PIPELINE_NOTES = frozenset({OVERRIDE_NOTE, CUT_NOTE, NO_EXCHANGE_NOTE, NOT_PRESENT_NOTE, MISSED_NOTE})


def coder_note(line: dict | None) -> str:
    """The coder's own words: not a note the import wrote, and without the page's "not sure" marker ([unsure]),
    which `cards` shows as its own column."""
    note = ((line or {}).get("note") or "").strip()
    return "" if note in PIPELINE_NOTES else " ".join(note.replace(UNSURE, " ").split())
# how a row of the tables is ordered: the order in which a coder meets the fields on the task page, a video's
# presentation cards first, then its question-answer cards and the first decision, then the comparisons (a class
# not named here, such as the structural prepared_window row, follows them)
CLASS_ORDER = ("stage_a", "prep_pres", "venture", "turn_code", "cut", OUTCOME, "pair")
CLASS_WORDS = {"stage_a": "Prepared presentation: delivery (Stage A)",
               "prep_pres": "Prepared presentation: content", "venture": "The venture and the ask",
               "turn_code": "Question and answer codes",
               "cut": "When the first decision is spoken", OUTCOME: "Made or joined an offer",
               "pair": "Which of two pitches comes across as more … (comparisons)",
               "prepared_window": "Where the prepared presentation is"}
# A class outside CLASS_ORDER (the structural prepared_window row, or the exchange topics of a sheet exported with
# `--exchanges-per-video`, which the full-video sheet no longer draws by default) is kept, never dropped: its cards
# and its agreement row follow the ordered classes, named here or, failing that, by the class key itself. The
# reporting file (validation_reporting.yaml) does not report it, so the paper table has no row for it either.
OTHER_CLASS_WORDS = {"exchange_topic": "Topic of a line of questioning (only on sheets that asked for it)"}


def class_word(cls: str) -> str:
    """The plain name of a field class: ordered classes first, then the others, then the key itself."""
    return CLASS_WORDS.get(cls) or OTHER_CLASS_WORDS.get(cls) or cls


def class_rank(cls: str) -> int:
    """Where a class sorts: its place in CLASS_ORDER, every other class after them (ties broken by the caller)."""
    return CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)


SERIES = (("coder_coder", "Coder vs coder (the ceiling)", pf.BLACK, "o", True),
          ("coder_a_model", "{a} vs model", pf.BLUE, "s", True),
          ("coder_b_model", "{b} vs model", pf.VERMILLION, "^", True),
          ("pass_pass", "Model pass 1 vs pass 2", pf.GREY, "D", False))


def review_panel():
    """scripts/review_panel.py as a module (it is a script, not a package)."""
    name = "review_panel_for_human_agreement"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "review_panel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class TrainingAgreement:
    run: Path
    coders: list[str]
    videos: list[str]
    logs: dict[str, list[dict]]
    agreements: dict[str, dict | None]
    pooled: dict
    books: PanelBooks
    bar: float
    import_failed: bool
    import_log: str = ""
    pairs: dict = field(default_factory=dict)     # review_clips.pair_agreement of the clip session ({} without one)
    reporting: object = None                       # review_report.Reporting (validation_reporting.yaml)


def load(run: Path | str, results_dir: Path | str, bar: float | None = None,
         training_dir: Path | str | None = None) -> TrainingAgreement:
    """The coders' training sheets in `results_dir`, pooled as the validation pools: the full-video sheets
    (`<video_id>__<coder>.csv`) and, when the training had a clip session, the clip-session sheets
    (`clips__<coder>.csv`), whose cards join the same videos' rows and whose pair judgements give `pairs`."""
    rp = review_panel()
    run = Path(run)
    books = load_panel_books(load_release())
    said = io.StringIO()
    with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):     # the import's messages, kept
        logs, status, agreements, pairs = rp.training_imports(run, Path(results_dir), books,
                                                              Path(training_dir) if training_dir else None)
    videos, tolerances, by_reviewer = rp.pool_training(logs, agreements)
    if not by_reviewer:
        raise ValueError(f"no training sheet was imported from {results_dir}")
    # a stratified validation sample gives the stratum table (agreement_by_stratum); the training videos are never in
    # it, so their pooled figures carry no strata
    try:
        reporting = rp.load_reporting(books)
    except ConfigError as exc:
        # a reporting file the release refuses: the pooled summary's figures without plan D's section, and the
        # default bar from the Landis & Koch floors directly
        print(f"note: the agreement per kind of code is left out: {exc}", file=sys.stderr)
        reporting = None
    pooled = rp.pooled_summary(videos, tolerances, by_reviewer=by_reviewer,
                               pairs=rp.review_clips.pooled_pairs(pairs) if pairs else None,
                               strata=rp.sample_strata(run), reporting=reporting)
    if bar is None:
        bar = (reporting.band_floor(rp.CALIBRATION_BAND) if reporting is not None
               else REPORTING_BAND_FLOORS["landis_koch"][rp.CALIBRATION_BAND])
    return TrainingAgreement(run=run, coders=sorted(by_reviewer), videos=sorted(videos), logs=logs,
                             agreements=agreements, pooled=pooled, books=books, bar=float(bar),
                             import_failed=bool(status), import_log=said.getvalue(), pairs=pairs,
                             reporting=reporting)


# ---- per card ------------------------------------------------------------------------------------------

def value_of(line: dict) -> str | None:
    """What one coder's line says the value is (None for "I can't tell" and "the unit is wrong")."""
    resolution = line.get("resolution")
    if resolution in UNDECIDED:
        return None
    if resolution == "confirm_primary":
        typed = (line.get("typed") or "").strip()
        return typed or line.get("primary")
    return line.get("new")


def code_label(variable: str, path: str, books: PanelBooks) -> str:
    """A plain name for a variable: the codebook entry's name (an item's id when it has none)."""
    if is_outcome(path):
        return CLASS_WORDS[OUTCOME]
    entry = entry_for(path, books)
    if entry is None:
        return variable
    return entry.name or entry.id.replace("_", " ").capitalize()


def class_of(line: dict) -> str:
    return line.get("field_class") or "unknown"


def cards(data: TrainingAgreement) -> pd.DataFrame:
    """One row per card a coder judged on a training video: both coders' answers, the model's two passes, the
    coders' notes and flags, and whether the coders agree (as the pipeline compares values)."""
    rp = review_panel()
    a, b = (data.coders + [None, None])[:2]
    rows = []
    for video_id in data.videos:
        by_coder = rp._lines_by_reviewer(data.logs[video_id])
        tolerances = Tolerances.of(data.agreements.get(video_id))
        paths = sorted({p for lines in by_coder.values() for p in lines})
        for path in paths:
            lines = {c: by_coder.get(c, {}).get(path) for c in data.coders}
            any_line = next(line for line in lines.values() if line)
            row = {"video": video_id, "path": path, "class": class_of(any_line), "variable": variable_of(path),
                   "code": code_label(variable_of(path), path, data.books), "level": any_line.get("level"),
                   # the model's claim count (PRES-EVID) shown as the band the coders chose from (final review M4)
                   "model": banded(path, any_line.get("primary")),
                   "model_pass_2": banded(path, any_line.get("replicate"))}
            for c, line in lines.items():
                row[c] = value_of(line) if line else None
                row[f"{c} flag"] = (line or {}).get("resolution") if (line or {}).get("resolution") in UNDECIDED else ""
                row[f"{c} unsure"] = bool(line) and is_unsure(line)       # a best guess ticked "not sure"
                row[f"{c} note"] = coder_note(line)
            va, vb = (row.get(a), row.get(b)) if b else (None, None)
            both = va is not None and vb is not None
            row["coders agree"] = bool(agrees(path, va, vb, tolerances)) if both else None
            if both and path == 'pitch:first_decision':
                context = (lines[a] or {}).get('decision_turn_context')
                if context != (lines[b] or {}).get('decision_turn_context'):
                    context = None
                row['coders agree'] = decision_turn_comparison(va, vb, context)['agreed']
            row["within one"] = (bool(within_one(va, vb, path)) if both and row["level"] == "ordinal" else None)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["class_rank"] = frame["class"].map(class_rank)
    return frame.sort_values(["class_rank", "variable", "video", "path"]).drop(columns="class_rank").reset_index(drop=True)


def sheets_summary(data: TrainingAgreement) -> pd.DataFrame:
    """Per coder and video: cards answered, "I can't tell", "the unit is wrong", "not sure" (a best guess), cards with
    a note."""
    rp = review_panel()
    rows = []
    for video_id in data.videos:
        for coder, lines in rp._lines_by_reviewer(data.logs[video_id]).items():
            values = list(lines.values())
            rows.append({"coder": coder, "video": video_id, "cards answered": len(values),
                         "I can't tell": sum(l.get("resolution") == "unclear" for l in values),
                         "the unit is wrong": sum(l.get("resolution") == "structural" for l in values),
                         "not sure": sum(is_unsure(l) for l in values),
                         "with a note": sum(bool(coder_note(l)) for l in values)})
    return pd.DataFrame(rows).sort_values(["coder", "video"]).reset_index(drop=True)


# ---- agreement tables ----------------------------------------------------------------------------------

def _figure_row(key: str, scope: str, data: TrainingAgreement, label: str) -> dict:
    pooled = data.pooled
    a, b = (data.coders + [None, None])[:2]
    if scope == "class":
        hh = (pooled.get("human_human") or {}).get(key) or {}
        model = (pooled.get("sampled") or {}).get(key) or {}
        per = {c: ((pooled.get("by_reviewer") or {}).get(c, {}).get("sampled") or {}).get(key) or {} for c in data.coders}
    else:
        hh = ((pooled.get("human_human") or {}).get("by_variable") or {}).get(key) or {}
        model = (pooled.get("by_variable") or {}).get(key) or {}
        per = {c: ((pooled.get("by_reviewer") or {}).get(c, {}).get("by_variable") or {}).get(key) or {}
               for c in data.coders}
    share = lambda f, k="human_vs_primary": (f.get(k) or {}).get("agree")             # noqa: E731
    alpha = hh.get("alpha")
    # the coders against the model (pooled) without the rows a coder ticked "not sure"
    without = pooled.get("without_unsure") or {}
    sure = ((without.get("sampled") if scope == "class" else without.get("by_variable")) or {}).get(key) or {}
    row = {"scope": scope, "key": key, "code": label, "videos": hh.get("videos") or model.get("videos"),
           "rows (coder pairs)": hh.get("n"), "α units": hh.get("alpha_units"),
           "coder vs coder: agreement": hh.get("agree"), "coder vs coder: α": alpha,
           "coder vs coder: within one": (hh.get("within_one") or {}).get("agree"),
           f"{a} vs model: α": per.get(a, {}).get("alpha"), f"{a} vs model: agreement": share(per.get(a, {})),
           "coders vs model: α without unsure rows": sure.get("alpha"),
           "coders vs model: agreement without unsure rows": share(sure),
           "model pass vs pass: α": model.get("alpha_primary_vs_replicate"),
           "model pass vs pass: agreement": share(model, "primary_vs_replicate"),
           "level": hh.get("alpha_level") or model.get("alpha_level"),
           # at the two decimals the tables print, as `review_panel.py calibrate` reads it
           "reached the bar": None if alpha is None else round(alpha, 2) >= round(data.bar, 2)}
    if b:
        row.update({f"{b} vs model: α": per.get(b, {}).get("alpha"), f"{b} vs model: agreement": share(per.get(b, {}))})
    return row


# ---- plan D's reporting (pooled["reporting"]), per code ---------------------------------------------------------

# the comparisons of pooled["reporting"] and the words of their columns
REPORTED = (("human_human", "coder vs coder"), ("coder_model", "coders vs model"), ("pass_to_pass", "model pass vs pass"))
# the share that leads a row of each kind (review_report.HEADLINE's first figure)
LEADING = {"categorical": "exact", "binary": "exact", "ordinal": "exact", "clock": "same_turn", "exact": "exact",
           "comparison": "agreement"}
FLAG_WORDS = {"prevalence": "prevalence (read AC1/AC2)", "small_n": "small n"}
NOT_REPORTED = "not reported (its frame is)"


def reported_rows(pooled: dict, reporting=None) -> dict[str, dict[str, dict]]:
    """{comparison: {name: row}} of plan D's reporting (`pooled["reporting"]`), as review_report.paper_rows gives them
    (figures, alpha, its band, flags); {} when the pooled figures carry no reporting (an older pooled summary)."""
    if not pooled.get("reporting"):
        return {}
    return {which: {r["field"]: r for r in paper_rows(pooled, reporting, which) if not r["missing"]}
            for which, _ in REPORTED}


def reported_name(variable: str, reporting=None) -> str:
    """The name a pooled variable has in the reporting: a Stage A item and a prepared-presentation code by bare id."""
    if variable.startswith("pair:"):
        return variable
    if reporting is not None:
        return reporting.aliases.get(variable, variable)
    unit, _, rest = variable.partition(":")
    return rest if unit in ("stage_a", "prep_pres") and rest else variable


def _value(row: dict | None, name: str):
    return (((row or {}).get("figures") or {}).get(name) or {}).get("value")


def _reported_figures(rows: dict[str, dict[str, dict]], name: str) -> dict:
    """One code's figures of every comparison the reporting has it in: raw agreement, within one (ordinal), alpha
    (categorical, binary, ordinal), AC1 or AC2, alpha's band, the flags, the units."""
    out = {}
    for which, words in REPORTED:
        r = rows.get(which, {}).get(name)
        if r is None:
            continue
        kind = r["kind"]
        out["kind"] = out.get("kind") or kind
        out[f"{words}: agreement"] = _value(r, LEADING.get(kind, "exact"))
        out[f"{words}: within one"] = _value(r, "within_one") if kind == "ordinal" else None
        out[f"{words}: α"] = (r.get("alpha") or {}).get("value")
        out[f"{words}: AC1/AC2"] = _value(r, "ac2" if kind == "ordinal" else "ac1") if kind in ALPHA_KINDS else None
        out[f"{words}: band of α"] = r.get("band")
        out[f"{words}: flags"] = ", ".join(FLAG_WORDS.get(f, f) for f in r.get("flags") or [])
        out[f"{words}: n"] = r.get("n")
    return out


def _with_reporting(row: dict, data: TrainingAgreement, rows: dict[str, dict[str, dict]]) -> dict:
    """A per-code row of `_figure_row` with plan D's figures where the reporting has the code: the comparisons'
    agreement, within one and alpha replaced by the reported ones (no alpha for the clock and the ask), AC1/AC2, the
    band and the flags added; each coder's alpha against the model left out for a kind without alpha. The bar is
    read against the reported coder-against-coder alpha."""
    reporting = getattr(data, "reporting", None)
    name = reported_name(row["key"], reporting)
    if reporting is not None and name in reporting.not_reported:
        return row | {"kind": NOT_REPORTED}
    figures = _reported_figures(rows, name)
    if not figures:
        return row
    row = row | figures
    if figures["kind"] not in ALPHA_KINDS:
        # no alpha for the clock or the ask: the older pooled columns' alpha, its units and level go too, and the
        # clock shows its median seconds apart instead
        for c in data.coders:
            row[f"{c} vs model: α"] = None
        row.update({"coders vs model: α without unsure rows": None, "α units": None, "level": None})
        if figures["kind"] == "clock":
            for which, words in REPORTED:
                r = rows.get(which, {}).get(name)
                row[f"{words}: median seconds apart"] = _value(r, "median_abs") if r is not None else None
            # without the unsure rows, the same turn-identity share as the reported agreement
            sure = (rows.get("coder_model", {}).get(name) or {}).get("without_unsure") or {}
            row["coders vs model: agreement without unsure rows"] = sure.get("same_turn")
    alpha = row.get("coder vs coder: α")
    row["reached the bar"] = None if alpha is None else round(alpha, 2) >= round(data.bar, 2)
    return row


def _derived_rows(data: TrainingAgreement, rows: dict[str, dict[str, dict]], source: dict) -> list[dict]:
    """The reporting's rows derived from a code (the question's frame, PR-FRAME, from PR-CONCEPT), placed after it."""
    reporting = getattr(data, "reporting", None)
    if reporting is None:
        return []
    code = reported_name(source["key"], reporting)
    out = []
    for name, entry in reporting.derived.items():
        if entry.get("from") != code:
            continue
        figures = _reported_figures(rows, name)
        if not figures:
            continue
        alpha = figures.get("coder vs coder: α")
        out.append({"scope": "variable", "key": name, "code": f"{source['code']}: its frame ({name})",
                    "class": source.get("class"), "videos": None, "rows (coder pairs)": figures.get("coder vs coder: n"),
                    "level": "nominal" if entry.get("kind") in ("categorical", "binary") else entry.get("kind"),
                    **figures, "reached the bar": None if alpha is None else round(alpha, 2) >= round(data.bar, 2)})
    return out


def agreement_by_class(data: TrainingAgreement) -> pd.DataFrame:
    """One row per field class (Stage A is one class of ten items; its alpha is stratified by item)."""
    keys = [k for k in (data.pooled.get("human_human") or {}) if k != "by_variable"]
    keys += [k for k in (data.pooled.get("sampled") or {}) if k not in keys]
    rows = [_figure_row(k, "class", data, class_word(k)) for k in sorted(keys, key=lambda k: (class_rank(k), k))]
    return _columns(pd.DataFrame(rows), data)


def agreement_by_code(data: TrainingAgreement) -> pd.DataFrame:
    """One row per code: each turn code, each Stage A item, and the one-code classes. Where the pooled figures carry
    plan D's reporting, each code shows its kind and, per comparison, the reported figures (`_with_reporting`), and
    the question concept's frame follows the concept as a row of its own; the concept itself keeps its pooled
    figures, marked as not reported. Without the reporting the rows are the pooled summary's alpha and agreement."""
    table = cards(data)
    if table.empty:
        return pd.DataFrame()
    firsts = table.drop_duplicates("variable")
    reporting = getattr(data, "reporting", None)
    reported = reported_rows(data.pooled, reporting)
    rows = []
    for r in firsts.to_dict("records"):
        row = _figure_row(r["variable"], "variable", data, r["code"]) | {"class": r["class"]}
        if reported:
            row = _with_reporting(row, data, reported)
        rows.append(row)
        if reported:
            rows.extend(_derived_rows(data, reported, row))
    return _columns(pd.DataFrame(rows), data)


def _columns(frame: pd.DataFrame, data: TrainingAgreement) -> pd.DataFrame:
    a, b = (data.coders + [None, None])[:2]
    wanted = ["code", "class", "kind", "videos", "rows (coder pairs)", "α units", "coder vs coder: agreement",
              "coder vs coder: within one", "coder vs coder: α", "coder vs coder: AC1/AC2", "coder vs coder: band of α",
              "coder vs coder: flags", "reached the bar", f"{a} vs model: α", f"{a} vs model: agreement",
              *([f"{b} vs model: α", f"{b} vs model: agreement"] if b else []),
              "coders vs model: agreement", "coders vs model: within one", "coders vs model: α",
              "coders vs model: AC1/AC2", "coders vs model: band of α", "coders vs model: flags",
              "coders vs model: α without unsure rows", "coders vs model: agreement without unsure rows",
              "model pass vs pass: agreement", "model pass vs pass: within one", "model pass vs pass: α",
              "model pass vs pass: AC1/AC2", "coder vs coder: median seconds apart",
              "coders vs model: median seconds apart", "model pass vs pass: median seconds apart", "level", "key",
              "scope"]
    return frame[[c for c in wanted if c in frame.columns]]


def agreement_by_stratum(data) -> pd.DataFrame:
    """When the coded videos are a sample stratified by outcome (pooled["strata"], plan C Task 7f): one row per row of
    the paper table (review_report.stratum_rows, from pooled["reporting"]), each stratum's coder-against-model alpha
    and leading agreement, and over the stratified videos the alpha and agreement reweighted to the run's mix beside
    the pooled (unweighted) ones. Empty for an unstratified set (the training videos)."""
    strata = data.pooled.get("strata")
    if not strata:
        return pd.DataFrame()
    rows = []
    for r in stratum_rows(data.pooled, getattr(data, "reporting", None)):
        row = {"key": r["field"], "kind": r["kind"], "rows": r["rows"]}
        for s, f in r["by_stratum"].items():
            word = stratum_word(strata.get("stratify"), s)
            row.update({f"{word}: α": f["alpha"], f"{word}: agreement": f["share"], f"{word}: rows": f["n"]})
        row.update({"reweighted α": r["reweighted_alpha"], "unweighted α": r["unweighted_alpha"],
                    "reweighted agreement": r["reweighted_share"], "unweighted agreement": r["unweighted_share"]})
        rows.append(row)
    return pd.DataFrame(rows)


def disagreements(data: TrainingAgreement) -> pd.DataFrame:
    """The cards on which the coders differ, or one of them could not decide: what a calibration session goes
    through, with the model's answer beside theirs and both coders' notes."""
    table = cards(data)
    if table.empty:
        return table
    a, b = (data.coders + [None, None])[:2]
    keep = (table["coders agree"] == False) | (table[f"{a} flag"] != "") | ((table[f"{b} flag"] != "") if b else False)  # noqa: E712
    cols = ["code", "video", "path", a, *([b] if b else []), "model", "model_pass_2", f"{a} flag", f"{a} note",
            *([f"{b} flag", f"{b} note"] if b else [])]
    return table.loc[keep, cols].reset_index(drop=True)


def confusion(data: TrainingAgreement, variable: str) -> pd.DataFrame:
    """Coder against coder for one categorical code: how often each pair of answers was given."""
    table = cards(data)
    a, b = data.coders[:2]
    part = table[table["variable"] == variable]
    if part.empty:
        return pd.DataFrame()
    return pd.crosstab(part[a].fillna("(no answer)").rename(a), part[b].fillna("(no answer)").rename(b))


# ---- figures -------------------------------------------------------------------------------------------

def _labels(frame: pd.DataFrame) -> list[str]:
    return [f"{code}" + (f"  (n={int(n)})" if pd.notna(n) else "") for code, n in zip(frame["code"], frame["rows (coder pairs)"])]


def figure_agreement(data: TrainingAgreement, by_code: pd.DataFrame | None = None, width=pf.DEFAULT_WIDTH):
    """Two panels, one row per code: A, Krippendorff's alpha; B, plain agreement. In each, coder vs coder (the
    ceiling), each coder vs the model, and the model's two passes; the bar is the vertical line in panel A."""
    import matplotlib.pyplot as plt

    by_code = agreement_by_code(data) if by_code is None else by_code
    a, b = (data.coders + [None, None])[:2]
    columns = {"alpha": {"coder_coder": "coder vs coder: α", "coder_a_model": f"{a} vs model: α",
                         "coder_b_model": f"{b} vs model: α", "pass_pass": "model pass vs pass: α"},
               "share": {"coder_coder": "coder vs coder: agreement", "coder_a_model": f"{a} vs model: agreement",
                         "coder_b_model": f"{b} vs model: agreement", "pass_pass": "model pass vs pass: agreement"}}
    labels = _labels(by_code)
    with pf._context():
        fig, axes = plt.subplots(1, 2, figsize=(pf.width_inches(width), 0.24 * len(labels) + 1.4), dpi=pf.SCREEN_DPI,
                                 layout="constrained", sharey=True)
        offsets = {"coder_coder": -0.27, "coder_a_model": -0.09, "coder_b_model": 0.09, "pass_pass": 0.27}
        for i, (ax, measure, title) in enumerate(zip(axes, ("alpha", "share"),
                                                     ("Krippendorff's α", "Agreement (share of rows)"))):
            for key, words, colour, marker, filled in SERIES:
                column = columns[measure].get(key)
                if column not in by_code.columns:
                    continue
                for y, value in enumerate(by_code[column]):
                    if value is not None and pd.notna(value):
                        pf._point(ax, max(float(value), -0.5), y + offsets[key], colour, marker, filled, size=4.5)
            ax.set_xlim(-0.55 if measure == "alpha" else 0, 1.02)
            classes = list(by_code["class"]) if "class" in by_code.columns else [None] * len(labels)
            for y in range(len(labels) - 1):
                boundary = classes[y] != classes[y + 1]
                ax.axhline(y + 0.5, color=pf.GREY if boundary else pf.GRID, linewidth=0.8 if boundary else 0.4, zorder=0)
            if measure == "alpha":
                ax.axvline(data.bar, color=pf.INK, linewidth=0.8, linestyle="--", zorder=1)
                ax.axvline(0, color=pf.GRID, linewidth=0.6, zorder=0)
                pf._reference_ticks(ax, data.bar, "bar")
                ax.set_xticks(sorted({-0.5, 0, 0.5, 1, data.bar}))
            pf._letter(ax, i, title)
        axes[0].set_yticks(range(len(labels)), labels)          # shared rows: set once, on the left panel
        axes[0].invert_yaxis()
        axes[1].tick_params(labelleft=False)
        if (by_code[[c for c in columns["alpha"].values() if c in by_code.columns]].astype(float) < -0.5).any().any():
            axes[0].text(0.0, -0.06, "an alpha below −0.5 is drawn at −0.5", transform=axes[0].transAxes,
                         ha="left", va="top", color=pf.GREY)
        handles = [pf._handle(colour, marker, filled, words.format(a=a, b=b), line=False)
                   for key, words, colour, marker, filled in SERIES if columns["alpha"].get(key) in by_code.columns]
        fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)))
    return fig


def stage_a_ratings(data: TrainingAgreement) -> pd.DataFrame:
    """Each Stage A item's rating per video by each coder and by the model's two passes."""
    table = cards(data)
    part = table[table["class"] == "stage_a"]
    cols = ["code", "video", *data.coders, "model", "model_pass_2"]
    return part[cols].reset_index(drop=True)


def figure_stage_a(data: TrainingAgreement, width=pf.DEFAULT_WIDTH):
    """One panel per training video: each Stage A item's rating (1-5) by each coder and by the model."""
    import matplotlib.pyplot as plt

    ratings = stage_a_ratings(data)
    if ratings.empty:
        return None
    a, b = (data.coders + [None, None])[:2]
    raters = [(a, pf.BLUE, "s", True, a), *([(b, pf.VERMILLION, "^", True, b)] if b else []),
              ("model", pf.BLACK, "o", False, "model (pass 1)")]
    items = list(dict.fromkeys(ratings["code"]))
    with pf._context():
        fig, axes = plt.subplots(1, len(data.videos), figsize=(pf.width_inches(width), 0.26 * len(items) + 1.4),
                                 dpi=pf.SCREEN_DPI, layout="constrained", sharey=True, squeeze=False)
        for i, (ax, video) in enumerate(zip(axes[0], data.videos)):
            part = ratings[ratings["video"] == video].set_index("code")
            for j, (column, colour, marker, filled, _) in enumerate(raters):
                for y, item in enumerate(items):
                    value = part[column].get(item) if item in part.index else None
                    try:
                        pf._point(ax, float(value), y + (j - 1) * 0.18, colour, marker, filled, size=4.5)
                    except (TypeError, ValueError):
                        continue
            ax.set_xlim(0.6, 5.4)
            ax.set_xticks(range(1, 6))
            for y in range(len(items)):
                ax.axhline(y + 0.5, color=pf.GRID, linewidth=0.4, zorder=0)
            pf._letter(ax, i, video)
            ax.set_xlabel("rating")
        axes[0][0].set_yticks(range(len(items)), items)          # shared rows: set once, on the left panel
        axes[0][0].invert_yaxis()
        for ax in axes[0][1:]:
            ax.tick_params(labelleft=False)
        fig.legend(handles=[pf._handle(c, m, f, w, line=False) for _, c, m, f, w in raters],
                   loc="outside lower center", ncol=len(raters))
    return fig


# ---- the clip session's pair judgements --------------------------------------------------------------------------

def pair_agreement_table(pairs: dict, pooled: dict | None = None) -> pd.DataFrame:
    """One row per Stage A item from `review_clips.pair_agreement`: the pairs answered, each coder against the model,
    coder against coder (chance 0.5), the share of "close" ticks on the narrowest third of score gaps, and each
    coder's coherence with their own ratings. Given pooled figures that carry plan D's reporting, each row adds the
    reported comparison (`pair:<item>`): the coders' pooled agreement with the model, k of n, with its exact
    (Clopper-Pearson) 95% interval and two-sided p against 0.5, the same for coder against coder, and the close ticks;
    no target."""
    reported = reported_rows(pooled or {})
    rows = []
    for code, f in pairs.items():
        row = {"dimension": code, "pairs": f["n_pairs"], "videos": f["videos"]}
        for coder in f["coders"]:
            row[f"{coder} vs model"] = f["coder_model"][coder]
            ci = f["ci95"]["coder_model"].get(coder)
            row[f"{coder} vs model 95% CI"] = None if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"
        row["coder vs coder"] = f["coder_coder"]
        ci = f["ci95"]["coder_coder"]
        row["coder vs coder 95% CI"] = None if ci is None else f"{ci[0]:.2f}-{ci[1]:.2f}"
        row["close ticks in the lowest third"] = f["close_in_lowest_tercile_share"]
        for coder in f["coders"]:
            row[f"{coder} coherent with own ratings"] = f["coherence"].get(coder)
        if reported:
            row.update(_pair_figures(reported, f"pair:{code}"))
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_figures(reported: dict[str, dict[str, dict]], name: str) -> dict:
    """A comparison's reported figures: coders against the model (pooled) and coder against coder, each as k/n, its
    agreement, Clopper-Pearson interval and p against 0.5; the close ticks."""
    out = {}
    for which in ("coder_model", "human_human"):
        r = reported.get(which, {}).get(name)
        if r is None:
            continue
        counts = r.get("counts") or {}
        prefix = "" if which == "coder_model" else "coder vs coder: "
        if which == "coder_model":           # coder against coder's share is pair_agreement's own column
            out["coders vs model (pooled)"] = _value(r, "agreement")
        out[f"{prefix}k/n"] = f"{counts.get('agree')}/{counts.get('n')}" if counts.get("n") else None
        out[f"{prefix}95% interval (Clopper–Pearson)"] = _value(r, "binomial_interval")
        out[f"{prefix}p against 0.5"] = _value(r, "p_vs_half")
        if which == "coder_model":
            out["close ticks"] = counts.get("close")
    return out


def pairs_table(data: TrainingAgreement) -> pd.DataFrame:
    """The training session's comparisons (which of two pitches comes across as more ...), one row per Stage A item:
    `pair_agreement_table`'s columns, the coders' pooled agreement with the model, and its exact (Clopper-Pearson) 95%
    interval and two-sided p against 0.5, as `review_panel.py calibrate` reports them (no target: plan D). Empty
    without a session sheet (`clips__<coder>.csv`)."""
    if not data.pairs:
        return pd.DataFrame()
    table = pair_agreement_table(data.pairs, data.pooled)
    if "coders vs model (pooled)" not in table.columns:          # no reporting: calibrate's own figures
        rp = review_panel()
        pooled = data.pooled.get("pairs") or {}
        rows = {r["field"].partition(":")[2]: r for r in rp.calibration_rows({"pairs": pooled}, data.bar)}
        table["coders vs model (pooled)"] = [rows[d]["agree"] for d in table["dimension"]]
        table["95% interval (Clopper–Pearson)"] = [rows[d]["interval"] for d in table["dimension"]]
        table["p against 0.5"] = [rows[d]["p_vs_half"] for d in table["dimension"]]
    pooled_column = table.pop("coders vs model (pooled)")
    table.insert(3, "coders vs model (pooled)", pooled_column)
    return table


def figure_pairs(pairs, width=pf.DEFAULT_WIDTH):
    """One panel per Stage A item: agreement on which of two pitches comes across as more ..., in thirds of the
    model's score gap (narrowest
    first), each coder against the model and coder against coder; chance (0.5) dashed. `pairs` is
    `review_clips.pair_agreement`'s result or a TrainingAgreement (its `pairs`). None without pairs."""
    import matplotlib.pyplot as plt

    if isinstance(pairs, TrainingAgreement):
        pairs = pairs.pairs
    if not pairs:
        return None
    colours = [(pf.BLUE, "s"), (pf.VERMILLION, "^"), (pf.GREEN, "D"), (pf.PURPLE, "v")]
    with pf._context():
        fig, axes = plt.subplots(1, len(pairs), figsize=(pf.width_inches(width), 2.6), dpi=pf.SCREEN_DPI,
                                 layout="constrained", sharey=True, squeeze=False)
        handles = {}
        for i, (ax, (code, f)) in enumerate(zip(axes[0], pairs.items())):
            thirds = f["by_gap_tercile"]
            x = [t["tercile"] for t in thirds]
            series = [(f"{c} vs model", [t["coder_model"].get(c) for t in thirds], *colours[k % len(colours)], True)
                      for k, c in enumerate(f["coders"])]
            series.append(("coder vs coder", [t["coder_coder"] for t in thirds], pf.BLACK, "o", False))
            for label, ys, colour, marker, filled in series:
                points = [(a, b) for a, b in zip(x, ys) if b is not None]
                if not points:
                    continue
                ax.plot(*zip(*points), color=colour, linewidth=1, zorder=2)
                for a, b in points:
                    pf._point(ax, a, b, colour, marker, filled)
                handles[label] = pf._handle(colour, marker, filled, label)
            ax.axhline(0.5, color=pf.GREY, linewidth=0.8, linestyle="--", zorder=1)
            ax.set_xticks([1, 2, 3], ["narrowest", "middle", "widest"])
            ax.set_xlim(0.6, 3.4)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("third of the model's score gap")
            pf._letter(ax, i, code.replace("_", " "))
        axes[0][0].set_ylabel("agreement (share of pairs)")
        fig.legend(handles=list(handles.values()), loc="outside lower center", ncol=min(4, len(handles)))
    return fig
