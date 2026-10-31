"""Readable views of a panel run for the run-explorer notebook: the instrument, the records, agreement.

Everything here reads files a run already has and returns a table (or a figure, or text) ready to
show. Nothing is written except by `export_tables`, and only where it is told to. No provider is
called. The instrument is read from the selected video's own `bundle/` folder (the codebooks and
prompt template as that run used them), and only when a run kept no bundle from the live release;
`Instrument.source` says which.

People are shown by name with their id, e.g. "Mark Cuban (S1)"; times as M:SS; codes by the display
names of the codebooks.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from html import escape
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.panel_bottleneck import boot_p
from analysis.panel_dataset import (
    BOOKS_BUNDLE, BOOKS_LIVE, BUNDLE_DIR, NEXT_MOVES, NOT_VIDEO_DIRS, PANEL_MOVES, answer_followups, build_data,
    bundle_books, panel_next_move, substance_level, support_level,
)
from pipeline.errors import ConfigError
from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import seconds_to_clock
from pipeline.stage_b.panel_books import CONCEPT_CODE_OF, PanelBooks, load_panel_books, other_concepts, turn_frame

REPO = Path(__file__).resolve().parents[1]
PASS_FILES = {"primary": "primary.json", "replicate": "replicate.json", "reviewed": "reviewed.json"}
PROMPT_FILE = Path("prompts") / "panel_full.txt"
EXPLORER_DIR = Path("analysis") / "explorer"
COMPARISON_FILE = "comparison_vs_structured.json"
DASH = "—"
NO_EXCHANGE = "outside any exchange"
TURN_COLOURS = {'presents': '#c778ac', 'answer': '#e7a000', 'question': '#007bac',
                'accepts': '#00a58b', 'challenges': '#d95b24', 'negotiates': '#999999',
                'decides': '#000000', 'social': '#e5df47', 'hostile': '#ad3940', 'other': '#999999'}
KIND_COLOURS = {"substantive": "#3b6fb6", "social": "#e0a530", "demonstration": "#4aa564",
                "negotiation": "#b6453b", "presentation": "#8d6cab", NO_EXCHANGE: "#9a9a9a"}
KIND_LABELS = {"substantive": "Question and answer"}
VENTURE_CODES = {"primary_industry": "VC-INDUSTRY", "product_form": "VC-FORM", "customer_type": "VC-CUSTOMER",
                 "founder_gender": "VC-GENDER"}
FLAG_NAMES = {
    "unmapped_actor": "person in one pass only", "actor_mapped_by_order": "person matched by order, not name",
    "function_disagreement": "same turn, different function", "unmatched_turn": "turn in one pass only",
    "unmatched_exchange": "exchange in one pass only", "exchange_boundary_disagreement": "exchange drawn differently",
    "exchange_kind_disagreement": "exchange kind differs", "exchange_topic_disagreement": "exchange topic differs",
    "code_disagreement": "rating differs",
    "link_disagreement": "responds-to link differs", "unmatched_decision": "decision in one pass only",
    "ever_offer_disagreement": "whether an investor offered differs",
    "offer_membership_disagreement": "who is in an offer differs", "time_outlier": "times far apart",
    "time_drift": "times drift apart", "stage_a_disagreement": "Stage A rating differs",
    "venture_disagreement": "venture field differs",
}


# ---- small things both notebooks use --------------------------------------------------------

def resolve_run(run_name: str | Path, repo: Path = REPO) -> Path:
    """The folder of a run: `runs/<run_name>` of the repository, or the path itself when absolute."""
    path = Path(run_name)
    root = path if path.is_absolute() else Path(repo) / "runs" / path
    if not root.is_dir():
        runs = Path(repo) / "runs"
        available = sorted(p.name for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []
        raise ConfigError(f"run {str(run_name)!r} not found at {root}. Runs available: {', '.join(available) or 'none'}.")
    return root


def resolve_video_ids(run_root: Path | str, video_ids) -> list[str] | None:
    """The videos a notebook or the analysis script is limited to, as a list of ids (None = every video).

    `video_ids` is None, a list of ids, or the name of a selection file inside the run folder (or its
    full path), e.g. "selection_sample43.json". A selection file is JSON: {"videos": [{"video_id": ...},
    ...]}, {"video_ids": [...]} or a plain list of ids. The ids come back in the file's order.
    """
    if video_ids is None:
        return None
    if not isinstance(video_ids, (str, Path)):
        return [str(v) for v in video_ids]
    path = Path(video_ids)
    path = path if path.is_absolute() else Path(run_root) / path
    if not path.is_file():
        found = sorted(p.name for p in Path(run_root).glob("selection*.json"))
        raise ConfigError(f"VIDEO_IDS names the selection file {str(video_ids)!r}, which is not in {run_root}. "
                          f"Selection files there: {', '.join(found) or 'none'}. Give a file name, a list of video "
                          "ids, or None for every video.")
    content = _read(path)
    listed = content.get("videos", content.get("video_ids")) if isinstance(content, dict) else content
    ids = [v.get("video_id") if isinstance(v, dict) else v for v in listed] if isinstance(listed, list) else []
    if not ids or not all(isinstance(v, str) and v for v in ids):
        raise ConfigError(f'{path} is not a selection file: it must hold {{"videos": [{{"video_id": ...}}, ...]}}, '
                          '{"video_ids": [...]} or a list of video ids')
    return ids


def clock(seconds) -> str:
    """M:SS, or the dash when there is no time."""
    return DASH if seconds is None or seconds != seconds else seconds_to_clock(float(seconds))


def clock_range(start, end) -> str:
    return f"{clock(start)}–{clock(end)}"


def wide(table: pd.DataFrame, max_height: int | None = 520):
    """A table shown whole: long text wrapped rather than cut, left-aligned, scrolling when tall."""
    from IPython.display import HTML

    with pd.option_context("display.max_colwidth", None):
        body = table.to_html(index=False, escape=True, na_rep="", border=0)
    return HTML(_scroll(body, max_height))


def _scroll(body: str, max_height: int | None) -> str:
    style = "overflow:auto;" + (f"max-height:{int(max_height)}px;" if max_height else "")
    body = body.replace("<td>", '<td style="text-align:left;vertical-align:top;white-space:normal;">')
    body = body.replace("<th>", '<th style="text-align:left;vertical-align:bottom;">')
    return f'<div style="{style}">{body}</div>'


def grouped(table: pd.DataFrame, by: str, max_height: int | None = 640):
    """A table with one heading row per run of equal values in column `by`, which is then left out."""
    from IPython.display import HTML

    columns = [c for c in table.columns if c != by]
    rows = ["<table><thead><tr>" + "".join(f"<th>{escape(str(c))}</th>" for c in columns) + "</tr></thead><tbody>"]
    previous = object()
    for record in table.to_dict("records"):
        if record[by] != previous:
            previous = record[by]
            rows.append(f'<tr><td colspan="{len(columns)}" style="text-align:left;background:#eef1f5;color:#111;'
                        f'font-weight:bold;">{escape(str(previous))}</td></tr>')
        rows.append("<tr>" + "".join(f"<td>{escape('' if record[c] is None else str(record[c]))}</td>"
                                     for c in columns) + "</tr>")
    rows.append("</tbody></table>")
    return HTML(_scroll("".join(rows), max_height))


def show(table, empty: str, whole: bool = False) -> None:
    """Display a table, or print the one-line explanation `empty` when there is nothing to show.

    `whole=True` shows long text unshortened (see `wide`).
    """
    from IPython.display import display

    if table is None or (isinstance(table, pd.DataFrame) and table.empty):
        print(empty)
    else:
        display(wide(table) if whole else table)


def display_name(entry) -> str:
    """A code's display name from its codebook entry; its id in words when the book gives none."""
    return entry.name or entry.id.replace("_", " ")


def _read(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _names(record: dict) -> dict[str, str]:
    """actor id -> "Name (id)"."""
    return {a["actor_id"]: f"{a['label']} ({a['actor_id']})" if a.get("label") else a["actor_id"]
            for a in record["actors"]}


def _who(names: dict[str, str], actor) -> str:
    return names.get(actor, str(actor))


# ---- the run --------------------------------------------------------------------------------

def video_dirs(run_root: Path | str, video_ids: list[str] | None = None) -> list[Path]:
    """Every video folder of a run that holds at least one pass, in id order.

    `video_ids` keeps the view to those videos: no other folder of the run is looked at, so a run that
    is still being annotated can be read through the videos that are finished. A named video without
    any pass is an error.
    """
    run_root = Path(run_root)
    if video_ids is None:
        return sorted(d for d in run_root.iterdir() if d.is_dir() and d.name not in NOT_VIDEO_DIRS
                      and passes_present(d))
    dirs = [run_root / str(video_id) for video_id in sorted(set(video_ids))]
    missing = [d.name for d in dirs if not passes_present(d)]
    if missing:
        raise ConfigError(f"VIDEO_IDS names {', '.join(missing)}, which {'has' if len(missing) == 1 else 'have'} no "
                          f"annotation pass under {run_root}. Set VIDEO_IDS = None to see every annotated video.")
    return dirs


def pick_video(run_root: Path | str, video_id: str | None = None, video_ids: list[str] | None = None) -> Path:
    """The folder of `video_id`, or of the first video when None; among `video_ids` when given."""
    dirs = video_dirs(run_root, video_ids)
    if not dirs:
        raise ConfigError(f"no annotated video under {run_root}")
    if video_id is None:
        return dirs[0]
    for d in dirs:
        if d.name == video_id:
            return d
    raise ConfigError(f"video {video_id!r} is not in this run. Videos: {', '.join(d.name for d in dirs)}.")


def passes_present(video_dir: Path | str) -> list[str]:
    return [name for name, file in PASS_FILES.items() if (Path(video_dir) / "results" / file).exists()]


def load_pass(video_dir: Path | str, pass_name: str = "primary") -> tuple[dict, str, str]:
    """(result, the pass actually read, a one-line note). A pass the video does not have falls back to
    the primary pass, and the note says so."""
    video_dir = Path(video_dir)
    if pass_name not in PASS_FILES:
        raise ConfigError(f"PASS must be one of {', '.join(PASS_FILES)}, not {pass_name!r}")
    present = passes_present(video_dir)
    if not present:
        raise ConfigError(f"{video_dir.name}: no pass under {video_dir / 'results'}")
    used = pass_name if pass_name in present else ("primary" if "primary" in present else present[0])
    note = (f"Showing the {used} pass of {video_dir.name}." if used == pass_name else
            f"{video_dir.name} has no {pass_name} pass (it has: {', '.join(present)}); showing the {used} pass instead.")
    return _read(video_dir / "results" / PASS_FILES[used]), used, note


def _validation(video_dir: Path, pass_name: str) -> dict:
    return _read(video_dir / "results" / f"{pass_name}_validation.json") or {}


def _calls(video_dir: Path) -> list[dict]:
    path = video_dir / "calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analysis_pass(video_dir: Path | str) -> str:
    """The pass the analysis reads for a video (pipeline.stage_b.panel_pipeline.load_best_record): "reviewed"
    when a human-corrected record exists, else "primary", or "replicate" when the primary pass has
    validation errors and the replicate has fewer. The dash when the video has no primary pass."""
    from pipeline.stage_b.panel_pipeline import load_best_record

    try:
        return load_best_record(video_dir)[1]
    except (ConfigError, OSError, ValueError):
        return DASH


def run_overview(run_root: Path | str, input_usd_per_m: float = 2.0, output_usd_per_m: float = 12.0,
                 video_ids: list[str] | None = None) -> pd.DataFrame:
    """One row per video: what it is, which passes exist and which one the analysis reads, validation
    counts, review, calls, tokens, cost.

    The cost is tokens times the prices given (output and thinking tokens at the output price): an
    estimate from the call log, not an invoice. Validation counts are `errors / flags` per pass.
    """
    rows = []
    dirs = video_dirs(run_root, video_ids)
    in_run = [name for name in PASS_FILES if any(name in passes_present(d) for d in dirs)]
    for vdir in dirs:
        experiment = _read(vdir / "experiment.json") or {}
        present = passes_present(vdir)
        first = _read(vdir / "results" / PASS_FILES[present[0]]) or {}
        venture = (first.get("record") or {}).get("venture") or {}
        calls = _calls(vdir)
        tokens = {k: sum(int(c.get(k) or 0) for c in calls) for k in ("input_tokens", "output_tokens", "thinking_tokens")}
        cost = (tokens["input_tokens"] * input_usd_per_m
                + (tokens["output_tokens"] + tokens["thinking_tokens"]) * output_usd_per_m) / 1e6
        checks = {}
        for name in present:
            v = _validation(vdir, name)
            checks[name] = f"{len(v.get('errors') or [])} / {len(v.get('flags') or [])}" if v else "not checked"
        rows.append({
            "Video": vdir.name, "Venture": venture.get("product_category") or venture.get("description") or DASH,
            "Release": experiment.get("release_id") or DASH, "Length": clock(experiment.get("duration_s")),
            "Passes": ", ".join(present), "Analysis reads": analysis_pass(vdir),
            **{f"{name}: errors / flags": checks.get(name, DASH) for name in in_run},
            "Human-reviewed": "yes" if "reviewed" in present else "no",
            "Model calls": len(calls), "Failed calls": sum(1 for c in calls if not c.get("ok", True)),
            "Input tokens": tokens["input_tokens"], "Output tokens": tokens["output_tokens"],
            "Thinking tokens": tokens["thinking_tokens"], "Estimated cost (USD)": round(cost, 2)})
    return pd.DataFrame(rows)


def overview_summary(overview: pd.DataFrame) -> str:
    """One sentence under the overview table."""
    if overview.empty:
        return "No annotated video in this run."
    releases = sorted(set(overview["Release"]))
    text = (f"{len(overview)} videos; instrument release {', '.join(releases)}; "
            f"{int((overview['Human-reviewed'] == 'yes').sum())} human-reviewed; "
            f"{int(overview['Model calls'].sum())} model calls; estimated cost "
            f"${overview['Estimated cost (USD)'].sum():.2f} at the prices in the configuration cell.")
    if "Analysis reads" in overview.columns:
        reads = overview["Analysis reads"].value_counts()
        text += (" The analysis reads: " + ", ".join(f"the {name} pass of {int(n)}" for name, n in reads.items()
                                                      if name != DASH)
                 + " (the replicate only where the primary pass has validation errors and the replicate has fewer).")
    return text


# ---- the instrument as the model saw it -----------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    books: PanelBooks
    source: str                 # "bundle" (the run's own copy) | "live" (the current release)
    release_id: str | None
    prompt_text: str | None
    prompt_hash: str | None     # of the prompt file as it is on disk
    recorded_hash: str | None   # what experiment.json says was sent
    differences: pd.DataFrame   # bundle files against the live release


def instrument(video_dir: Path | str) -> Instrument:
    """The codebooks and prompt a video was annotated with, from its own folder."""
    video_dir = Path(video_dir)
    books, source = bundle_books(video_dir), BOOKS_BUNDLE
    if books is None:
        books, source = load_panel_books(load_release()), BOOKS_LIVE
    experiment = _read(video_dir / "experiment.json") or {}
    prompt_path = video_dir / PROMPT_FILE
    text = prompt_path.read_text() if prompt_path.exists() else None
    digest = None if text is None else hashlib.sha256(text.encode()).hexdigest()[:16]
    return Instrument(books=books, source=source, release_id=experiment.get("release_id"), prompt_text=text,
                      prompt_hash=digest, recorded_hash=experiment.get("prompt_hash"),
                      differences=bundle_differences(video_dir))


def bundle_differences(video_dir: Path | str) -> pd.DataFrame:
    """Each file of a video's bundle against the file of the same name in the live release."""
    bundle = Path(video_dir) / BUNDLE_DIR
    columns = ["File", "In the live release", "Same content"]
    if not bundle.is_dir():
        return pd.DataFrame(columns=columns)
    live = load_release()
    rows = []
    for path in sorted(p for p in bundle.iterdir() if p.is_file()):
        current = next((d / path.name for d in (live.canonical_dir, live.templates_dir) if (d / path.name).exists()), None)
        rows.append({"File": path.name, "In the live release": "yes" if current else "no",
                     "Same content": "yes" if current and current.read_bytes() == path.read_bytes() else "no"})
    return pd.DataFrame(rows, columns=columns)


def instrument_note(inst: Instrument) -> str:
    """In words: where the instrument was read from and whether it still is the live one."""
    live_id = load_release().release_id
    if inst.source == BOOKS_LIVE:
        return (f"This video kept no copy of its instrument, so the codebooks below are the LIVE release ({live_id}); "
                "they may differ from what the model saw.")
    changed = list(inst.differences.loc[inst.differences["Same content"] == "no", "File"])
    head = f"Read from this video's own bundle folder (release {inst.release_id or 'not recorded'}). "
    if not changed:
        return head + f"Every bundle file is identical to the live release ({live_id})."
    return head + (f"The live release ({live_id}) has since changed in {len(changed)} of {len(inst.differences)} files: "
                   f"{', '.join(changed)}. The tables below show what the model saw, not the live version.")


def prompt_note(inst: Instrument) -> str:
    if inst.prompt_text is None:
        return "This video kept no prompt file."
    check = ("matches the hash recorded for the run" if inst.prompt_hash == inst.recorded_hash else
             f"does NOT match the hash recorded for the run ({inst.recorded_hash})")
    return (f"Prompt: {len(inst.prompt_text):,} characters, about {len(inst.prompt_text.split()):,} words; "
            f"hash {inst.prompt_hash}, which {check}.")


def prompt_excerpt(inst: Instrument, full: bool = False, head: int = 2500, tail: int = 800) -> str:
    """The prompt text; unless `full`, its beginning and end with the middle left out and counted."""
    text = inst.prompt_text or ""
    if full or len(text) <= head + tail:
        return text
    return (text[:head] + f"\n\n[... {len(text) - head - tail:,} characters not shown; set SHOW_FULL_PROMPT = True "
            "to see all of it ...]\n\n" + text[-tail:])


def _allowed(entry) -> str:
    if entry.type == "categorical":
        return "one of: " + ", ".join(entry.values or [])
    if entry.type == "binary":
        return "true or false"
    if entry.anchors:
        return " · ".join(f"{anchor.value:g} = {anchor.description}" for anchor in entry.anchors)
    if entry.scale:
        return f"{entry.scale[0]:g} to {entry.scale[1]:g}"
    return entry.type


def codebook_table(book) -> pd.DataFrame:
    """A codebook as a table: one row per code, in the book's order."""
    return pd.DataFrame([{
        "Code": e.id, "Name": display_name(e), "Section": e.dimension,
        "Applies to": f"{e.unit.lower()}" + (f" ({e.applies_to})" if e.applies_to else ""),
        "Type": e.type, "Allowed values or scale": _allowed(e), "Definition": e.definition,
        "Use when": e.include_if, "Do not use when": e.exclude_if} for e in book.entries])


# ---- one video's record ---------------------------------------------------------------------

def actors_table(result: dict) -> pd.DataFrame:
    return pd.DataFrame([{"Id": a["actor_id"], "Role": a["role"], "Name": a.get("label") or DASH}
                         for a in result["record"]["actors"]], columns=["Id", "Role", "Name"])


def window_table(result: dict) -> pd.DataFrame:
    """The prepared presentation window, what was left out of the footage, and the record's notes."""
    record = result["record"]
    rows = [("Prepared presentation", "; ".join(clock_range(s["start_s"], s["end_s"])
                                                for s in record.get("prepared_spans") or []) or "none recorded")]
    for s in record.get("excluded_spans") or []:
        rows.append((f"Left out ({s.get('kind', 'unspecified')})", clock_range(s["start_s"], s["end_s"])))
    for g in record.get("gaps") or []:
        rows.append(("Gap in the footage", clock_range(g.get("start_s"), g.get("end_s"))))
    rows.append(("Footage complete", "yes" if record.get("source_complete") else "no"))
    if record.get("notes"):
        rows.append(("Annotator's notes", record["notes"]))
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _rank_scores(run_root: Path | None) -> dict:
    scores = _read(Path(run_root) / "ranking" / "scores.json") if run_root is not None else None
    return {code: block["pl_score"] for code, block in (scores or {}).items()
            if isinstance(block, dict) and isinstance(block.get("pl_score"), dict)}


def stage_a_table(video_dir: Path | str, books: PanelBooks, run_root: Path | str | None = None) -> pd.DataFrame:
    """Stage A ratings of the prepared presentation: every pass side by side, the rank score when the run
    has a ranking, and the model's note from the first pass that has one."""
    video_dir = Path(video_dir)
    passes = {name: (_read(video_dir / "results" / PASS_FILES[name]) or {}).get("record") or {}
              for name in passes_present(video_dir)}
    by_pass = {name: {i["code"]: i for i in record.get("stage_a") or []} for name, record in passes.items()}
    ranks = _rank_scores(run_root)
    rows = []
    for entry in books.stage_a.entries:
        row = {"Code": entry.id, "Name": display_name(entry)}
        for name, items in by_pass.items():
            item = items.get(entry.id)
            row[name] = DASH if item is None else ("uncodable" if item.get("value") is None else item["value"])
        if ranks:
            score = ranks.get(entry.id, {}).get(video_dir.name)
            row["Rank score"] = DASH if score is None else round(float(score), 2)
        row["Note"] = next((items[entry.id].get("note") for items in by_pass.values()
                            if entry.id in items and items[entry.id].get("note")), "")
        rows.append(row)
    return pd.DataFrame(rows)


def _records(video_dir: Path) -> dict[str, dict]:
    """{pass name: record} of every pass the video has, in PASS_FILES order."""
    return {name: (_read(video_dir / "results" / PASS_FILES[name]) or {}).get("record") or {}
            for name in passes_present(video_dir)}


def prepared_table(video_dir: Path | str, books: PanelBooks) -> pd.DataFrame:
    """The ratings of the prepared presentation's content (the PREP-PRES codes, PRES-SUBS and the others, 1 to 5),
    every pass side by side; "missing" with the reason the record gives when a pass has no value."""
    video_dir = Path(video_dir)
    by_pass = {name: {item["code"]: item for item in record.get("prepared") or []}
               for name, record in _records(video_dir).items()}
    rows = []
    for entry in books.prepared_codes():
        row = {"Code": entry.id, "Name": display_name(entry)}
        for name, items in by_pass.items():
            item = items.get(entry.id)
            if item is None:
                row[name] = DASH
            elif item.get("value") in (None, "", "NONE"):
                row[name] = f"missing ({item.get('missing') or 'no reason given'})"
            else:
                row[name] = item["value"]
        rows.append(row)
    return pd.DataFrame(rows)


def ask_table(video_dir: Path | str) -> pd.DataFrame:
    """The ask, per pass: the amount in US dollars, the equity offered for it, and whether the ask was stated."""
    video_dir = Path(video_dir)
    ventures = {name: record.get("venture") or {} for name, record in _records(video_dir).items()}
    fields = [("ask_amount_usd", "Amount asked (USD)"), ("ask_equity_pct", "Equity offered (%)"),
              ("ask_missing", "What is missing from the ask")]
    rows = []
    for field, label in fields:
        row = {"Field": label}
        for name, venture in ventures.items():
            value = venture.get(field)
            row[name] = DASH if value in (None, "") else (f"{value:,.0f}" if field == "ask_amount_usd" else value)
        rows.append(row)
    return pd.DataFrame(rows)


def venture_table(video_dir: Path | str, books: PanelBooks) -> pd.DataFrame:
    """Venture context per pass, the presenting team's gender composition included."""
    video_dir = Path(video_dir)
    ventures = {name: ((_read(video_dir / "results" / PASS_FILES[name]) or {}).get("record") or {}).get("venture") or {}
                for name in passes_present(video_dir)}
    fields = [("description", "What the venture is"), ("product_category", "Product category")]
    fields += [(field, display_name(books.venture.by_id(code))) for field, code in VENTURE_CODES.items()]
    fields += [("secondary_industries", "Secondary industries"), ("note", "Note")]
    rows = []
    for field, label in fields:
        row = {"Field": label}
        for name, venture in ventures.items():
            value = venture.get(field)
            row[name] = ", ".join(value) if isinstance(value, list) else (DASH if value in (None, "") else value)
        rows.append(row)
    return pd.DataFrame(rows)


def _exchange_spans(record: dict) -> dict[int, tuple[float, float, int]]:
    spans: dict[int, tuple[float, float, int]] = {}
    for t in record["turns"]:
        if t["exchange_id"] is not None:
            start, end, n = spans.get(t["exchange_id"], (t["start_s"], t["end_s"], 0))
            spans[t["exchange_id"]] = (min(start, t["start_s"]), max(end, t["end_s"]), n + 1)
    return spans


def exchange_table(result: dict) -> pd.DataFrame:
    """One row per exchange: kind, topic, label, clock range and number of turns."""
    record = result["record"]
    spans = _exchange_spans(record)
    rows = []
    for e in record["exchanges"]:
        start, end, n = spans.get(e["exchange_id"], (None, None, 0))
        rows.append({"Exchange": e["exchange_id"], "Kind": KIND_LABELS.get(e['kind'], e['kind']), "Topic": e["topic"], "Label": e["label"],
                     "Clock": clock_range(start, end) if n else DASH, "Turns": n})
    return pd.DataFrame(rows, columns=["Exchange", "Kind", "Topic", "Label", "Clock", "Turns"])


def turn_table(result: dict, books: PanelBooks, group_by_exchange: bool = True) -> pd.DataFrame:
    """Every turn as a readable row. With `group_by_exchange` the turns of an exchange stay together (the
    exchanges in the order they open; turns outside any exchange where they occur); otherwise by time."""
    record = result["record"]
    names = _names(record)
    exchange = {e["exchange_id"]: e for e in record["exchanges"]}
    first_pos: dict[int, int] = {}
    for pos, t in enumerate(record["turns"]):
        if t["exchange_id"] is not None:
            first_pos.setdefault(t["exchange_id"], pos)
    rows = []
    for pos, t in enumerate(record["turns"]):
        e = exchange.get(t["exchange_id"])
        heading = NO_EXCHANGE if e is None else (
            f"Exchange {e['exchange_id']} · {KIND_LABELS.get(e['kind'], e['kind'])}"
            + (f" · topic: {e['topic']}" if e["topic"] not in (None, "NONE") else "")
            + (f" · {e['label']}" if e.get("label") else ""))
        values: dict[str, str] = {}
        for c in t["codes"]:
            values.setdefault(c["code"], c["value"] if c["value"] is not None else f"missing ({c.get('missing')})")
        concept_code = CONCEPT_CODE_OF.get(t["function"])
        concept = values.get(concept_code) if concept_code else None
        if concept is not None and t.get("other_concept"):
            concept = f"{concept}: {t['other_concept']}"
        ratings = "; ".join(f"{display_name(books.panel.by_id(code))}: {value}" for code, value in values.items()
                            if code != concept_code)
        rows.append({
            "_key": (first_pos.get(t["exchange_id"], pos), pos) if group_by_exchange else (pos, pos),
            "Exchange": heading, "Turn": t["turn_id"], "Clock": clock_range(t["start_s"], t["end_s"]),
            "Speaker": _who(names, t["speaker"]), "Function": t["function"], "Concept": concept or "",
            "Frame (derived)": turn_frame(t["function"], [(c["code"], c["value"]) for c in t["codes"]]) or "",
            "Ratings": ratings, "Summary": t["summary"],
            "Responds to": ", ".join(f"turn {i}" for i in t["responds_to"]),
            "Addressed to": ", ".join(_who(names, a) for a in t["addressed"])})
    rows.sort(key=lambda r: r["_key"])
    return pd.DataFrame(rows).drop(columns="_key") if rows else pd.DataFrame()


def decisions_table(result: dict) -> pd.DataFrame:
    record = result["record"]
    names = _names(record)
    return pd.DataFrame([{
        "Decision": d["decision_id"], "Clock": clock(d["time_s"]), "Who": _who(names, d["actor"]), "Kind": d["kind"],
        "Offer": d["offer_id"] or "", "Terms": d.get("terms") or "", "Note": d.get("note") or "",
        "Stated in": f"turn {d['turn_id']}" if d["turn_id"] is not None else "no turn"}
        for d in record["decisions"]], columns=["Decision", "Clock", "Who", "Kind", "Offer", "Terms", "Note", "Stated in"])


def other_concepts_table(result: dict) -> pd.DataFrame:
    """Matters the model coded as `other`, with the name it gave each."""
    record = result["record"]
    summary = {t["turn_id"]: t["summary"] for t in record["turns"]}
    label = {e["exchange_id"]: e["label"] for e in record["exchanges"]}
    return pd.DataFrame([{
        "Clock": item["clock"] or DASH, "Where": f"{item['where']} {item['id']}", "Code": item["code"],
        "Name given": item["name"] or "(unnamed)",
        "Context": summary.get(item["id"], "") if item["where"] == "turn" else label.get(item["id"], "")}
        for item in other_concepts(record)], columns=["Clock", "Where", "Code", "Name given", "Context"])


# ---- what happens after an answer ------------------------------------------------------------

ASKER_MOVE_WORDS = {
    "positive_receipt": "asker accepts",
    "follow_up_question": "asker asks again", "challenge": "asker challenges",
    "other_by_asker": "asker says something else", "other_investor": "another investor speaks first",
    "no_reaction": "no investor reacts"}
PANEL_MOVE_WORDS = {
    "receipt": "receipt (an accepts turn)", "question": "a question",
    "challenge": "a challenge", "other": "something else", "none": "no reaction"}
assert tuple(ASKER_MOVE_WORDS) == NEXT_MOVES and tuple(PANEL_MOVE_WORDS) == PANEL_MOVES
FOLLOWUP_HEADERS = ["Video", "Asked by", "Question", "Severity", "Question summary", "Answered by", "Answer",
                    "Substance", "Checkable support", "Answer summary", "Asker's next move", "Panel's next move",
                    "Next investor turn", "Next turn summary"]


def followup_rows(followups: pd.DataFrame, summaries: dict[str, dict[int, str]],
                  names: dict[tuple[str, str], str]) -> pd.DataFrame:
    """Rows of `answer_followups` as readable turns: the question, the answer, and the investor turn that
    follows, each with its clock time and the annotator's summary.

    `summaries` maps video id -> turn id -> summary; `names` maps (video id, actor id) -> "Name (id)".
    """
    def turn(turn_id, start) -> str:
        return f"turn {int(turn_id)} at {clock(start)}"

    def who(video_id, actor) -> str:
        return names.get((video_id, actor), str(actor))

    rows = []
    for r, panel in zip(followups.itertuples(), panel_next_move(followups)):
        said = summaries.get(r.video_id, {})
        has_next = r.next_turn == r.next_turn
        rows.append({
            "Video": r.video_id, "Asked by": who(r.video_id, r.asker), "Question": turn(r.question_turn, r.question_start_s),
            "Severity": "" if r.question_sev != r.question_sev else int(r.question_sev),
            "Question summary": said.get(int(r.question_turn), ""), "Answered by": who(r.video_id, r.answerer),
            "Answer": turn(r.answer_turn, r.answer_start_s),
            "Substance": substance_level(r.subs).replace("_", " "),
            "Checkable support": support_level(r.evid).replace("_", " "),
            "Answer summary": said.get(int(r.answer_turn), ""),
            "Asker's next move": ASKER_MOVE_WORDS.get(r.next_move, r.next_move),
            "Panel's next move": PANEL_MOVE_WORDS.get(panel, panel),
            "Next investor turn": (f"{turn(r.next_turn, r.next_start_s)}: {who(r.video_id, r.next_actor)}, "
                                   f"{r.next_function}" if has_next else DASH),
            "Next turn summary": said.get(int(r.next_turn), "") if has_next else ""})
    return pd.DataFrame(rows, columns=FOLLOWUP_HEADERS)


def followups_table(video_dir: Path | str, result: dict, books: PanelBooks | None = None) -> pd.DataFrame:
    """For one video and pass: each answer to an investor question, with that question, the answer's
    substance and checkable support, and what the asker and the panel did next (read from turn order,
    whatever the exchange; see analysis.panel_dataset.answer_followups). Whole encounter, no cut."""
    video_id, record = Path(video_dir).name, result["record"]
    data = build_data([{"video_id": video_id, "record": record, "source": "primary", "books": books}])
    table = followup_rows(answer_followups(data), {video_id: {t["turn_id"]: t["summary"] for t in record["turns"]}},
                          {(video_id, actor): name for actor, name in _names(record).items()})
    return table.drop(columns="Video")


def validation_table(video_dir: Path | str, pass_name: str) -> pd.DataFrame:
    """The validation errors (rules a record must meet) and flags (things worth a look) of one pass."""
    report = _validation(Path(video_dir), pass_name)
    rows = []
    for level in ("errors", "flags"):
        for message in report.get(level) or []:
            rule = re.match(r"\[([^\]]+)\]\s*(.*)", message)
            rows.append({"Level": level[:-1], "Rule": rule.group(1) if rule else "", "Message": rule.group(2) if rule else message})
    return pd.DataFrame(rows, columns=["Level", "Rule", "Message"])


def timeline_figure(result: dict, title: str = ""):
    """Who speaks when: one row per person, each turn a bar coloured by the kind of its exchange, each
    decision a marker with its kind."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    record = result["record"]
    names = _names(record)
    kind_of = {e["exchange_id"]: e["kind"] for e in record["exchanges"]}
    actors = [a["actor_id"] for a in record["actors"]]
    row = {actor: i for i, actor in enumerate(reversed(actors))}
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(actors) + 1.6))
    used = set()
    for t in record["turns"]:
        if t["speaker"] not in row:
            continue
        kind = "presentation" if t["function"] == "presents" else kind_of.get(t["exchange_id"], NO_EXCHANGE)
        used.add(kind)
        ax.broken_barh([(t["start_s"], max(1.0, t["end_s"] - t["start_s"]))], (row[t["speaker"]] - 0.35, 0.7),
                       facecolors=KIND_COLOURS.get(kind, KIND_COLOURS[NO_EXCHANGE]))
    for d in record["decisions"]:
        if d["actor"] in row:
            ax.plot(d["time_s"], row[d["actor"]] + 0.42, marker="v", color="black", markersize=6)
            ax.annotate(d["kind"], (d["time_s"], row[d["actor"]] + 0.5), fontsize=7, rotation=30, ha="left", va="bottom")
    ax.set_yticks(list(row.values()))
    ax.set_yticklabels([names[a] for a in row])
    ax.set_ylim(-0.6, len(actors) + 0.4)
    ax.xaxis.set_major_locator(MultipleLocator(60))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda s, _: seconds_to_clock(max(0.0, s))))
    ax.set_xlabel("time in the video (M:SS)")
    handles = [Patch(color=KIND_COLOURS[k], label=KIND_LABELS.get(k, k)) for k in KIND_COLOURS if k in used]
    handles.append(plt.Line2D([], [], marker="v", color="black", linestyle="", label="decision"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=len(handles), frameon=False, fontsize=8)
    ax.set_title(title or "Turns by person over time")
    fig.tight_layout()
    return fig


def annotated_timeline_figure(result: dict, *, start_s: float, end_s: float, title: str = ""):
    """Expand a record window: turn functions, question codes, answer codes and saved response links.

    Display only: no inferred links, ratings or new annotations. Missing ratings are a dash;
    checkable support is marked only for an explicit true RS-EVID value. Actor rows include
    founders and investors even when they do not speak in the selected window.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, FancyArrowPatch
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if end_s <= start_s:
        raise ValueError('end_s must be later than start_s')
    record = result['record']
    actors = [a for a in record['actors'] if a['role'] in ('entrepreneur', 'founder', 'investor')]
    turns = [t for t in record['turns'] if t['end_s'] > start_s and t['start_s'] < end_s]
    if not turns:
        raise ValueError('No recorded turns in this window')
    rows = {a['actor_id']: i for i, a in enumerate(reversed(actors))}
    names = _names(record)
    colours = TURN_COLOURS
    severity = {'1': 'ask', '2': 'substantiate', '3': 'contest'}
    def words(value):
        return str(value).replace('_', ' ').capitalize() if value not in (None, 'NONE') else 'Unspecified'
    fig, ax = plt.subplots(figsize=(12, 4.5))
    spans = _exchange_spans(record)
    for i, exchange in enumerate(record['exchanges']):
        span = spans.get(exchange['exchange_id'])
        if not span or span[1] <= start_s or span[0] >= end_s:
            continue
        left, right = max(start_s, span[0]), min(end_s, span[1])
        ax.axvspan(left, right, color='#f2f2f2' if i % 2 == 0 else '#e7e7e7', zorder=0)
        topic = words(exchange.get('topic'))
        if topic == 'Operating efficiency':
            topic = 'Op. efficiency'
        ax.text((left + right) / 2, len(rows) - .12, topic, ha='center', va='bottom', fontsize=8, color='#555555')
    by_id = {t['turn_id']: t for t in turns if t['speaker'] in rows}
    used = set()
    for t in by_id.values():
        y = rows[t['speaker']]
        left, right = max(start_s, t['start_s']), min(end_s, t['end_s'])
        function = t['function']
        used.add(function)
        ax.barh(y, right - left, left=left, height=.62, color=colours.get(function, '#999999'), zorder=3)
        codes = {c['code']: c['value'] for c in t['codes'] if c.get('missing', 'none') == 'none'}
        if function == 'question':
            concept = words(codes.get('PR-CONCEPT'))
            if concept == 'Operating efficiency':
                concept = 'Op. efficiency'
            ax.text(left, y + .36, f"Q: {concept}, {severity.get(str(codes.get('PR-SEV')), 'unrated')}",
                    fontsize=7, va='bottom', zorder=4)
        elif function == 'answer':
            label = str(codes.get('PREP-SUBS') or DASH)
            if str(codes.get('RS-EVID')).lower() == 'true':
                label += ' ✓'
            ax.text((left + right) / 2, y, label, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)
            for target in t['responds_to']:
                question = by_id.get(target)
                if question is None or question['function'] != 'question':
                    continue
                arrow = FancyArrowPatch((left, y - .35),
                                        (min(end_s, question['end_s']), rows[question['speaker']] + .35),
                                        connectionstyle='arc3,rad=-.10', arrowstyle='->',
                                        mutation_scale=7, linewidth=.7, color='#888888', zorder=2)
                arrow.set_gid(f"answer-{t['turn_id']}-to-question-{target}")
                ax.add_patch(arrow)
        elif function == 'accepts':
            ax.text(left, y + .36, 'Accepts', va='bottom', fontsize=7)
    for decision in record['decisions']:
        if start_s <= decision['time_s'] < end_s and decision['actor'] in rows:
            ax.plot(decision['time_s'], rows[decision['actor']], marker='v', color='black', zorder=5)
    ax.set_yticks(list(rows.values()), labels=[names[a] for a in rows])
    ax.set_xlim(start_s, end_s)
    ax.set_ylim(-.6, len(rows) + .15)
    ax.xaxis.set_major_locator(MultipleLocator(15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: seconds_to_clock(value)))
    ax.set_xlabel('Time in the video (M:SS)')
    ax.set_title(title or 'Question and answer: the annotations', loc='left', pad=14)
    ax.spines[['top', 'right']].set_visible(False)
    handles = [Patch(color=colours.get(k, '#999999'), label=k.capitalize()) for k in colours if k in used]
    handles.append(Line2D([], [], color='#888888', linewidth=.7, label='Answer → question it responds to'))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.24), ncol=5, frameon=False, fontsize=8)
    fig.text(.5, .10, 'Inside each answer: substance (1 = no substance to 5 = in depth); ✓ = checkable support. '
             'Questions: ask, substantiate or contest.', ha='center', fontsize=8)
    fig.subplots_adjust(left=.20, right=.98, top=.87, bottom=.25)
    return fig


def record_timeline_figure(result: dict, *, start_s: float = 125, end_s: float = 232,
                           title: str = ''):
    """Figure 1 view: full episode by turn function above the annotated zoom window."""
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    fig = annotated_timeline_figure(result, start_s=start_s, end_s=end_s,
                                    title=f'Annotated close-up · {clock(start_s)}–{clock(end_s)}')
    fig.set_size_inches(13, 9)
    detail = fig.axes[0]
    detail.set_position([.11, .18, .86, .44])
    detail.get_legend().remove()
    record = result['record']
    actors = [a for a in record['actors'] if a['role'] in ('entrepreneur', 'founder', 'investor')]
    rows = {a['actor_id']: i for i, a in enumerate(reversed(actors))}
    names, counts = {}, {}
    founders = sum(a['role'] in ('entrepreneur', 'founder') for a in actors)
    for actor in actors:
        role = 'Investor' if actor['role'] == 'investor' else 'Founder'
        counts[role] = counts.get(role, 0) + 1
        names[actor['actor_id']] = (role if role == 'Founder' and founders == 1
                                   else f'{role} {counts[role]}')
    detail.set_yticks(list(rows.values()), labels=[names[a] for a in rows])
    overview = fig.add_axes([.11, .74, .86, .20])
    used = set()
    for turn in record['turns']:
        if turn['speaker'] not in rows:
            continue
        used.add(turn['function'])
        overview.barh(rows[turn['speaker']], turn['end_s'] - turn['start_s'],
                      left=turn['start_s'], height=.62,
                      color=TURN_COLOURS.get(turn['function'], TURN_COLOURS['other']))
    labels = {'exit': 'out', 'offer': 'offer', 'joins_offer': 'joins offer',
              'founder_counters': 'counters', 'founder_accepts': 'accepts offer'}
    last_time, levels = {}, {}
    for decision in sorted(record['decisions'], key=lambda d: d['time_s']):
        actor, time = decision['actor'], decision['time_s']
        if actor not in rows:
            continue
        level = levels.get(actor, 0) + 1 if time - last_time.get(actor, -100) < 25 else 0
        last_time[actor], levels[actor] = time, level
        y = rows[actor] + .43
        overview.plot(time, y, marker='v', color='black', markersize=6)
        overview.annotate(labels.get(decision['kind'], decision['kind'].replace('_', ' ')),
                          (time, y), xytext=(4, 4 + level * 13), textcoords='offset points',
                          fontsize=7, fontweight='bold', va='bottom')
    overview.add_patch(Rectangle((start_s, -.6), end_s - start_s, len(rows) + .25,
                                fill=False, linestyle='--', linewidth=1.2, edgecolor='#333333'))
    overview.set_yticks(list(rows.values()), labels=[names[a] for a in rows])
    overview.set_xlim(0, max(t['end_s'] for t in record['turns']) + 25)
    overview.set_ylim(-.6, len(rows) + 1)
    overview.xaxis.set_major_locator(MultipleLocator(60))
    overview.xaxis.set_major_formatter(FuncFormatter(lambda value, _: clock(value)))
    overview.spines[['top', 'right']].set_visible(False)
    overview.tick_params(labelsize=8)
    overview.set_title(title or 'The record of one episode as timeline and annotations',
                       loc='left', fontsize=12, fontweight='bold', pad=12)
    # One legend for both panels, including functions outside the zoomed window.
    handles = [Patch(color=colour, label=function.capitalize())
               for function, colour in TURN_COLOURS.items() if function in used]
    handles += [Line2D([], [], marker='v', color='black', linestyle='', label='Decision'),
                Line2D([], [], color='#888888', linewidth=.7, label='Answer → question it responds to')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .015),
               ncol=5, frameon=False, fontsize=8)
    for caption in fig.texts:
        caption.set_position((.54, .09))
    return fig


# ---- how often each value of each code is used ----------------------------------------------

#: The groups the value-frequency charts are drawn in: codes with the same kind of values side by side.
VALUE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("turn_function", "What each turn does", ("TURN-FUNC",)),
    ("concepts", "Business concepts (the same sixteen concepts plus other)", ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC")),
    ("categories", "Other categories", ("EX-KIND", "DEC-KIND", "RS-RESP")),
    ("yes_no", "Yes or no", ("RS-EVID", "RS-UNCERT")),
    ("scales", "Rating scales of questions and answers", ("PR-SEV", "PREP-SUBS", "PREP-COH", "PREP-CTX")),
    ("prepared", "Preparedness of the pitch", ("PRES-SUBS", "PRES-COH", "PRES-CTX", "PRES-EVID")),
    ("consistency", "Recorded claim conflicts", ("PREP-CONSIST",)),
    # the Stage A codes are read from the books when the counts are made (code_value_counts)
    ("stage_a", "Stage A ratings of the prepared presentation (1 to 5)", ()),
    ("venture", "Venture and team", ("VC-INDUSTRY", "VC-FORM", "VC-CUSTOMER", "VC-GENDER")),
)
#: What one counted unit is, per code (the denominator of its shares).
VALUE_UNITS = {
    "TURN-FUNC": "turns", "EX-KIND": "exchanges", "EX-TOPIC": "substantive exchanges", "DEC-KIND": "decisions",
    "PR-CONCEPT": "question turns", "PR-SEV": "question turns", **{c: "answer turns" for c in (
        "RS-RESP", "RS-CONCEPT", "RS-EVID", "RS-UNCERT", "PREP-SUBS", "PREP-COH", "PREP-CTX")},
    "VC-INDUSTRY": "pitches", "VC-FORM": "pitches", "VC-CUSTOMER": "pitches", "VC-GENDER": "pitches"}
_VENTURE_FIELDS = {"VC-INDUSTRY": "primary_industry", "VC-FORM": "product_form", "VC-CUSTOMER": "customer_type",
                   "VC-GENDER": "founder_gender"}
_MISSING_WORDS = {"not_applicable", "no_opportunity", "not_observable", "edited_gap", "ambiguous", "too_short",
                  "missing"}
VALUE_COUNT_COLUMNS = ["group", "code", "name", "unit", "value", "kind", "order", "count", "n", "share"]
DOMINANT_SHARE = 0.8        # a value used for at least this share of a code's coded units is flagged as dominant
RARE_SHARE = 0.01           # a value used for less than this share (but not never) is flagged as rare


def _allowed_values(entry) -> list[str]:
    """The values a code may take, in the codebook's order; for a scale every whole step from low to high."""
    if entry.type == "categorical":
        return list(entry.values or [])
    if entry.type == "binary":
        return ["true", "false"]
    if entry.type == "ordinal" and entry.scale:
        low, high = entry.scale
        return [f"{v:g}" for v in np.arange(low, high + 1)]
    return []


def _observations(record: dict, books: PanelBooks) -> dict[str, list[tuple[str | None, str | None]]]:
    """Every observation of every counted code in one record: code -> [(value, missing reason)]."""
    from pipeline.stage_b.panel_books import canonical_value

    entries = {e.id: e for book in (books.panel, books.stage_a, books.venture) for e in book.entries}
    seen: dict[str, list] = {}

    def add(code, value, missing=None):
        entry = entries.get(code)
        if entry is None:
            return
        value = None if value is None else canonical_value(entry, str(value))
        if value is not None and entry.type == "ordinal":
            try:
                value = f"{float(value):g}"
            except ValueError:
                pass
        seen.setdefault(code, []).append((value, missing))

    for turn in record.get("turns") or []:
        add("TURN-FUNC", turn.get("function"))
        for c in turn.get("codes") or []:
            missing = c.get("missing")
            missing = None if missing in (None, "none") else missing
            value = c.get("value")
            if not missing and isinstance(value, str) and value.strip().casefold() in _MISSING_WORDS:
                missing, value = value.strip().casefold(), None       # a missing reason written as the value
            add(c.get("code"), None if missing else value, missing)
    for exchange in record.get("exchanges") or []:
        add("EX-KIND", exchange.get("kind"))
        if exchange.get("kind") == "substantive":
            add("EX-TOPIC", exchange.get("topic"))
    for decision in record.get("decisions") or []:
        add("DEC-KIND", decision.get("kind"))
    for item in record.get("stage_a") or []:
        uncodable = bool(item.get("uncodable"))
        add(item.get("code"), None if uncodable else item.get("value"), "uncodable" if uncodable else None)
    for item in record.get("prepared") or []:
        missing = item.get("missing")
        missing = None if missing in (None, "none") else missing
        add(item.get("code"), None if missing else item.get("value"), missing)
    if "claim_conflicts" in record:
        # The codebook defines this count as the length of the recorded pair list;
        # reconciliation is a separate attribute of each pair.
        add("PREP-CONSIST", len(record["claim_conflicts"]))
    venture = record.get("venture") or {}
    for code, field in _VENTURE_FIELDS.items():
        if field in venture:
            add(code, venture.get(field))
    return seen


def code_value_counts(run_root: Path | str, books: PanelBooks, video_ids: list[str] | None = None,
                      pass_name: str = "analysis") -> pd.DataFrame:
    """How often each value of each code is used, over the videos of a run: one row per code and value.

    Every value the codebook allows is listed, with a count of 0 when it was never used, so values that
    were never applied are visible. Missing reasons (a code left without a value: too short, not
    applicable, ...) and values outside the codebook are listed after the allowed values, with `kind`
    "missing" or "not in codebook". `share` is the count over `n`, the units on which the code was
    recorded (value or missing reason). `pass_name` is "analysis" (the pass the analysis reads for each
    video, see analysis_pass), "primary", "replicate" or "reviewed".
    """
    entries = {e.id: e for book in (books.panel, books.stage_a, books.venture) for e in book.entries}
    totals: dict[str, dict[tuple[str, str], int]] = {}
    for video_dir in video_dirs(run_root, video_ids):
        used = analysis_pass(video_dir) if pass_name == "analysis" else pass_name
        if used not in PASS_FILES or not (video_dir / "results" / PASS_FILES[used]).exists():
            continue
        result = _read(video_dir / "results" / PASS_FILES[used])
        for code, observations in _observations(result.get("record") or {}, books).items():
            bucket = totals.setdefault(code, {})
            for value, missing in observations:
                key = ("missing", missing) if missing else ("value", value if value is not None else "(blank)")
                bucket[key] = bucket.get(key, 0) + 1
    rows = []
    for group, _, codes in VALUE_GROUPS:
        for code in (books.stage_a_ids() if group == "stage_a" else codes):
            entry = entries.get(code)
            if entry is None:
                continue
            bucket = totals.get(code, {})
            n = sum(bucket.values())
            allowed = _allowed_values(entry)
            if entry.type == "count":
                observed = [int(v) for k, v in bucket if k == "value" and str(v).isdigit()]
                allowed = [str(v) for v in range(max(observed, default=0) + 1)]
            values = [("value", v) for v in allowed]
            extra = sorted(k for k in bucket if k not in values)
            for order, (kind, value) in enumerate(values + extra):
                count = bucket.get((kind, value), 0)
                label = kind if kind == "missing" else ("value" if value in allowed else "not in codebook")
                rows.append({"group": group, "code": code, "name": display_name(entry),
                             "unit": VALUE_UNITS.get(code, "pitches"), "value": value, "kind": label, "order": order,
                             "count": count, "n": n, "share": count / n if n else np.nan})
    return pd.DataFrame(rows, columns=VALUE_COUNT_COLUMNS)


def value_usage_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Per code: how many units, the most common value and its share, the values never used, the rare
    values (used, but for under 1% of units) and the share left missing. Codes with a dominant value
    (at least 80%) or with unused values are the ones to look at."""
    rows = []
    for (group, code), part in counts.groupby(["group", "code"], sort=False):
        values = part[part.kind == "value"]
        n = int(part.n.iloc[0]) if len(part) else 0
        top = values.loc[values["count"].idxmax()] if len(values) and values["count"].max() > 0 else None
        # NONE ("no value applies") is a legitimate code but not a substantive one: leaving it unused is no
        # sign of a dead category, so it is not listed as never used (the charts still show it)
        never = values[(values["count"] == 0) & (values.value != "NONE")].value.tolist()
        rare = values[(values["count"] > 0) & (values.share < RARE_SHARE)].value.tolist()
        missing = part[part.kind == "missing"]["count"].sum()
        rows.append({
            "Code": code, "Name": part.name.iloc[0], "Counted units": f"{n} {part.unit.iloc[0]}",
            "Most common value": None if top is None else top.value,
            "Its share": None if top is None else _round(float(top.share)),
            "Dominant (80% or more)": bool(top is not None and top.share >= DOMINANT_SHARE),
            "Values never used": ", ".join(never) or "", "Rare values (under 1%)": ", ".join(rare) or "",
            "Left missing (share)": _round(missing / n) if n else None,
            "Values outside the codebook": ", ".join(part[part.kind == "not in codebook"].value) or ""})
    return pd.DataFrame(rows)


def value_frequency_figure(counts: pd.DataFrame, group: str, title: str | None = None, ncols: int | None = None):
    """Small multiples for one group of codes: per code, a horizontal bar per value with its share of the
    code's units and its count. Unused values are labelled "never used"; a dominant value (80% or more)
    is drawn dark blue; a rare value (used, but for under 1% of units) is orange; missing reasons are
    hatched grey; values outside the codebook are red. None when the group has no counted unit."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    part = counts[counts.group == group]
    codes = [c for c in dict.fromkeys(part.code) if part[part.code == c].n.iloc[0] > 0]
    if not codes:
        return None
    ncols = ncols or (1 if group == "turn_function" else min(len(codes), 3 if group != "stage_a" else 5))
    nrows = -(-len(codes) // ncols)
    longest = max(len(part[part.code == c]) for c in codes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols + (2.2 if group in ("concepts", "turn_function") else 0),
                                                    0.32 * longest * nrows + 1.2 * nrows + 0.6), squeeze=False)
    blue, dark, grey, red, orange = "#9ecae1", "#08519c", "#bdbdbd", "#d62728", "#fd8d3c"
    for ax, code in zip(axes.flat, codes):
        rows = part[part.code == code].sort_values("order")
        shares = (rows.share.fillna(0) * 100).to_numpy()
        colours, hatches = [], []
        for r in rows.itertuples():
            if r.kind == "missing":
                colours.append(grey); hatches.append("///")
            elif r.kind == "not in codebook":
                colours.append(red); hatches.append("")
            else:
                colours.append(dark if r.share >= DOMINANT_SHARE else orange if 0 < r.share < RARE_SHARE else blue)
                hatches.append("")
        labels = [("missing: " + v) if k == "missing" else v for v, k in zip(rows.value, rows.kind)]
        y = np.arange(len(rows))
        bars = ax.barh(y, shares, color=colours, edgecolor="white")
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax.set_yticks(y, labels, fontsize=9)
        ax.invert_yaxis()
        top = max(shares.max(), 1)
        ax.set_xlim(0, min(top * 1.35, 125))
        for yi, (share, count, kind) in enumerate(zip(shares, rows["count"], rows.kind)):
            if count == 0 and kind == "value":
                ax.text(0.5, yi, "never used", va="center", fontsize=8, color=red, style="italic")
            else:
                ax.text(share + top * 0.02, yi, f"{share:.1f}%  ({count})", va="center", fontsize=8)
        n = int(rows.n.iloc[0])
        ax.set_title(f"{rows.name.iloc[0]} ({code})\nn = {n} {rows.unit.iloc[0]}", fontsize=10, loc="left")
        ax.set_xlabel("share of units (%)", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    for ax in list(axes.flat)[len(codes):]:
        ax.set_visible(False)
    handles = [Patch(color=blue, label="value"), Patch(color=dark, label="dominant value (80% or more)"),
               Patch(color=orange, label="rare value (under 1%)"),
               Patch(facecolor=grey, hatch="///", label="left missing, with its reason")]
    if (part.kind == "not in codebook").any():
        handles.append(Patch(color=red, label="value outside the codebook"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97 if title else 1))
    return fig


# ---- pass-to-pass agreement -----------------------------------------------------------------

def agreement_reports(run_root: Path | str, video_ids: list[str] | None = None) -> dict[str, dict]:
    """video id -> its agreement.json (the two passes compared), for the videos that have one."""
    out = {}
    for vdir in video_dirs(run_root, video_ids):
        report = _read(vdir / "agreement.json")
        if isinstance(report, dict) and "turns" in report:
            out[vdir.name] = report
    return out


PART_OUTCOMES, PART_VENTURE, PART_PREPARED = "Outcomes and decisions", "Venture context", "Prepared presentation"
PART_TURNS, PART_CODES, PART_EXCHANGES, PART_STAGE_A = "Turns", "Turn codes", "Exchanges", "Stage A ratings"
AGREEMENT_PARTS = (PART_OUTCOMES, PART_VENTURE, PART_PREPARED, PART_TURNS, PART_CODES, PART_EXCHANGES, PART_STAGE_A)
AGREEMENT_COLUMNS = ["Part", "Field", "Compared", "Agree", "Share agreeing", "Within one (count)", "Within one point",
                     "What is counted"]


def _agreement_counts(report: dict, books: PanelBooks | None) -> list[tuple]:
    """(part, field, n compared, n agreeing, n within one or None, what n counts) of one report."""
    rows: list[tuple] = []
    turns, exchanges, decisions = report.get("turns") or {}, report.get("exchanges") or {}, report.get("decisions") or {}
    if isinstance(decisions.get("matched"), list):
        both = decisions.get("any_offer") or {}
        if both.get("primary") is not None and both.get("replicate") is not None:
            rows.append((PART_OUTCOMES, "Whether the pitch got any offer", 1, int(both["primary"] == both["replicate"]),
                         None, "pitches"))
        ever = decisions.get("ever_offer") or {}
        rows.append((PART_OUTCOMES, "Whether each investor ever offered", len(ever),
                     sum(1 for v in ever.values() if v.get("primary") == v.get("replicate")), None, "investors"))
        matched, alone = len(decisions["matched"]), len(decisions.get("unmatched_primary") or [])
        rows.append((PART_OUTCOMES, "Decisions found by both passes", matched + alone, matched, None,
                     "decisions of the primary pass"))
        rows.append((PART_OUTCOMES, "Decisions of the replicate pass also found in the primary",
                     matched + len(decisions.get("unmatched_replicate") or []), matched, None,
                     "decisions of the replicate pass"))
        offers = decisions.get("offers") or []
        rows.append((PART_OUTCOMES, "Who is part of each offer", len(offers), sum(1 for o in offers if o.get("agree")),
                     None, "offers"))
    venture = {k: v for k, v in (report.get("venture") or {}).items() if isinstance(v, dict) and "agree" in v}
    if venture:
        rows.append((PART_VENTURE, "Venture context (industry, form, customer, team gender)", len(venture),
                     sum(1 for v in venture.values() if v["agree"]), None, "fields"))
        for field, v in venture.items():
            name = field
            if books is not None and field in VENTURE_CODES:
                name = f"{display_name(books.venture.by_id(VENTURE_CODES[field]))} ({field})"
            rows.append((PART_VENTURE, name, 1, int(bool(v["agree"])), None, "pitches"))
    prepared = report.get("prepared") or {}
    if all(isinstance(prepared.get(k), (int, float)) for k in ("primary_s", "replicate_s", "iou")):
        # seconds in both windows and in either, recovered from the two lengths and their overlap ratio
        total, iou = float(prepared["primary_s"]) + float(prepared["replicate_s"]), float(prepared["iou"])
        shared = iou * total / (1 + iou)
        rows.append((PART_PREPARED, "Prepared presentation window (seconds in both passes' windows)", total - shared,
                     shared, None, "seconds in either pass's window"))
    if isinstance(turns.get("matched"), list):
        same, differ = len(turns["matched"]), len(turns.get("function_mismatched") or [])
        rows.append((PART_TURNS, "Turns found by both passes with the same function", turns.get("n_primary", 0), same,
                     None, "turns of the primary pass"))
        rows.append((PART_TURNS, "Turn function, among turns paired across passes", same + differ, same, None,
                     "paired turns"))
    for code, block in (report.get("codes") or {}).items():
        if not isinstance(block, dict) or "both_valued" not in block:
            continue
        name = code
        if books is not None:
            base = code.split("(")[0].replace("-FRAME", "-CONCEPT")
            try:
                entry_name = display_name(books.panel.by_id(base))
                name = (f"{entry_name.replace('concept', 'frame')} (derived from the concept)" if "derived" in code
                        else f"{entry_name} ({code})")
            except KeyError:
                pass
        within = block.get("within_one")
        rows.append((PART_CODES, name, block["both_valued"], block["exact"], within,
                     "paired turns with a value in both passes"))
    if isinstance(exchanges.get("matched"), list):
        pairs = len(exchanges["matched"])
        rows.append((PART_EXCHANGES, "Exchanges paired with at least half their turns in common",
                     exchanges.get("n_primary", 0), exchanges.get("matched_at_half", 0), None,
                     "exchanges of the primary pass"))
        rows.append((PART_EXCHANGES, "Exchange kind, among paired exchanges", pairs, exchanges.get("kind_agree", 0), None,
                     "paired exchanges"))
        rows.append((PART_EXCHANGES, "Exchange topic, among paired substantive exchanges", exchanges.get("topic_n", 0),
                     exchanges.get("topic_agree", 0), None, "paired substantive exchanges"))
    stage_a = report.get("stage_a") or {}
    summary = stage_a.get("_summary") or {}
    if summary:
        rows.append((PART_STAGE_A, "Stage A ratings of the prepared presentation", summary.get("both_rated", 0),
                     summary.get("exact", 0), summary.get("within_one"), "items rated in both passes"))
    for code, item in stage_a.items():
        if code.startswith("_") or not isinstance(item, dict):
            continue
        rated = item.get("primary") is not None and item.get("replicate") is not None
        name = f"Stage A: {code}"
        if books is not None:
            try:
                name = f"Stage A: {display_name(books.stage_a.by_id(code))} ({code})"
            except KeyError:
                pass
        apart = abs(item["primary"] - item["replicate"]) if rated else None
        rows.append((PART_STAGE_A, name, int(rated), int(rated and apart == 0), int(rated and apart <= 1),
                     "pitches rated on the item in both passes"))
    return rows


def agreement_table(reports: dict[str, dict] | dict, books: PanelBooks | None = None) -> pd.DataFrame:
    """The agreement of the two passes, field by field, pooled over the reports given.

    `reports` maps video ids to agreement reports (one entry for one video). Counts are summed over
    videos before the share is taken, so a long video weighs more than a short one. `Within one` is
    given for scale ratings only: the number, and the share, whose two values are at most one scale
    point apart. Rows are grouped by `Part`: outcomes and decisions, venture context, the prepared
    window, turns, turn codes, exchanges, Stage A (all ten items together, then each item).
    """
    totals: dict[str, list] = {}
    for report in reports.values():
        for part, field, n, agree, within, unit in _agreement_counts(report, books):
            entry = totals.setdefault(field, [0, 0, None, unit, part])
            entry[0] += n or 0
            entry[1] += agree or 0
            if within is not None:
                entry[2] = (entry[2] or 0) + int(within)
    rows = [{"Part": part, "Field": field, "Compared": int(round(n)), "Agree": int(round(agree)),
             "Share agreeing": round(agree / n, 2) if n else None, "Within one (count)": within,
             "Within one point": round(within / n, 2) if within is not None and n else None, "What is counted": unit}
            for field, (n, agree, within, unit, part) in totals.items()]
    rows.sort(key=lambda r: AGREEMENT_PARTS.index(r["Part"]))
    table = pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
    table["Within one (count)"] = table["Within one (count)"].astype("Int64")
    return table


def function_agreement_table(reports: dict[str, dict]) -> pd.DataFrame:
    """Per turn function, pooled over the reports: how many turns each pass gave that function, how many
    of them were paired with a turn of the same function in the other pass, and the agreement
    2 x matched / (primary + replicate), which is 1 when both passes found the same turns with that
    function and 0 when none was paired. Most frequent function first; the last row is all functions."""
    counts: dict[str, list[int]] = {}
    for report in reports.values():
        for function, block in ((report.get("turns") or {}).get("by_function") or {}).items():
            entry = counts.setdefault(function, [0, 0, 0])
            for i, key in enumerate(("primary", "replicate", "matched")):
                entry[i] += int(block.get(key) or 0)
    columns = ["Function", "Primary turns", "Replicate turns", "Matched with the same function", "Agreement"]
    if not counts:
        return pd.DataFrame(columns=columns)
    ordered = sorted(counts.items(), key=lambda kv: (-(kv[1][0] + kv[1][1]), kv[0]))
    ordered.append(("all functions", [sum(v[i] for v in counts.values()) for i in range(3)]))
    return pd.DataFrame([{"Function": function, "Primary turns": p, "Replicate turns": r,
                          "Matched with the same function": m, "Agreement": round(2 * m / (p + r), 2) if p + r else None}
                         for function, (p, r, m) in ordered], columns=columns)


def agreement_extras(reports: dict[str, dict]) -> pd.DataFrame:
    """Agreement figures that are averages rather than counts, per video."""
    rows = []
    for video_id, r in reports.items():
        turns, exchanges, time = r.get("turns") or {}, r.get("exchanges") or {}, r.get("time") or {}
        rows.append({"Video": video_id, "Turns (primary / replicate)": f"{turns.get('n_primary')} / {turns.get('n_replicate')}",
                     "Exchanges (primary / replicate)": f"{exchanges.get('n_primary')} / {exchanges.get('n_replicate')}",
                     "Who speaks, second by second": _round(turns.get("speaker_time_agreement")),
                     "Order of turns": _round(turns.get("sequence_agreement")),
                     "Mean overlap of paired exchanges": _round(exchanges.get("mean_jaccard")),
                     "Prepared window overlap": _round((r.get("prepared") or {}).get("iou")),
                     "Median start difference (s)": time.get("median_abs_start_diff_s")})
    return pd.DataFrame(rows)


def _round(value, digits: int = 2):
    return None if value is None else round(float(value), digits)


def confusion_table(reports: dict[str, dict], what: str) -> pd.DataFrame:
    """Primary value (rows) by replicate value (columns), counts pooled over the reports.

    `what` is a code id of the agreement report's codes block (e.g. "PR-CONCEPT", "RS-CONCEPT",
    "PR-SEV"), or "topic", "kind" (exchanges) or "function" (turns paired with different
    functions only). Empty when no report has it.
    """
    counts: dict[tuple[str, str], int] = {}
    for report in reports.values():
        exchanges = report.get("exchanges") or {}
        block = {"topic": exchanges.get("topic_confusion"), "kind": exchanges.get("kind_confusion"),
                 "function": (report.get("turns") or {}).get("function_confusion")}.get(what)
        if block is None:
            block = ((report.get("codes") or {}).get(what) or {}).get("confusion") if what in (report.get("codes") or {}) else None
        for key, n in (block or {}).items():
            a, _, b = key.partition("|")
            counts[(a, b)] = counts.get((a, b), 0) + int(n)
    if not counts:
        return pd.DataFrame()
    labels = sorted({a for a, _ in counts} | {b for _, b in counts})
    table = pd.DataFrame(0, index=pd.Index(labels, name="primary ↓ / replicate →"), columns=labels)
    for (a, b), n in counts.items():
        table.loc[a, b] = n
    return table


_PAIR = re.compile(r"^(turns|exchanges) (\d+)\|(\d+)")
_TURN_FLAGS = ("function_disagreement", "unmatched_turn", "link_disagreement")
_EXCHANGE_FLAGS = ("unmatched_exchange", "exchange_boundary_disagreement", "exchange_kind_disagreement",
                   "exchange_topic_disagreement")


def disagreements(report: dict, primary: dict | None, replicate: dict | None, kinds: list[str] | None = None) -> pd.DataFrame:
    """Each specific disagreement the comparison flagged, with the clock time in each pass.

    `primary` and `replicate` are the two results (for the times). `kinds` keeps only those flag
    kinds (the raw names of the agreement report, e.g. "code_disagreement"). Sorted by time.
    """
    def starts(result) -> tuple[dict, dict, dict]:
        record = (result or {}).get("record") or {"turns": [], "decisions": []}
        turn = {t["turn_id"]: t["start_s"] for t in record["turns"]}
        exchange = {k: v[0] for k, v in _exchange_spans(record).items()}
        return turn, exchange, {d["decision_id"]: d["time_s"] for d in record["decisions"]}

    times = {"primary": starts(primary), "replicate": starts(replicate)}
    rows = []
    for flag in report.get("flags") or []:
        kind, side, ref, detail = flag["kind"], flag.get("pass"), flag.get("ref"), flag.get("detail") or ""
        if kinds is not None and kind not in kinds:
            continue
        at = {"primary": None, "replicate": None}
        pair = _PAIR.match(detail)
        if pair:
            index = 0 if pair.group(1) == "turns" else 1
            at["primary"] = times["primary"][index].get(int(pair.group(2)))
            at["replicate"] = times["replicate"][index].get(int(pair.group(3)))
        elif isinstance(ref, int):
            index = (0 if kind in _TURN_FLAGS or (kind == "time_outlier" and "turn" in detail)
                     else 1 if kind in _EXCHANGE_FLAGS else 2 if "decision" in kind or "decision" in detail else None)
            name = "replicate" if side == "replicate" else "primary"      # a reference of both passes is the primary's
            if index is not None:
                at[name] = times[name][index].get(ref)
        first = next((v for v in (at["primary"], at["replicate"]) if v is not None), None)
        rows.append({"_t": float("inf") if first is None else first, "What": FLAG_NAMES.get(kind, kind),
                     "Primary clock": clock(at["primary"]), "Replicate clock": clock(at["replicate"]),
                     "About": "" if ref is None else str(ref), "Detail": detail, "Kind": kind})
    rows.sort(key=lambda r: r["_t"])
    return pd.DataFrame(rows, columns=["_t", "What", "Primary clock", "Replicate clock", "About", "Detail", "Kind"]).drop(columns="_t")


def video_disagreements(video_dir: Path | str, kinds: list[str] | None = None) -> pd.DataFrame:
    """`disagreements` for a video folder: empty when the video has no comparison of two passes."""
    video_dir = Path(video_dir)
    report = _read(video_dir / "agreement.json")
    if not isinstance(report, dict):
        return disagreements({}, None, None)
    return disagreements(report, _read(video_dir / "results" / PASS_FILES["primary"]),
                         _read(video_dir / "results" / PASS_FILES["replicate"]), kinds)


# ---- human review ---------------------------------------------------------------------------

def _review_script():
    path = REPO / "scripts" / "review_panel.py"
    spec = importlib.util.spec_from_file_location("review_panel_for_explorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review_status(run_root: Path | str, video_ids: list[str] | None = None) -> pd.DataFrame:
    """Per video: whether a review sheet was exported, how many rows a coder reviewed, whether a reviewed
    record exists and how many corrections it applied, and the human-against-model headline."""
    script = _review_script()
    rows = []
    for vdir in video_dirs(run_root, video_ids):
        review = vdir / script.REVIEW_DIR
        entry = script._manifest_entry(review, vdir.name)
        reviewed = _read(vdir / "results" / PASS_FILES["reviewed"])
        agreement = _read(review / script.AGREEMENT)
        rows.append({"Video": vdir.name,
                     "Review sheet": f"exported ({len(entry['rows'])} rows)" if entry else "not exported",
                     "Rows reviewed": len(script._reviewed_lines(script._log_lines(review / script.LOG))),
                     "Reviewed record": "yes" if reviewed else "no",
                     "Corrections applied": (reviewed or {}).get("corrections", 0) if reviewed else 0,
                     "Human against model": script._headline(agreement) if agreement else "not yet imported"})
    return pd.DataFrame(rows)


def review_summary(run_root: Path | str, video_ids: list[str] | None = None) -> str | None:
    """The pooled human-review summary (the text `review_panel.py summary` prints); None when no sheet
    has been imported."""
    script = _review_script()
    videos, tolerances = {}, {}
    for vdir in video_dirs(run_root, video_ids):
        lines = script.validity_lines(script._log_lines(vdir / script.REVIEW_DIR / script.LOG))
        if lines:
            videos[vdir.name] = lines
            tolerances[vdir.name] = script.Tolerances.of(_read(vdir / "agreement.json"))
    if not videos:
        return None
    return script.summary_text(Path(run_root), script.pooled_summary(videos, tolerances))


def review_commands(run_root: Path | str, video_id: str, python: str = ".venv/bin/python") -> str:
    """The commands that export a review sheet and import it when filled. Text to copy; nothing is run."""
    run = Path(run_root)
    try:
        run = run.relative_to(REPO)
    except ValueError:
        pass
    sheet = run / video_id / "review" / f"{video_id}__review.csv"
    return (f"# 1. write a review sheet (a CSV to fill in a spreadsheet) and its instructions\n"
            f"{python} scripts/review_panel.py export --run {run} --video-ids {video_id}\n\n"
            f"# 2. after filling the human_value / resolution columns, check and apply it\n"
            f"{python} scripts/review_panel.py import --run {run} --sheet {sheet} --reviewer YOUR_NAME\n\n"
            f"# 3. see where every video stands, and the pooled agreement\n"
            f"{python} scripts/review_panel.py status --run {run}\n"
            f"{python} scripts/review_panel.py summary --run {run}")


# ---- Stage A ranking ------------------------------------------------------------------------

def ranking_batches(run_root: Path | str) -> list[dict]:
    """The result of every ranking batch on disk (each a strict order of its clips per dimension)."""
    batches = Path(run_root) / "ranking" / "batches"
    if not batches.is_dir():
        return []
    results = [_read(d / "result.json") for d in sorted(batches.iterdir()) if d.is_dir()]
    return [r for r in results if isinstance(r, dict) and isinstance(r.get("rankings"), dict)]


def ranking_orders(run_root: Path | str) -> pd.DataFrame:
    """Per dimension and batch, the order the model gave: strongest first."""
    rows = []
    for batch in ranking_batches(run_root):
        for code, order in batch["rankings"].items():
            rows.append({"Dimension": code, "Batch": batch.get("batch_id"),
                         **{f"{i + 1}{' (strongest)' if i == 0 else ''}": v for i, v in enumerate(order)}})
    table = pd.DataFrame(rows)
    return table.sort_values(["Dimension", "Batch"], kind="stable").reset_index(drop=True) if len(table) else table


def ranking_scores(run_root: Path | str, video_ids: list[str] | None = None) -> pd.DataFrame:
    """The pooled rank score of every video on every dimension (higher is stronger; relative within the
    ranking as it was run). `video_ids` keeps the rows of those videos."""
    scores = _rank_scores(Path(run_root))
    if not scores:
        return pd.DataFrame()
    table = pd.DataFrame(scores).round(2)
    if video_ids is not None:
        table = table[table.index.isin(list(video_ids))]
    table.index.name = "Video"
    return table


def pairwise_order_agreement(orders: list[list[str]]) -> dict:
    """How consistently several orders rank the same pairs.

    For every pair of items that appear together in at least two orders: is the pair ordered the
    same way every time? Returns the number of such pairs, how many are always ordered the same
    way, and that share (None when no pair is seen twice).
    """
    seen: dict[tuple[str, str], list[bool]] = {}
    for order in orders:
        rank = {item: i for i, item in enumerate(order)}
        for a, b in combinations(sorted(rank), 2):
            seen.setdefault((a, b), []).append(rank[a] < rank[b])
    repeated = [votes for votes in seen.values() if len(votes) >= 2]
    consistent = sum(1 for votes in repeated if len(set(votes)) == 1)
    return {"pairs_seen_twice": len(repeated), "consistent": consistent,
            "share": consistent / len(repeated) if repeated else None}


def ranking_agreement(run_root: Path | str) -> pd.DataFrame:
    """Per dimension: over the pairs of videos ranked together in more than one batch, the share ordered
    the same way every time."""
    by_code: dict[str, list[list[str]]] = {}
    for batch in ranking_batches(run_root):
        for code, order in batch["rankings"].items():
            by_code.setdefault(code, []).append(list(order))
    rows = []
    for code, orders in by_code.items():
        figures = pairwise_order_agreement(orders)
        rows.append({"Dimension": code, "Batches": len(orders), "Pairs ranked more than once": figures["pairs_seen_twice"],
                     "Ordered the same way every time": figures["consistent"],
                     "Share consistent": _round(figures["share"])})
    return pd.DataFrame(rows, columns=["Dimension", "Batches", "Pairs ranked more than once",
                                       "Ordered the same way every time", "Share consistent"])


def _pearson(x, y) -> float:
    """Pearson correlation of two equally long sequences; NaN with fewer than three pairs or no spread."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() <= 1e-12 or y.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    """Spearman rank correlation: the Pearson correlation of the ranks, ties sharing their mean rank."""
    return _pearson(pd.Series(list(x), dtype=float).rank(), pd.Series(list(y), dtype=float).rank())


def batch_standings(run_root: Path | str) -> pd.DataFrame:
    """One row per dimension, video and ranking batch: the video's standing in that batch,
    1 - position / (batch size - 1), so 1 is the strongest clip of the batch and 0 the weakest.
    `appearance` numbers a video's batches in batch order (1 = its first)."""
    rows = []
    for batch in ranking_batches(run_root):
        for code, order in batch["rankings"].items():
            if len(order) < 2:
                continue
            rows += [{"code": code, "video_id": video, "batch_id": str(batch.get("batch_id")),
                      "standing": 1 - position / (len(order) - 1)} for position, video in enumerate(order)]
    table = pd.DataFrame(rows, columns=["code", "video_id", "batch_id", "standing"])
    table = table.sort_values(["code", "video_id", "batch_id"], kind="stable").reset_index(drop=True)
    table["appearance"] = table.groupby(["code", "video_id"]).cumcount() + 1
    return table


def _stage_a_ratings(run_root: Path, video_ids) -> dict[str, dict[str, dict]]:
    """pass -> video id -> Stage A code -> rating (None when uncodable), for the videos that have the pass."""
    out: dict[str, dict[str, dict]] = {"primary": {}, "replicate": {}}
    for video_id in video_ids:
        for name in out:
            result = _read(Path(run_root) / video_id / "results" / PASS_FILES[name])
            items = ((result or {}).get("record") or {}).get("stage_a")
            if items is not None:
                out[name][video_id] = {i["code"]: i.get("value") for i in items}
    return out


STABILITY_COLUMNS = ["Dimension", "Videos ranked in two batches", "Batch-to-batch correlation of standing",
                     "95% interval, low", "95% interval, high", "p-value", "Videos rated in both passes",
                     "Primary vs replicate rating (Spearman)", "Videos with a rating and a rank score",
                     "Primary rating vs rank score (Spearman)", "Batches per video"]


def ranking_stability(run_root: Path | str, n_boot: int = 1999, seed: int = 42) -> pd.DataFrame:
    """How stable the Stage A ranking is when each video is ranked in two different batches, per dimension.

    No pair of videos needs to meet twice. For every video ranked in at least two batches, its
    standing in its first batch is set against its standing in its second (see `batch_standings`):
    the Pearson correlation over videos, with a 95% percentile interval from resampling videos
    (`n_boot` draws). Beside it, over the same ranked videos: the Spearman correlation of the 1-5
    ratings the two annotation passes gave, and of the primary pass's rating with the pooled rank
    score. All ranked videos of the run are used, because a standing is relative to batch-mates.
    Empty when the run has no ranking batches.
    """
    standings = batch_standings(run_root)
    if standings.empty:
        return pd.DataFrame(columns=STABILITY_COLUMNS)
    scores = _rank_scores(Path(run_root))
    ratings = _stage_a_ratings(Path(run_root), sorted(set(standings.video_id)))
    rng = np.random.default_rng(seed)
    rows = []
    for code, part in standings.groupby("code", sort=False):
        # every video appears in the same number of batches by design; with more than two, the figure is the
        # mean correlation over the pairs of appearances (first with second, first with third, ...)
        wide_ = part.pivot(index="video_id", columns="appearance", values="standing")
        wide_ = wide_.dropna(axis=1, how="all").dropna() if wide_.shape[1] >= 2 else wide_.iloc[0:0]
        matrix = wide_.to_numpy()
        pairs = list(combinations(range(matrix.shape[1]), 2)) if len(matrix) else []

        def mean_pairwise(rows):
            values = [_pearson(rows[:, a], rows[:, b]) for a, b in pairs]
            values = [v for v in values if v == v]
            return float(np.mean(values)) if values else np.nan

        first = matrix[:, 0] if pairs else []
        draws = [mean_pairwise(matrix[i]) for i in rng.integers(len(first), size=(n_boot, len(first)))] if len(first) else []
        draws = [d for d in draws if d == d]
        low, high = np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan)
        both = [(ratings["primary"][v].get(code), ratings["replicate"][v].get(code))
                for v in ratings["primary"] if v in ratings["replicate"]]
        both = [(a, b) for a, b in both if a is not None and b is not None]
        ranked = [(ratings["primary"][v].get(code), scores.get(code, {}).get(v)) for v in ratings["primary"]]
        ranked = [(a, b) for a, b in ranked if a is not None and b is not None]
        rows.append({
            "Dimension": code, "Videos ranked in two batches": len(first),
            "Batch-to-batch correlation of standing": _round(mean_pairwise(matrix)) if pairs else None,
            "95% interval, low": _round(low), "95% interval, high": _round(high),
            "p-value": boot_p(draws) if draws else None,           # two-sided, from the same draws; null: no correlation
            "Videos rated in both passes": len(both),
            "Primary vs replicate rating (Spearman)": _round(spearman(*zip(*both))) if both else None,
            "Videos with a rating and a rank score": len(ranked),
            "Primary rating vs rank score (Spearman)": _round(spearman(*zip(*ranked))) if ranked else None,
            "Batches per video": matrix.shape[1] if pairs else None})
    return pd.DataFrame(rows, columns=STABILITY_COLUMNS)


def ranking_stability_reading(table: pd.DataFrame) -> str:
    """What the batch-to-batch correlations say, in plain words, with the smallest and largest."""
    column = "Batch-to-batch correlation of standing"
    known = table[table[column].notna()] if len(table) else table
    if known.empty:
        return ("No video was ranked in two batches (or the run has no ranking), so the stability of the ranking "
                "cannot be computed.")
    low, high = known.loc[known[column].idxmin()], known.loc[known[column].idxmax()]
    k = int(known["Batches per video"].max()) if "Batches per video" in known and known["Batches per video"].notna().any() else 2
    if k > 2:
        r = float(known[column].mean())
        return (f"Over {int(known['Videos ranked in two batches'].max())} videos, each ranked in {k} different batches, a "
                f"video's standing in one batch correlates with its standing in another (mean over the pairs of its "
                f"batches) between {low[column]:.2f} ({low['Dimension']}) and {high[column]:.2f} ({high['Dimension']}).\n"
                "How to read a value: 1 would mean a video always lands in the same place whoever it is compared with; "
                "0 that its place in one batch says nothing about its place in another. Around 0.5 is moderate. One "
                "batch gives only a coarse standing (a place among a handful of clips, which also depends on which "
                f"clips happened to be in the batch). The pooled rank score uses all {k} of a video's batches and is "
                f"steadier than any one: at the mean single-batch correlation here ({r:.2f}), a score pooled over {k} "
                f"batches is expected to correlate about {k * r / (1 + (k - 1) * r):.2f} with another such score "
                "(Spearman-Brown, k r / (1 + (k - 1) r)).\n"
                "The two rating columns are a different question: do the two annotation passes give the same 1-5 "
                "rating, and does the rating order the videos as the ranking does? A low value there usually means the "
                "rating barely varies on that dimension. All of it is model-coded: stability, not validity.")
    return (f"Over {int(known['Videos ranked in two batches'].max())} videos, each ranked in two different batches, a "
            f"video's standing in one batch correlates with its standing in the other between {low[column]:.2f} "
            f"({low['Dimension']}) and {high[column]:.2f} ({high['Dimension']}).\n"
            "How to read a value: 1 would mean a video always lands in the same place whoever it is compared with; 0 "
            "that its place in one batch says nothing about its place in another. Around 0.5 is moderate. One batch "
            "gives only a coarse standing (a place among a handful of clips, which also depends on which clips "
            "happened to be in the batch). The pooled rank score uses both of a video's batches and is steadier "
            "than either one: if single batches correlate 0.5, the average of two is expected to correlate about "
            "0.67 with another such average (2r / (1 + r)).\n"
            "The two rating columns are a different question: do the two annotation passes give the same 1-5 "
            "rating, and does the rating order the videos as the ranking does? A low value there usually means the "
            "rating barely varies on that dimension. All of it is model-coded: stability, not validity.")


# ---- comparison with the archived structured pilot ------------------------------------------

def comparison_tables(run_root: Path | str) -> dict[str, pd.DataFrame] | None:
    """Coverage and cost of this run beside the archived structured pilot, from comparison_vs_structured.json;
    None when the run has no such file."""
    report = _read(Path(run_root) / COMPARISON_FILE)
    if not isinstance(report, dict) or not isinstance(report.get("videos"), dict):
        return None
    coverage, cost = [], []
    for video_id, v in sorted(report["videos"].items()):
        c, k = v.get("coverage") or {}, v.get("cost") or {}
        coverage.append({
            "Video": video_id, "Old pipeline stopped at": clock(c.get("old_cutoff_s")),
            "This run's last turn ends": clock(c.get("new_last_turn_end_s")),
            "Old episodes": c.get("old_episodes"), "Turns": c.get("new_turns"),
            "Turns after the old stop": c.get("new_turns_after_cutoff"),
            "Questions": c.get("new_questions"), "Questions after the old stop": c.get("new_questions_after_cutoff"),
            "Answers": c.get("new_answers"), "Answers after the old stop": c.get("new_answers_after_cutoff"),
            "Substantive exchanges after the old stop": c.get("new_substantive_exchanges_after_cutoff")})
        cost.append({"Video": video_id,
                     **{f"{label} ({side})": k.get(f"{side_key}_{key}") for key, label in (
                         ("calls", "Calls"), ("input_tokens", "Input tokens"), ("output_tokens", "Output tokens"),
                         ("thinking_tokens", "Thinking tokens"))
                        for side_key, side in (("old", "structured pilot"), ("new", "this run"))}})
    return {"coverage": pd.DataFrame(coverage), "cost": pd.DataFrame(cost),
            "old_run": pd.DataFrame([{"Compared with": report.get("old_run")}])}


# ---- export ---------------------------------------------------------------------------------

def export_tables(run_root: Path | str, tables: dict[str, pd.DataFrame]) -> Path:
    """Write the given tables as CSV files under <run>/analysis/explorer/ and return that folder."""
    out = Path(run_root) / EXPLORER_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(out / f"{name}.csv", index=not isinstance(table.index, pd.RangeIndex))
    return out


# ---- human validation -----------------------------------------------------------------------

def human_validation(run: Path | str):
    """The paper's validation table of a run (review/paper_table.json, written by
    `scripts/review_panel.py summary --paper-table`) as a DataFrame, or None when no table has been written."""
    path = Path(run) / "review" / "paper_table.json"
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))["rows"])
