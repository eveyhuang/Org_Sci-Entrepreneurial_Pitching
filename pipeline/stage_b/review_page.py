"""One coding page per coder: the exported rows, the record's people and the codebooks' wording, rendered
into a self-contained HTML file that plays the video from the coder's disk and writes the import's CSV.

This half builds the page model: a plain dict per coder, holding nothing the model's passes judged."""
from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord, clock_to_seconds, seconds_to_clock
from pipeline.stage_b.panel_books import PanelBooks
from pipeline.stage_b.panel_pipeline import _read, prepared_window
from pipeline.stage_b.review_apply import value_kind
from pipeline.stage_b.review_items import (ASK, NOT_AN_EXCHANGE, OUTCOME, SHEET_COLUMNS, allowed_values, banded,
                                           canonical_full, is_banded, is_cut, is_outcome, locator, locator_span,
                                           parse_path)
from pipeline.stage_b.panel_books import is_other
from pipeline.stage_b.review_labels import (DEFAULT_EXCHANGE_UNIT, DEFAULT_HINTS, DEFAULT_SECTIONS, PAIR_VALUES, Labels,
                                            book_holding, coder_anchors, definition_html, entry_for, missing_options,
                                            path_glosses, question_gloss, question_text, scale_anchors, section_of,
                                            structural_text, value_labels)

TEMPLATE = Path(__file__).with_name("review_page_template.html")
MANIFEST = "export_manifest.json"
# a turn's summary ends where the next turn begins: at ` | turn N (`, at ` -> answer turn N (` or at the end
_TURN = re.compile(r"turn (\d+) \(([^)]+)\) (\w+) \(([^)]+)\): (.*?)(?= \| turn \d+ \(| -> answer turn \d+ \(|$)")
# the full-video page's sections, then the clip session's (each clip with its cards, then the pairs)
SECTION_ORDER = ("pitch", "conversation", "outcomes", "clips", "pairs")


class PageError(ValueError):
    pass


def _turns_in(context: str) -> list[dict]:
    """The turns a blind context names: `turn 9 (2:00-2:01) S4 (Kevin): summary | turn ...`."""
    tail = context.split("questions: ", 1)[1] if "questions: " in context else context
    return [{"turn": m.group(1), "clock": m.group(2), "who": m.group(4), "summary": m.group(5).strip()}
            for m in _TURN.finditer(tail)]


def _kind(path: str, field_class: str) -> str:
    """What a card is about, for ordering and the template: `OUTCOME`, a code id for a turn code, else the field class."""
    if is_outcome(path):
        return OUTCOME
    ref = parse_path(path)
    return ref.sub if ref.unit == "turn" and ref.field == "code" and ref.sub else field_class


def _input(path: str, books: PanelBooks) -> str | None:
    """The control a card's answer is typed or picked with, beyond a list of values: "clock" (M:SS, with a button
    that reads the player), "number" (a count), "ask" (dollars and per cent), "choice" (left or right of a pair, with
    a close tick); None for a card answered from its values (or, the prepared window, by the structural tick)."""
    kind = value_kind(path, books)
    if kind == "clock":
        return "clock"
    ref = parse_path(path)
    if kind == "number":
        return "ask" if ref.unit == "venture" and ref.unit_id == ASK else "number"
    return "choice" if ref.unit == "pair" else None


def _card(row: dict, number: int, allowed: tuple[str, ...] | None, record: dict, books: PanelBooks,
          labels: Labels) -> dict:
    """One item to judge. `row` is a manifest row with the blind `context`, `clock_start` and `clock_end` filled in."""
    path, field_class = row["path"], row["field_class"]
    entry = entry_for(path, books)
    values = list(allowed or ())
    # a pair asks which clip shows more of a Stage A item: left or right, never the item's own scale
    pair = parse_path(path).unit == "pair"
    if pair:
        values = list(PAIR_VALUES)
    anchors = {} if pair else scale_anchors(entry)
    if anchors and not values:
        values = list(anchors)
    defs = anchors or path_glosses(path, entry, tuple(values), labels, book_holding(books, entry))
    # the coders' own wording of a code (the labels' `anchors`) in place of the codebook's scale text; the codebook
    # definition below the card keeps the codebook's (the model's) wording
    coder = None if pair else entry
    defs = {**defs, **{v: text for v, text in coder_anchors(coder, labels).items() if v in values}}
    html_def = definition_html(entry)
    unit = ""
    m = re.match(r"^exchange (\d+), ([\d:]+-[\d:]+)", row["context"])
    if m:
        unit = f"{labels.sections.get('exchange_unit', DEFAULT_EXCHANGE_UNIT)} {m.group(1)} ({m.group(2)})"
    # `number` is what the card's header shows; it is the sheet's `row`, so a coder and the researcher name a card alike
    return {"row": number, "number": number, "input": _input(path, books), "path": path, "field_class": field_class,
            "kind": _kind(path, field_class),
            "q": question_text(path, record, books, labels),
            # the line under the question, and what a value's pill reads where it is not the value (`yes` for `true`);
            # a pair's line is its item's definition, first sentence (the Stage A codebook, shared with the model)
            "gloss": _first_sentence(entry.definition) if pair and entry is not None else question_gloss(coder, labels),
            "value_labels": {v: shown for v, shown in value_labels(coder, labels).items() if v in values},
            # the values whose choice needs the matter named in the note (`other`)
            "name_values": [v for v in values if is_other(v)],
            "start": row["clock_start"] or "", "end": row["clock_end"] or "", "unit_label": unit,
            "turns": _turns_in(row["context"]), "values": values, "defs": defs, "html_def": html_def,
            # the "cannot be rated" answers the import takes at this path (`missing:<reason>`); none for most
            "missing": missing_options(path, labels),
            "structural": structural_text(path, labels),
            # the codebook entry the card asks about, so a page that carries the codebook can point to it
            "entry": entry.id if entry is not None else None}


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=\.)\s+", " ".join(str(text or "").split()), maxsplit=1)[0]


MISSING_PREFIX = "missing:"
NO_REASON = "The model records no reason for this code; below is its summary of the turn."


def _clock(seconds) -> str:
    return seconds_to_clock(float(seconds)) if seconds is not None else ""


def _turn_line(t: dict, names: dict[str, str]) -> dict:
    return {"turn": str(t["turn_id"]), "clock": f"{_clock(t['start_s'])}-{_clock(t['end_s'])}",
            "who": names.get(t["speaker"], t["speaker"]), "summary": t.get("summary") or ""}


def model_side(row: dict, d: dict) -> dict:
    """What the model answered for one card and what it recorded in support, for a walkthrough with the coders:
    {"answer", "second", "reason", "turns"}. `answer` is the value of the pass the sheet was drawn from, `second`
    the other pass's value (None when it has none), `reason` the model's own note where the record keeps one, else
    a sentence saying what is shown instead, and `turns` the turns the model points to (with its summaries)."""
    path = row["path"]
    names = {a["actor_id"]: a["label"] for a in d["actors"]}
    turns = {t["turn_id"]: t for t in d["turns"]}
    out = {"answer": row.get("primary"), "second": row.get("replicate"), "reason": "", "turns": []}
    if is_outcome(path) or is_cut(path):
        investors = {a["actor_id"] for a in d["actors"] if a["role"] == "investor"}
        own = sorted((x for x in d["decisions"] if x["actor"] in investors), key=lambda x: x["time_s"])
        own = own[:1] if is_cut(path) else own
        out["reason"] = " · ".join(f"{names.get(x['actor'], x['actor'])}: {x['kind'].replace('_', ' ')} at "
                                   f"{_clock(x['time_s'])}" + (f" ({x['terms']})" if x.get("terms") else "")
                                   + (f": {x['note']}" if x.get("note") else "") for x in own) \
            or "The model records no decision of the panel."
        out["turns"] = [_turn_line(turns[x["turn_id"]], names) for x in own if x.get("turn_id") in turns]
        return out
    if path == "prepared_window":
        spans = ", ".join(f"{_clock(s['start_s'])}-{_clock(s['end_s'])}" for s in d.get("prepared_spans") or [])
        out["reason"] = (f"The model places the prepared presentation at {spans}." if spans
                         else "The model places no prepared presentation.")
        return out
    ref = parse_path(path)
    if ref.unit == "prep_pres":
        item = next((i for i in d.get("prepared") or [] if i["code"] == ref.unit_id), None) or {}
        out["reason"] = ("The model records no note for the prepared presentation's ratings"
                         + (f"; it marks this one missing ({item['missing'].replace('_', ' ')})."
                            if item.get("missing") not in (None, "none") else "."))
    elif ref.unit == "stage_a":
        item = next((i for i in d["stage_a"] if i["code"] == ref.unit_id), None) or {}
        moment = f" (a moment it points to: {_clock(item['moment_s'])})" if item.get("moment_s") is not None else ""
        out["reason"] = (item.get("note") or "The model records no note for this rating.") + moment
    elif ref.unit == "venture":
        out["reason"] = d["venture"].get("note") or ""
    elif ref.unit == "exchange":
        exchange = next((e for e in d["exchanges"] if e["exchange_id"] == ref.unit_id), None) or {}
        out["reason"] = (f"The model's own name for this line of questioning: {exchange['label']}."
                         if exchange.get("label") else "")
    elif ref.unit == "turn":
        out["reason"] = NO_REASON
        turn = turns.get(ref.unit_id)
        if turn is not None:
            out["turns"] = [_turn_line(turn, names)]
            if ref.sub and (turn.get("other_concept") or "") and ref.sub.endswith("CONCEPT"):
                out["reason"] += f" It named the matter: {turn['other_concept']}."
    return out


def _banded_side(side: dict, path: str) -> dict:
    """The model's side of a banded card (PRES-EVID): its count shown as the band a coder chooses, the count kept in
    the reason. Any other card's side is returned as it is."""
    if not is_banded(path) or side.get("answer") in (None, ""):
        return side
    count = side["answer"]
    answer = banded(path, count)
    if answer == count:                 # missing:<reason> or no count: nothing to band
        return side
    return {**side, "answer": answer, "second": banded(path, side.get("second")),
            "reason": " ".join(x for x in (side.get("reason") or "", f"Its count: {count}; a coder chooses its band.")
                               if x)}


def video_model(vdir: Path, entry: dict, books: PanelBooks, labels: Labels, with_model: bool = False) -> dict:
    if not entry.get("blind"):
        raise PageError(f"{vdir.name}: the export is not blind; a coding page is built from a blind export")
    # the pass the sheet was drawn from (final review I1; an older manifest names none: the primary)
    result = _read(vdir / "results" / f"{entry.get('pass') or 'primary'}.json")
    record = canonical_full(PanelRecord.model_validate(result["record"]), books)
    d = record.model_dump(mode="json")
    from pipeline.stage_b.review_clips import video_file
    file_name = video_file(vdir)
    first: dict[str, float] = {}
    for t in sorted(d["turns"], key=lambda t: t["start_s"]):
        first.setdefault(t["speaker"], t["start_s"])
    window = ", ".join(prepared_window(record))     # every span, as review_clips.window_text (final review M8)
    qa_order = [code for _, code in (entry.get("sample") or {}).get("requested", {}).get("qa_codes") or []]
    rank = {"exchange_topic": 0, **{code: 1 + i for i, code in enumerate(qa_order)}}
    cards = []
    # the number is the row's place on the exported sheet, as export_video numbers it
    for number, row in enumerate(entry["rows"], 1):
        allowed = allowed_values(row["path"], record, books, coder=True)        # a coding page is a coder's
        if allowed and row["field_class"] == "exchange_topic" and row["path"].endswith("/topic"):
            allowed = (*allowed, NOT_AN_EXCHANGE)
        start, end = locator_span(record, row["path"])      # what a blind sheet shows, not the model's times
        shown = {**row, "context": row.get("blind_context") or locator(record, row["path"]),
                 "clock_start": start, "clock_end": end}
        card = _card(shown, number, allowed, d, books, labels)
        if with_model:
            card["model"] = _banded_side(model_side(row, d), row["path"])
        cards.append(card)
    by_section = {key: [] for key in SECTION_ORDER}
    for c in cards:
        by_section[section_of(c["path"])].append(c)
    stage_order = {code: i for i, code in enumerate(books.stage_a_ids())}
    by_section["pitch"].sort(key=lambda c: stage_order.get(c["path"].partition(":")[2], 99))
    by_section["conversation"].sort(key=lambda c: (clock_to_seconds(c["start"]) if c["start"] else -1,
                                                    rank.get(c["kind"], 99), c["path"]))
    by_section["outcomes"].sort(key=lambda c: (not is_outcome(c["path"]), c["path"]))     # the outcome, then the cut
    # a sheet that rates neither delivery nor the prepared content only locates the prepared presentation
    rated = any(c["field_class"] in ("stage_a", "prep_pres") for c in by_section["pitch"])
    intros = {"pitch": None if rated else {**DEFAULT_HINTS, **labels.hints}["clip_session_intro"]}
    # the outcomes section without an outcome card holds the first decision only, and is headed so
    titled = {key: labels.sections.get(key) or DEFAULT_SECTIONS[key] for key in SECTION_ORDER}
    if by_section["outcomes"] and not any(is_outcome(c["path"]) for c in by_section["outcomes"]):
        titled["outcomes"] = labels.sections.get("first_decision") or DEFAULT_SECTIONS["first_decision"]
    return {"video_id": vdir.name, "title": os.path.splitext(file_name)[0], "file_name": file_name,
            "file_alts": fallback_names(vdir.name, file_name),
            "actors": {a["actor_id"]: a["label"] for a in d["actors"]},
            "actors_first": {a: seconds_to_clock(s) for a, s in first.items()},
            "window": window,
            # `intro`: the section's own intro text; None for the page's hint (hints.pitch_intro for the pitch)
            "sections": [{"key": key, "title": titled[key], "intro": intros.get(key), "cards": by_section[key]}
                         for key in SECTION_ORDER if by_section[key]]}


VALIDATION, TRAINING, WALKTHROUGH, CLIPS = "validation", "training", "walkthrough", "clips"
TRAINING_BANNER = ("Training: these sheets are for practice and for calibrating the coders against each other. "
                   "They are not part of the validation and are never imported into it.")
WALKTHROUGH_BANNER = ("Model answers, for the researcher's walkthrough with the coders after they have coded the "
                      "training videos. Do not give this page to a coder before then: it shows the model's answers.")
HEADINGS = {VALIDATION: "Coding sheets", TRAINING: "Training sheets", WALKTHROUGH: "Model answers",
            CLIPS: "Clip session"}
BANNERS = {VALIDATION: "", TRAINING: TRAINING_BANNER, WALKTHROUGH: WALKTHROUGH_BANNER, CLIPS: ""}
LIST_TITLES = {VALIDATION: "Videos in your sample", TRAINING: "Training videos", WALKTHROUGH: "Training videos",
               CLIPS: "Videos in your clip session"}
TRAINING_CLIPS_HEADING, TRAINING_CLIPS_LIST = "Training clip session", "Training videos (clip session)"
# the task page (review_panel.py session-export; training --clips): one page per coder
TASKS, TRAINING_TASKS = "tasks", "training_tasks"
TASK_HEADINGS = {TASKS: "Coding tasks", TRAINING_TASKS: "Training tasks"}
TASK_LISTS = {TASKS: "Your tasks", TRAINING_TASKS: "Your training tasks"}
COMPARE_PREFIX = "compare-"     # a comparison task's entry: compare-<batch id> (a name no video of a session takes)
TASK_HINTS = ("pitch_card", "prep_pres_card", "outcome_card", "cut_card", "protocol_for_coders", "compare_intro")


def page_model(run: Path, video_ids: list[str], coder: str, books: PanelBooks, labels: Labels,
               out_dirs: dict[str, Path] | None = None, order_seed: int | None = None, mode: str = VALIDATION,
               codebook: list[dict] | None = None) -> dict:
    """The page of one coder: the videos in the order given. `order_seed` records the seed the caller shuffled
    that order with (None when it did not). `mode` is "validation" or "training": a training page says so on
    every screen and keeps its answers apart from a validation page of the same run and coder. `codebook`
    (from review_codebook.codebook_model) lays the whole instrument out on the page when given."""
    if mode not in HEADINGS or mode == CLIPS:      # the task pages are tasks_page_model's
        raise PageError(f"mode must be one of {', '.join(repr(m) for m in HEADINGS if m != CLIPS)}, not {mode!r}"
                        + ("; a clip session's pages are written by `review_panel.py clips-export`" if mode == CLIPS
                           else ""))
    videos = []
    for video_id in video_ids:
        vdir = run / video_id
        review_dir = (out_dirs or {}).get(video_id) or vdir / "review"
        manifest = review_dir / MANIFEST
        if not manifest.exists():
            raise PageError(f"{video_id}: no export at {manifest}; run `review_panel.py export --blind` first")
        entry = (_read(manifest).get("videos") or {}).get(video_id)
        if entry is None:
            raise PageError(f"{manifest} lists no video {video_id}")
        videos.append(video_model(vdir, entry, books, labels, with_model=mode == WALKTHROUGH))
    return {"coder": coder, "run": run.name, "release_id": load_release().release_id, "columns": list(SHEET_COLUMNS),
            "order_seed": order_seed, "hints": {**DEFAULT_HINTS, **labels.hints}, "videos": videos,
            "mode": mode, "heading": HEADINGS[mode], "banner": BANNERS[mode], "list_title": LIST_TITLES[mode],
            "codebook": codebook or []}


def clips_page_model(run: Path, manifest: dict, coder: str, books: PanelBooks, labels: Labels,
                     codebook: list[dict] | None = None, training: bool = False) -> dict:
    """A coder's page of the clip session (mode "clips"), from the session's manifest (review_clips): one entry per
    video in the coder's own order, with its seven cards, its full video (from the coder's folder, by its source
    base name) and its prepared window (every span; the ▶ button seeks to the first), then one entry holding the
    pairs in design order, each with its two videos and windows and its three left/right cards. The cards carry the
    sheet's row and nothing of the model's side; their own clocks are left empty, as the window button locates them.
    A `training` session's page carries the training banner (its answers are kept apart by the session's design
    id). A manifest written before 2026-10-21 names clip files (`clip_files`) and no windows: its page plays those."""
    from pipeline.stage_b.review_clips import PAIRS_ID, pairs_intro, record_of
    hints = {**DEFAULT_HINTS, **labels.hints}
    sections = {key: labels.sections.get(key) or DEFAULT_SECTIONS[key] for key in ("clips", "pairs")}
    order = (manifest.get("order") or {}).get(coder)
    if order is None:
        raise PageError(f"the clip session's manifest has no video order for {coder!r}; export the session for them")
    files, windows = _session_files(manifest)
    intro = hints["protocol_for_coders"]        # not pitch_intro: the coders watch the full video at its window
    shown = lambda row: {**row, "clock_start": "", "clock_end": ""}     # noqa: E731  (the window button locates them)
    videos = []
    for video_id in order:
        vdir = run / video_id
        record = record_of(run, video_id, books, manifest)
        d = record.model_dump(mode="json")
        experiment = _read(vdir / "experiment.json") if (vdir / "experiment.json").exists() else {}
        cards = [_card(shown(row), row["row"], tuple(row["allowed_values"] or ()) or None, d, books, labels)
                 for row in manifest["rows"] if row.get("video_id") == video_id]
        videos.append({"video_id": video_id,
                       "title": os.path.splitext(os.path.basename(experiment.get("source") or video_id))[0],
                       "file_name": files[video_id], "file_alts": fallback_names(video_id, files[video_id]),
                       "actors": {}, "actors_first": {}, "window": windows[video_id],
                       "sections": [{"key": "clips", "title": sections["clips"], "intro": intro, "cards": cards}]})
    any_record = d if order else {}
    pair_cards = [_card(shown(row), row["row"], None, any_record, books, labels)
                  for row in manifest["rows"] if row["field_class"] == "pair"]
    videos.append({"video_id": PAIRS_ID, "title": sections["pairs"], "file_name": "", "actors": {},
                   "actors_first": {}, "window": "",
                   "pairs": [_pair_entry(p, files, windows) for p in manifest["pairs"]],
                   "sections": [{"key": "pairs", "title": sections["pairs"],
                                 "intro": pairs_intro(hints["pairs_intro"], len(order)),
                                 "cards": pair_cards}]})
    return {"coder": coder, "run": run.name, "release_id": manifest.get("release_id") or load_release().release_id,
            "columns": list(SHEET_COLUMNS), "order_seed": manifest.get("seed"), "hints": hints, "videos": videos,
            "mode": CLIPS, "heading": TRAINING_CLIPS_HEADING if training else HEADINGS[CLIPS],
            "banner": TRAINING_BANNER if training else BANNERS[CLIPS],
            "list_title": TRAINING_CLIPS_LIST if training else LIST_TITLES[CLIPS],
            # the page keeps its answers per design and writes the design on its sheet
            "design_id": manifest.get("design_id"), "codebook": codebook or []}


def short_title(title: str) -> str:
    """A video's title for a comparison row: the part before its first " | " (the episode's own name, without the
    channel or show that follows)."""
    return str(title).split(" | ", 1)[0].strip() or str(title)


def task_order(order: list[str], pairs: list[dict]) -> list[tuple[str, str]]:
    """The task page's list for one coder: ("video", id) in the coder's own order, each followed by ("pair", batch id)
    for every pair, in design order, whose two videos are now both on the list; a pair whose videos are not all in
    `order` comes last (it cannot happen for a session's own manifest)."""
    seen, placed, out = set(), set(), []
    for video_id in order:
        out.append(("video", video_id))
        seen.add(video_id)
        for p in pairs:
            if p["batch_id"] not in placed and p["left"] in seen and p["right"] in seen:
                placed.add(p["batch_id"])
                out.append(("pair", p["batch_id"]))
    return out + [("pair", p["batch_id"]) for p in pairs if p["batch_id"] not in placed]


def tasks_page_model(run: Path, manifest: dict, coder: str, books: PanelBooks, labels: Labels,
                     out_dirs: dict[str, Path] | None = None, codebook: list[dict] | None = None,
                     training: bool = False) -> dict:
    """One page per coder (2026-10-21, the researcher's "order A"): a list of tasks. A video task holds that video's
    seven cards of the session (under "The prepared presentation", its window on a ▶ button) and then the cards of
    its blind full-video export (under "The questions and answers", as the full-video page orders them, the first
    decision last); a comparison task holds one pair of the session's design, its two videos and windows and its
    three left/right cards. The order is `task_order`: the coder's video order, each comparison as soon as both its
    videos have come. The page writes the clip-session sheet (clips__<coder>.csv, its cards' rows of the session's
    manifest) and each video's full-video sheet (<video_id>__<coder>.csv), as the two separate pages did, so the
    imports read them unchanged. `out_dirs` gives a video's export directory (default <run>/<video>/review)."""
    from pipeline.stage_b.review_clips import PAIRS_ID, record_of
    hints = {**DEFAULT_HINTS, **labels.hints}
    sections = {key: labels.sections.get(key) or DEFAULT_SECTIONS[key] for key in ("prepared", "qa", "compare")}
    order = (manifest.get("order") or {}).get(coder)
    if order is None:
        raise PageError(f"the session's manifest has no video order for {coder!r}; export the session for them")
    taken = [v for v in order if v == PAIRS_ID or v.startswith(COMPARE_PREFIX)]
    if taken:
        raise PageError(f"a video may not be called {taken[0]!r}: the task page names its entries so")
    files, windows = _session_files(manifest)
    shown = lambda row: {**row, "clock_start": "", "clock_end": ""}     # noqa: E731  (the window button locates them)
    tasks: dict[tuple[str, str], dict] = {}
    titles: dict[str, str] = {}
    for video_id in order:
        vdir = run / video_id
        record = record_of(run, video_id, books, manifest)
        d = record.model_dump(mode="json")
        prepared = [_card(shown(row), row["row"], tuple(row["allowed_values"] or ()) or None, d, books, labels)
                    for row in manifest["rows"] if row.get("video_id") == video_id]
        export = ((out_dirs or {}).get(video_id) or vdir / "review") / MANIFEST
        entry = ((_read(export) if export.exists() else {}).get("videos") or {}).get(video_id)
        if entry is None:
            raise PageError(f"{video_id}: no blind export at {export}; run `review_panel.py session-export` (or "
                            "`export --blind`) first")
        full = video_model(vdir, entry, books, labels)
        qa = [c for sec in full["sections"] for c in sec["cards"]]
        clash = sorted({c["path"] for c in prepared} & {c["path"] for c in qa})
        if clash:
            raise PageError(f"{video_id}: {', '.join(clash)} is on both the session's sheet and the full-video sheet; "
                            "export the full-video sheet without it (the session codes it)")
        titles[video_id] = full["title"]
        tasks[("video", video_id)] = {
            "video_id": video_id, "task": "video", "title": f"{short_title(full['title'])} — presentation and Q&A",
            "short_title": short_title(full["title"]), "file_name": files[video_id],
            "file_alts": fallback_names(video_id, files[video_id]), "actors": full["actors"],
            "actors_first": full["actors_first"], "window": windows[video_id],
            # `sheet`: which sheet a section's cards are written to, the session's or the video's own
            "sections": [{"key": "prepared", "title": sections["prepared"], "intro": hints["protocol_for_coders"],
                          "sheet": "session", "cards": prepared},
                         {"key": "qa", "title": sections["qa"], "intro": None, "sheet": "video", "cards": qa}]}
    by_batch: dict[str, list[dict]] = {}
    for row in manifest["rows"]:
        if row["field_class"] == "pair":
            by_batch.setdefault(row["batch_id"], []).append(_card(shown(row), row["row"], None, d if order else {},
                                                                  books, labels))
    named: dict[frozenset, int] = {}
    for p in manifest["pairs"]:
        pair = {**_pair_entry(p, files, windows), "left_title": short_title(titles.get(p["left"], p["left"])),
                "right_title": short_title(titles.get(p["right"], p["right"]))}
        title = f"Compare: {pair['left_title']} vs {pair['right_title']}"
        both = frozenset((p["left"], p["right"]))
        named[both] = named.get(both, 0) + 1            # a pair shown again (training) is told apart by its count
        tasks[("pair", p["batch_id"])] = {
            "video_id": f"{COMPARE_PREFIX}{p['batch_id']}", "task": "compare",
            "title": title + (f" ({named[both]})" if named[both] > 1 else ""), "file_name": "", "actors": {},
            "actors_first": {}, "window": "", "pair": pair, "videos": [p["left"], p["right"]],
            "sections": [{"key": "compare", "title": sections["compare"], "intro": hints["compare_intro"],
                          "sheet": "session", "cards": by_batch.get(p["batch_id"], [])}]}
    mode = TRAINING_TASKS if training else TASKS
    # only the hints the task page reads: the protocol's own paragraph (pitch_intro) speaks of a clip, and the page
    # shows the coders' version (protocol_for_coders) in its place
    page_hints = {**{k: hints[k] for k in TASK_HINTS}, "pitch_intro": hints["protocol_for_coders"]}
    return {"coder": coder, "run": run.name, "release_id": manifest.get("release_id") or load_release().release_id,
            "columns": list(SHEET_COLUMNS), "order_seed": manifest.get("seed"), "hints": page_hints,
            "videos": [tasks[key] for key in task_order(order, manifest["pairs"])],
            "mode": mode, "heading": TASK_HEADINGS[mode], "banner": TRAINING_BANNER if training else "",
            "list_title": TASK_LISTS[mode], "design_id": manifest.get("design_id"), "codebook": codebook or []}


def _session_files(manifest: dict) -> tuple[dict[str, str], dict[str, str]]:
    """({video_id: file in the coder's folder}, {video_id: prepared window}) of a clip session's manifest: the full
    videos and their windows; a manifest written before 2026-10-21 names its clip files and no windows."""
    files = manifest.get("video_files") or manifest.get("clip_files") or {}
    windows = manifest.get("windows") or {}
    return files, {v: windows.get(v, "") for v in manifest.get("videos") or files}


def fallback_names(video_id: str, file_name: str) -> list[str]:
    """The names a page tries after a video's own file name (its source's base name): `<video_id>.mp4`, for a folder
    whose files are named by video id (2026-10-22); none when that is the file name already."""
    neutral = f"{video_id}.mp4"
    return [] if not file_name or file_name == neutral else [neutral]


def _pair_entry(p: dict, files: dict[str, str], windows: dict[str, str], name=lambda v: v) -> dict:
    """A pair as the page draws it: each side's entry name, file (and the fallback name) and window."""
    return {"batch_id": p["batch_id"], "left": name(p["left"]), "right": name(p["right"]),
            "left_file": files[p["left"]], "right_file": files[p["right"]],
            "left_alts": fallback_names(p["left"], files[p["left"]]),
            "right_alts": fallback_names(p["right"], files[p["right"]]),
            "left_window": windows.get(p["left"], ""), "right_window": windows.get(p["right"], "")}


def _signed(x: float) -> str:
    return f"{x:+.2f}"


def clips_walkthrough(run: Path, manifest: dict, books: PanelBooks, labels: Labels,
                      scores: dict[str, dict[str, float]]) -> list[dict]:
    """The clip session's entries of the model-answers page: each video at its prepared window with its seven cards
    and the model's side (a Stage A card adds the video's rank score; PRES-EVID's count is shown as its band), then
    the pairs with the model's choice and both rank scores. `scores` is {stage_a item: {video_id: rank score}}
    (review_clips.model_scores). A clip-session entry is named `<video_id>__prepared` (the clips' old file name), so
    it never shares a name with the full video's own entry on the same page, though both play the same file."""
    from pipeline.stage_b.review_clips import PAIRS_ID, clip_file, pairs_intro, record_of
    hints = {**DEFAULT_HINTS, **labels.hints}
    sections = {key: labels.sections.get(key) or DEFAULT_SECTIONS[key] for key in ("clips", "pairs")}
    files, windows = _session_files(manifest)
    intro = hints["protocol_for_coders"]
    entry_id = lambda video_id: os.path.splitext(clip_file(video_id))[0]          # noqa: E731
    shown = lambda row: {**row, "clock_start": "", "clock_end": ""}               # noqa: E731
    videos = []
    for video_id in manifest["videos"]:
        vdir = run / video_id
        record = record_of(run, video_id, books, manifest)
        d = record.model_dump(mode="json")
        experiment = _read(vdir / "experiment.json") if (vdir / "experiment.json").exists() else {}
        cards = []
        for row in manifest["rows"]:
            if row.get("video_id") != video_id:
                continue
            card = _card(shown(row), row["row"], tuple(row["allowed_values"] or ()) or None, d, books, labels)
            card["model"] = _banded_side(model_side(row, d), row["path"])
            code = row["path"].partition(":")[2]
            if row["field_class"] == "stage_a" and video_id in scores.get(code, {}):
                card["model"]["reason"] += (f" Its rank score for this presentation, from the ranking of the whole run: "
                                            f"{_signed(scores[code][video_id])}.")
            cards.append(card)
        title = os.path.splitext(os.path.basename(experiment.get("source") or video_id))[0]
        videos.append({"video_id": entry_id(video_id), "title": f"{title} (clip session)",
                       "file_name": files[video_id], "file_alts": fallback_names(video_id, files[video_id]),
                       "actors": {}, "actors_first": {}, "window": windows[video_id],
                       "sections": [{"key": "clips", "title": sections["clips"], "intro": intro, "cards": cards}]})
    pair_cards = []
    for row in manifest["rows"]:
        if row["field_class"] != "pair":
            continue
        card = _card(shown(row), row["row"], None, {}, books, labels)      # a pair card reads no record
        own = scores.get(row["code"], {})
        both = (f"The model's rank scores: left ({row['left']}) {_signed(own[row['left']])}, right ({row['right']}) "
                f"{_signed(own[row['right']])}" if row["left"] in own and row["right"] in own else
                "The model's rank scores are not in the manifest")
        card["model"] = {"answer": row["primary"], "second": None, "turns": [],
                         "reason": both + (": they tie, so the model prefers neither presentation."
                                           if row["primary"] is None else
                                           f"; it chooses the {row['primary']} video's prepared presentation "
                                           f"(gap {row['gap']:.2f}).")}
        pair_cards.append(card)
    videos.append({"video_id": PAIRS_ID, "title": sections["pairs"], "file_name": "", "actors": {},
                   "actors_first": {}, "window": "",
                   "pairs": [_pair_entry(p, files, windows, entry_id) for p in manifest["pairs"]],
                   "sections": [{"key": "pairs", "title": sections["pairs"],
                                 "intro": pairs_intro(hints["pairs_intro"], len(manifest["videos"])),
                                 "cards": pair_cards}]})
    return videos


def embed_json(model: dict) -> str:
    """The model as JSON that cannot close the script element it is written into (ASCII, `</` escaped)."""
    return json.dumps(model).replace("</", "<\\/")


def render_page(model: dict) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    title = f"{model.get('heading') or 'Coding sheets'} · {model['coder']}"
    return template.replace("__TITLE__", html.escape(title)).replace("__PAGE__", embed_json(model))


def render_notebook_preview(model: dict) -> str:
    """Read-only HTML from the coding page's own cards and submitted state, with no scripts.

    Works in notebook HTML outputs without executing the standalone application's JavaScript.
    All cards, anchors, notes and flags remain inspectable; editing and video selection belong
    to the standalone page. Values are escaped, including the human's free-text notes.
    """
    esc = lambda value: html.escape(str(value if value is not None else ''), quote=True)
    parts = ['<div style="font:14px/1.5 system-ui;color:#243746;background:#f5f7f9;'
             'padding:20px;border:1px solid #ccd5dd;border-radius:10px;max-width:960px;max-height:560px;overflow:auto">',
             f'<h3 style="margin:0">{esc(model.get("heading", "Coding sheet"))}</h3>',
             '<p>Read-only notebook preview · expand a section to inspect its cards. '
             'Open the full sheet in your browser to use the video and coding controls.</p>']
    answers = model.get('completed_answers', {})
    for video in model['videos']:
        parts.append(f'<h4>{esc(video["title"])}</h4>')
        state = answers.get(video['video_id'], {})
        for index, section in enumerate(video['sections']):
            parts.append(f'<details{" open" if index == 0 else ""}><summary style="cursor:pointer;padding:10px;font-weight:600">'
                         f'{esc(section["title"])} · {len(section["cards"])} cards</summary>')
            for card in section['cards']:
                answer = state.get(card['path'], {})
                value = str(answer.get('value', ''))
                parts.append(f'<article style="background:white;border:1px solid #ccd5dd;'
                             f'border-radius:8px;padding:16px;margin:10px 0" data-path="{esc(card["path"])}">'
                             f'<b>{esc(card["q"])}</b>')
                if card.get('start'):
                    parts.append(f'<p>{esc(card["start"])}–{esc(card.get("end"))}</p>')
                if card.get('gloss'):
                    parts.append(f'<p>{esc(card["gloss"])}</p>')
                if card.get('turns'):
                    parts.append('<details><summary>Locating summaries from the model</summary>')
                    for turn in card['turns']:
                        parts.append(f'<p>{esc(turn["clock"])} · {esc(turn["who"])}: {esc(turn["summary"])}</p>')
                    parts.append('</details>')
                parts.append('<div style="margin:12px 0;line-height:2.6">')
                for option in card.get('values', []):
                    selected = str(option) == value
                    colour = 'background:#dcefeb;border:2px solid #287a67' if selected else 'background:#f7f9fb;border:1px solid #ccd5dd'
                    label = card.get('value_labels', {}).get(str(option), str(option).replace('_', ' '))
                    parts.append(f'<span style="display:inline-block;padding:0 9px;margin:3px;border-radius:5px;{colour}" '
                                 f'title="{esc(card.get("defs", {}).get(str(option), ""))}">'
                                 f'{"✓ " if selected else ""}{esc(label)}</span>')
                parts.append('</div>')
                parts.append(f'<p><b>{esc(model["coder"])}’s answer:</b> {esc(value) or "Not answered"}</p>')
                flags = [label for key, label in [('unsure', 'Unsure'), ('unclear', 'Cannot tell'),
                                                   ('structural', 'Problem with the unit')] if answer.get(key)]
                if flags:
                    parts.append(f'<p><b>Flags:</b> {esc(", ".join(flags))}</p>')
                if answer.get('note'):
                    parts.append(f'<p><b>Note:</b> {esc(answer["note"])}</p>')
                if card.get('defs'):
                    parts.append('<details><summary>Scale and definitions</summary><dl>')
                    for option, definition in card['defs'].items():
                        parts.append(f'<dt><b>{esc(option)}</b></dt><dd>{esc(definition)}</dd>')
                    parts.append('</dl></details>')
                parts.append('</article>')
            parts.append('</details>')
    parts.append('</div>')
    return ''.join(parts)


def on_sheet(model: dict) -> set[str]:
    """The ids of the codebook entries the page's cards ask about."""
    return {c["entry"] for v in model["videos"] for sec in v["sections"] for c in sec["cards"] if c.get("entry")}
