"""A discussion page for the researcher and one coder (2026-10-29, the researcher's request): the cases where the
coder and Gemini disagree, on the task page's interface. It is not part of the validation: its sheet
(<coder>__discussion.csv) is never imported, and the coder's imported answers are never touched.

Two kinds of task:
- per video with a question-answer turn the coder flagged as structurally wrong (`resolution == "structural"`, the
  coder's latest line per path) or a question concept the coder put in another frame than Gemini's (promotion,
  prevention, neither; 2026-10-29), the cards of every question-answer pair holding one (all six codes of the pair),
  the cards that brought it marked;
- per drawn comparison (`draw_comparisons`), the one row of a pair where the coder's choice is not the model's.

Every card carries, read-only, the coder's answer and the model's (the pass the analysis reads, and the other pass),
then the card's own inputs for the answer after discussion and a reason."""
from __future__ import annotations

import random
import re
from pathlib import Path

from pipeline.instrument import load_release
from pipeline.stage_b import review_clips
from pipeline.stage_b.panel_books import PanelBooks, frame_of
from pipeline.stage_b.panel_pipeline import _read
from pipeline.stage_b.review_apply import SUMMARY_SHOWN, UNSURE
from pipeline.stage_b.review_items import SHEET_COLUMNS, parse_path
from pipeline.stage_b.review_labels import DEFAULT_HINTS, DEFAULT_SECTIONS, PAIR_VALUES, Labels
from pipeline.stage_b.review_page import (COMPARE_PREFIX, MANIFEST, TASK_HINTS, PageError, _card, _pair_entry,
                                          _session_files, short_title, video_model)

DISCUSSION = "discussion"                       # the page's mode, and the last part of its sheet's name
PAGE = "{coder}__discussion.html"               # under <run>/review/pages/
SHEET = "{coder}__discussion.csv"               # never <video>__<coder>.csv or clips__<coder>.csv: no import reads it
DIR = "discussion"                              # <run>/review/discussion/<coder>/manifest.json
EXTRA_COLUMNS = ["coder_value", "gemini_value", "gemini_other_pass"]
STRUCTURAL = "structural"
FLAGGED_SECTION = "flagged"
CONCEPT = "PR-CONCEPT"
STRUCTURAL_FLAG, FRAME_DISAGREEMENT, CONTEXT = "structural flag", "frame disagreement", "context of its pair"
BANNER = ("Discussion page, for the researcher and {coder} together: {coder}'s answers beside Gemini's. Not part of "
          "the validation: its sheet ({sheet}) is never imported, and {coder}'s imported answers stay as they are.")
FLAGGED_INTRO = ("The question–answer pairs in which {coder} marked a turn as structurally wrong, or gave a question "
                 "concept of another frame than Gemini's (promotion, prevention, neither), with all six codes of each "
                 "pair; the cards that brought the pair here are marked. Each card shows {coder}'s answer and "
                 "Gemini's; the inputs below them are for the answer after discussion (optional) and the reason.")
COMPARE_INTRO = ("A comparison on which {coder}'s choice is not Gemini's (Gemini's choice is implied by its rank "
                 "scores). The inputs below are for the answer after discussion (optional) and the reason.")
RULE = ("Candidates: {coder}'s standing answer (the last pairs_log line) per (batch, code) of the live design "
        "{design}, where it is left or right and differs from Gemini's implied choice (a tie in the rank scores, an "
        "unanswered or withdrawn row is no candidate). Drawn without replacement with random.Random({seed}): the "
        "three dimensions (Stage A items) take turns in the codebook's order, one row each, until {n} rows are drawn "
        "or none is left; within a dimension a row is drawn with probability proportional to its model gap (the "
        "difference of the two rank scores) among the rows left, a row of a pair already drawn being taken only when "
        "the dimension has no other.")
_PAIR_CONTEXT = re.compile(r"question turn (\d+) .*?-> answer turn (\d+)")
_MARKERS = (UNSURE, SUMMARY_SHOWN, review_clips.CLOSE)


def latest_lines(lines: list[dict], coder: str) -> dict[str, dict]:
    """The coder's last log line per path (role coder; a line a function change implied is no answer)."""
    last: dict[str, dict] = {}
    for line in lines:
        if line.get("reviewer") == coder and line.get("role", "adjudicator") == "coder" and not line.get("implied"):
            last[line["path"]] = line
    return last


def flagged_turns(last: dict[str, dict]) -> set[int]:
    """The turns of the question-answer cards the coder's latest answer marks as structurally wrong."""
    out = set()
    for path, line in last.items():
        ref = parse_path(path)
        if ref.unit == "turn" and line.get("resolution") == STRUCTURAL:
            out.add(int(ref.unit_id))
    return out


def qa_pair_of(row: dict) -> tuple[int, ...]:
    """(question turn, answer turn) a turn-code row belongs to, from its blind context; (turn,) when it names none."""
    m = _PAIR_CONTEXT.search(row.get("blind_context") or row.get("context") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (int(parse_path(row["path"]).unit_id),)


def _clean(note) -> str:
    text = str(note or "")
    for marker in _MARKERS:
        text = text.replace(marker, " ")
    return " ".join(text.split())


def coder_side(line: dict | None) -> dict:
    """The coder's imported answer to a card: {"value", "resolution", "unsure", "summary_shown", "note", "cell"};
    `value` is what the coder entered (the log's `typed`, else its `new` when a value was given), `cell` the sheet's
    coder_value (the value, then [unsure], [unclear] or [structural])."""
    if line is None:
        return {"value": "", "resolution": "", "unsure": False, "summary_shown": False, "note": "", "cell": "",
                "answered": False}
    resolution = str(line.get("resolution") or "")
    boxed = resolution in ("unclear", STRUCTURAL)
    value = line.get("typed") if "typed" in line else (None if boxed else line.get("new"))
    value = "" if value is None else str(value)
    note = str(line.get("note") or "")
    unsure = UNSURE in note
    cell = " ".join(x for x in (value, UNSURE if unsure else "", f"[{resolution}]" if boxed else "") if x)
    return {"value": value, "resolution": resolution if boxed else "", "unsure": unsure,
            "summary_shown": SUMMARY_SHOWN in note, "note": _clean(note), "cell": cell, "answered": True}


def draw_comparisons(lines: list[dict], coder: str, design: str | None, n: int, seed: int,
                     codes: list[str]) -> tuple[list[dict], dict]:
    """The comparisons to discuss and how they were drawn (RULE): (the drawn pairs_log lines, sorted by batch and
    code, {"rule", "candidates", "by_dimension"})."""
    standing = review_clips.last_pairs(review_clips.of_design(lines, design))
    candidates = [line for (who, _, _), line in standing.items() if who == coder
                  and line.get("value") in PAIR_VALUES and line.get("model") in PAIR_VALUES
                  and line["value"] != line["model"]]
    order = [c for c in codes if any(l["code"] == c for l in candidates)]
    order += sorted({l["code"] for l in candidates} - set(order))
    pools = {c: sorted((l for l in candidates if l["code"] == c), key=lambda l: l["batch_id"]) for c in order}
    rng = random.Random(seed)
    drawn, used = [], set()
    while len(drawn) < n and any(pools.values()):
        for code in order:
            pool = pools[code]
            if len(drawn) >= n or not pool:
                continue
            fresh = [l for l in pool if l["batch_id"] not in used] or pool
            weights = [max(float(l.get("gap") or 0.0), 1e-9) for l in fresh]
            point, pick = rng.random() * sum(weights), fresh[-1]
            for line, weight in zip(fresh, weights):
                point -= weight
                if point < 0:
                    pick = line
                    break
            pool.remove(pick)
            drawn.append(pick)
            used.add(pick["batch_id"])
    rank = {c: i for i, c in enumerate(order)}
    drawn.sort(key=lambda l: (l["batch_id"], rank[l["code"]]))
    return drawn, {"rule": RULE.format(coder=coder, design=design, seed=seed, n=n), "candidates": len(candidates),
                   "by_dimension": {c: {"candidates": sum(l["code"] == c for l in candidates),
                                        "drawn": sum(l["code"] == c for l in drawn)} for c in order}}


def _value(v):
    return None if v in (None, "") else str(v)


def frame(concept) -> str | None:
    """A question concept's frame as the discussion page names it (the researcher's grouping, 2026-10-29):
    promotion, prevention, or neither (deal_terms, other, NONE); None for no concept (no value, a "cannot be rated"
    answer, a ticked box)."""
    got = frame_of(concept) if concept not in (None, "") else None
    return got if got in (None, "promotion", "prevention") else "neither"


def frame_disagreements(rows: dict[str, dict], last: dict[str, dict]) -> set[str]:
    """The question-concept cards (PR-CONCEPT) on which the coder's concept and Gemini's (the pass the sheet was
    drawn from) fall in different frames; a card the coder gave no concept (a ticked box) is none."""
    out = set()
    for path, row in rows.items():
        ref = parse_path(path)
        if ref.unit != "turn" or ref.sub != CONCEPT:
            continue
        mine, gemini = frame(coder_side(last.get(path))["value"]), frame(row.get("primary"))
        if mine and gemini and mine != gemini:
            out.add(path)
    return out


def _video_task(run: Path, video_id: str, entry: dict, last: dict[str, dict], books: PanelBooks, labels: Labels,
                coder: str, file_name: str | None) -> tuple[dict, dict] | None:
    """A video's task, or None when it holds nothing to discuss; with what it holds and why. A card is selected for a
    structural flag (the coder's latest answer) or a frame disagreement (`frame_disagreements`); a selected card brings
    every card of its question-answer pair, for context (for a structural flag, every pair holding the flagged turn)."""
    rows = {row["path"]: row for row in entry["rows"]}
    turns = flagged_turns(last)
    framed = frame_disagreements(rows, last)
    if not turns and not framed:
        return None
    full = video_model(run / video_id, entry, books, labels, with_model=True)
    qa = {c["path"]: qa_pair_of(rows[c["path"]]) for sec in full["sections"] for c in sec["cards"]
          if parse_path(c["path"]).unit == "turn"}
    why: dict[tuple, set] = {}
    for path, pair in qa.items():
        if set(pair) & turns:
            why.setdefault(pair, set()).add(STRUCTURAL_FLAG)
        if path in framed:
            why.setdefault(pair, set()).add(FRAME_DISAGREEMENT)
    cards = []
    for sec in full["sections"]:
        for c in sec["cards"]:
            if qa.get(c["path"]) not in why:
                continue
            model = c.pop("model")
            mine = coder_side(last.get(c["path"]))
            reasons = [r for r, on in ((STRUCTURAL_FLAG, mine["resolution"] == STRUCTURAL),
                                       (FRAME_DISAGREEMENT, c["path"] in framed)) if on]
            c["discuss"] = {"flagged": mine["resolution"] == STRUCTURAL, "reasons": reasons, "coder": mine,
                            "gemini_value": _value(model.get("answer")), "gemini_other": _value(model.get("second")),
                            "context": video_id}
            if parse_path(c["path"]).sub == CONCEPT:
                c["discuss"]["frames"] = {"coder": frame(mine["value"]), "gemini": frame(model.get("answer")),
                                          "gemini_other": frame(model.get("second"))}
            cards.append(c)
    if not cards:
        return None
    task = {"video_id": video_id, "task": "video",
            "title": f"{short_title(full['title'])} — questions and answers to discuss",
            "short_title": short_title(full["title"]), "file_name": file_name or full["file_name"],
            "file_alts": full["file_alts"], "actors": full["actors"], "actors_first": full["actors_first"],
            # the cards play their own turns; the prepared window is not what is discussed here
            "window": "",
            "sections": [{"key": FLAGGED_SECTION, "title": "Questions and answers to discuss",
                          "intro": FLAGGED_INTRO.format(coder=coder), "sheet": "video", "cards": cards}]}
    held = {"video_id": video_id, "pass": entry.get("pass") or "primary",
            "flagged_paths": sorted(p for p, l in last.items() if l.get("resolution") == STRUCTURAL
                                    and parse_path(p).unit == "turn"),
            "frame_paths": sorted(framed),
            "pairs": [{"pair": list(p), "selected_for": sorted(r)} for p, r in sorted(why.items())],
            # each card and why it is on the page: its own structural flag or frame disagreement, else the context of
            # its pair
            "cards": [{"path": c["path"], "selected_for": c["discuss"]["reasons"] or [CONTEXT]} for c in cards]}
    return task, held


def _signed(x: float) -> str:
    return f"{x:+.2f}"


def discussion_page_model(run: Path, session: dict, coder: str, books: PanelBooks, labels: Labels,
                          logs: dict[str, list[dict]], pair_lines: list[dict], n_pairs: int, seed: int,
                          scores: dict[str, dict[str, float]] | None = None) -> tuple[dict, dict]:
    """(the page model, the manifest's content) of the discussion page for `coder`. `session` is the clip session's
    manifest (videos, coder order, design, files, pair rows); `logs` {video_id: its review_log.jsonl lines};
    `pair_lines` the pairs log; `scores` the model's rank scores ({item: {video: score}}) when at hand."""
    hints = {**DEFAULT_HINTS, **labels.hints}
    files, windows = _session_files(session)
    order = (session.get("order") or {}).get(coder) or session.get("videos") or []
    order = [*order, *sorted(set(session.get("videos") or []) - set(order))]
    tasks, held = [], []
    for video_id in order:
        export = run / video_id / "review" / MANIFEST
        entry = ((_read(export) if export.exists() else {}).get("videos") or {}).get(video_id)
        last = latest_lines(logs.get(video_id) or [], coder)
        if entry is None or not last:
            continue
        got = _video_task(run, video_id, entry, last, books, labels, coder, files.get(video_id))
        if got:
            tasks.append(got[0])
            held.append(got[1])
    design = session.get("design_id")
    drawn, how = draw_comparisons(pair_lines, coder, design, n_pairs, seed, list(books.stage_a_ids()))
    rows = {(r["batch_id"], r["code"]): r for r in session["rows"] if r["field_class"] == "pair"}
    pairs = {p["batch_id"]: p for p in session["pairs"]}
    titles = {t["video_id"]: t["short_title"] for t in tasks}
    for video_id in session.get("videos") or []:
        if video_id not in titles:
            titles[video_id] = short_title(Path(files.get(video_id) or video_id).stem)
    shown = lambda row: {**row, "clock_start": "", "clock_end": ""}     # noqa: E731  (the window button locates them)
    for line in drawn:
        row = rows.get((line["batch_id"], line["code"]))
        if row is None:
            raise PageError(f"pair {line['batch_id']}/{line['code']} of the pairs log is not a row of the live design")
        p = pairs[line["batch_id"]]
        card = _card(shown(row), row["row"], None, {}, books, labels)
        own = (scores or {}).get(line["code"]) or {}
        both = (f"rank scores: left {_signed(own[p['left']])}, right {_signed(own[p['right']])}; "
                if p["left"] in own and p["right"] in own else "")
        mine = {"value": line["value"], "resolution": "", "unsure": False, "summary_shown": False,
                "note": _clean(line.get("note")), "close": bool(line.get("close")), "answered": True,
                "cell": " ".join(x for x in (line["value"], review_clips.CLOSE if line.get("close") else "") if x)}
        card["discuss"] = {"flagged": False, "coder": mine, "gemini_value": line.get("model"), "gemini_other": None,
                           "gemini_detail": f"{both}gap {float(line.get('gap') or 0):.2f}",
                           "context": f"pair {p['batch_id']}: left {p['left']}, right {p['right']}"}
        pair = {**_pair_entry(p, files, windows), "left_title": titles.get(p["left"], p["left"]),
                "right_title": titles.get(p["right"], p["right"])}
        tasks.append({"video_id": f"{COMPARE_PREFIX}{p['batch_id']}-{line['code']}", "task": "compare",
                      "title": f"Compare: {pair['left_title']} vs {pair['right_title']} "
                               f"({line['code'].replace('_', ' ')})",
                      "file_name": "", "actors": {}, "actors_first": {}, "window": "", "pair": pair,
                      "videos": [p["left"], p["right"]],
                      "sections": [{"key": "compare", "title": labels.sections.get("compare") or DEFAULT_SECTIONS["compare"],
                                    "intro": COMPARE_INTRO.format(coder=coder), "sheet": "session",
                                    "cards": [card]}]})
    sheet = SHEET.format(coder=coder)
    page_hints = {**{k: hints[k] for k in TASK_HINTS}, "pitch_intro": hints["protocol_for_coders"]}
    model = {"coder": coder, "run": run.name, "release_id": session.get("release_id") or load_release().release_id,
             # the coder's sheet's columns, then the coder's answer and Gemini's two passes
             "columns": [*SHEET_COLUMNS, *EXTRA_COLUMNS], "order_seed": None, "hints": page_hints, "videos": tasks,
             "mode": DISCUSSION, "heading": "Discussion", "list_title": "Cases to discuss",
             "banner": BANNER.format(coder=coder, sheet=sheet), "sheet_name": sheet, "codebook": []}
    manifest = {"coder": coder, "run": run.name, "design_id": design, "seed": seed, "pairs_requested": n_pairs,
                "sheet": sheet, "not_validation": ("this page's answers never enter the agreement figures; no import "
                                                   "reads its sheet, and the coder's imported answers are unchanged"),
                "passes": {h["video_id"]: h["pass"] for h in held},
                "videos": held,
                "comparisons": {**how, "selected": [{k: line.get(k) for k in ("batch_id", "code", "left", "right",
                                                                               "value", "model", "gap", "close",
                                                                               "note")} for line in drawn]}}
    return model, manifest
