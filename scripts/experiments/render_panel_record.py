#!/usr/bin/env python3
"""Write a human-readable Markdown view of one pass of a v3 panel record.

Reads a results/<pass>.json and, when they exist beside it, ../experiment.json (video id and
release) and <pass>_validation.json. A concept of `other` is shown with the name the model gave it, and
all of them are listed after the timeline. Labels are shown in the current codebook's spelling. Offline; changes nothing but the output file. The view is
for reading a record, not for analysis: every value in it is model-coded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.instrument import load_release  # noqa: E402
from pipeline.schemas.stage_b_panel import seconds_to_clock  # noqa: E402
from pipeline.stage_b.panel_books import (  # noqa: E402
    CONCEPT_CODE_OF, frame_of, is_other, load_panel_books, other_concepts, other_name,
)

MISSING = "∅"
DASH = "—"
UNNAMED = "(unnamed)"  # an `other` concept or topic for which the record gives no name
TURN_HEADER = ["turn", "time", "who", "function", "codes", "summary"]
LINK = "↩"  # function cell: the turns this one responds to


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _row(cells: list) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _table(header: list[str], rows: list[list]) -> list[str]:
    return [_row(header), "|" + "|".join("---" for _ in header) + "|"] + [_row(r) for r in rows]


def _clock(seconds) -> str:
    return DASH if seconds is None else seconds_to_clock(seconds)


def _runs(turns: list[dict]) -> list[list[dict]]:
    """Consecutive turns of the same exchange (or of none), in turn order."""
    runs: list[list[dict]] = []
    for t in turns:
        if runs and runs[-1][0]["exchange_id"] == t["exchange_id"]:
            runs[-1].append(t)
        else:
            runs.append([t])
    return runs


def render(result: dict, experiment: dict | None = None, validation: dict | None = None) -> str:
    record = result["record"]
    names = {a["actor_id"]: a["label"] for a in record["actors"]}
    turns = record["turns"]
    turn_start = {}
    for t in turns:
        turn_start.setdefault(t["turn_id"], t["start_s"])
    exchanges = {}
    for e in record["exchanges"]:
        exchanges.setdefault(e["exchange_id"], e)

    def who(actor: str) -> str:
        return f"{actor} ({names[actor]})" if actor in names else actor

    def decision_row(d: dict) -> list:
        move = " ".join(x for x in ("**DECISION**", d["kind"], d["offer_id"]) if x)
        turn_id = "" if d["turn_id"] is None else d["turn_id"]
        return [turn_id, _clock(d["time_s"]), who(d["actor"]), move, "",
                ". ".join(x for x in (d["terms"], d["note"]) if x)]

    decisions_at: dict[int, list[dict]] = {}
    loose = []
    for d in record["decisions"]:
        if d["turn_id"] in turn_start:
            decisions_at.setdefault(d["turn_id"], []).append(d)
        else:
            loose.append(d)

    def code_text(t: dict, c: dict) -> str:
        if c["value"] is None:
            return f"{c['code']}={MISSING}({c['missing']})"
        text = f"{c['code']}={c['value']}"
        if c["code"] != CONCEPT_CODE_OF.get(t["function"]):
            return text
        # a concept is followed by the name given to an `other` and by the frame derived from it
        if is_other(c["value"]):
            name = other_name(t["other_concept"])
            text += f' ("{name}")' if name else f" {UNNAMED}"
        frame = frame_of(c["value"])
        return text + (f" ({frame})" if frame else "")

    def turn_rows(group: list[dict]) -> list[list]:
        rows = []
        for t in group:
            speaker = who(t["speaker"]) + (" → " + ", ".join(t["addressed"]) if t["addressed"] else "")
            codes = " ".join(code_text(t, c) for c in t["codes"])
            function = t["function"]
            if t["responds_to"]:
                function += f" {LINK} " + ", ".join(str(i) for i in t["responds_to"])
            rows.append([t["turn_id"], _clock(t["start_s"]), speaker, function, codes, t["summary"]])
            rows += [decision_row(d) for d in decisions_at.get(t["turn_id"], [])]
        return rows

    def exchange_title(exchange_id: int) -> str:
        e = exchanges.get(exchange_id)
        if e is None:
            return f"Exchange {exchange_id} · (not recorded)"
        parts = [f"Exchange {exchange_id}", e["kind"]]
        if e["topic"] != "NONE":
            parts.append(e["topic"])
        if e["label"]:
            parts.append(f'"{e["label"]}"')
        return " · ".join(parts)

    video_id = (experiment or {}).get("video_id")
    if video_id is None:
        parts = str(result.get("unit_key") or "").split("__")
        video_id = parts[1] if len(parts) == 3 else "unknown"
    window = result.get("prepared_window")
    if window is None:
        window = [f"{_clock(s['start_s'])}-{_clock(s['end_s'])}" for s in record["prepared_spans"]]
    pass_name = result.get("pass", "unknown")

    L = [f"# Panel record: {video_id} ({pass_name})", "",
         f"- Video: {video_id}", f"- Release: {(experiment or {}).get('release_id', 'unknown')}",
         f"- Pass: {pass_name}", f"- Prepared window: {', '.join(window) or 'none'}",
         "- Every value below is model-coded and not human-validated.", "",
         "## Actors", ""]
    L += _table(["id", "role", "name"], [[a["actor_id"], a["role"], a["label"]] for a in record["actors"]])
    L += ["", "## Stage A (prepared presentation only)", ""]
    L += _table(["code", "value", "moment", "note"],
                [[i["code"], "uncodable" if i["uncodable"] else i["value"], _clock(i["moment_s"]), i["note"]]
                 for i in record["stage_a"]])
    if "prepared" in record:     # the prepared presentation codes, since v3-dev9
        items = [f"{c['code'].removeprefix('PRES-')} "
                 + (c["value"] if c["value"] is not None else f"{MISSING}({c['missing']})") for c in record["prepared"]]
        L += ["", "prepared: " + " · ".join(items)]
    v = record["venture"]
    ask = ""
    if "ask_missing" in v:       # the ask, since v3-dev9
        amount, equity = v.get("ask_amount_usd"), v.get("ask_equity_pct")
        ask = "; ask: " + (f"${amount:,} for {equity:g}%" if amount is not None and equity is not None
                           else v["ask_missing"])
    L += ["", "## Venture", "", f"- {v['description']}",
          f"- Industry: {', '.join([v['primary_industry']] + v['secondary_industries'])}",
          f"- Product form: {v['product_form']}; customer type: {v['customer_type']}; "
          f"category: {v['product_category']}{ask}",
          f"- Presenting team: {v['founder_gender']}",
          f"- Earliest support: {_clock(v['earliest_s'])}. {v['note']}".rstrip(),
          "", "## Timeline", ""]

    members: dict[int, list[dict]] = {}
    for t in turns:
        if t["exchange_id"] is not None:
            members.setdefault(t["exchange_id"], []).append(t)
    shown: set[int] = set()
    for run in _runs(turns):
        exchange_id = run[0]["exchange_id"]
        if exchange_id is None:
            L += ["### (no exchange)", ""] + _table(TURN_HEADER, turn_rows(run)) + [""]
        elif exchange_id not in shown:
            shown.add(exchange_id)
            L += [f"### {exchange_title(exchange_id)}", ""]
            L += _table(TURN_HEADER, turn_rows(members[exchange_id])) + [""]
        else:
            label = exchanges.get(exchange_id, {}).get("label")
            name = f"Exchange {exchange_id}" + (f' ("{label}")' if label else "")
            count = f"{len(run)} turn{'s' if len(run) != 1 else ''}"
            L += [f"↪ {_clock(run[0]['start_s'])}: {name} resumes here for {count}, shown above.", ""]
    if loose:
        L += ["### Decisions without a turn", ""] + _table(TURN_HEADER, [decision_row(d) for d in loose]) + [""]

    L += ["## Other concepts named", ""]
    named = other_concepts(record)
    L += [f"- {o['clock'] or DASH} {o['where']} {o['id']} {o['code']}: {o['name'] or UNNAMED}" for o in named]
    L += [""] if named else ["None.", ""]

    L += ["## Claim conflicts", ""]
    if not record["claim_conflicts"]:
        L += ["None recorded.", ""]
    for c in record["claim_conflicts"]:
        def at(turn_id):
            return f"turn {turn_id} ({_clock(turn_start.get(turn_id))})"
        if c["reconciled"]:
            end = "reconciled" + (f" in {at(c['reconciled_turn'])}" if c["reconciled_turn"] is not None else "")
        else:
            end = "not reconciled"
        L.append(f"- {c['metric']}: {at(c['turn_a'])} against {at(c['turn_b'])}; {end}. {c['note']}".rstrip())
    if record["claim_conflicts"]:
        L.append("")
    if record["notes"]:
        L += ["## Notes", "", record["notes"], ""]
    if validation is not None:
        L += ["## Validation", ""]
        for title, key in (("Errors", "errors"), ("Flags", "flags")):
            items = validation.get(key) or []
            L += [f"{title} ({len(items)})", ""] + [f"- {x}" for x in items] + ([""] if items else [])
    return "\n".join(L).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", type=Path, required=True, help="a results/<pass>.json of a panel run")
    ap.add_argument("--out", type=Path, required=True, help="the Markdown file to write")
    args = ap.parse_args(argv)
    result = _read(args.record)
    if not isinstance(result, dict) or not isinstance(result.get("record"), dict):
        ap.error(f"{args.record} is not a readable panel result (no record)")
    result = {**result, "record": load_panel_books(load_release()).canonical_structure(result["record"])}
    experiment = _read(args.record.parent.parent / "experiment.json")
    validation = _read(args.record.with_name(f"{args.record.stem}_validation.json"))
    text = render(result, experiment if isinstance(experiment, dict) else None,
                  validation if isinstance(validation, dict) else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
