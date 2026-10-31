"""Which review items go on a coder's sheet, and why.

Every item of the primary record is either flagged (something about it asks for a human look) or
left for a seeded random sample. The flagged rows are a biased set by construction; the sampled
rows are what an unbiased agreement estimate rests on: a draw, proportional to each field class's
share, from the fields that nothing flagged and whose unit both passes recorded (the fields of a
unit one pass lacks are judged as a unit, in a unit_presence row, and are not in that pool). The exchange-level sheet (select_exchange_sheet)
is a seeded random sample only: outcomes, venture, a few Stage A items, a few question-answer pairs and, on request,
a few substantive exchanges, so every row of it is an estimate. A flag-driven sheet is kept short: a unit one pass lacks
is one row, a validation message flags only the field it is about, fields of little value are left
off unless asked for, and the flagged rows can be capped. No I/O: records and reports in, rows out.
"""

from __future__ import annotations

import random
import re

from pipeline.schemas.stage_b_panel import PanelRecord, clock_to_seconds, seconds_to_clock
from pipeline.stage_b.panel_books import CONCEPT_CODES, PanelBooks, is_other
from pipeline.stage_b.panel_validate import SUBSTANTIVE
from pipeline.stage_b.replicate_agreement import AlignmentSettings, align
from pipeline.stage_b.review_items import (
    ABSENT, ASK, CUT, CUT_PATH, OUTCOME, OUTCOME_PATH, PRESENT, SHEET_CLASSES, UNIT_PRESENCE, VENTURE_CODES, PathRef,
    any_offer, canonical_full, cut_position, first_decision_clock, flatten, main_path, parse_path, presence_path,
    read_value, replicate_only_path,
)

FLAGGED, SAMPLED = "flagged", "sampled"
DIFFER, NO_COUNTERPART, STAGE_A_GAP = "passes_differ", "no_counterpart", "stage_a_gap"
OTHER_CONCEPT, MAPPED_BY_ORDER = "other_concept", "actor_mapped_by_order"
VALIDATION = "validation:"          # followed by the tag of the message, e.g. validation:flag.orphan_turn
SOFT = "flag."                      # the tag of a soft flag starts so; any other tag is a validator error
STAGE_A_MIN_GAP = 2
UNALIGNED = "~"                     # prefix of a replicate id that has no counterpart in the primary
NO_VALUE = "NONE"                   # the codebooks' residual: no topic, unknown
# the order a coder can work in when time is short
PRIORITY_SAMPLED, PRIORITY_PRESENCE, PRIORITY_ERROR, PRIORITY_FLAGGED = 1, 2, 3, 4
# free text is shown for both passes but two wordings of one thing are not a disagreement
_FREE_TEXT = {("actor", "label"), ("exchange", "label"), ("turn", "other_concept")}
_ID_LISTS = {("turn", "responds_to")}
_UNITS = ("turn", "exchange", "decision")
_TAG = re.compile(r"^\[([^\]]+)\]\s*")
_NAMED = re.compile(r"\b(turn|exchange|decision) (\d+)\b")
_SUBJECT = re.compile(r"^(turn|exchange|decision) (\d+)\b")
_CODE = re.compile(r"\b([A-Z]{2,}-[A-Z]+)\b")
# validation messages that record a repair or a fact of the source; nothing in them asks for a human look
INFORMATIONAL = frozenset({"flag.value_case", "flag.value_with_missing", "flag.gap", "flag.unknown_addressee",
                           "flag.single_turn_exchange", "flag.venture_unknown"})
# the field a validation message is about, by its tag; a tag not listed here (or whose field is not on
# the sheet) falls back to the main field of the unit the message names
_TAG_FIELD = {
    "turn.function": "function", "flag.turn_role": "function", "flag.kind_function_mismatch": "function",
    "turn.exchange": "exchange", "flag.orphan_turn": "exchange", "flag.demonstration_without_exchange": "exchange",
    "turn.link": "responds_to", "flag.answer_link": "responds_to", "flag.unlinked_response": "responds_to",
    "flag.unanswered_question": "responds_to",
    "turn.other_concept": "other_concept",
    "exchange.kind": "kind", "exchange.topic": "topic",
}
_WINDOW_TAGS = ("flag.no_prepared", "flag.no_presents_turn")


def _empty(value: str | None) -> bool:
    """No value: nothing, blank, or the residual NONE."""
    return value is None or value.strip() in ("", NO_VALUE)


class _Counterparts:
    """Where each unit of the primary sits in the replicate, read off the agreement report."""

    def __init__(self, primary: dict, replicate: dict, agreement: dict):
        self.replicate = replicate
        actors = agreement["actors"]
        self.actor_r2p: dict[str, str] = dict(actors["mapping"])
        self.actor_p2r = {p: r for r, p in self.actor_r2p.items()}
        self.by_order = {self.actor_r2p[r] for r, how in actors["mapped_by"].items() if how == "order"}
        turns = agreement["turns"]
        pairs = [*turns["matched"], *turns["function_mismatched"]]
        self.unit_p2r = {
            "turn": {int(p): int(r) for p, r in pairs},
            "exchange": {int(m[0]): int(m[1]) for m in agreement["exchanges"]["matched"]},
            "decision": {int(p): int(r) for p, r in agreement["decisions"]["matched"]},
        }
        self.unit_r2p = {unit: {r: p for p, r in table.items()} for unit, table in self.unit_p2r.items()}
        settings = agreement.get("settings") or {}
        base = AlignmentSettings()
        self.event_tolerance_s = float(settings.get("event_tolerance_s", base.event_tolerance_s))
        self.decision_tolerance_s = float(settings.get("decision_tolerance_s", base.decision_tolerance_s))
        # the two passes name offers independently; an offer id is carried over by its matched decisions
        p_offer = {d["decision_id"]: d["offer_id"] for d in primary["decisions"]}
        r_offer = {d["decision_id"]: d["offer_id"] for d in replicate["decisions"]}
        self.offer_r2p: dict[str, str] = {}
        for p, r in sorted(self.unit_p2r["decision"].items()):
            if p_offer.get(p) is not None and r_offer.get(r) is not None:
                self.offer_r2p.setdefault(r_offer[r], p_offer[p])
        self.stage_a = {i["code"] for i in replicate["stage_a"]}

    def has(self, unit: str, unit_id: int) -> bool:
        """Whether a turn, exchange or decision of the primary has a counterpart in the replicate."""
        return unit_id in self.unit_p2r[unit]

    def _path(self, ref: PathRef) -> str | None:
        """The replicate path of the unit a primary path addresses; None when the unit has no counterpart."""
        if ref.unit == "prepared_window":
            return "prepared_window"
        if ref.unit == "venture":
            return f"venture:{ref.unit_id}"
        if ref.unit == "stage_a":
            return f"stage_a:{ref.unit_id}" if ref.unit_id in self.stage_a else None
        if ref.unit == "prep_pres":
            return f"prep_pres:{ref.unit_id}"
        if ref.unit == "actor":
            other = self.actor_p2r.get(ref.unit_id)
            return None if other is None else f"actor:{other}/{ref.field}"
        other = self.unit_p2r[ref.unit].get(ref.unit_id)
        if other is None:
            return None
        if ref.field == "code":
            return f"{ref.unit}:{other}/code:{ref.sub}"
        return f"{ref.unit}:{other}/{ref.field}"

    def _to_primary_ids(self, ref: PathRef, value: str) -> str:
        def actor(a: str) -> str:
            return self.actor_r2p.get(a, UNALIGNED + a)

        def unit(kind: str, text: str) -> str:
            found = self.unit_r2p[kind].get(int(text))
            return f"{UNALIGNED}{text}" if found is None else str(found)

        key = (ref.unit, ref.field)
        if key == ("turn", "exchange"):
            return unit("exchange", value)
        if key == ("turn", "responds_to"):
            return ",".join(unit("turn", part) for part in value.split(","))
        if key == ("decision", "actor"):
            return actor(value)
        if key == ("decision", "offer_id"):
            return self.offer_r2p.get(value, UNALIGNED + value)
        return value

    def value(self, ref: PathRef) -> tuple[bool, str | None]:
        """(the unit has a counterpart, the replicate's value there with its ids rewritten as primary ids)."""
        path = self._path(ref)
        if path is None:
            return False, None
        try:
            value = read_value(self.replicate, path)
        except ValueError:      # the unit is there but not this part of it (an investor the replicate does not list)
            return True, None
        return True, None if value is None else self._to_primary_ids(ref, value)

    def differ(self, ref: PathRef, a: str | None, b: str | None) -> bool:
        key = (ref.unit, ref.field)
        if key in _FREE_TEXT or (_empty(a) and _empty(b)):
            return False
        if _empty(a) or _empty(b):
            return True
        if ref.unit == "stage_a":
            return abs(int(a) - int(b)) >= STAGE_A_MIN_GAP
        if key == ("decision", "time"):
            return abs(clock_to_seconds(a) - clock_to_seconds(b)) > self.decision_tolerance_s
        if ref.unit == "prepared_window":
            ends_a, ends_b = _boundaries(a), _boundaries(b)
            return len(ends_a) != len(ends_b) or any(
                abs(x - y) > self.event_tolerance_s for x, y in zip(ends_a, ends_b))
        if key in _ID_LISTS:
            return set(a.split(",")) != set(b.split(","))
        return a != b


def _boundaries(window: str) -> list[float]:
    return [clock_to_seconds(end) for span in window.split(";") for end in span.split("-")]


def _is_error(reason: str) -> bool:
    return reason.startswith(VALIDATION) and not reason.startswith(VALIDATION + SOFT)


def priority(row: dict) -> int:
    """1 = sampled, 2 = unit_presence, 3 = a validator error points at it, 4 = any other flagged row."""
    if not row["reasons"]:
        return PRIORITY_SAMPLED
    if row["field_class"] == UNIT_PRESENCE:
        return PRIORITY_PRESENCE
    return PRIORITY_ERROR if any(_is_error(r) for r in row["reasons"]) else PRIORITY_FLAGGED


def _message_target(message: str, paths: set[str]) -> tuple[str, list[str]]:
    """(tag, paths the message flags): the one field it is about, else the main field of the unit it names."""
    found = _TAG.match(message)
    if not found or found.group(1) in INFORMATIONAL:
        return "", []
    tag, body = found.group(1), message[found.end():]
    if tag == "flag.multiple_negotiations":
        wanted = [main_path("exchange", int(n)) for n in re.findall(r"\d+", body.partition("]")[0])]
        return tag, [path for path in wanted if path in paths]
    subject = _SUBJECT.match(body)
    if subject is None:
        wanted = None
        if tag.startswith("stage_a.") and tag != "stage_a.codes":
            wanted = f"stage_a:{body.partition(':')[0].split(' ')[0]}"
        elif tag == "venture.value":
            wanted = f"venture:{body.partition(':')[0]}"
        elif tag in _WINDOW_TAGS or (tag == "span.time" and body.startswith("prepared")):
            wanted = "prepared_window"
        else:
            named = _NAMED.search(body)
            if named:
                wanted = main_path(named.group(1), int(named.group(2)))
        return tag, [wanted] if wanted in paths else []
    unit, unit_id = subject.group(1), int(subject.group(2))
    base = f"{unit}:{unit_id}"
    field = None
    if unit == "turn" and tag in ("turn.codes", "flag.other_concept"):
        code = _CODE.search(body)
        field = f"code:{code.group(1)}" if code else None
    elif unit == "turn" and tag == "flag.other_concept_unused":
        field = next((f"code:{c}" for c in sorted(CONCEPT_CODES) if f"{base}/code:{c}" in paths), None)
    elif unit == "exchange" and tag == "flag.other_concept":
        field = "topic"
    elif unit != "decision":        # every message about a decision is about what kind of move it is
        field = _TAG_FIELD.get(tag)
    for path in (f"{base}/{field}" if field else None, main_path(unit, unit_id)):
        if path in paths:
            return tag, [path]
    return tag, []


def _validation_targets(validation: dict, paths: set[str]) -> dict[str, list[str]]:
    """path -> tags of the validation messages that are about that field (informational messages flag nothing)."""
    out: dict[str, list[str]] = {}
    for message in [*(validation.get("errors") or []), *(validation.get("flags") or [])]:
        tag, targets = _message_target(message, paths)
        for path in targets:
            tags = out.setdefault(path, [])
            if tag not in tags:
                tags.append(tag)
    return out


def _other_units(primary: dict) -> set[tuple[str, int]]:
    """The turns whose concept is `other` and the exchanges whose topic is."""
    out = set()
    for t in primary["turns"]:
        if any(c["code"] in CONCEPT_CODES and is_other(c["value"]) for c in t["codes"]):
            out.add(("turn", t["turn_id"]))
    out |= {("exchange", e["exchange_id"]) for e in primary["exchanges"] if is_other(e["topic"])}
    return out


def _is_other_item(ref: PathRef, value: str | None, others: set[tuple[str, int]]) -> bool:
    if (ref.unit, ref.unit_id) not in others:
        return False
    if ref.unit == "turn":
        return ref.field == "other_concept" or (ref.field == "code" and ref.sub in CONCEPT_CODES and is_other(value))
    return ref.field in ("topic", "label")


def _trivial_links(primary: dict) -> set[int]:
    """Answers whose only link is to the question right before them: the link nobody would code otherwise."""
    out = set()
    for before, t in zip(primary["turns"], primary["turns"][1:]):
        if t["function"] == "answer" and before["function"] == "question" and t["responds_to"] == [before["turn_id"]]:
            out.add(t["turn_id"])
    return out


class _Sheet:
    """Which items of the primary can be on a sheet at all."""

    def __init__(self, primary: dict, field_classes: tuple[str, ...] | None):
        self.asked = field_classes
        self.others = _other_units(primary)
        self.trivial = _trivial_links(primary)
        self.substantive = {e["exchange_id"] for e in primary["exchanges"] if e["kind"] == SUBSTANTIVE}

    def takes(self, ref: PathRef, cls: str) -> bool:
        if self.asked is not None and cls not in self.asked:
            return False
        # a topic is a judgment only of a substantive exchange; elsewhere it is NONE by rule
        if ref.unit == "exchange" and ref.field == "topic" and ref.unit_id not in self.substantive:
            return False
        return self.asked is not None or not self._low_value(ref)      # a class asked for comes whole

    def _low_value(self, ref: PathRef) -> bool:
        if ref.unit == "actor":
            return True
        named = (ref.unit, ref.unit_id) in self.others      # there the free text names the `other` matter
        if (ref.unit, ref.field) in (("exchange", "label"), ("turn", "other_concept")):
            return not named
        return (ref.unit, ref.field) == ("turn", "responds_to") and ref.unit_id in self.trivial


def _sample(paths_by_class: dict[str, list[str]], budget: int, rng: random.Random) -> set[str]:
    """`budget` paths: one from each class first (from as many classes as the budget covers), the rest at random."""
    classes = [c for c in SHEET_CLASSES if paths_by_class.get(c)]
    everything = [path for c in classes for path in sorted(paths_by_class[c])]
    if budget >= len(everything):
        return set(everything)
    if budget <= 0:
        return set()
    first = classes if budget >= len(classes) else rng.sample(classes, budget)
    chosen = {rng.choice(sorted(paths_by_class[c])) for c in first}
    rest = [path for path in everything if path not in chosen]
    return chosen | set(rng.sample(rest, budget - len(chosen)))


UNFLAGGED_NOTE = (
    "The sampled rows are a seeded draw from the unflagged pool: the fields of the primary that nothing flagged "
    "and whose unit both passes recorded. Each field class gets its share of the draw, proportional to its "
    "population (no class is forced onto the sheet), and rows are drawn at random within a class; population gives "
    "the pool per field class and on_sheet the rows drawn. The sample estimates agreement on such fields. It says "
    "nothing about the flagged fields, and not about the fields of units one pass lacks (excluded_no_counterpart), "
    "which are judged as units.")


def _proportional(paths_by_class: dict[str, list[str]], budget: int, rng: random.Random) -> set[str]:
    """`budget` paths, each class getting its share of the population (largest remainders first, ties in the
    order of SHEET_CLASSES) and its rows drawn at random; everything when the budget covers it."""
    classes = [c for c in SHEET_CLASSES if paths_by_class.get(c)]
    sizes = {c: len(set(paths_by_class[c])) for c in classes}
    total = sum(sizes.values())
    if budget >= total:
        return {path for c in classes for path in paths_by_class[c]}
    if budget <= 0:
        return set()
    quota = {c: budget * sizes[c] / total for c in classes}
    take = {c: int(quota[c]) for c in classes}
    by_remainder = sorted(classes, key=lambda c: -(quota[c] - take[c]))     # stable: ties keep the class order
    for c in by_remainder[:budget - sum(take.values())]:
        take[c] += 1
    return {path for c in classes for path in rng.sample(sorted(set(paths_by_class[c])), take[c])}


def _count_by_class(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["field_class"]] = counts.get(row["field_class"], 0) + 1
    return {c: counts[c] for c in SHEET_CLASSES if c in counts}


def _by_class(rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        out.setdefault(row["field_class"], []).append(row["path"])
    return out


def _presence_rows(items: dict, primary: dict, other: _Counterparts) -> list[dict]:
    """One row for every turn, exchange and decision of the primary that the replicate lacks."""
    rows = []
    for unit, key, id_field in (("turn", "turns", "turn_id"), ("exchange", "exchanges", "exchange_id"),
                                ("decision", "decisions", "decision_id")):
        for unit_id in dict.fromkeys(u[id_field] for u in primary[key]):
            if other.has(unit, unit_id):
                continue
            item = items[main_path(unit, unit_id)]
            rows.append({"path": presence_path(unit, unit_id), "field_class": UNIT_PRESENCE,
                         "clock_start": item.clock_start, "clock_end": item.clock_end, "context": item.context,
                         "primary": PRESENT, "replicate": ABSENT, "reasons": [NO_COUNTERPART]})
    return rows


def _replicate_only_rows(primary: dict, other: _Counterparts) -> list[dict]:
    """One row for every turn, exchange and decision only the replicate recorded, in the primary's actor ids."""
    replicate = other.replicate
    names: dict[str, str] = {}
    for a in primary["actors"]:
        names.setdefault(a["actor_id"], a["label"])

    def who(actor: str) -> str:
        mapped = other.actor_r2p.get(actor)
        return f"{mapped} ({names[mapped]})" if mapped in names else "an actor the primary does not have"

    def clock(seconds: float | None) -> str | None:
        return None if seconds is None else seconds_to_clock(seconds)

    def text(*parts) -> str:
        return " ".join(", ".join(str(p) for p in parts if p).split())

    members: dict[int, list[dict]] = {}
    for t in replicate["turns"]:
        if t["exchange_id"] is not None:
            members.setdefault(t["exchange_id"], []).append(t)
    found: list[tuple[str, float | None, float | None, str]] = []
    for t in replicate["turns"]:
        if t["turn_id"] not in other.unit_r2p["turn"]:
            found.append(("turn", t["start_s"], t["end_s"], f"turn by {who(t['speaker'])}: "
                          + text(t["function"], t["summary"])))
    for e in replicate["exchanges"]:
        if e["exchange_id"] not in other.unit_r2p["exchange"]:
            turns = members.get(e["exchange_id"], [])
            found.append(("exchange", min((t["start_s"] for t in turns), default=None),
                          max((t["end_s"] for t in turns), default=None),
                          f"exchange of {len(turns)} turn{'' if len(turns) == 1 else 's'}: "
                          + text(e["kind"], e["label"])))
    for x in replicate["decisions"]:
        if x["decision_id"] not in other.unit_r2p["decision"]:
            found.append(("decision", x["time_s"], None, f"decision by {who(x['actor'])}: "
                          + text(x["kind"], x["terms"])))
    rows, used = [], set()
    for unit, start, end, described in found:
        path = replicate_only_path(unit, clock(start))
        n = 1
        while path in used:     # two units of one kind starting on the same second
            n += 1
            path = f"{replicate_only_path(unit, clock(start))}#{n}"
        used.add(path)
        rows.append({"path": path, "field_class": UNIT_PRESENCE, "clock_start": clock(start), "clock_end": clock(end),
                     "context": f"only in the replicate, {described}", "primary": ABSENT, "replicate": PRESENT,
                     "reasons": [NO_COUNTERPART]})
    return rows


def _cap(flagged: list[dict], max_flagged: int | None, seed: int) -> list[dict]:
    """The flagged rows a sheet keeps: all of them, or under a cap every unit_presence row and every row a
    validator error points at, then a seeded draw of the others that reaches each field class first."""
    if max_flagged is None or len(flagged) <= max_flagged:
        return flagged
    must = [row for row in flagged if row["priority"] in (PRIORITY_PRESENCE, PRIORITY_ERROR)]
    rest = [row for row in flagged if row["priority"] == PRIORITY_FLAGGED]
    drawn = _sample(_by_class(rest), max(0, max_flagged - len(must)), random.Random(f"{seed}:flagged"))
    return must + [row for row in rest if row["path"] in drawn]


def _tally(rows: list[dict]) -> dict:
    by_class: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        by_class[row["field_class"]] = by_class.get(row["field_class"], 0) + 1
        for reason in row["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    return {"n": len(rows), "by_field_class": {c: by_class[c] for c in SHEET_CLASSES if c in by_class},
            "by_reason": dict(sorted(by_reason.items()))}


def _aligned(primary: PanelRecord, replicate: PanelRecord | None, agreement: dict | None, books: PanelBooks):
    """(canonical primary, its dict, where its units sit in the replicate or None without one)."""
    primary = canonical_full(primary, books)
    p_dict = primary.model_dump(mode="json")
    other = None
    if replicate is not None:
        replicate = canonical_full(replicate, books)
        if agreement is None:
            agreement = align(primary, replicate, books)
        other = _Counterparts(p_dict, replicate.model_dump(mode="json"), agreement)
    return primary, p_dict, other


def _field_rows(primary: PanelRecord, p_dict: dict, other: _Counterparts | None, validation: dict,
                books: PanelBooks, field_classes: tuple[str, ...] | None) -> tuple[dict, list[dict]]:
    """(every item of the primary by path, a row with its reasons for each item a sheet can take)."""
    sheet = _Sheet(p_dict, field_classes)
    items = {item.path: item for item in flatten(primary, books)}
    candidates = [(item, ref) for item, ref in ((item, parse_path(item.path)) for item in items.values())
                  if sheet.takes(ref, item.field_class)]
    targets = _validation_targets(validation or {}, {item.path for item, _ in candidates})
    rows = []
    for item, ref in candidates:
        reasons: list[str] = []
        replicate_value = None
        if other is not None:
            found, replicate_value = other.value(ref)
            if not found:
                if ref.unit not in _UNITS:      # a missing turn, exchange or decision is one unit_presence row
                    reasons.append(NO_COUNTERPART)
            elif other.differ(ref, item.value, replicate_value):
                both_rated = item.value is not None and replicate_value is not None
                reasons.append(STAGE_A_GAP if ref.unit == "stage_a" and both_rated else DIFFER)
            if ref.unit == "actor" and ref.unit_id in other.by_order:
                reasons.append(MAPPED_BY_ORDER)
        reasons += [VALIDATION + tag for tag in targets.get(item.path, [])]
        if _is_other_item(ref, item.value, sheet.others):
            reasons.append(OTHER_CONCEPT)
        rows.append({"path": item.path, "field_class": item.field_class, "clock_start": item.clock_start,
                     "clock_end": item.clock_end, "context": item.context, "primary": item.value,
                     "replicate": replicate_value, "reasons": reasons})
    return items, rows


def _names(d: dict) -> dict[str, str]:
    names: dict[str, str] = {}
    for a in d["actors"]:
        names.setdefault(a["actor_id"], a["label"])
    return names


def _pitch_rows(p_dict: dict, other: _Counterparts | None, field_classes: tuple[str, ...] | None = None) -> list[dict]:
    """The pitch-level rows: the outcome (did any investor make an offer) and the cut (when is the first decision
    spoken), each read off the decisions of every pass; only the classes of `field_classes` when it is given.

    `blind_context` is what a blind sheet shows instead of the decisions.
    """
    investors = {a["actor_id"] for a in p_dict["actors"] if a["role"] == "investor"}
    own = sorted((x for x in p_dict["decisions"] if x["actor"] in investors), key=lambda x: x["time_s"])
    moves = ", ".join(f"{x['actor']} {x['kind']}@{seconds_to_clock(x['time_s'])}" for x in own) or "no decision recorded"
    context = f"the pitch: {moves}"
    tolerance = other.decision_tolerance_s if other is not None else AlignmentSettings().decision_tolerance_s

    def differ_outcome(a, b) -> bool:
        return b not in (None, a)

    def differ_cut(a, b) -> bool:
        if a is None or b is None:
            return a is not b
        return abs(clock_to_seconds(a) - clock_to_seconds(b)) > tolerance

    rows = []
    for path, cls, read, ask, differ in (
            (OUTCOME_PATH, OUTCOME, any_offer, "Did any investor make an offer? Watch to the end.", differ_outcome),
            (CUT_PATH, CUT, first_decision_clock, "When is the first decision spoken?", differ_cut)):
        if field_classes is not None and cls not in field_classes:
            continue
        value = read(p_dict)
        replicate_value = None if other is None else read(other.replicate)
        compared = other is not None and (cls == CUT or replicate_value is not None)
        rows.append({"path": path, "field_class": cls, "clock_start": None, "clock_end": None,
                     "context": context, "blind_context": f"the pitch: {ask}",
                     "primary": value, "replicate": replicate_value,
                     "reasons": [DIFFER] if compared and differ(value, replicate_value) else []})
    return rows


# what the full-video sheet draws per video; Stage A and the venture are coded in the clip session, not on this sheet.
# `exchanges` is 0 since 2026-10-21: no reported analysis reads the model's grouping of turns into exchanges (plan
# B2), so its topic cards would cost coder time and validate nothing used. The drawing code stays, and
# `--exchanges-per-video N` asks for N topic cards per video when a sheet is wanted with them.
EXCHANGE_DEFAULTS = {"stage_a": 0, "exchanges": 0, "qa": 3, "venture": 0}
# the pitch-level row every full-video sheet carries: the first decision. Since 2026-10-21 (the researcher's pilot
# coding) the sheet draws neither the outcome (pitch:any_offer: the two passes agree on it for 100 of 100 pitches and
# the outcome is a public fact of the show) nor the prepared-window check (the coder does not check the window); each
# is drawn only when --field-classes names its class (`outcome`, `prepared_window`), and old sheets with those rows
# still import and count.
WINDOW_ASK = "Is the prepared presentation where the page says it is?"
VENTURE_ALWAYS = "founder_gender"
# answer coherence (PREP-COH) joined on 2026-10-21 (plan C Task 7e): the strongest single feature on v3-dev9, not
# human-coded before
QA_CODES = (("question", "PR-CONCEPT"), ("question", "PR-SEV"), ("answer", "RS-RESP"), ("answer", "RS-EVID"),
            ("answer", "PREP-SUBS"), ("answer", "PREP-COH"))
ALL = "all"
MAX_QUESTIONS_SHOWN = 3


def _draw(rng: random.Random, population: list, k: int) -> list:
    """`k` of the population (all of it when it is smaller), in the population's order."""
    if k >= len(population):
        return list(population)
    chosen = set(rng.sample(range(len(population)), max(0, k)))
    return [x for i, x in enumerate(population) if i in chosen]


def _span_text(start: float, end: float) -> str:
    return f"{seconds_to_clock(start)}-{seconds_to_clock(end)}"


def _tidy(text) -> str:
    return " ".join(str(text).split())


def select_exchange_sheet(primary: PanelRecord, replicate: PanelRecord | None, agreement: dict | None,
                          validation: dict, books: PanelBooks, *, seed: int, key: str = "",
                          stage_a: int | str = EXCHANGE_DEFAULTS["stage_a"],
                          exchanges: int = EXCHANGE_DEFAULTS["exchanges"],
                          qa: int = EXCHANGE_DEFAULTS["qa"], venture: int = EXCHANGE_DEFAULTS["venture"],
                          qa_codes: tuple[tuple[str, str], ...] = QA_CODES,
                          field_classes: tuple[str, ...] | None = None) -> dict:
    """{"rows", "sample"}: the exchange-level sheet of one video, a seeded random sample and nothing else.

    Rows: the pitch's cut (pitch:first_decision); the pitch's outcome (pitch:any_offer) and the prepared_window
    row (whose only answer is the structural tick "the prepared presentation is not where the page says it is")
    only when `field_classes` names `outcome` or `prepared_window` (not by default since 2026-10-21); for
    `exchanges` substantive exchanges the topic (none by default, see EXCHANGE_DEFAULTS); for `qa` question-answer
    pairs (an answer turn and the first question it responds to, both before the pitch's cut: the first decision of
    any kind by any investor, review_items.cut_position, the whole video without one) the codes of `qa_codes` (by default the
    question's concept and severity, the answer's responsiveness, evidence, substance and coherence); `venture` venture fields (the founders' gender first, then
    seeded others) and `stage_a` Stage A items (every item for "all"), both none by default: the clip session
    codes them. Nothing is selected because it was flagged, so
    every row is `sampled` with priority 1; `reasons` still says what the passes or the validator have
    against that field. A row's `blind_context` names the moment (clocks, speakers, turn summaries) without
    the value to be judged.

    The draws use `seed` and `key` (the video id, so that two videos do not draw the same Stage A items).
    "sample" records what was asked for and drawn, and per field class the rows on the sheet and the size
    of the population they stand for.
    """
    if field_classes is not None:
        unknown = [c for c in field_classes if c not in SHEET_CLASSES]
        if unknown:
            raise ValueError(f"unknown field classes {unknown}; choose from {list(SHEET_CLASSES)}")
    known = {(e.applies_to, e.id) for function in ("question", "answer") for e in books.codes_for(function)}
    bad = [f"{role}:{code}" for role, code in qa_codes if (role, code) not in known]
    if bad:
        raise ValueError(f"qa_codes {bad} are not question or answer codes of this instrument")
    primary, p_dict, other = _aligned(primary, replicate, agreement, books)
    _, field_rows = _field_rows(primary, p_dict, other, validation, books, SHEET_CLASSES)
    by_path = {row["path"]: row for row in field_rows}
    names = _names(p_dict)

    def rng(part: str) -> random.Random:
        return random.Random(f"{seed}:{key}:{part}")

    def who(actor: str) -> str:
        return f"{actor} ({names[actor]})" if actor in names else actor

    def turn_text(t: dict) -> str:
        return f"turn {t['turn_id']} ({_span_text(t['start_s'], t['end_s'])}) {who(t['speaker'])}: {_tidy(t['summary'])}"

    chosen: list[dict] = []
    population: dict[str, int] = {}

    def take(path: str, context: str | None = None, blind_context: str | None = None) -> None:
        row = by_path.get(path)
        if row is None or any(r["path"] == path for r in chosen):
            return
        row = dict(row)
        if context is not None:
            row["context"] = context
        if blind_context is not None:
            row["blind_context"] = blind_context
        chosen.append(row)

    # a. the pitch: its cut; its outcome and the check of the prepared window only when field_classes names them
    requested = set(field_classes or ())
    pitch_rows = _pitch_rows(p_dict, other, (OUTCOME, CUT) if OUTCOME in requested else (CUT,))
    chosen += pitch_rows
    for row in pitch_rows:
        population[row["field_class"]] = 1
    if "prepared_window" in requested:
        take("prepared_window", blind_context=WINDOW_ASK)
        population["prepared_window"] = 1

    # b. venture: the founders' gender, then others
    # the ask is coded in the clip session, not drawn onto this sheet
    others = [name for name in VENTURE_CODES if name not in (VENTURE_ALWAYS, ASK)]
    venture_fields = ([VENTURE_ALWAYS] + _draw(rng("venture"), others, max(0, venture - 1)))[:venture]
    for name in venture_fields:
        take(f"venture:{name}")
    population["venture"] = len(VENTURE_CODES) - 1

    # c. Stage A
    codes = sorted(i["code"] for i in p_dict["stage_a"])
    drawn_stage_a = list(codes) if stage_a == ALL else _draw(rng("stage_a"), codes, stage_a)
    for code in drawn_stage_a:
        take(f"stage_a:{code}")
    population["stage_a"] = len(codes)

    # d. substantive exchanges: the topic
    members: dict[int, list[dict]] = {}
    turns_by_id: dict[int, dict] = {}
    for t in p_dict["turns"]:
        turns_by_id.setdefault(t["turn_id"], t)
        if t["exchange_id"] is not None:
            members.setdefault(t["exchange_id"], []).append(t)
    substantive = [e for e in p_dict["exchanges"] if e["kind"] == SUBSTANTIVE]
    drawn_exchanges = _draw(rng("exchanges"), substantive, exchanges)
    investor_ids = {a["actor_id"] for a in p_dict["actors"] if a["role"] == "investor"}
    for e in drawn_exchanges:
        turns = members.get(e["exchange_id"], [])
        span = _span_text(min(t["start_s"] for t in turns), max(t["end_s"] for t in turns)) if turns else "no turns"
        # the investors who speak in it (the exchange's own list up to v3-dev8 named the same people)
        speaking = dict.fromkeys(t["speaker"] for t in turns if t["speaker"] in investor_ids)
        listed = ", ".join(who(actor) for actor in speaking) or "none listed"
        asked = [turn_text(t) for t in turns if t["function"] == "question"][:MAX_QUESTIONS_SHOWN]
        questions = f"; questions: {' | '.join(asked)}" if asked else ""
        tail = f"; investors: {listed}{questions}"
        base = f"exchange {e['exchange_id']}"
        label = f" \"{_tidy(e['label'])}\"" if e["label"] else ""
        # the label is the model's own name for the topic, so a blind sheet does not show it
        at = f"exchange:{e['exchange_id']}"
        take(f"{at}/topic", f"{base}{label}, {span}{tail}", f"{base}, {span}{tail}")
    population["exchange_topic"] = len(substantive)

    # e. question-answer pairs: an answer and the first investor's question it responds to, both before the pitch's cut (the
    # first decision of any kind by any investor; the whole video without one): the answers the analysis reads
    # (plan C Task 7f, 2026-10-21)
    cut = cut_position(p_dict)
    # an investor's question only, as analysis.panel_dataset.answer_followups reads them (final review M1)
    investors = {a["actor_id"] for a in p_dict["actors"] if a["role"] == "investor"}
    position = {}
    for pos, t in enumerate(p_dict["turns"]):
        position.setdefault(t["turn_id"], pos)
    pairs = []
    for pos, t in enumerate(p_dict["turns"]):
        if t["function"] != "answer" or pos >= cut:
            continue
        question = next((turns_by_id[i] for i in t["responds_to"]
                         if i in turns_by_id and turns_by_id[i]["function"] == "question"
                         and turns_by_id[i]["speaker"] in investors), None)
        if question is not None and position[question["turn_id"]] < cut:
            pairs.append((question, t))
    drawn_pairs = _draw(rng("qa"), pairs, qa)
    for question, answer in drawn_pairs:
        text = f"question {turn_text(question)} -> answer {turn_text(answer)}"
        for role, code in qa_codes:
            take(f"turn:{(question if role == 'question' else answer)['turn_id']}/code:{code}", text, text)
    qa_paths = {f"turn:{(q if role == 'question' else a)['turn_id']}/code:{code}"
                for q, a in pairs for role, code in qa_codes}
    population["turn_code"] = len(qa_paths & set(by_path))

    if field_classes is not None:
        chosen = [row for row in chosen if row["field_class"] in field_classes]
        # the pitch rows the sheet carries, so the manifest never lists one it does not (`--field-classes outcome`)
        pitch_rows = [row for row in pitch_rows if row["field_class"] in field_classes]
        for cls in (OUTCOME, CUT):
            if not any(row["field_class"] == cls for row in pitch_rows):
                population.pop(cls, None)
    for row in chosen:
        row["selected_as"], row["priority"] = SAMPLED, PRIORITY_SAMPLED
    chosen.sort(key=lambda row: (
        -1.0 if row["clock_start"] is None else clock_to_seconds(row["clock_start"]), row["path"]))
    on_sheet: dict[str, int] = {}
    for row in chosen:
        on_sheet[row["field_class"]] = on_sheet.get(row["field_class"], 0) + 1
    return {"rows": chosen, "sample": {
        "seed": seed, "key": key,
        "requested": {"stage_a": stage_a, "exchanges": exchanges, "qa": qa, "venture": venture,
                      "qa_codes": [list(pair) for pair in qa_codes]},
        "drawn": {"pitch": [row["path"].partition(":")[2] for row in pitch_rows], "venture": venture_fields,
                  "stage_a": drawn_stage_a,
                  "exchanges": [e["exchange_id"] for e in drawn_exchanges],
                  "qa": [[q["turn_id"], a["turn_id"]] for q, a in drawn_pairs]},
        "available": {"venture": len(VENTURE_CODES) - 1, "stage_a": len(codes),
                      "exchanges": len(substantive), "qa": len(pairs)},
        # the pairs were drawn before the pitch's cut only, and `available` and the turn_code population count those
        "qa_before_cut": True,
        # per field class: rows on this sheet, and the number of such fields in the record they were drawn from
        "on_sheet": {c: on_sheet[c] for c in SHEET_CLASSES if c in on_sheet},
        "population": {c: population[c] for c in SHEET_CLASSES if c in population},
    }}


def select_sheet(primary: PanelRecord, replicate: PanelRecord | None, agreement: dict | None,
                 validation: dict, books: PanelBooks, *, sample_unflagged: int, seed: int,
                 field_classes: tuple[str, ...] | None = None, max_flagged: int | None = None,
                 replicate_only: bool = True, presence: bool = True) -> dict:
    """{"rows": the sheet's rows, "flagged": what was flagged, kept and left out, "unflagged": the pool the
    sampled rows were drawn from} for one video.

    A row is flagged when the passes differ on it (Stage A: by two points or more, or rated in one
    pass only; a decision time or the prepared window: beyond the agreement's tolerances; free text
    never; nothing, blank and NONE are one value), when a message of the primary's validation
    report is about that field (see _TAG_FIELD; informational messages flag nothing), when it is an
    `other` concept or topic, or when its actor was mapped by order. A turn, exchange or decision
    one pass lacks is one unit_presence row (primary `present`, replicate `absent`, or the reverse
    under a `replicate_only:` path) and flags none of its fields; `replicate_only=False` leaves out
    the rows that would show what the replicate holds, and `presence=False` every unit_presence row
    (a blind sheet: such a row exists only because the other pass lacks the unit).

    The sampled rows are `sample_unflagged` rows drawn from the unflagged pool, each field class in
    proportion to its population: the fields nothing flagged, without those of a turn, exchange or
    decision the replicate lacks. "unflagged" records the pool and the draw per field class.

    Not on a sheet at all: the topic of an exchange that is not substantive.
    Not on a sheet unless `field_classes` names their class: actors, free text that names no
    `other` matter, and the link of an answer to the question right before it.

    With more than `max_flagged` flagged rows the sheet keeps every unit_presence row and every row
    a validator error points at, and a seeded draw of the rest; "flagged" counts what was left out
    by field class and by reason. `agreement` is the stored agreement report; it is computed when
    absent. Without a replicate nothing is compared. Rows are ordered by clock, then path.
    """
    if field_classes is not None:
        unknown = [c for c in field_classes if c not in SHEET_CLASSES]
        if unknown:
            raise ValueError(f"unknown field classes {unknown}; choose from {list(SHEET_CLASSES)}")
    primary, p_dict, other = _aligned(primary, replicate, agreement, books)
    items, rows = _field_rows(primary, p_dict, other, validation, books, field_classes)
    if field_classes is not None and OUTCOME in field_classes:      # computed, so only when asked for
        rows += _pitch_rows(p_dict, other, (OUTCOME,))       # the first decision is on the full-video sheet only
    if presence and other is not None and (field_classes is None or UNIT_PRESENCE in field_classes):
        rows += _presence_rows(items, p_dict, other)
        if replicate_only:
            rows += _replicate_only_rows(p_dict, other)
    for row in rows:
        row["selected_as"] = FLAGGED if row["reasons"] else SAMPLED
        row["priority"] = priority(row)

    flagged = [row for row in rows if row["reasons"]]
    kept = _cap(flagged, max_flagged, seed)
    kept_paths = {row["path"] for row in kept}

    def lacks_counterpart(row: dict) -> bool:
        """A field of a turn, exchange or decision the replicate does not have: judged as a unit, not sampled."""
        if other is None or row["field_class"] in (UNIT_PRESENCE, OUTCOME):
            return False
        ref = parse_path(row["path"])
        return ref.unit in _UNITS and not other.has(ref.unit, ref.unit_id)

    unflagged = [row for row in rows if not row["reasons"]]
    lonely = [row for row in unflagged if lacks_counterpart(row)]
    lonely_paths = {row["path"] for row in lonely}
    pool = [row for row in unflagged if row["path"] not in lonely_paths]
    sampled = _proportional(_by_class(pool), sample_unflagged, random.Random(seed))
    chosen = [row for row in rows if row["path"] in kept_paths or row["path"] in sampled]
    chosen.sort(key=lambda row: (
        -1.0 if row["clock_start"] is None else clock_to_seconds(row["clock_start"]), row["path"]))
    return {"rows": chosen,
            "flagged": {"max_flagged": max_flagged, "found": _tally(flagged), "on_sheet": _tally(kept),
                        "left_out": _tally([row for row in flagged if row["path"] not in kept_paths])},
            "unflagged": {"requested": sample_unflagged, "seed": seed, "population": _count_by_class(pool),
                          "on_sheet": _count_by_class([row for row in pool if row["path"] in sampled]),
                          "excluded_no_counterpart": {"n": len(lonely), "by_field_class": _count_by_class(lonely)},
                          "note": UNFLAGGED_NOTE}}
