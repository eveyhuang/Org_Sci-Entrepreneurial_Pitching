"""Review items: every field of a panel record a human coder can judge, addressed by a stable path.

A path names one field of one unit by the record's own ids:

    actor:<actor_id>/label | role
    turn:<turn_id>/function | exchange | responds_to | other_concept | code:<CODE>
    exchange:<exchange_id>/kind | topic | label
    decision:<decision_id>/kind | actor | time | offer_id
    stage_a:<code>
    prep_pres:<CODE>                    a prepared presentation code of record["prepared"] (PRES-SUBS, ...)
    venture:primary_industry | product_form | customer_type | founder_gender | ask
    prepared_window

The ask reads `<usd>;<pct>` (`200000;10`) or `missing:<reason>` and writes the three ask fields of the venture.
`prep_pres:PRES-EVID` holds a count of checkable claims; on a coder's sheet it is answered in bands (`EVID_BANDS`,
`evid_band`), which agreement compares with the model's count banded the same way; the record keeps the count.

Two more paths appear on a review sheet only, for a unit one pass recorded and the other did not
(field class `unit_presence`, values `present` | `absent`); they address no field and cannot be written:

    turn:<turn_id>/present | exchange:<exchange_id>/present | decision:<decision_id>/present
    replicate_only:turn@<clock> | replicate_only:exchange@<clock> | replicate_only:decision@<clock>

Two pitch-level paths are sheet-only too, computed from the decisions rather than stored, so they are judged
and never written (an import stores a differing judgment beside the record):

    pitch:any_offer         did any investor make or join an offer (field class `outcome`, yes | no | unknown)
    pitch:first_decision    the clock of the panel's first decision (field class `cut`, a clock or empty)

and a pair judgement of the clip session names two prepared clips of a pair design and a Stage A item; its model
side comes from the ranking, not from a record (field class `pair`, values `left` | `right`):

    pair:<batch_id>/<stage_a_code>

Values travel as text. An empty field (no exchange, no link, an uncodable Stage A item, no offer id)
is None; a code that is missing for a reason reads `missing:<reason>`; lists are comma-separated;
times are clock strings. No I/O here: records in, items or an edited record dict out.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import get_args

from pydantic import ValidationError

from pipeline.schemas.stage_b_panel import MISSING_REASONS, Actor, PanelRecord, clock_to_seconds, seconds_to_clock
from pipeline.stage_b.panel_books import CONCEPT_CODE_OF, PanelBooks, canonical_value, implies_offer
# the validator's own test of a code value, so a human value is accepted exactly when validation would accept it
from pipeline.stage_b.panel_validate import OFFER_MOVE_KINDS, _value_problem, _whole_number

FIELD_CLASSES = ("actor", "turn_function", "turn_code", "turn_exchange", "turn_link", "exchange_kind",
                 "exchange_topic", "decision", "stage_a", "prep_pres", "venture", "prepared_window", "cut", "pair")
ASK = "ask"                         # the venture field that stands for ask_amount_usd, ask_equity_pct and ask_missing
VENTURE_CODES = {"primary_industry": "VC-INDUSTRY", "product_form": "VC-FORM", "customer_type": "VC-CUSTOMER",
                 "founder_gender": "VC-GENDER", ASK: "VC-ASK"}
# `<usd>;<pct>` as a coder may type it: `200000;10`, `$200,000; 10%`
ASK_RE = re.compile(r"^\s*\$?([\d,]+)\s*;\s*([\d.]+)\s*%?\s*$")
ACTOR_ROLES: tuple[str, ...] = get_args(Actor.model_fields["role"].annotation)
# the count of checkable claims in the prepared presentation cannot be made reliably at viewing speed, so a coder
# chooses a band (researcher's decision, 2026-10-21); the model's count is banded the same way for agreement only
EVID_CODE = "PRES-EVID"
EVID_PATH = f"prep_pres:{EVID_CODE}"
EVID_BANDS = ("0", "1-2", "3-5", "6+")
MISSING_PREFIX = "missing:"
_FIELDS = {
    "actor": {"label": "actor", "role": "actor"},
    "turn": {"function": "turn_function", "exchange": "turn_exchange", "responds_to": "turn_link",
             "other_concept": "turn_code", "code": "turn_code"},
    "exchange": {"kind": "exchange_kind", "topic": "exchange_topic", "label": "exchange_topic"},
    "decision": {"kind": "decision", "actor": "decision", "time": "decision", "offer_id": "decision"},
}
_INT_UNITS = ("turn", "exchange", "decision")
UNIT_PRESENCE = "unit_presence"                     # is a unit one pass lacks really in the footage?
OUTCOME = "outcome"                                 # did any investor make an offer?
CUT = "cut"                                         # when does the panel first decide?
OUTCOME_PATH, CUT_PATH = "pitch:any_offer", "pitch:first_decision"
_PITCH = {"any_offer": OUTCOME, "first_decision": CUT}      # the pitch-level paths: judged, never written
PAIR = "pair"
SHEET_CLASSES = (UNIT_PRESENCE, OUTCOME, *FIELD_CLASSES)     # every field class a sheet row can have
# the columns of a review sheet, in order: the CSV an export writes, a coding page downloads and an import reads
SHEET_COLUMNS = ["priority", "row", "path", "field_class", "clock_start", "clock_end", "context", "primary", "replicate",
                 "reasons", "selected_as", "allowed_values", "human_value", "resolution", "note"]
OUTCOME_VALUES = ("yes", "no", "unknown")
NOT_AN_EXCHANGE = "not_an_exchange"                 # a human's answer to an exchange topic: there is no such exchange
PRESENT, ABSENT = "present", "absent"
PRESENCE_VALUES = (PRESENT, ABSENT)
REPLICATE_ONLY = "replicate_only"
MAIN_FIELD = {"turn": "function", "exchange": "kind", "decision": "kind"}     # the field that says what a unit is
_PRESENCE = re.compile(r"^(turn|exchange|decision):(\d+)/present$")
_REPLICATE_ONLY = re.compile(rf"^{REPLICATE_ONLY}:(turn|exchange|decision)@\S+$")
STAGE_A_CHANGED = "[value changed in review]"       # appended to the note of a Stage A item whose value a coder changed
DECISION_FIELDS = ("kind", "actor", "time", "offer_id")
HOUR_S = 3600.0


@dataclass(frozen=True)
class ReviewItem:
    path: str            # stable address, see the module docstring
    field_class: str     # one of FIELD_CLASSES
    clock_start: str | None
    clock_end: str | None
    context: str         # one line a coder needs: speaker name, summary, exchange label
    value: str | None    # the record's value as text; None means the field is empty


@dataclass(frozen=True)
class PathRef:
    unit: str                  # actor | turn | exchange | decision | stage_a | prep_pres | venture | prepared_window
                               # | pitch | pair
    unit_id: str | int | None  # the record's id of the unit (an int for turns, exchanges and decisions; the field
                               # name for venture and pitch; the batch id for pair)
    field: str                 # the field of the unit; "value" for the single-segment units
    sub: str | None = None     # the code id of a turn code, the Stage A item of a pair

    @property
    def field_class(self) -> str:
        if self.unit in _FIELDS:
            return _FIELDS[self.unit][self.field]
        if self.unit == "pitch":
            return _PITCH[self.unit_id]
        return self.unit


@dataclass(frozen=True)
class PresenceRef:
    unit: str                  # turn | exchange | decision
    unit_id: int | None        # the primary's id; None for a unit only the replicate has
    replicate_only: bool

    @property
    def primary_value(self) -> str:
        return ABSENT if self.replicate_only else PRESENT

    @property
    def replicate_value(self) -> str:
        return PRESENT if self.replicate_only else ABSENT


def presence_ref(path: str | None) -> PresenceRef | None:
    """The unit a unit_presence path asks about; None for every other path."""
    found = _PRESENCE.match(path or "")
    if found:
        return PresenceRef(found.group(1), int(found.group(2)), False)
    found = _REPLICATE_ONLY.match(path or "")
    return PresenceRef(found.group(1), None, True) if found else None


def is_outcome(path: str | None) -> bool:
    """Whether a path asks whether any investor made an offer (pitch:any_offer)."""
    return (path or "").strip() == OUTCOME_PATH


def is_cut(path: str | None) -> bool:
    """Whether a path asks when the panel first decides (pitch:first_decision)."""
    return (path or "").strip() == CUT_PATH


def _investor_decisions(d: dict) -> dict[str, list[dict]]:
    """investor id -> that investor's decisions, in record order, for every investor of a record dict."""
    investors = dict.fromkeys(a["actor_id"] for a in d["actors"] if a["role"] == "investor")
    return {actor: [x for x in d["decisions"] if x["actor"] == actor] for actor in investors}


def any_offer(d: dict) -> str:
    """Whether any investor of a record dict made or joined an offer: yes when a decision implies one (also when
    the record holds only its revision or withdrawal, or an offer after an exit and a re-entry); no when every
    investor exits and none ever offers; else unknown (an investor never seen deciding, or no investor).

    The rule of analysis.panel_dataset.pitch_outcome, so a reviewed verdict is compared with the value the
    analysis reads.
    """
    own = _investor_decisions(d)
    if any(implies_offer(x["kind"], x["offer_id"]) for decisions in own.values() for x in decisions):
        return "yes"
    if own and all(any(x["kind"] == "exit" for x in decisions) for decisions in own.values()):
        return "no"
    return "unknown"


def first_decision_clock(d: dict) -> str | None:
    """The clock of the earliest decision of any kind by any investor of a record dict (an exit as much as an
    offer); None without one.

    This takes the earliest decision `time_s`, while analysis.panel_dataset.pitch_cut takes the earliest decision
    by turn position; the two can name different decisions when the record's decision times are out of turn order.
    The clock is for the paper table's agreement on the first decision only and never moves the analysis cut."""
    times = [x["time_s"] for decisions in _investor_decisions(d).values() for x in decisions]
    return seconds_to_clock(min(times)) if times else None


def cut_position(d: dict) -> int:
    """The turn position at which a record dict's history stops: analysis.panel_dataset.pitch_cut's rule on the
    record. A turn's position is its place in the record's turn list (a repeated turn id takes its first place); a
    decision sits at the position of the turn that speaks it, else at the last turn started by its time (the first
    turn when none is). The cut is the earliest position of a decision of any kind by any investor (an exit as much as
    an offer); without one it is the number of turns, so the whole video is history. The full-video sheet draws its
    question-answer pairs before it (plan C Task 7f, 2026-10-21): the answers the analysis reads."""
    turns = d["turns"]
    pos_of: dict[int, int] = {}
    for pos, t in enumerate(turns):
        pos_of.setdefault(t["turn_id"], pos)
    investors = {a["actor_id"] for a in d["actors"] if a["role"] == "investor"}
    found = []
    for x in d["decisions"]:
        if x["actor"] not in investors:
            continue
        pos = pos_of.get(x["turn_id"])
        if pos is None:
            started = [p for p, t in enumerate(turns) if t["start_s"] <= x["time_s"]]
            pos = max(started) if started else 0
        found.append(pos)
    return min(found) if found else len(turns)


def presence_path(unit: str, unit_id: int) -> str:
    return f"{unit}:{unit_id}/present"


def replicate_only_path(unit: str, clock: str | None) -> str:
    return f"{REPLICATE_ONLY}:{unit}@{clock or 'unplaced'}"


def main_path(unit: str, unit_id: int) -> str:
    """The path of the field that says what a turn, exchange or decision is."""
    return f"{unit}:{unit_id}/{MAIN_FIELD[unit]}"


def _anchor(path: str) -> str:
    """The field path that stands for a unit_presence path of the primary when locating it; other paths as they are."""
    presence = presence_ref(path)
    if presence is None:
        return path
    if presence.replicate_only:
        raise _no_path(path, "a unit only the replicate has is not in the record")
    return main_path(presence.unit, presence.unit_id)


def _no_path(path: str, why: str = "") -> ValueError:
    return ValueError(f"{path}: no such path" + (f" ({why})" if why else ""))


def parse_path(path: str) -> PathRef:
    """The parts of a path; ValueError when it does not follow the grammar (whatever the record holds)."""
    if path == "prepared_window":
        return PathRef("prepared_window", None, "value")
    segments = path.split("/")
    unit, colon, unit_id = segments[0].partition(":")
    if not colon or not unit_id:
        raise _no_path(path)
    if unit in ("stage_a", "prep_pres", "venture", "pitch"):
        if (len(segments) != 1 or (unit == "venture" and unit_id not in VENTURE_CODES)
                or (unit == "pitch" and unit_id not in _PITCH)):
            raise _no_path(path)
        return PathRef(unit, unit_id, "value")
    if unit == PAIR:
        if len(segments) != 2 or not segments[1] or ":" in segments[1]:
            raise _no_path(path, "a pair judgement is pair:<batch_id>/<stage_a_code>")
        return PathRef(unit, unit_id, "value", segments[1])
    if unit not in _FIELDS:
        raise _no_path(path)
    ident: str | int = unit_id
    if unit in _INT_UNITS:
        if not (unit_id.isascii() and unit_id.isdigit()):
            raise _no_path(path, f"{unit_id!r} is not a {unit} id")
        ident = int(unit_id)
    rest = segments[1:]
    if len(rest) == 1:
        name, colon, sub = rest[0].partition(":")
        if unit == "turn" and name == "code" and sub:
            return PathRef(unit, ident, "code", sub)
        if not colon and name in _FIELDS[unit] and name != "code":
            return PathRef(unit, ident, name)
    raise _no_path(path)


def normalize_path(path: str | None) -> str:
    """A path as this module writes it, so that one field has one path: `turn:0005/function` is
    `turn:5/function`. A path that follows no grammar is returned as given (stripped), for the caller to refuse."""
    text = (path or "").strip()
    found = _PRESENCE.match(text)
    if found:
        return presence_path(found.group(1), int(found.group(2)))
    if _REPLICATE_ONLY.match(text):
        return text
    try:
        ref = parse_path(text)
    except ValueError:
        return text
    if ref.unit not in _INT_UNITS:
        return text
    base = f"{ref.unit}:{ref.unit_id}"
    if ref.field == "code":
        return f"{base}/code:{ref.sub}"
    return f"{base}/{ref.field}"


def field_class(path: str) -> str:
    return UNIT_PRESENCE if presence_ref(path) is not None else parse_path(path.strip()).field_class


def canonical_full(record: PanelRecord, books: PanelBooks) -> PanelRecord:
    """The record with every codebook value in the codebook's spelling: structure labels, code values, venture."""
    d = books.canonical_structure(record.model_dump(mode="json"))
    panel_ids = set(books.panel.ids)
    for t in d["turns"]:
        for c in t["codes"]:
            if c["code"] in panel_ids:
                c["value"] = canonical_value(books.panel.by_id(c["code"]), c["value"])
    for c in d["prepared"]:
        if c["code"] in panel_ids:
            c["value"] = canonical_value(books.panel.by_id(c["code"]), c["value"])
    for name, code_id in VENTURE_CODES.items():
        if name != ASK:
            d["venture"][name] = books.canonical_venture(code_id, d["venture"][name])
    d["venture"]["secondary_industries"] = [books.canonical_venture("VC-INDUSTRY", s)
                                            for s in d["venture"]["secondary_industries"]]
    return PanelRecord.model_validate(d)


# ---- finding a unit in a record dict ------------------------------------------------------

_LISTS = {"actor": ("actors", "actor_id"), "turn": ("turns", "turn_id"), "exchange": ("exchanges", "exchange_id"),
          "decision": ("decisions", "decision_id"), "stage_a": ("stage_a", "code"), "prep_pres": ("prepared", "code")}


def _unit(d: dict, ref: PathRef, path: str) -> dict:
    """The dict of the unit a path addresses; the first one when an id is repeated."""
    if ref.unit == "venture":
        return d["venture"]
    if ref.unit not in _LISTS:      # pitch and pair: judged from the footage or the clips, no field of a record
        raise _no_path(path, f"a {ref.unit} path is judged on a sheet and is no field of the record")
    key, id_field = _LISTS[ref.unit]
    for item in d[key]:
        if item[id_field] == ref.unit_id:
            return item
    raise _no_path(path, f"{ref.unit} {ref.unit_id} is not in the record")


def _placed(d: dict, ref: PathRef, path: str) -> dict | None:
    """The unit a path addresses, for placing it in time; None for the prepared window and the pitch, which are
    no unit of a list."""
    return None if ref.unit in ("prepared_window", "pitch") else _unit(d, ref, path)


def _first(items: list[dict], field: str, value) -> dict | None:
    return next((i for i in items if i[field] == value), None)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _text(value) -> str | None:
    """A cell as a value: None for nothing or blank, otherwise the stripped text."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _window_text(spans: list[dict]) -> str | None:
    return ";".join(f"{seconds_to_clock(s['start_s'])}-{seconds_to_clock(s['end_s'])}" for s in spans) or None


def _code_text(code: dict | None) -> str | None:
    if code is None:
        return None
    return f"{MISSING_PREFIX}{code['missing']}" if code["value"] is None else code["value"]


def _number_text(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def _ask_text(venture: dict) -> str | None:
    if venture["ask_missing"] != "none":
        return f"{MISSING_PREFIX}{venture['ask_missing']}"
    if venture["ask_amount_usd"] is None or venture["ask_equity_pct"] is None:
        return None
    return f"{venture['ask_amount_usd']};{_number_text(venture['ask_equity_pct'])}"


def read_value(d: dict, path: str) -> str | None:
    """The value at a path of a record dict, as written. ValueError when the path is not in the record.

    A turn that does not carry a code reads None for it. The pitch paths read what the decisions give
    (any_offer, first_decision_clock); they are no field of the record and cannot be written.
    """
    ref = parse_path(path)
    if ref.unit == "prepared_window":
        return _window_text(d["prepared_spans"])
    if ref.unit == "pitch":
        return any_offer(d) if ref.unit_id == "any_offer" else first_decision_clock(d)
    unit = _unit(d, ref, path)
    if ref.unit == "stage_a":
        return None if unit["value"] is None else str(unit["value"])
    if ref.unit == "prep_pres":
        return _code_text(unit)
    if ref.unit == "venture":
        return _ask_text(unit) if ref.unit_id == ASK else _text(unit[ref.unit_id])
    if ref.unit == "turn":
        if ref.field == "exchange":
            return None if unit["exchange_id"] is None else str(unit["exchange_id"])
        if ref.field == "responds_to":
            return ",".join(str(i) for i in unit["responds_to"]) or None
        if ref.field == "code":
            return _code_text(_first(unit["codes"], "code", ref.sub))
    if ref.unit == "decision" and ref.field == "time":
        return seconds_to_clock(unit["time_s"])
    return _text(unit[ref.field])


# ---- flatten ------------------------------------------------------------------------------

def _ranges(ids: list[int]) -> str:
    """Turn ids as compact runs: [2, 3, 4, 7] gives "2-4, 7"."""
    runs: list[list[int]] = []
    for i in sorted(set(ids)):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


class _Reader:
    """Where each unit of one record dict sits in time, and how to name it to a coder."""

    def __init__(self, d: dict):
        self.d = d
        self.names: dict[str, str] = {}
        for a in d["actors"]:
            self.names.setdefault(a["actor_id"], a["label"])
        self.turns: dict[int, dict] = {}
        self.members: dict[int, list[dict]] = {}
        self.first_speech: dict[str, float] = {}
        for t in d["turns"]:
            self.turns.setdefault(t["turn_id"], t)
            self.first_speech.setdefault(t["speaker"], t["start_s"])
            if t["exchange_id"] is not None:
                self.members.setdefault(t["exchange_id"], []).append(t)

    def who(self, actor: str) -> str:
        return f"{actor} ({self.names[actor]})" if actor in self.names else actor

    def span(self, ref: PathRef, unit: dict | None, blind: bool = False) -> tuple[float | None, float | None]:
        """Where to look for a unit. `blind` leaves out a time that is itself a judged value: a decision
        is then placed by the turn it is spoken in, and the prepared window is not placed at all, nor
        is a Stage A item, whose clocks would be that window."""
        d = self.d
        if blind and ref.unit == "decision":
            turn = self.turns.get(unit["turn_id"])
            return (None, None) if turn is None else (turn["start_s"], turn["end_s"])
        if blind and ref.unit in ("prepared_window", "stage_a", "prep_pres"):   # where the prepared presentation
            return None, None                                                   # is, is judged
        if ref.unit == "pitch":         # the whole encounter; when the panel decides is judged
            return None, None
        if ref.unit == "actor":
            return self.first_speech.get(ref.unit_id), None
        if ref.unit == "turn":
            return unit["start_s"], unit["end_s"]
        if ref.unit == "exchange":
            turns = self.members.get(ref.unit_id, [])
            if not turns:
                return None, None
            return min(t["start_s"] for t in turns), max(t["end_s"] for t in turns)
        if ref.unit == "decision":
            return unit["time_s"], None
        if ref.unit == "venture":
            return d["venture"]["earliest_s"], None
        if not d["prepared_spans"]:     # stage_a, prep_pres and prepared_window: the prepared presentation
            return None, None
        return (min(s["start_s"] for s in d["prepared_spans"]), max(s["end_s"] for s in d["prepared_spans"]))

    def locator(self, ref: PathRef, unit: dict | None) -> str:
        """Which unit this is, in words that carry no judged value: ids, speakers and turn numbers only."""
        if ref.unit == "actor":
            return f"actor {ref.unit_id}"
        if ref.unit == "turn":
            return f"turn {ref.unit_id}, {self.who(unit['speaker'])}"
        if ref.unit == "exchange":
            turns = _ranges([t["turn_id"] for t in self.members.get(ref.unit_id, [])])
            return f"exchange {ref.unit_id}, " + (f"turns {turns}" if turns else "no turns")
        if ref.unit == "decision":
            return f"decision {ref.unit_id}" + ("" if unit["turn_id"] is None else f", turn {unit['turn_id']}")
        if ref.unit == "pitch":
            return "the pitch"
        return "the venture" if ref.unit == "venture" else "prepared presentation"

    def describe(self, ref: PathRef, unit: dict | None) -> str:
        if ref.unit == "actor":
            return f"{unit['label']} ({unit['role']})"
        if ref.unit == "turn":
            return unit["summary"]
        if ref.unit == "exchange":
            return ", ".join(x for x in (unit["kind"], unit["label"]) if x)
        if ref.unit == "decision":
            return ", ".join(x for x in (f"{self.who(unit['actor'])} {unit['kind']}", unit["terms"]) if x)
        if ref.unit == "stage_a":
            return unit["note"]
        if ref.unit == "venture":
            return self.d["venture"]["description"]
        return ""


def _paths(d: dict, books: PanelBooks) -> list[str]:
    """Every path of a (canonical) record dict, in record order; a repeated id is listed once."""
    out: list[str] = []
    for a in d["actors"]:
        out += [f"actor:{a['actor_id']}/label", f"actor:{a['actor_id']}/role"]
    for t in d["turns"]:
        base = f"turn:{t['turn_id']}"
        out += [f"{base}/function", f"{base}/exchange", f"{base}/responds_to"]
        codes = [c["code"] for c in t["codes"]] + [e.id for e in books.codes_for(t["function"])]
        out += [f"{base}/code:{code}" for code in codes]
        if t["function"] in CONCEPT_CODE_OF or t["other_concept"] is not None:
            out.append(f"{base}/other_concept")
    for e in d["exchanges"]:
        base = f"exchange:{e['exchange_id']}"
        out += [f"{base}/kind", f"{base}/topic", f"{base}/label"]
    for x in d["decisions"]:
        base = f"decision:{x['decision_id']}"
        out += [f"{base}/kind", f"{base}/actor", f"{base}/time", f"{base}/offer_id"]
    out += [f"stage_a:{i['code']}" for i in d["stage_a"]]
    out += [f"prep_pres:{c['code']}" for c in d["prepared"]]
    out += [f"venture:{name}" for name in VENTURE_CODES]
    out.append("prepared_window")
    return list(dict.fromkeys(out))


def flatten(record: PanelRecord, books: PanelBooks) -> list[ReviewItem]:
    """One item per judgeable field, in record order, values in the codebook's spelling.

    A turn has an item for every code it carries and for every code its function requires (None when
    the turn lacks it), and an other_concept item when it is a question or an answer or carries a name.
    """
    d = canonical_full(record, books).model_dump(mode="json")
    reader = _Reader(d)
    items = []
    for path in _paths(d, books):
        ref = parse_path(path)
        unit = _placed(d, ref, path)
        start, end = reader.span(ref, unit)
        described = _one_line(reader.describe(ref, unit))
        items.append(ReviewItem(
            path=path, field_class=ref.field_class,
            clock_start=None if start is None else seconds_to_clock(start),
            clock_end=None if end is None else seconds_to_clock(end),
            context=_one_line(reader.locator(ref, unit)) + (f": {described}" if described else ""),
            value=read_value(d, path),
        ))
    return items


def locator(record: PanelRecord, path: str) -> str:
    """Which unit a path belongs to, without any value a coder is asked to judge (for a blind sheet)."""
    d = record.model_dump(mode="json")
    path = _anchor(path)
    ref = parse_path(path)
    unit = _placed(d, ref, path)
    return _one_line(_Reader(d).locator(ref, unit))


def locator_span(record: PanelRecord, path: str) -> tuple[str | None, str | None]:
    """(clock_start, clock_end) of where to look for a path, without a time a coder is asked to judge."""
    d = record.model_dump(mode="json")
    path = _anchor(path)
    ref = parse_path(path)
    unit = _placed(d, ref, path)
    start, end = _Reader(d).span(ref, unit, blind=True)
    return (None if start is None else seconds_to_clock(start), None if end is None else seconds_to_clock(end))


# ---- allowed values -----------------------------------------------------------------------

def _scale_values(entry) -> tuple[str, ...] | None:
    """The values of an ordinal scale ("1", ..., "5"); None for an entry without a scale (a count, a number)."""
    if not getattr(entry, "scale", None):
        return None
    lo, hi = entry.scale
    return tuple(str(n) for n in range(int(lo), int(hi) + 1))


def _code_entry(books: PanelBooks, code_id: str):
    return books.panel.by_id(code_id) if code_id in set(books.panel.ids) else None


def _prepared_entry(books: PanelBooks, code_id: str):
    """The codebook entry of a prepared presentation code; None for any other code id."""
    return next((e for e in books.prepared_codes() if e.id == code_id), None)


def choices_of(entry) -> tuple[str, ...] | None:
    """The values a code takes as a choice: categorical values, true/false, an ordinal scale; None for a count."""
    if entry.type == "categorical":
        return tuple(entry.values or [])
    if entry.type == "binary":
        return ("true", "false")
    return _scale_values(entry) if entry.type == "ordinal" else None


def evid_band(count) -> str:
    """The band of a count of checkable claims: 0 → "0", 1-2 → "1-2", 3-5 → "3-5", 6 or more → "6+"."""
    n = int(str(count).strip())
    if n < 0:
        raise ValueError(f"a count of checkable claims is 0 or more, not {n}")
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else "6+"


def is_banded(path: str | None) -> bool:
    """Whether a path is answered in bands on a coder's sheet and compared in bands (`prep_pres:PRES-EVID`)."""
    return path == EVID_PATH


def as_band(text) -> str | None:
    """A band as a coder may type it on a sheet: "3-5", "3–5" (an en or em dash), "3 - 5", "6 +"; None for anything
    that is not one of EVID_BANDS."""
    if text is None:
        return None
    folded = re.sub(r"\s+", "", str(text)).replace("\u2013", "-").replace("\u2014", "-")
    return folded if folded in EVID_BANDS else None


def banded(path: str | None, value):
    """A value at a banded path as agreement compares it: a band as given, a whole-number count as its band;
    anything else (empty, `missing:<reason>`) and every value at another path unchanged."""
    if not is_banded(path) or value is None:
        return value
    text = str(value).strip()
    if as_band(text):
        return as_band(text)
    return evid_band(text) if _whole_number(text) else value


def band_rank(path: str | None, value):
    """A value at a banded path as an ordinal number (the band's place, "0" to "3"); else unchanged."""
    band = banded(path, value)
    return str(EVID_BANDS.index(band)) if is_banded(path) and band in EVID_BANDS else value


def allowed_values(path: str, record: PanelRecord, books: PanelBooks, coder: bool = False) -> tuple[str, ...] | None:
    """The values a field takes, or None when it is free text, a number, a list or a time.

    A code may also be given as `missing:<reason>`. ValueError when the path is not in the record. `coder`: the
    values on a coder's (blind) sheet, where `prep_pres:PRES-EVID` takes a band (EVID_BANDS) rather than a count.
    """
    d = record.model_dump(mode="json")
    presence = presence_ref(path)
    if presence is not None:
        if not presence.replicate_only:
            read_value(d, _anchor(path))    # raises when the unit is not in the record
        return PRESENCE_VALUES
    ref = parse_path(path)
    read_value(d, path)     # raises when the unit is not in the record
    if ref.unit == "pitch":
        return OUTCOME_VALUES if ref.unit_id == "any_offer" else None      # the first decision is a clock
    if ref.unit == "actor":
        return ACTOR_ROLES if ref.field == "role" else None
    if ref.unit == "turn":
        if ref.field == "function":
            return books.turn_functions()
        if ref.field == "exchange":
            return tuple(dict.fromkeys(str(e["exchange_id"]) for e in d["exchanges"]))
        if ref.field == "code":
            entry = _code_entry(books, ref.sub)
            return None if entry is None else choices_of(entry)
        return None
    if ref.unit == "exchange":
        return {"kind": books.exchange_kinds(), "topic": books.exchange_topics()}.get(ref.field)
    if ref.unit == "decision":
        if ref.field == "kind":
            return books.decision_kinds()
        return tuple(dict.fromkeys(a["actor_id"] for a in d["actors"])) if ref.field == "actor" else None
    if ref.unit == "stage_a":
        return _scale_values(books.stage_a.by_id(ref.unit_id))
    if ref.unit == "prep_pres":
        if coder and is_banded(path):
            return EVID_BANDS
        entry = _prepared_entry(books, ref.unit_id)
        return None if entry is None else choices_of(entry)
    if ref.unit == "venture":
        return None if ref.unit_id == ASK else books.venture_values(VENTURE_CODES[ref.unit_id])
    return None


# ---- writing ------------------------------------------------------------------------------

def _choice(text: str | None, allowed: tuple[str, ...]) -> str:
    if text is None:
        raise ValueError(f"needs one of {list(allowed)}")
    folded = text.casefold()
    for value in allowed:
        if value.casefold() == folded:
            return value
    raise ValueError(f"{text!r} is not one of {list(allowed)}")


def _integer(text: str, lo: float, hi: float) -> int:
    try:
        n = int(text)
    except ValueError:
        raise ValueError(f"{text!r} is not an integer") from None
    if not lo <= n <= hi:
        raise ValueError(f"{n} is outside {int(lo)}-{int(hi)}")
    return n


def _missing_reason(text: str, what: str) -> str | None:
    """The reason of a `missing:<reason>` value; None for any other text; ValueError for a reason the schema lacks."""
    if not text.casefold().startswith(MISSING_PREFIX):
        return None
    reason = text[len(MISSING_PREFIX):].strip().casefold()
    if reason == "none" or reason not in MISSING_REASONS:
        raise ValueError(f"{what}: {reason!r} is not a missing reason "
                         f"({', '.join(r for r in MISSING_REASONS if r != 'none')})")
    return reason


def _code_item(entry, text: str | None) -> dict:
    """The record form of a human code value: a value the codebook allows (a whole number of 0 or more for a
    count), or `missing:<reason>`."""
    if text is None:
        raise ValueError(f"{entry.id} needs a value or {MISSING_PREFIX}<reason>")
    reason = _missing_reason(text, entry.id)
    if reason is not None:
        return {"code": entry.id, "value": None, "missing": reason}
    if entry.type == "count":
        if not _whole_number(text):
            raise ValueError(f"{entry.id} {text.strip()!r} is not a whole number of 0 or more")
        return {"code": entry.id, "value": str(int(text.strip())), "missing": "none"}
    value = canonical_value(entry, text)
    problem = _value_problem(entry, value)
    if problem:
        raise ValueError(f"{entry.id} {problem}")
    if entry.type == "ordinal":
        value = str(int(value))
    return {"code": entry.id, "value": value, "missing": "none"}


def _function_of(turn: dict, books: PanelBooks) -> str:
    return canonical_value(books.panel.by_id("TURN-FUNC"), turn["function"])


def _write_function(turn: dict, function: str | None, supplied: dict[str, str | None], books: PanelBooks) -> list[str]:
    new = _choice(function, books.turn_functions())
    required = [e.id for e in books.codes_for(new)]
    stray = [code for code in supplied if code not in required]
    if stray:
        raise ValueError(f"codes {stray} do not apply to function {new!r}")
    carried: dict[str, dict] = {}
    for c in turn["codes"]:
        carried.setdefault(c["code"], c)
    changed = new != _function_of(turn, books)
    if changed:
        needed = [code for code in required if code not in carried and code not in supplied]
        if needed:
            raise ValueError(f"function {new!r} needs codes {needed}; give them in the same import "
                             f"(rows turn:{turn['turn_id']}/code:<CODE>)")
    items = {code: _code_item(books.panel.by_id(code), text) for code, text in supplied.items()}
    turn["function"] = new
    if not changed:
        for code, item in items.items():
            _put_code(turn, item)
        return []
    turn["codes"] = [items.get(code) or carried[code] for code in required]
    return [code for code in carried if code not in required]


def _put_code(turn: dict, item: dict) -> None:
    for index, c in enumerate(turn["codes"]):
        if c["code"] == item["code"]:
            turn["codes"][index] = item
            return
    turn["codes"].append(item)


def _ids(text: str | None, what: str) -> list[int]:
    out = []
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if not (part.isascii() and part.isdigit()):
            raise ValueError(f"{part!r} is not {what}")
        out.append(int(part))
    return out


def _write_turn(d: dict, turn: dict, ref: PathRef, path: str, text: str | None, books: PanelBooks) -> None:
    if ref.field == "function":
        _write_function(turn, text, {}, books)
    elif ref.field == "exchange":
        ids = _ids(text, "an exchange id")
        if len(ids) > 1:
            raise ValueError(f"{text!r} is not an exchange id")
        if ids and _first(d["exchanges"], "exchange_id", ids[0]) is None:
            raise ValueError(f"exchange {ids[0]} is not a recorded exchange")
        turn["exchange_id"] = ids[0] if ids else None
    elif ref.field == "responds_to":
        ids = _ids(text, "a turn id")
        earlier = set()
        for t in d["turns"]:
            if t is turn:
                break
            earlier.add(t["turn_id"])
        for i in ids:
            if i not in earlier:
                raise ValueError(f"{i} is not an earlier turn")
        turn["responds_to"] = ids
    elif ref.field == "other_concept":
        turn["other_concept"] = text
    else:
        entry = _code_entry(books, ref.sub)
        function = _function_of(turn, books)
        carried = _first(turn["codes"], "code", ref.sub) is not None
        if entry is None or not (carried or entry.applies_to == function):
            raise _no_path(path, f"turn {ref.unit_id} does not carry {ref.sub} and its function "
                                 f"{function!r} does not require it")
        _put_code(turn, _code_item(entry, text))


def _write_exchange(exchange: dict, ref: PathRef, text: str | None, books: PanelBooks) -> None:
    if ref.field == "kind":
        exchange["kind"] = _choice(text, books.exchange_kinds())
    elif ref.field == "topic":
        exchange["topic"] = _choice(text, books.exchange_topics())
    elif ref.field == "label":
        exchange["label"] = text or ""


def _clock(text: str, duration_s: float | None) -> float:
    """A clock string as seconds, refused when it cannot be a moment of this video.

    On a video shorter than an hour a time in H:MM:SS form is refused: a spreadsheet that reads
    `1:48` as a time of day writes it back as `1:48:00`, an hour and 48 minutes.
    """
    seconds = clock_to_seconds(text)
    if duration_s is None:
        return seconds
    if text.count(":") == 2 and duration_s < HOUR_S:
        raise ValueError(f"{text.strip()!r} is in H:MM:SS form, but the video is {seconds_to_clock(duration_s)} long; "
                         "write M:SS (a spreadsheet may have turned M:SS into a time of day)")
    if seconds > duration_s:
        raise ValueError(f"{text.strip()} is beyond the video's duration ({seconds_to_clock(duration_s)})")
    return seconds


def _check_decision(d: dict, decision: dict, written: set[str], books: PanelBooks) -> None:
    """What a decision must satisfy after `written` fields changed: the actor's role fits the kind, and a
    kind the decision was changed to has the offer id it needs."""
    kind = canonical_value(books.panel.by_id("DEC-KIND"), decision["kind"])
    base = f"decision:{decision['decision_id']}"
    if written & {"kind", "actor"}:
        role = next((a["role"] for a in d["actors"] if a["actor_id"] == decision["actor"]), None)
        needed = "founder" if kind.startswith("founder_") else "investor"
        if role is not None and role != needed:
            article = "an" if needed == "investor" else "a"
            raise ValueError(f"{kind} needs {article} {needed}, and {decision['actor']} is "
                             f"{'an' if role == 'investor' else 'a'} {role}; correct {base}/kind and {base}/actor "
                             "in the same import")
    if "kind" in written and kind in OFFER_MOVE_KINDS and _text(decision["offer_id"]) is None:
        raise ValueError(f"{kind} needs an offer id and decision {decision['decision_id']} has none; give "
                         f"{base}/offer_id in the same import")


def _write_decision_field(d: dict, decision: dict, field: str, text: str | None, books: PanelBooks,
                          duration_s: float | None) -> None:
    if field == "kind":
        decision["kind"] = _choice(text, books.decision_kinds())
    elif field == "actor":
        actors = tuple(dict.fromkeys(a["actor_id"] for a in d["actors"]))
        if text is None or text.casefold() not in {a.casefold() for a in actors}:
            raise ValueError(f"{text!r} is not an actor ({', '.join(actors)})")
        decision["actor"] = next(a for a in actors if a.casefold() == text.casefold())
    elif field == "time":
        if text is None:
            raise ValueError("needs a clock time (M:SS)")
        decision["time_s"] = _clock(text, duration_s)
    else:
        decision["offer_id"] = text


def _write_decision(d: dict, decision: dict, ref: PathRef, text: str | None, books: PanelBooks,
                    duration_s: float | None = None) -> None:
    _write_decision_field(d, decision, ref.field, text, books, duration_s)
    _check_decision(d, decision, {ref.field}, books)


def _spans(text: str | None, duration_s: float | None = None) -> list[dict]:
    spans = []
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        ends = part.split("-")
        if len(ends) != 2:
            raise ValueError(f"{part!r} is not a clock range (M:SS-M:SS)")
        start, end = _clock(ends[0], duration_s), _clock(ends[1], duration_s)
        if end <= start:
            raise ValueError(f"{part!r} ends before it starts")
        spans.append({"start_s": start, "end_s": end})
    return spans


def _write(d: dict, ref: PathRef, path: str, text: str | None, books: PanelBooks,
           duration_s: float | None = None) -> None:
    if ref.unit == "prepared_window":
        d["prepared_spans"] = _spans(text, duration_s)
        return
    if ref.unit in ("pitch", PAIR):
        raise _no_path(path, "judged on a sheet and never written to the record")
    unit = _unit(d, ref, path)
    if ref.unit == "actor":
        if ref.field == "role":
            unit["role"] = _choice(text, ACTOR_ROLES)
        elif text is None:
            raise ValueError("needs a label")
        else:
            unit["label"] = text
    elif ref.unit == "turn":
        _write_turn(d, unit, ref, path, text, books)
    elif ref.unit == "exchange":
        _write_exchange(unit, ref, text, books)
    elif ref.unit == "decision":
        _write_decision(d, unit, ref, text, books, duration_s)
    elif ref.unit == "stage_a":
        if ref.unit_id not in books.stage_a_ids():
            raise _no_path(path, f"{ref.unit_id!r} is not a Stage A item")
        old = unit["value"]
        if text is None:
            unit["value"], unit["uncodable"] = None, True
        else:
            unit["value"], unit["uncodable"] = _integer(text, *books.stage_a.by_id(ref.unit_id).scale), False
        if unit["value"] != old and STAGE_A_CHANGED not in (unit.get("note") or ""):
            # the note was written for the old value; say so rather than leave it contradicting the new one
            unit["note"] = " ".join(x for x in ((unit.get("note") or "").strip(), STAGE_A_CHANGED) if x)
    elif ref.unit == "prep_pres":
        entry = _prepared_entry(books, ref.unit_id)
        if entry is None:
            raise _no_path(path, f"{ref.unit_id!r} is not a prepared presentation code")
        unit.update(_code_item(entry, text))
    elif ref.unit_id == ASK:
        unit["ask_amount_usd"], unit["ask_equity_pct"], unit["ask_missing"] = _ask(text)
    else:
        code_id = VENTURE_CODES[ref.unit_id]
        unit[ref.unit_id] = _choice(text, books.venture_values(code_id))


def _ask(text: str | None) -> tuple[int | None, float | None, str]:
    """(ask_amount_usd, ask_equity_pct, ask_missing) of a human ask: `<usd>;<pct>` with any `$`, thousands commas,
    blanks and `%`, or `missing:<reason>`."""
    reason = _missing_reason(text or "", "the ask")
    if reason is not None:
        return None, None, reason
    if text is None:
        raise ValueError(f"an empty value is not amount;equity (use {MISSING_PREFIX}<reason>)")
    found = ASK_RE.match(text)
    try:
        amount, equity = int(found.group(1).replace(",", "")), float(found.group(2))
    except (AttributeError, ValueError):
        raise ValueError(f"{text!r} is not amount;equity (whole US dollars;per cent, e.g. 200000;10) or "
                         f"{MISSING_PREFIX}<reason>") from None
    if not 0 <= equity <= 100:
        raise ValueError(f"equity {equity:g} is outside 0-100 per cent")
    return amount, equity, "none"


def _edit(record_dict: dict, path: str, change) -> object:
    """Run `change` on a copy; keep the result only when it succeeds and is still a record."""
    work = copy.deepcopy(record_dict)
    try:
        result = change(work)
        PanelRecord.model_validate(work)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ValueError(f"{path}: {first['msg']} ({'.'.join(str(p) for p in first['loc'])})") from None
    except ValueError as exc:
        text = str(exc)
        raise ValueError(text if text.startswith(f"{path}:") else f"{path}: {text}") from None
    record_dict.clear()
    record_dict.update(work)
    return result


def set_value(record_dict: dict, path: str, value: str | None, books: PanelBooks,
              duration_s: float | None = None) -> None:
    """Write one human value into a record dict. ValueError, with the path and the reason, changes nothing.

    A categorical field takes only a codebook value (any case); a code may instead be
    `missing:<reason>`; None or blank empties a field that can be empty. A turn's function can
    change only while its codes still fit (see set_function for a change that brings its codes).
    A decision's actor must have the role its kind needs, and a kind that moves an offer needs an
    offer id (see set_decision for changes that are valid only together). With `duration_s`, a
    decision time or a prepared window beyond the video, or written as H:MM:SS on a video shorter
    than an hour, is refused. A changed Stage A value marks its note. A prepared presentation count
    takes a whole number of 0 or more; the ask takes `<usd>;<pct>` or `missing:<reason>`. The pitch
    paths and pair judgements are judged, never written: "no such path".
    """
    ref = parse_path(path)
    _edit(record_dict, path, lambda work: _write(work, ref, path, _text(value), books, duration_s))


def set_decision(record_dict: dict, decision_id: int, fields: dict[str, str | None], books: PanelBooks,
                 duration_s: float | None = None) -> None:
    """Write several fields of one decision together ({"kind" | "actor" | "time" | "offer_id": value}).

    For corrections that are valid only jointly: a kind and the actor whose role it needs, or a kind
    that moves an offer and its offer id. ValueError changes nothing.
    """
    path = main_path("decision", decision_id)
    ref = parse_path(path)
    unknown = [f for f in fields if f not in DECISION_FIELDS]
    if unknown:
        raise ValueError(f"{path}: {', '.join(unknown)} is not a field of a decision ({', '.join(DECISION_FIELDS)})")

    def change(work: dict) -> None:
        decision = _unit(work, ref, path)
        for field in DECISION_FIELDS:
            if field in fields:
                _write_decision_field(work, decision, field, _text(fields[field]), books, duration_s)
        _check_decision(work, decision, set(fields), books)

    _edit(record_dict, path, change)


def set_function(record_dict: dict, path: str, function: str | None, codes: dict[str, str | None],
                 books: PanelBooks) -> list[str]:
    """Change a turn's function together with the code values the same import supplies for that turn.

    `path` is the turn's function path and `codes` maps code ids to human values. Every code the
    new function requires must be carried already or supplied, and a supplied code must be one the
    new function takes; otherwise ValueError names the codes. Codes the turn carried that the new
    function does not take are removed and returned.
    """
    ref = parse_path(path)
    if ref.unit != "turn" or ref.field != "function":
        raise _no_path(path, "not a turn function")
    supplied = {code: _text(text) for code, text in codes.items()}
    return _edit(record_dict, path,
                 lambda work: _write_function(_unit(work, ref, path), _text(function), supplied, books))
