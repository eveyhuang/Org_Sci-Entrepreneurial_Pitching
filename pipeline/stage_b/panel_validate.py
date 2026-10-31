"""Local validation of one PanelRecord against the codebooks. No I/O: objects in, report out.

Hard errors mean the record violates the contract; soft flags mean a reviewer should look.
Every message starts with a stable bracketed tag so reports can be counted by rule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pipeline.schemas.stage_b_panel import ADDRESS_TOKENS, PanelRecord
from pipeline.stage_b.panel_books import (CONCEPT_CODES, PanelBooks, canonical_value, creates_offer_id, is_other,
                                          other_name)

# decision kinds that belong to an offer and so must carry its offer_id
OFFER_MOVE_KINDS = ("offer", "joins_offer", "revises_offer", "withdraws_offer",
                    "founder_accepts", "founder_rejects", "founder_counters")
# turn functions that by definition react to an earlier turn
RESPONSE_FUNCTIONS = ("accepts", "challenges", "repair")
ANSWERABLE_FUNCTIONS = ("question", "challenges")
# who is expected to speak a turn of a given function (a soft flag otherwise); functions in neither
# tuple are open to anyone
INVESTOR_FUNCTIONS = ("question", "challenges", "accepts")
FOUNDER_FUNCTIONS = ("answer", "presents", "repair")
CODED_FUNCTIONS = ("question", "answer")
SUBSTANTIVE = "substantive"
NEGOTIATION = "negotiation"
# a question or an answer inside these kinds, or social talk inside a substantive exchange, is flagged
NOT_FOR_QUESTIONING = ("social", "demonstration")
SOCIAL_FUNCTIONS = ("social",)
MIN_COVERAGE = 0.75


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    flags: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def _dupes(ids: list) -> list:
    seen: set = set()
    out = []
    for i in ids:
        if i in seen and i not in out:
            out.append(i)
        seen.add(i)
    return out


def _case_flags(written: PanelRecord, r: PanelRecord, flags: list[str]) -> None:
    """One flag per structure label whose case or surrounding whitespace was repaired; `r` is the canonical form of `written`."""
    def note(where: str, old: str, new: str) -> None:
        if old != new:
            flags.append(f"[flag.value_case] {where} {old!r} → {new!r}")

    for a, b in zip(written.turns, r.turns):
        note(f"turn {a.turn_id}: function", a.function, b.function)
    for a, b in zip(written.exchanges, r.exchanges):
        note(f"exchange {a.exchange_id}: kind", a.kind, b.kind)
        note(f"exchange {a.exchange_id}: topic", a.topic, b.topic)
    for a, b in zip(written.decisions, r.decisions):
        note(f"decision {a.decision_id}: kind", a.kind, b.kind)


def _check_ids(r: PanelRecord, err: list[str]) -> None:
    for tag, ids in (("actor.id", [a.actor_id for a in r.actors]), ("turn.id", [t.turn_id for t in r.turns]),
                     ("exchange.id", [e.exchange_id for e in r.exchanges]), ("decision.id", [d.decision_id for d in r.decisions])):
        for d in _dupes(ids):
            err.append(f"[{tag}] duplicate id {d!r}")


def _check_times(r: PanelRecord, dur: float, tol: float, actors: set[str], err: list[str]) -> None:
    limit = dur + tol
    for t in r.turns:
        if t.speaker not in actors:
            err.append(f"[turn.speaker] turn {t.turn_id}: {t.speaker!r} is not an actor")
        if t.end_s < t.start_s or t.start_s > limit or t.end_s > limit:
            err.append(f"[turn.time] turn {t.turn_id}: {t.start_s}-{t.end_s} is reversed or beyond {dur}s")
    for a, b in zip(r.turns, r.turns[1:]):
        if b.start_s < a.start_s:
            err.append(f"[turn.order] turn {b.turn_id} starts before turn {a.turn_id}")
    for kind, spans in (("prepared", r.prepared_spans), ("excluded", r.excluded_spans)):
        for s in spans:
            if s.end_s <= s.start_s or s.start_s > limit or s.end_s > limit:
                err.append(f"[span.time] {kind} span {s.start_s}-{s.end_s} is empty, reversed or beyond {dur}s")


def _value_problem(entry, value: str) -> str | None:
    if entry.type == "categorical":
        return None if value in (entry.values or []) else f"{value!r} is not one of {entry.values}"
    if entry.type == "ordinal":
        lo, hi = entry.scale
        try:
            n = int(value)
        except ValueError:
            return f"{value!r} is not an integer"
        return None if lo <= n <= hi else f"{n} is outside {int(lo)}-{int(hi)}"
    if entry.type == "binary":
        return None if value in ("true", "false") else f"{value!r} is not true or false"
    return None


def _position(r: PanelRecord) -> dict[int, int]:
    """turn id -> place in the turn list, which is the order of the encounter."""
    pos: dict[int, int] = {}
    for i, t in enumerate(r.turns):
        pos.setdefault(t.turn_id, i)
    return pos


def _check_turns(r: PanelRecord, books: PanelBooks, actors: set[str], err: list[str], flags: list[str]) -> None:
    roles = {a.actor_id: a.role for a in r.actors}
    by_id = {}
    for t in r.turns:
        by_id.setdefault(t.turn_id, t)
    pos = _position(r)
    functions = books.turn_functions()
    exchange_ids = {e.exchange_id for e in r.exchanges}
    panel_ids = set(books.panel.ids)
    for index, t in enumerate(r.turns):
        tag = f"turn {t.turn_id}"
        if t.function not in functions:
            err.append(f"[turn.function] {tag}: {t.function!r} is not a TURN-FUNC value")
        for a in t.addressed:
            if a not in actors and a not in ADDRESS_TOKENS:
                err.append(f"[turn.addressed] {tag}: {a!r} is neither an actor nor PANEL/UNKNOWN")
        if t.exchange_id is not None and t.exchange_id not in exchange_ids:
            err.append(f"[turn.exchange] {tag}: exchange_id {t.exchange_id} is not a recorded exchange")
        for rid in t.responds_to:
            if rid not in pos:
                err.append(f"[turn.link] {tag}: responds_to {rid} does not exist")
            elif pos[rid] >= index:
                err.append(f"[turn.link] {tag}: responds_to {rid}, which is not an earlier turn")
        _check_codes(t, books, panel_ids, err, flags)
        _check_other_concept(t, err, flags)
        if t.function == "answer" and not any(
                by_id[i].function in ANSWERABLE_FUNCTIONS for i in t.responds_to if i in by_id):
            flags.append(f"[flag.answer_link] {tag}: answer responds to no question or challenges turn")
        role = roles.get(t.speaker)
        if role is not None:      # an unknown speaker is already a [turn.speaker] error
            for needed, group in (("investor", INVESTOR_FUNCTIONS), ("founder", FOUNDER_FUNCTIONS)):
                if t.function in group and role != needed:
                    flags.append(f"[flag.turn_role] {tag}: {t.function} is expected from "
                                 f"{'an' if needed == 'investor' else 'a'} {needed}, got {role} {t.speaker!r}")


def _check_codes(t, books: PanelBooks, panel_ids: set[str], err: list[str], flags: list[str]) -> None:
    tag = f"turn {t.turn_id}"
    for d in _dupes([c.code for c in t.codes]):
        err.append(f"[turn.codes] {tag}: duplicate code {d}")
    present = {c.code for c in t.codes}
    for c in t.codes:
        if c.code not in panel_ids:
            err.append(f"[turn.codes] {tag}: {c.code} is not a panel code")
            continue
        entry = books.panel.by_id(c.code)
        if entry.applies_to != t.function:
            err.append(f"[turn.codes] {tag}: {c.code} applies to {entry.applies_to!r}, not {t.function!r}")
        if c.value is not None:
            value = canonical_value(entry, c.value)
            problem = _value_problem(entry, value)
            if problem:
                err.append(f"[turn.codes] {tag}: {c.code} {problem}")
            elif value != c.value:
                flags.append(f"[flag.value_case] {tag}: {c.code} {c.value!r} → {value!r}")
    for entry in books.codes_for(t.function):
        if entry.id not in present:
            err.append(f"[turn.codes] {tag}: required code {entry.id} is absent")


def _check_other_concept(t, err: list[str], flags: list[str]) -> None:
    """A concept of `other` needs its name in other_concept and is listed for review; a name without one is flagged."""
    tag = f"turn {t.turn_id}"
    name = other_name(t.other_concept)
    others = []
    for c in t.codes:
        if c.code in CONCEPT_CODES and c.code not in others and is_other(c.value):
            others.append(c.code)
    for code_id in others:
        if not name:
            err.append(f"[turn.other_concept] {tag}: {code_id} is other but other_concept does not name the matter")
        flags.append(f"[flag.other_concept] {tag}: {code_id} other is " + (f'"{name}"' if name else "unnamed"))
    if t.other_concept is not None and not others:
        flags.append(f"[flag.other_concept_unused] {tag}: other_concept is {t.other_concept!r} but no concept "
                     "of the turn is other")


def _check_exchanges(r: PanelRecord, books: PanelBooks, err: list[str], flags: list[str]) -> None:
    kinds, topics = books.exchange_kinds(), books.exchange_topics()
    members: dict[int, list] = {}
    for t in r.turns:
        if t.exchange_id is not None:
            members.setdefault(t.exchange_id, []).append(t)
    for e in r.exchanges:
        tag = f"exchange {e.exchange_id}"
        turns = members.get(e.exchange_id, [])
        substantive = e.kind == SUBSTANTIVE
        if e.kind not in kinds:
            err.append(f"[exchange.kind] {tag}: {e.kind!r} is not an EX-KIND value a recorded exchange can have")
        if e.topic not in topics:
            err.append(f"[exchange.topic] {tag}: {e.topic!r} is not an EX-TOPIC value")
        elif substantive and e.topic == "NONE":
            err.append(f"[exchange.topic] {tag}: a substantive exchange needs a topic, got NONE")
        elif not substantive and e.topic != "NONE":
            err.append(f"[exchange.topic] {tag}: topic is {e.topic!r} but the exchange is not substantive")
        elif is_other(e.topic) and not other_name(e.label):
            err.append(f"[exchange.topic] {tag}: topic is other but the label does not name the matter")
        if is_other(e.topic):
            name = other_name(e.label)
            flags.append(f"[flag.other_concept] {tag}: EX-TOPIC other is " + (f'"{name}"' if name else "unnamed"))
        if not turns:
            err.append(f"[exchange.empty] {tag}: no turn points to it")
        if substantive and len(turns) == 1:
            flags.append(f"[flag.single_turn_exchange] {tag}: substantive exchange of one turn")


def _check_decisions(r: PanelRecord, books: PanelBooks, dur: float, tol: float, err: list[str]) -> None:
    roles = {a.actor_id: a.role for a in r.actors}
    turn_ids = {t.turn_id for t in r.turns}
    offer_ids = {d.offer_id for d in r.decisions if creates_offer_id(d.kind, d.offer_id)}
    for d in r.decisions:
        tag = f"decision {d.decision_id}"
        if d.kind not in books.decision_kinds():
            err.append(f"[decision.kind] {tag}: {d.kind!r} is not a DEC-KIND value")
        role = roles.get(d.actor)
        if role is None:
            err.append(f"[decision.actor] {tag}: unknown actor {d.actor!r}")
        elif d.kind.startswith("founder_"):
            if role != "founder":
                err.append(f"[decision.actor] {tag}: {d.kind} needs a founder, got {role}")
        elif role != "investor":
            err.append(f"[decision.actor] {tag}: {d.kind} needs an investor, got {role}")
        if d.time_s > dur + tol:
            err.append(f"[decision.time] {tag}: {d.time_s}s is beyond {dur}s")
        if d.turn_id is not None and d.turn_id not in turn_ids:
            err.append(f"[decision.turn] {tag}: unknown turn_id {d.turn_id}")
        if d.kind in OFFER_MOVE_KINDS:
            if d.offer_id is None:
                err.append(f"[decision.offer] {tag}: {d.kind} has no offer_id")
            elif not creates_offer_id(d.kind, d.offer_id) and d.offer_id not in offer_ids:
                err.append(f"[decision.offer] {tag}: {d.kind} names offer_id {d.offer_id!r}, which no offer decision carries")


def _check_stage_a(r: PanelRecord, books: PanelBooks, tol: float, err: list[str]) -> None:
    codes = [i.code for i in r.stage_a]
    expected = set(books.stage_a_ids())
    if sorted(codes) != sorted(expected):
        missing = sorted(expected - set(codes))
        extra = sorted(set(codes) - expected)
        dup = _dupes(codes)
        err.append(f"[stage_a.codes] missing {missing}, extra {extra}, duplicate {dup}")
    for i in r.stage_a:
        if not i.uncodable and not r.prepared_spans:
            err.append(f"[stage_a.no_prepared] {i.code} is rated but there is no prepared span")
            continue
        if i.moment_s is not None and not any(
                s.start_s - tol <= i.moment_s <= s.end_s + tol for s in r.prepared_spans):
            err.append(f"[stage_a.moment] {i.code}: {i.moment_s}s is outside every prepared span")


def _whole_number(text: str) -> bool:
    text = text.strip()
    return text.isascii() and text.isdigit()


def _check_prepared(r: PanelRecord, books: PanelBooks, err: list[str]) -> None:
    """Every prepared presentation code exactly once, with a valid value; all not_applicable without a presentation."""
    entries = {e.id: e for e in books.prepared_codes()}
    codes = [c.code for c in r.prepared]
    for code_id in entries:
        n = codes.count(code_id)
        if n == 0:
            err.append(f"[prepared] {code_id} is not recorded")
        elif n > 1:
            err.append(f"[prepared] {code_id} is recorded twice")
    for c in r.prepared:
        entry = entries.get(c.code)
        if entry is None:
            err.append(f"[prepared] {c.code} is not a prepared presentation code")
            continue
        if not r.prepared_spans:
            if c.value is not None:
                err.append(f"[prepared] {c.code}: a value with no prepared presentation; use missing not_applicable")
            elif c.missing != "not_applicable":
                err.append(f"[prepared] {c.code}: missing reason {c.missing} with no prepared presentation; "
                           "use not_applicable")
            continue
        if c.value is None:
            continue
        if entry.type == "count":
            if not _whole_number(c.value):
                err.append(f"[prepared] {c.code}: value must be a whole number of 0 or more")
            continue
        problem = _value_problem(entry, canonical_value(entry, c.value))
        if problem:
            err.append(f"[prepared] {c.code}: {problem}")


def _check_venture(r: PanelRecord, books: PanelBooks, err: list[str], flags: list[str]) -> None:
    v = r.venture
    fields = [("VC-INDUSTRY", "primary_industry", v.primary_industry), ("VC-FORM", "product_form", v.product_form),
              ("VC-CUSTOMER", "customer_type", v.customer_type), ("VC-GENDER", "founder_gender", v.founder_gender)]
    fields += [("VC-INDUSTRY", "secondary_industries", s) for s in v.secondary_industries]
    for code_id, name, raw in fields:
        value = books.canonical_venture(code_id, raw)
        if value not in books.venture_values(code_id):
            err.append(f"[venture.value] {name}: {raw!r} is not a {code_id} value")
            continue
        if value != raw:
            flags.append(f"[flag.value_case] venture {name}: {raw!r} → {value!r}")
        if value == "NONE":
            flags.append(f"[flag.venture_unknown] {name} is NONE")
    asked = (v.ask_amount_usd, v.ask_equity_pct)
    if v.ask_missing == "none" and None in asked:
        err.append("[venture.ask] amount and equity must both be given unless ask_missing is not none")
    elif v.ask_missing != "none" and asked != (None, None):
        err.append(f"[venture.ask] values are given although ask_missing is {v.ask_missing!r}")


def _check_claims(r: PanelRecord, err: list[str]) -> None:
    pos = _position(r)
    for c in r.claim_conflicts:
        tag = f"conflict {c.turn_a}-{c.turn_b}"
        unknown = [name for name, t in (("turn_a", c.turn_a), ("turn_b", c.turn_b)) if t not in pos]
        for name in unknown:
            err.append(f"[claim.turn] {tag}: {name} is not a recorded turn")
        if not unknown and pos[c.turn_a] >= pos[c.turn_b]:
            err.append(f"[claim.turn] {tag}: turn_a is not earlier than turn_b")
        if c.reconciled_turn is None:
            continue
        if c.reconciled_turn not in pos:
            err.append(f"[claim.turn] {tag}: reconciled_turn {c.reconciled_turn} is not a recorded turn")
        elif c.turn_b in pos and pos[c.reconciled_turn] <= pos[c.turn_b]:
            err.append(f"[claim.turn] {tag}: reconciled_turn {c.reconciled_turn} is not later than turn_b")


def _decision_flags(r: PanelRecord, flags: list[str]) -> None:
    """Where decisions sit among the turns and exchanges: a decision is spoken in a decides turn, an
    exit belongs to no exchange, and offer moves belong to the one negotiation exchange."""
    turns = {}
    for t in r.turns:
        turns.setdefault(t.turn_id, t)
    kind_of = {}
    for e in r.exchanges:
        kind_of.setdefault(e.exchange_id, e.kind)
    named = set()
    for d in r.decisions:
        turn = turns.get(d.turn_id)
        if turn is None:
            continue      # no turn_id is a [flag.decision_turn]; an unknown one is a [decision.turn] error
        named.add(turn.turn_id)
        tag = f"decision {d.decision_id}"
        kind = kind_of.get(turn.exchange_id)
        if turn.function != "decides":
            flags.append(f"[flag.decision_function] {tag}: turn {turn.turn_id} has function {turn.function!r}, "
                         "not decides")
        if d.kind == "exit" and turn.exchange_id is not None and kind != NEGOTIATION:
            flags.append(f"[flag.exit_in_exchange] {tag}: exit turn {turn.turn_id} belongs to exchange "
                         f"{turn.exchange_id}, which is not the negotiation")
        if d.kind in OFFER_MOVE_KINDS and kind != NEGOTIATION:
            flags.append(f"[flag.offer_outside_negotiation] {tag}: {d.kind} turn {turn.turn_id} is not in a "
                         "negotiation exchange")
    for t in r.turns:
        if t.function == "decides" and t.turn_id not in named:
            flags.append(f"[flag.decides_without_decision] turn {t.turn_id}: no decision names this decides turn")
    negotiations = [e.exchange_id for e in r.exchanges if e.kind == NEGOTIATION]
    if len(negotiations) > 1:
        flags.append(f"[flag.multiple_negotiations] exchanges {negotiations} are all negotiation; there is one "
                     "negotiation exchange per encounter")


def _flags(r: PanelRecord, dur: float, tol: float, flags: list[str]) -> None:
    answered = {i for t in r.turns if t.function == "answer" for i in t.responds_to}
    kind_of = {}
    for e in r.exchanges:
        kind_of.setdefault(e.exchange_id, e.kind)
    for t in r.turns:
        tag = f"turn {t.turn_id}"
        if t.function == "question" and t.turn_id not in answered:
            flags.append(f"[flag.unanswered_question] {tag}: no answer responds to it")
        if t.function in RESPONSE_FUNCTIONS and not t.responds_to:
            flags.append(f"[flag.unlinked_response] {tag}: {t.function} responds to no recorded turn")
        if t.function in CODED_FUNCTIONS and t.exchange_id is None:
            flags.append(f"[flag.orphan_turn] {tag}: {t.function} belongs to no exchange")
        kind = kind_of.get(t.exchange_id)
        if (t.function in CODED_FUNCTIONS and kind in NOT_FOR_QUESTIONING) or (
                t.function in SOCIAL_FUNCTIONS and kind == SUBSTANTIVE):
            flags.append(f"[flag.kind_function_mismatch] {tag}: {t.function} inside {kind} exchange {t.exchange_id}")
        if "UNKNOWN" in t.addressed:
            flags.append(f"[flag.unknown_addressee] {tag}: addressee unknown")
        if t.function == "demonstrates" and t.exchange_id is None:
            flags.append(f"[flag.demonstration_without_exchange] {tag}: demonstrates belongs to no exchange")
    if r.prepared_spans and not any(t.function == "presents" for t in r.turns):
        flags.append("[flag.no_presents_turn] there is a prepared span but no presents turn")
    turns = {t.turn_id: t for t in r.turns}
    for d in r.decisions:
        tag = f"[flag.decision_turn] decision {d.decision_id}"
        if d.turn_id is None:
            flags.append(f"{tag}: no turn_id")
            continue
        turn = turns.get(d.turn_id)
        if turn is None:
            continue  # already a [decision.turn] error
        if turn.speaker != d.actor:
            flags.append(f"{tag}: actor {d.actor!r} is not the speaker of turn {d.turn_id} ({turn.speaker!r})")
        elif not turn.start_s - tol <= d.time_s <= turn.end_s + tol:
            flags.append(f"{tag}: {d.time_s}s is outside turn {d.turn_id} ({turn.start_s}-{turn.end_s})")
    _decision_flags(r, flags)
    last_end = max((t.end_s for t in r.turns), default=0.0)
    if last_end < MIN_COVERAGE * dur:
        flags.append(f"[flag.short_coverage] last turn ends at {last_end}s, before {MIN_COVERAGE:.0%} of {dur}s")
    deciders = {d.actor for d in r.decisions}
    for a in r.actors:
        if a.role == "investor" and a.actor_id not in deciders:
            flags.append(f"[flag.no_decision] investor {a.actor_id} has no decision")
    for g in r.gaps:
        flags.append(f"[flag.gap] gap at {g.at_s}s")
    if not r.source_complete:
        flags.append("[flag.incomplete_source] source is not complete")
    if not r.prepared_spans:
        flags.append("[flag.no_prepared] no prepared span")


def validate_record(record: PanelRecord, books: PanelBooks, duration_s: float, tolerance_s: float = 2.0,
                    normalizations: Sequence[str] = ()) -> ValidationReport:
    """`normalizations` are the notes of the wire conversion (values dropped because they came with a
    missing reason); each becomes a soft flag so the report shows what the record no longer does."""
    errors: list[str] = []
    flags: list[str] = [f"[flag.value_with_missing] {note}" for note in normalizations]
    written, record = record, books.canonical_record(record)   # every rule below reads the canonical labels
    _case_flags(written, record, flags)
    actors = {a.actor_id for a in record.actors}
    _check_ids(record, errors)
    _check_times(record, duration_s, tolerance_s, actors, errors)
    _check_turns(record, books, actors, errors, flags)
    _check_exchanges(record, books, errors, flags)
    _check_decisions(record, books, duration_s, tolerance_s, errors)
    _check_stage_a(record, books, tolerance_s, errors)
    _check_prepared(record, books, errors)
    _check_venture(record, books, errors, flags)
    _check_claims(record, errors)
    _flags(record, duration_s, tolerance_s, flags)
    return ValidationReport(tuple(errors), tuple(flags))
