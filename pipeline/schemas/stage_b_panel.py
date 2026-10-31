"""Contract for the v3 panel annotation: one record per full-video pass.

PanelRecord is the normalized record (times in seconds) that validation, alignment and
comparison consume. PanelWire, at the end of this file, is what the provider is asked for:
the same record with times as M:SS clock strings, converted by wire_to_record.

The model fills every field while seeing the whole video. Code values travel as strings and
are checked against the codebooks locally (pipeline.stage_b.panel_validate), which keeps this
schema small enough for the provider and keeps the codebook the only place a code is defined.
Each turn carries its own function and codes; exchanges group turns into the back-and-forth
about one matter. Decisions are a separate list so analysis can withhold them later.
"""

from __future__ import annotations

from typing import Collection, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MissingReason = Literal[
    "none", "not_applicable", "no_opportunity", "not_observable", "edited_gap", "ambiguous", "too_short"
]
MISSING_REASONS = ("none", "not_applicable", "no_opportunity", "not_observable", "edited_gap", "ambiguous", "too_short")
ADDRESS_TOKENS = ("PANEL", "UNKNOWN")
ANNOTATION_CONTEXT = "descriptive_full_context"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Actor(Strict):
    actor_id: str = Field(min_length=1, description="F1.. founders, S1.. investors, H1.. host or narrator, in order of first appearance")
    role: Literal["founder", "investor", "host", "other"]
    label: str = Field(min_length=1, max_length=120, description="name if spoken or shown, else a brief visual description")


class Span(Strict):
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)


class ExcludedSpan(Span):
    kind: Literal["narration", "teaser", "interview", "recap", "other"]


class Gap(Strict):
    at_s: float = Field(ge=0)
    note: str = Field(max_length=200)


class CodeValue(Strict):
    code: str
    value: str | None = Field(description="categorical value, ordinal integer as text, or true/false; null only when missing is not none")
    missing: MissingReason

    @model_validator(mode="after")
    def _value_xor_missing(self) -> "CodeValue":
        if (self.value is None) == (self.missing == "none"):
            raise ValueError(f"{self.code}: value must be null exactly when missing is not 'none'")
        return self


_TURN_FUNCTION = "one TURN-FUNC value; NONE when the turn has none of the functions"
_TURN_ADDRESSED = "actor ids, or PANEL, or UNKNOWN; empty for the prepared presentation"
_TURN_EXCHANGE = "id of the exchange this turn belongs to; null when it belongs to none"
_TURN_RESPONDS = "ids of earlier turns this one responds to; empty when none is supported"
_TURN_OTHER = "a few words naming the matter when the turn's PR-CONCEPT or RS-CONCEPT is other; otherwise null"
_TURN_SUMMARY = "neutral summary, about 20 words, no quotes"


class Turn(Strict):
    turn_id: int = Field(ge=1)
    speaker: str
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    function: str = Field(description=_TURN_FUNCTION)
    addressed: list[str] = Field(description=_TURN_ADDRESSED)
    exchange_id: int | None = Field(ge=1, description=_TURN_EXCHANGE)
    responds_to: list[int] = Field(description=_TURN_RESPONDS)
    codes: list[CodeValue]
    other_concept: str | None = Field(max_length=80, description=_TURN_OTHER)
    summary: str = Field(min_length=1, max_length=240, description=_TURN_SUMMARY)


class Exchange(Strict):
    exchange_id: int = Field(ge=1)
    kind: str = Field(description="one EX-KIND value other than NONE")
    topic: str = Field(description="one EX-TOPIC value; NONE unless the exchange is substantive")
    label: str = Field(max_length=80, description="a few words naming the specific matter; required when the topic is other")


class Decision(Strict):
    decision_id: int = Field(ge=1)
    actor: str
    kind: str = Field(description="one DEC-KIND value")
    time_s: float = Field(ge=0)
    turn_id: int | None = Field(description="turn in which the decision is spoken")
    offer_id: str | None = Field(description="shared by all moves of the same offer, e.g. O1; null for an exit")
    terms: str = Field(max_length=200)
    note: str = Field(max_length=200)


class ClaimConflict(Strict):
    turn_a: int = Field(ge=1, description="turn holding the earlier claim")
    turn_b: int = Field(ge=1, description="turn holding the later claim")
    metric: str = Field(max_length=120)
    note: str = Field(max_length=200)
    reconciled: bool
    reconciled_turn: int | None = Field(ge=1, description="turn that reconciled the pair; null when there is none")

    @model_validator(mode="after")
    def _reconciling_turn_needs_reconciled(self) -> "ClaimConflict":
        if self.reconciled_turn is not None and not self.reconciled:
            raise ValueError("reconciled_turn is given only when reconciled is true")
        return self


class StageAItem(Strict):
    code: str
    value: int | None = Field(default=None, ge=1, le=5)
    uncodable: bool
    moment_s: float | None = Field(ge=0, description="source second of the single moment that best shows the rating; null when there is none")
    note: str = Field(max_length=200)

    @model_validator(mode="after")
    def _value_xor_uncodable(self) -> "StageAItem":
        if self.uncodable == (self.value is not None):
            raise ValueError(f"{self.code}: value must be null exactly when uncodable is true")
        return self


class Venture(Strict):
    description: str = Field(min_length=1, max_length=300)
    primary_industry: str
    secondary_industries: list[str]
    product_form: str
    customer_type: str
    founder_gender: str = Field(description="one VC-GENDER value: the presenting team's gender composition")
    product_category: str = Field(max_length=120)
    earliest_s: float | None = Field(default=None, ge=0)
    note: str = Field(max_length=200)
    ask_amount_usd: int | None = Field(default=None, ge=0)
    ask_equity_pct: float | None = Field(default=None, ge=0, le=100)
    ask_missing: MissingReason = "none"


class PanelRecord(Strict):
    actors: list[Actor]
    prepared_spans: list[Span]
    excluded_spans: list[ExcludedSpan]
    gaps: list[Gap]
    source_complete: bool
    stage_a: list[StageAItem]
    prepared: list[CodeValue]
    venture: Venture
    turns: list[Turn]
    exchanges: list[Exchange]
    decisions: list[Decision]
    claim_conflicts: list[ClaimConflict]
    notes: str = Field(max_length=600)


# ---- wire form ----------------------------------------------------------------------------
# The provider is asked for clock times (M:SS) because models write raw second counts
# unreliably (1:48 came back as 148). These classes are the request schema only; a response is
# converted with wire_to_record and everything downstream consumes PanelRecord in seconds.
#
# The wire is also more tolerant than the record in one respect: a model sometimes gives a value
# together with a missing reason (or a Stage A value together with uncodable). That is accepted
# here, and the conversion drops the value and reports it, so the record keeps "missing is not
# low" strictly. The opposite (no value and no reason) is still an invalid response.
#
# A model also sometimes writes the word instead of leaving the value empty ("missing", "n/a" as
# the value of an ordinal code, with missing "none"). Such a value is read as missing too.

TIME_FORMAT = "clock_on_wire_seconds_in_record"
MISSING_WORDS = ("missing", "null", "n/a", "na", "not applicable")   # never a code value, in any case
RESIDUAL_WORD = "none"      # a missing word too, except on a categorical code, where NONE can be a value
WORD_MISSING_REASON = "not_applicable"
_CLOCK = "clock time M:SS from the start of the video file, e.g. 1:48"


def _digits(text: str) -> bool:
    return text.isascii() and text.isdigit()


def clock_to_seconds(text: str) -> float:
    """'M:SS', 'MM:SS' or 'H:MM:SS', optionally with fractional seconds, as seconds.

    The seconds field has exactly two digits before any fraction: '1:5' is not a clock time.
    """
    parts = text.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"{text!r} is not a clock time (M:SS)")
    *head, last = parts
    whole, dot, fraction = last.partition(".")
    if (not all(_digits(p) for p in head) or not _digits(whole) or len(whole) != 2
            or (dot and not _digits(fraction))):
        raise ValueError(f"{text!r} is not a clock time (M:SS)")
    seconds = float(last)
    if seconds >= 60:
        raise ValueError(f"{text!r}: seconds must be below 60")
    minutes = int(head[-1])
    hours = int(head[0]) if len(head) == 2 else 0
    if len(head) == 2 and minutes >= 60:
        raise ValueError(f"{text!r}: minutes must be below 60 when hours are given")
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_clock(seconds: float) -> str:
    """M:SS, or H:MM:SS from one hour, to the nearest second."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class WireSpan(Strict):
    start: str = Field(description=_CLOCK)
    end: str = Field(description=_CLOCK)


class WireExcludedSpan(WireSpan):
    kind: Literal["narration", "teaser", "interview", "recap", "other"]


class WireGap(Strict):
    at: str = Field(description=_CLOCK)
    note: str = Field(max_length=200)


class WireCodeValue(Strict):
    code: str
    value: str | None = Field(description="categorical value, ordinal integer as text, or true/false; null only when missing is not none")
    missing: MissingReason

    @model_validator(mode="after")
    def _missing_none_needs_a_value(self) -> "WireCodeValue":
        if self.value is None and self.missing == "none":
            raise ValueError(f"{self.code}: a null value needs a missing reason other than 'none'")
        return self


class WireTurn(Strict):
    turn_id: int = Field(ge=1)
    speaker: str
    start: str = Field(description=_CLOCK)
    end: str = Field(description=_CLOCK)
    function: str = Field(description=_TURN_FUNCTION)
    addressed: list[str] = Field(description=_TURN_ADDRESSED)
    exchange_id: int | None = Field(ge=1, description=_TURN_EXCHANGE)
    responds_to: list[int] = Field(description=_TURN_RESPONDS)
    codes: list[WireCodeValue]
    other_concept: str | None = Field(max_length=80, description=_TURN_OTHER)
    summary: str = Field(min_length=1, max_length=240, description=_TURN_SUMMARY)


class WireDecision(Strict):
    decision_id: int = Field(ge=1)
    actor: str
    kind: str = Field(description="one DEC-KIND value")
    time: str = Field(description=_CLOCK)
    turn_id: int | None = Field(description="turn in which the decision is spoken")
    offer_id: str | None = Field(description="shared by all moves of the same offer, e.g. O1; null for an exit")
    terms: str = Field(max_length=200)
    note: str = Field(max_length=200)


class WireStageAItem(Strict):
    code: str
    value: int | None = Field(default=None, ge=1, le=5)
    uncodable: bool
    moment: str | None = Field(description=f"{_CLOCK}; the single moment that best shows the rating, null when there is none")
    note: str = Field(max_length=200)

    @model_validator(mode="after")
    def _rated_item_needs_a_value(self) -> "WireStageAItem":
        if self.value is None and not self.uncodable:
            raise ValueError(f"{self.code}: a null value needs uncodable to be true")
        return self


class WireVenture(Strict):
    description: str = Field(min_length=1, max_length=300)
    primary_industry: str
    secondary_industries: list[str]
    product_form: str
    customer_type: str
    founder_gender: str = Field(description="one VC-GENDER value: the presenting team's gender composition")
    product_category: str = Field(max_length=120)
    earliest: str | None = Field(default=None, description=_CLOCK)
    note: str = Field(max_length=200)
    # Required but nullable on the wire, so the model answers the ask every time.
    ask_amount_usd: int | None = Field(ge=0, description="VC-ASK: whole US dollars asked, or null with ask_missing")
    ask_equity_pct: float | None = Field(ge=0, le=100,
                                         description="VC-ASK: per cent of equity offered, or null with ask_missing")
    ask_missing: MissingReason = Field(description="none when both ask values are given, otherwise why not")


class PanelWire(Strict):
    actors: list[Actor]
    prepared_spans: list[WireSpan]
    excluded_spans: list[WireExcludedSpan]
    gaps: list[WireGap]
    source_complete: bool
    stage_a: list[WireStageAItem]
    prepared: list[WireCodeValue]
    venture: WireVenture
    turns: list[WireTurn]
    exchanges: list[Exchange]
    decisions: list[WireDecision]
    claim_conflicts: list[ClaimConflict]
    notes: str = Field(max_length=600)


def _is_missing_word(code: str, value: str, categorical_codes: Collection[str] | None) -> bool:
    word = value.strip().casefold()
    if word in MISSING_WORDS:
        return True
    return word == RESIDUAL_WORD and categorical_codes is not None and code not in categorical_codes


def _normalized_code(path: str, code: str, value: str | None, missing: str,
                     categorical_codes: Collection[str] | None, notes: list[str]) -> dict:
    """One code item with "missing is not low" restored; what was changed is appended to `notes`."""
    if value is not None and missing != "none":
        notes.append(f"{path} {code}: value {value!r} dropped because missing is {missing!r}")
        value = None
    elif value is not None and _is_missing_word(code, value, categorical_codes):
        notes.append(f"{path} {code}: value {value!r} read as missing, reason set to {WORD_MISSING_REASON!r}")
        value, missing = None, WORD_MISSING_REASON
    return {"code": code, "value": value, "missing": missing}


def normalize_missing_words(record: dict, categorical_codes: Collection[str] | None = None) -> list[str]:
    """Repair, in place, the turn codes of a record dict whose value is a word for missing.

    For records stored before the wire conversion did this. Returns one note per repaired code;
    a record without such a value is left untouched and gives no note.
    """
    notes: list[str] = []
    for i, turn in enumerate(record["turns"]):
        turn["codes"] = [_normalized_code(f"turns[{i}].codes", c["code"], c["value"], c["missing"],
                                          categorical_codes, notes) for c in turn["codes"]]
    return notes


def wire_to_record(wire: PanelWire) -> PanelRecord:
    """The normalized record of a wire response. ValueError names the first malformed time."""
    return wire_to_record_with_notes(wire)[0]


def wire_to_record_with_notes(wire: PanelWire,
                              categorical_codes: Collection[str] | None = None) -> tuple[PanelRecord, list[str]]:
    """The normalized record, and one note per code value it had to turn into a missing value.

    That is a value given together with a missing reason, and a value that is itself a word for
    missing (MISSING_WORDS in any case). `categorical_codes` are the ids of the categorical codes:
    when given, the word "none" is read as missing as well on every other code. On a categorical
    code NONE can be the codebook's residual value, so it is never touched there; without the set
    no code's type is known and "none" is left alone everywhere.
    """
    notes: list[str] = []

    def secs(path: str, text: str | None) -> float | None:
        if text is None:
            return None
        try:
            return clock_to_seconds(text)
        except ValueError as exc:
            raise ValueError(f"{path}: {exc}") from None

    def span(path: str, s: WireSpan) -> dict:
        return {"start_s": secs(f"{path}.start", s.start), "end_s": secs(f"{path}.end", s.end)}

    def stage_a_value(i: int, item: WireStageAItem) -> int | None:
        if item.uncodable and item.value is not None:
            notes.append(f"stage_a[{i}] {item.code}: value {item.value} dropped because uncodable is true")
            return None
        return item.value

    def codes(i: int, turn: WireTurn) -> list[dict]:
        return [_normalized_code(f"turns[{i}].codes", c.code, c.value, c.missing, categorical_codes, notes)
                for c in turn.codes]

    data = {
        "actors": [a.model_dump() for a in wire.actors],
        "prepared_spans": [span(f"prepared_spans[{i}]", s) for i, s in enumerate(wire.prepared_spans)],
        "excluded_spans": [{**span(f"excluded_spans[{i}]", s), "kind": s.kind}
                           for i, s in enumerate(wire.excluded_spans)],
        "gaps": [{"at_s": secs(f"gaps[{i}].at", g.at), "note": g.note} for i, g in enumerate(wire.gaps)],
        "source_complete": wire.source_complete,
        "stage_a": [{"code": x.code, "value": stage_a_value(i, x), "uncodable": x.uncodable, "note": x.note,
                     "moment_s": secs(f"stage_a[{i}].moment", x.moment)}
                    for i, x in enumerate(wire.stage_a)],
        "prepared": [_normalized_code(f"prepared[{i}]", c.code, c.value, c.missing, categorical_codes, notes)
                     for i, c in enumerate(wire.prepared)],
        "venture": {**wire.venture.model_dump(exclude={"earliest"}),
                    "earliest_s": secs("venture.earliest", wire.venture.earliest)},
        "turns": [{**t.model_dump(exclude={"start", "end", "codes"}), "codes": codes(i, t), **span(f"turns[{i}]", t)}
                  for i, t in enumerate(wire.turns)],
        "exchanges": [e.model_dump() for e in wire.exchanges],
        "decisions": [{**d.model_dump(exclude={"time"}), "time_s": secs(f"decisions[{i}].time", d.time)}
                      for i, d in enumerate(wire.decisions)],
        "claim_conflicts": [c.model_dump() for c in wire.claim_conflicts],
        "notes": wire.notes,
    }
    return PanelRecord.model_validate(data), notes
