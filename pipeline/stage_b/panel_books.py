"""The three codebooks behind the panel prompt, and the one prompt rendered from them."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pipeline.codebook import Codebook, CodeEntry, load_codebook
from pipeline.errors import ConfigError
from pipeline.instrument import InstrumentRelease
from pipeline.prompts import render_prompt
from pipeline.schemas.stage_b_panel import PanelRecord, seconds_to_clock

PANEL_TEMPLATE = "panel_full.j2"
# The model records the concept of a question or an answer; its regulatory frame is derived here
# (Kanze et al. 2018, Appendix B). The two tuples are the PR-CONCEPT / RS-CONCEPT / EX-TOPIC values minus
# deal_terms, other and NONE.
PROMOTION_CONCEPTS = ("customer_acquisition", "go_to_market", "market_opportunity", "assets", "self_promotion",
                      "sales", "usage", "vision", "forecast")
PREVENTION_CONCEPTS = ("execution", "safety_security", "liability", "competition", "operating_efficiency",
                       "customer_retention", "team_capabilities")
CONCEPT_CODE_OF = {"question": "PR-CONCEPT", "answer": "RS-CONCEPT"}   # turn function -> its concept code
CONCEPT_CODES = tuple(CONCEPT_CODE_OF.values())
OTHER = "other"                # a real business matter outside the sixteen concepts; the record names it
UNCLASSIFIED = "unclassified"  # the frame of `other` and `deal_terms`: the concept list does not say
NO_FRAME_CONCEPTS = (OTHER, "deal_terms")   # deal_terms is this project's own concept, outside Kanze's sixteen


OFFER_MAKING_KINDS = ("offer", "joins_offer")


def _has_offer_id(offer_id) -> bool:
    return isinstance(offer_id, str) and bool(offer_id.strip())


def makes_offer(kind, offer_id=None) -> bool:
    """A decision makes an offer when it is an offer or a joined offer, or a re-entry that carries an offer id
    (the investor came back with a new offer and the model recorded it as one decision)."""
    return kind in OFFER_MAKING_KINDS or (kind == "re_entry" and _has_offer_id(offer_id))


OFFER_FOLLOW_UP_KINDS = ("revises_offer", "withdraws_offer")


def implies_offer(kind, offer_id=None) -> bool:
    """A decision that shows its investor made an offer at some point: one that makes an offer, or one that
    revises or withdraws an offer, which must have existed even when the record does not hold the move that
    made it."""
    return makes_offer(kind, offer_id) or kind in OFFER_FOLLOW_UP_KINDS


def creates_offer_id(kind, offer_id=None) -> bool:
    """A decision that brings its offer id into being: an offer, or an offer-making re-entry."""
    return (kind == "offer" and offer_id is not None) or (kind == "re_entry" and _has_offer_id(offer_id))


def frame_of(concept: str | None) -> str | None:
    """The frame a concept stands for: promotion, prevention, NEUTRAL for NONE.

    `other` and `deal_terms` have no frame in Kanze et al.'s list; both give the one value UNCLASSIFIED.

    None when there is no concept.

    Case and surrounding whitespace are ignored; a value that is not a concept gives None.
    """
    if concept is None:
        return None
    folded = concept.strip().casefold()
    if folded in PROMOTION_CONCEPTS:
        return "promotion"
    if folded in PREVENTION_CONCEPTS:
        return "prevention"
    if folded in NO_FRAME_CONCEPTS:
        return UNCLASSIFIED
    return "NEUTRAL" if folded == "none" else None


def is_other(value: str | None) -> bool:
    """Whether a concept or topic value is `other`, whatever its case or surrounding whitespace."""
    return value is not None and value.strip().casefold() == OTHER


def other_name(text: str | None) -> str:
    """A name given for an `other` matter with its whitespace collapsed; empty when none was given."""
    return " ".join((text or "").split())


def other_concepts(record: dict) -> list[dict]:
    """Every `other` in a record dict, turns first (in turn order) and then exchanges, each with its name.

    A turn's name is its other_concept, an exchange's is its label; the clock is the start of the
    turn, or of the exchange's first turn (None when no turn points to it). An unnamed item has
    an empty name. One item per concept code of a turn.
    """
    out = []
    first_start: dict[int, float] = {}
    for t in record["turns"]:
        if t["exchange_id"] is not None:
            first_start.setdefault(t["exchange_id"], t["start_s"])
        seen = set()
        for c in t["codes"]:
            if c["code"] in CONCEPT_CODES and c["code"] not in seen and is_other(c["value"]):
                out.append({"where": "turn", "id": t["turn_id"], "code": c["code"],
                            "name": other_name(t["other_concept"]), "clock": seconds_to_clock(t["start_s"])})
            seen.add(c["code"])
    for e in record["exchanges"]:
        if is_other(e["topic"]):
            start = first_start.get(e["exchange_id"])
            out.append({"where": "exchange", "id": e["exchange_id"], "code": "EX-TOPIC", "name": other_name(e["label"]),
                        "clock": None if start is None else seconds_to_clock(start)})
    return out


def turn_frame(function: str, codes: Iterable[tuple[str, str | None]]) -> str | None:
    """The derived frame of a question or answer turn from its (code id, value) items; None for any other turn."""
    wanted = CONCEPT_CODE_OF.get(function)
    for code_id, value in codes:
        if wanted is not None and code_id == wanted:
            return frame_of(value)
    return None

# The structural codes of the panel book, by the role they play in a record (the field classes of review_items).
STRUCTURE_CODES = {"turn_function": "TURN-FUNC", "exchange_kind": "EX-KIND", "exchange_topic": "EX-TOPIC",
                   "decision": "DEC-KIND"}

_REQUIRED_BOOKS = ("stage_b_panel", "stage_a", "venture_context")
# the applies_to of the codes rated once per pitch on the prepared presentation; it is not a turn function, so no
# turn requires them
PREPARED = "prepared"


@dataclass(frozen=True)
class PanelBooks:
    panel: Codebook
    stage_a: Codebook
    venture: Codebook

    def structure_entry(self, role: str) -> CodeEntry | None:
        """The panel entry of a structural role (a key of STRUCTURE_CODES); None when the role or the entry is missing."""
        code_id = STRUCTURE_CODES.get(role)
        return next((e for e in self.panel.entries if e.id == code_id), None) if code_id else None

    def turn_functions(self) -> tuple[str, ...]:
        """Every TURN-FUNC value; NONE is a valid function (a remark with none of the others)."""
        return tuple(self.panel.by_id(STRUCTURE_CODES["turn_function"]).values or [])

    def exchange_kinds(self) -> tuple[str, ...]:
        return tuple(v for v in (self.panel.by_id(STRUCTURE_CODES["exchange_kind"]).values or []) if v != "NONE")

    def exchange_topics(self) -> tuple[str, ...]:
        return tuple(self.panel.by_id(STRUCTURE_CODES["exchange_topic"]).values or [])

    def decision_kinds(self) -> tuple[str, ...]:
        return tuple(v for v in (self.panel.by_id(STRUCTURE_CODES["decision"]).values or []) if v != "NONE")

    def categorical_code_ids(self) -> frozenset[str]:
        """Panel codes whose values are categories, so NONE may be one of their values."""
        return frozenset(e.id for e in self.panel.entries if e.type == "categorical")

    def codes_for(self, function: str) -> tuple[CodeEntry, ...]:
        return tuple(e for e in self.panel.entries if e.applies_to == function)

    def prepared_codes(self) -> tuple[CodeEntry, ...]:
        """The codes rated once per pitch on the prepared presentation, in book order."""
        return self.codes_for(PREPARED)

    def canonical_structure(self, record: dict) -> dict:
        """A copy of a record dict whose structure labels carry the codebook's spelling.

        The labels are a turn's function, an exchange's kind and topic, and a decision's kind. Only
        case and surrounding whitespace are repaired; a label that is not a codebook value in any
        case is left as written. Everything that reads these labels (validation,
        alignment, derived fields, the renderer, the comparison) goes through here.
        """
        def fix(code_id: str, item: dict, field: str) -> None:
            item[field] = canonical_value(self.panel.by_id(code_id), item[field])

        out = copy.deepcopy(record)
        for t in out["turns"]:
            fix("TURN-FUNC", t, "function")
        for e in out["exchanges"]:
            fix("EX-KIND", e, "kind")
            fix("EX-TOPIC", e, "topic")
        for d in out["decisions"]:
            fix("DEC-KIND", d, "kind")
        return out

    def canonical_record(self, record: PanelRecord) -> PanelRecord:
        return PanelRecord.model_validate(self.canonical_structure(record.model_dump(mode="json")))

    def stage_a_ids(self) -> tuple[str, ...]:
        return self.stage_a.ids

    def venture_values(self, code_id: str) -> tuple[str, ...]:
        return tuple(self.venture.by_id(code_id).values or [])

    def other_concepts(self, record: dict) -> list[dict]:
        return other_concepts(record)

    def canonical_venture(self, code_id: str, text: str | None) -> str | None:
        return canonical_value(self.venture.by_id(code_id), text)


def canonical_value(entry: CodeEntry, text: str | None) -> str | None:
    """The codebook's spelling of a value that differs only in case or surrounding whitespace; else the text."""
    if text is None:
        return None
    if entry.type == "categorical":
        folded = text.strip().casefold()
        for value in entry.values or []:
            if value.casefold() == folded:
                return value
        return text
    if entry.type == "binary":
        lowered = text.strip().lower()
        return lowered if lowered in ("true", "false") else text
    if entry.type == "ordinal":
        return text.strip()
    return text


def load_panel_books(release: InstrumentRelease, sync: bool = True) -> PanelBooks:
    """The three books of a release. `sync=False` reads the YAML files as they are and writes nothing
    (for the copy of an instrument a run keeps in its bundle folder)."""
    for key in _REQUIRED_BOOKS:
        if key not in release.books:
            raise ConfigError(f"instrument release {release.release_id!r} has no book {key!r}")
    return PanelBooks(
        panel=load_codebook(release.books["stage_b_panel"].yaml, sync=sync),
        stage_a=load_codebook(release.books["stage_a"].yaml, sync=sync),
        venture=load_codebook(release.books["venture_context"].yaml, sync=sync),
    )


def render_panel_prompt(books: PanelBooks, prompt_dir: Path) -> str:
    return render_prompt(
        PANEL_TEMPLATE,
        codebook=books.panel,
        context={"prepared_entries": books.prepared_codes(), "stage_a_entries": books.stage_a.entries,
                 "venture_entries": books.venture.entries},
        prompt_dir=prompt_dir,
    )
