"""Stage B hand-written response shapes: segmentation.

These are structure, not codebook constructs — EP-BOUND and EP-CUT from the project plan
are the span and `cut` fields below. The per-episode coding record is generated from the
codebook instead (pipeline/schemas/generate.py). The speaking timeline (roster + turns) is
`pipeline.schemas.stage_b_timeline`, kept in its own module because four paid dev-set results
are keyed to that dotted path. The consistency pass is deferred (Revision R1); its records
are not built.

Revision R2: investor decisions (`out` / `offer` / `joins_offer`) and the on-air `deal` are
the study's OUTCOMES, not predictor episodes — they are recorded here in `decisions`/`deal`
and the episode pass never sees them (see the "Once the first investor begins to decide"
segmentation rule and the EP-BOUND codebook entry).

Constraints are limited to ge/le/min_length/max_length, which the live API accepted in
Stage A; anything else (the speaker-label pattern, orderings) is enforced in validators so
it never reaches the wire schema.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pipeline.schemas.stage_b_timeline import clock_to_seconds

_SPEAKER = re.compile(r"^speaker-\d+$")
Role = Literal["founder", "investor", "host", "other"]
Resolution = Literal["resolved", "dropped", "conceded", "reframed", "displaced", "unclear"]
Decision = Literal["out", "offer", "joins_offer", "no_decision_shown"]


def _speaker_label(value: str) -> str:
    if not _SPEAKER.match(value):
        raise ValueError(f"speaker labels look like 'speaker-3', never a name; got {value!r}")
    return value


class EpisodeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str
    start_turn: int = Field(ge=0)
    end_turn: int = Field(ge=0)
    probing_speaker: str
    topic: str = Field(min_length=1, max_length=120)
    resolution: Resolution
    cut: bool
    evidence: str = Field(min_length=1, max_length=600)

    _label = field_validator("probing_speaker")(_speaker_label)

    @model_validator(mode="after")
    def _ordered(self) -> "EpisodeSpan":
        if self.end_turn < self.start_turn:
            raise ValueError("start_turn must not be after end_turn")
        if clock_to_seconds(self.end) < clock_to_seconds(self.start):
            raise ValueError("end must not be before start")
        return self


class InvestorDecision(BaseModel):
    """One investor's on-air decision — an OUTCOME, never coded as a predictor episode."""

    model_config = ConfigDict(extra="forbid")

    speaker: str
    decision: Decision
    turn: int | None = Field(default=None, ge=0)
    terms: str | None = Field(default=None, max_length=200)  # offers: as stated, e.g. "$100k for 20%"

    _label = field_validator("speaker")(_speaker_label)

    @model_validator(mode="after")
    def _turn_and_terms(self) -> "InvestorDecision":
        if self.decision != "no_decision_shown" and self.turn is None:
            raise ValueError("turn is required unless decision is no_decision_shown")
        if self.terms is not None and self.decision not in ("offer", "joins_offer"):
            raise ValueError("terms is only reported for offer/joins_offer decisions")
        return self


class SegmentationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_pitches: int = Field(ge=0)
    prepared_end: str
    summary: str = Field(min_length=1, max_length=600)
    episodes: list[EpisodeSpan]
    decisions: list[InvestorDecision]
    deal: bool

    @model_validator(mode="after")
    def _ordered_and_disjoint(self) -> "SegmentationRecord":
        clock_to_seconds(self.prepared_end)
        for previous, current in zip(self.episodes, self.episodes[1:]):
            if current.start_turn <= previous.end_turn:
                raise ValueError("episodes must be ordered and must not overlap in turns")
        speakers = [d.speaker for d in self.decisions]
        if len(speakers) != len(set(speakers)):
            raise ValueError("decisions must have at most one entry per speaker")
        if self.deal and not any(d.decision in ("offer", "joins_offer") for d in self.decisions):
            raise ValueError("deal cannot be True unless some decision is offer/joins_offer")
        return self
