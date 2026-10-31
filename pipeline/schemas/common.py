"""Shared response types.

Every rating carries its own evidence string and an explicit `uncodable` escape,
so a missing measurement is recorded as missing instead of silently coerced to a
number — the design doc's residual-code rule, enforced by the schema.

Validation runs in pydantic's default lax mode on purpose: a well-formed number
that arrives as the JSON string "0.4" is accepted and coerced, because discarding
an otherwise valid annotation over JSON typing would cost a paid API call and gain
nothing. What is never accepted is missing or out-of-range data — those surface as
SCHEMA_INVALID rather than being repaired.

SCALE CONTRACT — read this before forking the codebook. The shipped `Rating`
encodes a graded human-style judgment as an ORDINAL INTEGER on 1–5
(`ge=1, le=5`), matching a five-anchor codebook scale: each construct's YAML
entry declares `scale: [1, 5]` plus five `anchors`, and the prompt template
renders those anchors as the model's explicit response set. The codebook's
`scale` and this model's bounds must agree — a codebook declaring a different
scale (say a 1–7 Likert scale with seven anchors) would instruct the model to
answer 1–7 and then this model would reject every well-formed response as
SCHEMA_INVALID. A fork using any other scale must define its own Rating model
(and a record model referencing it) and point the pass's `schema:` field at
that dotted path — do NOT widen the bounds here, because 1–5 is the contract
the published Stage A instrument was validated on.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Confidence = Literal["low", "medium", "high"]


class Rating(BaseModel):
    """A single construct rating on an ordinal 1–5 scale with auditable evidence."""

    model_config = ConfigDict(extra="forbid")

    value: int | None = Field(
        default=None, ge=1, le=5,
        description="ordinal 1-5 rating chosen from the codebook's five anchors; null only when uncodable is true",
    )
    uncodable: bool = Field(
        default=False,
        description="true when the evidence needed for this construct is absent",
    )
    confidence: Confidence
    evidence: str = Field(
        min_length=1, max_length=600,
        description="timestamps or behaviors this rating is based on",
    )

    @field_validator("evidence")
    @classmethod
    def _evidence_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _value_matches_codability(self) -> "Rating":
        if self.uncodable and self.value is not None:
            raise ValueError("uncodable ratings must have a null value")
        if not self.uncodable and self.value is None:
            raise ValueError("value is required unless uncodable is true")
        return self
