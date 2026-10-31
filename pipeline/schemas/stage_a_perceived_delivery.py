"""Stage A v2: provisional shared human–model delivery judgments, not legacy features."""
from __future__ import annotations

import re
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STAGE_A_DELIVERY_CODES = ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")


class DeliveryRating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int | None = Field(ge=1, le=5, strict=True)
    uncodable: bool
    evidence: str = Field(min_length=1, max_length=1200)

    @field_validator("evidence")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def evidence_and_codability(self):
        if self.uncodable != (self.value is None):
            raise ValueError("value must be null exactly when uncodable")
        if not self.uncodable and not re.search(r"\b\d+:[0-5]\d(?:[:][0-5]\d)?\b", self.evidence):
            raise ValueError("scored evidence requires a clip-relative M:SS timestamp")
        return self


class StageADeliveryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pitch_positivity: DeliveryRating
    expressed_enthusiasm: DeliveryRating
    interpersonal_warmth: DeliveryRating
    overall_notes: str = Field(default="", max_length=1500)
