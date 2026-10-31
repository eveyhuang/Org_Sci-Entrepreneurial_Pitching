"""Response schema generated from a codebook.

For codebook-driven passes the codebook is the single source of truth for both the prompt
and the response contract: adding or retyping a code is a Markdown edit, never a code edit.

Every field is {value | null, uncodable, evidence, turn_refs} with the Stage A invariant —
a value OR uncodable, never both, never neither. All four keys are required on the wire, so
an omitted `value` can never be mistaken for a judgment. There is no `confidence` field:
Stage A returned "high" on 997 of 1,000 ratings, so it carried no information.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, field_validator, model_validator

from pipeline.codebook import Codebook, CodeEntry
from pipeline.errors import ConfigError


class CodedValue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncodable: bool = Field(description="true when the evidence needed to judge this code is absent")
    evidence: str = Field(min_length=1, max_length=600,
                          description="what was said or done that this judgment rests on")
    turn_refs: list[int] = Field(description="transcript turn indices (tN) the judgment rests on")

    @field_validator("evidence")
    @classmethod
    def _evidence_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def _value_matches_codability(self) -> "CodedValue":
        value = getattr(self, "value", None)
        if self.uncodable and value is not None:
            raise ValueError("uncodable codes must have a null value")
        if not self.uncodable and value is None:
            raise ValueError("value is required unless uncodable is true")
        return self


def _value_type(entry: CodeEntry) -> Any:
    if entry.type == "ordinal":
        low, high = entry.scale
        return Annotated[int, Field(ge=int(low), le=int(high))]
    if entry.type == "binary":
        return bool
    if entry.type == "categorical":
        return Literal[tuple(entry.values)]
    if entry.type == "count":
        return Annotated[int, Field(ge=0)]
    raise ConfigError(
        f"{entry.id}: code type {entry.type!r} cannot be generated into a response field "
        f"(supported: ordinal, binary, categorical, count)"
    )


def build_record_model(
    codebook: Codebook, name: str = "EpisodeRecord", unit: str = "EPISODE"
) -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for entry in codebook.entries:
        if entry.unit != unit:
            continue
        attribute = re.sub(r"\W", "_", entry.id)
        if attribute[0].isdigit() or attribute.startswith("_"):
            attribute = f"code_{attribute.lstrip('_')}"
        if attribute in fields:
            raise ConfigError(f"code ids collide once made into field names: {entry.id!r}")
        value_model = create_model(
            f"{name}_{attribute}", __base__=CodedValue,
            value=(_value_type(entry) | None, Field(description=f"{entry.type} answer; null only when uncodable")),
        )
        fields[attribute] = (value_model, Field(alias=entry.id))
    if not fields:
        raise ConfigError(f"the codebook has no {unit} codes to build a response schema from")
    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def schema_fingerprint(model: type[BaseModel]) -> str:
    """Identity of the response contract. Descriptions are excluded: wording is the prompt's
    business (and the prompt hash's); this tracks only keys, types and bounds."""

    def strip(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in ("description", "title")}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    canonical = json.dumps(strip(model.model_json_schema()), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
