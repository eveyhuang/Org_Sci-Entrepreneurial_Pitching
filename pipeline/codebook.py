"""Codebook as data.

The YAML files under pipeline/codebooks/ are the published codebook: prompts are
rendered from them and the paper's codebook table is generated from them, so the
instrument's documentation cannot drift from what the model was actually asked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from pipeline.errors import ConfigError

Tier = Literal["T1", "T2", "T3"]
Side = Literal["founder", "investor", "dyadic", "none"]
CodeType = Literal["categorical", "ordinal", "binary", "count", "continuous", "span", "number"]
Unit = Literal["PITCH", "CHUNK", "TURN", "EPISODE", "EXCHANGE", "OBSERVATION", "EVENT", "DECISION"]
Observability = Literal["observable", "interpretive"]


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)
    provenance: Literal["real", "illustrative"]
    source: str | None = None


class Anchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    description: str = Field(min_length=1)

    @field_validator("description", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class CodeEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    dimension: str
    parent: str | None = None
    side: Side = "none"
    tier: Tier
    type: CodeType
    scale: tuple[float, float] | None = None
    unit: Unit
    observability: Observability
    definition: str = Field(min_length=1)
    include_if: str = Field(min_length=1)
    exclude_if: str = Field(min_length=1)
    examples: list[Example] = Field(default_factory=list)
    anchors: list[Anchor] = Field(default_factory=list)
    provenance: str | None = None
    values: list[str] | None = None
    name: str | None = None          # human-readable display name for tables and plots
    applies_to: str | None = None    # event kind this rating code is required for

    @field_validator("definition", "include_if", "exclude_if", "provenance", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        # YAML folded scalars carry a trailing newline and Markdown does not; normalise so
        # the two sources of a codebook parse to identical entries.
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _scaled_types_have_scale(self) -> "CodeEntry":
        if self.type in ("ordinal", "continuous") and self.scale is None:
            raise ValueError(f"{self.id}: type {self.type} requires a scale [min, max]")
        if self.type in ("ordinal", "continuous"):
            if not self.anchors:
                raise ValueError(f"{self.id}: type {self.type} requires at least one anchor")
            lo, hi = self.scale
            for anchor in self.anchors:
                if not (lo <= anchor.value <= hi):
                    raise ValueError(
                        f"{self.id}: anchor value {anchor.value} is outside scale [{lo}, {hi}]"
                    )
        if self.type == "ordinal":
            lo, hi = self.scale
            have = sorted(a.value for a in self.anchors)
            want = list(range(int(lo), int(hi) + 1))
            if have != want:
                raise ValueError(
                    f"{self.id}: an ordinal code needs exactly one anchor per scale point "
                    f"{want}; found {have}"
                )
        if self.type == "categorical":
            if not self.values:
                raise ValueError(f"{self.id}: type categorical requires a `values` list")
            if not any(v in ("NEUTRAL", "NONE") for v in self.values):
                raise ValueError(
                    f"{self.id}: categorical values must include a residual (NEUTRAL or NONE)"
                )
        elif self.values is not None:
            raise ValueError(f"{self.id}: `values` is only allowed on categorical codes")
        return self


class Codebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[CodeEntry]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> "Codebook":
        seen: set[str] = set()
        for entry in self.entries:
            if entry.id in seen:
                raise ValueError(f"duplicate code id {entry.id!r}")
            seen.add(entry.id)
        return self

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    def by_id(self, code_id: str) -> CodeEntry:
        for entry in self.entries:
            if entry.id == code_id:
                return entry
        raise KeyError(code_id)


def load_codebook(path: Path | str, sync: bool = True) -> Codebook:
    path = Path(path)
    # The Markdown sibling, when there is one, is the master: regenerate the YAML from it
    # before reading, so no run can use a codebook older than its Markdown. `sync=False` reads the
    # YAML as it is and never writes (a run's own copy of its instrument).
    if sync:
        from pipeline.codebook_md import ensure_synced

        ensure_synced(path)
    if not path.exists():
        raise ConfigError(f"codebook not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or []
    except yaml.YAMLError as exc:
        raise ConfigError(f"codebook {path} is not valid YAML: {exc}") from exc
    if not isinstance(data, list):
        raise ConfigError(f"codebook must be a YAML list of code entries: {path}")
    try:
        book = Codebook(entries=data)
    except ValidationError as exc:
        raise ConfigError(f"invalid codebook {path}:\n{exc}") from exc
    seen: set[str] = set()
    for entry in book.entries:
        if entry.id in seen:
            raise ConfigError(f"duplicate code id {entry.id!r} in {path}")
        seen.add(entry.id)
    return book
