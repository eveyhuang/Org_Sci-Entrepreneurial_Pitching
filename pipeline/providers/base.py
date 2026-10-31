"""Provider protocol.

Isolating the model behind this protocol is what lets the whole pipeline run in CI
against recorded fixtures, and what keeps 'swap the provider' a contained change
for anyone forking the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from pipeline.config import ModelConfig


@dataclass(frozen=True)
class MediaHandle:
    """A media file as the provider refers to it."""

    name: str
    uri: str | None = None
    local_path: Path | None = None


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    media: MediaHandle | None
    schema: type[BaseModel]
    model: ModelConfig
    unit_key: str  # opaque tracing key; also the fixture key for FakeProvider
    # Several labelled clips in one call, as (label, handle) in presentation order. Used
    # instead of `media`, never with it.
    media_parts: tuple[tuple[str, MediaHandle], ...] = ()


@dataclass(frozen=True)
class ProviderResponse:
    raw_text: str
    parsed: BaseModel | None
    model_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    finish_reason: str | None = None
    safety_blocked: bool = False


class Provider(Protocol):
    def upload_media(
        self, local_path: Path, file_sha256: str | None = None
    ) -> MediaHandle: ...

    def generate(self, request: GenerateRequest) -> ProviderResponse: ...
