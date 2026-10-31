"""Fixture-replaying provider.

Fixtures are named `<unit_key>.json`, where unit_key is
`<pass>__<video_id>[__<start>_<end>]__k<k>`, and contain the response body the model
would return. A fixture named just `<pass>__<video_id>.json` is used as a fallback
for every unit of that video, so a whole-video pass needs only one file.
A request's `media_parts` (several labelled clips) are recorded with the request and
otherwise ignored, like `media`.
`error_plan` queues exceptions per (full) unit_key so retry and failure paths are
testable.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from pipeline.errors import ProviderError, SchemaInvalidError
from pipeline.providers.base import GenerateRequest, MediaHandle, ProviderResponse


class FakeProvider:
    def __init__(
        self,
        fixture_dir: Path | str,
        error_plan: dict[str, list[Exception]] | None = None,
        model_version: str = "fake-model-v1",
    ) -> None:
        self.fixture_dir = Path(fixture_dir)
        self.error_plan = {k: list(v) for k, v in (error_plan or {}).items()}
        self.model_version = model_version
        self.calls: list[str] = []
        self.uploads: list[Path] = []
        self.requests: list[GenerateRequest] = []

    def upload_media(self, local_path: Path, file_sha256: str | None = None) -> MediaHandle:
        self.uploads.append(Path(local_path))
        return MediaHandle(name=f"fake-files/{Path(local_path).name}", local_path=Path(local_path))

    def _fixture_for(self, unit_key: str) -> Path:
        """Exact unit_key first, then the pass__video prefix shared by all its units."""
        exact = self.fixture_dir / f"{unit_key}.json"
        if exact.exists():
            return exact
        parts = unit_key.split("__")
        if len(parts) > 2:
            shared = self.fixture_dir / f"{parts[0]}__{parts[1]}.json"
            if shared.exists():
                return shared
        return exact

    def generate(self, request: GenerateRequest) -> ProviderResponse:
        self.requests.append(request)
        self.calls.append(request.unit_key)
        queued = self.error_plan.get(request.unit_key)
        if queued:
            raise queued.pop(0)

        fixture = self._fixture_for(request.unit_key)
        if not fixture.exists():
            raise ProviderError(f"no fixture for unit_key {request.unit_key!r} ({fixture})")
        raw_text = fixture.read_text()
        try:
            parsed = request.schema.model_validate_json(raw_text)
        except ValidationError as exc:
            raise SchemaInvalidError(
                f"fixture {fixture.name} failed schema validation: {exc}", raw_text=raw_text
            ) from exc
        # Token counts are synthesized, not measured: FakeProvider exists so the
        # pipeline can run without an API, and nothing downstream should treat
        # these as real accounting. The real provider reports usage_metadata.
        return ProviderResponse(
            raw_text=raw_text,
            parsed=parsed,
            model_version=self.model_version,
            input_tokens=max(1, len(request.prompt) // 4),
            output_tokens=max(1, len(raw_text) // 4),
            thinking_tokens=0,
            finish_reason="STOP",
        )
