"""Gemini provider.

Two rules this file exists to enforce:
  * one pinned model per request — there is no fallback list anywhere;
  * no JSON salvage — a response that does not validate is a SCHEMA_INVALID failure
    with its raw text preserved for inspection.
"""

from __future__ import annotations

import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from pipeline.errors import (
    ErrorClass,
    ProviderError,
    RunFatalError,
    SchemaInvalidError,
    classify_exception,
)
from pipeline.ingest import sha256_file
from pipeline.providers.base import GenerateRequest, MediaHandle, ProviderResponse
from pipeline.registry import Registry

ACTIVE_STATES = {"ACTIVE"}
FAILED_STATES = {"FAILED"}

_SCHEMA_DROP_KEYS = frozenset({"additionalProperties", "$defs", "title", "default"})


def gemini_response_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a pydantic model into a Gemini-compatible response schema.

    Gemini's response_schema accepts only a subset of JSON Schema: it rejects
    `additionalProperties` (which `extra="forbid"` emits), cannot follow `$ref`,
    and expresses optionality as `nullable` rather than an anyOf-with-null branch.
    Strict validation stays on the pydantic side — an invented field still fails at
    parse time; only the wire format is relaxed.
    """
    raw = model.model_json_schema()
    defs = raw.get("$defs", {})

    def convert(node: Any) -> Any:
        if isinstance(node, list):
            return [convert(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return convert(defs[name])
        if "anyOf" in node:
            branches = [b for b in node["anyOf"] if b.get("type") != "null"]
            nullable = len(branches) != len(node["anyOf"])
            if len(branches) == 1:
                merged = convert(branches[0])
                for key, value in node.items():
                    if key not in ("anyOf", *_SCHEMA_DROP_KEYS) and key not in merged:
                        merged[key] = convert(value)
                if nullable:
                    merged["nullable"] = True
                return merged
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _SCHEMA_DROP_KEYS:
                continue
            out[key] = convert(value)
        return out

    return convert(raw)


class GeminiProvider:
    def __init__(
        self,
        client,
        registry: Registry,
        upload_max_age_h: float = 44.0,
        poll_interval_s: float = 5.0,
        upload_timeout_s: float = 900.0,
        sleep=time.sleep,
    ) -> None:
        self.client = client
        self.registry = registry
        self.upload_max_age_h = upload_max_age_h
        self.poll_interval_s = poll_interval_s
        self.upload_timeout_s = upload_timeout_s
        self._sleep = sleep

    # --- media -----------------------------------------------------------
    def upload_media(self, local_path: Path, file_sha256: str | None = None) -> MediaHandle:
        """Upload with a content-hash cache. Files expire remotely at 48h.

        `file_sha256` lets the caller supply a digest already computed at ingest
        time; without it every work unit re-reads the whole source file, which at
        k replicates over a large corpus is hundreds of gigabytes of redundant IO.
        """
        local_path = Path(local_path)
        digest = file_sha256 or sha256_file(local_path)
        cached = self.registry.get_upload(digest, max_age_h=self.upload_max_age_h)
        if cached:
            try:
                remote = self.client.files.get(name=cached)
                if _state_name(remote) in ACTIVE_STATES:
                    return MediaHandle(
                        name=cached, uri=getattr(remote, "uri", None), local_path=local_path
                    )
            except Exception as exc:
                if classify_exception(exc) is ErrorClass.RUN_FATAL:
                    raise RunFatalError(str(exc)) from exc
                if classify_exception(exc) is ErrorClass.RETRYABLE:
                    raise
                # Remote copy is genuinely gone or unusable; re-upload below.
                self.registry.delete_upload(digest)

        remote = self._raise_as_pipeline_error(lambda: self._upload(local_path))
        remote = self._await_active(remote, digest)
        self.registry.put_upload(digest, remote.name)
        return MediaHandle(
            name=remote.name, uri=getattr(remote, "uri", None), local_path=local_path
        )

    def _upload(self, local_path: Path):
        # Given a path, the SDK sends the file's basename in an HTTP header, which must be ASCII;
        # titles with curly quotes or apostrophes fail. Such files are uploaded as an open handle.
        if local_path.name.isascii():
            return self.client.files.upload(file=str(local_path))
        mime = mimetypes.guess_type(local_path.name)[0] or "video/mp4"
        with open(local_path, "rb") as fh:
            return self.client.files.upload(file=fh, config={"mime_type": mime})

    def _await_active(self, remote, digest: str):
        deadline = time.monotonic() + self.upload_timeout_s
        while _state_name(remote) not in ACTIVE_STATES:
            if _state_name(remote) in FAILED_STATES:
                # A FAILED remote file is a transient processing problem, not a fatal
                # one: the caller retries, and must re-upload rather than reuse a cache
                # entry that now points at a dead file.
                self.registry.delete_upload(digest)
                raise ProviderError(
                    f"Gemini file processing failed for {remote.name}", status=503
                )
            if time.monotonic() > deadline:
                raise ProviderError(f"timed out waiting for {remote.name} to become ACTIVE", status=504)
            self._sleep(self.poll_interval_s)
            remote = self.client.files.get(name=remote.name)
        return remote

    # --- generation ------------------------------------------------------
    def generate(self, request: GenerateRequest) -> ProviderResponse:
        if request.media_parts and request.media is not None:
            raise ValueError(f"{request.unit_key}: a request carries either media or media_parts, not both")
        contents: list = [request.prompt]
        if request.media is not None:
            contents.append(self._media_part(request.media))
        for label, handle in request.media_parts:
            contents.append(f"Clip {label}:")
            contents.append(self._media_part(handle))

        response = self._raise_as_pipeline_error(
            lambda: self.client.models.generate_content(
                model=request.model.name,  # exactly one model, always
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": gemini_response_schema(request.schema),
                    "temperature": request.model.temperature,
                    "max_output_tokens": request.model.max_output_tokens,
                },
            )
        )

        raw_text = getattr(response, "text", None) or ""
        finish_reason = _finish_reason(response)
        blocked = _is_blocked(response)
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        thinking_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
        model_version = getattr(response, "model_version", None) or request.model.name

        def billed(exc: Exception) -> Exception:
            # A response arrived and was paid for even though it is rejected here; the caller
            # logs these so failed attempts are not accounted as free.
            exc.usage = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                         "thinking_tokens": thinking_tokens}
            exc.model_version = model_version
            return exc

        if blocked:
            raise billed(ProviderError(
                f"{request.unit_key}: response blocked by safety filters",
                status=400,
                raw_text=raw_text,
            ))
        if finish_reason and finish_reason not in ("STOP", "FINISH_REASON_STOP"):
            raise billed(SchemaInvalidError(
                f"{request.unit_key}: response did not finish cleanly ({finish_reason})",
                raw_text=raw_text,
            ))

        # The SDK only auto-parses into the model when it is handed a model CLASS.
        # We send a converted dict schema (see gemini_response_schema), so `parsed`
        # comes back as a plain dict — which would skip validation entirely. Always
        # validate the raw body ourselves unless the SDK already produced the model.
        parsed = getattr(response, "parsed", None)
        if not isinstance(parsed, request.schema):
            if not raw_text.strip():
                raise billed(SchemaInvalidError(f"{request.unit_key}: empty response body", raw_text=raw_text))
            try:
                parsed = request.schema.model_validate_json(raw_text)
            except ValidationError as exc:
                raise billed(SchemaInvalidError(f"{request.unit_key}: {exc}", raw_text=raw_text)) from exc

        return ProviderResponse(
            raw_text=raw_text or parsed.model_dump_json(),
            parsed=parsed,
            model_version=model_version,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            thinking_tokens=thinking_tokens,
            finish_reason=finish_reason,
            safety_blocked=blocked,
        )

    def _media_part(self, media: MediaHandle):
        """Prefer the SDK's file object; fall back to the handle's name."""
        try:
            return self._raise_as_pipeline_error(
                lambda: self.client.files.get(name=media.name)
            )
        except RunFatalError:
            raise
        except Exception as exc:
            raise ProviderError(f"media handle {media.name} unusable: {exc}", status=400) from exc

    @staticmethod
    def _raise_as_pipeline_error(call):
        try:
            return call()
        except Exception as exc:
            if classify_exception(exc) is ErrorClass.RUN_FATAL:
                raise RunFatalError(str(exc)) from exc
            raise


def _state_name(remote) -> str:
    state = getattr(remote, "state", None)
    name = getattr(state, "name", state)
    return str(name) if name is not None else "PROCESSING"


def _finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = getattr(candidates[0], "finish_reason", None)
    return str(getattr(reason, "name", reason)) if reason is not None else None


def _is_blocked(response) -> bool:
    feedback = getattr(response, "prompt_feedback", None)
    return bool(getattr(feedback, "block_reason", None))


def build_gemini_provider(registry: Registry, api_key: str | None = None) -> GeminiProvider:
    """Construct a provider from the environment (loads .env if present)."""
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RunFatalError("GOOGLE_API_KEY is not set (put it in .env at the repo root)")
    return GeminiProvider(genai.Client(api_key=key), registry)
