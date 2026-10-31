"""Error taxonomy for the annotation pipeline.

Three classes drive execution policy:
  RETRYABLE  — back off and try again; on exhaustion the unit fails, run continues.
  ITEM_FATAL — this unit cannot succeed; record and continue with other units.
  RUN_FATAL  — nothing can succeed; stop the run (registry keeps partial progress).
"""

from __future__ import annotations

from enum import Enum


class ErrorClass(str, Enum):
    RETRYABLE = "RETRYABLE"
    ITEM_FATAL = "ITEM_FATAL"
    RUN_FATAL = "RUN_FATAL"


class PipelineError(Exception):
    """Base class for pipeline errors."""


class ConfigError(PipelineError):
    """Configuration is missing, malformed, or internally inconsistent."""


class SchemaInvalidError(PipelineError):
    """A provider response could not be validated against the pass schema."""

    def __init__(self, message: str, raw_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class RunFatalError(PipelineError):
    """The run cannot continue (auth failure, unusable environment)."""


class ProviderError(PipelineError):
    """An API call failed. `status` carries the HTTP status when known.

    `raw_text` carries the response body when there was one — a safety block is
    exactly the failure class the published taxonomy needs bodies for, and
    annotate.py persists whatever is here before recording the failure.
    """

    def __init__(
        self, message: str, status: int | None = None, raw_text: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.raw_text = raw_text


RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
ITEM_FATAL_STATUS = frozenset({400, 404, 405, 413, 415, 422})
RUN_FATAL_STATUS = frozenset({401, 403})

_RETRYABLE_EXCEPTIONS = (TimeoutError, ConnectionError)


def classify_exception(exc: BaseException) -> ErrorClass:
    """Map an exception to its execution policy."""
    if isinstance(exc, RunFatalError):
        return ErrorClass.RUN_FATAL
    if isinstance(exc, (SchemaInvalidError, ConfigError)):
        return ErrorClass.ITEM_FATAL
    status = _status_of(exc)
    if status is not None:
        if status in RUN_FATAL_STATUS:
            return ErrorClass.RUN_FATAL
        if status in RETRYABLE_STATUS:
            return ErrorClass.RETRYABLE
        if status in ITEM_FATAL_STATUS:
            return ErrorClass.ITEM_FATAL
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return ErrorClass.RETRYABLE
    # Unknown failures fail one unit rather than the whole run.
    return ErrorClass.ITEM_FATAL


def _status_of(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction across SDK exception shapes."""
    for attr in ("status", "status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None
