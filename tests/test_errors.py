import pytest

from pipeline.errors import (
    ErrorClass,
    ProviderError,
    RunFatalError,
    SchemaInvalidError,
    classify_exception,
)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retryable(status):
    assert classify_exception(ProviderError("boom", status=status)) is ErrorClass.RETRYABLE


@pytest.mark.parametrize("status", [400, 404, 413, 422])
def test_bad_request_statuses_are_item_fatal(status):
    assert classify_exception(ProviderError("boom", status=status)) is ErrorClass.ITEM_FATAL


@pytest.mark.parametrize("status", [401, 403])
def test_auth_statuses_are_run_fatal(status):
    assert classify_exception(ProviderError("boom", status=status)) is ErrorClass.RUN_FATAL


def test_schema_invalid_is_item_fatal():
    assert classify_exception(SchemaInvalidError("bad json")) is ErrorClass.ITEM_FATAL


def test_run_fatal_error_is_run_fatal():
    assert classify_exception(RunFatalError("no key")) is ErrorClass.RUN_FATAL


def test_timeouts_and_connection_errors_are_retryable():
    assert classify_exception(TimeoutError()) is ErrorClass.RETRYABLE
    assert classify_exception(ConnectionError()) is ErrorClass.RETRYABLE


def test_unknown_exception_is_item_fatal_not_run_fatal():
    # An unexpected bug must fail one unit, never silently kill the whole run.
    assert classify_exception(ValueError("surprise")) is ErrorClass.ITEM_FATAL
