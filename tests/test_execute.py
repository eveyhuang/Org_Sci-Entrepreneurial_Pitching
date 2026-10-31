import threading

import pytest

from pipeline.errors import ProviderError, RunFatalError, SchemaInvalidError
from pipeline.execute import TokenBucket, UnitOutcome, run_units, with_retries


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def test_token_bucket_allows_burst_then_paces():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, capacity=2, clock=clock.time, sleep=clock.sleep)
    bucket.acquire()
    bucket.acquire()
    assert clock.sleeps == []  # burst capacity consumed without waiting
    bucket.acquire()
    assert clock.sleeps and clock.sleeps[0] == pytest.approx(1.0, abs=0.05)


def test_token_bucket_rejects_nonpositive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        TokenBucket(rate_per_minute=60, capacity=0)
    with pytest.raises(ValueError, match="rate_per_minute"):
        TokenBucket(rate_per_minute=0)


def test_with_retries_retries_transient_and_returns_value():
    clock = FakeClock()
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderError("503", status=503)
        return "ok"

    result = with_retries(
        flaky, max_attempts=6, base_s=5.0, cap_s=120.0, sleep=clock.sleep, jitter=lambda: 1.0
    )
    assert result == "ok"
    assert attempts["n"] == 3
    assert clock.sleeps == [5.0, 10.0]


def test_with_retries_caps_backoff():
    clock = FakeClock()

    def always_503():
        raise ProviderError("503", status=503)

    with pytest.raises(ProviderError):
        with_retries(
            always_503, max_attempts=8, base_s=5.0, cap_s=20.0,
            sleep=clock.sleep, jitter=lambda: 1.0,
        )
    assert max(clock.sleeps) == 20.0


def test_with_retries_does_not_retry_item_fatal():
    calls = {"n": 0}

    def bad_schema():
        calls["n"] += 1
        raise SchemaInvalidError("nope")

    with pytest.raises(SchemaInvalidError):
        with_retries(bad_schema, max_attempts=6, base_s=1.0, cap_s=2.0, sleep=lambda s: None)
    assert calls["n"] == 1


def test_with_retries_does_not_retry_run_fatal():
    calls = {"n": 0}

    def auth_failure():
        calls["n"] += 1
        raise RunFatalError("403")

    with pytest.raises(RunFatalError):
        with_retries(auth_failure, max_attempts=6, base_s=1.0, cap_s=2.0, sleep=lambda s: None)
    assert calls["n"] == 1


def test_run_units_executes_all_and_records_failures():
    def work(item: int) -> int:
        if item == 3:
            raise SchemaInvalidError("bad item")
        return item * 2

    outcomes = run_units(list(range(5)), work, max_concurrency=3)
    assert len(outcomes) == 5
    by_item = {o.item: o for o in outcomes}
    assert by_item[2].value == 4 and by_item[2].ok is True
    assert by_item[3].ok is False
    assert by_item[3].error_class == "ITEM_FATAL"


def test_run_units_respects_concurrency_limit():
    active = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def work(item):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        try:
            return item
        finally:
            with lock:
                active["now"] -= 1

    run_units(list(range(50)), work, max_concurrency=4)
    assert active["peak"] <= 4


def test_run_units_stops_on_run_fatal():
    seen: list[int] = []

    def work(item):
        seen.append(item)
        if item == 0:
            raise RunFatalError("auth")
        return item

    with pytest.raises(RunFatalError):
        run_units(list(range(20)), work, max_concurrency=1)
    assert len(seen) < 20


def test_run_units_invokes_callback_per_result():
    seen: list[UnitOutcome] = []
    run_units([1, 2], lambda i: i, max_concurrency=2, on_result=seen.append)
    assert len(seen) == 2
