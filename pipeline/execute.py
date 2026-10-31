"""Bounded concurrent execution with rate limiting and typed retries.

Threads rather than asyncio: the Gemini SDK, ffmpeg, and sqlite3 are all blocking,
so a thread pool keeps every other module synchronous and readable for people
forking this pipeline. Retry policy is driven by the error taxonomy — transient
failures back off, item-fatal failures fail once, run-fatal failures stop the run.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from pipeline.config import LimitsConfig
from pipeline.errors import ErrorClass, RunFatalError, classify_exception


class TokenBucket:
    """Simple thread-safe token bucket: `rate_per_minute` sustained, `capacity` burst."""

    def __init__(
        self,
        rate_per_minute: float,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive when given")
        self.rate_per_s = rate_per_minute / 60.0
        self.capacity = float(capacity if capacity is not None else max(1.0, rate_per_minute / 4))
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = self._clock()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_s)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                deficit = n - self._tokens
                wait_s = deficit / self.rate_per_s
            self._sleep(wait_s)


def with_retries(
    call: Callable[[], Any],
    *,
    max_attempts: int,
    base_s: float,
    cap_s: float,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Any:
    """Call `call`, retrying only RETRYABLE failures with capped exponential backoff."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return call()
        except BaseException as exc:
            error_class = classify_exception(exc)
            if error_class is not ErrorClass.RETRYABLE or attempt >= max_attempts:
                raise
            delay_s = min(cap_s, base_s * (2 ** (attempt - 1))) * jitter()
            if on_retry:
                on_retry(attempt, exc, delay_s)
            sleep(delay_s)


@dataclass
class UnitOutcome:
    item: Any
    ok: bool
    value: Any = None
    error: BaseException | None = None
    error_class: str | None = None
    attempts: int = 1


def run_units(
    items: Iterable[Any],
    work_fn: Callable[[Any], Any],
    *,
    max_concurrency: int,
    bucket: TokenBucket | None = None,
    limits: LimitsConfig | None = None,
    on_result: Callable[[UnitOutcome], None] | None = None,
) -> list[UnitOutcome]:
    """Run `work_fn` over `items` with bounded concurrency.

    Item failures become UnitOutcome rows; a RunFatalError stops submission and is
    re-raised once in-flight work settles.
    """
    items = list(items)
    if not items:
        return []
    limits = limits or LimitsConfig()
    stop = threading.Event()
    outcomes: list[UnitOutcome] = []

    def guarded(item: Any) -> UnitOutcome:
        if stop.is_set():
            return UnitOutcome(item=item, ok=False, error_class="SKIPPED_RUN_STOPPED")
        attempts = {"n": 0}

        def attempt_once():
            attempts["n"] += 1
            if bucket is not None:
                bucket.acquire()
            return work_fn(item)

        try:
            value = with_retries(
                attempt_once,
                max_attempts=limits.max_attempts,
                base_s=limits.backoff_base_s,
                cap_s=limits.backoff_cap_s,
            )
            return UnitOutcome(item=item, ok=True, value=value, attempts=attempts["n"])
        except RunFatalError as exc:
            stop.set()
            return UnitOutcome(
                item=item, ok=False, error=exc,
                error_class=ErrorClass.RUN_FATAL.value, attempts=attempts["n"],
            )
        except BaseException as exc:
            error_class = classify_exception(exc)
            label = (
                "RETRY_EXHAUSTED"
                if error_class is ErrorClass.RETRYABLE
                else error_class.value
            )
            return UnitOutcome(
                item=item, ok=False, error=exc, error_class=label, attempts=attempts["n"]
            )

    callback_errors: list[BaseException] = []
    pool = ThreadPoolExecutor(max_workers=max(1, max_concurrency))
    try:
        futures = {pool.submit(guarded, item): item for item in items}
        for future in as_completed(futures):
            outcome = future.result()
            outcomes.append(outcome)
            if on_result:
                try:
                    on_result(outcome)
                except BaseException as exc:  # a bookkeeping bug must not abort the drain
                    callback_errors.append(exc)
    except BaseException:
        # Ctrl-C or a callback error: drop queued work instead of draining it.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        pool.shutdown(wait=True)

    fatal = next((o for o in outcomes if o.error_class == ErrorClass.RUN_FATAL.value), None)
    if fatal and fatal.error:
        raise fatal.error
    if callback_errors:
        raise callback_errors[0]
    return outcomes
