"""Seeded batch design for cross-video ranking.

Each video is ranked `appearances` times, each time in a small batch with different
neighbours where the number of videos allows. The design is a pure function of
(video ids, batch size, appearances, seed), so a stored design can be checked on resume.
"""

from __future__ import annotations

import random
import string
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

RESHUFFLES = 200
MAX_BATCH_SIZE = len(string.ascii_uppercase)


@dataclass(frozen=True)
class Batch:
    batch_id: str                 # "b01", "b02", ...
    video_ids: tuple[str, ...]    # in presentation order
    labels: tuple[str, ...]       # "A", "B", ... aligned with video_ids


def _chunks(order: list[str], batch_size: int) -> list[list[str]]:
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        if len(chunks[-2]) > 2:
            chunks[-1].insert(0, chunks[-2].pop())
        else:
            # batch_size 2: taking one from the previous pair would only move the lone video
            chunks[-2].extend(chunks.pop())
    return chunks


def _repeats(chunks: list[list[str]], met: set[frozenset]) -> int:
    return sum(1 for chunk in chunks for pair in combinations(chunk, 2) if frozenset(pair) in met)


def _repair(chunks: list[list[str]], met: set[frozenset]) -> list[list[str]]:
    """Swap videos between batches while a swap lowers the number of repeated pairs.

    Random reshuffling alone almost never reaches zero once there are many batches, so the
    best shuffle is finished off deterministically: batch sizes and the seed-dependence of the
    result are unchanged.
    """
    def clashes(video: str, chunk: list[str], without: str) -> int:
        return sum(1 for other in chunk if other != without and other != video
                   and frozenset((video, other)) in met)

    improved = True
    while improved and _repeats(chunks, met):
        improved = False
        for a, chunk_a in enumerate(chunks):
            for i in range(len(chunk_a)):
                if not clashes(chunk_a[i], chunk_a, chunk_a[i]):
                    continue
                for b, chunk_b in enumerate(chunks):
                    if a == b:
                        continue
                    for j in range(len(chunk_b)):
                        x, y = chunk_a[i], chunk_b[j]
                        before = clashes(x, chunk_a, x) + clashes(y, chunk_b, y)
                        after = clashes(x, chunk_b, y) + clashes(y, chunk_a, x)
                        if after < before:
                            chunk_a[i], chunk_b[j] = y, x
                            improved = True
                            break
                    if improved and not clashes(chunk_a[i], chunk_a, chunk_a[i]):
                        break
    return chunks


def make_design(video_ids: Sequence[str], batch_size: int = 5, appearances: int = 2, seed: int = 0) -> list[Batch]:
    ids = sorted(video_ids)
    if len(ids) < 2:
        raise ValueError(f"ranking needs at least 2 videos, got {len(ids)}")
    if len(set(ids)) != len(ids):
        dupes = sorted(v for v, n in Counter(ids).items() if n > 1)
        raise ValueError(f"duplicate video ids: {dupes}")
    if not 2 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 2 and {MAX_BATCH_SIZE}, got {batch_size}")
    if appearances < 1:
        raise ValueError(f"appearances must be at least 1, got {appearances}")

    single_batch = len(ids) <= batch_size
    met: set[frozenset] = set()       # pairs that shared a batch in an earlier round
    used: set[tuple[str, ...]] = set()  # whole-round orders already used
    rounds: list[list[list[str]]] = []
    for round_index in range(appearances):
        rng = random.Random(seed * 1000 + round_index)
        best: list[str] | None = None
        best_cost: tuple[int, int] | None = None
        for _ in range(1 + (RESHUFFLES if round_index else 0)):
            order = list(ids)
            rng.shuffle(order)
            cost = (_repeats(_chunks(order, batch_size), met), int(tuple(order) in used))
            if best_cost is None or cost < best_cost:
                best, best_cost = order, cost
            # with one batch per round every pair repeats; all that can differ is the order
            if cost == (0, 0) or (single_batch and cost[1] == 0):
                break
        assert best is not None and best_cost is not None
        chunks = _chunks(best, batch_size)
        if best_cost[0] and not single_batch:
            chunks = _repair(chunks, met)
        used.add(tuple(best))
        rounds.append(chunks)
        met.update(frozenset(pair) for chunk in chunks for pair in combinations(chunk, 2))

    batches: list[Batch] = []
    for chunks in rounds:
        for chunk in chunks:
            batches.append(Batch(batch_id=f"b{len(batches) + 1:02d}", video_ids=tuple(chunk),
                                 labels=tuple(string.ascii_uppercase[:len(chunk)])))
    return batches


def design_summary(batches: list[Batch]) -> dict:
    appearances = Counter(v for b in batches for v in b.video_ids)
    pairs = Counter(frozenset(p) for b in batches for p in combinations(b.video_ids, 2))
    return {
        "n_videos": len(appearances),
        "n_batches": len(batches),
        "appearances": dict(sorted(appearances.items())),
        # a pair that shares k batches counts k - 1 times
        "repeated_pairs": sum(n - 1 for n in pairs.values()),
        "batch_sizes": [len(b.video_ids) for b in batches],
    }
