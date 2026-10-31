"""Aggregate batch rankings into one score per item: Plackett-Luce worths and mean rank.

Scores are relative to the items ranked together. Nothing here is an absolute rating.
"""

from __future__ import annotations

import math
from typing import Sequence


def _canonical(orders: Sequence[Sequence[str]]) -> list[tuple[str, ...]]:
    """Orders as sorted tuples, so the result does not depend on the order they arrive in."""
    out = sorted(tuple(order) for order in orders)
    for order in out:
        if len(set(order)) != len(order):
            raise ValueError(f"an order names an item more than once: {list(order)}")
    return out


def plackett_luce(orders: Sequence[Sequence[str]], prior: float = 0.5, iterations: int = 500,
                  tol: float = 1e-9) -> dict[str, float]:
    """Log-worths of a Plackett-Luce model, centred to mean zero, by Hunter's (2004) MM algorithm.

    An order of k items is k - 1 successive choices: the item in each position is chosen from
    those not yet placed. Every item also has `prior` wins and `prior` losses against a virtual
    item of worth 1, as paired comparisons inside the same update; that keeps the score of an
    item that is always first, or always last, finite.
    """
    if prior <= 0:
        raise ValueError(f"prior must be positive, got {prior}")
    ranked = _canonical(orders)
    items = sorted({item for order in ranked for item in order})
    if not items:
        return {}
    # an item wins every choice it is picked in: once per order in which it is not last
    wins = {item: prior for item in items}
    for order in ranked:
        for item in order[:-1]:
            wins[item] += 1.0
    worth = {item: 1.0 for item in items}
    for _ in range(iterations):
        denominator = {item: 2.0 * prior / (worth[item] + 1.0) for item in items}
        for order in ranked:
            remaining = sum(worth[item] for item in order)
            share = 0.0
            last = len(order) - 1
            for position, item in enumerate(order):
                if position < last:  # the last item is placed without a choice
                    share += 1.0 / remaining
                    remaining -= worth[item]
                denominator[item] += share
        updated = {item: wins[item] / denominator[item] for item in items}
        change = max(abs(math.log(updated[item]) - math.log(worth[item])) for item in items)
        worth = updated
        if change < tol:
            break
    logs = {item: math.log(worth[item]) for item in items}
    centre = sum(logs.values()) / len(logs)
    return {item: logs[item] - centre for item in items}


def mean_normalised_rank(orders: Sequence[Sequence[str]]) -> dict[str, float]:
    """Mean over an item's appearances of 1 - position/(k-1): first is 1, last is 0."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for order in _canonical(orders):
        k = len(order)
        for position, item in enumerate(order):
            value = 0.5 if k == 1 else 1.0 - position / (k - 1)
            totals[item] = totals.get(item, 0.0) + value
            counts[item] = counts.get(item, 0) + 1
    return {item: totals[item] / counts[item] for item in sorted(totals)}


def aggregate(rankings: dict[str, list[list[str]]]) -> dict:
    """Per code: scores for every item ranked on it, and the items from strongest to weakest."""
    out = {}
    for code, orders in rankings.items():
        scores = plackett_luce(orders)
        out[code] = {
            "pl_score": scores,
            "mean_rank": mean_normalised_rank(orders),
            "n_orders": len(orders),
            "order": sorted(scores, key=lambda item: (-scores[item], item)),
        }
    return out
