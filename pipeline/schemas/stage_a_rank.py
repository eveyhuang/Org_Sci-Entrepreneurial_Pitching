"""Wire schema for a Stage A ranking call: one strict order of the batch's clips per dimension.

The model returns orders and short notes, never scores. The schema cannot express "a
permutation of exactly these labels", so `check_rank_response` does: a response that parses
but ties, omits or invents a clip, or misses a dimension, is invalid output.
"""

from __future__ import annotations

from collections import Counter

from pydantic import Field

from pipeline.schemas.stage_b_panel import Strict


class DimensionRanking(Strict):
    code: str
    order: list[str]            # clip labels, strongest first
    note: str = Field(max_length=300)   # what separates the top from the bottom, about 25 words


class RankResponse(Strict):
    rankings: list[DimensionRanking]
    notes: str = Field(max_length=400)


def check_rank_response(resp: RankResponse, codes: tuple[str, ...], labels: tuple[str, ...]) -> list[str]:
    """Problems with a parsed response; empty when every dimension has one strict order of `labels`."""
    problems: list[str] = []
    seen = Counter(r.code for r in resp.rankings)
    for code in codes:
        if code not in seen:
            problems.append(f"dimension {code!r} is missing")
    for code, n in seen.items():
        if code not in codes:
            problems.append(f"unknown dimension {code!r}")
        elif n > 1:
            problems.append(f"dimension {code!r} is ranked more than once ({n} times)")
    for ranking in resp.rankings:
        placed = Counter(ranking.order)
        faults = [f"{label!r} is missing" for label in labels if label not in placed]
        faults += [f"{label!r} is not a clip of this batch" for label in placed if label not in labels]
        faults += [f"{label!r} appears {n} times" for label, n in placed.items() if n > 1 and label in labels]
        if faults:
            problems.append(f"{ranking.code}: order {ranking.order} is not a strict order of "
                            f"{list(labels)}: " + "; ".join(faults))
    return problems
