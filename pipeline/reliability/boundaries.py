"""Agreement between two segmentations of one video. Pure functions over turn spans.

Used for stability (model replicate vs replicate) and validity (model vs human). Unitizing
is reported separately from categorical agreement: a kappa over codes means little if the
units being coded do not line up.
"""

from __future__ import annotations

from itertools import combinations

Span = tuple[int, int]  # inclusive turn indices


def span_jaccard(a: Span, b: Span) -> float:
    overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)
    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - overlap
    return overlap / union


def episode_jaccard(a: list[Span], b: list[Span]) -> float:
    """Mean best-match Jaccard, averaged over both directions."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    forward = sum(max(span_jaccard(x, y) for y in b) for x in a) / len(a)
    backward = sum(max(span_jaccard(y, x) for x in a) for y in b) / len(b)
    return (forward + backward) / 2


def boundary_points(spans: list[Span]) -> list[int]:
    return sorted({start for start, _ in spans} | {end + 1 for _, end in spans})


def boundary_agreement(a: list[Span], b: list[Span], tolerance: int = 2) -> dict:
    """One-to-one matching of boundary points within ±tolerance turns."""
    pa, pb = boundary_points(a), boundary_points(b)
    matched = i = j = 0
    while i < len(pa) and j < len(pb):  # greedy is optimal for points on a line
        if abs(pa[i] - pb[j]) <= tolerance:
            matched, i, j = matched + 1, i + 1, j + 1
        elif pa[i] < pb[j]:
            i += 1
        else:
            j += 1
    if not pa and not pb:
        precision = recall = f1 = 1.0
    else:
        precision = matched / len(pa) if pa else 0.0
        recall = matched / len(pb) if pb else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "matched": matched, "n_a": len(pa), "n_b": len(pb)}


def pairwise_agreement(by_replicate: dict[int, list[Span]], tolerance: int = 2) -> dict:
    pairs = list(combinations(sorted(by_replicate), 2))
    if not pairs:
        return {"pairs": 0, "jaccard": None, "f1": None}
    jaccards = [episode_jaccard(by_replicate[x], by_replicate[y]) for x, y in pairs]
    f1s = [boundary_agreement(by_replicate[x], by_replicate[y], tolerance)["f1"] for x, y in pairs]
    return {"pairs": len(pairs), "jaccard": sum(jaccards) / len(pairs), "f1": sum(f1s) / len(pairs)}
