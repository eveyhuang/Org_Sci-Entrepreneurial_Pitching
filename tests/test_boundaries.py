import pytest

from pipeline.reliability.boundaries import (
    boundary_agreement, boundary_points, episode_jaccard, pairwise_agreement, span_jaccard,
)

A = [(2, 4), (5, 7)]
B = [(2, 5), (6, 7)]


def test_span_jaccard_counts_inclusive_turns():
    assert span_jaccard((2, 4), (2, 5)) == pytest.approx(3 / 4)
    assert span_jaccard((5, 7), (2, 5)) == pytest.approx(1 / 6)
    assert span_jaccard((2, 4), (6, 7)) == 0.0


def test_episode_jaccard_is_symmetric_best_match():
    # A->B: max(.75, 0)=.75 and max(1/6, 2/3)=2/3; B->A: .75 and 2/3. Mean = 17/24.
    assert episode_jaccard(A, B) == pytest.approx(17 / 24)
    assert episode_jaccard(B, A) == pytest.approx(17 / 24)
    assert episode_jaccard(A, A) == 1.0
    assert episode_jaccard([], []) == 1.0 and episode_jaccard(A, []) == 0.0


def test_boundary_points_are_starts_and_one_past_each_end():
    assert boundary_points(A) == [2, 5, 8]      # 5 is both an end+1 and a start
    assert boundary_points(B) == [2, 6, 8]


def test_boundary_agreement_with_and_without_tolerance():
    strict = boundary_agreement(A, B, tolerance=0)
    assert (strict["matched"], strict["n_a"], strict["n_b"]) == (2, 3, 3)
    assert strict["f1"] == pytest.approx(2 / 3)
    assert boundary_agreement(A, B, tolerance=2)["f1"] == 1.0
    assert boundary_agreement([], [])["f1"] == 1.0
    assert boundary_agreement(A, [])["f1"] == 0.0


def test_matching_is_one_to_one():
    # points [2, 11] vs [4, 5]: 2<->4 match; 5 cannot also match 2; 11 matches nothing.
    result = boundary_agreement([(2, 10)], [(4, 4)], tolerance=2)
    assert (result["matched"], result["precision"], result["recall"]) == (1, 0.5, 0.5)


def test_pairwise_means_over_replicate_pairs():
    result = pairwise_agreement({0: A, 1: A, 2: B}, tolerance=0)
    assert result["pairs"] == 3
    assert result["jaccard"] == pytest.approx((1.0 + 17 / 24 + 17 / 24) / 3)
    assert result["f1"] == pytest.approx((1.0 + 2 / 3 + 2 / 3) / 3)
    assert pairwise_agreement({0: A}) == {"pairs": 0, "jaccard": None, "f1": None}
