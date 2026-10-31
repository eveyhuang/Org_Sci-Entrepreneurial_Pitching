"""Seeded batch design: who is ranked against whom, fixed by (ids, batch size, appearances, seed)."""

from collections import Counter
from itertools import combinations

import pytest

from pipeline.stage_a.rank_design import Batch, design_summary, make_design


def _ids(n):
    return [f"v{i:03d}" for i in range(n)]


def _appearances(batches):
    return Counter(v for b in batches for v in b.video_ids)


@pytest.mark.parametrize("n,size,appearances", [(50, 5, 2), (11, 5, 2), (5, 5, 2), (7, 3, 3), (23, 4, 1),
                                                (2, 5, 2), (9, 2, 2), (6, 5, 3)])
def test_every_video_appears_exactly_appearances_times(n, size, appearances):
    batches = make_design(_ids(n), batch_size=size, appearances=appearances, seed=3)
    assert _appearances(batches) == {v: appearances for v in _ids(n)}
    for b in batches:
        assert len(set(b.video_ids)) == len(b.video_ids)  # never twice in one batch


def test_batches_are_numbered_and_labelled_in_order():
    batches = make_design(_ids(12), batch_size=5, appearances=2, seed=0)
    assert [b.batch_id for b in batches] == [f"b{i:02d}" for i in range(1, len(batches) + 1)]
    for b in batches:
        assert isinstance(b, Batch)
        assert b.labels == tuple("ABCDE"[:len(b.video_ids)])
        assert isinstance(b.video_ids, tuple) and isinstance(b.labels, tuple)


def test_deterministic_for_a_seed_and_different_for_another():
    a = make_design(_ids(30), batch_size=5, appearances=2, seed=7)
    assert a == make_design(_ids(30), batch_size=5, appearances=2, seed=7)
    assert a != make_design(_ids(30), batch_size=5, appearances=2, seed=8)


def test_input_order_is_irrelevant():
    ids = _ids(17)
    forward = make_design(ids, batch_size=4, appearances=2, seed=1)
    assert forward == make_design(list(reversed(ids)), batch_size=4, appearances=2, seed=1)
    assert forward == make_design(ids[5:] + ids[:5], batch_size=4, appearances=2, seed=1)


def test_fifty_videos_make_twenty_batches_with_no_repeated_pair():
    batches = make_design(_ids(50), batch_size=5, appearances=2, seed=0)
    summary = design_summary(batches)
    assert len(batches) == 20 and summary["n_batches"] == 20 and summary["n_videos"] == 50
    assert summary["batch_sizes"] == [5] * 20
    assert summary["repeated_pairs"] == 0
    pairs = Counter(frozenset(p) for b in batches for p in combinations(b.video_ids, 2))
    assert max(pairs.values()) == 1


@pytest.mark.parametrize("seed", range(6))
def test_no_repeated_pair_for_other_seeds_and_three_rounds(seed):
    assert design_summary(make_design(_ids(50), 5, 2, seed))["repeated_pairs"] == 0
    assert design_summary(make_design(_ids(60), 5, 3, seed))["repeated_pairs"] == 0


@pytest.mark.parametrize("seed", range(8))
def test_five_videos_make_two_batches_of_all_five_in_different_orders(seed):
    batches = make_design(_ids(5), batch_size=5, appearances=2, seed=seed)
    assert len(batches) == 2
    first, second = batches
    assert sorted(first.video_ids) == sorted(second.video_ids) == _ids(5)
    assert first.video_ids != second.video_ids
    summary = design_summary(batches)
    assert summary["repeated_pairs"] == 10  # every pair meets twice; unavoidable and counted
    assert summary["batch_sizes"] == [5, 5]


def test_fewer_videos_than_the_batch_size_is_one_batch_per_round():
    batches = make_design(_ids(3), batch_size=5, appearances=3, seed=0)
    assert [len(b.video_ids) for b in batches] == [3, 3, 3]
    assert len({b.video_ids for b in batches}) == 3
    assert all(b.labels == ("A", "B", "C") for b in batches)


@pytest.mark.parametrize("seed", range(5))
def test_eleven_videos_have_no_batch_of_one(seed):
    batches = make_design(_ids(11), batch_size=5, appearances=2, seed=seed)
    sizes = [len(b.video_ids) for b in batches]
    assert sizes == [5, 4, 2, 5, 4, 2]
    assert min(sizes) >= 2


def test_batch_size_two_with_an_odd_count_has_no_batch_of_one():
    batches = make_design(_ids(7), batch_size=2, appearances=2, seed=0)
    sizes = [len(b.video_ids) for b in batches]
    assert min(sizes) >= 2 and sum(sizes) == 14


def test_design_summary_counts():
    batches = [Batch("b01", ("x", "y", "z"), ("A", "B", "C")), Batch("b02", ("y", "x"), ("A", "B"))]
    assert design_summary(batches) == {
        "n_videos": 3, "n_batches": 2, "appearances": {"x": 2, "y": 2, "z": 1},
        "repeated_pairs": 1, "batch_sizes": [3, 2],
    }


@pytest.mark.parametrize("kwargs", [
    {"video_ids": ["only"]},
    {"video_ids": []},
    {"video_ids": ["a", "b", "c"], "batch_size": 1},
    {"video_ids": ["a", "b", "c"], "appearances": 0},
    {"video_ids": ["a", "b", "a"]},
    {"video_ids": [f"v{i}" for i in range(40)], "batch_size": 27},
])
def test_invalid_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        make_design(**kwargs)
