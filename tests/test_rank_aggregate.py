"""Plackett-Luce and mean-rank aggregation of batch rankings."""

import math
import random

import pytest

from pipeline.stage_a.rank_aggregate import aggregate, mean_normalised_rank, plackett_luce


def _spearman(a: dict, b: dict) -> float:
    def ranks(scores):
        ordered = sorted(scores, key=lambda k: scores[k])
        return {k: i for i, k in enumerate(ordered)}
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((ra[k] - rb[k]) ** 2 for k in a)
    return 1 - 6 * d2 / (n * (n * n - 1))


def _simulated_orders(worths: dict, n_batches: int, size: int, seed: int) -> list[list[str]]:
    """Plackett-Luce samples: sort a random batch by log-worth plus Gumbel noise."""
    rng = random.Random(seed)
    items = sorted(worths)
    orders = []
    for _ in range(n_batches):
        batch = rng.sample(items, size)
        noisy = {i: worths[i] - math.log(-math.log(rng.random())) for i in batch}
        orders.append(sorted(batch, key=lambda i: -noisy[i]))
    return orders


def test_single_order_gives_strictly_decreasing_scores():
    scores = plackett_luce([["A", "B", "C"]])
    assert set(scores) == {"A", "B", "C"}
    assert scores["A"] > scores["B"] > scores["C"]


def test_reversing_all_pairwise_orders_negates_the_scores():
    # Batches of two are where the Plackett-Luce likelihood is symmetric under reversal.
    orders = [["A", "B"], ["A", "B"], ["B", "A"], ["B", "C"], ["C", "D"], ["A", "D"], ["D", "B"], ["A", "C"],
              ["C", "D"]]
    forward = plackett_luce(orders)
    backward = plackett_luce([list(reversed(o)) for o in orders])
    assert len({round(v, 6) for v in forward.values()}) == 4  # a non-trivial fit
    for item in forward:
        assert backward[item] == pytest.approx(-forward[item], abs=1e-6)


def test_reversing_a_longer_order_mirrors_the_scores_without_negating_them():
    # With three or more clips per order Plackett-Luce is not symmetric under reversal: the
    # first place is a choice among all the clips, the last place among none. Reversing
    # A>B>C therefore hands A's score to C and back, but the middle clip does not sit at zero.
    single = plackett_luce([["A", "B", "C"]])
    mirrored = plackett_luce([["C", "B", "A"]])
    assert mirrored["C"] == pytest.approx(single["A"], abs=1e-9)
    assert mirrored["B"] == pytest.approx(single["B"], abs=1e-9)
    assert mirrored["A"] == pytest.approx(single["C"], abs=1e-9)
    assert mirrored["C"] > mirrored["B"] > mirrored["A"]
    assert abs(single["B"]) > 1e-3 and mirrored["B"] != pytest.approx(-single["B"], abs=1e-6)


def test_planted_worths_are_recovered_from_noisy_batches():
    planted = {f"v{i:02d}": 0.35 * i for i in range(20)}
    orders = _simulated_orders(planted, n_batches=200, size=5, seed=11)
    scores = plackett_luce(orders)
    assert set(scores) == set(planted)
    assert _spearman(scores, planted) >= 0.9
    assert _spearman(mean_normalised_rank(orders), planted) >= 0.8


def test_always_first_and_always_last_items_have_finite_scores():
    orders = [["top", "a", "b"], ["top", "b", "c"], ["top", "c", "a"], ["a", "b", "bottom"], ["c", "a", "bottom"]]
    scores = plackett_luce(orders)
    assert all(math.isfinite(v) for v in scores.values())
    assert scores["top"] == max(scores.values()) and scores["bottom"] == min(scores.values())
    assert abs(scores["top"]) < 10 and abs(scores["bottom"]) < 10


def test_a_weaker_prior_lets_an_unbeaten_item_run_further():
    orders = [["top", "a", "b"], ["top", "b", "a"]]
    assert plackett_luce(orders, prior=0.05)["top"] > plackett_luce(orders, prior=0.5)["top"]


def test_scores_are_centred():
    orders = _simulated_orders({f"v{i}": 0.5 * i for i in range(8)}, n_batches=30, size=4, seed=2)
    scores = plackett_luce(orders)
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-9)


def test_permuting_the_list_of_orders_does_not_change_the_result():
    orders = _simulated_orders({f"v{i}": 0.5 * i for i in range(8)}, n_batches=30, size=4, seed=5)
    shuffled = list(orders)
    random.Random(1).shuffle(shuffled)
    assert shuffled != orders
    assert plackett_luce(shuffled) == plackett_luce(orders)
    assert mean_normalised_rank(shuffled) == pytest.approx(mean_normalised_rank(orders))


def test_items_and_edge_cases():
    assert plackett_luce([]) == {}
    assert plackett_luce([["solo"]]) == {"solo": 0.0}
    scores = plackett_luce([["A", "B"], ["C"]])
    assert list(scores) == ["A", "B", "C"]  # sorted, deterministic
    assert scores["A"] > scores["C"] > scores["B"]
    with pytest.raises(ValueError):
        plackett_luce([["A", "B", "A"]])
    with pytest.raises(ValueError):
        plackett_luce([["A", "B"]], prior=0)


def test_mean_normalised_rank_hand_computed():
    orders = [["A", "B", "C"], ["B", "A"], ["C", "A", "B", "D", "E"], ["E"]]
    result = mean_normalised_rank(orders)
    assert result == pytest.approx({
        "A": (1.0 + 0.0 + 0.75) / 3,
        "B": (0.5 + 1.0 + 0.5) / 3,
        "C": (0.0 + 1.0) / 2,
        "D": 0.25,
        "E": (0.0 + 0.5) / 2,
    })
    assert list(result) == ["A", "B", "C", "D", "E"]


def test_aggregate_shapes_one_block_per_code():
    rankings = {
        "pitch_positivity": [["v1", "v2", "v3"], ["v2", "v1", "v3"], ["v1", "v3", "v2"]],
        "expressed_enthusiasm": [["v3", "v2", "v1"]],
    }
    out = aggregate(rankings)
    assert set(out) == {"pitch_positivity", "expressed_enthusiasm"}
    block = out["pitch_positivity"]
    assert set(block) == {"pl_score", "mean_rank", "n_orders", "order"}
    assert block["n_orders"] == 3 and out["expressed_enthusiasm"]["n_orders"] == 1
    assert block["pl_score"] == plackett_luce(rankings["pitch_positivity"])
    assert block["mean_rank"] == mean_normalised_rank(rankings["pitch_positivity"])
    assert block["order"] == ["v1", "v2", "v3"]
    assert out["expressed_enthusiasm"]["order"] == ["v3", "v2", "v1"]


def test_aggregate_breaks_score_ties_by_id():
    out = aggregate({"x": [["b", "a"], ["a", "b"]]})
    assert out["x"]["pl_score"]["a"] == pytest.approx(out["x"]["pl_score"]["b"])
    assert out["x"]["order"] == ["a", "b"]
