import random
import time

import numpy as np
import pytest

from pipeline.reliability.agreement import (MIN_DEFINED_SHARE, alpha_units, bootstrap_alpha, bootstrap_alpha_detail,
                                            krippendorff_alpha, stratified_alpha)
from pipeline.stage_b.review_apply import alpha_level

# Krippendorff (2011), four observers, twelve units; published values
_ = None
DATA = [[1, 1, _, 1], [2, 2, 3, 2], [3, 3, 3, 3], [3, 3, 3, 3], [2, 2, 2, 2], [1, 2, 3, 4],
        [4, 4, 4, 4], [1, 1, 2, 1], [2, 2, 2, 2], [_, 5, 5, 5], [_, _, 1, 1], [_, 3, _, _]]


def test_published_values():
    assert krippendorff_alpha(DATA, "nominal") == pytest.approx(0.743, abs=0.001)
    assert krippendorff_alpha(DATA, "interval") == pytest.approx(0.849, abs=0.001)
    assert krippendorff_alpha(DATA, "ordinal") == pytest.approx(0.815, abs=0.001)


def test_strings_perfect_agreement_and_degenerate_cases():
    assert krippendorff_alpha([["a", "a"], ["b", "b"], ["a", "a"]], "nominal") == 1.0
    assert krippendorff_alpha([["a", "b"], ["b", "a"]], "nominal") < 0
    assert krippendorff_alpha([["a", "a"], ["a", "a"]], "nominal") is None          # no expected disagreement
    assert krippendorff_alpha([["a", None], [None, "b"]], "nominal") is None        # no unit with two values
    assert krippendorff_alpha([], "nominal") is None
    assert krippendorff_alpha([["1", "2"], ["3", "3"], ["x", "1"]], "ordinal") == \
        krippendorff_alpha([[1, 2], [3, 3]], "ordinal")                             # the unconvertible unit drops


def test_alpha_ignores_the_order_of_coders_and_of_units():
    shuffled = [list(reversed(u)) for u in reversed(DATA)]
    for level in ("nominal", "ordinal", "interval"):
        assert krippendorff_alpha(shuffled, level) == pytest.approx(krippendorff_alpha(DATA, level), abs=1e-12)
    rotated = [u[1:] + u[:1] for u in DATA]
    assert krippendorff_alpha(rotated, "nominal") == pytest.approx(krippendorff_alpha(DATA, "nominal"), abs=1e-12)


def test_two_coders_skewed_field_alpha_is_far_below_percent_agreement():
    # 18 units both "a", 1 both "b", 1 split ("a","b"): percent agreement 19/20 = 0.95.
    # Coincidences: o_aa = 36, o_bb = 2, o_ab = o_ba = 1 (the split unit, m=2, adds 1/(m-1) each way).
    # n_a = 37, n_b = 3, n = 40. Observed disagreement = 2 (the two off-diagonal cells).
    # Expected = 2 * n_a * n_b / (n - 1) = 2 * 37 * 3 / 39 = 5.6923. alpha = 1 - 2 / 5.6923 = 0.6486.
    units = [["a", "a"]] * 18 + [["b", "b"]] + [["a", "b"]]
    assert krippendorff_alpha(units, "nominal") == pytest.approx(0.649, abs=0.0005)


def test_bootstrap_interval_contains_the_point_estimate_and_is_seeded():
    units = {f"v{i}": [[1, 1], [2, 2], [1, 2 if i % 3 else 1]] for i in range(8)}
    low, high = bootstrap_alpha(units, "nominal", n_boot=200, seed=1)
    point = krippendorff_alpha([u for us in units.values() for u in us], "nominal")
    assert low <= point <= high and bootstrap_alpha(units, "nominal", n_boot=200, seed=1) == [low, high]
    assert bootstrap_alpha({}, "nominal") is None


def test_alpha_level_from_the_path():
    assert alpha_level("stage_a:interpersonal_warmth") == "ordinal"
    assert alpha_level("turn:4/code:PR-SEV") == "ordinal"
    assert alpha_level("decision:2/time") == "interval"
    assert alpha_level("exchange:1/topic") == "nominal"
    assert alpha_level("prepared_window") is None


def test_stratified_alpha_with_one_stratum_is_the_plain_alpha():
    for level in ("nominal", "ordinal", "interval"):
        assert stratified_alpha({"x": DATA}, level) == pytest.approx(krippendorff_alpha(DATA, level), abs=1e-12)
    rng = random.Random(3)
    for _ in range(20):
        units = [[rng.choice([1, 2, 3, 4, None]) for _ in range(rng.choice([2, 3]))] for _ in range(rng.randint(2, 15))]
        for level in ("nominal", "ordinal", "interval"):
            got, want = stratified_alpha({"x": units}, level), krippendorff_alpha(units, level)
            assert (got is None and want is None) or got == pytest.approx(want, abs=1e-12)
    assert stratified_alpha({}, "nominal") is None
    assert stratified_alpha({"a": [["a", "a"]], "b": [["a", "b"]]}, "nominal") is None     # no stratum has two units


def test_stratified_alpha_keeps_variables_apart_where_the_pooled_alpha_is_inflated():
    # two Stage A items at opposite ends of the scale; each item: units [1,2] and [2,2] -> n = (1, 3), observed 2,
    # expected 2*1*3/3 = 2, alpha 0. Pooled over both items the between-item spread enters the expected
    # disagreement: alpha = 31/38.
    a, b = [[1, 2], [2, 2]], [[4, 5], [5, 5]]
    assert krippendorff_alpha(a, "ordinal") == pytest.approx(0.0, abs=1e-12)
    assert krippendorff_alpha(b, "ordinal") == pytest.approx(0.0, abs=1e-12)
    assert stratified_alpha({"a": a, "b": b}, "ordinal") == pytest.approx(0.0, abs=1e-12)
    assert krippendorff_alpha(a + b, "ordinal") == pytest.approx(31 / 38)


def _reference_interval(strata_by_video, level, n_boot, seed):
    """The straightforward bootstrap: recompute stratified alpha from the picked videos' units on every draw."""
    names = list(strata_by_video.values())
    picked = np.random.default_rng(seed).integers(len(names), size=(n_boot, len(names)))
    draws = []
    for row in picked:
        strata = {}
        for i in row:
            for s, units in names[i].items():
                strata.setdefault(s, []).extend(units)
        a = stratified_alpha(strata, level)
        if a is not None:
            draws.append(a)
    return np.quantile(draws, [0.025, 0.975]).tolist(), len(draws)


@pytest.mark.parametrize("level", ["nominal", "ordinal", "interval"])
def test_matrix_sum_bootstrap_equals_a_recomputation_per_draw(level):
    rng = random.Random(5)
    videos = {f"v{i}": {s: [[rng.choice([1, 2, 3, 4]), rng.choice([1, 2, 3, 4])] for _ in range(rng.randint(2, 6))]
                        for s in ("x", "y")} for i in range(9)}
    for i, per in enumerate(videos.values()):          # a stratum of one unit in some videos: it counts only in draws
        if i % 3 == 0:                                  # that pick it twice
            per["z"] = [[rng.choice([1, 2, 3]), rng.choice([1, 2, 3])]]
    detail = bootstrap_alpha_detail(videos, level, n_boot=300, seed=7)
    want, defined = _reference_interval(videos, level, 300, 7)
    assert detail["n_boot"] == 300 and detail["n_defined"] == defined
    assert detail["ci95"] == pytest.approx(want, abs=1e-9)


def test_a_bootstrap_whose_draws_are_often_undefined_gives_no_interval():
    # one video carries the only disagreement: a draw without it (about a third of them) has no variation
    videos = {f"v{i}": {"": [[1, 1], [1, 1]]} for i in range(5)}
    videos["v5"] = {"": [[1, 2], [2, 2]]}
    detail = bootstrap_alpha_detail(videos, "nominal", n_boot=500, seed=0)
    assert detail["n_boot"] == 500 and detail["n_defined"] < 500
    assert detail["n_defined"] < MIN_DEFINED_SHARE * 500 and detail["ci95"] is None
    assert bootstrap_alpha({v: s[""] for v, s in videos.items()}, "nominal", n_boot=500, seed=0) is None
    assert bootstrap_alpha_detail({}, "nominal") is None and bootstrap_alpha_detail({"v": {"": [["a"]]}}, "nominal") is None


def test_the_bootstrap_of_thirty_big_videos_is_fast():
    rng = random.Random(1)
    videos = {f"v{i}": [[c, c if rng.random() < 0.8 else rng.choice("abc")] for c in
                        (rng.choice("abc") for _ in range(200))] for i in range(30)}
    start = time.perf_counter()
    interval = bootstrap_alpha(videos, "nominal")
    assert time.perf_counter() - start < 2.0 and interval[0] < interval[1]


def test_single_unit_strata_estimate_nothing_and_do_not_move_a_real_stratum():
    # three items, one unit each, one of them disagreeing: no stratum can estimate its expected disagreement
    assert stratified_alpha({"a": [["x", "x"]], "b": [["x", "y"]], "c": [["y", "y"]]}, "nominal") is None
    # ten ordinal items, nine disagreeing by three points / one disagreeing by one point
    assert stratified_alpha({f"i{k}": [[1, 4] if k else [1, 1]] for k in range(10)}, "ordinal") is None
    assert stratified_alpha({f"i{k}": [[1, 2] if k == 0 else [3, 3]] for k in range(10)}, "ordinal") is None
    # a real stratum: units [a,a] [a,b] [b,b] [b,b] -> n_a = 3, n_b = 5, observed 2, expected 2*3*5/7; alpha 8/15
    real = [["a", "a"], ["a", "b"], ["b", "b"], ["b", "b"]]
    assert krippendorff_alpha(real, "nominal") == pytest.approx(8 / 15)
    for single in ([["p", "q"]], [["p", "p"]]):
        extra = {f"s{k}": single for k in range(5)}
        assert stratified_alpha({"real": real, **extra}, "nominal") == pytest.approx(8 / 15)


def test_alpha_units_counts_the_pairable_units_of_the_strata_that_qualify():
    rated = [["1", "2"], ["3", "3"], ["2", "missing:too_short"], ["", "1"], [None, "2"]]
    assert alpha_units({"a": rated}, "ordinal") == 2                    # a side that is no number drops the unit
    assert alpha_units({"a": rated}, "nominal") == 3                    # `missing:too_short` is a category there
    assert alpha_units({"a": rated, "b": [["1", "1"]]}, "ordinal") == 2      # a stratum of one unit adds none
    assert alpha_units({}, "nominal") == 0
