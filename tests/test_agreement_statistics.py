"""The agreement statistics per kind of code (plan D Task 1). Every expected value is worked out by hand here."""
import math

import pytest

from pipeline.reliability.agreement import (binomial_interval, binomial_p_vs_half, bias_index, chance_exact,
                                            chance_within_one, clock_agreement, gwet_ac1, gwet_ac2,
                                            krippendorff_alpha, landis_koch_band, mean_difference, modal_share,
                                            modal_shares_by_rater, prevalence_index, spearman, specific_agreement, within_one_share)

SCALE = [1, 2, 3, 4, 5]


def test_ac1_two_categories_balanced():
    pairs = [("Y", "Y")] * 4 + [("N", "N")] * 4 + [("Y", "N"), ("N", "Y")]
    # pa = 8/10; each rater says Y 5 of 10 -> pi_Y = pi_N = 0.5; pe = (0.5*0.5 + 0.5*0.5) / (2 - 1) = 0.5
    pa, pe = 0.8, 0.5
    assert gwet_ac1(pairs, ["Y", "N"]) == pytest.approx((pa - pe) / (1 - pe))
    assert gwet_ac1(pairs, ["Y", "N"]) == pytest.approx(0.6)


def test_ac1_is_stable_under_prevalence_where_kappa_collapses():
    pairs = [("A", "A")] * 90 + [("B", "B")] * 5 + [("A", "B")] * 3 + [("B", "A")] * 2
    # pa = 95/100; rater a says A 93 times, rater b 92 times -> pi_A = (0.93 + 0.92)/2 = 0.925, pi_B = 0.075
    pa = 0.95
    pe = (0.925 * 0.075 + 0.075 * 0.925) / (2 - 1)                  # 0.13875
    ac1 = gwet_ac1(pairs, ["A", "B"])
    assert ac1 == pytest.approx((pa - pe) / (1 - pe))               # 0.9419
    assert ac1 > 0.85
    # alpha: 200 values, n_A = 185, n_B = 15; Do = 10/200, De = 2*185*15/(200*199) -> 1 - 0.05/0.13945 = 0.641
    alpha = krippendorff_alpha([list(p) for p in pairs], "nominal")
    assert alpha == pytest.approx(1 - (10 / 200) / (2 * 185 * 15 / (200 * 199)))
    assert alpha < 0.65


def test_ac1_three_categories_with_a_dominant_one():
    # table of counts, rows rater a, columns rater b, categories X Y Z
    table = [[80, 4, 2], [3, 5, 1], [2, 1, 2]]
    cats = ["X", "Y", "Z"]
    pairs = [(cats[i], cats[j]) for i in range(3) for j in range(3) for _ in range(table[i][j])]
    pa = (80 + 5 + 2) / 100
    # rows 86, 9, 5; columns 85, 10, 5 -> pi = (.855, .095, .05)
    spread = 0.855 * 0.145 + 0.095 * 0.905 + 0.05 * 0.95             # 0.25745
    pe = spread / (3 - 1)                                            # 0.128725: the 1/(q - 1) factor matters
    assert gwet_ac1(pairs, cats) == pytest.approx((pa - pe) / (1 - pe))
    assert gwet_ac1(pairs, cats) == pytest.approx(0.850793, abs=1e-6)
    assert gwet_ac1(pairs, ["Z", "X", "Y"]) == pytest.approx(gwet_ac1(pairs, cats))     # order of categories


def test_ac1_and_ac2_none_with_fewer_than_two_pairs():
    assert gwet_ac1([("A", "A")], ["A", "B"]) is None
    assert gwet_ac1([], ["A", "B"]) is None
    assert gwet_ac2([(1, 2)], SCALE) is None


def test_ac2_quadratic_equals_ac1_when_two_categories():
    pairs = [("A", "A")] * 90 + [("B", "B")] * 5 + [("A", "B")] * 3 + [("B", "A")] * 2
    # q = 2: weights 1 - ((i - j)/1)^2 are the identity, sum of weights 2 = q(q - 1), so pe is AC1's
    assert gwet_ac2(pairs, ["A", "B"]) == pytest.approx(gwet_ac1(pairs, ["A", "B"]))
    assert gwet_ac2(pairs, ["A", "B"]) == pytest.approx((0.95 - 0.13875) / (1 - 0.13875))


def test_ac2_rewards_near_misses():
    pairs = [(1, 2), (2, 3), (3, 4), (4, 5), (2, 1), (3, 2), (4, 3), (5, 4)]
    # every pair one step apart: weight 1 - (1/4)^2 = 15/16 -> pa = 15/16
    pa = 15 / 16
    # each rater: 1 once, 2-4 twice, 5 once of 8 -> pi = (1/8, 2/8, 2/8, 2/8, 1/8)
    spread = 2 * (1 / 8) * (7 / 8) + 3 * (2 / 8) * (6 / 8)            # 50/64
    # sum of (i - j)^2 over the 5x5 table: 2 * (4*1 + 3*4 + 2*9 + 1*16) = 100 -> sum of weights 25 - 100/16 = 18.75
    pe = spread * 18.75 / (5 * 4)                                    # 0.732421875
    ac2 = gwet_ac2(pairs, SCALE)
    assert ac2 == pytest.approx((pa - pe) / (1 - pe))               # 0.766
    assert ac2 > 0.5
    assert sum(a == b for a, b in pairs) / len(pairs) == 0
    # chance exact: sum of (share of a) x (share of b) = 2 * (1/8)^2 + 3 * (2/8)^2
    assert chance_exact(pairs) == pytest.approx(2 * (1 / 8) ** 2 + 3 * (2 / 8) ** 2)    # 14/64


def test_gwet_rejects_a_value_outside_the_categories():
    with pytest.raises(ValueError):
        gwet_ac1([("A", "A"), ("A", "C")], ["A", "B"])
    with pytest.raises(ValueError):
        gwet_ac2([(1, 1), (2, 2)], [1, 2, 2, 3])                      # a category listed twice


def test_ordinal_functions_reject_off_scale_and_mixed_types():
    off = [(4, 4), (6, 5)]
    mixed = [("4", 4), (5, 5)]                                       # the string "4" is not the category 4
    for bad in (off, mixed):
        for f in (within_one_share, chance_within_one, mean_difference, spearman):
            with pytest.raises(ValueError):
                f(bad, SCALE)
        with pytest.raises(ValueError):
            chance_exact(bad, SCALE)
    # without a scale "4" is a value of its own: a says "4" .5, 5 .5; b says 4 .5, 5 .5 -> only 5 meets 5
    assert chance_exact(mixed) == pytest.approx(0.5 * 0.0 + 0.0 * 0.5 + 0.5 * 0.5)


def test_specific_agreement_and_indices_on_a_2x2():
    # a = both yes, b = rater a yes and rater b no, c = rater a no and rater b yes, d = both no
    pairs = [(1, 1)] * 18 + [(1, 0)] * 9 + [(0, 1)] * 2 + [(0, 0)] * 9
    positive, negative = specific_agreement(pairs, 1)
    assert positive == pytest.approx(2 * 18 / (36 + 11))
    assert negative == pytest.approx(2 * 9 / (18 + 11))
    assert prevalence_index(pairs, 1) == pytest.approx(abs(18 - 9) / 38)
    value, who = bias_index(pairs, 1)
    assert value == pytest.approx(abs(9 - 2) / 38)
    assert who == "a"                                               # rater a says yes 27 times, rater b 20
    reversed_pairs = [(b, a) for a, b in pairs]
    assert bias_index(reversed_pairs, 1)[1] == "b"
    assert bias_index([(1, 0), (0, 1)], 1) == (0.0, "none")


def test_binary_functions_reject_a_third_value_or_another_type():
    # the review labels are strings, a model's JSON answer a boolean
    mixed = [("true", "true"), (True, True), (False, False)]
    for f in (specific_agreement, prevalence_index, bias_index):
        with pytest.raises(ValueError):
            f(mixed, True)                                           # two values besides True: "true" and False
        with pytest.raises(ValueError):
            f(mixed, True, False)                                    # "true" is neither True nor False
        with pytest.raises(ValueError):
            f([(1, 1), (True, 0)], 1, 0)                             # True is not 1
        with pytest.raises(ValueError):
            f([(1, 1)], 1, 1)                                        # positive and negative the same
    # strings on both sides: the default negative is the one other value
    strings = [("true", "true"), ("true", "false"), ("false", "false")]
    assert specific_agreement(strings, "true") == specific_agreement(strings, "true", "false")
    assert specific_agreement(strings, "true") == pytest.approx((2 * 1 / (2 + 1), 2 * 1 / (2 + 1)))


def test_specific_agreement_none_without_a_positive_or_negative():
    assert specific_agreement([(0, 0), (0, 0)], 1) == (None, 1.0)
    assert specific_agreement([(1, 1)], 1) == (1.0, None)


def test_modal_share():
    pairs = [("A", "A")] * 90 + [("B", "B")] * 5 + [("A", "B")] * 3 + [("B", "A")] * 2
    assert modal_share(pairs) == pytest.approx(185 / 200)
    assert modal_share([("x", "y"), ("y", "z")]) == pytest.approx(2 / 4)
    # per rater: a says A 93 of 100, b says A 92 of 100
    assert modal_shares_by_rater(pairs) == pytest.approx((93 / 100, 92 / 100))


def test_empty_input_gives_nan_or_none():
    nan = [prevalence_index([], 1), bias_index([], 1)[0], modal_share([]), within_one_share([], SCALE),
           chance_within_one([], SCALE), chance_exact([]), mean_difference([], SCALE), *modal_shares_by_rater([]),
           *binomial_interval(0, 0), binomial_p_vs_half(0, 0)]
    assert all(math.isnan(x) for x in nan)
    assert bias_index([], 1)[1] == "none"
    assert specific_agreement([], 1) == (None, None)
    assert spearman([], SCALE) is None and gwet_ac1([], [1, 2]) is None and gwet_ac2([], SCALE) is None


def test_chance_within_one_equals_marginal_product():
    # rater a uses only 4 and 5, rater b 2 to 5
    pairs = [(4, 2), (5, 3), (4, 4), (5, 5)]
    # a: 4 -> 0.5, 5 -> 0.5; b: 2, 3, 4, 5 -> 0.25 each
    # within one: a=4 with b in {3, 4, 5} -> 0.5 * 0.75; a=5 with b in {4, 5} -> 0.5 * 0.5
    assert chance_within_one(pairs, SCALE) == pytest.approx(0.5 * 0.75 + 0.5 * 0.5)       # 0.625
    assert within_one_share(pairs, SCALE) == pytest.approx(2 / 4)
    # exact: a=4 with b=4, a=5 with b=5
    assert chance_exact(pairs) == pytest.approx(0.5 * 0.25 + 0.5 * 0.25)                   # 0.25


def test_mean_difference_and_spearman_signs():
    up = [(2, 1), (3, 2), (4, 3), (5, 4)]
    assert mean_difference(up, SCALE) == pytest.approx(1.0)          # a is one step above b each time
    assert spearman(up, SCALE) == pytest.approx(1.0)
    down = [(1, 4), (2, 3), (3, 2), (4, 1)]
    assert mean_difference(down, SCALE) == pytest.approx((-3 - 1 + 1 + 3) / 4)
    assert spearman(down, SCALE) == pytest.approx(-1.0)
    labelled = [("low", "high"), ("high", "high")]
    assert mean_difference(labelled, ["low", "mid", "high"]) == pytest.approx(((0 - 2) + (2 - 2)) / 2)


def test_spearman_none_when_constant():
    assert spearman([(3, 1), (3, 2), (3, 4)], SCALE) is None
    assert spearman([(1, 3), (2, 3), (4, 3)], SCALE) is None
    assert spearman([(1, 3)], SCALE) is None


def test_clock_agreement_fields():
    diffs = [2, -3, 4, 6, 10, 14, 16, 19, 20, 20]
    out = clock_agreement(diffs)
    assert out["n"] == 10
    assert out["share_within_5"] == pytest.approx(3 / 10)           # 2, -3, 4
    assert out["share_within_30"] == pytest.approx(1.0)
    assert out["median_abs"] == pytest.approx((10 + 14) / 2)        # |d| sorted: 2 3 4 6 10 14 16 19 20 20
    assert out["mean_signed"] == pytest.approx(108 / 10)
    # squared deviations from 10.8 sum to 611.6; sample sd = sqrt(611.6 / 9)
    sd = math.sqrt(611.6 / 9)
    assert out["loa_low"] == pytest.approx(10.8 - 1.96 * sd)
    assert out["loa_high"] == pytest.approx(10.8 + 1.96 * sd)


def test_clock_agreement_empty_and_single():
    empty = clock_agreement([])
    assert empty["n"] == 0 and empty["share_within_5"] is None and empty["loa_low"] is None
    one = clock_agreement([-7])
    assert one["share_within_5"] == 0.0 and one["median_abs"] == 7.0 and one["loa_low"] is None


@pytest.mark.parametrize('a,b,expected', [
    (340, 352, True),  # same decision turn despite a 12-second difference
    (354, 355, False),  # adjacent turns despite a one-second difference
    (340, 355, False),  # the right endpoint belongs to the following turn
    (339, 340, False),
    (None, 340, None), (float('nan'), 340, None), (float('inf'), 340, None),
    (380, 340, None),  # outside all reference turns
])
def test_same_turn_agreement(a, b, expected):
    from pipeline.reliability.agreement import same_turn_agreement
    turns = [(30, 326, 340), (31, 340, 355), (32, 355, 375)]
    assert same_turn_agreement(a, b, turns)['agreed'] is expected


def test_same_turn_agreement_ambiguous_segmentation_is_not_counted():
    from pipeline.reliability.agreement import same_turn_agreement
    assert same_turn_agreement(352, 340, [(31, 340, 355), (32, 350, 375)])['agreed'] is None


def test_clock_agreement_rejects_non_finite():
    with pytest.raises(ValueError):
        clock_agreement([1, float("nan"), 3])
    with pytest.raises(ValueError):
        clock_agreement([1, float("inf")])


def _upper_tail(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def test_binomial_interval_and_p():
    low, high = binomial_interval(15, 25)
    assert low == pytest.approx(0.387, abs=0.001)
    assert high == pytest.approx(0.789, abs=0.001)
    # Clopper-Pearson by definition: P(X >= 15 | low) = 0.025 and P(X <= 15 | high) = 0.025
    assert _upper_tail(15, 25, low) == pytest.approx(0.025, abs=1e-6)
    assert 1 - _upper_tail(16, 25, high) == pytest.approx(0.025, abs=1e-6)
    # two-sided against 0.5, symmetric: 2 * P(X >= 15) = 2 * sum C(25, i) / 2^25 for i = 15..25
    p = 2 * sum(math.comb(25, i) for i in range(15, 26)) / 2 ** 25
    assert binomial_p_vs_half(15, 25) == pytest.approx(p)
    assert binomial_p_vs_half(15, 25) == pytest.approx(0.424, abs=0.001)


def test_binomial_interval_edges():
    low, high = binomial_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(1 - 0.025 ** (1 / 10))
    low, high = binomial_interval(10, 10)
    assert high == 1.0 and low == pytest.approx(0.025 ** (1 / 10))
    for k, n in ((5, 3), (-1, 3), (1, -1)):
        with pytest.raises(ValueError):
            binomial_interval(k, n)
        with pytest.raises(ValueError):
            binomial_p_vs_half(k, n)


def test_landis_koch_bands():
    assert landis_koch_band(-0.01) == "poor"
    assert landis_koch_band(0.0) == "slight"
    assert landis_koch_band(0.20) == "slight"
    assert landis_koch_band(0.21) == "fair"
    assert landis_koch_band(0.40) == "fair"
    assert landis_koch_band(0.41) == "moderate"
    assert landis_koch_band(0.60) == "moderate"
    assert landis_koch_band(0.61) == "substantial"
    assert landis_koch_band(0.80) == "substantial"
    assert landis_koch_band(0.81) == "almost perfect"
    assert landis_koch_band(1.0) == "almost perfect"
    # banded at two decimals, as printed: 0.2001 and 0.204 print 0.20, 0.8001 prints 0.80, -0.004 prints -0.00
    assert landis_koch_band(0.2001) == "slight"
    assert landis_koch_band(0.204) == "slight"
    assert landis_koch_band(0.1 * 3 - 0.1) == "slight"                # 0.20000000000000004
    assert landis_koch_band(0.8001) == "substantial"
    assert landis_koch_band(-0.004) == "slight"
    assert landis_koch_band(-0.006) == "poor"                        # prints -0.01
    assert landis_koch_band(None) is None
    assert landis_koch_band(float("nan")) is None
