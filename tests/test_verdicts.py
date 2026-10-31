"""The pre-registered decision rules of the pitch-level test (refocus spec section 6), on hand-built deltas."""
import numpy as np
import pandas as pd
import pytest

from analysis.verdicts import AUC_USED, NOT_DISTINGUISHABLE, RULES, SUPPORTED, VERDICT_COLUMNS, verdicts


def _metrics(**meaningful: bool) -> pd.DataFrame:
    return pd.DataFrame({"model": list(meaningful), "auc_meaningful": list(meaningful.values())})


FINE = _metrics(C2p=True, C3q=True, C3=True)      # every AUC of the ladder meaningful: the AUC route is used


def _deltas(rows: dict[str, dict[str, tuple[float, float, float]]]) -> pd.DataFrame:
    """A deltas frame as `evaluate` gives it: {comparison: {metric: (delta, low, high)}}."""
    return pd.DataFrame([{"comparison": comparison, "metric": metric, "delta": d, "ci_low": low, "ci_high": high}
                         for comparison, metrics in rows.items() for metric, (d, low, high) in metrics.items()])


def test_the_rules_are_the_pre_registered_ones():
    assert RULES == {"H1'": {"pair": ("C2p", "C3"), "auc_min": 0.05, "skill_alternative": True},
                     "H8": {"pair": ("C3q", "C3"), "auc_min": 0.03, "skill_alternative": False}}


def test_verdict_supported_on_auc_gain_with_interval_above_zero():
    deltas = _deltas({"C3 minus C2p": {"auc": (0.08, 0.02, 0.14), "skill": (0.5, -1.0, 2.0)},
                      "C3 minus C3q": {"auc": (0.08, 0.02, 0.14), "skill": (0.5, -1.0, 2.0)}})
    table = verdicts(deltas, n_pitches=100, n_no_offer=28, metrics=FINE)
    assert list(table.columns) == VERDICT_COLUMNS and list(table.hypothesis) == ["H1'", "H8"]
    row = table.set_index("hypothesis").loc["H1'"]
    assert row.comparison == "C3 minus C2p" and row.verdict == SUPPORTED == "supported"
    assert (row.auc_delta, row.auc_low, row.auc_high) == (0.08, 0.02, 0.14)
    assert (row.skill_delta, row.skill_low, row.skill_high) == (0.5, -1.0, 2.0)
    assert table.set_index("hypothesis").loc["H8", "verdict"] == SUPPORTED
    # a gain below the threshold, or an interval that reaches zero, is not enough
    small = verdicts(_deltas({"C3 minus C2p": {"auc": (0.04, 0.01, 0.07)}, "C3 minus C3q": {"auc": (0.02, 0.01, 0.03)}}),
                     100, 28, FINE)
    assert set(small.verdict) == {NOT_DISTINGUISHABLE}
    wide = verdicts(_deltas({"C3 minus C2p": {"auc": (0.08, -0.01, 0.17)}, "C3 minus C3q": {"auc": (0.08, 0.0, 0.16)}}),
                    100, 28, FINE)
    assert set(wide.verdict) == {NOT_DISTINGUISHABLE}


def test_verdict_supported_on_skill_alone_for_h1_only():
    numbers = {"auc": (0.02, -0.01, 0.05), "skill": (3.0, 0.5, 5.5)}
    table = verdicts(_deltas({"C3 minus C2p": numbers, "C3 minus C3q": numbers}), n_pitches=100, n_no_offer=28, metrics=FINE)
    by = table.set_index("hypothesis")
    assert by.verdict["H1'"] == SUPPORTED                          # a skill gain whose interval excludes zero
    assert by.verdict["H8"] == NOT_DISTINGUISHABLE == "not distinguishable from zero"   # H8 reads the AUC only
    # a skill loss is not a gain, however clear
    loss = verdicts(_deltas({"C3 minus C2p": {"auc": numbers["auc"], "skill": (-3.0, -5.5, -0.5)}}), 100, 28, FINE)
    assert loss.set_index("hypothesis").verdict["H1'"] == NOT_DISTINGUISHABLE


def test_verdict_not_distinguishable_prints_the_power_sentence():
    deltas = _deltas({"C3 minus C2p": {"auc": (0.01, -0.06, 0.08), "skill": (0.2, -2.0, 2.4)},
                      "C3 minus C3q": {"auc": (0.0, -0.08, 0.08), "skill": (0.1, -1.0, 1.2)},
                      "C2 minus C1": {"auc": (0.1, 0.04, 0.16), "log_loss": (-0.02, -0.05, 0.01)}})
    table = verdicts(deltas, n_pitches=100, n_no_offer=28, metrics=FINE)
    assert set(table.verdict) == {NOT_DISTINGUISHABLE}
    resolution = np.mean([0.14, 0.16, 0.12]) / 2                 # half the mean width of every AUC interval
    sentence = (f"With 28 pitches without an offer among 100, AUC differences smaller than about ±{resolution:.2f} "
                "cannot be told from zero; the rule's thresholds (+0.05, +0.03) are below that resolution, so a pass "
                "rests on the interval excluding zero.")
    assert (table.power == sentence).all() and "±0.07" in sentence
    # a comparison the deltas do not hold is reported, not raised, and is never supported
    missing = verdicts(deltas[deltas.comparison != "C3 minus C3q"], 100, 28, FINE).set_index("hypothesis")
    assert missing.verdict["H8"] == NOT_DISTINGUISHABLE and np.isnan(missing.auc_delta["H8"])
    assert pytest.approx(missing.auc_delta["H1'"]) == 0.01


def test_auc_meaningful_is_diagnostic_only_and_never_withholds_the_auc_route():
    """Ruling I3 (plan B final review): under the base-rate-adjusted score the spread rule no longer withholds the
    AUC route; `auc_meaningful` stays a diagnostic column of the metrics."""
    deltas = _deltas({"C3 minus C2p": {"auc": (0.08, 0.02, 0.14), "skill": (0.5, -1.0, 2.0)},
                      "C3 minus C3q": {"auc": (0.08, 0.02, 0.14), "skill": (0.5, -1.0, 2.0)}})
    flat = verdicts(deltas, 100, 28, metrics=_metrics(C2p=False, C3q=False, C3=False)).set_index("hypothesis")
    assert set(flat.auc_route) == {AUC_USED} == {"used"} and set(flat.verdict) == {SUPPORTED}
    fine = verdicts(deltas, 100, 28, metrics=_metrics(C2p=True, C3q=True, C3=True))
    assert fine.equals(verdicts(deltas, 100, 28, metrics=_metrics(C2p=False, C3q=False, C3=False)))
    assert not any("not meaningful" in route for route in flat.auc_route)


def test_the_auc_route_is_withheld_when_a_models_auc_interval_lies_below_one_half():
    """Plan B task 7 review, round 2: the backstop for an anti-ranking AUC (interval wholly below 0.5)."""
    deltas = _deltas({"C3 minus C2p": {"auc": (0.56, 0.34, 0.82), "skill": (0.5, -1.0, 2.0)},
                      "C3 minus C3q": {"auc": (0.56, 0.34, 0.82), "skill": (0.5, -1.0, 2.0)}})
    metrics = _metrics(C2p=True, C3q=True, C3=True).assign(auc_high=[0.48, 0.30, 0.82])
    by = verdicts(deltas, 23, 7, metrics=metrics).set_index("hypothesis")
    assert by.auc_route["H1'"] == "withheld: AUC interval below 0.5 for C2p" and by.verdict["H1'"] == NOT_DISTINGUISHABLE
    assert by.auc_route["H8"] == "withheld: AUC interval below 0.5 for C3q" and by.verdict["H8"] == NOT_DISTINGUISHABLE
    # an interval that reaches 0.5, or a missing one, leaves the route open, whatever auc_meaningful says
    open_ = verdicts(deltas, 23, 7, metrics=metrics.assign(auc_high=[0.5, np.nan, 0.82],
                                                            auc_meaningful=False)).set_index("hypothesis")
    assert set(open_.auc_route) == {AUC_USED} and set(open_.verdict) == {SUPPORTED}
    both = metrics.assign(auc_meaningful=[False, True, True])
    assert verdicts(deltas, 23, 7, metrics=both).auc_route.iloc[0] == "withheld: AUC interval below 0.5 for C2p"


def test_the_auc_threshold_is_met_within_a_tolerance_and_a_missing_interval_is_worded():
    edge = _deltas({"C3 minus C2p": {"auc": (0.05 - 1e-13, 0.01, 0.09)}, "C3 minus C3q": {"auc": (0.03 - 1e-9, 0.01, 0.05)}})
    by = verdicts(edge, 100, 28, FINE).set_index("hypothesis").verdict
    assert by["H1'"] == SUPPORTED and by["H8"] == NOT_DISTINGUISHABLE
    nothing = verdicts(_deltas({"C3 minus C2p": {"auc": (np.nan, np.nan, np.nan)}}), 100, 28, FINE)
    assert "±nan" not in nothing.power.iloc[0] and "the interval widths could not be estimated" in nothing.power.iloc[0]


def test_the_metrics_frame_is_required():
    deltas = _deltas({"C3 minus C2p": {"auc": (0.08, 0.02, 0.14)}})
    with pytest.raises(TypeError):
        verdicts(deltas, 100, 28)                                   # noqa: the guard cannot be left out silently
    with pytest.raises(ValueError, match="metrics"):
        verdicts(deltas, 100, 28, None)


def test_the_power_sentence_sets_the_resolution_against_the_thresholds():
    """Fix round 1 of plan B task 9: the sentence says whether the rule's thresholds are below what can be resolved."""
    def power(width):
        return verdicts(_deltas({"C3 minus C2p": {"auc": (0.0, -width / 2, width / 2)}}), 100, 27, FINE).power.iloc[0]
    assert power(0.20).endswith("about ±0.10 cannot be told from zero; the rule's thresholds (+0.05, +0.03) are below "
                                "that resolution, so a pass rests on the interval excluding zero.")
    assert power(0.08).endswith("the thresholds +0.03 are below that resolution (a pass on them rests on the interval "
                                "excluding zero) and +0.05 at or above it.")
    assert power(0.04).endswith("the rule's thresholds (+0.05, +0.03) are at or above that resolution.")
    assert "what the data can see" not in power(0.2)


def test_the_power_sentence_counts_one_pitch_in_the_singular():
    deltas = _deltas({"C3 minus C2p": {"auc": (0.01, -0.06, 0.08)}})
    assert verdicts(deltas, 30, 1, FINE).power.iloc[0].startswith("With 1 pitch without an offer among 30, ")
    assert verdicts(deltas, 30, 2, FINE).power.iloc[0].startswith("With 2 pitches without an offer among 30, ")
