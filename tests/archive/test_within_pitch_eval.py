"""analysis.archive.within_pitch_eval: does a model rank the sharks of one pitch, not only the pitches?"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analysis.panel_bottleneck import DELTA_COLUMNS
from analysis.archive.within_pitch_eval import (
    between_pitch_auc, conditional_logit_check, informative_pitches, within_pitch_auc, within_pitch_deltas,
    within_pitch_table,
)


def table() -> pd.DataFrame:
    """Pitch a ranks its offerer first, pitch b ranks it last, pitch c has only offers (no within-pitch pair)."""
    rows = [("a", "S1", 1, 0.9, 0.5), ("a", "S2", 0, 0.2, 0.5),
            ("b", "S1", 1, 0.3, 0.4), ("b", "S2", 0, 0.6, 0.4),
            ("c", "S1", 1, 0.8, 0.7), ("c", "S2", 1, 0.7, 0.7)]
    frame = pd.DataFrame(rows, columns=["video_id", "investor", "y", "p_M", "p_K"])
    frame["weight"] = 0.5
    return frame


def test_informative_pitches_have_both_outcomes():
    assert informative_pitches(table()) == ["a", "b"]


def test_within_pitch_auc_is_pooled_over_within_pitch_pairs():
    assert within_pitch_auc(table(), "p_M") == (0.5, 2)
    assert within_pitch_auc(table(), "p_M", pitches=["a"]) == (1.0, 1)
    empty = table()[table().video_id == "c"]
    auc, pairs = within_pitch_auc(empty, "p_M")
    assert math.isnan(auc) and pairs == 0


def test_a_constant_within_pitch_column_ranks_nothing():
    out = within_pitch_table(table(), ["K", "M"], n_boot=50, seed=1).set_index("model")
    assert out.loc["K", "within_auc"] == 0.5 and bool(out.loc["K", "constant_within"])
    assert out.loc["K", "note"] == "ranking uninformative"
    assert not bool(out.loc["M", "constant_within"]) and out.loc["M", "note"] == ""
    assert (out.loc["M", "pairs"], out.loc["M", "pitches"]) == (2, 2)


def test_between_pitch_auc_reads_the_pitch_mean():
    # pitch means of p_K: a 0.5, b 0.4, c 0.7; any offer in all three -> one class only: NaN
    assert math.isnan(between_pitch_auc(table(), "p_K"))
    frame = pd.concat([table(), pd.DataFrame([("d", "S1", 0, 0.45, 0.45), ("d", "S2", 0, 0.45, 0.45)],
                                             columns=["video_id", "investor", "y", "p_M", "p_K"])])
    # offer pitches a, b, c (means 0.5, 0.4, 0.7) against no-offer pitch d (0.45): 2 of 3 above
    assert between_pitch_auc(frame, "p_K") == pytest.approx(2 / 3)


def test_the_bootstrap_interval_holds_the_point_and_is_reproducible():
    rng = np.random.default_rng(0)
    rows = []
    for v in range(30):
        for s in range(4):
            y = int(s < 2)
            rows.append((f"v{v}", f"S{s}", y, 0.5 + 0.2 * y + rng.normal(0, 0.2)))
    frame = pd.DataFrame(rows, columns=["video_id", "investor", "y", "p_M"])
    first = within_pitch_table(frame, ["M"], n_boot=200, seed=3).iloc[0]
    again = within_pitch_table(frame, ["M"], n_boot=200, seed=3).iloc[0]
    assert first.within_low <= first.within_auc <= first.within_high
    assert (first.within_low, first.within_high) == (again.within_low, again.within_high)


def test_a_column_against_itself_has_no_delta():
    out = within_pitch_deltas(table(), [("M", "M")], n_boot=50, seed=1)
    assert list(out.columns) == DELTA_COLUMNS
    r = out.iloc[0]
    assert (r.comparison, r.metric, r.delta, r.ci_low, r.ci_high) == ("M minus M", "within_auc", 0.0, 0.0, 0.0)


def test_conditional_logit_recovers_a_planted_sign():
    rng = np.random.default_rng(7)
    rows = []
    for v in range(60):
        x = rng.normal(size=4)
        noise = rng.normal(size=4)
        offer = np.zeros(4, dtype=int)
        offer[np.argmax(1.5 * x + rng.gumbel(size=4))] = 1
        for s in range(4):
            rows.append((f"v{v}", f"S{s}", offer[s], x[s], noise[s]))
    frame = pd.DataFrame(rows, columns=["video_id", "investor", "ever_offer", "x", "noise"])
    out = conditional_logit_check(frame, "ever_offer", ["x", "noise"])
    assert out["beta"]["x"] > 0.5 and abs(out["beta"]["noise"]) < out["beta"]["x"]
    assert out["lr_p"] < 0.001 and out["pitches"] == 60
