"""analysis.answer_level: H9, whether a more prepared answer changes what the panel does next."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import analysis.answer_level as al
from analysis.answer_level import (
    ANSWER_TABLE_COLUMNS, BINARY_OUTCOMES, BINARY_PREDICTORS, CONTINUOUS_OUTCOMES, H9_COLUMNS, MODEL_COLUMNS, MODELS,
    OUTCOMES, POPULATIONS, SENSITIVITY_OUTCOMES, answer_table, conditional_logit, fixed_effects, h9, h9_sensitivity,
    h9_summary,
)
from analysis.panel_dataset import build_data, pitch_cut
from tests.panel_analysis_fixtures import talk_record

FULL = MODELS["full"]


def _talk(turns, **kw):
    return build_data([{"video_id": "vid", "record": talk_record(turns, **kw), "source": "primary"}])


def planted_table(n_pitches=24, n_answers=20, subs_effect=-1.0, seed=7) -> pd.DataFrame:
    """An answer table in which, within each pitch, a challenge grows less likely as substance rises
    (log-odds `subs_effect` per substance point) and further questions fall with substance, on top of a
    pitch-level intercept that the pitch strata absorb. The other predictors have no effect."""
    rng, frames = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    rows = []
    for p in range(n_pitches):
        intercept = rng.normal(0.0, 1.0)
        level = rng.normal(3.0, 1.0)
        for a in range(n_answers):
            subs = float(rng.integers(1, 6))
            challenge = rng.random() < 1.0 / (1.0 + math.exp(-(intercept + subs_effect * (subs - 3.0))))
            rows.append({
                "video_id": f"p{p:02d}", "answer_turn": a + 2, "question_turn": a + 1,
                "hard_question": bool(rng.random() < 0.5),
                "resp": 1.0, "evid": float(rng.random() < 0.5), "uncert": float(rng.random() < 0.3),
                "subs": subs, "coh": float(rng.integers(1, 6)), "ctx": float(rng.integers(1, 6)),
                "repair_before_next": bool(rng.random() < 0.2),
                "next_accept": float(rng.random() < 0.3), "next_challenge": float(challenge),
                "next_further_question": float(rng.random() < 0.4),
                "next_question_sev": float(rng.integers(1, 4)) if rng.random() < 0.6 else np.nan,
                "next_question_same_exchange": float(rng.random() < 0.5),
                "further_questions_in_exchange": int(max(0, round(level - 0.8 * (subs - 3.0) + rng.normal(0, 0.5)))),
                # the plan B2 columns come from their own stream, so the planted draws above are unchanged
                "q_promotion": float(frames.random() < 0.6),
                "next_question_same_frame": float(frames.random() < 0.5),
                "further_questions_same_frame": int(frames.integers(0, 3))})
    return pd.DataFrame(rows)


# ---- the answer table --------------------------------------------------------------------------------

def test_answer_table_is_cut_at_the_pitch_cut():
    turns = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4),      # 3
             ("F1", "repair", 1, (), None),                                   # 4: a repair before the reaction
             ("S2", "challenges", 1, (), None),                               # 5: the panel challenges 3
             ("S1", "question", 1, (), 3), ("F1", "answer", 1, (6,), 2),      # 7
             ("S2", "question", 1, (), 1), ("F1", "answer", 1, (8,), 3),      # 9: S2 asks again after 7
             ("S1", "accepts", 1, (), None),                                  # 10: the first decision is here
             ("S2", "question", 1, (), 2), ("F1", "answer", 1, (11,), 3)]     # 12: after the cut
    data = _talk(turns, decisions=(("S1", "exit", 10),))
    cut = pitch_cut(data)
    assert int(cut.cut_pos.iloc[0]) == 9
    table = answer_table(data, cut)
    assert list(table.columns) == list(ANSWER_TABLE_COLUMNS)
    assert table.answer_turn.tolist() == [3, 7, 9]                           # answer 12 lies after the cut
    rows = {int(r.answer_turn): r for r in table.itertuples()}
    assert (rows[3].next_accept, rows[3].next_challenge, rows[3].next_further_question) == (0.0, 1.0, 0.0)
    assert (rows[7].next_accept, rows[7].next_challenge, rows[7].next_further_question) == (0.0, 0.0, 1.0)
    late = rows[9]                                                            # its reaction is at the cut
    assert all(math.isnan(v) for v in (late.next_accept, late.next_challenge, late.next_further_question,
                                       late.next_question_sev, late.next_question_same_exchange))
    assert math.isnan(rows[3].next_question_sev) and rows[7].next_question_sev == 1.0
    assert [bool(r.repair_before_next) for r in table.itertuples()] == [True, False, False]
    assert table.further_questions_in_exchange.tolist() == [2, 1, 0]          # question 11 is after the cut
    assert table.next_question_same_exchange.tolist()[:2] == [1.0, 1.0]
    assert table.hard_question.tolist() == [True, True, False]
    assert (rows[3].subs, rows[3].evid, rows[3].uncert) == (4.0, 1.0, 0.0)


def test_answer_table_without_rows_has_the_columns():
    data = _talk([("S1", "accepts", 1, (), None)])
    table = answer_table(data, pitch_cut(data))
    assert len(table) == 0 and list(table.columns) == list(ANSWER_TABLE_COLUMNS)


# ---- the models -------------------------------------------------------------------------------------

def test_conditional_logit_recovers_a_planted_within_pitch_effect():
    table = planted_table()
    out = conditional_logit(table, "next_challenge", FULL, n_boot=99, seed=42)
    assert list(out.columns) == list(MODEL_COLUMNS)
    assert out.predictor.tolist() == list(FULL) and set(out.outcome) == {"next_challenge"}
    subs = out.set_index("predictor").loc["subs"]
    assert subs.coef < 0 and subs.high < 0 and subs.low < subs.coef < subs.high
    assert subs.n_failed < 10 and subs.n_pitches_informative == 24 and subs.n_answers == 480


def test_fixed_effects_recover_a_planted_within_pitch_effect():
    out = fixed_effects(planted_table(), "further_questions_in_exchange", FULL, n_boot=99, seed=42)
    assert list(out.columns) == list(MODEL_COLUMNS)
    subs = out.set_index("predictor").loc["subs"]
    assert subs.coef < 0 and subs.high < 0 and subs.n_failed == 0
    for name in ("evid", "coh", "uncert"):                       # no planted effect: the interval holds zero
        row = out.set_index("predictor").loc[name]
        assert row.low < 0 < row.high, name


def test_single_answer_pitches_are_dropped_and_counted():
    table = planted_table(n_pitches=20)
    single = planted_table(n_pitches=3, n_answers=1, seed=11).assign(video_id=lambda t: "single_" + t.video_id)
    flat = planted_table(n_pitches=2, seed=12).assign(video_id=lambda t: "flat_" + t.video_id, next_challenge=0.0)
    both = pd.concat([table, single, flat], ignore_index=True)
    out = conditional_logit(both, "next_challenge", FULL, n_boot=19, seed=42).set_index("predictor")
    assert (out.n_pitches_informative == 20).all() and (out.n_pitches_dropped == 5).all()
    assert (out.n_answers == 400).all()                          # only the rows of informative pitches count
    alone = conditional_logit(table, "next_challenge", FULL, n_boot=19, seed=42).set_index("predictor")
    assert np.allclose(out.coef, alone.coef)                    # the dropped strata carry no information
    fe = fixed_effects(both, "next_challenge", FULL, n_boot=19, seed=42)
    assert (fe.n_pitches_informative == 22).all() and (fe.n_pitches_dropped == 3).all()   # flat pitches inform OLS
    assert (fe.n_answers == 440).all()


def test_rows_missing_the_outcome_or_a_predictor_are_left_out_of_that_model_only():
    table = planted_table()
    table.loc[:9, "next_challenge"] = np.nan
    table.loc[10:14, "coh"] = np.nan
    out = conditional_logit(table, "next_challenge", FULL, n_boot=0, seed=42)
    assert (out.n_answers == 465).all() and out.low.isna().all() and out.coef.notna().all()


def test_a_predictor_without_within_pitch_variation_has_no_coefficient():
    table = planted_table().assign(uncert=0.0)
    out = conditional_logit(table, "next_challenge", FULL, n_boot=19, seed=42).set_index("predictor")
    assert math.isnan(out.loc["uncert", "coef"]) and math.isnan(out.loc["uncert", "low"])
    assert out.loc["subs", "coef"] < 0


def test_a_duplicated_pitch_in_a_bootstrap_draw_is_its_own_stratum():
    groups = np.array(["a", "a", "b", "b", "b"])
    rows, strata = al._draw(groups, np.array(["a", "b"]), np.array([0, 0]))
    assert rows.tolist() == [0, 1, 0, 1] and strata.tolist() == [0, 0, 1, 1]


def test_too_few_converged_draws_give_no_interval(monkeypatch):
    real = al._fit_clogit
    calls = {"n": 0}

    def flaky(y, x, groups, start=None):
        calls["n"] += 1
        return real(y, x, groups, start) if calls["n"] == 1 or calls["n"] % 5 else None   # 1 in 5 draws fails
    monkeypatch.setattr(al, "_fit_clogit", flaky)
    out = conditional_logit(planted_table(), "next_challenge", FULL, n_boot=20, seed=42)
    assert (out.n_failed == 4).all() and out.low.isna().all() and out.coef.notna().all()


# ---- H9 and its summary ---------------------------------------------------------------------------------

def test_h9_runs_every_outcome_for_all_and_hard_questions(monkeypatch):
    table = planted_table(n_pitches=12, n_answers=16)
    monkeypatch.setattr(al, "answer_table", lambda data, cut: table)
    result = h9(None, None, n_boot=9, seed=42)
    assert set(result) == {"all", "hard"}
    assert set(MODELS) == {"support", "substance", "full", "responsiveness"}
    assert MODELS["responsiveness"] == ("uncert", "repair_before_next")
    assert all("repair_before_next" not in MODELS[m] for m in ("support", "substance", "full"))
    for frame in result.values():
        assert list(frame.columns) == list(H9_COLUMNS)
        assert frame.model.unique().tolist() == list(MODELS)
        for model, predictors in MODELS.items():
            part = frame[frame.model == model]
            # the two same-frame outcomes also enter the answered question's frame, a covariate
            assert len(part) == 6 * len(predictors) + 2 and set(part.predictor) == set(predictors) | {"q_promotion"}
            assert set(part[part.method == "conditional_logit"].outcome) == set(BINARY_OUTCOMES)
            assert set(part[part.method == "fixed_effects"].outcome) == set(CONTINUOUS_OUTCOMES)
            assert set(part.population) == {POPULATIONS[model]}
    assert result["hard"].n_answers.max() < result["all"].n_answers.max()


def test_h9_outcomes_are_the_frame_outcomes_and_the_exchange_outcomes_are_a_sensitivity(monkeypatch):
    assert BINARY_OUTCOMES == ("next_accept", "next_challenge", "next_further_question", "next_question_same_frame")
    assert CONTINUOUS_OUTCOMES == ("next_question_sev", "further_questions_same_frame")
    assert OUTCOMES == (*BINARY_OUTCOMES, *CONTINUOUS_OUTCOMES)
    assert SENSITIVITY_OUTCOMES == ("next_question_same_exchange", "further_questions_in_exchange")
    assert not set(SENSITIVITY_OUTCOMES) & set(OUTCOMES)
    for name in ("q_promotion", "next_question_same_frame", "further_questions_same_frame",
                 "next_question_same_concept", *SENSITIVITY_OUTCOMES):
        assert name in ANSWER_TABLE_COLUMNS, name
    table = planted_table(n_pitches=12, n_answers=16)
    monkeypatch.setattr(al, "answer_table", lambda data, cut: table)
    main = h9(None, None, n_boot=9, seed=42)
    for frame in main.values():
        assert set(frame.outcome) == set(OUTCOMES)
    sensitivity = h9_sensitivity(None, None, n_boot=9, seed=42, workers=1)
    assert set(sensitivity) == {"all", "hard"}
    for name, frame in sensitivity.items():
        assert list(frame.columns) == list(H9_COLUMNS), name
        assert frame.model.unique().tolist() == list(MODELS)
        assert set(frame.outcome) == set(SENSITIVITY_OUTCOMES)
        methods = dict(zip(frame.outcome, frame.method))          # binary first, continuous second
        assert methods == {"next_question_same_exchange": "conditional_logit",
                           "further_questions_in_exchange": "fixed_effects"}
        assert "q_promotion" not in set(frame.predictor)          # the covariate belongs to the frame outcomes
        for model, predictors in MODELS.items():
            assert len(frame[frame.model == model]) == 2 * len(predictors)
    # the same model, outcome and draws give the same numbers whichever function runs it
    alone = conditional_logit(table, "next_question_same_exchange", MODELS["support"], n_boot=9, seed=42)
    mine = sensitivity["all"][(sensitivity["all"].model == "support")
                              & (sensitivity["all"].outcome == "next_question_same_exchange")]
    np.testing.assert_allclose(mine.coef.to_numpy(), alone.coef.to_numpy())


def test_q_promotion_enters_only_the_same_frame_models(monkeypatch):
    assert "q_promotion" in BINARY_PREDICTORS                        # entered as 0/1, never standardised
    assert all("q_promotion" not in predictors for predictors in MODELS.values())
    table = planted_table(n_pitches=12, n_answers=16)
    monkeypatch.setattr(al, "answer_table", lambda data, cut: table)
    result = h9(None, None, n_boot=9, seed=42)
    for frame in result.values():
        with_it = frame[frame.predictor == "q_promotion"]
        assert set(with_it.outcome) == {"next_question_same_frame", "further_questions_same_frame"}
        assert set(with_it.model) == set(MODELS)                     # every nested model
        for (model, outcome), part in frame.groupby(["model", "outcome"]):
            expected = MODELS[model] + (("q_promotion",) if outcome in ("next_question_same_frame",
                                                                          "further_questions_same_frame") else ())
            assert tuple(part.predictor) == expected, (model, outcome)
    # the covariate is not a predictor of interest: the summary states no direction for it
    summary = h9_summary(result)
    assert "q_promotion" not in set(summary.predictor)
    assert len(summary) == len(result["all"]) - (result["all"].predictor == "q_promotion").sum()
    # the answered question's frame: 1 promotion, 0 prevention, NaN unclassified (deal_terms) or NEUTRAL (NONE)
    turns = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "question", 1, (), 2),
             ("F1", "answer", 1, (4,), 4), ("S2", "question", 1, (), 2), ("F1", "answer", 1, (6,), 4),
             ("S2", "question", 1, (), 2), ("F1", "answer", 1, (8,), 4), ("S1", "decides", None, (), None)]
    record = talk_record(turns, decisions=(("S1", "exit", 10),))
    concepts = {2: "sales", 4: "execution", 6: "deal_terms", 8: "NONE"}
    for t in record["turns"]:
        for c in t["codes"]:
            if c["code"] == "PR-CONCEPT" and t["turn_id"] in concepts:
                c["value"] = concepts[t["turn_id"]]
    data = build_data([{"video_id": "vid", "record": record, "source": "primary"}])
    table = answer_table(data, pitch_cut(data)).set_index("question_turn")
    assert table.q_promotion[2] == 1.0 and table.q_promotion[4] == 0.0
    assert math.isnan(table.q_promotion[6]) and math.isnan(table.q_promotion[8])
    assert table.next_question_same_frame[2] == 0.0 and table.further_questions_same_frame[2] == 0


def _result(rows_all, rows_hard):
    def frame(rows):
        return pd.DataFrame([{"model": "prepared", "method": "conditional_logit", "outcome": o, "predictor": p,
                              "coef": (lo + hi) / 2, "low": lo, "high": hi} for o, p, lo, hi in rows])
    return {"all": frame(rows_all), "hard": frame(rows_hard)}


def test_direction_claimed_only_when_hard_and_all_agree():
    nan = np.nan
    result = _result(
        [("next_accept", "subs", 0.1, 0.5), ("next_accept", "evid", 0.1, 0.5), ("next_challenge", "subs", -0.9, -0.2),
         ("next_challenge", "evid", 0.2, 0.4), ("next_challenge", "coh", nan, nan)],
        [("next_accept", "subs", 0.2, 0.8), ("next_accept", "evid", -0.1, 0.5), ("next_challenge", "subs", -0.5, -0.1),
         ("next_challenge", "evid", -0.4, -0.2), ("next_challenge", "coh", 0.1, 0.2)])
    summary = h9_summary(result)
    assert list(summary.columns) == ["model", "outcome", "predictor", "direction", "hard_and_all_agree"]
    summary = summary.set_index(["outcome", "predictor"])
    assert summary.loc[("next_accept", "subs"), "direction"] == "positive"
    assert summary.loc[("next_challenge", "subs"), "direction"] == "negative"
    for key in (("next_accept", "evid"), ("next_challenge", "evid"), ("next_challenge", "coh")):
        assert summary.loc[key, "direction"] == "not distinguishable", key
        assert not summary.loc[key, "hard_and_all_agree"], key
    assert summary.hard_and_all_agree.tolist() == [True, False, True, False, False]


# ---- fix round 1 ----------------------------------------------------------------------------------------

def _separating_table() -> pd.DataFrame:
    """The reviewer's case: admitted uncertainty in 30 of 100 pitches, rare, and never followed by a
    challenge inside the informative strata; evid lowers the challenge odds; uncertainty has no effect."""
    rng = np.random.default_rng(3)
    rows = []
    for p in range(100):
        a0 = rng.normal(-2.4, 0.7)
        for _ in range(15):
            unc = float(p < 30 and rng.random() < 0.10)
            evid = float(rng.random() < .58)
            challenge = rng.random() < 1 / (1 + np.exp(-(a0 - 0.8 * evid)))
            rows.append(dict(video_id=f"p{p:03d}", next_challenge=float(challenge), evid=evid, uncert=unc,
                             repair_before_next=bool(rng.random() < 0.03), subs=float(rng.integers(1, 6))))
    return pd.DataFrame(rows)


def test_a_separating_predictor_is_flagged_and_never_given_a_direction():
    table = _separating_table()
    informative = table[table.groupby("video_id").next_challenge.transform("nunique") > 1]
    assert informative.uncert.sum() > 0
    assert ((informative.uncert == 1) & (informative.next_challenge == 1)).sum() == 0     # the empty cell
    out = conditional_logit(table, "next_challenge", ("evid", "uncert"), n_boot=19, seed=42)
    row = out.set_index("predictor")
    assert bool(row.loc["uncert", "separated"]) and not bool(row.loc["evid", "separated"])
    assert all(math.isnan(row.loc["uncert", c]) for c in ("coef", "low", "high"))
    alone = conditional_logit(table, "next_challenge", ("evid",), n_boot=0, seed=42)
    assert np.isclose(row.loc["evid", "coef"], alone.coef.iloc[0])     # refitted without the separated one
    summary = h9_summary({"all": out.assign(model="support"), "hard": out.assign(model="support")})
    assert summary.set_index("predictor").loc["uncert", "direction"] == "not distinguishable"
    fe = fixed_effects(table, "next_challenge", ("evid", "uncert"), n_boot=0, seed=42)
    assert not fe.separated.any()                                     # separation is a logit matter


def test_a_large_bootstrap_coefficient_is_a_failed_draw_for_that_predictor_only(monkeypatch):
    real = al._fit_clogit
    calls = {"n": 0}

    def wild(y, x, groups, start=None):
        calls["n"] += 1
        params = real(y, x, groups, start)
        if calls["n"] > 1 and calls["n"] % 4 == 0 and params is not None:
            params = params.copy()
            params[0] = 12.0                                         # the first predictor, subs, runs away
        return params
    monkeypatch.setattr(al, "_fit_clogit", wild)
    out = conditional_logit(planted_table(), "next_challenge", ("subs", "evid"), n_boot=20, seed=42)
    row = out.set_index("predictor")
    assert row.loc["subs", "n_failed"] == 5 and math.isnan(row.loc["subs", "low"])
    assert row.loc["evid", "n_failed"] == 0 and row.loc["evid", "low"] == row.loc["evid", "low"]


def test_a_main_fit_coefficient_beyond_the_bound_takes_the_separated_path(monkeypatch):
    """Plan B task 4 carry and Task 9 pre-run: MAX_ABS_LOGIT applies to the fit on the data too; a predictor whose
    coefficient is beyond it is reported separated (no coefficient, interval or direction), the model is refitted
    without it and the other predictors keep their bootstrap intervals."""
    real = al._fit_clogit
    calls = {"n": 0}

    def runaway_main(y, x, groups, start=None):
        calls["n"] += 1
        params = real(y, x, groups, start)
        if calls["n"] == 1 and params is not None:                 # the first call is the fit on the data
            params = params.copy()
            params[0] = -(al.MAX_ABS_LOGIT + 1.0)                    # subs runs away
        return params
    monkeypatch.setattr(al, "_fit_clogit", runaway_main)
    out = conditional_logit(planted_table(), "next_challenge", ("subs", "evid"), n_boot=20, seed=42)
    row = out.set_index("predictor")
    assert not hasattr(al, "BEYOND_BOUND") and set(out.estimable) == {al.ESTIMABLE}
    assert bool(row.loc["subs", "separated"]) and not bool(row.loc["evid", "separated"])
    assert math.isnan(row.loc["subs", "coef"]) and math.isnan(row.loc["subs", "low"])
    # the refit without subs equals a model that never had it, and evid keeps its interval
    monkeypatch.setattr(al, "_fit_clogit", real)
    alone = conditional_logit(planted_table(), "next_challenge", ("evid",), n_boot=20, seed=42).set_index("predictor")
    for column in ("coef", "low", "high"):
        assert row.loc["evid", column] == pytest.approx(alone.loc["evid", column])
    assert not out.main_failed.any()
    summary = h9_summary({name: out.assign(model="m") for name in ("all", "hard")}).set_index("predictor")
    assert summary.loc["subs", "direction"] == "not distinguishable"
    # within the bound the fit is kept and bootstrapped as before
    fine = conditional_logit(planted_table(), "next_challenge", ("subs", "evid"), n_boot=20, seed=42)
    assert set(fine.estimable) == {al.ESTIMABLE} and fine.low.notna().all() and not fine.separated.any()
    # with one informative pitch, too few pitches stays the reason, and a runaway coefficient is still no estimate
    calls["n"] = 0
    monkeypatch.setattr(al, "_fit_clogit", runaway_main)
    one = conditional_logit(planted_table()[lambda t: t.video_id == "p00"], "next_challenge", ("subs", "evid"),
                            n_boot=20, seed=42).set_index("predictor")
    assert set(one.estimable) == {al.TOO_FEW_PITCHES} and math.isnan(one.loc["subs", "coef"])
    assert bool(one.loc["subs", "separated"]) and one[["low", "high"]].isna().all().all()


def test_a_rare_predictor_leaves_the_other_intervals_standing():
    rng = np.random.default_rng(5)
    rows = []
    for p in range(50):
        a0 = rng.normal(-1.0, 0.7)
        for a in range(15):
            evid = float(rng.random() < .58)
            challenge = rng.random() < 1 / (1 + np.exp(-(a0 - 0.8 * evid)))
            repair = p == 0 and a < 2                    # one pitch, one repair challenged and one not
            rows.append(dict(video_id=f"p{p:03d}", next_challenge=float(challenge if not repair else a == 0),
                             evid=evid, uncert=float(rng.random() < .04), repair_before_next=repair,
                             subs=float(rng.integers(1, 6))))
    table = pd.DataFrame(rows)
    out = conditional_logit(table, "next_challenge", ("subs", "evid", "uncert", "repair_before_next"),
                            n_boot=39, seed=42).set_index("predictor")
    assert out.loc["evid", "high"] < 0 and out.loc["evid", "n_failed"] <= 3          # the planted effect
    assert out.loc["subs", "low"] < 0 < out.loc["subs", "high"]
    assert not out.separated.any()
    assert math.isnan(out.loc["repair_before_next", "low"])                          # in one pitch only
    assert out.loc["repair_before_next", "n_failed"] > out.loc["evid", "n_failed"]


def test_a_decision_at_the_cut_leaves_the_next_move_unknown():
    turns = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "decides", 1, (), None),
             ("S2", "question", 1, (), 2), ("F1", "answer", 1, (5,), 4)]
    data = _talk(turns, decisions=(("S1", "exit", 4),))
    cut = pitch_cut(data)
    assert int(cut.cut_pos.iloc[0]) == 3
    table = answer_table(data, cut)
    assert table.answer_turn.tolist() == [3]
    r = table.iloc[0]
    assert all(math.isnan(v) for v in (r.next_accept, r.next_challenge, r.next_further_question, r.next_question_sev))
    # a decision before the cut that ends the exchange still reads as no reaction (0.0 on all three)
    data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "decides", 1, (), None)])
    r = answer_table(data, pitch_cut(data)).iloc[0]
    assert (r.next_accept, r.next_challenge, r.next_further_question) == (0.0, 0.0, 0.0)


def test_models_read_a_copy_on_write_frame():
    out = fixed_effects(planted_table(), "next_question_sev", ("subs", "evid"), n_boot=0)
    assert out.coef.notna().all() and out.low.isna().all()
    out = conditional_logit(planted_table(), "next_challenge", ("subs", "evid"), n_boot=0)
    assert out.coef.notna().all()


def test_main_failed_is_flagged_apart_from_no_usable_predictor(monkeypatch):
    table = planted_table(n_pitches=8, n_answers=10)
    flat = fixed_effects(table.assign(subs=3.0), "further_questions_in_exchange", ("subs",), n_boot=0)
    assert not flat.main_failed.any() and flat.coef.isna().all()
    monkeypatch.setattr(al, "_fit_fixed_effects", lambda y, x, groups, start=None: None)
    failed = fixed_effects(table, "further_questions_in_exchange", ("subs", "evid"), n_boot=5)
    assert failed.main_failed.all() and failed.coef.isna().all() and (failed.n_failed == 0).all()


def test_n_pitches_dropped_counts_pitches_lost_to_missing_values():
    table = planted_table(n_pitches=10, n_answers=10)
    table.loc[table.video_id == "p00", "next_question_sev"] = np.nan
    table.loc[table.video_id == "p01", "coh"] = np.nan
    out = fixed_effects(table, "next_question_sev", FULL, n_boot=0)
    assert (out.n_pitches_informative == 8).all() and (out.n_pitches_dropped == 2).all()


def test_bootstrap_draws_split_across_workers_give_the_same_interval():
    table = planted_table(n_pitches=10, n_answers=12)
    rows = table[table.groupby("video_id").next_challenge.transform("nunique") > 1]
    groups = rows.video_id.to_numpy()
    x = np.array(rows[["subs", "evid"]], dtype=float)
    y = np.array(rows.next_challenge, dtype=float)
    pitches = np.array(sorted(set(groups)), dtype=object)
    matrix = al._draw_matrix(len(pitches), 20, 42)
    assert matrix.shape == (20, len(pitches))
    assert np.array_equal(matrix, al._draw_matrix(len(pitches), 20, 42))
    whole = al._bootstrap(y, x, groups, pitches, matrix, al.CLOGIT)
    split = np.vstack([al._bootstrap(y, x, groups, pitches, matrix[:9], al.CLOGIT),
                       al._bootstrap(y, x, groups, pitches, matrix[9:], al.CLOGIT)])
    np.testing.assert_array_equal(whole, split)
    for a, b in zip(al._interval(whole), al._interval(split)):
        np.testing.assert_array_equal(a, b)


def test_a_predictor_separated_in_the_hard_questions_gets_no_direction():
    frame = pd.DataFrame([{"model": "support", "method": "conditional_logit", "outcome": "next_accept",
                           "predictor": "uncert", "coef": 1.0, "low": 0.5, "high": 1.5, "separated": False}])
    hard = frame.assign(separated=True)
    summary = h9_summary({"all": frame, "hard": hard})
    assert summary.direction.tolist() == ["not distinguishable"] and not summary.hard_and_all_agree.any()
    assert h9_summary({"all": frame, "hard": frame}).direction.tolist() == ["positive"]


def test_h9_with_two_workers_equals_h9_with_one(tmp_path):
    """The command line splits each model's pre-drawn bootstrap matrix over a process pool (--workers): the result
    must be the single-process result, number for number, on a real (synthetic) run."""
    from analysis.panel_dataset import load_run
    from tests.panel_analysis_fixtures import write_run

    data = load_run(write_run(tmp_path, 12, seed=4, planted="own", strength=4.0))
    cut = pitch_cut(data)
    one = h9(data, cut, n_boot=40, seed=42, workers=1)
    two = h9(data, cut, n_boot=40, seed=42, workers=2)
    assert set(one) == set(two) == {"all", "hard"}
    for name in one:
        assert one[name]["low"].notna().any(), name                    # intervals were actually drawn
        pd.testing.assert_frame_equal(one[name], two[name])


def test_the_split_runs_the_blocks_in_a_pool_and_keeps_their_order():
    table = planted_table(n_pitches=8, n_answers=10)
    rows = table[table.groupby("video_id").next_challenge.transform("nunique") > 1]
    groups = rows.video_id.to_numpy()
    x = np.array(rows[["subs", "evid"]], dtype=float)
    y = np.array(rows.next_challenge, dtype=float)
    pitches = np.array(sorted(set(groups)), dtype=object)
    matrix = al._draw_matrix(len(pitches), 11, 3)
    pool = al.worker_pool(3)
    try:
        split = al._bootstrap_split(y, x, groups, pitches, matrix, al.CLOGIT, pool, 3)
    finally:
        pool.shutdown()
    np.testing.assert_array_equal(split, al._bootstrap(y, x, groups, pitches, matrix, al.CLOGIT))
    assert al.worker_pool(1) is None and al.worker_pool(0) is None


def test_one_informative_pitch_gives_no_interval_and_no_direction(monkeypatch):
    """Resampling one pitch redraws it every time: a zero-width interval that must not carry a direction."""
    table = planted_table(n_pitches=12, n_answers=16, subs_effect=-2.0)
    table["hard_question"] = table.video_id == "p00"                 # the hard questions all sit in one pitch
    monkeypatch.setattr(al, "answer_table", lambda data, cut: table)
    result = h9(None, None, n_boot=19, seed=42)
    hard = result["hard"]
    one = hard[hard.n_pitches_informative == 1]
    assert len(one) and set(one.estimable) == {al.TOO_FEW_PITCHES} == {"too few informative pitches"}
    assert one.low.isna().all() and one.high.isna().all()
    assert set(result["all"].estimable) == {al.ESTIMABLE} == {"yes"}
    summary = h9_summary(result)
    assert set(summary.direction) == {"not distinguishable"} and not summary.hard_and_all_agree.any()


@pytest.mark.parametrize("method", [al.CLOGIT, al.FIXED_EFFECTS])
def test_a_rank_deficient_design_gives_no_estimate(method):
    table = planted_table(n_pitches=12, n_answers=16)
    table["ctx"] = table["coh"]                                     # two predictors that are one column
    fit = conditional_logit if method == al.CLOGIT else fixed_effects
    outcome = "next_challenge" if method == al.CLOGIT else "further_questions_in_exchange"
    frame = fit(table, outcome, ("subs", "coh", "ctx"), n_boot=9, seed=1)
    assert set(frame.estimable) == {al.RANK_DEFICIENT} == {"rank-deficient design"}
    assert frame[["coef", "low", "high"]].isna().all().all()
    fine = fit(table, outcome, ("subs", "coh"), n_boot=9, seed=1)
    assert set(fine.estimable) == {"yes"} and fine.low.notna().all()
    rows = {name: fine for name in ("all", "hard")}
    rows["hard"] = frame.iloc[:2].assign(predictor=["subs", "coh"])
    assert set(h9_summary({k: v.assign(model="m") for k, v in rows.items()}).direction) == {"not distinguishable"}


def test_a_rank_deficient_bootstrap_draw_is_a_failed_draw():
    table = planted_table(n_pitches=6, n_answers=12)
    rows = table.copy()
    collinear = rows.video_id == "p00"
    rows.loc[collinear, "coh"] = rows.loc[collinear, "subs"]          # within p00 the two columns are one
    groups = rows.video_id.to_numpy()
    x = np.array(rows[["subs", "coh"]], dtype=float)
    y = np.array(rows.further_questions_in_exchange, dtype=float)
    pitches = np.array(sorted(set(groups)), dtype=object)
    matrix = np.array([[0] * len(pitches), list(range(len(pitches)))])     # p00 only, then every pitch once
    draws = al._bootstrap(y, x, groups, pitches, matrix, al.FIXED_EFFECTS)
    assert np.isnan(draws[0]).all() and np.isfinite(draws[1]).all()
    assert not al.design_rank_ok(x[collinear.to_numpy()], groups[collinear.to_numpy()], al.FIXED_EFFECTS)
    assert al.design_rank_ok(x, groups, al.FIXED_EFFECTS) and al.design_rank_ok(x, groups, al.CLOGIT)
