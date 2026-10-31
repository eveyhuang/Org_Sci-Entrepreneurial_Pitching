"""Heads B0-B5 on synthetic runs with planted structure: the retired shark-level model sets, archived with their
functions in plan B task 3 (analysis.archive.shark_level_models). Moved from tests/test_panel_bottleneck.py with
their expected values. Illustrative data, never empirical."""
import numpy as np
import pandas as pd
import pytest

from analysis import panel_bottleneck as pb
from analysis.panel_bottleneck import (
    BY_OFFER_COLUMNS, CONDITIONAL, DEMONSTRATION, NEAR_CONSTANT_SPREAD, NOTE_COUNT,
    NOTE_DECISION_ORDER, PENALTY_GRID, SENSITIVITY_NOTE, SINGLE_COLUMNS, choose_penalty, editable_features,
    fold_design, fold_detail, metrics_of, penalty_setting, ridge_logistic, skill_over_base,
)
from analysis.archive.shark_level_models import (
    ANSWER_QUALITY_STATS, ARCHIVED_C_FAMILY, MODEL_FAMILY, PRIMARY_FEATURES, PRIMARY_MODELS,
    PRIMARY_SENSITIVITY_FEATURES, PRIMARY_STAGE_A_RATINGS, PRIMARY_STATUS, QUESTIONS_ASKED, REMOVAL_CAVEAT, SENSITIVITY,
    answer_quality_by_offer, answer_quality_comparisons, answer_quality_models, check_models, count_behind_a_mean,
    design_matrix, edited_prediction, evaluate, evaluate_answer_quality, evaluate_question_control, family_columns,
    feature_status, fit_design, fit_head, o1_reaction_comparisons, o1_reaction_models, pooled_vs_attributed,
    prediction_breakdown, predictor_names, primary_models, question_control_comparisons, question_control_models,
    reaction_columns, reaction_comparisons, reaction_models, removal_ranking, shark_level_rows, single_features,
)
from analysis.archive.shark_level_features import (
    ARCHIVE_FEATURES, COMBINED_FEATURES as FEATURES, combined_dictionary, o2_features,
)
from analysis.panel_dataset import load_run
from analysis.panel_features import FAMILIES, FEATURES as PITCH_FEATURES, pitch_features
from analysis.archive.within_pitch_eval import within_pitch_auc
from tests.panel_analysis_fixtures import write_run



# Every planted-structure test runs with the default penalty: chosen inside each training fold by
# cross-validation over its pitches. No test hands a head a penalty that suits the planted effect.
BOOT = 199


def _tables(tmp_path_factory, planted, n=30, seed=1, ranking=False, strength=4.0):
    root = write_run(tmp_path_factory.mktemp(f"run_{planted}_{n}"), n, seed=seed, planted=planted,
                     strength=strength, ranking=ranking)
    data = load_run(root)
    return o2_features(data), pitch_features(data)


@pytest.fixture(scope="module")
def own(tmp_path_factory):
    return _tables(tmp_path_factory, "own")


@pytest.fixture(scope="module")
def panel(tmp_path_factory):
    # what the other investors said (agreed, challenged) drives the offers; their earlier exits and offers
    # are no longer part of this family, so the planted effect is the only thing it can pick up
    return _tables(tmp_path_factory, "panel_reactions", strength=6.0)


@pytest.fixture(scope="module")
def own_result(own):
    return evaluate(own[0], "ever_offer", n_boot=BOOT)


@pytest.fixture(scope="module")
def panel_result(panel):
    return evaluate(panel[0], "ever_offer", n_boot=BOOT)


def _frame(n=12, **columns):
    """A small feature table: `hist_turns` 1..n and whatever else a test sets."""
    base = {"video_id": [f"v{i // 2}" for i in range(n)], "hist_turns": np.arange(1.0, n + 1)}
    return pd.DataFrame({**base, **columns})


# ---- the shark-level reading of the generic machinery -------------------------------------------

def test_the_shark_level_reading_is_put_back_on_leaving_also_after_an_error():
    live = (pb.FEATURES, pb.FAMILIES, pb.KEY_COLUMNS, pb.NOT_PREDICTORS, pb.feature_status)
    assert pb.KEY_COLUMNS == () and pb.NOT_PREDICTORS == () and "own" not in pb.FAMILIES
    with pytest.raises(RuntimeError):
        with shark_level_rows():
            assert pb.FEATURES is FEATURES and pb.KEY_COLUMNS == ("investor",) and "own" in pb.FAMILIES
            with shark_level_rows():                                    # nested: still the shark-level reading
                assert pb.NOT_PREDICTORS == ("investor_label",)
            assert pb.FEATURES is FEATURES and pb.feature_status is feature_status
            raise RuntimeError
    assert (pb.FEATURES, pb.FAMILIES, pb.KEY_COLUMNS, pb.NOT_PREDICTORS, pb.feature_status) == live
    # the live machinery does not know the shark-level columns; their shark-level forms do
    assert "own_questions" not in pb.predictor_names() and "own_questions" in predictor_names()
    with pytest.raises(ValueError, match="unknown feature or family"):
        pb.check_models({"B3": MODEL_FAMILY["B3"]})


# ---- the model family -----------------------------------------------------------------------

def test_model_family_is_cumulative_and_ordered():
    assert list(MODEL_FAMILY) == ["B0", "B1", "B2", "B3", "B4a", "B4", "B5", "B5d"]
    assert MODEL_FAMILY["B0"] == () and MODEL_FAMILY["B1"] == ("control", "exposure")
    names = list(MODEL_FAMILY)
    for earlier, later in zip(names[:-1], names[1:]):
        assert MODEL_FAMILY[later][:len(MODEL_FAMILY[earlier])] == MODEL_FAMILY[earlier]
        assert len(MODEL_FAMILY[later]) == len(MODEL_FAMILY[earlier]) + (2 if earlier == "B0" else 1)
    assert MODEL_FAMILY["B5"][2:] == ("stage_a", "own", "panel_content", "panel_reactions", "rapport_repair")
    # the order in which the other investors decided is its own family, outside B4 and B5
    assert MODEL_FAMILY["B5d"] == MODEL_FAMILY["B5"] + ("decision_history",)
    assert all("decision_history" not in MODEL_FAMILY[m] for m in MODEL_FAMILY if m != "B5d")
    assert list(SENSITIVITY) == ["B5d", "P5d"] and SENSITIVITY["P5d"] == SENSITIVITY["B5d"] and "exits usually precede offers" in SENSITIVITY["B5d"]
    assert "adds the order in which other investors decided" in SENSITIVITY["B5d"]


def test_decision_history_reaches_b5d_only_and_is_reported_as_a_sensitivity(own, own_result):
    o2 = own[0]
    def used(model):
        columns = family_columns(o2, MODEL_FAMILY[model])
        return set(columns["categorical"] + columns["indicator"] + columns["numeric"])
    history = {"oth_exits_before", "oth_offers_before"}
    assert not used("B4") & history and not used("B5") & history and used("B5d") - used("B5") == history
    metrics = own_result["metrics"].set_index("model")
    assert metrics.loc["B5d", "role"] == "sensitivity" and metrics.loc["B5d", "note"] == SENSITIVITY["B5d"]
    assert (metrics.drop("B5d").role == "main").all() and (metrics.drop("B5d").note == "").all()
    assert "B5d minus B5" in set(own_result["deltas"].comparison)
    for fold in own_result["folds"]:
        assert not history & set(fold["models"]["B5"]["columns"]) and not history & set(fold["models"]["B4"]["columns"])


def test_only_named_features_can_enter_a_head(own):
    o2 = own[0]
    columns = family_columns(o2, MODEL_FAMILY["B5"])
    used = columns["categorical"] + columns["indicator"] + columns["numeric"]
    assert set(used) <= set(FEATURES) and "investor_label" not in used
    assert not set(used) & {"video_id", "investor", "source", "cut_pos", "ever_offer", "first_decision_kind",
                            "first_decision_pos"}
    assert all(FEATURES[c]["family"] in MODEL_FAMILY["B5"] for c in used)
    # a free-text column, a label and the outcome itself under another name change nothing
    noisy = o2.assign(summary_text=o2.investor_label + " offers", leaked_outcome=o2.ever_offer,
                      investor_label=o2.ever_offer.astype(str))
    a = evaluate(o2, "ever_offer", models={"B5": MODEL_FAMILY["B5"]}, penalty=10.0, n_boot=20)
    b = evaluate(noisy, "ever_offer", models={"B5": MODEL_FAMILY["B5"]}, penalty=10.0, n_boot=20)
    assert np.allclose(a["predictions"].p_B5, b["predictions"].p_B5)
    assert all("summary" not in c and "leaked" not in c and "label" not in c
               for f in b["folds"] for c in f["models"]["B5"]["columns"])


# ---- evaluation -----------------------------------------------------------------------------

def test_every_model_sees_identical_rows_and_folds(own, own_result):
    o2 = own[0]
    known = o2[o2.ever_offer.notna()]
    pred = own_result["predictions"]
    assert own_result["n_rows"] == len(known) == len(pred) and len(known) < len(o2)   # unknown outcomes left out
    assert own_result["n_groups"] == known.video_id.nunique() == len(own_result["folds"])
    assert own_result["n_events"] == int(known.ever_offer.sum())
    assert list(zip(pred.video_id, pred.investor)) == list(zip(known.video_id, known.investor))
    assert (pred.y.to_numpy() == known.ever_offer.to_numpy()).all()
    columns = [f"p_{m}" for m in MODEL_FAMILY]
    assert pred[columns].notna().all().all() and ((pred[columns] > 0) & (pred[columns] < 1)).all().all()
    assert list(own_result["metrics"].model) == list(MODEL_FAMILY)
    for fold in own_result["folds"]:
        assert set(fold["models"]) == set(MODEL_FAMILY)
        assert fold["held_out"] not in fold["train_groups"]
        assert fold["n_test"] == int((known.video_id == fold["held_out"]).sum())
        assert fold["n_train"] + fold["n_test"] == len(known)          # all rows of a pitch stay together
    assert sorted(f["held_out"] for f in own_result["folds"]) == sorted(known.video_id.unique())
    assert own_result["label"] == "pitch-held-out estimate" and own_result["coding"] == {"primary": 30}


def test_b0_predicts_the_training_base_rate(own, own_result):
    known = own[0][own[0].ever_offer.notna()]
    pred = own_result["predictions"]
    for video_id, rows in pred.groupby("video_id"):
        expected = known.ever_offer[known.video_id != video_id].mean()
        assert np.allclose(rows.p_B0, expected)
    assert own_result["folds"][0]["models"]["B0"]["columns"] == []


def test_a_planted_own_feature_makes_b3_beat_b2(own_result):
    loss = own_result["metrics"].set_index("model").log_loss
    assert loss.B3 < loss.B2 - 0.1
    deltas = own_result["deltas"].set_index(["comparison", "metric"])
    step = deltas.loc[("B3 minus B2", "log_loss")]
    assert step.delta == pytest.approx(loss.B3 - loss.B2) and step.ci_high < 0
    assert step.better == "negative" and step.uncertainty == CONDITIONAL and step.n_boot == BOOT


def test_a_planted_panel_feature_makes_b4_beat_b3(panel_result):
    result = panel_result
    loss = result["metrics"].set_index("model").log_loss
    assert loss.B4 < loss.B3 - 0.05
    deltas = result["deltas"].set_index(["comparison", "metric"])
    assert deltas.loc[("B4 minus B3", "log_loss"), "delta"] == pytest.approx(loss.B4 - loss.B3)
    assert set(result["deltas"].comparison) == {"B1 minus B0", "B2 minus B1", "B3 minus B2", "B4a minus B3",
                                                "B4 minus B4a", "B5 minus B4", "B5d minus B5", "B4 minus B3",
                                                "B5 minus B2"}
    assert set(result["deltas"].metric) == {"log_loss", "brier", "auc"}


def test_removal_ranking_puts_the_planted_family_first(own, panel):
    for tables, planted in ((own, "own"), (panel, "panel_reactions")):
        ranking = removal_ranking(tables[0], "ever_offer", n_boot=BOOT)
        assert set(ranking.family) == {"stage_a", "own", "panel_content", "panel_reactions", "rapport_repair"}
        assert ranking.family.iloc[0] == planted
        assert ranking.delta_vs_full.iloc[0] > 0.1 and ranking.ci_low.iloc[0] > 0
        assert list(ranking.delta_vs_full) == sorted(ranking.delta_vs_full, reverse=True)
        assert np.allclose(ranking.delta_vs_full, ranking.log_loss_without - ranking.log_loss_full)
        assert ranking.log_loss_full.nunique() == 1 and (ranking.caveat == REMOVAL_CAVEAT).all()
        assert "not evidence of irrelevance" in REMOVAL_CAVEAT
    # the full model of the ranking is B5 on the same folds
    full = evaluate(own[0], "ever_offer", penalty=10.0, n_boot=20).get("metrics").set_index("model").log_loss.B5
    assert removal_ranking(own[0], "ever_offer", penalty=10.0, n_boot=20).log_loss_full.iloc[0] == pytest.approx(full)


def test_removal_ranking_treats_decision_history_like_any_family_of_its_base_model(own, own_result):
    with_history = removal_ranking(own[0], "ever_offer", base_model="B5d", penalty=10.0, n_boot=20)
    assert set(with_history.family) == {"stage_a", "own", "panel_content", "panel_reactions", "rapport_repair",
                                        "decision_history"}
    assert (with_history.base_model == "B5d").all() and with_history.family.iloc[0] == "own"
    default = removal_ranking(own[0], "ever_offer", penalty=10.0, n_boot=20)
    assert "decision_history" not in set(default.family) and (default.base_model == "B5").all()


def test_evaluation_is_seeded_and_reproducible(own, own_result):
    again = evaluate(own[0], "ever_offer", n_boot=BOOT)
    pd.testing.assert_frame_equal(again["deltas"], own_result["deltas"])
    pd.testing.assert_frame_equal(again["metrics"], own_result["metrics"])
    other = evaluate(own[0], "ever_offer", penalty=10.0, n_boot=BOOT, seed=7)
    other = evaluate(own[0], "ever_offer", n_boot=BOOT, seed=7)
    defined = own_result["deltas"].delta.notna()          # the AUC step from the constant head has no value
    assert not np.allclose(other["deltas"].ci_low[defined], own_result["deltas"].ci_low[defined])
    assert np.allclose(other["deltas"].delta, own_result["deltas"].delta, equal_nan=True)


# ---- preprocessing is fitted on training pitches only ------------------------------------------

def test_preprocessing_uses_training_rows_only():
    train = _frame(own_sev_mean=[1.0, 2.0, 3.0, np.nan] * 3, own_sev_mean_missing=[0, 0, 0, 1] * 3,
                   founder_gender=["all_men"] * 6 + ["all_women"] * 6)
    test = _frame(4, own_sev_mean=[np.nan, 2.0, 50.0, 2.0], own_sev_mean_missing=[1, 0, 0, 0],
                  founder_gender=["all_men", "all_women", "mixed", "NONE"])
    families = ("control", "exposure", "own")
    # each gender is seen in three training pitches (two rows a pitch), so three is the bar here
    x_train, x_test, names = design_matrix(train, test, families, min_category=3)
    assert names == ["hist_turns", "own_sev_mean", "own_sev_mean_missing", "founder_gender=all_men",
                     "founder_gender=all_women"]
    assert np.allclose(x_train[:, :2].mean(axis=0), 0) and np.allclose(x_train[:, :2].std(axis=0), 1)
    filled = np.where(np.isnan(train.own_sev_mean), 2.0, train.own_sev_mean)        # training median is 2
    assert np.allclose(x_test[:, 1], (np.array([2.0, 2.0, 50.0, 2.0]) - filled.mean()) / filled.std())
    assert x_test[:, 2].tolist() == [1, 0, 0, 0] and set(x_train[:, 2]) == {0.0, 1.0}      # indicator stays 0/1
    assert x_test[:, 3:].tolist() == [[1, 0], [0, 1], [0, 0], [0, 0]]
    # other held-out values leave the fold's medians, scaler and training matrix untouched
    wild = test.assign(own_sev_mean=[1e6, -1e6, np.nan, 3.0], hist_turns=[9e9] * 4, founder_gender=["mixed"] * 4)
    a, b = fit_design(train, families, min_category=3), fit_design(train, families, min_category=3)
    assert a.numeric == b.numeric and a.levels == b.levels
    x_train_2, x_wild, names_2 = design_matrix(train, wild, families, min_category=3)
    assert np.array_equal(x_train, x_train_2) and names == names_2 and not np.array_equal(x_test, x_wild)


def test_held_out_values_do_not_change_the_fold_that_holds_them_out(own):
    o2 = own[0]
    held = o2.video_id == "v003"
    changed = o2.copy()
    changed.loc[held, "own_resp_mean"] = 99.0
    changed.loc[held, "hist_turns"] = 5000
    changed.loc[held, "founder_gender"] = "a_level_never_seen"
    models = {"B3": MODEL_FAMILY["B3"]}
    a = evaluate(o2, "ever_offer", models=models, n_boot=10)
    b = evaluate(changed, "ever_offer", models=models, n_boot=10)
    fold_a = next(f for f in a["folds"] if f["held_out"] == "v003")["models"]["B3"]
    fold_b = next(f for f in b["folds"] if f["held_out"] == "v003")["models"]["B3"]
    assert fold_a == fold_b                                    # medians, scaler, levels, penalty, coefficients
    assert fold_a["penalty"] in PENALTY_GRID and fold_a["penalty_mode"] == "inner_cv"
    assert fold_a["medians"]["own_resp_mean"] <= 1.0
    pa = a["predictions"].set_index(["video_id", "investor"]).p_B3
    pb = b["predictions"].set_index(["video_id", "investor"]).p_B3
    assert not np.allclose(pa.loc["v003"], pb.loc["v003"])      # only its own predictions move






def test_constant_and_all_missing_columns_are_dropped_and_listed():
    train = _frame(prepared_seconds=[55.0] * 12, own_sev_mean=[np.nan] * 12, own_sev_mean_missing=[1] * 12,
                   founder_gender=["mixed"] * 12, own_questions=[0, 1] * 6)
    design = fit_design(train, ("control", "exposure", "own"))
    assert design.names == ["hist_turns", "own_questions"]
    assert design.dropped == ["prepared_seconds", "own_sev_mean", "own_sev_mean_missing", "founder_gender=mixed"]
    x_train, x_test, names = design_matrix(train, train.head(3), ("control", "exposure", "own"))
    assert x_train.shape == (12, 2) and x_test.shape == (3, 2) and np.isfinite(x_train).all()
    # no families: an empty matrix, not an error
    assert design_matrix(train, train.head(3), ())[0].shape == (12, 0)




def test_single_class_training_fold_falls_back_to_the_base_rate():
    train = _frame(own_questions=[0, 1, 2] * 4).assign(y=1.0)
    test = _frame(2, own_questions=[5, 0])
    prob, detail = fit_head(train, test, "y", ("exposure", "own"))
    assert prob.tolist() == [1.0, 1.0] and detail["single_class_training"] is True and detail["coefficients"] == {}
    # inside evaluate: two pitches, each fold trains on one class
    rows = pd.DataFrame({"video_id": ["a", "a", "b", "b"], "hist_turns": [1.0, 2.0, 3.0, 4.0],
                         "ever_offer": [1.0, 1.0, 0.0, 0.0]})
    result = evaluate(rows, "ever_offer", models={"B0": (), "B1": MODEL_FAMILY["B1"]}, n_boot=20)
    assert result["predictions"].p_B1.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert result["metrics"].set_index("model").loc["B1", "penalty"] == "none fitted"
    assert all(f["models"]["B1"]["single_class_training"] for f in result["folds"])
    assert result["metrics"].set_index("model").loc["B1", "single_class_folds"] == 2
    assert np.isfinite(result["metrics"].log_loss).all()






# ---- pooled versus attributed ---------------------------------------------------------------------

def test_pooled_versus_attributed_uses_the_same_pitches_for_both_arms(own):
    o2, o1 = own
    extra = o1.iloc[[0]].assign(video_id="only_in_o1")
    result = pooled_vs_attributed(pd.concat([o1, extra], ignore_index=True), o2, n_boot=BOOT)
    pred = result["predictions"]
    known = o1[o1.any_offer.notna()]
    assert result["pitches"] == sorted(known.video_id) == list(pred.video_id) and "only_in_o1" not in result["pitches"]
    assert result["n_groups"] == len(known) and result["n_events"] == int(known.any_offer.sum())
    assert pred[["p_collective", "p_attributed", "p_attributed_max"]].notna().all().all()
    assert result["penalty"]["mode"] == "inner_cv" and result["pitches_without_investor_rows"] == ["only_in_o1"]
    assert (pred.y.to_numpy() == known.sort_values("video_id").any_offer.to_numpy()).all()
    assert (pred.n_investors.to_numpy() == o2.groupby("video_id").size().loc[pred.video_id].to_numpy()).all()
    for arm in ("collective", "attributed", "attributed_max"):
        assert set(result[arm]) == {"log_loss", "brier", "auc", "mean_pred", "base_rate"}
        assert result[arm]["base_rate"] == pytest.approx(known.any_offer.mean())
    assert result["collective"]["log_loss"] == pytest.approx(metrics_of(pred.y, pred.p_collective)["log_loss"])
    diff = result["difference"]["log_loss"]
    assert diff["delta"] == pytest.approx(result["attributed"]["log_loss"] - result["collective"]["log_loss"])
    assert diff["ci_low"] <= diff["delta"] <= diff["ci_high"] and result["uncertainty"] == CONDITIONAL


def test_attributed_arm_is_reported_as_noisy_or_and_as_the_largest_investor_probability(own):
    o2, o1 = own
    result = pooled_vs_attributed(o1, o2, penalty=10.0, n_boot=20)
    assert result["attributed_primary"] == "noisy_or" and "primary attributed estimate" in result["attributed_note"]
    assert result["penalty"] == penalty_setting(10.0)
    pitches = result["pitches"]
    held = pitches[2]
    train = o2[o2.video_id.isin(pitches) & (o2.video_id != held) & o2.ever_offer.notna()].assign(
        y=lambda d: d.ever_offer)
    prob, _ = fit_head(train, o2[o2.video_id == held], "y", MODEL_FAMILY["B5"], 10.0)
    row = result["predictions"].set_index("video_id").loc[held]
    assert row.p_attributed_max == pytest.approx(prob.max()) and len(prob) == row.n_investors
    assert row.p_attributed == pytest.approx(1 - np.prod(1 - prob)) and row.p_attributed >= row.p_attributed_max
    pred = result["predictions"]
    assert (pred.p_attributed >= pred.p_attributed_max - 1e-12).all()
    assert result["attributed"]["log_loss"] == pytest.approx(metrics_of(pred.y, pred.p_attributed)["log_loss"])
    assert result["attributed_max"]["log_loss"] == pytest.approx(metrics_of(pred.y, pred.p_attributed_max)["log_loss"])
    assert result["difference"]["log_loss"]["delta"] == pytest.approx(
        result["attributed"]["log_loss"] - result["collective"]["log_loss"])
    assert result["difference_max"]["log_loss"]["delta"] == pytest.approx(
        result["attributed_max"]["log_loss"] - result["collective"]["log_loss"])


def test_a_pitch_without_investor_rows_is_left_out_and_named_instead_of_failing(own):
    o2, o1 = own
    lonely = o1[o1.any_offer.notna()].video_id.iloc[3]
    result = pooled_vs_attributed(o1, o2[o2.video_id != lonely], penalty=10.0, n_boot=20)
    assert result["status"] == "estimated" and lonely not in result["pitches"]
    assert result["pitches_without_investor_rows"] == [lonely]
    assert result["n_groups"] == int(o1.any_offer.notna().sum()) - 1
    assert np.isfinite(result["predictions"][["p_attributed", "p_attributed_max"]]).all().all()
    assert pooled_vs_attributed(o1, o2, penalty=10.0, n_boot=20)["pitches_without_investor_rows"] == []
    # an investor table that is empty altogether: stated, not raised
    nothing = pooled_vs_attributed(o1, o2.iloc[:0], penalty=10.0, n_boot=20)
    assert nothing["status"].startswith("not estimable") and len(nothing["pitches_without_investor_rows"]) > 0


# ---- small samples ------------------------------------------------------------------------------

def test_five_pitches_run_and_carry_the_demonstration_label(tmp_path_factory):
    o2, o1 = _tables(tmp_path_factory, None, n=5, seed=3)
    result = evaluate(o2, "ever_offer", n_boot=50)
    assert result["label"] == DEMONSTRATION == "demonstration only" and result["n_groups"] == 5
    assert len(result["metrics"]) == 8 and np.isfinite(result["metrics"].log_loss).all()
    ranking = removal_ranking(o2, "ever_offer", n_boot=50)
    assert len(ranking) == 5
    both = pooled_vs_attributed(o1, o2, n_boot=50)
    assert both["label"] == DEMONSTRATION and both["status"] == "estimated"
    # one pitch cannot be held out from itself: stated, not raised
    single = evaluate(o2[o2.video_id == "v000"], "ever_offer", n_boot=10)
    assert single["status"].startswith("not estimable") and single["metrics"].empty and single["label"] == DEMONSTRATION
    assert removal_ranking(o2[o2.video_id == "v000"], "ever_offer", n_boot=10).empty
    assert pooled_vs_attributed(o1[o1.video_id == "v000"], o2, n_boot=10)["status"].startswith("not estimable")



# ---- the penalty is chosen inside each training fold -----------------------------------------









def test_every_fold_records_the_penalty_each_model_chose_from_its_training_pitches_only(own, own_result):
    o2 = own[0]
    assert own_result["penalty"] == penalty_setting(None) and own_result["penalty"]["mode"] == "inner_cv"
    assert own_result["penalty"]["grid"] == [1.0, 10.0, 100.0, 1000.0]
    assert own_result["penalty"]["text"] == "penalty: inner_cv"
    for fold in own_result["folds"]:
        assert fold["models"]["B0"]["penalty"] is None
        for name in list(MODEL_FAMILY)[1:]:
            detail = fold["models"][name]
            assert detail["penalty"] in PENALTY_GRID and detail["penalty_mode"] == "inner_cv"
            losses = {float(k): v for k, v in detail["penalty_cv_log_loss"].items()}
            assert set(losses) == set(PENALTY_GRID)
            best = min(losses.values())
            assert detail["penalty"] == max(k for k, v in losses.items() if v <= best + 1e-12)
    metrics = own_result["metrics"].set_index("model")
    assert metrics.loc["B0", "penalty"] == "none fitted" and metrics.loc["B3", "penalty"].startswith("inner_cv: ")
    assert sum(int(part.split(" x")[1]) for part in metrics.loc["B3", "penalty"][len("inner_cv: "):].split(", ")) == 30
    # the choice of one fold, recomputed from that fold's training pitches alone
    fold = own_result["folds"][4]
    known = o2[o2.ever_offer.notna()].assign(y=lambda d: d.ever_offer)
    train = known[known.video_id != fold["held_out"]]
    design = fit_design(train, MODEL_FAMILY["B3"])
    chosen, _ = choose_penalty(design.transform(train), train.y.to_numpy(), train.video_id.to_numpy())
    assert chosen == fold["models"]["B3"]["penalty"]
    # the held-out pitch has no say: other values there, the same choice everywhere in that fold
    changed = o2.copy()
    held = changed.video_id == fold["held_out"]
    changed.loc[held, "own_resp_mean"] = 1 - changed.loc[held, "own_resp_mean"]
    changed.loc[held & changed.ever_offer.notna(), "ever_offer"] = 1 - changed.loc[
        held & changed.ever_offer.notna(), "ever_offer"]
    again = evaluate(changed, "ever_offer", models={"B3": MODEL_FAMILY["B3"]}, n_boot=10)
    assert next(f for f in again["folds"] if f["held_out"] == fold["held_out"])["models"]["B3"] == fold["models"]["B3"]


def test_a_fixed_penalty_is_an_explicit_override_and_is_recorded_as_one(own):
    result = evaluate(own[0], "ever_offer", models={"B0": (), "B3": MODEL_FAMILY["B3"]}, penalty=5.0, n_boot=10)
    assert result["penalty"] == {"mode": "fixed", "value": 5.0, "text": "penalty: fixed 5"}
    assert {f["models"]["B3"]["penalty"] for f in result["folds"]} == {5.0}
    assert {f["models"]["B3"]["penalty_mode"] for f in result["folds"]} == {"fixed"}
    assert result["metrics"].set_index("model").loc["B3", "penalty"] == "fixed 5"
    assert all(f["models"]["B3"]["penalty_cv_log_loss"] == {} for f in result["folds"])


def test_the_constant_head_has_no_auc(own_result):
    metrics = own_result["metrics"].set_index("model")
    assert np.isnan(metrics.loc["B0", "auc"]) and np.isnan(metrics.loc["B0", "auc_rows"])
    assert np.isfinite(metrics.loc["B0", ["log_loss", "brier"]].astype(float)).all()
    assert np.isfinite(metrics.drop("B0").auc).all()
    deltas = own_result["deltas"].set_index(["comparison", "metric"])
    step = deltas.loc[("B1 minus B0", "auc")]
    assert np.isnan(step.delta) and np.isnan(step.ci_low) and step.n_boot == 0
    assert np.isfinite(deltas.loc[("B1 minus B0", "log_loss"), "delta"])
    assert np.isfinite(deltas.loc[("B2 minus B1", "auc"), "delta"])


# ---- models named by feature, and one prediction taken apart and edited ---------------------------


NAMED = ["own_questions", "own_resp_mean", "oth_agrees", "founder_gender"]


@pytest.fixture(scope="module")
def custom_result(own):
    o2, _ = own
    models = {"B0": [], "B3": MODEL_FAMILY["B3"], "named": NAMED, "mixed": ["control", "own_resp_mean"]}
    return evaluate(o2, "ever_offer", models=models, n_boot=BOOT)


def test_a_model_can_be_a_list_of_families_of_feature_names_or_of_both(own, custom_result):
    o2, _ = own
    assert list(custom_result["metrics"].model) == ["B0", "B3", "named", "mixed"]
    named = family_columns(o2, tuple(NAMED))
    assert named == {"categorical": ["founder_gender"], "indicator": [],
                     "numeric": ["own_questions", "own_resp_mean", "oth_agrees"]}
    mixed = family_columns(o2, ("control", "own_resp_mean"))
    assert mixed["numeric"] == ["own_resp_mean"] and "primary_industry" in mixed["categorical"]
    for fold in custom_result["folds"]:      # nothing but the named columns reaches the head
        assert {c.split("=", 1)[0] for c in fold["models"]["named"]["columns"]} <= set(NAMED)
    # the planted effect lives in the answers to an investor's own questions: three named features carry it
    metrics = custom_result["metrics"].set_index("model")
    assert metrics.log_loss["named"] < metrics.log_loss["B0"]
    assert metrics.families["named"] == "+".join(NAMED)


def test_an_unknown_feature_or_family_name_is_an_error_that_lists_the_valid_names(own):
    o2, _ = own
    with pytest.raises(ValueError) as caught:
        evaluate(o2, "ever_offer", models={"B0": [], "typo": ["own_questions", "own_respmean", "panel"]}, n_boot=9)
    message = str(caught.value)
    assert "model 'typo'" in message and "'own_respmean'" in message and "'panel'" in message
    assert "'own_questions'" not in message.split("Valid families")[0]
    assert all(family in message for family in FAMILIES) and "own_resp_mean" in message
    # identity and keys are not predictors, an O1-only feature is not in the O2 table, and a bare string is refused
    for bad in (["investor_label"], ["cut_pos"], ["q_askers"]):
        with pytest.raises(ValueError, match="unknown feature or family"):
            check_models({"m": bad}, o2)
    with pytest.raises(ValueError, match="give a list of names"):
        check_models({"m": "own"}, o2)
    assert check_models({"m": ["q_askers"]}) == {"m": ("q_askers",)}       # valid without a table
    assert "investor_label" not in predictor_names() and "own_questions" in predictor_names(o2)


def test_a_stored_fold_reapplied_to_its_rows_reproduces_the_held_out_predictions(own, custom_result):
    o2, _ = own
    pred = custom_result["predictions"]
    for model in ("B3", "named"):
        for video_id, investor, p in list(zip(pred.video_id, pred.investor, pred[f"p_{model}"]))[:12]:
            same = edited_prediction(custom_result, o2, model, video_id, investor, {})
            assert same["original"] == pytest.approx(p, abs=1e-12) and same["change"] == 0.0
    detail = fold_detail(custom_result, "named", pred.video_id.iloc[0])
    assert fold_design(detail).names == detail["columns"]


def test_the_breakdown_adds_up_to_the_prediction(own, custom_result):
    o2, _ = own
    pred = custom_result["predictions"]
    video_id, investor = pred.video_id.iloc[3], pred.investor.iloc[3]
    table = prediction_breakdown(custom_result, o2, "named", video_id, investor)
    assert table.feature.iloc[0] == "(intercept)" and set(table.feature[1:]) == set(
        fold_detail(custom_result, "named", video_id)["columns"])
    logit = float(table.contribution.astype(float).sum())
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(float(pred.p_named.iloc[3]), abs=1e-9)
    sizes = table.contribution.astype(float).abs().to_numpy()[1:]
    assert list(sizes) == sorted(sizes, reverse=True)
    assert table.label[table.feature == "own_resp_mean"].iloc[0] == FEATURES["own_resp_mean"]["label"]
    assert prediction_breakdown(custom_result, o2, "B0", video_id, investor).empty       # a base rate has no parts


def test_editing_a_feature_moves_the_prediction_the_way_its_coefficient_points(own, custom_result):
    o2, _ = own
    pred = custom_result["predictions"]
    video_id, investor = pred.video_id.iloc[0], pred.investor.iloc[0]
    detail = fold_detail(custom_result, "named", video_id)
    assert "own_resp_mean" in editable_features(custom_result, "named", video_id)
    up = edited_prediction(custom_result, o2, "named", video_id, investor, {"own_resp_mean": 1.0})
    down = edited_prediction(custom_result, o2, "named", video_id, investor, {"own_resp_mean": 0.0})
    assert detail["coefficients"]["own_resp_mean"] > 0            # planted: good answers to own questions
    assert up["edited"] > down["edited"] and up["original"] == down["original"]
    assert up["change"] == pytest.approx(up["edited"] - up["original"])
    assert SENSITIVITY_NOTE in up["note"] and "not a causal effect" in up["note"]
    before = o2.copy()
    edited_prediction(custom_result, o2, "named", video_id, investor, {"own_resp_mean": 0.0})
    pd.testing.assert_frame_equal(o2, before)                      # the table itself is never edited


def test_an_edit_outside_the_model_is_refused_and_a_base_rate_cannot_be_edited(own, custom_result):
    o2, _ = own
    pred = custom_result["predictions"]
    video_id, investor = pred.video_id.iloc[0], pred.investor.iloc[0]
    with pytest.raises(ValueError) as caught:
        edited_prediction(custom_result, o2, "named", video_id, investor, {"oth_challenges": 3})
    assert "not a feature of model 'named'" in str(caught.value) and "own_resp_mean" in str(caught.value)
    with pytest.raises(ValueError, match="none: it predicts the base rate"):
        edited_prediction(custom_result, o2, "B0", video_id, investor, {"own_resp_mean": 1.0})
    assert "no edit can change it" in edited_prediction(custom_result, o2, "B0", video_id, investor, {})["note"]
    with pytest.raises(ValueError, match="unknown model"):
        edited_prediction(custom_result, o2, "B9", video_id, investor, {})
    with pytest.raises(ValueError, match="was not held out"):
        edited_prediction(custom_result, o2, "named", "nope", investor, {})
    with pytest.raises(ValueError, match="expected exactly one row"):
        edited_prediction(custom_result, o2, "named", video_id, None, {})


def test_a_feature_the_fold_dropped_can_be_named_but_changes_nothing():
    # `flat` is constant in every training fold, so each fold drops it
    frame = _frame(n=24, flat=[1.0] * 24, slope=list(np.linspace(-1, 1, 24)))
    frame["video_id"] = [f"v{i // 3}" for i in range(24)]
    frame["ever_offer"] = [float(i % 3 == 0) for i in range(24)]
    from analysis.archive.shark_level_features import ARCHIVE_FEATURES       # family `own` is a shark-level one
    ARCHIVE_FEATURES["flat"] = {"family": "own", "label": "flat", "unit": "o2", "kind": "concept", "description": ""}
    ARCHIVE_FEATURES["slope"] = {"family": "own", "label": "slope", "unit": "o2", "kind": "concept", "description": ""}
    try:
        result = evaluate(frame, "ever_offer", models={"m": ["flat", "slope"]}, penalty=1.0, n_boot=9)
        frame["investor"] = [f"S{i % 3}" for i in range(24)]
        edit = edited_prediction(result, frame, "m", "v0", "S0", {"flat": 99.0})
        assert edit["change"] == 0.0 and "flat was dropped in this fold" in edit["note"]
        assert editable_features(result, "m", "v0") == ["slope"]
    finally:
        del ARCHIVE_FEATURES["flat"], ARCHIVE_FEATURES["slope"]


# ---- the draft pre-specified model set ----------------------------------------------------------

def test_the_primary_features_are_a_short_explicit_list_per_family_and_the_set_is_marked_draft():
    assert PRIMARY_STATUS == "draft — to be confirmed by the researcher before the scale run"
    assert PRIMARY_FEATURES == {
        "control": ("founder_gender", "product_form", "primary_industry"),
        "exposure": ("hist_turns", "hist_questions_all"),
        "stage_a": ("sa_rank_pitch_positivity", "sa_rank_expressed_enthusiasm", "sa_rank_interpersonal_warmth"),
        "own": ("own_questions", "own_subs_mean", "own_evid_share", "own_subs_hard_mean", "own_receipt_share",
                "own_satisfied", "own_unsatisfied"),
        "panel_content": ("oth_questions", "oth_subs_mean", "oth_evid_share"),
        "panel_reactions": ("oth_agrees", "oth_challenges", "oth_unsatisfied"),
        "rapport_repair": ("own_social", "n_social_exchanges"),
    }
    assert PRIMARY_STAGE_A_RATINGS == ("sa_pitch_positivity", "sa_expressed_enthusiasm", "sa_interpersonal_warmth")
    assert PRIMARY_SENSITIVITY_FEATURES == ("oth_exits_before", "oth_offers_before")
    named = [f for names in PRIMARY_FEATURES.values() for f in names]
    assert len(named) == len(set(named)) == 23
    for name in (*named, *PRIMARY_STAGE_A_RATINGS, *PRIMARY_SENSITIVITY_FEATURES):
        assert name in FEATURES and FEATURES[name]["family"] is not None, name
    # deliberately absent: directness, uncertainty, coherence and context
    assert not [f for f in named if f.endswith(("resp_mean", "uncert_share", "coh_mean", "ctx_mean"))]


def test_the_primary_models_are_cumulative_in_the_order_of_the_b_family():
    assert list(PRIMARY_MODELS) == ["P0", "P1", "P2", "P3", "P4", "P5", "P5d"]
    f = PRIMARY_FEATURES
    assert PRIMARY_MODELS["P0"] == ()
    assert PRIMARY_MODELS["P1"] == f["control"] + f["exposure"]
    assert PRIMARY_MODELS["P2"] == PRIMARY_MODELS["P1"] + f["stage_a"]
    assert PRIMARY_MODELS["P3"] == PRIMARY_MODELS["P2"] + f["own"]
    assert PRIMARY_MODELS["P4"] == PRIMARY_MODELS["P3"] + f["panel_content"] + f["panel_reactions"]
    assert PRIMARY_MODELS["P5"] == PRIMARY_MODELS["P4"] + f["rapport_repair"]
    assert PRIMARY_MODELS["P5d"] == PRIMARY_MODELS["P5"] + PRIMARY_SENSITIVITY_FEATURES
    assert primary_models() == PRIMARY_MODELS
    # a model of the set is named features only: no whole family can ride in
    assert not {name for spec in PRIMARY_MODELS.values() for name in spec} & set(MODEL_FAMILY["B5d"])


def test_the_primary_stage_a_features_are_rank_scores_when_the_table_has_them_and_ratings_otherwise(
        own, tmp_path_factory):
    plain = primary_models(own[0])                                       # this run has no ranking
    assert plain["P2"][-len(PRIMARY_STAGE_A_RATINGS):] == PRIMARY_STAGE_A_RATINGS and not [n for n in plain["P5d"] if n.startswith("sa_rank_")]
    assert check_models(plain, own[0]) == plain
    ranked = _tables(tmp_path_factory, None, n=6, seed=4, ranking=True)[0]
    assert primary_models(ranked) == PRIMARY_MODELS
    assert check_models(primary_models(ranked), ranked) == PRIMARY_MODELS


def test_the_primary_set_evaluates_like_any_models_and_its_decision_history_step_is_a_sensitivity(own):
    models = primary_models(own[0])
    result = evaluate(own[0], "ever_offer", models=models, n_boot=49, penalty=10.0)
    metrics = result["metrics"].set_index("model")
    assert list(metrics.index) == list(models) and np.isfinite(metrics.log_loss).all()
    assert metrics.loc["P5d", "role"] == "sensitivity" and (metrics.drop("P5d").role == "main").all()
    comparisons = set(result["deltas"].comparison)
    assert {"P3 minus P2", "P4 minus P3", "P5d minus P5", "P5 minus P2"} <= comparisons
    for fold in result["folds"]:
        used = {c.split("=")[0] for c in fold["models"]["P5"]["columns"]}
        assert used <= set(models["P5"])                                # exactly the named columns, no companion
        assert not {"oth_exits_before", "oth_offers_before"} & used
    # the planted effect (answers to the investor's own questions) is in the short own list
    assert metrics.loc["P3", "log_loss"] < metrics.loc["P2", "log_loss"]


def test_every_feature_has_a_status_and_the_dictionary_shows_it():
    # Plan B task 3: the pitch columns now have the pitch-level status of the ladder (the draft pitch list that
    # made some of them primary and the a_* answer statistics descriptive was dropped); the shark-level columns
    # keep theirs, which this test pins as before.
    named = {f for names in PRIMARY_FEATURES.values() for f in names} | set(PRIMARY_STAGE_A_RATINGS)
    shark = [name for name in ARCHIVE_FEATURES if name not in PITCH_FEATURES]
    dictionary = combined_dictionary().set_index("name")
    assert {name for name in shark if feature_status(name) == "primary"} == named & set(shark)
    assert set(dictionary.index[dictionary.status == "primary"]) & set(shark) == named & set(shark)
    for name in ("own_resp_mean", "oth_resp_mean", "own_uncert_share", "oth_coh_mean", "own_resp_mean_n"):
        assert feature_status(name) == "descriptive" == dictionary.status[name], name
    for name in ("own_sev_mean", "own_subs_mean_n", "oth_exits_before", "own_followup_share"):
        assert feature_status(name) == "exploratory" == dictionary.status[name], name
    # a pitch column has its pitch-level status from the families alone: the ladder's exclusion of ruling I4 (plan B
    # final review) leaves the archive and the legacy notebook's dictionary unchanged
    for name in PITCH_FEATURES:
        assert feature_status(name) == pb.feature_status(name, ladder_exclusion=False) == dictionary.status[name], name
    assert pb.feature_status("demo_seconds") == "descriptive" and feature_status("demo_seconds") == "primary"
    assert feature_status("video_id") is None and feature_status("investor") is None and dictionary.status["cut_pos"] is None
    assert set(dictionary.status.dropna()) == {"primary", "descriptive", "exploratory"}







# ---- each feature on its own ---------------------------------------------------------------------

def _toy(n_pitches=40, rows=4, seed=0):
    """Investor rows with one planted feature, one pure-noise feature, one sparse and one constant feature."""
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i:02d}" for i in range(n_pitches)], rows)
    n = len(groups)
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-2.0 * signal))).astype(float)
    questions = rng.poisson(3, size=n).astype(float)
    sparse = np.where(rng.random(n) < 0.2, rng.normal(size=n), np.nan)          # observed on about 20% of the rows
    partly = np.where(rng.random(n) < 0.6, -signal + rng.normal(scale=0.5, size=n), np.nan)
    return pd.DataFrame({"video_id": groups, "investor": [f"S{i % rows + 1}" for i in range(n)], "ever_offer": y,
                         "own_satisfied": signal, "own_unsatisfied": rng.normal(size=n), "own_questions": questions,
                         "own_sev_mean": partly, "own_sev_mean_n": questions, "own_sev_mean_missing": np.isnan(partly) * 1.0,
                         "own_resp_mean": sparse, "own_agrees": np.ones(n), "oth_offers_before": rng.integers(0, 3, n) * 1.0,
                         "founder_gender": "all_men"})


def test_single_features_rank_each_numeric_feature_by_its_own_association_with_the_outcome():
    frame = _toy()
    table = single_features(frame, "ever_offer", n_boot=300, seed=1)
    assert list(table.columns) == SINGLE_COLUMNS
    by = table.set_index("feature")
    # ranked: numeric features with enough coverage that vary; not the control, the indicator, the sparse or the constant one
    assert set(by.index) == {"own_satisfied", "own_unsatisfied", "own_questions", "own_sev_mean", "own_sev_mean_n",
                             "oth_offers_before"}
    assert table.feature.iloc[0] in ("own_satisfied", "own_sev_mean") and list(table.correlation.abs()) == sorted(
        table.correlation.abs(), reverse=True)
    planted = by.loc["own_satisfied"]
    assert planted.correlation == pytest.approx(np.corrcoef(frame.own_satisfied, frame.ever_offer)[0, 1])
    assert 0 < planted.corr_low < planted.correlation < planted.corr_high and planted.coefficient > 0.5
    assert planted.log_loss_gain > 0.05 and planted.gain_low > 0 and planted.n == len(frame) and planted.coverage == 1.0
    assert (planted.label, planted.family, planted.status) == (FEATURES["own_satisfied"]["label"], "own", "primary")
    noise = by.loc["own_unsatisfied"]
    assert noise.corr_low < 0 < noise.corr_high and abs(noise.log_loss_gain) < 0.02 and noise.gain_low < 0
    # the opposite direction, on the rows that have a value only
    partly = by.loc["own_sev_mean"]
    seen = frame.own_sev_mean.notna()
    assert partly.n == int(seen.sum()) and partly.coverage == pytest.approx(seen.mean()) and partly.coverage < 1
    assert partly.correlation == pytest.approx(np.corrcoef(frame.own_sev_mean[seen], frame.ever_offer[seen])[0, 1])
    assert partly.correlation < 0 and partly.coefficient < 0 and partly.corr_high < 0
    # notes: a count behind a mean, and the order of deciding
    assert by.loc["own_sev_mean_n", "note"] == NOTE_COUNT and by.loc["oth_offers_before", "note"] == NOTE_DECISION_ORDER
    assert by.loc["own_satisfied", "note"] == "" and count_behind_a_mean("own_sev_mean_n")
    assert not count_behind_a_mean("own_questions") and not count_behind_a_mean("hist_turns")
    # adjusted for the number of questions asked: a partial correlation; the count itself (and its copy) has none
    assert set(by.adjusted_for) == {"own_questions", ""} and np.isnan(by.loc["own_questions", "adjusted_correlation"])
    assert np.isnan(by.loc["own_sev_mean_n", "adjusted_correlation"])                 # identical to the count: nothing left
    x, q, y = frame.own_satisfied, frame.own_questions, frame.ever_offer
    r = lambda a, b: np.corrcoef(a, b)[0, 1]                                          # noqa: E731
    expected = (r(x, y) - r(x, q) * r(y, q)) / np.sqrt((1 - r(x, q) ** 2) * (1 - r(y, q) ** 2))
    assert planted.adjusted_correlation == pytest.approx(expected) and planted.adj_low < expected < planted.adj_high
    assert single_features(frame, "ever_offer", n_boot=300, seed=1).equals(table)      # seeded
    assert "own_resp_mean" in set(single_features(frame, "ever_offer", n_boot=20, min_coverage=0.1).feature)
    assert single_features(frame[frame.video_id == "p00"], "ever_offer", n_boot=20).empty


def test_the_single_feature_gain_is_held_out_and_weights_pitches_equally():
    """The gain of a feature is base-rate loss minus one-feature-head loss on pitches the head never saw."""
    frame = _toy(n_pitches=12, rows=3, seed=3)
    frame = pd.concat([frame, frame[frame.video_id == "p00"]], ignore_index=True)       # one pitch has twice the rows
    table = single_features(frame, "ever_offer", n_boot=50).set_index("feature")
    y, x = frame.ever_offer.to_numpy(), frame.own_satisfied.to_numpy()
    gains = []
    for held_out in sorted(frame.video_id.unique()):
        test = (frame.video_id == held_out).to_numpy()
        mean, sd = x[~test].mean(), x[~test].std()
        beta = ridge_logistic(((x[~test] - mean) / sd)[:, None], y[~test], 1.0)
        p = 1 / (1 + np.exp(-(beta[0] + beta[1] * (x[test] - mean) / sd)))
        base = y[~test].mean()
        loss = lambda q: -(y[test] * np.log(q) + (1 - y[test]) * np.log(1 - q))       # noqa: E731
        gains.append(float((loss(np.full(test.sum(), base)) - loss(p)).mean()))        # one number per pitch
    assert table.loc["own_satisfied", "log_loss_gain"] == pytest.approx(np.mean(gains))
    # a held-out pitch's own values never reach the head that predicts it
    moved = frame.copy()
    moved.loc[moved.video_id == "p05", "own_unsatisfied"] = 50.0
    again = single_features(moved, "ever_offer", n_boot=50).set_index("feature")
    assert again.loc["own_satisfied", "log_loss_gain"] == pytest.approx(table.loc["own_satisfied", "log_loss_gain"])


# ---- questions asked as a control ----------------------------------------------------------------

def test_the_question_control_family_puts_the_count_with_the_controls_and_never_enters_it_again(own):
    o2, o1 = own
    for frame, unit, names in ((o2, "o2", ["Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]),
                               (o1, "o1", ["QC0", "QC1", "QC2", "QC3", "QC4", "QC5"])):
        models = question_control_models(frame)
        question = QUESTIONS_ASKED[unit]
        assert list(models) == names and models[names[0]] == () and question in models[names[1]]
        controls = {c for columns in family_columns(frame, ("control", "exposure")).values() for c in columns}
        assert set(models[names[1]]) == controls | {question}
        for earlier, later in zip(names[:-1], names[1:]):
            assert models[later][:len(models[earlier])] == models[earlier] and len(models[later]) > len(models[earlier])
        full = models[names[-1]]
        assert len(full) == len(set(full)) and full.count(question) == 1 and check_models(models, frame) == models
        added = full[len(models[names[1]]):]
        assert not [c for c in added if count_behind_a_mean(c)]
        assert [c for c in added if c.endswith("_missing")]                      # the none-observed indicators stay
        families = [FEATURES[c]["family"] for c in added]
        assert list(dict.fromkeys(families)) == (
            ["stage_a", "own", "panel_content", "panel_reactions", "rapport_repair"] if unit == "o2"
            else ["stage_a", "questions", "answers", "reactions", "rapport_repair"])
        assert "decision_history" not in families
        pairs = question_control_comparisons(models)
        assert pairs[:len(names) - 1] == list(zip(names[:-1], names[1:]))
        assert pairs[len(names) - 1:] == [(names[1], later) for later in names[3:]]
    # the pitch-level first model is C1: the exposure controls already hold the count
    assert set(question_control_models(o1)["QC1"]) == {c for columns in family_columns(o1, ARCHIVED_C_FAMILY["C1"]).values()
                                                       for c in columns}


def test_the_primary_question_control_sets_are_the_primary_features_with_the_count_in_the_first_step(own):
    o2, o1 = own
    models = question_control_models(o2, primary=True)
    plain = primary_models(o2)
    assert list(models) == ["QP0", "QP1", "QP2", "QP3", "QP4", "QP5"]
    assert models["QP1"] == plain["P1"] + ("own_questions",)
    assert set(models["QP5"]) == set(plain["P5"]) and models["QP5"].count("own_questions") == 1
    assert "own_questions" not in models["QP3"][len(models["QP2"]):]            # the own step no longer carries it
    # plan B task 3 dropped the draft pitch-level list, and its question-control set (QCP0-QCP5) with it
    with pytest.raises(ValueError, match="QCP0-QCP5"):
        question_control_models(o1, primary=True)


def test_question_control_families_are_evaluated_with_every_step_compared_to_the_controls_plus_questions(own):
    o2, o1 = own
    # the pitch rows' case reads QC (plan B task 3 dropped the draft named-feature QCP set it read before)
    for frame, outcome, primary, control in ((o2, "ever_offer", False, "Q1"), (o1, "any_offer", False, "QC1")):
        result = evaluate_question_control(frame, outcome, primary=primary, penalty=10.0, n_boot=30)
        names = list(result["models"])
        assert result["status"] == "estimated" and list(result["metrics"].model) == names
        comparisons = list(dict.fromkeys(result["deltas"].comparison))
        assert comparisons == [f"{b} minus {a}" for a, b in question_control_comparisons(result["models"])]
        assert {f"{later} minus {control}" for later in names[2:]} <= set(comparisons)
        metrics = result["metrics"].set_index("model").log_loss
        d = result["deltas"][result["deltas"].metric == "log_loss"].set_index("comparison")
        assert d.delta[f"{names[-1]} minus {control}"] == pytest.approx(metrics[names[-1]] - metrics[control])
    # the planted effect of this run is in the answers to the investor's own questions: it shows beyond the count
    result = evaluate_question_control(o2, "ever_offer", n_boot=30)
    metrics = result["metrics"].set_index("model").log_loss
    assert metrics["Q3"] < metrics["Q1"]


def test_single_features_rank_nothing_when_the_outcome_does_not_vary():
    frame = _toy().assign(ever_offer=1.0)
    assert single_features(frame, "ever_offer", n_boot=20).empty
    assert list(single_features(frame, "ever_offer", n_boot=20).columns) == SINGLE_COLUMNS


# ---- answer quality: own questions against the other investors' ------------------------------------

def test_answer_quality_models_add_each_block_to_controls_plus_questions_and_nothing_else(own):
    o2, o1 = own
    models = answer_quality_models(o2)
    plain = question_control_models(o2)
    assert list(models) == ["Q0", "Q1", "Q1_own_answers", "Q1_other_answers", "Q1_both_answers"]
    assert models["Q0"] == plain["Q0"] and models["Q1"] == plain["Q1"] and check_models(models, o2) == models
    for name, prefix in (("Q1_own_answers", "own_"), ("Q1_other_answers", "oth_")):
        added = models[name][len(models["Q1"]):]
        assert models[name][:len(models["Q1"])] == models["Q1"]
        assert [c for c in added if not c.endswith("_missing")] == [prefix + stat for stat in ANSWER_QUALITY_STATS]
        assert {f"{prefix}{stat}_missing" for stat in ANSWER_QUALITY_STATS} <= set(added)
        assert not [c for c in added if count_behind_a_mean(c)] and "own_questions" not in added
    both = models["Q1_both_answers"]
    assert set(both) == set(models["Q1_own_answers"]) | set(models["Q1_other_answers"]) and len(both) == len(set(both))
    assert answer_quality_comparisons(models) == [
        ("Q0", "Q1"), ("Q1", "Q1_own_answers"), ("Q1", "Q1_other_answers"), ("Q1", "Q1_both_answers"),
        ("Q1_own_answers", "Q1_both_answers"), ("Q1_other_answers", "Q1_both_answers")]
    pitch = answer_quality_models(o1)                                    # pitch rows have the pooled block only
    assert list(pitch) == ["QC0", "QC1", "QC1_all_answers"]
    assert [c for c in pitch["QC1_all_answers"][len(pitch["QC1"]):] if not c.endswith("_missing")] == [
        "a_" + stat for stat in ANSWER_QUALITY_STATS]
    assert answer_quality_comparisons(pitch) == [("QC0", "QC1"), ("QC1", "QC1_all_answers")]
    result = evaluate_answer_quality(o2, "ever_offer", penalty=10.0, n_boot=30)
    assert result["status"] == "estimated" and list(result["metrics"].model) == list(models)
    assert list(dict.fromkeys(result["deltas"].comparison)) == [f"{b} minus {a}" for a, b in answer_quality_comparisons(models)]
    loss = result["metrics"].set_index("model").log_loss                 # the planted effect is in the own answers
    assert loss["Q1_own_answers"] < loss["Q1"] and loss["Q1_own_answers"] < loss["Q1_other_answers"]


def test_answer_quality_by_offer_compares_the_means_of_offering_and_non_offering_rows():
    frame = _toy(n_pitches=30, rows=4, seed=4)
    rng = np.random.default_rng(0)
    quality = frame.own_satisfied.to_numpy()                             # goes with offers in the toy
    frame = frame.assign(own_subs_mean=3 + quality, own_evid_share=rng.random(len(frame)),
                         own_subs_hard_mean=np.where(rng.random(len(frame)) < 0.4, 3 + quality, np.nan),
                         oth_subs_mean=rng.normal(size=len(frame)), oth_evid_hard_share=rng.random(len(frame)))
    frame.loc[::7, "ever_offer"] = np.nan                                # unknown outcomes are left out
    table = answer_quality_by_offer(frame, "ever_offer", n_boot=200, seed=2)
    assert list(table.columns) == BY_OFFER_COLUMNS
    assert list(table.feature) == ["own_subs_mean", "own_evid_share", "oth_subs_mean", "own_subs_hard_mean",
                                   "oth_evid_hard_share"]                # the columns the frame has, in table order
    assert list(table.questions) == ["all questions"] * 3 + ["hard questions only (severity 2-3)"] * 2
    assert list(table.whose)[:3] == ["the shark's own questions"] * 2 + ["other sharks' questions"]
    known = frame[frame.ever_offer.notna()]
    by = table.set_index("feature")
    for feature in table.feature:
        seen = known[known[feature].notna()]
        row = by.loc[feature]
        assert row.n == len(seen) and row.n_offered == int((seen.ever_offer == 1).sum())
        assert row.n_not_offered == int((seen.ever_offer == 0).sum())
        assert row.mean_offered == pytest.approx(seen.loc[seen.ever_offer == 1, feature].mean())
        assert row.mean_not_offered == pytest.approx(seen.loc[seen.ever_offer == 0, feature].mean())
        assert row.difference == pytest.approx(row.mean_offered - row.mean_not_offered)
        assert row.correlation == pytest.approx(np.corrcoef(seen[feature], seen.ever_offer)[0, 1])
        assert row.corr_low < row.correlation < row.corr_high and row.n_boot == 200
    assert by.loc["own_subs_mean", "corr_low"] > 0 and by.loc["own_subs_hard_mean", "n"] < by.loc["own_subs_mean", "n"]
    assert by.loc["oth_subs_mean", "corr_low"] < 0 < by.loc["oth_subs_mean", "corr_high"]
    assert answer_quality_by_offer(frame, "ever_offer", n_boot=200, seed=2).equals(table)      # seeded
    pitch = frame.drop(columns="investor").assign(a_subs_mean=1.0 + frame.own_satisfied, a_evid_hard_share=0.5)
    pooled = answer_quality_by_offer(pitch, "ever_offer", n_boot=20)
    assert list(pooled.feature) == ["a_subs_mean", "a_evid_hard_share"]
    assert set(pooled.whose) == {"all sharks' questions (pooled)"} and np.isnan(pooled.correlation.iloc[1])   # a constant
    assert answer_quality_by_offer(frame.assign(ever_offer=np.nan), "ever_offer", n_boot=20).empty


# ---- readable metrics ------------------------------------------------------------------------------

def test_skill_over_the_base_rate_is_zero_for_the_base_model_and_reported_with_every_metric_row(own):
    assert skill_over_base(0.3, 0.6) == pytest.approx(50.0) and skill_over_base(0.6, 0.6) == 0.0
    assert skill_over_base(0.9, 0.6) == pytest.approx(-50.0) and skill_over_base(0.0, 0.6) == 100.0
    assert np.isnan(skill_over_base(0.5, 0.0)) and np.isnan(skill_over_base(np.nan, 0.6))
    o2, _ = own
    result = evaluate(o2, "ever_offer", penalty=10.0, n_boot=40)
    m, pred, base = result["metrics"].set_index("model"), result["predictions"], result["base"]
    assert base["log_loss"] == pytest.approx(m.log_loss["B0"]) and base["coin_log_loss"] == pytest.approx(np.log(2))
    assert base["predicted_rate"] == pytest.approx(float((pred.weight * pred.p_B0).sum() / pred.weight.sum()))
    assert (m.base_log_loss == base["log_loss"]).all() and m.skill_pct["B0"] == 0.0
    assert m.skill_pct["B3"] == pytest.approx(100 * (1 - m.log_loss["B3"] / m.log_loss["B0"])) and m.skill_pct["B3"] > 0
    assert (m.skill_low["B0"], m.skill_high["B0"]) == (0.0, 0.0) and m.skill_low["B3"] < m.skill_pct["B3"] < m.skill_high["B3"]
    assert m.auc_low["B3"] < m.auc["B3"] < m.auc_high["B3"] and np.isnan(m.auc_low["B0"]) and np.isnan(m.auc_high["B0"])
    # the base rate of a held-out pitch runs against that pitch's own offer rate: the leave-one-pitch-out artefact
    per = pred.groupby("video_id").agg(p=("p_B0", "mean"), y=("y", "mean"))
    assert base["corr_with_pitch_rate"] == pytest.approx(np.corrcoef(per.p, per.y)[0, 1]) and base["corr_with_pitch_rate"] < -0.9
    assert not m.auc_meaningful["B0"] and m.pred_spread["B0"] == 0.0 and m.auc_meaningful["B3"]
    assert (m.auc_meaningful == (m.auc.notna() & (m.pred_spread >= NEAR_CONSTANT_SPREAD))).all()
    assert evaluate(o2[o2.video_id == o2.video_id.iloc[0]], "ever_offer", n_boot=5)["base"] == {}






# ---- the reaction-trajectory ladders ----------------------------------------------------------



@pytest.fixture(scope="module")
def reactions(tmp_path_factory):
    return _tables(tmp_path_factory, "reaction_events", n=12, strength=4.0)


def test_reaction_columns_follow_the_dictionary_and_can_drop_the_course(reactions):
    o2, o1 = reactions
    own = reaction_columns(o2, ("rx_own_",))
    assert own and all(c.startswith("rx_own_") for c in own) and "rx_own_last_receipt_missing" in own
    assert list(own) == [c for c in FEATURES if c in own]
    counts = reaction_columns(o2, ("rx_own_",), counts_only=True)
    assert "rx_own_receipt_share" in counts and not any("_last_" in c or "_trend_" in c for c in counts)
    assert "rx_own_trend_receipt" in own
    assert reaction_columns(o1, ("rx_",)) and not reaction_columns(o1, ("rx_own_",))     # the pitch row's rx_ shares


def test_the_reaction_ladders_and_their_comparisons(reactions):
    o2, o1 = reactions
    models = reaction_models(o2)
    assert list(models) == ["B1", "B3", "R1", "B4a", "R2", "B4", "R3", "B5", "R4"]
    assert check_models(models, o2)
    b4, r3 = family_columns(o2, models["B4"]), family_columns(o2, models["R3"])
    assert all(set(b4[k]) <= set(r3[k]) for k in b4) and any(c.startswith("rx_pan_") for c in r3["numeric"])
    assert not any(c.startswith("rx_") for k in b4 for c in b4[k])
    assert reaction_comparisons() == [("B3", "R1"), ("B4a", "R2"), ("B4", "R3"), ("B5", "R4")]
    assert list(reaction_models(o2, counts_only=True)) == ["B1", "B3", "R1c", "B4a", "R2c", "B4", "R3c", "B5", "R4c"]
    assert reaction_comparisons(counts_only=True)[2] == ("B4", "R3c")
    assert list(o1_reaction_models(o1)) == ["C1", "C3", "R-C3", "C4", "R-C4"]
    assert check_models(o1_reaction_models(o1), o1)
    assert o1_reaction_comparisons() == [("C3", "R-C3"), ("C4", "R-C4")]


def test_the_planted_reaction_signal_ranks_the_sharks_of_a_pitch(reactions):
    o2, _ = reactions
    result = evaluate(o2, "ever_offer", models=reaction_models(o2), comparisons=reaction_comparisons(), n_boot=99)
    assert list(result["metrics"].model) == list(reaction_models(o2))
    assert set(result["deltas"].comparison) == {"R1 minus B3", "R2 minus B4a", "R3 minus B4", "R4 minus B5"}
    pred = result["predictions"]
    assert within_pitch_auc(pred, "p_R3")[0] > within_pitch_auc(pred, "p_B4")[0] + 0.1
