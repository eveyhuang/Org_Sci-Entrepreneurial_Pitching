"""The pitch-level heads: the pre-registered ladder C0-C5 (with C2p and C3q) on synthetic runs with planted
structure, and the machinery every head shares. The retired shark-level model sets are tested in
tests/archive/test_shark_level_models.py. Illustrative data, never empirical."""
import numpy as np
import pandas as pd
import pytest

from analysis.panel_bottleneck import (
    BASE_RATE_ADJUSTED, BASELINE, CLIP, CUMULATIVE, CUMULATIVE_OF, INNER_FOLDS, ONE_FAMILY, RAW, adjusted_score, INNER_LOO_MAX, LADDER, LADDER_COMPARISONS, NEAR_CONSTANT_SPREAD, PENALTY_GRID,
    POOLED, SENSITIVITY_LADDERS, STAGE_A_FALLBACK, STAGE_A_FALLBACK_NOTE, base_rate_corr, base_reference, check_models,
    choose_penalty, C1_COLUMNS, LADDER_NOT_FITTED, LADDER_NOT_FITTED_NOTE, ladder_columns, model_coefficients, permutation_test,
    design_matrix, evaluate, family_columns, feature_status, fit_design, inner_folds,
    metric_intervals, metrics_of, paired_deltas, pitch_weights, ridge_logistic, roc_points, weighted_auc,
)
from analysis.panel_dataset import load_run

# a diagnostic only (plan B task 7 review): proposed as a second auc_meaningful rule and not used, because on pitch
# rows the fold base rate takes two values and |Spearman| with it cannot reach this limit (the AUC is adjusted for
# the base rate instead: BASE_RATE_ADJUSTED). Kept here, where it is checked; the module no longer names it.
BASE_RATE_CORR_LIMIT = 0.9


def evaluate_sensitivity(rows: pd.DataFrame, outcome: str, name: str, **kw) -> dict:
    """`evaluate` of the sensitivity ladder `name` (SENSITIVITY_LADDERS) with LADDER_COMPARISONS and without the
    columns of LADDER_NOT_FITTED, as scripts/analysis/analyze_panel.py evaluates a sensitivity; `rows` must already be
    built as the variant says. The result also carries `sensitivity` (the name) and its `note`."""
    variant = SENSITIVITY_LADDERS[name]
    result = evaluate(rows, outcome, models=variant["models"], penalty=variant["penalty"],
                      comparisons=LADDER_COMPARISONS, not_fitted=kw.pop("not_fitted", LADDER_NOT_FITTED), **kw)
    return {**result, "sensitivity": name, "sensitivity_note": variant["note"]}
from analysis.panel_features import FAMILIES, FEATURES, STAGE_A_CODES, _BOOKS, feature_dictionary, pitch_features
from tests.panel_analysis_fixtures import write_run

# Every planted-structure test runs with the default penalty: chosen inside each training fold by
# cross-validation over its pitches. No test hands a head a penalty that suits the planted effect.
BOOT = 199


# ---- preprocessing is fitted on training pitches only ------------------------------------------

def _frame(n=12, **columns):
    """A small feature table: `hist_turns` 1..n and whatever else a test sets."""
    base = {"video_id": [f"v{i // 2}" for i in range(n)], "hist_turns": np.arange(1.0, n + 1)}
    return pd.DataFrame({**base, **columns})


def test_rare_and_unseen_categories_are_pooled_and_unknown_stays_its_own_level():
    # two rows per pitch: food_beverage in 8 pitches, pets in 6, NONE in 2, toys in 2, garden in 2
    train = _frame(40, primary_industry=["food_beverage"] * 16 + ["pets"] * 12 + ["NONE"] * 4 + ["toys"] * 4
                                         + ["garden"] * 4)
    test = _frame(5, primary_industry=["food_beverage", "toys", "never_seen", "NONE", None])
    design = fit_design(train, ("control",), min_category=5)
    assert design.pooled["primary_industry"] == ["garden", "toys"]
    assert design.names == ["primary_industry=NONE", "primary_industry=food_beverage",
                            f"primary_industry={POOLED}", "primary_industry=pets"]
    x_train, x_test, names = design_matrix(train, test, ("control",), min_category=5)
    assert x_train.sum(axis=0).tolist() == [4, 16, 8, 12]
    assert x_test.tolist() == [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
    # with nothing pooled in training there is no pooled column: an unseen level is all zeros
    _, x_unseen, names = design_matrix(train, test, ("control",), min_category=1)
    assert f"primary_industry={POOLED}" not in names and x_unseen[2].sum() == 0


def test_a_category_is_rare_by_its_pitches_not_by_its_rows():
    # toys: 8 rows but one pitch (a panel of eight); pets: 5 rows in 5 pitches; the rest food_beverage
    videos = ["big"] * 8 + [f"p{i}" for i in range(5)] + [f"f{i // 2}" for i in range(12)]
    train = pd.DataFrame({"video_id": videos, "hist_turns": np.arange(1.0, 26),
                          "primary_industry": ["toys"] * 8 + ["pets"] * 5 + ["food_beverage"] * 12})
    design = fit_design(train, ("control",), min_category=5)
    assert design.pooled["primary_industry"] == ["toys"]                 # eight rows, one pitch: rare
    assert "primary_industry=pets" in design.names                       # five rows, five pitches: kept
    assert fit_design(train, ("control",), min_category=6).pooled["primary_industry"] == ["pets", "toys"]
    # a table without the pitch column can only count rows
    assert fit_design(train.drop(columns="video_id"), ("control",), min_category=6).pooled["primary_industry"] == [
        "pets"]


def test_rank_scores_are_used_only_when_every_training_pitch_has_one():
    code = STAGE_A_CODES[0]
    complete = _frame(**{f"sa_rank_{code}": np.linspace(-1, 1, 12)})
    assert f"sa_rank_{code}" in fit_design(complete, ("stage_a",)).names
    partial = complete.copy()
    partial.loc[3, f"sa_rank_{code}"] = np.nan
    design = fit_design(partial, ("stage_a",))
    assert f"sa_rank_{code}" not in design.names and f"sa_rank_{code}" in design.dropped
    # a missing value among the held-out rows is imputed, not a reason to drop
    _, x_test, _ = design_matrix(complete, partial.iloc[[3]], ("stage_a",))
    assert np.isfinite(x_test).all()


# ---- evaluation and metrics ------------------------------------------------------------------

def test_metrics_weight_pitches_equally_and_also_report_rows():
    pred = pd.DataFrame({"video_id": ["a", "a", "a", "a", "b"], "y": [1.0, 1.0, 1.0, 0.0, 0.0],
                         "p_m": [0.8, 0.8, 0.8, 0.6, 0.1]})
    w = pitch_weights(pred.video_id)
    assert w.tolist() == [0.25, 0.25, 0.25, 0.25, 1.0]
    pitch = metrics_of(pred.y, pred.p_m, w)
    loss_a = -(3 * np.log(0.8) + np.log(0.4)) / 4
    loss_b = -np.log(0.9)
    assert pitch["log_loss"] == pytest.approx((loss_a + loss_b) / 2)
    assert pitch["brier"] == pytest.approx(((3 * 0.04 + 0.36) / 4 + 0.01) / 2)
    assert pitch["base_rate"] == pytest.approx((0.75 + 0.0) / 2) and pitch["mean_pred"] == pytest.approx(0.425)
    rows = metrics_of(pred.y, pred.p_m)
    assert rows["log_loss"] == pytest.approx((4 * loss_a + loss_b) / 5) and rows["base_rate"] == pytest.approx(0.6)
    # weighted AUC: positive-negative pairs weigh the product of their row weights
    assert weighted_auc([1, 1, 0, 0], [0.9, 0.2, 0.5, 0.1], [1, 1, 1, 1]) == pytest.approx(0.75)
    assert weighted_auc([1, 1, 0, 0], [0.9, 0.2, 0.5, 0.1], [1, 3, 1, 1]) == pytest.approx((2 + 3) / 8)
    assert np.isnan(weighted_auc([1, 1], [0.3, 0.4], [1, 1])) and np.isnan(metrics_of([0, 0], [0.3, 0.4])["auc"])
    # evaluate reports both weightings
    frame = pd.DataFrame({"video_id": list("aaaabbcc"), "hist_turns": [1.0, 2, 3, 4, 5, 6, 7, 8],
                          "ever_offer": [1.0, 1, 1, 0, 0, 1, 0, 1]})
    result = evaluate(frame, "ever_offer", models={"B0": ()}, n_boot=20)
    p, m = result["predictions"], result["metrics"].iloc[0]
    assert p.weight.tolist() == [0.25] * 4 + [0.5] * 4
    assert m.log_loss == pytest.approx(metrics_of(p.y, p.p_B0, p.weight)["log_loss"])
    assert m.log_loss_rows == pytest.approx(metrics_of(p.y, p.p_B0)["log_loss"]) and m.log_loss != m.log_loss_rows
    assert {"brier_rows", "auc_rows", "mean_pred_rows", "base_rate_rows"} <= set(result["metrics"].columns)


def test_paired_deltas_bootstrap_whole_pitches():
    pred = pd.DataFrame({"video_id": list("aabbccdd"), "y": [1.0, 0, 1, 0, 1, 0, 1, 0],
                         "p_x": [0.5] * 8, "p_y": [0.9, 0.1] * 4})
    deltas = paired_deltas(pred, [("x", "y")], n_boot=200, seed=3).set_index("metric")
    assert deltas.loc["log_loss", "delta"] == pytest.approx(-np.log(0.9) + np.log(0.5))
    # every pitch improves by the same amount, so every draw of pitches gives the same difference
    assert deltas.loc["log_loss", "ci_low"] == pytest.approx(deltas.loc["log_loss", "ci_high"])
    assert deltas.loc["auc", "delta"] == pytest.approx(0.5) and deltas.loc["auc", "n_boot"] == 200
    assert (deltas.comparison == "y minus x").all()


def test_o1_heads_are_at_chance_when_offers_follow_a_draw_no_feature_sees(tmp_path_factory):
    """Offers here depend only on a pitch-level draw unrelated to any feature. A cut that waits for the
    first offer (and for the end of the encounter when there is none) lets the history itself tell the
    two apart; a cut at the first decision of any kind does not."""
    aucs, family_aucs = {"reactions": [], "all": []}, {}
    for seed in (1, 2):
        root = write_run(tmp_path_factory.mktemp(f"null_{seed}"), 60, seed=seed, planted="pitch_draw", strength=0.0)
        o1 = pitch_features(load_run(root))
        assert o1.any_offer.notna().sum() >= 45 and 0.2 < o1.any_offer.mean() < 0.8
        null_models = {"B0": (), "reactions": ("reactions",), "all": LADDER["C5"]}
        for name in ("reactions", "all"):         # each model reads pitch columns, so the test cannot go vacuous
            assert any(family_columns(o1, null_models[name]).values()), name
        assert {c for cols in family_columns(o1, null_models["all"]).values() for c in cols} >= {
            "q_sev_mean", "a_subs_mean", "rx_receipt_share", "nm_receipt_share", "social_turns"}
        result = evaluate(o1, "any_offer", models=null_models, n_boot=20)
        metrics = result["metrics"].set_index("model")
        for name in aucs:
            aucs[name].append(metrics.auc[name])
            # AUC alone can pass for the wrong reason (a head shrunk to a constant scores near 0 under
            # leave-one-out), so the log loss must not beat the base rate either
            assert metrics.log_loss[name] > metrics.log_loss["B0"] - 0.03, (seed, name, metrics.log_loss.to_dict())
        # the ladder the analysis reports (C0-C5 with C2p and C3q): no step beats the base rate
        for family, base in ((LADDER, "C0"),):
            metrics = evaluate(o1, "any_offer", models=family, n_boot=20)["metrics"].set_index("model")
            assert list(metrics.index) == list(family)
            for name in list(family)[1:]:
                assert metrics.log_loss[name] > metrics.log_loss[base] - 0.03, (seed, name, metrics.log_loss.to_dict())
                family_aucs.setdefault(name, []).append(metrics.auc[name])
    assert np.mean(aucs["reactions"]) < 0.65, aucs
    assert np.mean(aucs["all"]) < 0.65, aucs
    # ten models with features: the seven cumulative steps and the three one-family models (amendment 6.9)
    assert len(family_aucs) == len(LADDER) - 1 == 10
    assert all(np.mean(values) < 0.65 for values in family_aucs.values()), family_aucs


# ---- the penalty is chosen inside each training fold -----------------------------------------

def _separable(n_pitches=12, rows=3, seed=0, signal=2.5):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"p{i:02d}" for i in range(n_pitches)], rows)
    x = rng.normal(size=(len(groups), 3))
    y = (rng.random(len(groups)) < 1 / (1 + np.exp(-signal * x[:, 0]))).astype(float)
    return x, y, groups


def test_inner_folds_leave_one_pitch_out_up_to_25_pitches_and_use_five_grouped_folds_beyond():
    assert (INNER_LOO_MAX, INNER_FOLDS, PENALTY_GRID) == (25, 5, (1.0, 10.0, 100.0, 1000.0))
    groups = np.repeat([f"p{i:02d}" for i in range(25)], 2)
    folds = inner_folds(groups)
    assert len(folds) == 25 and all(len(set(groups[f])) == 1 and f.sum() == 2 for f in folds)
    groups = np.repeat([f"p{i:02d}" for i in range(26)], 3)
    folds = inner_folds(groups)
    assert len(folds) == 5 and np.sum(folds, axis=0).tolist() == [1] * len(groups)      # every row held out once
    for f in folds:                                   # whole pitches, never a pitch on both sides
        assert not set(groups[f]) & set(groups[~f]) and len(set(groups[f])) in (5, 6)


def test_choose_penalty_scores_candidates_by_pitch_weighted_inner_log_loss():
    x, y, groups = _separable()
    chosen, losses = choose_penalty(x, y, groups)
    assert set(losses) == set(PENALTY_GRID) and chosen == min(losses, key=losses.get) and chosen < 1000.0
    # the score of one candidate, by hand: leave each pitch out, fit on the rest, weigh pitches equally
    p = np.empty(len(y))
    for g in sorted(set(groups)):
        held = groups == g
        beta = ridge_logistic(x[~held], y[~held], 10.0)
        p[held] = 1 / (1 + np.exp(-(beta[0] + x[held] @ beta[1:])))
    assert losses[10.0] == pytest.approx(metrics_of(y, p, pitch_weights(pd.Series(groups)))["log_loss"])
    # pure noise: nothing to learn, so the heaviest shrinkage wins
    rng = np.random.default_rng(3)
    assert choose_penalty(rng.normal(size=(36, 3)), (rng.random(36) < 0.5).astype(float), groups)[0] >= 100.0
    # a narrower grid is honoured
    assert choose_penalty(x, y, groups, grid=(5.0, 50.0))[0] in (5.0, 50.0)


def test_choose_penalty_gives_ties_to_the_larger_penalty_and_needs_two_pitches():
    groups = np.repeat(["a", "b", "c"], 2)
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    # no column at all: every candidate predicts the same base rates, a four-way tie
    chosen, losses = choose_penalty(np.zeros((6, 0)), y, groups)
    assert len(set(round(v, 12) for v in losses.values())) == 1 and chosen == 1000.0
    # every inner training fold has one class: base rates again, the same for every candidate
    y_split = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    assert choose_penalty(np.arange(6.0).reshape(6, 1), y_split, np.repeat(["a", "b"], 3))[0] == 1000.0
    assert choose_penalty(np.ones((4, 1)), np.array([1.0, 0, 1, 0]), np.array(["a"] * 4)) == (1000.0, {})


# ---- readable metrics ------------------------------------------------------------------------------

def test_a_model_that_hardly_leaves_the_offer_rate_has_an_auc_that_is_not_meaningful():
    rng = np.random.default_rng(5)
    groups = pd.Series(np.repeat([f"p{i:02d}" for i in range(20)], 4))
    y = (rng.random(80) < 0.3).astype(float)
    w = pitch_weights(groups)
    base_p = np.array([y[(groups != g).to_numpy()].mean() for g in groups])
    near = metric_intervals(groups, y, base_p + 0.001 * rng.normal(size=80), base_p, w, n_boot=100)
    assert not near["auc_meaningful"] and near["pred_spread"] < 0.005 and near["auc_low"] <= near["auc_high"]
    informed = np.clip(base_p + 0.3 * (y - 0.3) + 0.05 * rng.normal(size=80), 0.01, 0.99)
    far = metric_intervals(groups, y, informed, base_p, w, n_boot=100)
    assert far["auc_meaningful"] and far["pred_spread"] > NEAR_CONSTANT_SPREAD and far["skill_low"] > 0 and far["auc_low"] > 0.5
    constant = metric_intervals(groups, y, base_p, base_p, w, n_boot=100, ranks=False)
    assert not constant["auc_meaningful"] and np.isnan(constant["auc_low"]) and constant["skill_low"] == constant["skill_high"] == 0.0
    reference = base_reference(groups, y, base_p, w)
    assert reference["corr_with_pitch_rate"] < -0.99 and reference["predicted_rate"] == pytest.approx(float((w * base_p).sum() / w.sum()))


def test_the_base_rate_correlation_is_a_diagnostic_and_does_not_decide_auc_meaningful():
    """`auc_base_rate_corr` is recorded; `auc_meaningful` is the spread rule alone (round 2 of the task 7 review)."""
    rng = np.random.default_rng(3)
    groups = pd.Series(np.repeat([f"p{i:02d}" for i in range(20)], 4))      # four rows a pitch: many base rates
    y = (rng.random(80) < 0.4).astype(float)
    w = pitch_weights(groups)
    base_p = np.array([y[(groups != g).to_numpy()].mean() for g in groups])
    amplified = np.clip(0.4 + 6.0 * (base_p - base_p.mean()), 0.01, 0.99)    # far from the base rate, in its order
    follows = metric_intervals(groups, y, amplified, base_p, w, n_boot=50)
    assert follows["auc_base_rate_corr"] >= BASE_RATE_CORR_LIMIT and follows["pred_spread"] >= NEAR_CONSTANT_SPREAD
    assert follows["auc_meaningful"]                                        # the correlation does not decide it
    assert np.isnan(base_rate_corr(base_p, np.full(80, 0.4))) and base_rate_corr(base_p, base_p) == pytest.approx(1.0)


def test_on_pitch_rows_the_base_rate_correlation_is_bounded_below_the_limit(tmp_path_factory):
    """Why a correlation rule was not used against the leave-one-pitch-out artefact: with one row per pitch the
    fold's base rate (S - y) / (N - 1) takes two values only, so |Spearman| with any prediction is at most
    sqrt(3 n0 n1 / (n^2 - 1)) <= 0.866 < BASE_RATE_CORR_LIMIT and such a rule could never fire. A planted-effect
    model is meaningful, and its correlation is recorded."""
    table = _pitches(tmp_path_factory, planted="prepared", n=80, seed=2)
    result = evaluate(table, "any_offer", n_boot=49, comparisons=LADDER_COMPARISONS)
    m = result["metrics"].set_index("model")
    n, n1 = result["n_groups"], result["n_events"]
    bound = np.sqrt(3 * n1 * (n - n1) / (n ** 2 - 1))
    assert (m.auc_base_rate_corr.drop("C0").abs() <= bound + 1e-9).all() and bound < BASE_RATE_CORR_LIMIT
    assert m.auc_meaningful["C2p"] and m.auc["C2p"] > 0.75 and abs(m.auc_base_rate_corr["C2p"]) < BASE_RATE_CORR_LIMIT


def test_the_ladder_auc_ranks_the_base_rate_adjusted_score_and_an_intercept_model_gets_exactly_one_half(
        tmp_path_factory):
    """Round 2 of the task 7 review: the AUC of `evaluate` ranks logit(p) - logit(b_fold). A pure intercept model
    (C0, and a model fitted to the base rate only) scores 0 everywhere: AUC exactly 0.5 with a degenerate interval.
    The probabilities, log loss, Brier and skill are those of the raw predictions; RAW reproduces the old AUC."""
    table = _pitches(tmp_path_factory, n=24, seed=10)
    models = {"C0": (), "C1": LADDER["C1"], "C5": LADDER["C5"]}
    adjusted = evaluate(table, "any_offer", models=models, comparisons=[("C1", "C5"), ("C0", "C1")], n_boot=49)
    raw = evaluate(table, "any_offer", models=models, comparisons=[("C1", "C5"), ("C0", "C1")], n_boot=49,
                   auc_score=RAW)
    assert adjusted["auc_score"] == BASE_RATE_ADJUSTED and raw["auc_score"] == RAW
    pred, m = adjusted["predictions"], adjusted["metrics"].set_index("model")
    base_p = pred.p_C0.to_numpy()                                   # the intercept-only head predicts b_fold
    for name in models:
        expected = adjusted_score(pred[f"p_{name}"], base_p)
        assert np.allclose(pred[f"score_{name}"], expected)
        assert m.auc[name] == pytest.approx(weighted_auc(pred.y, expected, pred.weight))
    assert (pred.score_C0 == 0).all() and m.auc["C0"] == 0.5 and (m.auc_low["C0"], m.auc_high["C0"]) == (0.5, 0.5)
    assert not m.auc_meaningful["C0"] and list(pred.columns[pred.columns.str.startswith(("p_", "score_"))]) == [
        "p_C0", "score_C0", "p_C1", "score_C1", "p_C5", "score_C5"]
    r = raw["metrics"].set_index("model")
    for column in ("log_loss", "brier", "skill_pct", "skill_low", "skill_high"):
        assert np.allclose(m[column], r[column], equal_nan=True), column
    assert np.isnan(r.auc["C0"]) and "score_C1" not in raw["predictions"]
    d = adjusted["deltas"].set_index(["comparison", "metric"]).delta
    assert d[("C5 minus C1", "auc")] == pytest.approx(m.auc["C5"] - m.auc["C1"])
    assert d[("C1 minus C0", "auc")] == pytest.approx(m.auc["C1"] - 0.5)          # defined against the base rate now
    assert np.isnan(raw["deltas"].set_index(["comparison", "metric"]).delta[("C1 minus C0", "auc")])
    # a fitted model whose predictions equal b_fold is an intercept model too
    assert weighted_auc(pred.y, adjusted_score(base_p, base_p), pred.weight) == 0.5
    assert adjusted_score([0.0, 1.0], [0.5, 0.5])[0] == pytest.approx(np.log(CLIP / (1 - CLIP)))
    with pytest.raises(ValueError, match="auc_score must be one of"):
        evaluate(table, "any_offer", models=models, auc_score="ranks")


def test_on_the_balanced_run_the_near_base_rate_models_no_longer_rank_backwards(tmp_path_factory):
    """The run that showed the artefact (24 pitches, seed 10, strength 2.5: 16 of 23 with an offer): C1 to C3q had a
    raw AUC of 0.00 with interval [0, 0]; adjusted for the base rate they rank near chance, and C3 minus C3q is no
    longer a gain of more than half the AUC scale."""
    root = write_run(tmp_path_factory.mktemp("balanced"), 24, seed=10, planted="own", ranking=True, strength=2.5)
    table = pitch_features(load_run(root))
    adjusted = evaluate(table, "any_offer", n_boot=49, comparisons=LADDER_COMPARISONS)
    raw = evaluate(table, "any_offer", n_boot=49, comparisons=LADDER_COMPARISONS, auc_score=RAW)
    a, r = adjusted["metrics"].set_index("model").auc, raw["metrics"].set_index("model").auc
    near = ["C1", "C2", "C2p", "C3q"]
    assert (r[near] == 0).all() and (a[near] > 0.2).all()
    gain = adjusted["deltas"].set_index(["comparison", "metric"]).delta[("C3 minus C3q", "auc")]
    assert gain < 0.3


def test_roc_points_trace_the_curve_whose_area_is_the_weighted_auc():
    fpr, tpr = roc_points([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1])
    assert list(fpr) == [0, 0, 0.5, 0.5, 1] and list(tpr) == [0, 0.5, 0.5, 1, 1]
    fpr, tpr = roc_points([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])                 # a constant: the diagonal
    assert list(fpr) == [0, 1] and list(tpr) == [0, 1]
    rng = np.random.default_rng(1)
    y, p, w = (rng.random(60) < 0.4).astype(float), np.round(rng.random(60), 1), rng.random(60) + 0.1
    fpr, tpr = roc_points(y, p, w)
    area = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    assert area == pytest.approx(weighted_auc(y, p, w)) and (fpr[0], tpr[0], fpr[-1], tpr[-1]) == (0, 0, 1, 1)
    assert len(roc_points([1, 1], [0.2, 0.3])[0]) == 0


# ---- the pre-registered pitch-level ladder (refocus spec section 6) --------------------------------

def _pitches(tmp_path_factory, planted=None, n=24, seed=1, ranking=True):
    root = write_run(tmp_path_factory.mktemp(f"pitch_{planted}_{n}_{seed}"), n, seed=seed, planted=planted,
                     ranking=ranking)
    return pitch_features(load_run(root))


def test_ladder_steps_add_one_family_each_and_c2_reads_the_primary_rank_score():
    assert list(LADDER) == ["C0", "C1", "C2", "C2p", "C3q", "C3", "C4", "C5", "C3a", "C4r", "C5r"]
    assert list(CUMULATIVE) == list(LADDER)[:8]
    assert _BOOKS.stage_a_ids()[0] == STAGE_A_CODES[0] == "pitch_positivity"        # the book's primary item
    assert LADDER["C0"] == () and LADDER["C1"] == ("control", "exposure")
    assert LADDER["C2"] == LADDER["C1"] + ("sa_rank_pitch_positivity",)
    added = {"C2p": "prepared", "C3q": "questions", "C3": "answers", "C4": "reactions", "C5": "rapport_repair"}
    names = list(CUMULATIVE)
    for earlier, later in zip(names[2:-1], names[3:]):
        assert LADDER[later] == LADDER[earlier] + (added[later],), later
    # the one-family models (amendment 6.9): the baseline plus one family alone, each paired with the baseline
    assert BASELINE == "C2p" and ONE_FAMILY == {"C3q": "questions", "C3a": "answers", "C4r": "reactions",
                                                "C5r": "rapport_repair"}
    for model, family in ONE_FAMILY.items():
        assert LADDER[model] == LADDER[BASELINE] + (family,), model
        assert (BASELINE, model) in LADDER_COMPARISONS and (BASELINE, CUMULATIVE_OF[model]) in LADDER_COMPARISONS
    # every family of the pitch rows is in the full model, and only the primary Stage A item
    assert set(LADDER["C5"]) - {"sa_rank_pitch_positivity"} == set(FAMILIES) - {"stage_a"}
    assert LADDER_COMPARISONS == [("C1", "C2"), ("C2", "C2p"), ("C2p", "C3"), ("C3q", "C3"), ("C3", "C4"),
                                  ("C4", "C5"), ("C2p", "C3q"), ("C2p", "C3a"), ("C2p", "C4r"), ("C2p", "C5r"),
                                  ("C2p", "C4"), ("C2p", "C5")]
    assert check_models(LADDER) == LADDER


def test_c2_falls_back_to_the_rating_with_a_note_when_no_rank_scores(tmp_path_factory):
    plain = _pitches(tmp_path_factory, n=8, seed=3, ranking=False)
    assert not [c for c in plain.columns if c.startswith("sa_rank_")] and "sa_pitch_positivity" in plain
    assert STAGE_A_FALLBACK["sa_rank_pitch_positivity"] == "sa_pitch_positivity"
    result = evaluate(plain, "any_offer", n_boot=19, penalty=10.0)
    metrics = result["metrics"].set_index("model")
    assert list(metrics.index) == list(LADDER)
    assert STAGE_A_FALLBACK_NOTE == "rank score absent; rating used" and (metrics.note == STAGE_A_FALLBACK_NOTE).all()
    assert metrics.families["C2"] == "control+exposure+sa_pitch_positivity"
    for fold in result["folds"]:
        assert "sa_pitch_positivity" in fold["models"]["C2"]["columns"] + fold["models"]["C2"]["dropped"]
    # with the rank scores the rank score is read and nothing is noted
    ranked = _pitches(tmp_path_factory, n=8, seed=3, ranking=True)
    metrics = evaluate(ranked, "any_offer", n_boot=19, penalty=10.0)["metrics"].set_index("model")
    assert metrics.families["C2"] == "control+exposure+sa_rank_pitch_positivity" and (metrics.note == "").all()


def test_the_ladder_is_evaluated_with_its_comparisons_and_skill_differences(tmp_path_factory):
    table = _pitches(tmp_path_factory, n=8, seed=3)
    result = evaluate(table, "any_offer", n_boot=19, penalty=10.0)
    deltas = result["deltas"]
    assert list(dict.fromkeys(deltas.comparison)) == [f"{b} minus {a}" for a, b in LADDER_COMPARISONS]
    assert set(deltas.metric) == {"log_loss", "brier", "auc", "skill"}
    metrics = result["metrics"].set_index("model")
    skill = deltas[deltas.metric == "skill"].set_index("comparison")
    assert skill.delta["C3 minus C2p"] == pytest.approx(metrics.skill_pct["C3"] - metrics.skill_pct["C2p"])
    assert skill.better["C3 minus C2p"] == "positive" and (skill.ci_low <= skill.ci_high).all()
    # a skill difference has the sign of the log-loss improvement in every draw
    loss = deltas[deltas.metric == "log_loss"].set_index("comparison")
    assert (np.sign(skill.delta) == -np.sign(loss.delta)).all()


def test_the_sensitivity_ladders_are_named_variants_of_the_same_ladder(tmp_path_factory):
    assert list(SENSITIVITY_LADDERS) == ["rates", "all_stage_a", "fixed_penalty_10"]
    assert SENSITIVITY_LADDERS["rates"]["models"] == LADDER and SENSITIVITY_LADDERS["rates"]["features"] == {
        "exposure": "rates"}
    assert SENSITIVITY_LADDERS["fixed_penalty_10"]["penalty"] == 10.0
    every = SENSITIVITY_LADDERS["all_stage_a"]["models"]
    assert list(every) == list(LADDER) and every["C2"] == LADDER["C1"] + tuple(f"sa_rank_{c}" for c in STAGE_A_CODES)
    assert every["C5"][len(every["C2"]):] == LADDER["C5"][len(LADDER["C2"]):]
    table = _pitches(tmp_path_factory, n=8, seed=3)
    fixed = evaluate_sensitivity(table, "any_offer", "fixed_penalty_10", n_boot=19)
    assert fixed["sensitivity"] == "fixed_penalty_10" and fixed["penalty"]["value"] == 10.0
    assert list(fixed["metrics"].model) == list(LADDER)
    wide = evaluate_sensitivity(table, "any_offer", "all_stage_a", n_boot=19)
    assert wide["metrics"].set_index("model").families["C2"].count("sa_rank_") == len(STAGE_A_CODES)
    assert list(dict.fromkeys(wide["deltas"].comparison)) == [f"{b} minus {a}" for a, b in LADDER_COMPARISONS]


def test_the_ladder_reads_only_declared_pitch_columns_and_each_step_adds_its_family(tmp_path_factory):
    o1 = _pitches(tmp_path_factory, n=8, seed=3)
    names = list(CUMULATIVE)
    added = {}
    for earlier, later in zip(names[:-1], names[1:]):
        before = {c for cols in family_columns(o1, LADDER[earlier]).values() for c in cols}
        after = {c for cols in family_columns(o1, LADDER[later]).values() for c in cols}
        assert before < after, later
        added[later] = after - before
    # each one-family model reads the baseline's columns plus its family's, and nothing else (amendment 6.9)
    base = {c for cols in family_columns(o1, LADDER[BASELINE]).values() for c in cols}
    for model, family in ONE_FAMILY.items():
        own = {c for cols in family_columns(o1, LADDER[model]).values() for c in cols} - base
        assert own and {FEATURES[c]["family"] for c in own} == {family}, model
    for column in set().union(*added.values()):
        assert FEATURES[column]["unit"] in ("o1", "both") and not column.startswith(("own_", "oth_")), column
    assert added["C2"] == {"sa_rank_pitch_positivity"}
    for model, family in (("C2p", "prepared"), ("C3q", "questions"), ("C3", "answers"), ("C4", "reactions"),
                          ("C5", "rapport_repair")):
        assert {FEATURES[c]["family"] for c in added[model]} == {family}, model
    assert {"q_askers", "q_deal_terms_share", "q_sev_mean"} <= added["C3q"] and "a_subs_mean" in added["C3"]
    assert {"accepts_turns", "challenge_turns", "nm_receipt_share", "rx_receipt_share"} <= added["C4"]
    assert {"social_turns", "founder_social", "founder_repairs"} <= added["C5"]
    assert "hist_questions" in added["C1"] and not [c for c in added["C3q"] if c == "hist_questions"]


def test_the_ladder_runs_through_the_evaluation_on_pitch_rows(tmp_path_factory):
    o1 = _pitches(tmp_path_factory, n=8, seed=3)
    known = o1[o1.any_offer.notna()]
    result = evaluate(o1, "any_offer", n_boot=49)
    names = list(LADDER)
    assert result["status"] == "estimated" and result["n_rows"] == result["n_groups"] == len(known)
    assert result["n_events"] == int(known.any_offer.sum()) and result["penalty"]["mode"] == "inner_cv"
    metrics = result["metrics"].set_index("model")
    assert list(metrics.index) == names and np.isfinite(metrics.log_loss).all() and (metrics.role == "main").all()
    assert metrics.auc["C0"] == 0.5                                      # the base rate ranks nothing: all ties
    pred = result["predictions"]
    assert list(pred.video_id) == sorted(known.video_id) and (pred.weight == 1.0).all() and "investor" not in pred
    for fold in result["folds"][:3]:          # one pitch held out at a time: the base rate is that of the others
        rest = known[known.video_id != fold["held_out"]]
        assert fold["n_test"] == 1 and fold["n_train"] == len(rest)
        assert pred.loc[pred.video_id == fold["held_out"], "p_C0"].iloc[0] == pytest.approx(rest.any_offer.mean())
    logloss = result["deltas"][result["deltas"].metric == "log_loss"].set_index("comparison")
    assert logloss.delta["C3 minus C2p"] == pytest.approx(metrics.log_loss["C3"] - metrics.log_loss["C2p"])
    for fold in result["folds"]:
        used = {c.split("=")[0] for c in fold["models"]["C5"]["columns"]}
        assert not [c for c in used if c.startswith(("own_", "oth_"))] and "any_offer" not in used


def test_every_pitch_feature_has_the_status_of_what_the_ladder_reads():
    dictionary = feature_dictionary().set_index("name")
    for name in ("hist_turns", "ask_log_amount", "customer_type", "q_deal_terms_share", "a_subs_mean", "a_resp_mean",
                 "a_ctx_mean_missing", "nm_receipt_share", "social_turns", "sa_rank_pitch_positivity",
                 "sa_pitch_positivity"):
        assert feature_status(name) == "primary" == dictionary.status[name], name
    others = [f"sa_rank_{code}" for code in STAGE_A_CODES[1:]] + [f"sa_{code}" for code in STAGE_A_CODES[1:]]
    for name in [*others, "sa_pitch_positivity_missing"]:       # read by the all-Stage-A sensitivity only
        assert feature_status(name) == "exploratory" == dictionary.status[name], name
    assert feature_status("video_id") is None and dictionary.status["cut_pos"] is None
    # ruling I4: what the ladder never fits is a descriptive column (the archive keeps the family status)
    for name in ("q_sev_mean_n", "a_subs_mean_n", "a_frame_match_share", "a_frame_match_share_missing",
                 "demo_seconds", "prepared_seconds"):
        assert feature_status(name) == "descriptive" == dictionary.status[name], name
        assert feature_status(name, ladder_exclusion=False) == "primary", name
    assert set(dictionary.index[dictionary.status == "descriptive"]) == set(LADDER_NOT_FITTED)
    assert set(dictionary.status.dropna()) == {"primary", "exploratory", "descriptive"}
    assert {dictionary.family[n] for n in dictionary.index[dictionary.status == "exploratory"]} == {"stage_a"}


SPEC_C1 = ["founder_gender", "primary_industry", "product_form", "customer_type", "ask_log_amount", "ask_equity_pct",
           "ask_missing", "hist_turns", "hist_seconds", "hist_questions"]


def _no_count_twin_or_frame_match(columns, where):
    bad = [c for c in columns if c.split("=")[0].endswith("_n") or "a_frame_match_share" in c
           or c.split("=")[0] == "rx_answers"]
    assert not bad, (where, bad)


def test_ladder_designs_follow_the_spec_table_c1_exactly_and_no_count_twin(tmp_path_factory):
    """Ruling I4 (plan B final review): C1 is the spec's list; no `_n` count twin and no a_frame_match_share in any
    ladder design, main or sensitivity; the columns stay in the feature table (descriptive), never fitted."""
    assert list(C1_COLUMNS) == SPEC_C1
    assert {"demo_seconds", "prepared_seconds", "a_frame_match_share", "rx_answers"} <= set(LADDER_NOT_FITTED)
    assert "rx_answers" in LADDER_NOT_FITTED_NOTE                             # a denominator, not a share (pre-run)
    assert "n_social_exchanges" not in FEATURES                               # dropped in plan B2 (archive only)
    assert {n for n in FEATURES if n.endswith("_n") and n[:-2] in FEATURES} <= set(LADDER_NOT_FITTED)
    assert not [n for n in LADDER_NOT_FITTED if n.endswith("_missing") and n != "a_frame_match_share_missing"]
    o1 = _pitches(tmp_path_factory, n=8, seed=3)
    assert set(LADDER_NOT_FITTED) <= set(o1.columns)                         # kept in the table, not deleted
    assert ladder_columns(o1, LADDER["C1"]) == SPEC_C1
    for name, spec in LADDER.items():
        _no_count_twin_or_frame_match(ladder_columns(o1, spec), name)
    result = evaluate(o1, "any_offer", n_boot=19, penalty=10.0)              # the default ladder leaves them out
    assert result["design_columns"]["C1"] == SPEC_C1 and result["not_fitted"] == list(LADDER_NOT_FITTED)
    assert result["design_columns"]["C0"] == [] and set(result["design_columns"]) == set(LADDER)
    for fold in result["folds"]:
        assert {c.split("=")[0] for c in fold["models"]["C1"]["columns"]} <= set(SPEC_C1)
        for name, detail in fold["models"].items():
            _no_count_twin_or_frame_match(detail["columns"], (fold["held_out"], name))
    for variant in SENSITIVITY_LADDERS:
        sens = evaluate_sensitivity(o1, "any_offer", variant, n_boot=19)
        assert sens["design_columns"]["C1"] == SPEC_C1, variant
        for name, columns in sens["design_columns"].items():
            _no_count_twin_or_frame_match(columns, (variant, name))
    permuted = permutation_test(o1, "any_offer", result, n_perm=5)          # refits the same designs
    assert len(permuted) == len(LADDER) - 1
    coefficients = model_coefficients(o1, "any_offer", LADDER["C5"], 10.0, model="C5", n_boot=9,
                                      not_fitted=LADDER_NOT_FITTED)
    _no_count_twin_or_frame_match(list(coefficients.feature), "coefficients")
    # models given by the caller fit what they name: the archived and legacy paths are unchanged
    given = evaluate(o1, "any_offer", models={"C1": LADDER["C1"]}, n_boot=19, penalty=10.0)
    assert given["not_fitted"] == [] and "demo_seconds" in given["design_columns"]["C1"]


def test_c2p_reads_the_prepared_ratings_and_c3q_the_questions_in_every_fold(tmp_path_factory):
    """A regression guard: the steps that test H1' and H8 read the families they name, and the step before does not."""
    table = _pitches(tmp_path_factory, n=8, seed=3)
    result = evaluate(table, "any_offer", n_boot=9, penalty=10.0)

    def read(detail):
        return {c.split("=", 1)[0] for c in detail["columns"] + detail["dropped"]}
    for fold in result["folds"]:
        c2, c2p, c3q = (read(fold["models"][m]) for m in ("C2", "C2p", "C3q"))
        assert {"pres_subs", "pres_coh", "pres_ctx", "pres_evid"} <= c2p and not [c for c in c2 if c.startswith("pres_")]
        assert {"q_askers", "q_sev_mean"} <= c3q and not [c for c in c2p if c.startswith("q_")]


def test_planted_prepared_effect_shows_in_c2p_minus_c2(tmp_path_factory):
    """planted="prepared": the offers depend on the prepared presentation's substance beyond the pitch's appeal. Over
    several runs of 80 pitches, C2p's skill over C2 is positive on every one and larger on average than in the
    unplanted twin of the same seed (offers from the appeal alone). Skill, not AUC: C2 is often a near-constant head
    whose AUC is not meaningful. Takes about 30 seconds."""
    models = {name: LADDER[name] for name in ("C0", "C1", "C2", "C2p")}

    def skill_gain(planted, seed):
        table = _pitches(tmp_path_factory, planted=planted, n=80, seed=seed)
        deltas = evaluate(table, "any_offer", models=models, comparisons=[("C2", "C2p")], n_boot=199)["deltas"]
        return float(deltas.set_index("metric").delta["skill"])
    planted = [skill_gain("prepared", seed) for seed in PREPARED_SEEDS]
    twins = [skill_gain(None, seed) for seed in PREPARED_SEEDS]
    assert all(gain > 0 for gain in planted), planted
    assert np.mean(planted) > np.mean(twins), (planted, twins)


PREPARED_SEEDS = (1, 2, 3, 4)
