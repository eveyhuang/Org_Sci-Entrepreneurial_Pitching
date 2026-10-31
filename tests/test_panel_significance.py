"""Significance in the panel analysis: bootstrap p-values, the pitch-level permutation test, FDR-adjusted
single-feature tables, and the coefficients of the full model. Synthetic runs only; nothing empirical."""
import numpy as np
import pandas as pd
import pytest

from analysis import panel_report as pr
from analysis import pitch_figures as pfig
from analysis import panel_bottleneck as pb
from analysis.panel_bottleneck import (
    BY_OFFER_COLUMNS, COEFFICIENT_COLUMNS, DELTA_COLUMNS, FAMILY_SUMMARY_COLUMNS, NOTE_COUNT, PERMUTATION_COLUMNS,
    SINGLE_COLUMNS, _ridge_logistic_many, add_permutation_p, bh_q, boot_p, modal_penalty, permutation_orders,
    permutation_p_corr, ridge_logistic,
)
# the investor rows are read by the archived shark-level forms of the same functions (plan B task 3)
from analysis.archive.shark_level_models import (
    MODEL_FAMILY, answer_quality_by_offer, evaluate, family_summary, fit_design, model_coefficients, permutation_test,
    removal_ranking, single_features,
)
from analysis.panel_dataset import load_run
from analysis.archive.shark_level_features import o2_features
from analysis.panel_features import pitch_features
from tests.panel_analysis_fixtures import write_run

SMALL = {"B0": (), "B1": MODEL_FAMILY["B1"], "B3": MODEL_FAMILY["B3"]}


def _tables(tmp_path_factory, planted, n=30, seed=1):
    data = load_run(write_run(tmp_path_factory.mktemp(f"sig_{planted}"), n, seed=seed, planted=planted, strength=4.0))
    return o2_features(data), pitch_features(data)


@pytest.fixture(scope="module")
def own(tmp_path_factory):
    return _tables(tmp_path_factory, "own")


@pytest.fixture(scope="module")
def null(tmp_path_factory):
    return _tables(tmp_path_factory, "pitch_draw")


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    """The pitch-level analysis (analyze_panel.compute through the notebook's panel_report.compute), quick settings."""
    root = write_run(tmp_path_factory.mktemp("sig_analysis"), 30, seed=1, planted="own", strength=4.0)
    return pr.compute(root, None, 10, 49, permutations=19, single_draws=199, run_h9=False)


# ---- the two small formulas ------------------------------------------------------------------------

def test_boot_p_is_twice_the_smaller_tail_floored_at_one_over_draws_plus_one():
    draws = np.arange(1, 100, dtype=float)                           # 99 draws, all above zero
    assert boot_p(draws) == pytest.approx(1 / 100)                   # never 0: the floor
    assert boot_p(-draws) == pytest.approx(1 / 100)
    assert boot_p(np.r_[-np.ones(10), np.ones(90)]) == pytest.approx(0.2)        # 10% on the other side, doubled
    assert boot_p(np.r_[-np.ones(50), np.ones(50)]) == 1.0
    assert boot_p(np.zeros(20)) == 1.0                               # an estimate that never leaves the null
    assert boot_p(draws, null=50.0) == pytest.approx(1.0) and boot_p(draws, null=5.0) == pytest.approx(2 * 5 / 99)
    assert np.isnan(boot_p([])) and np.isnan(boot_p([np.nan, np.nan]))
    assert boot_p([np.nan, 1.0, 2.0, 3.0]) == pytest.approx(1 / 4)   # missing draws are left out


def test_bh_q_values_follow_the_benjamini_hochberg_rule():
    q = bh_q([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(q, [0.02, 0.04, 0.04, 0.02])                  # p x m / rank, made monotone from the largest down
    with_missing = bh_q([0.01, np.nan, 0.04, 0.03, 0.005])
    assert np.isnan(with_missing[1]) and np.allclose(np.delete(with_missing, 1), q)      # a missing p is no test
    assert np.allclose(bh_q([0.5, 0.9]), [0.9, 0.9]) and bh_q([0.9, 0.99]).max() <= 1.0
    assert len(bh_q([])) == 0
    p = np.random.default_rng(0).random(50)
    assert (bh_q(p) >= p - 1e-12).all() and (np.diff(bh_q(p)[np.argsort(p)]) >= -1e-12).all()


def test_every_interval_of_the_model_tables_has_a_bootstrap_p_value(own):
    result = evaluate(own[0], "ever_offer", penalty=10.0, models=SMALL, n_boot=199)
    m, d = result["metrics"].set_index("model"), result["deltas"]
    assert list(d.columns) == DELTA_COLUMNS and "p_value" in DELTA_COLUMNS
    assert np.isnan(m.skill_p_value["B0"]) and np.isnan(m.auc_p_value["B0"])       # nothing to test in the base rate
    for name in ("B1", "B3"):
        assert 1 / 200 <= m.skill_p_value[name] <= 1 and 1 / 200 <= m.auc_p_value[name] <= 1
    # an interval that excludes the null goes with p below 0.05, one that straddles it widely does not
    for r in d[d.metric == "log_loss"].itertuples():
        if r.ci_low > 0 or r.ci_high < 0:
            assert r.p_value < 0.05
        else:
            assert r.p_value >= 0.05 - 2 / 199
    assert d[(d.metric == "auc") & d.comparison.str.contains("B0")].p_value.isna().all()
    assert m.skill_p_value["B3"] < 0.05 < m.skill_p_value["B1"]                   # the planted family, and controls only
    ranking = removal_ranking(own[0], "ever_offer", penalty=10.0, n_boot=99)
    assert "p_value" in ranking.columns and ranking.p_value.between(1 / 100, 1).all()


# ---- the permutation test --------------------------------------------------------------------------

def test_permutation_orders_move_whole_pitches_between_pitches_of_the_same_size():
    groups = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 2 + ["d"] * 3 + ["e"] * 2 + ["f"])
    orders = permutation_orders(groups, 200, seed=3)
    assert orders.shape == (200, len(groups))
    assert (np.sort(orders, axis=1) == np.arange(len(groups))).all()               # every row is used exactly once
    moved = 0
    for order in orders:
        for g in set(groups):
            sources = set(groups[order[groups == g]])
            assert len(sources) == 1                                               # a pitch's outcomes travel together
            source = sources.pop()
            assert (groups == source).sum() == (groups == g).sum()                 # ... to a pitch of the same size
            moved += source != g
    assert moved > 200                                                              # and they do move
    assert (orders[:, -1] == len(groups) - 1).all()                                 # the only pitch of its size stays
    assert len({tuple(o[:3]) for o in orders}) > 6                                  # the order within a set is shuffled too
    assert (permutation_orders(groups, 200, seed=3) == orders).all()
    assert not (permutation_orders(groups, 200, seed=4) == orders).all()
    pitches = np.array([f"p{i}" for i in range(12)])                                # pitch rows: a plain shuffle
    plain = permutation_orders(pitches, 50, seed=0)
    assert (np.sort(plain, axis=1) == np.arange(12)).all() and len({tuple(o) for o in plain}) == 50


def test_many_fits_at_once_equal_the_single_fit(own):
    known = own[0][own[0].ever_offer.notna()].reset_index(drop=True)
    x = fit_design(known, MODEL_FAMILY["B3"]).transform(known)
    y = known.ever_offer.to_numpy(dtype=float)
    targets = np.vstack([y, y[::-1], np.ones(len(y)), np.r_[1.0, np.zeros(len(y) - 1)]])
    for penalty in (1.0, 100.0):
        many = _ridge_logistic_many(x, targets, penalty)
        assert np.allclose(many[0], ridge_logistic(x, y, penalty), atol=1e-5)
        assert np.allclose(many[1], ridge_logistic(x, y[::-1], penalty), atol=1e-5)
        assert np.isnan(many[2]).all() and not np.isnan(many[3]).any()             # one class only: no fit


def test_permutation_test_refits_every_fold_and_finds_the_planted_family_but_nothing_in_a_null_run(own, null):
    result = evaluate(own[0], "ever_offer", penalty=10.0, models=SMALL, n_boot=19)
    table = permutation_test(own[0], "ever_offer", result, n_perm=39, seed=5)
    assert list(table.columns) == PERMUTATION_COLUMNS and list(table.model) == ["B1", "B3"]     # models with features
    m = result["metrics"].set_index("model")
    by = table.set_index("model")
    for name in ("B1", "B3"):
        # with the penalty fixed, the observed statistic of the test is the run's own leave-one-pitch-out result
        assert by.log_loss[name] == pytest.approx(m.log_loss[name], abs=1e-6)
        assert by.log_loss_inner_cv[name] == m.log_loss[name]
        assert by.auc[name] == pytest.approx(m.auc[name], abs=1e-4)
        assert by.penalty[name] == 10.0 == modal_penalty(result, name) and by.n_perm[name] == 39
        for column in ("p_log_loss", "p_auc"):
            assert by[column][name] * 40 == pytest.approx(round(by[column][name] * 40)) and 1 / 40 <= by[column][name] <= 1
    assert by.p_log_loss["B3"] == 1 / 40 and by.p_auc["B3"] == 1 / 40              # better than every permutation
    assert by.perm_log_loss_mean["B3"] > by.log_loss["B3"]
    pd.testing.assert_frame_equal(table, permutation_test(own[0], "ever_offer", result, n_perm=39, seed=5))
    assert modal_penalty(result, "B0") is None
    # nothing before the decision predicts the outcome of a null run: the full model does not beat permuted data
    flat = evaluate(null[0], "ever_offer", penalty=10.0, models=SMALL, n_boot=19)
    assert permutation_test(null[0], "ever_offer", flat, n_perm=39, seed=5).set_index("model").p_log_loss["B3"] > 0.05
    # pitch rows, the penalty chosen inside the folds: the modal choice is what the test fixes
    o1 = pb.evaluate(own[1], "any_offer", models={"C0": (), "C2": ("control", "exposure", "stage_a")}, n_boot=19)
    pitch = pb.permutation_test(own[1], "any_offer", o1, n_perm=19, seed=1)
    assert list(pitch.model) == ["C2"] and pitch.penalty[0] == modal_penalty(o1, "C2") and pitch.n_rows[0] == pitch.n_groups[0]
    # not run, or nothing estimated: an empty table with the columns
    assert permutation_test(own[0], "ever_offer", result, n_perm=0).empty
    one = evaluate(own[0][own[0].video_id == "v000"], "ever_offer", models=SMALL, n_boot=9)
    assert list(permutation_test(own[0], "ever_offer", one, n_perm=9).columns) == PERMUTATION_COLUMNS
    with_p = add_permutation_p(result, table)["metrics"].set_index("model")
    assert np.isnan(with_p.perm_p_log_loss["B0"]) and with_p.perm_p_log_loss["B3"] == 1 / 40 and with_p.n_perm["B3"] == 39
    assert add_permutation_p(flat, table.iloc[:0])["metrics"].perm_p_auc.isna().all()


def test_permutation_p_of_a_correlation_counts_permutations_at_least_as_large():
    rng = np.random.default_rng(0)
    groups = np.repeat([f"p{i:02d}" for i in range(40)], 4)
    y = (rng.random(160) < 0.4).astype(float)
    permuted = y[permutation_orders(groups, 499, seed=1)]
    assert permutation_p_corr(y + rng.normal(0, 0.3, 160), y, permuted) == 1 / 500       # a strong association
    assert permutation_p_corr(rng.normal(size=160), y, permuted) > 0.05                  # none
    assert permutation_p_corr(-(y + rng.normal(0, 0.3, 160)), y, permuted) == 1 / 500    # two-sided
    assert np.isnan(permutation_p_corr(np.ones(160), y, permuted)) and np.isnan(permutation_p_corr(y, y, permuted[:0]))
    # a feature that is non-zero in two rows cannot reach a small p, whatever a bootstrap of it would say
    sparse = np.zeros(160)
    sparse[np.flatnonzero(y == 0)[:2]] = 1.0
    assert permutation_p_corr(sparse, y, permuted) > 0.2


# ---- single features, by offer, families, coefficients ---------------------------------------------

def test_single_feature_and_by_offer_tables_carry_permutation_p_and_fdr_q(own):
    table = single_features(own[0], "ever_offer", n_boot=499, seed=2)
    assert list(table.columns) == SINGLE_COLUMNS and (table.n_perm == 499).all()
    assert table.p_value.between(1 / 500, 1).all() and table.p_boot.between(1 / 500, 1).all()
    assert np.allclose(table.q_value, bh_q(table.p_value)) and (table.q_value >= table.p_value - 1e-12).all()
    assert list(table.significant_fdr) == list(table.q_value < 0.05) and table.significant_fdr.dtype == bool
    assert table.gain_p_value.between(1 / 500, 1).all() and table.adj_p_value.dropna().between(1 / 500, 1).all()
    planted = table[table.feature.isin(("own_resp_mean", "own_subs_mean", "own_evid_share"))]
    assert planted.significant_fdr.any() and table.significant_fdr.sum() < len(table)      # found, and not everything is
    assert 0 < (table.p_value < 0.05).sum() >= table.significant_fdr.sum()
    by_offer = answer_quality_by_offer(own[0], "ever_offer", n_boot=199)
    assert list(by_offer.columns) == BY_OFFER_COLUMNS
    known = by_offer[by_offer.correlation.notna()]
    assert np.allclose(known.q_value, bh_q(by_offer.p_value)[by_offer.correlation.notna()])
    assert known.p_value.between(1 / 200, 1).all() and list(known.significant_fdr) == list(known.q_value < 0.05)
    # per family: what was ranked, what survives, without the counts behind a mean
    summary = family_summary(table)
    assert list(summary.columns) == FAMILY_SUMMARY_COLUMNS
    kept = table[table.note != NOTE_COUNT]
    assert summary.features.sum() == len(kept) and summary.survive_fdr.sum() == int(kept.significant_fdr.sum())
    own_row = summary.set_index("family").loc["own"]
    part = kept[kept.family == "own"]
    assert own_row.p_below_05 == int((part.p_value < 0.05).sum())
    assert own_row.median_abs_correlation == pytest.approx(part.correlation.abs().median())
    assert own_row.largest_feature == part.loc[part.correlation.abs().idxmax()].feature
    assert family_summary(table, counts=True).features.sum() == len(table)


def test_model_coefficients_are_the_full_fit_with_pitch_bootstrap_intervals(own):
    spec = MODEL_FAMILY["B3"]
    table = model_coefficients(own[0], "ever_offer", spec, 10.0, model="B3", n_boot=99, seed=4)
    assert list(table.columns) == COEFFICIENT_COLUMNS and (table.model == "B3").all() and (table.penalty == 10.0).all()
    known = own[0][own[0].ever_offer.notna()].reset_index(drop=True)
    design = fit_design(known, spec)
    beta = ridge_logistic(design.transform(known), known.ever_offer.to_numpy(dtype=float), 10.0)
    assert dict(zip(table.feature, table.coefficient)) == pytest.approx(dict(zip(design.names, beta[1:])))
    assert list(table.coefficient.abs()) == sorted(table.coefficient.abs(), reverse=True)        # largest first
    assert (table.ci_low <= table.ci_high).all() and table.p_value.between(1 / 100, 1).all() and (table.n_boot == 99).all()
    inside = ((table.ci_low <= table.coefficient) & (table.coefficient <= table.ci_high)).mean()
    assert inside > 0.9                                                    # a percentile interval around the estimate
    top = table.iloc[0]
    assert top.family == "own" and top.p_value < 0.05                      # the planted family carries the model
    category = table[table.feature.str.contains("=")].iloc[0]
    assert " = " in category.label and category.family == "control"
    assert table.n_rows.iloc[0] == len(known) and table.n_groups.iloc[0] == known.video_id.nunique()
    assert model_coefficients(own[0], "ever_offer", (), 10.0).empty
    assert model_coefficients(own[0][own[0].ever_offer == 1], "ever_offer", spec, 10.0).empty   # the outcome must vary


# ---- the analysis and what the notebook shows ------------------------------------------------------

def test_the_analysis_runs_the_permutation_test_for_every_model_of_the_ladder(analysis):
    a = analysis
    assert a.n_perm == 19 and a.single_draws == 199
    m = a.result["metrics"].set_index("model")
    with_features = [name for name in m.index if m.families[name] != "(base rate)"]
    assert list(a.permutation.model) == with_features and (a.permutation.n_perm == 19).all()
    part = a.permutation.set_index("model")
    for name in with_features:
        assert m.perm_p_log_loss[name] == part.p_log_loss[name] and m.perm_p_auc[name] == part.p_auc[name]
    assert (a.single.n_perm == 199).all() and (a.by_offer.n_boot == 199).all()
    assert (a.coefficients.model == "C5").all() and a.coefficients.penalty.iloc[0] == 10.0
    assert (a.coefficients.n_boot <= 49).all()
    assert "p-value" in a.ranking_agreement.columns and "p_value" in a.h4_shares.columns
    assert a.h4_shares.p_value.between(1 / 50, 1).all()


def test_without_permutations_the_columns_are_empty(analysis):
    quick = pr.compute(analysis.run_root, None, 10, 19, permutations=0, single_draws=19, run_h9=False)
    metrics = quick.result["metrics"]
    assert quick.n_perm == 0 and quick.permutation.empty
    assert "perm_p_log_loss" not in metrics or metrics.perm_p_log_loss.isna().all()
    assert pr.metrics_table(quick.result)[pr.PERM_LOSS_HEADER].eq("").all()
    from analysis import pitch_report

    assert pitch_report.p_kind(quick.result) == "bootstrap" and pitch_report.p_kind(analysis.result) == "permutation"
    row = quick.result["metrics"].set_index("model").loc["C3"]
    assert pitch_report.model_p(row) == (row.skill_p_value, "bootstrap")
    assert pitch_report.model_p(row, "auc") == (row.auc_p_value, "bootstrap")
    tested = analysis.result["metrics"].set_index("model").loc["C3"]
    assert pitch_report.model_p(tested) == (tested.perm_p_log_loss, "permutation")


def test_p_values_are_written_the_same_way_everywhere():
    assert [pr.p_text(p) for p in (0.0004, 0.001, 0.0123, 0.05, 0.0996, 0.1, 0.314, 1.0, np.nan, None)] == [
        "<0.001", "0.001", "0.012", "0.050", "0.100", "0.10", "0.31", "1.00", "", ""]
    assert [pr.stars(p) for p in (0.0004, 0.001, 0.009, 0.01, 0.049, 0.05, 0.5, np.nan)] == [
        "***", "**", "**", "*", "*", "", "", ""]
    assert pr.p_phrase(0.0004) == "p < 0.001" and pr.p_phrase(0.02) == "p = 0.020" and pr.p_phrase(np.nan) == ""
    assert pr.STARS_NOTE == "* p < .05, ** p < .01, *** p < .001"
    plain = pr.delta_reading("C3 minus C2p", "log_loss", -0.02, -0.03, -0.01)
    assert pr.delta_reading("C3 minus C2p", "log_loss", -0.02, -0.03, -0.01, 0.004) == plain[:-1] + " (bootstrap p = 0.004)."
    assert pr.delta_reading("C3 minus C2p", "log_loss", -0.02, -0.03, -0.01, np.nan) == plain
    assert "(an improvement)" in plain and "(an improvement)" in pr.delta_reading("C3 minus C2p", "skill", 2.0, 1.0, 3.0)
    assert pfig.significance_look(0.2, 0.01, True)[2] and not pfig.significance_look(0.2, 0.01, False)[2]


def test_the_notebook_tables_show_p_q_families_survivors_and_coefficients(analysis):
    a = analysis
    m = pr.metrics_table(a.result)
    assert {pr.BOOT_SKILL_HEADER, pr.PERM_LOSS_HEADER, pr.PERM_AUC_HEADER} <= set(m.columns)
    assert m[pr.PERM_LOSS_HEADER][0] == "" and m[pr.PERM_LOSS_HEADER][3] == pr.p_text(a.result["metrics"].perm_p_log_loss[3])
    assert pr.P_HEADER in pr.delta_table(a.result).columns
    assert all("(bootstrap p " in line for line in pr.delta_readings(a.result) if "is not defined" not in line)
    table = a.single
    view = pr.single_feature_view(a, show_all=True)
    assert {pr.P_HEADER, pr.Q_HEADER, "Survives FDR 5%"} <= set(view.columns)
    assert list(view[pr.Q_HEADER]) == [pr.p_text(q) for q in table.q_value]
    assert (view["Survives FDR 5%"] == "yes").sum() == int(table.significant_fdr.sum())
    k = int(table.significant_fdr.sum())
    sentence = (f"{k} of {len(table)} features survive FDR 5% (Benjamini-Hochberg q < 0.05 over the {len(table)} "
                f"tested here); {int((table.p_value < 0.05).sum())} have an unadjusted permutation p below 0.05.")
    assert pr.fdr_sentence(table) == sentence and sentence in pr.single_feature_reading(a)
    assert "p permutes pitch outcomes (199 permutations)" in pr.single_feature_caption(a)
    families = pr.family_summary_view(a)
    assert list(families.columns) == ["Feature family", "Features ranked", "Unadjusted p < 0.05", "Survive FDR 5%",
                                      "Median size of the correlation", "Largest"]
    survivors = pr.survivors_text(a)
    assert survivors.startswith(f"Surviving FDR 5% ({k})") if k else survivors == "No feature survives FDR 5%."
    coefficients = pr.coefficient_view(a)
    assert len(coefficients) == min(20, len(a.coefficients))
    assert list(coefficients["Column"]) == list(a.coefficients.feature[:20]) and len(pr.coefficient_view(a, 5)) == 5
    caption = pr.coefficient_caption(a)
    assert "Model C5 refitted on all " in caption and "penalty 10" in caption
    assert "conditional associations within a penalised model, not independent effects" in caption
    assert {pr.P_HEADER, pr.Q_HEADER} <= set(pr.by_offer_view(a).columns)
    assert "comparisons survive FDR 5%" in pr.by_offer_reading(a)
