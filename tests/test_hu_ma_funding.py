"""The Hu & Ma funding analysis: inputs, Pitch Factor, the small-sample models, the whole run."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from analysis.hu_ma_funding import (
    CODES, HU_MA_COLUMNS, accelerator_family, auc, auc_permutation, external_agreement, external_feature_agreement,
    firth_logit, load_hu_ma_pitch_factor, load_outcomes, loading_agreement, logit_ame, method_convergence,
    pitch_factor, predictor_row, rank_scores, rating_means, rating_repeatability, run_analysis,
)

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "analysis" / "analyze_hu_ma_stage_a.py"


# --- synthetic corpus ---------------------------------------------------------------------------

def _measures(n=50, seed=0):
    """Ten per-video measures with a one-factor structure: positives rise with the factor,
    negatives fall; plus a true funding label driven by the factor."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=n)
    wide = {}
    for code in CODES:
        sign = -1.0 if "negative" in code else 1.0
        wide[code] = sign * 0.8 * factor + rng.normal(scale=0.6, size=n)
    frame = pd.DataFrame(wide, index=[f"v{i:02d}" for i in range(n)])
    frame.index.name = "video_id"
    accepted = (rng.uniform(size=n) < expit(-1.5 + 1.5 * factor)).astype(int)
    return frame, factor, accepted


def _annotations(frame: pd.DataFrame, k=3, seed=1) -> pd.DataFrame:
    """Long annotations like tables/annotations.parquet: k replicates of a 1–5 rating per code,
    replicates jittered around a rounded version of the measure."""
    rng = np.random.default_rng(seed)
    rows = []
    for video_id, measures in frame.iterrows():
        for code in CODES:
            base = int(np.clip(np.round(3 + measures[code]), 1, 5))
            for rep in range(k):
                value = int(np.clip(base + rng.choice([-1, 0, 0, 0, 1]), 1, 5))
                rows.append({"run_id": "r", "video_id": video_id, "code_id": code, "replicate_k": rep,
                             "value": float(value), "uncodable": False, "confidence": "high", "evidence": "t",
                             "prompt_hash": "abc", "pass": "holistic"})
    return pd.DataFrame(rows)


def _scores_csv(frame: pd.DataFrame, path: Path, seed=2) -> Path:
    rng = np.random.default_rng(seed)
    rows = []
    for code in CODES:
        noisy = frame[code] + rng.normal(scale=0.3, size=len(frame))
        pl = (noisy - noisy.mean()).to_numpy()
        rank = pd.Series(pl).rank().to_numpy() / len(pl)
        for video_id, p, r in zip(frame.index, pl, rank):
            rows.append({"video_id": video_id, "code": code, "pl_score": p, "mean_rank": r, "n_orders": 4})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _outcomes_csv(frame: pd.DataFrame, accepted, path: Path, accelerators=None) -> Path:
    accelerators = accelerators or ["YC", "Techstars", "YC W18", "NA", "MIT bootcamp"]
    rows = []
    for i, (video_id, a) in enumerate(zip(frame.index, accepted)):
        acc = accelerators[i % len(accelerators)]
        rows.append({"video_id": video_id, "url": f"https://youtu.be/{video_id}", "title": "t", "channel": "c",
                     "company_name": "co", "Applied to": acc, "Accepted?": "NA" if acc == "NA" else str(int(a)),
                     "Notes": ""})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _hu_ma_csv(frame: pd.DataFrame, path: Path, seed=3, noise=0.4) -> Path:
    """A file like the one Hu & Ma's code writes: their ten features under their column names as
    noisy versions of the measures, and pitch_factor = the first component of those features."""
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame({name: frame[code] + rng.normal(scale=noise, size=len(frame))
                          for name, code in HU_MA_COLUMNS.items()}, index=frame.index)
    pf = pitch_factor(feats.rename(columns=HU_MA_COLUMNS)).scores
    out = feats.copy()
    out.insert(0, "pitch_factor", pf.reindex(feats.index) * 1.7 - 0.3)   # their file is not standardized
    out.index.name = "video_id"
    out.reset_index().to_csv(path, index=False)
    return path


# --- inputs -------------------------------------------------------------------------------------

def test_accelerator_family():
    assert accelerator_family("YC") == accelerator_family("YC W18") == accelerator_family("Y Combinator S19") == "YC"
    assert accelerator_family("Techstars") == accelerator_family("Cedars-Sinai Techstars 2018") == "Techstars"
    assert accelerator_family("MIT Global Entrepreneurship Bootcamp") == "other"
    assert accelerator_family("NA") == accelerator_family("") == "missing"


def test_load_outcomes_reads_labels_and_leaves_na_missing(tmp_path):
    frame, _, accepted = _measures(10)
    path = _outcomes_csv(frame, accepted, tmp_path / "o.csv")
    out = load_outcomes(path)
    assert list(out.columns) == ["video_id", "accepted", "accelerator", "applied_to", "notes"]
    assert len(out) == 10 and out["accepted"].isna().sum() == 2  # every fifth row is NA
    assert set(out["accelerator"]) == {"YC", "Techstars", "missing", "other"}
    assert out.loc[out["accelerator"] == "missing", "accepted"].isna().all()


def test_load_outcomes_rejects_a_label_that_is_not_0_1_or_na(tmp_path):
    (tmp_path / "bad.csv").write_text("video_id,Applied to,Accepted?\nv1,YC,2\n")
    with pytest.raises(ValueError, match="Accepted"):
        load_outcomes(tmp_path / "bad.csv")


def test_rating_means_average_replicates_and_drop_uncodable():
    frame, _, _ = _measures(4)
    ann = _annotations(frame, k=2)
    ann.loc[(ann.video_id == "v00") & (ann.code_id == "visual_positive") & (ann.replicate_k == 0), ["value", "uncodable"]] = [np.nan, True]
    wide = rating_means(ann)
    assert list(wide.columns) == list(CODES) and wide.shape == (4, 10)
    v00 = ann[(ann.video_id == "v00") & (ann.code_id == "visual_positive") & (ann.replicate_k == 1)]["value"].iloc[0]
    assert wide.loc["v00", "visual_positive"] == v00
    v01 = ann[(ann.video_id == "v01") & (ann.code_id == "vocal_arousal")]["value"]
    assert wide.loc["v01", "vocal_arousal"] == pytest.approx(v01.mean())


def test_rating_repeatability_icc_is_one_for_identical_replicates_and_near_zero_for_noise():
    frame, _, _ = _measures(20)
    exact = _annotations(frame, k=3)
    exact["value"] = exact.groupby(["video_id", "code_id"])["value"].transform("first")
    rep = rating_repeatability(exact)
    assert set(rep["code"]) == set(CODES)
    assert (rep["exact_repeat_fraction"] == 1.0).all() and (rep["icc1"] > 0.999).all()
    rng = np.random.default_rng(0)
    noise = exact.copy()
    noise["value"] = rng.integers(1, 6, size=len(noise)).astype(float)
    assert rating_repeatability(noise)["icc1"].abs().mean() < 0.25


def test_rank_scores_wide(tmp_path):
    frame, _, _ = _measures(6)
    path = _scores_csv(frame, tmp_path / "scores.csv")
    wide = rank_scores(path)
    assert wide.shape == (6, 10) and list(wide.columns) == list(CODES)
    raw = pd.read_csv(path)
    assert wide.loc["v03", "vocal_valence"] == pytest.approx(
        raw[(raw.video_id == "v03") & (raw.code == "vocal_valence")]["pl_score"].iloc[0])


# --- Pitch Factor -------------------------------------------------------------------------------

def test_pitch_factor_signs_scale_and_loadings():
    frame, factor, _ = _measures(80)
    pf = pitch_factor(frame)
    assert pf.scores.mean() == pytest.approx(0, abs=1e-9) and pf.scores.std(ddof=1) == pytest.approx(1)
    assert np.corrcoef(pf.scores.loc[frame.index], factor)[0, 1] > 0.9
    load = pf.loadings.set_index("code")["loading"]
    assert (load[[c for c in CODES if "negative" not in c]] > 0).all()
    assert (load[[c for c in CODES if "negative" in c]] < 0).all()
    assert ((pf.loadings["uniqueness"] >= 0) & (pf.loadings["uniqueness"] <= 1)).all()
    assert 0.3 < pf.explained_variance_share < 1 and pf.dropped == {}


def test_pitch_factor_drops_constant_measures_and_names_them():
    frame, _, _ = _measures(30)
    frame["vocal_negative"] = 1.0
    frame.loc["v00", "verbal_warmth"] = np.nan
    pf = pitch_factor(frame)
    assert pf.dropped == {"vocal_negative": "no variance across videos"}
    assert "vocal_negative" not in set(pf.loadings["code"]) and "v00" not in pf.scores.index


def test_pitch_factor_needs_two_varying_measures():
    frame = pd.DataFrame({"visual_positive": [1.0, 1.0, 1.0], "visual_negative": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="at least two"):
        pitch_factor(frame)


# --- models -------------------------------------------------------------------------------------

def test_logit_ame_recovers_a_strong_effect():
    frame, factor, accepted = _measures(400, seed=3)
    out = logit_ame(accepted, factor)
    assert out["method"] == "logit_ml" and out["converged"] and out["note"] == ""
    assert out["ame"] > 0.1 and out["p"] < 1e-6 and 0 < out["pseudo_r2"] < 1
    assert out["n"] == 400 and out["events"] == accepted.sum()


def test_logit_ame_flags_separation_instead_of_crashing():
    x = np.linspace(-1, 1, 30)
    y = (x > 0).astype(int)
    out = logit_ame(y, x)
    assert "separation" in out["note"] or "not estimable" in out["note"]
    assert all(np.isnan(out[k]) for k in ("ame", "ame_se", "p", "pseudo_r2"))


def test_firth_logit_is_finite_under_complete_separation_and_close_to_ml_otherwise():
    x = np.linspace(-1, 1, 30)
    y = (x > 0).astype(int)
    sep = firth_logit(y, x[:, None])
    assert np.all(np.isfinite(sep["coef"])) and np.all(np.isfinite(sep["se"])) and sep["coef"][1] > 0
    assert sep["converged"]
    frame, factor, accepted = _measures(600, seed=5)
    ml = logit_ame(accepted, factor)
    fr = firth_logit(accepted, factor[:, None])
    assert fr["coef"][1] == pytest.approx(ml["coef"], rel=0.1)
    assert fr["ame"][1] == pytest.approx(ml["ame"], rel=0.15)
    assert fr["p"][1] < 1e-6


def test_auc_and_exact_permutation():
    score = np.arange(12, dtype=float)
    y = np.zeros(12, dtype=int)
    y[-3:] = 1  # the three highest scores are the funded ones
    assert auc(y, score) == 1.0
    perm = auc_permutation(y, score)
    assert perm["exact"] and perm["permutations"] == 220 and perm["auc"] == 1.0
    assert perm["p_two_sided"] == pytest.approx(2 / 220)  # the top three or the bottom three
    y_mid = np.zeros(12, dtype=int)
    y_mid[[0, 6, 11]] = 1
    mid = auc_permutation(y_mid, score)
    assert 0.3 < mid["p_two_sided"] <= 1.0
    tie = auc(np.array([1, 0]), np.array([2.0, 2.0]))
    assert tie == 0.5


def test_auc_permutation_falls_back_to_monte_carlo_for_large_designs():
    rng = np.random.default_rng(0)
    score = rng.normal(size=60)
    y = (rng.uniform(size=60) < 0.5).astype(int)
    perm = auc_permutation(y, score, reps=500, seed=1)
    assert not perm["exact"] and perm["permutations"] == 500 and 0 < perm["p_two_sided"] <= 1


def test_predictor_row_has_every_column_and_adds_accelerator_controls():
    frame, factor, accepted = _measures(50, seed=7)
    acc = pd.Series(["YC", "Techstars", "YC", "other", "YC"] * 10, index=frame.index)
    row = predictor_row("f", accepted, pd.Series(factor, index=frame.index), acc)
    for key in ("ml_ame", "ml_ame_se", "ml_p", "firth_coef", "firth_se", "firth_p", "firth_ame", "firth_fe_coef",
                "firth_fe_controls", "auc", "auc_p_exact", "mann_whitney_p", "mean_funded", "mean_not_funded"):
        assert key in row
    assert row["firth_fe_controls"].startswith("accelerator=") and "(ref YC)" in row["firth_fe_controls"]
    assert row["auc"] > 0.5 and row["mean_funded"] > row["mean_not_funded"]


def test_method_convergence_is_high_when_both_methods_see_the_same_order(tmp_path):
    frame, _, _ = _measures(40)
    ratings = rating_means(_annotations(frame, k=3))
    ranks = rank_scores(_scores_csv(frame, tmp_path / "s.csv"))
    conv = method_convergence(ratings, ranks)
    assert len(conv) == 10 and (conv["videos"] == 40).all() and conv["spearman"].min() > 0.5


# --- the whole run ------------------------------------------------------------------------------

def test_run_analysis_writes_every_table_and_the_summary(tmp_path):
    frame, factor, accepted = _measures(50, seed=11)
    ann = _annotations(frame)
    (tmp_path / "tables").mkdir()
    ann.to_parquet(tmp_path / "tables" / "annotations.parquet", index=False)
    scores = _scores_csv(frame, tmp_path / "scores.csv")
    outcomes = _outcomes_csv(frame, accepted, tmp_path / "outcomes.csv")
    out = tmp_path / "out"
    result = run_analysis(tmp_path / "tables" / "annotations.parquet", scores, outcomes, out, provenance={"x": 1})
    for name in ("rating_means.csv", "rating_repeatability.csv", "rank_scores_wide.csv",
                 "pitch_factor_loadings_ratings.csv", "pitch_factor_loadings_ranks.csv", "pitch_factor_scores.csv",
                 "method_convergence.csv", "table3_pitch_factor.csv", "table4_features.csv", "summary.json", "summary.md"):
        assert (out / name).exists(), name
    assert result["videos_in_outcomes_file"] == 50 and result["videos_with_outcome_label"] == 40
    assert result["videos_rated"] == result["videos_ranked"] == 50
    assert {r["predictor"] for r in result["table3"]} == {"Pitch Factor (ratings)", "Pitch Factor (ranks)"}
    assert result["table4_rows"] == 20 and result["provenance"] == {"x": 1}
    assert result["pitch_factor_methods_spearman"]["spearman"] > 0.8
    text = (out / "summary.md").read_text()
    assert "Table III analogue" in text and "Table IV analogue" in text and "not human-validated" in text
    assert json.loads((out / "summary.json").read_text())["acceptances"] == int(accepted[np.arange(50) % 5 != 3].sum())
    scores_frame = pd.read_csv(out / "pitch_factor_scores.csv", index_col="video_id")
    assert np.corrcoef(scores_frame["pitch_factor_ratings"], factor)[0, 1] > 0.8


def test_run_analysis_without_a_ranking_run(tmp_path):
    frame, factor, accepted = _measures(30, seed=12)
    (tmp_path / "tables").mkdir()
    _annotations(frame).to_parquet(tmp_path / "tables" / "annotations.parquet", index=False)
    outcomes = _outcomes_csv(frame, accepted, tmp_path / "outcomes.csv")
    result = run_analysis(tmp_path / "tables" / "annotations.parquet", None, outcomes, tmp_path / "out")
    assert result["videos_ranked"] == 0 and [r["predictor"] for r in result["table3"]] == ["Pitch Factor (ratings)"]
    assert not (tmp_path / "out" / "rank_scores_wide.csv").exists()


def test_run_analysis_with_no_acceptances_still_writes_measures(tmp_path):
    frame, _, _ = _measures(20, seed=13)
    (tmp_path / "tables").mkdir()
    _annotations(frame).to_parquet(tmp_path / "tables" / "annotations.parquet", index=False)
    outcomes = _outcomes_csv(frame, np.zeros(20, dtype=int), tmp_path / "outcomes.csv")
    result = run_analysis(tmp_path / "tables" / "annotations.parquet", None, outcomes, tmp_path / "out")
    assert result["acceptances"] == 0 and result["table3"] == []
    assert "Not estimable" in (tmp_path / "out" / "summary.md").read_text()


def test_cli_end_to_end(tmp_path):
    frame, factor, accepted = _measures(30, seed=14)
    run = tmp_path / "runs" / "stage_a_x"
    (run / "tables").mkdir(parents=True)
    _annotations(frame).to_parquet(run / "tables" / "annotations.parquet", index=False)
    (run / "manifest.json").write_text(json.dumps({"model_requested": "m", "passes": {"holistic": {"done": 90, "prompt_hash": "h"}}}))
    rank = tmp_path / "runs" / "rank_x"
    (rank / "ranking").mkdir(parents=True)
    _scores_csv(frame, rank / "ranking" / "scores.csv")
    (rank / "ranking" / "design.json").write_text(json.dumps({"settings": {"batch_size": 5}, "windows_from": "whole_videos"}))
    outcomes = _outcomes_csv(frame, accepted, tmp_path / "outcomes.csv")
    out = subprocess.run([sys.executable, str(SCRIPT), "--stage-a-run", str(run), "--rank-run", str(rank),
                          "--outcomes", str(outcomes)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Pitch Factor (ratings): ML AME" in out.stdout and "written:" in out.stdout
    summary = json.loads((run / "analysis" / "hu_ma_funding" / "summary.json").read_text())
    assert summary["provenance"]["rank_windows_from"] == "whole_videos"
    assert summary["provenance"]["stage_a_passes"]["holistic"]["prompt_hash"] == "h"
    missing = subprocess.run([sys.executable, str(SCRIPT), "--stage-a-run", str(tmp_path / "nowhere")],
                             capture_output=True, text=True)
    assert missing.returncode == 1 and "aggregate.py" in missing.stderr


# --- Hu & Ma's own Pitch Factor -------------------------------------------------------------------

def test_load_hu_ma_pitch_factor_maps_their_columns_and_rejects_missing_ones(tmp_path):
    frame, _, _ = _measures(12, seed=31)
    hu = load_hu_ma_pitch_factor(_hu_ma_csv(frame, tmp_path / "pf.csv"))
    assert list(hu.columns) == ["pitch_factor", *CODES] and hu.index.name == "video_id" and len(hu) == 12
    bad = pd.read_csv(tmp_path / "pf.csv").drop(columns=["Vocal-Valence"])
    bad.to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="Vocal-Valence"):
        load_hu_ma_pitch_factor(tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="pitch_factor"):
        load_hu_ma_pitch_factor(_outcomes_csv(frame, np.zeros(12, dtype=int), tmp_path / "o.csv"))


def test_external_agreement_is_high_when_hu_ma_tracks_the_same_factor_and_near_zero_when_not(tmp_path):
    frame, factor, _ = _measures(60, seed=32)
    scores = pd.DataFrame({"pitch_factor_ratings": pitch_factor(frame).scores,
                           "pitch_factor_ranks": pitch_factor(frame + 0.1).scores})
    hu = load_hu_ma_pitch_factor(_hu_ma_csv(frame, tmp_path / "pf.csv"))
    agree = external_agreement(scores, hu).set_index("method")
    assert list(agree.index) == ["ratings", "ranks"] and (agree["videos"] == 60).all()
    assert (agree["pearson"] > 0.8).all() and (agree["spearman"] > 0.8).all() and (agree["kendall_tau"] > 0.6).all()
    assert (agree["pearson_low"] < agree["pearson"]).all() and (agree["pearson"] < agree["pearson_high"]).all()
    assert (agree["pearson_low"] > 0.6).all() and (agree["same_tercile_share"] > 0.6).all()
    other, _, _ = _measures(60, seed=99)
    other.index = frame.index
    unrelated = external_agreement(scores, load_hu_ma_pitch_factor(_hu_ma_csv(other, tmp_path / "pf2.csv")))
    assert (unrelated["pearson"].abs() < 0.35).all() and (unrelated["pearson_low"] < 0).all()
    assert (unrelated["pearson_high"] > 0).all()
    few = external_agreement(scores.iloc[:3], hu)
    assert few["pearson"].isna().all() and (few["videos"] == 3).all()


def test_external_feature_agreement_and_loading_agreement(tmp_path):
    frame, _, _ = _measures(50, seed=33)
    ratings = rating_means(_annotations(frame, k=3))
    ranks = rank_scores(_scores_csv(frame, tmp_path / "s.csv"))
    hu = load_hu_ma_pitch_factor(_hu_ma_csv(frame, tmp_path / "pf.csv"))
    feats = external_feature_agreement(ratings, ranks, hu)
    assert list(feats["code"]) == list(CODES) and (feats["videos"] == 50).all()
    assert (feats["rating_spearman"] > 0.4).all() and (feats["rank_score_spearman"] > 0.5).all()
    assert (feats["hu_ma_distinct_values"] == 50).all()
    factors = {"ratings": pitch_factor(ratings), "ranks": pitch_factor(ranks)}
    table, pairs = loading_agreement(factors, pitch_factor(hu.drop(columns="pitch_factor")))
    assert list(table.columns) == ["code", "loading_ratings", "loading_ranks", "loading_hu_ma"] and len(table) == 10
    assert set(pairs) == {"ratings_vs_ranks", "ratings_vs_hu_ma", "ranks_vs_hu_ma"} and min(pairs.values()) > 0.8
    table2, pairs2 = loading_agreement(factors, None)
    assert "loading_hu_ma" not in table2.columns and set(pairs2) == {"ratings_vs_ranks"}


def test_run_analysis_with_hu_ma_adds_the_comparison_a_table3_row_and_the_files(tmp_path):
    frame, factor, accepted = _measures(50, seed=34)
    (tmp_path / "tables").mkdir()
    _annotations(frame).to_parquet(tmp_path / "tables" / "annotations.parquet", index=False)
    scores = _scores_csv(frame, tmp_path / "scores.csv")
    outcomes = _outcomes_csv(frame, accepted, tmp_path / "outcomes.csv")
    hu = _hu_ma_csv(frame, tmp_path / "pf.csv")
    out = tmp_path / "out"
    result = run_analysis(tmp_path / "tables" / "annotations.parquet", scores, outcomes, out, hu_ma_csv=hu)
    info = result["hu_ma_pitch_factor"]
    assert info["videos_in_file"] == info["videos_matched"] == 50
    assert abs(info["reproduced_from_features_corr"]) > 0.999 and 0.3 < info["explained_variance_share"] < 1
    assert [r["method"] for r in info["agreement"]] == ["ratings", "ranks"]
    assert all(r["pearson"] > 0.8 for r in info["agreement"])
    assert set(result["loading_correlations"]) == {"ratings_vs_ranks", "ratings_vs_hu_ma", "ranks_vs_hu_ma"}
    assert {r["predictor"] for r in result["table3"]} == {"Pitch Factor (ratings)", "Pitch Factor (ranks)",
                                                           "Pitch Factor (Hu & Ma)"}
    assert result["table4_rows"] == 30
    for name in ("hu_ma_agreement.csv", "hu_ma_feature_agreement.csv", "loading_agreement.csv"):
        assert (out / name).exists(), name
    pf = pd.read_csv(out / "pitch_factor_scores.csv", index_col="video_id")
    assert "pitch_factor_hu_ma" in pf.columns and abs(pf["pitch_factor_hu_ma"].std(ddof=1) - 1) < 1e-6
    text = (out / "summary.md").read_text()
    assert "Against Hu & Ma's own Pitch Factor" in text and "Loadings of each Pitch Factor" in text
    assert "ratings vs hu_ma" in text
    # the file is kept out when not given
    plain = run_analysis(tmp_path / "tables" / "annotations.parquet", scores, outcomes, tmp_path / "out2")
    assert "hu_ma_pitch_factor" not in plain and plain["table4_rows"] == 20
    assert not (tmp_path / "out2" / "hu_ma_agreement.csv").exists()
    assert "pitch_factor_hu_ma" not in pd.read_csv(tmp_path / "out2" / "pitch_factor_scores.csv").columns


def test_cli_hu_ma_option(tmp_path):
    frame, factor, accepted = _measures(30, seed=35)
    run = tmp_path / "runs" / "stage_a_x"
    (run / "tables").mkdir(parents=True)
    _annotations(frame).to_parquet(run / "tables" / "annotations.parquet", index=False)
    (run / "manifest.json").write_text(json.dumps({"model_requested": "m", "passes": {"holistic": {"done": 90, "prompt_hash": "h"}}}))
    outcomes = _outcomes_csv(frame, accepted, tmp_path / "outcomes.csv")
    hu = _hu_ma_csv(frame, tmp_path / "pf.csv")
    out = subprocess.run([sys.executable, str(SCRIPT), "--stage-a-run", str(run), "--outcomes", str(outcomes),
                          "--hu-ma-pitch-factor", str(hu)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Pitch Factor (ratings) vs Hu & Ma: Pearson" in out.stdout and "Pitch Factor (Hu & Ma): ML AME" in out.stdout
    skipped = subprocess.run([sys.executable, str(SCRIPT), "--stage-a-run", str(run), "--outcomes", str(outcomes),
                              "--hu-ma-pitch-factor", "none", "--out", str(tmp_path / "o2")], capture_output=True, text=True)
    assert skipped.returncode == 0 and "vs Hu & Ma" not in skipped.stdout
    missing = subprocess.run([sys.executable, str(SCRIPT), "--stage-a-run", str(run), "--outcomes", str(outcomes),
                              "--hu-ma-pitch-factor", str(tmp_path / "nowhere.csv")], capture_output=True, text=True)
    assert missing.returncode == 1 and "not found" in missing.stderr
