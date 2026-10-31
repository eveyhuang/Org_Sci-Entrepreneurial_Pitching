"""analysis.panel_report: the tables, captions and plain readings of the pitch-level analysis notebook, and the H4
shares behind figure 7. Synthetic runs only; nothing empirical."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.panel_report as pr
from analysis.answer_level import COVARIATES
from analysis.panel_bottleneck import H4_SHARE_COLUMNS, H4_SHARE_MOVES, LADDER, LADDER_NOT_FITTED, h4_support_shares
from analysis.panel_dataset import eligible_turns, h4_tables, pitch_cut
from analysis.panel_features import FEATURES
from tests.panel_analysis_fixtures import write_run, write_video

BOOT = 49


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Twenty-four pitches with rank scores; v002 is human-reviewed and its review left a validation error open."""
    root = write_run(tmp_path_factory.mktemp("report"), 24, seed=1, planted="own", ranking=True, strength=4.0)
    primary = json.loads((root / "v002" / "results" / "primary.json").read_text())["record"]
    write_video(root, "v002", primary, reviewed=primary)
    (root / "v002" / "results" / "reviewed_validation.json").write_text(json.dumps({"errors": ["[x] open"], "flags": []}))
    return root


@pytest.fixture(scope="module")
def a(run):
    """The notebook's analysis of the whole run, H9 and its exchange-based sensitivity included, quick settings."""
    return pr.compute(run, None, 10, BOOT, permutations=9, single_draws=99, run_h9=True, h9_exchange_outcomes=True,
                      h9_sensitivity_draws=9)


def test_compute_is_the_script_function_with_the_notebook_settings(a, run):
    ap = pr.analyze_panel()
    assert isinstance(a, ap.Analysis) and a.n_boot == BOOT and a.n_perm == 9 and a.single_draws == 99
    assert a.penalty == 10.0 and a.h9 is not None and a.h9_sensitivity_draws == 9 and a.workers == 1
    with pytest.raises(ValueError, match='PENALTY must be "inner_cv" or a positive number'):
        pr.compute(run, penalty="auto")
    quick = pr.compute(run, ["v000", "v001", "v003"], "inner_cv", 9, permutations=0, single_draws=9, run_h9=False,
                       h9_exchange_outcomes=True, workers=None)
    assert quick.penalty is None and quick.h9 is None and quick.h9_sensitivity is None       # no H9, no sensitivity
    assert quick.workers == ap.default_workers()
    assert pr.h9_caption(quick).startswith("H9 was not run") and pr.h9_view(quick).empty


def test_export_writes_the_command_line_files_and_nothing_else(run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run, root)
    before = _snapshot(root)
    b = pr.compute(root, None, 10, 9, permutations=0, single_draws=9, run_h9=False)
    out = pr.export(b, "from_notebook")
    assert {k: v for k, v in _snapshot(root).items() if not k.startswith("analysis/from_notebook/")} == before
    assert {"report.md", "manifest.json", "metrics.csv", "verdicts.csv", "figures"} <= {p.name for p in out.iterdir()}
    assert pr.export_note(out).startswith("Saved ") and str(out.resolve()) in pr.export_note(out)
    pr.export(b, "from_notebook")                                                 # a re-run replaces its own export


def test_provenance_and_settings_say_who_coded_what_and_how(a):
    table = pr.provenance_table(a).set_index("Video")
    assert table.loc["v000", "Coded by"] == "model-coded (primary pass, no human review)"
    assert table.loc["v002", "Coded by"] == "human-reviewed, with issues still open"
    assert table.loc["v002", "Open validation errors after review"] == 1
    text = pr.status_text(a)
    assert text.startswith("24 videos: 23 model-coded only, 1 human-reviewed.") and "issues still open: v002" in text
    settings = pr.settings_table(a).set_index("Setting").Value
    assert settings["Bootstrap draws"] == str(BOOT) and settings["Permutations"] == "9"
    assert settings["H9 exchange-based sensitivity"] == "computed, not reported"
    assert settings["Stage A ranking (the rank score C2 reads)"] == "complete"
    assert settings["Pre-registered test quoted from"].endswith("2026-10-19-pitch-level-refocus.md, section 6")


def test_sample_and_outcomes_give_the_decisions_the_rule_and_the_cut(a):
    o = a.outcome.any_offer
    assert pr.sample_caption(a) == (f"n pitches = 24 ({int((o == 1).sum())} with an offer, {int((o == 0).sum())} "
                                    f"without, {int(o.isna().sum())} unknown)  [pitch-held-out estimate]")
    table = pr.outcome_table(a).set_index("Video")
    assert len(table) == 24 and set(table["Any offer"]) <= {"yes", "no", "unknown"}
    row = table.loc["v000"]
    assert row["Any offer"] == "yes" and "offer at " in row["Decisions, in order"]
    assert " by " in row["History cut at (the first decision of anyone)"]
    unknown = pr.unknown_outcomes(a)
    assert "never counted as" in unknown or unknown.startswith("No unknown outcomes")


def test_history_view_marks_the_cut_and_never_reads_the_decision(a):
    video = pr.default_video(a)
    assert video == "v000" and pr.default_video(a, "v003") == "v003"
    with pytest.raises(ValueError, match="INSPECT_VIDEO 'zzz' is not in this analysis"):
        pr.default_video(a, "zzz")
    view = pr.history_view(a, video)
    cut = int(pitch_cut(a.data).set_index("video_id").cut_pos[video])
    used = view[view["Feeds the features"] == "yes"]
    assert list(used["Position"]) == list(eligible_turns(a.data, video, cut).pos)
    assert (view[view["Position"] >= cut]["Feeds the features"] == "NO").all()
    assert "decides" not in set(used["Function"]) and view["Summary"].str.len().gt(0).all()
    text = pr.history_text(a, video)
    assert text.startswith(f"History cut for the pitch {video}: the first decision of any investor (")
    assert f"turn position {cut} of {len(view)}" in text and f"Turns that feed the features: {len(used)}." in text


def test_feature_tables_are_the_pitch_dictionary_with_its_statuses(a):
    dictionary = pr.dictionary_table()
    assert list(dictionary.columns) == ["Feature (column name)", "Family", "Plain-language label", "Kind", "Status",
                                        "Description"]
    assert set(dictionary["Feature (column name)"]) == set(FEATURES)
    assert set(dictionary["Status"]) <= {"primary", "descriptive", "exploratory", "(key)"}
    assert set(pr.dictionary_table("descriptive")["Feature (column name)"]) == set(LADDER_NOT_FITTED)
    with pytest.raises(ValueError, match="status must be None or one of 'primary', 'descriptive', 'exploratory'"):
        pr.dictionary_table("main")
    counts = pr.status_counts()
    assert f"{len(LADDER_NOT_FITTED)} descriptive" in counts and "rx_answers" in counts
    row = pr.feature_row(a, "v000").set_index("Feature")
    assert list(row.columns) == ["Label", "Value", "Denominator", "Family", "Description"]
    assert row.loc["q_sev_trend", "Denominator"].startswith("of ") and row.loc["hist_questions", "Denominator"] == ""
    readable = pr.readable_features(a)
    assert FEATURES["q_concept_entropy"]["label"] in readable.columns and len(readable) == 24


def test_descriptive_tables_count_what_the_data_hold(a):
    tables = pr.descriptive_tables(a)
    turns = a.data.turns
    questions = turns[turns.function == "question"]
    assert tables["question_concepts"]["Count"].sum() == questions.concept.notna().sum()
    assert tables["exchanges_by_kind"]["Count"].sum() == len(a.data.exchanges)
    assert int(tables["frame_match"].to_numpy().sum()) == int(a.h5.n.sum())
    assert tables["team_gender"]["Count"].sum() == 24 and len(tables["social_and_demonstration"]) == 24
    assert set(tables["prepared_ratings"]["Prepared presentation code"]) == set(a.data.prepared.code)
    assert tables["ask"]["Pitches"].iloc[0] == 24 and len(tables["stage_a_ranks"]) >= 1
    assert "closing_states" not in tables                                      # no closing state since v3-dev9
    for name in ("primary_industry", "exchanges_by_topic", "answer_substance", "answer_context", "question_severity",
                 "turn_functions", "concepts_by_team_gender"):
        assert not tables[name].empty, name
    assert f"n turns = {len(turns)}" in pr.descriptive_caption(a)


def test_self_consistency_and_human_agreement_say_what_exists(a):
    assert "n videos with two passes compared = 24 of 24" in pr.self_consistency_caption(a)
    assert "has not been collected for this run" in pr.human_agreement_text(a)
    assert "ALL ranked videos of the run, not only the 24 analysed here" in pr.ranking_agreement_caption(a)


# ---- H4 ----------------------------------------------------------------------------------------------

def test_h4_views_readings_and_turns_follow_the_counts(a):
    rows = a.followups
    tables = pr.h4_tables(a)
    asker = pr.h4_view(tables, "asker")
    assert asker.iloc[-1, 0] == "Total" and asker["All questions"].iloc[-1] == len(rows)
    by = pr.h4_view(tables, "panel_by_support", "hard")
    assert by["Answers"].iloc[-1] == int(rows.hard_question.sum())
    with pytest.raises(ValueError, match="unknown H4 table 'asker_by_mood'"):
        pr.h4_view(tables, "asker_by_mood")
    caption = pr.h4_caption(a)
    assert f"n pitches = 24, n answers to an investor question = {len(rows)} " in caption and "no history cut" in caption
    reading = pr.h4_reading(tables).split("\n")
    assert reading[0].startswith(f"After the {len(rows)} answers to an investor question: the asker reacted first ")
    again = pr.h4_turns(a, "asker", "follow_up_question")
    assert len(again) == int((rows.next_move == "follow_up_question").sum()) > 0
    assert pr.h4_selection(a, "asker", "follow_up_question") == (
        f"{len(again)} answers match (the asker's next move: asker asks again; all questions; all videos).")
    with pytest.raises(ValueError, match="H4_WHO must be None or one of 'asker', 'panel'"):
        pr.h4_turns(a, "founder")


def test_h4_support_shares_are_the_counts_of_the_h4_table_with_pitch_bootstrap_intervals(a):
    rows = a.followups
    table = h4_support_shares(rows, n_boot=199, seed=7)
    assert list(table.columns) == H4_SHARE_COLUMNS and len(table) == 2 * len(H4_SHARE_MOVES)
    counted = h4_tables(rows)["panel_by_support"].set_index(["questions", "support"])
    for r in table.itertuples():
        for key in ("yes", "no"):
            n, count = getattr(r, f"n_{key}"), getattr(r, f"count_{key}")
            assert n == counted.loc[(r.questions, key), "n"] and count == counted.loc[(r.questions, key), r.move]
            share, low, high = getattr(r, f"share_{key}"), getattr(r, f"{key}_low"), getattr(r, f"{key}_high")
            assert share == pytest.approx(count / n) and 0 <= low <= share + 1e-9 and share - 1e-9 <= high <= 1
        assert r.difference == pytest.approx(r.share_yes - r.share_no) and r.diff_low <= r.diff_high
        assert r.n_pitches == rows.video_id.nunique() and r.n_boot == 199
    pd.testing.assert_frame_equal(table, h4_support_shares(rows, n_boot=199, seed=7))      # the seed fixes them
    pd.testing.assert_frame_equal(a.h4_shares, h4_support_shares(rows, n_boot=BOOT, seed=a.seed))
    single = h4_support_shares(rows[rows.video_id == rows.video_id.iloc[0]], n_boot=50)   # one pitch: no spread
    known = single[single.share_yes.notna()]
    assert np.allclose(known.share_yes, known.yes_low) and np.allclose(known.share_yes, known.yes_high)
    with pytest.raises(ValueError, match="without a cut"):
        h4_support_shares(rows.assign(next_move=["after_cut", *rows.next_move[1:]]))
    view = pr.h4_shares_view(a)
    assert len(view) == len(a.h4_shares) and "Share, with support [95% interval]" in view.columns


# ---- the ladder, the verdicts, the sensitivities ------------------------------------------------------

def test_the_ladder_tables_follow_the_evaluation(a):
    guide = pr.ladder_guide(a).set_index("Model")
    assert list(guide.index) == list(LADDER) and guide.loc["C0", "Columns fitted"] == 0
    assert guide.loc["C3q", "Adds"] == "+ the sharks' questions"
    assert guide.loc["C1", "Design columns"] == ", ".join(a.result["design_columns"]["C1"])
    assert "rx_answers" in pr.not_fitted_note()
    m = pr.metrics_table(a.result)
    assert list(m.Model) == list(LADDER) and "Near-constant predictions (diagnostic)" in m.columns
    flagged = ~a.result["metrics"].auc_meaningful.astype(bool)
    assert list(m["Near-constant predictions (diagnostic)"] == "yes") == list(flagged)
    text = pr.how_to_read(a)
    assert text.startswith("HOW TO READ THESE NUMBERS") and "not comparable with those of the v3-dev8" in text
    best = pr.best_reading(a)
    metrics = a.result["metrics"].set_index("model")
    assert best.startswith(f"Lowest log loss: {metrics.log_loss.idxmin()} ") and pr.ladder_caption(a) in best
    deltas = pr.delta_table(a.result)
    assert set(deltas.Metric) == {"log loss", "Brier score", "AUC", "skill over the base rate"}
    readings = pr.delta_readings(a.result, "skill")
    assert len(readings) == len(a.result["deltas"][a.result["deltas"].metric == "skill"])
    penalties = pr.penalty_table(a)
    assert len(penalties) == a.result["n_groups"] and list(penalties.columns) == list(LADDER)


def test_verdicts_and_sensitivities_are_those_of_the_report(a):
    rules = pr.rules_table().set_index("Hypothesis")
    assert rules.loc["H1'", "Comparison"] == "C3 minus C2p" and "+0.05" in rules.loc["H1'", "Rule"]
    assert "skill" in rules.loc["H1'", "Rule"] and "skill" not in rules.loc["H8", "Rule"]
    view = pr.verdict_view(a).set_index("Hypothesis")
    assert list(view.Verdict) == list(a.verdicts.verdict) and list(view["AUC route"]) == list(a.verdicts.auc_route)
    r = a.verdicts.set_index("hypothesis").loc["H8"]
    assert view.loc["H8", "AUC gain [95% interval]"] == f"{r.auc_delta:+.3f} [{r.auc_low:+.3f}, {r.auc_high:+.3f}]"
    text = pr.verdict_text(a)
    assert text.count("**Verdict: ") == 2 and "cannot be told from zero" in text
    assert text.count("Passes on ") + text.count("Neither route passes.") == 2      # which route passed, per verdict
    sens = pr.sensitivity_view(a)
    assert list(sens.Setting) == ["main", *a.sensitivities] and sens.Pitches.iloc[0] == a.result["n_groups"]
    reading = pr.sensitivity_reading(a)
    assert reading.startswith("H1': main verdict '") and "not a second test" in reading
    assert pr.single_pass_text(a) == "Every video has both passes for both_passes."


# ---- H9, one prediction, limits, figures ----------------------------------------------------------------

def test_h9_tables_show_both_populations_and_claim_only_agreeing_directions(a):
    models = pr.h9_models_table()
    assert list(models.Model) == ["support", "substance", "full", "responsiveness"]
    view = pr.h9_view(a)
    assert len(view) == len(a.h9["all"]) and set(view.columns) >= {"All questions", "Hard questions", "Direction"}
    covariate = view[view.Predictor.isin(COVARIATES)]
    assert len(covariate) and set(covariate.Direction) == {"covariate, no direction"}
    claimed = pr.h9_view(a, claimed_only=True)
    assert len(claimed) == int(a.h9_summary.hard_and_all_agree.sum())
    text = pr.h9_directions(a)
    assert text.startswith(f"{len(claimed)} of {len(a.h9_summary)} predictor rows") and "exploratory pattern" in text
    assert "Not claimed: in " in text and all(line in text for line in pr.h9_unclaimed_lines(a, pr.H9_PREDICTOR_WORDS))
    assert ("ctx (Gemini only, not human-coded)" in set(view.Predictor)) == ("ctx" in set(a.h9["all"].predictor))
    note = pr.h9_sensitivity_note(a)
    assert "h9_exchange_sensitivity.csv" in note and "9 draws" in note and "next_question" not in note


def test_one_prediction_is_taken_apart(a):
    model = pr.default_model(a)
    assert model == "C3" and pr.default_model(a, "C5") == "C5"
    with pytest.raises(ValueError, match="INSPECT_MODEL 'B5' is not a model of the ladder"):
        pr.default_model(a, "B5")
    text = pr.prediction_text(a, model, "v000")
    assert text.startswith("Model C3, trained without v000, gives it a ") and "penalty 10" in text
    view = pr.prediction_view(a, model, "v000")
    assert view.Feature.iloc[0] == "(intercept)" and len(view) > 1
    assert "(the training offer rate: no features)" in pr.prediction_text(a, "C0", "v000")
    assert pr.prediction_view(a, "C0", "v000").empty


def test_limits_quote_plan_v4_with_the_runs_count_and_the_methods_limits(a):
    table = pr.limits_table(a)
    assert list(table.columns) == ["From", "Limit"] and len(table) == len(pr.plan_limits()) + len(pr.METHOD_LIMITS)
    without = int((a.outcome.any_offer == 0).sum())
    assert any(f"{without} pitches without an offer" in limit for limit in table.Limit)


def test_figure_files_and_their_note(run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run, root)
    b = pr.compute(root, None, 10, 9, permutations=0, single_draws=9, run_h9=False)
    assert pr.figure_files(b, "figs").empty
    assert pr.files_note(b, "figs", False).startswith("EXPORT and SAVE_FIGURES are False: nothing was written, and ")
    folder = pr.save_figures(b, "figs")
    assert sorted(p.name for p in (root / "analysis" / "figs").iterdir()) == ["figures"]
    listed = pr.figure_files(b, "figs")
    assert len(listed) == len(list(folder.glob("*.png"))) and listed.Caption.str.startswith("Figure").any()
    assert pr.files_note(b, "figs", True).startswith(f"Saved {len(listed)} figures (PNG at 300 dpi")
    assert "from an earlier export" in pr.files_note(b, "figs", False)
