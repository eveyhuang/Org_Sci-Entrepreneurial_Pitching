"""report.md of the pitch-level analysis (analysis.pitch_report): the texts it quotes, its headings and its H9
table. Synthetic runs only; the full report is tested through the command line in tests/test_analyze_panel_cli.py."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from analysis import pitch_report as pr
from tests.panel_analysis_fixtures import write_run


def test_the_report_texts_are_read_from_the_spec_and_the_plan(tmp_path):
    six = pr.section_six()
    assert six.startswith("Written before the v3-dev9 annotations are analysed")
    assert "**Decision rule.**" in six and "**6.7 H9, the answer-level test" in six and "## " not in six
    assert not six.endswith("---")
    limits = pr.plan_limits()
    assert len(limits) >= 7 and limits[0].startswith("Both passes see the outcome.")
    assert all("\n" not in bullet and "  " not in bullet for bullet in limits)
    plan = tmp_path / "plan.md"
    plan.write_text("## 11. What v4 does not fix\n\n- one\n  continued\n- two\n\n## Next\n- not a limit\n")
    assert pr.plan_limits(plan) == ["one continued", "two"]
    plan.write_text("## 10. Other\n- x\n")
    with pytest.raises(ValueError, match="## 11."):
        pr.plan_limits(plan)
    assert pr.git_commit(tmp_path) == "unknown" and re.fullmatch(r"[0-9a-f]{40}", pr.git_commit())


def test_report_markdown_has_the_ten_headings_in_order(tmp_path):
    ap = pr.analyze_panel()
    root = write_run(tmp_path, 8, seed=2, planted="own", strength=4.0)
    a = ap.compute(root, None, 10.0, 9, n_perm=0, skip_h9=True, single_draws=19)
    text = pr.report_markdown(a)
    assert re.findall(r"^## .*$", text, flags=re.M) == list(pr.REPORT_HEADINGS)
    assert pr.H9_SKIPPED_TEXT in text and "| C3q |" in text and "AUC route: " in text


def test_h9_table_says_why_the_frame_covariate_was_not_entered():
    """Fix round 1 (plan B2 Task 2), M4: the q_promotion row never gets a direction; when it separates the outcome or
    cannot be estimated in a population, the Direction cell says it was not entered and why."""
    from types import SimpleNamespace

    from analysis.answer_level import H9_COLUMNS, h9_summary

    def frame(q_promotion: dict) -> pd.DataFrame:
        base = {"model": "support", "population": "all answers", "method": "conditional_logit",
                "outcome": "next_question_same_frame", "n_answers": 40, "n_pitches_informative": 6,
                "n_pitches_dropped": 0, "n_failed": 0, "separated": False, "main_failed": False, "estimable": "yes"}
        rows = [{**base, "predictor": p, "coef": 0.3, "low": 0.1, "high": 0.5} for p in ("evid", "uncert")]
        rows.append({**base, "predictor": "q_promotion", "coef": 0.2, "low": -0.1, "high": 0.4, **q_promotion})
        return pd.DataFrame(rows)[list(H9_COLUMNS)]

    def cell(all_rows: dict, hard_rows: dict) -> str:
        result = {"all": frame(all_rows), "hard": frame(hard_rows)}
        a = SimpleNamespace(h9=result, h9_summary=h9_summary(result), data=SimpleNamespace(videos=range(6)))
        table = "\n".join(pr._h9_lines(a)).splitlines()
        return next(line for line in table if line.startswith("|") and "q_promotion" in line)
    nan = float("nan")
    entered = cell({}, {})
    assert "covariate, no direction" in entered and "not entered" not in entered
    separated = cell({"separated": True, "coef": nan, "low": nan, "high": nan},
                     {"separated": True, "coef": nan, "low": nan, "high": nan})
    assert "covariate, not entered: separated" in separated
    constant = cell({}, {"coef": nan, "low": nan, "high": nan})
    assert "covariate, not entered: no within-pitch variation (hard questions)" in constant
    few = cell({}, {"estimable": "too few informative pitches", "low": nan, "high": nan})
    assert "covariate, entered, no interval: too few informative pitches (hard questions)" in few
    assert "not entered" not in few
    assert "the Direction column says why" in " ".join(pr._h9_lines(SimpleNamespace(
        h9={"all": frame({}), "hard": frame({})}, h9_summary=h9_summary({"all": frame({}), "hard": frame({})}),
        data=SimpleNamespace(videos=range(6)))))


def test_a_missing_value_and_interval_are_one_dash():
    """Plan B Task 6 deferred to Task 9: "— —" never appears for a missing value without an interval."""
    nan = float("nan")
    assert pr._with_ci(nan, nan, nan) == "—"
    assert pr._with_ci(0.1234, nan, nan, 2, True) == "+0.12 —"
    assert pr._with_ci(0.1, 0.0, 0.2, 2) == "0.10 [0.00, 0.20]"


def _verdict(hypothesis, auc, skill, route="used"):
    from types import SimpleNamespace

    return SimpleNamespace(hypothesis=hypothesis, auc_delta=auc[0], auc_low=auc[1], auc_high=auc[2],
                           skill_delta=skill[0], skill_low=skill[1], skill_high=skill[2], auc_route=route)


def test_each_verdict_says_which_route_passed_and_the_gain_from_chance():
    """Fix round 1 of plan B task 9 (I1): the route that passed, and the gain measured from 0.5 when the earlier model
    of the pair ranks below chance (computed from the metrics)."""
    assert pr.route_sentence(_verdict("H1'", (0.245, 0.081, 0.402), (10.4, -3.8, 23.7))) == (
        "Passes on the AUC route only (the skill interval includes zero).")
    assert pr.route_sentence(_verdict("H8", (0.147, 0.032, 0.265), (11.4, -0.9, 22.7))) == (
        "Passes on the AUC route only (the skill interval includes zero; the rule has no skill route for this "
        "hypothesis).")
    assert pr.route_sentence(_verdict("H1'", (0.01, -0.05, 0.07), (3.0, 0.5, 5.5))) == (
        "Passes on the skill route only; the AUC route does not pass.")
    assert pr.route_sentence(_verdict("H1'", (0.01, -0.05, 0.07), (0.1, -2.0, 2.0))) == "Neither route passes."
    metrics = pd.DataFrame({"model": ["C0", "C1", "C2", "C2p", "C3q", "C3"],
                            "families": ["(base rate)", "a", "b", "c", "d", "e"],
                            "auc": [0.5, 0.424, 0.434, 0.441, 0.538, 0.685]})
    sentence = pr.baseline_sentence(_verdict("H1'", (0.245, 0.081, 0.402), (0, 0, 0)), metrics)
    assert "(C1 0.424, C2 0.434, C2p 0.441)" in sentence and "C3's AUC of 0.685 is +0.185 above chance" in sentence
    assert pr.baseline_sentence(_verdict("H8", (0.147, 0.032, 0.265), (0, 0, 0)), metrics) == ""   # C3q above 0.5
    assert pr.baseline_sentence(_verdict("H1'", (0.2, 0.1, 0.3), (0, 0, 0)), None) == ""


def test_h9_claims_are_set_against_the_hypothesis_and_unclaimed_patterns_listed():
    """Fix round 1 of plan B task 9 (I3, I4)."""
    from types import SimpleNamespace

    from analysis.answer_level import H9_COLUMNS, h9_summary

    def frame(rows):
        base = {"model": "full", "population": "p", "method": "fixed_effects", "n_answers": 100,
                "n_pitches_informative": 20, "n_pitches_dropped": 0, "n_failed": 0, "separated": False,
                "main_failed": False, "estimable": "yes"}
        return pd.DataFrame([{**base, **row} for row in rows])[list(H9_COLUMNS)]
    every = frame([{"outcome": "further_questions_same_frame", "predictor": "subs", "coef": 0.27, "low": 0.03, "high": 0.58},
                   {"outcome": "next_challenge", "predictor": "evid", "coef": -0.8, "low": -1.5, "high": -0.1},
                   {"outcome": "next_challenge", "predictor": "ctx", "coef": 0.4, "low": 0.01, "high": 0.7}])
    hard = frame([{"outcome": "further_questions_same_frame", "predictor": "subs", "coef": 0.48, "low": 0.05, "high": 0.96},
                  {"outcome": "next_challenge", "predictor": "evid", "coef": 0.3, "low": -0.7, "high": 1.3},
                  {"outcome": "next_challenge", "predictor": "ctx", "coef": 2.4, "low": float("nan"), "high": float("nan")}])
    result = {"all": every, "hard": hard}
    a = SimpleNamespace(h9=result, h9_summary=h9_summary(result))
    claims = pr.h9_claim_lines(a)
    assert claims == ["full / further_questions_same_frame / subs: positive, against the hypothesis, which expects "
                      "fewer consecutive questions in the same frame (less pressing on) after a more prepared answer: "
                      "here the panel asks more consecutive questions in the same frame: it presses on, more work for "
                      "the panel, not less"]
    unclaimed = pr.h9_unclaimed_lines(a, {"ctx": pr.labelled("ctx")})
    assert unclaimed == ["full / next_challenge / evid: all questions -0.80 [-1.50, -0.10]; hard questions +0.30 "
                         "[-0.70, +1.30]",
                         "full / next_challenge / ctx (Gemini only, not human-coded): all questions +0.40 [+0.01, "
                         "+0.70]; hard questions no interval"]
    assert pr.labelled("pres_ctx").endswith("(Gemini only, not human-coded)") and pr.labelled("subs") == "subs"
