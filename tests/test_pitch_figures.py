"""The pitch-level figures for the paper (analysis/pitch_figures.py), drawn from the pitch-level `Analysis` of
scripts/analysis/analyze_panel.py. Synthetic runs only."""
import dataclasses
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.text import Text  # noqa: E402

from analysis import panel_figures as style_shim  # noqa: E402
from analysis import pitch_report as pr  # noqa: E402
from analysis import pitch_figures as pfig  # noqa: E402
from analysis.panel_bottleneck import LADDER  # noqa: E402
from tests.panel_analysis_fixtures import write_run  # noqa: E402

ap = pr.analyze_panel()
MAIN = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
STEMS = {1: "model_ladder", 2: "auc_per_model", 3: "roc_curves", 4: "concept_forest", 5: "answer_quality_by_outcome",
         6: "calibration", 7: "after_an_answer", 8: "model_coefficients", 9: "h9_forest", 10: "gains_over_baseline",
         "S1": "shark_level_ladder", "S2": "gains_over_baseline_skill"}


@pytest.fixture(scope="module")
def a(tmp_path_factory):
    """Twenty-four pitches with rank scores and H9: a complete pitch-level analysis, without the shark-level path."""
    root = write_run(tmp_path_factory.mktemp("pitch_figures"), 24, seed=1, planted="own", ranking=True, strength=4.0)
    return ap.compute(root, None, 10.0, 19, n_perm=0, single_draws=49)


@pytest.fixture(scope="module")
def sharks(tmp_path_factory):
    """Twelve pitches with the shark-level path and without H9."""
    root = write_run(tmp_path_factory.mktemp("pitch_figures_sharks"), 12, seed=5, planted="own", strength=4.0)
    return ap.compute(root, None, 10.0, 9, n_perm=0, single_draws=19, skip_h9=True, with_shark_level=True)


def _texts(fig) -> list[Text]:
    return [t for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]


def _shown(fig) -> str:
    return " ".join(t.get_text() for t in _texts(fig))


def test_the_figures_are_the_pitch_level_set_with_their_stems():
    assert list(pfig.FIGURES) == [*MAIN, "S1", "S2"]
    assert {key: stem for key, (stem, _) in pfig.FIGURES.items()} == STEMS
    assert all(title and title[0].isupper() for _, title in pfig.FIGURES.values())
    assert pfig.file_names(9) == {"png": "figure_9_h9_forest.png", "pdf": "figure_9_h9_forest.pdf",
                                  "caption": "figure_9_caption.md"}
    assert pfig.file_names("S1")["png"] == "figure_S1_shark_level_ladder.png"
    with pytest.raises(ValueError, match="there is no figure 11"):
        pfig.figure(None, 11)
    # the old module name keeps only the style, re-exported for analysis.human_agreement (plan B task 9)
    assert not hasattr(style_shim, "FIGURES") and not hasattr(style_shim, "figure")
    assert style_shim.SAVE_DPI == pfig.SAVE_DPI == 300 and style_shim.WIDTHS is pfig.WIDTHS
    assert style_shim._context is pfig._context and style_shim.BLUE == pfig.BLUE


@pytest.mark.parametrize("width", ["double", "single"])
@pytest.mark.parametrize("number", MAIN, ids=[str(n) for n in MAIN])
def test_each_figure_renders_on_a_complete_analysis_and_is_readable(a, number, width):
    fig = pfig.figure(a, number, width)
    assert isinstance(fig, Figure), pfig.not_drawn(a, number)
    assert fig.get_size_inches()[0] == pytest.approx(pfig.WIDTHS[width]) and fig.get_size_inches()[1] <= pfig.MAX_HEIGHT
    texts = _texts(fig)
    assert texts and min(t.get_fontsize() for t in texts) >= 8
    assert {t.get_fontfamily()[0] for t in texts} == {pfig.font_name()}
    shown = _shown(fig)
    for column in ("_", "skill_pct", "any offer_"):
        assert column not in shown, column                               # plain-language labels only
    assert fig.legends or any(ax.get_legend() for ax in fig.axes)
    assert all(ax.lines or ax.collections or not ax.axison for ax in fig.axes)       # no empty panel
    plt.close(fig)


def test_the_ladder_names_every_model_and_draws_each_sensitivity_hollow(a):
    fig = pfig.figure(a, 1)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert [label.split()[0] for label in labels] == list(LADDER)
    assert any(label.startswith("C2p ") for label in labels) and any(label.startswith("C3q ") for label in labels)
    main = [line for line in ax.lines if line.get_marker() == "o"]
    assert len(main) == len(LADDER)
    for setting, (colour, marker) in pfig.SENSITIVITY_STYLE.items():
        drawn = [line for line in ax.lines if line.get_marker() == marker
                 and matplotlib.colors.to_hex(line.get_markerfacecolor()).upper() == "#FFFFFF"]
        expected = len(a.sensitivities[setting]["metrics"]) if a.sensitivities[setting]["status"] == "estimated" else 0
        assert len(drawn) == expected, setting
    legend = [t.get_text() for t in fig.legends[0].get_texts()]
    assert all(pfig.SENSITIVITY_WORDS[s] in " ".join(legend) for s in pfig.SENSITIVITY_STYLE)
    assert any("offer rate" in t.get_text() for t in ax.get_xticklabels())
    assert len(fig.axes) == 1                                            # one axis, no second scale
    plt.close(fig)
    fig = pfig.figure(a, 2)
    ax, m = fig.axes[0], a.result["metrics"]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "C2p" in " ".join(labels) and "C3q" in " ".join(labels)
    # every finite AUC is drawn (ruling I3); only the base-rate model is named in the margin instead
    margin = [t for t in ax.texts if t.get_text() == "base rate: no ranking"]
    assert len(margin) == 1 and all(t.get_position()[0] > 1 for t in margin)       # off the AUC axis
    drawn = [line for line in ax.lines if line.get_marker() == "o"]
    with_features = m[m.model != "C0"]
    assert len(drawn) == int(with_features.auc.notna().sum()) == len(m) - 1
    hollow = [line for line in drawn if matplotlib.colors.to_hex(line.get_markerfacecolor()).upper() == "#FFFFFF"]
    assert len(hollow) == int((~with_features.auc_meaningful.astype(bool)).sum())   # diagnostic mark only
    legend = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    assert (pfig.NEAR_CONSTANT_WORDS in legend) == bool(hollow)
    plt.close(fig)


def test_the_roc_curves_and_aucs_read_the_base_rate_adjusted_score(a):
    from analysis.panel_bottleneck import roc_points, weighted_auc

    pred, m = a.result["predictions"], a.result["metrics"].set_index("model")
    fig = pfig.figure(a, 3)
    ax = fig.axes[0]
    assert len(ax.lines) == 1 + len(pr.roc_models(a.result))           # every model with features is drawn
    for line, name in zip(ax.lines[1:], pr.roc_models(a.result)):
        assert pfig.roc_column(pred, name) == f"score_{name}"
        assert line.get_linewidth() == 1.4 and f"AUC {m.auc[name]:.2f}" in line.get_label()
        assert ("near-constant" in line.get_label()) == (not bool(m.auc_meaningful[name]))
        fpr, tpr = roc_points(pred.y, pred[f"score_{name}"], pred.weight)
        assert np.allclose(line.get_xdata(), fpr) and np.allclose(line.get_ydata(), tpr)
        assert m.auc[name] == pytest.approx(weighted_auc(pred.y, pred[f"score_{name}"], pred.weight))
    plt.close(fig)
    for number in (2, 3):
        assert "base-rate-adjusted held-out score" in pfig.caption(a, number)
        assert "not comparable with the AUCs of earlier (v3-dev8) analyses" in pfig.caption(a, number)


def test_h9_forest_draws_each_estimable_interval_and_lists_the_others_in_the_caption(a):
    fig = pfig.figure(a, 9)
    lettered = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert [t[0] for t in lettered] == ["A", "B"] and "All questions" in lettered[0] and "Hard questions" in lettered[1]
    drawn = pfig.h9_rows(a)
    points = sum(len([line for line in ax.lines if line.get_marker() in pfig.H9_MARKERS.values()]) for ax in fig.axes)
    assert points == len(drawn) > 0
    # the frame covariate (q_promotion) is neither drawn nor counted as left out: it is not a predictor of interest
    every = sum(int((frame.predictor != "q_promotion").sum()) for frame in a.h9.values())
    left_out = pfig.h9_left_out(a)
    assert len(drawn) + len(left_out) == every
    assert set(left_out.reason) <= {pfig.H9_TOO_FEW, pfig.H9_SEPARATED, pfig.H9_NOT_ESTIMABLE}
    # one row of axes per unit; within a row panels A and B share the x scale
    rows = [ax for ax in fig.axes if ax.get_xlabel()]
    assert len(rows) == 2 * len(pfig.H9_KINDS)
    for left, right in zip(rows[0::2], rows[1::2]):
        assert left.get_shared_x_axes().joined(left, right) and left.get_xlabel() == right.get_xlabel()
    assert len({ax.get_xlabel() for ax in rows}) == len(pfig.H9_KINDS)     # severity and count never share an axis
    # the caption counts what is not drawn per panel and reason, and names the files that list every row
    caption = pfig.caption(a, 9)
    counts = pfig.h9_left_out_counts(a)
    assert sum(sum(c.values()) for c in counts.values()) == len(left_out)
    if len(left_out):
        for letter, (questions, title) in zip("AB", pfig.H9_PANELS):
            words = ", ".join(f"{n} {reason}" for reason, n in counts[questions].items())
            assert f"panel {letter} ({title.lower()}): {words}" in caption
        assert "h9_all.csv / h9_hard.csv" in caption
        first = left_out.iloc[0]
        assert f"{first.outcome_words}, {first.predictor_words}, {first.model} model" not in caption   # no per-row list
    plt.close(fig)


def test_figure_9_draws_the_new_outcomes(a):
    from analysis.answer_level import OUTCOMES, SENSITIVITY_OUTCOMES

    kinds = [o for _, outcomes, _, _ in pfig.H9_KINDS for o in outcomes]
    assert sorted(kinds) == sorted(OUTCOMES) and len(kinds) == len(set(kinds))   # the import-time assertion holds
    assert {"next_question_same_frame", "further_questions_same_frame"} <= set(kinds)
    assert [k for k, outcomes, _, _ in pfig.H9_KINDS if outcomes == ("further_questions_same_frame",)]
    for outcome in OUTCOMES:
        assert outcome in pfig.H9_OUTCOME_WORDS and outcome in pfig.H9_NARROW_OUTCOME_WORDS, outcome
    for outcome in SENSITIVITY_OUTCOMES:                     # the exchange outcomes are a sensitivity, never drawn
        assert outcome not in kinds and outcome not in pfig.H9_OUTCOME_WORDS, outcome
        assert outcome not in pfig.H9_NARROW_OUTCOME_WORDS, outcome
    rows = pfig._h9_frame(a)
    assert "q_promotion" not in set(rows.predictor)
    assert {"next_question_same_frame", "further_questions_same_frame"} <= set(rows.outcome)
    fig = pfig.figure(a, 9)
    shown = _shown(fig)
    drawn = pfig.h9_rows(a)
    for outcome in set(drawn.outcome):
        assert pfig.H9_OUTCOME_WORDS[outcome] in shown, outcome
    plt.close(fig)
    for width in ("double", "single"):
        caption = pfig.caption(a, 9, width)
        assert "the answered question's frame (promotion or prevention) is entered as a covariate and not drawn" \
            in caption, width
        assert "q_promotion" not in caption and "--h9" not in caption, width            # plain words, no code names
        assert "in the exchange" not in caption and "same exchange" not in caption, width
        # the sensitivity file is named only when the sensitivity was run (fix round 1, M2)
        assert a.h9_sensitivity is None and "h9_exchange_sensitivity.csv" not in caption, width
        run = dataclasses.replace(a, h9_sensitivity={"all": a.h9["all"].iloc[:0]}, h9_sensitivity_draws=9)
        assert "h9_exchange_sensitivity.csv" in pfig.caption(run, 9, width), width


def test_captions_state_the_sample_the_intervals_and_how_to_read(a):
    for number in MAIN:
        text = pfig.caption(a, number)
        assert text.startswith(f"Figure {number}. {pfig.FIGURES[number][1]}.")
        n, offers = a.result["n_groups"], a.result["n_events"]
        assert f"Sample: {n} pitches with a known outcome, {offers} with at least one offer" in text
        assert "model-coded" in text and "exploratory" in text and "How to read it:" in text
    assert "C2p" in pfig.caption(a, 1) and "C3q" in pfig.caption(a, 1)
    for words in pfig.SENSITIVITY_WORDS.values():
        assert words in pfig.caption(a, 1)
    assert "bootstrap" in pfig.caption(a, 1) and "no permutation test was run" in pfig.caption(a, 1)
    assert pfig.caption(a, "S1").startswith("Supplementary Figure S1. ") and "Not drawn: shark-level path not run" in pfig.caption(a, "S1")


def test_write_figures_writes_three_files_per_figure_and_skips_s1_without_the_shark_level_path(a, tmp_path):
    from PIL import Image

    plt.close("all")
    skipped = {}
    written = pfig.write_figures(a, tmp_path / "out", "double", skipped=skipped)
    assert skipped == {"S1": "shark-level path not run"}                  # no figure 1-10 or S2 is skipped
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == sorted(written) and len(written) == 3 * (len(MAIN) + 1)
    assert not plt.get_fignums()
    for number in MAIN:
        names = pfig.file_names(number)
        assert [names["png"], names["pdf"], names["caption"]] == written[3 * MAIN.index(number):3 * MAIN.index(number) + 3]
        with Image.open(tmp_path / "out" / names["png"]) as image:
            assert image.size[0] == 2100 and round(image.info["dpi"][0]) == 300
        pdf = (tmp_path / "out" / names["pdf"]).read_bytes()
        assert pdf.startswith(b"%PDF") and b"/Subtype /Image" not in pdf
        assert (tmp_path / "out" / names["caption"]).read_text() == pfig.caption(a, number, "double") + "\n"
    assert pfig.figure(a, "S1") is None
    assert pfig.write_figures(a, tmp_path / "narrow", "single") == written     # without `skipped`: the same files
    with Image.open(tmp_path / "narrow" / "figure_1_model_ladder.png") as image:
        assert image.size[0] == 1050


def test_s1_is_drawn_when_the_shark_level_path_ran(sharks, tmp_path):
    fig = pfig.figure(sharks, "S1")
    assert isinstance(fig, Figure)
    labels = [t.get_text().split()[0] for t in fig.axes[0].get_yticklabels()]
    assert labels == list(sharks.shark_level["metrics"].model)
    plt.close(fig)
    skipped = {}
    written = pfig.write_figures(sharks, tmp_path / "figs", "double", skipped=skipped)
    assert {"figure_S1_shark_level_ladder.png", "figure_S1_shark_level_ladder.pdf", "figure_S1_caption.md"} <= set(written)
    assert skipped == {"9": "H9 skipped (--skip-h9)"}
    assert "reported, not tested" in pfig.caption(sharks, "S1")


def test_a_builder_that_raises_is_skipped_and_named_only_when_asked(a, tmp_path, monkeypatch):
    real = pfig.figure

    def failing(analysis, number, width=pfig.DEFAULT_WIDTH):
        if number == 2:
            raise KeyError("x")
        return real(analysis, number, width)
    monkeypatch.setattr(pfig, "figure", failing)
    skipped = {}
    written = pfig.write_figures(a, tmp_path / "figs", "double", skipped=skipped)
    assert skipped == {"2": "builder failed (KeyError)", "S1": "shark-level path not run"}
    assert not any(name.startswith("figure_2_") for name in written) and "figure_3_caption.md" in written
    with pytest.raises(KeyError):
        pfig.write_figures(a, tmp_path / "strict", "double")


def test_an_outcome_that_does_not_vary_draws_no_model_figure(a, tmp_path):
    """Every analysed pitch has an offer: the ladder runs, but figures 1, 2, 3 and 6 are not drawn and say why."""
    offered = sorted(a.outcome[a.outcome.any_offer == 1].video_id)
    every = ap.compute(a.run_root, offered, 10.0, 9, n_perm=0, single_draws=19, skip_h9=True)
    assert every.result["n_events"] == every.result["n_groups"] == len(offered)
    for number in (1, 2, 3, 6):
        why = pfig.not_drawn(every, number)
        assert pfig.figure(every, number) is None and why.startswith("the outcome does not vary"), number
        assert f"({len(offered)} of {len(offered)} pitches with an offer)" in why
    skipped = {}
    pfig.write_figures(every, tmp_path / "f", skipped=skipped)
    assert {"1", "2", "3", "6"} <= set(skipped)


def test_figure_five_needs_only_its_table_not_the_ladder(a):
    unestimated = dataclasses.replace(a, result={**a.result, "status": "not estimable: a test"})
    assert pfig.not_drawn(unestimated, 1) and not pfig.not_drawn(unestimated, 5)
    fig = pfig.figure(unestimated, 5)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_a_one_pitch_analysis_draws_only_what_needs_no_model(tmp_path):
    root = write_run(tmp_path, 3, seed=2)
    one = ap.compute(root, ["v000"], 10.0, 9, n_perm=0, single_draws=19, skip_h9=True)
    for number in (1, 2, 3, 4, 5, 6, 8, 9, 10, "S1", "S2"):
        assert pfig.figure(one, number) is None and pfig.not_drawn(one, number), number
        assert "Not drawn: " in pfig.caption(one, number)
    skipped = {}
    written = pfig.write_figures(one, tmp_path / "f", skipped=skipped)
    assert written == ["figure_7_after_an_answer.png", "figure_7_after_an_answer.pdf", "figure_7_caption.md"]
    assert set(skipped) == {"1", "2", "3", "4", "5", "6", "8", "9", "10", "S1", "S2"}
    assert json.dumps(skipped)                                            # plain strings for the manifest
    assert np.isfinite(pfig.width_inches("single"))


def test_without_a_ranking_the_figures_say_rating_not_rank_score(sharks):
    """M1 of plan B's final review: the wording keys off `stage_a_ranking`; this run has no ranking."""
    assert sharks.stage_a_ranking == ap.RANKING_ABSENT and pfig.rating_fallback(sharks)
    labels = list(pfig.ladder_table(sharks).label)
    assert labels[list(LADDER).index("C2")] == f"C2  {pfig.FALLBACK_C2_WORDS}"
    fig = pfig.figure(sharks, 1)
    shown = _shown(fig) + " " + " ".join(t.get_text() for t in fig.legends[0].get_texts())
    plt.close(fig)
    text = pfig.caption(sharks, 1)
    assert "rank score" not in shown and "Stage A rank score" not in text
    assert pfig.FALLBACK_ALL_STAGE_A_WORDS in shown and "the primary Stage A rating" in text
    assert pfig.sensitivity_words(sharks)["all_stage_a"] == pfig.FALLBACK_ALL_STAGE_A_WORDS


def test_a_sensitivity_not_run_has_no_marker_and_is_named_in_legend_and_caption(a):
    """I2 of plan B's final review: this run has no development videos, so that sensitivity is not run."""
    assert a.sensitivities["non_development"]["status"] == ap.NOT_RUN_IDENTICAL
    assert pfig.not_run_sensitivities(a) == {"non_development": ap.NOT_RUN_IDENTICAL}
    assert "non_development" not in set(pfig.sensitivity_table(a).setting)
    fig = pfig.figure(a, 1)
    colour, marker = pfig.SENSITIVITY_STYLE["non_development"]
    assert not [line for line in fig.axes[0].lines if line.get_marker() == marker]
    legend = " ".join(t.get_text() for t in fig.legends[0].get_texts())
    plt.close(fig)
    assert f"without the development videos: {ap.NOT_RUN_IDENTICAL}" in legend
    assert f"Without a marker: without the development videos ({ap.NOT_RUN_IDENTICAL})" in pfig.caption(a, 1)
