"""The pitch-level analysis script (scripts/analysis/analyze_panel.py) and its report.md (analysis.pitch_report).
Synthetic runs only."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import analyze_panel as ap  # noqa: E402
from analysis import pitch_report as pr  # noqa: E402
from analysis.answer_level import MODELS as H9_MODELS, POPULATIONS  # noqa: E402
from analysis.panel_bottleneck import (  # noqa: E402
    C1_COLUMNS, DEMONSTRATION, LADDER, LADDER_COMPARISONS, LADDER_NOT_FITTED, STAGE_A_FALLBACK_NOTE,
)
from analysis.panel_dataset import load_run, pitch_cut, pitch_outcome  # noqa: E402
from analysis.panel_features import feature_dictionary, pitch_features  # noqa: E402
from analysis.verdicts import VERDICT_COLUMNS  # noqa: E402
from tests.panel_analysis_fixtures import make_record, write_run, write_video  # noqa: E402

LISTED = ["videos.csv", "turns.csv", "exchanges.csv", "decisions.csv", "prepared.csv", "outcomes.csv", "features.csv",
          "feature_dictionary.csv", "metrics.csv", "chance_comparisons.csv", "deltas.csv", "predictions.csv", "folds.csv", "permutation.csv",
          "verdicts.csv", "sensitivity_metrics.csv", "sensitivity_deltas.csv", "single_feature.csv",
          "single_feature_families.csv", "answer_quality_by_offer.csv", "coefficients.csv", "h9_all.csv", "h9_hard.csv",
          "h9_responsiveness.csv", "h9_summary.csv", "h4_followups.csv", "h4_support_shares.csv", "h5_frame_match.csv", "self_consistency.csv",
          "self_consistency_turn_functions.csv", "stage_a_ranking_agreement.csv", "descriptives.json",
          "h4_tables.json", "manifest.json", "report.md"]
FIGURES = "figures"
SHARK_LEVEL = "shark_level"
HEADINGS = ["## 0. The pre-registered test", "## 1. Sample and outcomes", "## 2. The ladder", "## 3. Verdicts",
            "## 4. Sensitivities", "## 5. H9", "## 6. H4 and H5", "## 7. Self-consistency", "## 8. Limits",
            "## 9. Provenance"]
SETTINGS = ["main", "both_passes", "rates", "non_development", "fixed_penalty_10", "all_stage_a"]
SPEC = REPO / "docs" / "plans" / "design docs" / "2026-10-19-pitch-level-refocus.md"
PLAN = REPO / "docs" / "plans" / "project_plan_v4.md"
QUICK = ["--penalty", "10", "--workers", "1"]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): _sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _section(report: str, number: int) -> str:
    start = report.index(HEADINGS[number])
    end = report.index(HEADINGS[number + 1]) if number + 1 < len(HEADINGS) else len(report)
    return report[start:end]


def _plan_bullets() -> list[str]:
    """The bullets of plan v4 section 11, read here independently of the script: '- ' starts one, indented lines
    continue it."""
    lines = PLAN.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## 11."))
    bullets = []
    for line in lines[start + 1:]:
        if line.startswith("## "):
            break
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and line.strip() and bullets:
            bullets[-1] += " " + line.strip()
    return bullets


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A 30-pitch run with rank scores, one human-reviewed video and one video whose replicate pass is missing,
    analysed once from the command line."""
    tmp = tmp_path_factory.mktemp("cli")
    root = write_run(tmp, 30, seed=1, planted="own", ranking=True, strength=4.0)
    primary = json.loads((root / "v002" / "results" / "primary.json").read_text())["record"]
    write_video(root, "v002", primary, reviewed=primary)
    (root / "v002" / "review").mkdir()
    (root / "v002" / "review" / "human_agreement.json").write_text(json.dumps({"n_reviewed": 3, "by_class": {}}))
    (root / "v005" / "results" / "replicate.json").unlink()
    (root / "selection_all100.json").write_text(json.dumps({"video_ids": [f"v{i:03d}" for i in range(30)],
                                                             "pilot_development_videos": ["v001", "v003"]}))
    before = _snapshot(root)
    code = ap.main(["--run", str(root), "--analysis-id", "a1", "--n-boot", "49", "--n-perm", "9",
                    "--single-draws", "199", *QUICK])
    return root, root / "analysis" / "a1", before, code


def test_importing_the_script_loads_no_provider():
    code = ("import sys; sys.path.insert(0, 'scripts/analysis'); import analyze_panel; import analysis.pitch_figures; "
            "assert 'pipeline.providers.gemini' not in sys.modules; assert 'google' not in sys.modules; "
            "assert not [m for m in sys.modules if m.startswith('analysis.archive')], 'archive loaded'; "
            "assert 'analysis.panel_report' not in sys.modules; print('clean')")
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0 and done.stdout.strip() == "clean", done.stderr


def test_the_listed_files_are_written_and_nothing_outside_the_analysis_folder(exported):
    root, out, before, code = exported
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == sorted(LISTED + [FIGURES])
    assert not (out / SHARK_LEVEL).exists()                                 # only with --with-shark-level
    assert {k: v for k, v in _snapshot(root).items() if not k.startswith("analysis/a1/")} == before
    assert not any(name.endswith(("_o1.csv", "_o2.csv")) for name in LISTED)


def test_the_manifest_records_the_settings_the_passes_and_the_texts_quoted(exported):
    root, out, _, _ = exported
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["analysis_id"] == "a1" and manifest["run"] == str(root.resolve())
    assert manifest["release_ids"] == ["v3-dev9"] and manifest["n_boot"] == 49 and manifest["n_perm"] == 9
    assert manifest["seed"] == ap.SEED and manifest["workers"] == 1 and manifest["penalty"] == "fixed 10"
    assert manifest["with_shark_level"] is False and manifest["shark_level"] is None
    assert manifest["passes"] == {"main": "best", "both_passes": "both"}
    assert manifest["single_pass_videos"] == {"v005": "missing"}
    assert manifest["spec"] == {"path": str(SPEC.relative_to(REPO)), "sha256": _sha(SPEC), "section": "6"}
    assert manifest["limits"] == {"path": str(PLAN.relative_to(REPO)), "sha256": _sha(PLAN), "section": "11"}
    # the hash is the whole file's, and the report says so (final review M6)
    report = (out / "report.md").read_text()
    assert f"section 11 (the file's sha256 {_sha(PLAN)})" in report and f"section 6 (the file's sha256 {_sha(SPEC)})" in report
    assert f"(sha256 of the whole file {_sha(PLAN)})" in report
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", manifest["git_commit"])
    assert manifest["stage_a_ranking"] == "complete" and manifest["h9"] == "run"
    assert manifest["models"] == {name: list(spec) for name, spec in LADDER.items()}
    assert manifest["comparisons"] == [list(pair) for pair in LADDER_COMPARISONS]
    # ruling I4: each model's design columns; C1 is the spec's list, no count twin or frame-match share anywhere
    design = manifest["design_columns"]
    assert list(design) == list(LADDER) and design["C1"] == list(C1_COLUMNS) and design["C0"] == []
    assert manifest["not_fitted"]["columns"] == list(LADDER_NOT_FITTED)
    for setting, columns in [("main", design), *((n, i["design_columns"]) for n, i in manifest["sensitivities"].items())]:
        assert columns["C1"] == list(C1_COLUMNS), setting
        assert not [c for names in columns.values() for c in names
                    if c.endswith("_n") or "a_frame_match_share" in c], setting
    assert list(manifest["sensitivities"]) == SETTINGS[1:]
    assert manifest["sensitivities"]["non_development"]["excluded_videos"] == ["v001", "v003"]
    # the per-video books source is counted without the single-pass key of the both-passes read
    assert manifest["books"]["videos_by_source"] == {"live": 30}
    # the pitch-level figures (analysis.pitch_figures): all ten and S2 drawn, S1 only with the shark-level path
    assert manifest["figures_skipped"] == {"S1": "shark-level path not run"}
    assert len(manifest["figures"]["files"]) == 3 * 11
    drawn = [f"{FIGURES}/{p.name}" for p in (out / FIGURES).iterdir()]
    assert sorted(manifest["outputs"]) == sorted(LISTED + drawn)
    assert sorted(manifest["figures"]["files"]) == sorted(p.name for p in (out / FIGURES).iterdir())
    inputs = manifest["inputs"]
    assert inputs["v002/results/reviewed.json"] == _sha(root / "v002/results/reviewed.json")
    assert inputs["v000/results/replicate.json"] == _sha(root / "v000/results/replicate.json")     # sensitivity (a)
    assert inputs["ranking/scores.json"] == _sha(root / "ranking/scores.json")
    assert inputs["selection_all100.json"] == _sha(root / "selection_all100.json")       # read for the development ids
    assert all(inputs[k] == _sha(root / k) for k in inputs)
    for name in LISTED:
        if name.endswith(".csv"):
            assert manifest["rows"][name] == len(pd.read_csv(out / name)), name


def test_the_tables_are_those_of_the_modules(exported):
    root, out, _, _ = exported
    data = load_run(root)
    features = pd.read_csv(out / "features.csv")
    expected = pitch_features(data)
    assert list(features.video_id) == list(expected.video_id) and len(features) == 30
    np.testing.assert_allclose(features.any_offer, expected.any_offer, equal_nan=True)
    outcomes = pd.read_csv(out / "outcomes.csv")
    assert list(outcomes.cut_pos) == list(pitch_cut(data).cut_pos)
    np.testing.assert_allclose(outcomes.any_offer, pitch_outcome(data).any_offer, equal_nan=True)
    assert list(pd.read_csv(out / "feature_dictionary.csv").name) == list(feature_dictionary().name)
    prepared = pd.read_csv(out / "prepared.csv")
    assert len(prepared) == len(data.prepared) and "source" in prepared
    metrics = pd.read_csv(out / "metrics.csv")
    assert list(metrics.model) == list(LADDER) and (metrics.label == "pitch-held-out estimate").all()
    assert (metrics.n_pitches == 30).all() and (metrics.coding == "model-coded: 29 pitches; human-reviewed: 1 pitch").all()
    assert metrics.perm_p_log_loss.notna().sum() == len(LADDER) - 1          # every model with features
    chance = pd.read_csv(out / 'chance_comparisons.csv').set_index('model')
    matched = metrics.set_index('model').loc[chance.index]
    np.testing.assert_allclose(chance.above_chance, matched.auc - .5)
    np.testing.assert_allclose(chance.perm_p_auc, matched.perm_p_auc)
    deltas = pd.read_csv(out / "deltas.csv")
    assert {f"{b} minus {a}" for a, b in LADDER_COMPARISONS} == set(deltas.comparison)
    assert set(deltas.metric) == {"log_loss", "brier", "auc", "skill"}
    assert {f"p_{m}" for m in LADDER} <= set(pd.read_csv(out / "predictions.csv").columns)
    assert set(pd.read_csv(out / "folds.csv").model) == set(LADDER)
    assert set(pd.read_csv(out / "permutation.csv").model) == set(LADDER) - {"C0"}
    assert set(pd.read_csv(out / "coefficients.csv").model) == {"C5"}
    single = pd.read_csv(out / "single_feature.csv")
    assert len(single) and "investor" not in pd.read_csv(out / "features.csv").columns
    assert set(pd.read_csv(out / "single_feature_families.csv").family) <= set(single.family)
    by_offer = pd.read_csv(out / "answer_quality_by_offer.csv")                       # the table behind figure 5
    assert set(by_offer.feature) == {"a_subs_mean", "a_subs_hard_mean", "a_evid_share", "a_evid_hard_share"}
    assert (by_offer.n_pitches == 30).all() and by_offer.correlation.notna().all()
    assert len(pd.read_csv(out / "h4_followups.csv")) and len(pd.read_csv(out / "h4_support_shares.csv"))
    frames = pd.read_csv(out / "h5_frame_match.csv")
    assert list(frames.columns[:4]) == ["question_frame", "answer_frame", "n", "matches"]
    assert frames.n.sum() == int((data.turns.function == "answer").sum())
    for name in ("self_consistency.csv", "self_consistency_turn_functions.csv"):
        assert {"n_videos", "label"} <= set(pd.read_csv(out / name).columns), name
    assert json.loads((out / "h4_tables.json").read_text())["n_pitches"] == 30
    d = json.loads((out / "descriptives.json").read_text())
    assert d["counts"]["pitches"] == 30 and d["coding"] == {"primary": 29, "reviewed": 1}
    assert d["counts"]["pitches_with_offer"] + d["counts"]["pitches_without_offer"] + d["counts"]["pitches_unknown"] == 30


def test_verdicts_are_the_rules_on_the_ladder_and_its_sensitivities(exported):
    _, out, _, _ = exported
    table = pd.read_csv(out / "verdicts.csv")
    assert list(table.columns) == ["setting", *VERDICT_COLUMNS]
    main = table[table.setting == "main"]
    assert len(main) == 2 and list(main.hypothesis) == ["H1'", "H8"] and table.setting.iloc[0] == "main"
    assert list(dict.fromkeys(table.setting)) == SETTINGS and len(table) == 2 * len(SETTINGS)
    assert list(main.comparison) == ["C3 minus C2p", "C3 minus C3q"]
    assert set(table.verdict) <= {"supported", "not distinguishable from zero"}
    report = (out / "report.md").read_text()
    verdicts = _section(report, 3)
    for row in main.itertuples():
        assert f"AUC route: {row.auc_route}" in verdicts and row.verdict in verdicts
        # the power sentence follows its verdict
        assert verdicts.index(row.power, verdicts.index(row.hypothesis)) > verdicts.index(row.hypothesis)


def test_the_sensitivities_are_full_ladders_with_their_names(exported):
    _, out, _, _ = exported
    metrics, deltas = pd.read_csv(out / "sensitivity_metrics.csv"), pd.read_csv(out / "sensitivity_deltas.csv")
    assert list(dict.fromkeys(metrics.sensitivity)) == SETTINGS[1:] == list(dict.fromkeys(deltas.sensitivity))
    for name, part in metrics.groupby("sensitivity"):
        assert list(part.model) == list(LADDER), name
        assert part.sensitivity_note.notna().all() and (part.label == "pitch-held-out estimate").all(), name
    by = metrics.set_index(["sensitivity", "model"])
    assert by.loc[("non_development", "C1"), "n_pitches"] == 28                      # v001 and v003 left out
    assert by.loc[("fixed_penalty_10", "C3"), "penalty"] == "fixed 10"
    assert "+".join(LADDER["C2"]) != by.loc[("all_stage_a", "C2"), "families"]      # every Stage A rank score
    assert set(deltas.comparison) == {f"{b} minus {a}" for a, b in LADDER_COMPARISONS}
    report = _section((out / "report.md").read_text(), 4)
    for name in SETTINGS[1:]:
        assert name in report


def test_the_report_has_its_sections_and_quotes_the_rule_verbatim(exported):
    _, out, _, _ = exported
    report = (out / "report.md").read_text()
    assert re.findall(r"^## .*$", report, flags=re.M) == HEADINGS
    spec = SPEC.read_text()
    start, end = spec.index("**Decision rule.**"), spec.index("**6.4 Sensitivities")
    rule = spec[start:end].strip()
    assert rule in _section(report, 0) and "C3 − C2′ has an AUC gain of at least +0.05" in rule
    assert pr.section_six() in _section(report, 0)
    assert "**6.7 H9, the answer-level test" in _section(report, 0)
    # section 2 says what the AUC ranks and that it is not comparable with earlier analyses
    assert pr.AUC_ADJUSTED_SENTENCE in _section(report, 2) and "not comparable with the AUCs of earlier (v3-dev8)" in report
    predictions = pd.read_csv(out / "predictions.csv")
    assert {f"score_{m}" for m in LADDER} <= set(predictions.columns) and (predictions.score_C0 == 0).all()
    # section 8 is plan v4 section 11, read from the file however many bullets it has
    # with this run's number of pitches without an offer (plan v4 states the v3-dev8 count), then the method's limits
    limits = _section(report, 8)
    shown = [line[2:] for line in limits.splitlines() if line.startswith("- ")]
    without = int((pd.read_csv(out / "outcomes.csv").any_offer == 0).sum())

    def own(found):
        return (found.group(0) if int(found.group(1)) == without else
                f"{without} pitches without an offer in this run (plan v4 states {found.group(1)}, the v3-dev8 count)")
    plan = _plan_bullets()
    assert plan == pr.plan_limits() and len(plan) >= 7 and any("pitches without an offer" in b for b in plan)
    assert shown == [re.sub(r"(\d+) pitches without an offer", own, b) for b in plan] + list(pr.METHOD_LIMITS)
    assert any(f"{without} pitches without an offer" in b for b in shown)
    assert "residual leave-one-out bias against weak models" in limits
    # sections 4 and 5 state what plan B's reviews carried to the run
    assert pr.NON_DEVELOPMENT_TEXT in _section(report, 4)
    assert "the 1 video with a reviewed record averages the reviewed record with the raw replicate" in report
    h9 = _section(report, 5)
    assert pr.H9_UNITS_TEXT in h9 and re.search(r"for \d+ of \d+ answers \(\d+\.\d%\), in \d+ pitches", h9)
    assert "— —" not in report                                                 # a missing value is one dash
    # fix round 1 of plan B task 9: the routes, the diagnostic note, the Gemini-only label, the H9 claims
    assert pr.AUC_MEANINGFUL_NOTE in _section(report, 2) and pr.CONTEXT_SENTENCE in _section(report, 2)
    assert "pres_ctx (Gemini only, not human-coded)" in _section(report, 2)
    verdict_text = _section(report, 3)
    assert verdict_text.count("Passes on ") + verdict_text.count("Neither route passes.") == 2
    assert "cannot be told from zero" in verdict_text and "what the data can see" not in verdict_text
    assert pr.H9_EXPLORATORY_TEXT in h9 and "Directions claimed (" in h9 and "Not claimed: " in h9
    assert "ctx (Gemini only, not human-coded)" in h9
    provenance = _section(report, 9)
    manifest = json.loads((out / "manifest.json").read_text())
    for text in (manifest["git_commit"], "v3-dev9", "seed: 42", "n_boot: 49", "n_perm: 9", "workers: 1",
                 manifest["spec"]["sha256"]):
        assert text in provenance, text
    ladder = _section(report, 2)
    assert "| C3q |" in ladder and "C3 minus C2p" in ladder and "complete" in _section(report, 1) + ladder
    assert "Permutation p (log loss)" in ladder and "Permutation p (AUC)" in ladder
    assert "### Design columns" in ladder and f"| C1 | {len(C1_COLUMNS)} | {', '.join(C1_COLUMNS)} |" in ladder
    assert "demo_seconds, prepared_seconds" in ladder                             # named as not fitted
    metrics = pd.read_csv(out / "metrics.csv").set_index("model")
    row = next(line for line in ladder.splitlines() if line.startswith("| C3 |"))
    assert row.rstrip(" |").endswith(f"{metrics.perm_p_auc['C3']:.3f}")       # the last column is the AUC p


def test_section_six_fails_loudly_without_its_headings(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# Title\n\n## 5. Something\n\ntext\n\n## 7. Sequencing\n")
    with pytest.raises(ValueError, match="## 6."):
        pr.section_six(spec)
    spec.write_text("## 6. The test\n\nrule\n")
    with pytest.raises(ValueError, match="## 7."):
        pr.section_six(spec)
    spec.write_text("## 6. The test\n\nthe rule, verbatim\n\n---\n\n## 7. Next\n")
    assert pr.section_six(spec) == "the rule, verbatim"


def test_h9_tables_and_the_report_state_each_population(exported):
    _, out, _, _ = exported
    every, hard = pd.read_csv(out / "h9_all.csv"), pd.read_csv(out / "h9_hard.csv")
    responsiveness = pd.read_csv(out / "h9_responsiveness.csv")
    assert set(every.model) == set(hard.model) == set(H9_MODELS) - {"responsiveness"}
    assert set(responsiveness.model) == {"responsiveness"} and set(responsiveness.questions) == {"all", "hard"}
    summary = pd.read_csv(out / "h9_summary.csv")
    assert set(summary.model) == set(H9_MODELS) and summary.direction.notna().all()
    section = _section((out / "report.md").read_text(), 5)
    for model, population in POPULATIONS.items():
        assert f"**{model}**" in section and population in section, model
        n = int(pd.concat([every, responsiveness])[lambda f: f.model == model].n_answers.max())
        assert f"{n} answers" in section, model
    assert "not tested as a mediation" in section


def test_a_withheld_auc_route_reaches_the_verdicts_and_the_report(tmp_path, monkeypatch):
    """A hand-made AUC interval below 0.5 for C2p withholds the H1' AUC route in verdicts.csv and report.md (the
    backstop); a hand-made non-meaningful AUC for C3q withholds nothing (ruling I3: diagnostic only)."""
    root = write_run(tmp_path, 10, seed=2, planted="own", strength=4.0)
    real = ap.evaluate

    def below_c2p(*args, **kwargs):
        result = real(*args, **kwargs)
        metrics = result["metrics"]
        if len(metrics):
            metrics["auc_high"] = np.where(metrics.model == "C2p", 0.4, np.maximum(metrics.auc_high, 0.6))
            metrics["auc_meaningful"] = metrics.model != "C3q"
        return result
    monkeypatch.setattr(ap, "evaluate", below_c2p)
    a = ap.compute(root, None, 10.0, 19, n_perm=0, skip_h9=True, single_draws=19)
    out = ap.write_analysis(a, "flat")
    table = pd.read_csv(out / "verdicts.csv")
    h1 = table[(table.setting == "main") & (table.hypothesis == "H1'")].iloc[0]
    assert h1.auc_route == "withheld: AUC interval below 0.5 for C2p"
    assert table[(table.setting == "main") & (table.hypothesis == "H8")].auc_route.iloc[0] == "used"
    assert "AUC route: withheld: AUC interval below 0.5 for C2p" in _section((out / "report.md").read_text(), 3)
    assert "not meaningful" not in " ".join(table.auc_route)


def test_skip_h9_writes_empty_h9_tables_and_says_so(tmp_path, capsys):
    root = write_run(tmp_path, 6, seed=3)
    assert ap.main(["--run", str(root), "--analysis-id", "quick", "--n-boot", "9", "--n-perm", "0", "--skip-h9",
                    "--single-draws", "19", *QUICK]) == 0
    out = root / "analysis" / "quick"
    assert sorted(p.name for p in out.iterdir()) == sorted(LISTED + [FIGURES])
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["h9"] == "skipped" and manifest["label"] == DEMONSTRATION
    for name in ("h9_all.csv", "h9_hard.csv", "h9_responsiveness.csv", "h9_summary.csv"):
        assert len(pd.read_csv(out / name)) == 0 and (out / name).read_text().strip(), name     # header only
    assert "H9 was skipped" in _section((out / "report.md").read_text(), 5)
    assert "absent: rating used" == manifest["stage_a_ranking"]                    # this run has no ranking
    assert STAGE_A_FALLBACK_NOTE in _section((out / "report.md").read_text(), 2)
    assert "6 pitches" in capsys.readouterr().out
    # final review I1: the fallback note reaches every metrics row, main and sensitivities
    metrics = pd.read_csv(out / "metrics.csv")
    assert STAGE_A_FALLBACK_NOTE == "rank score absent; rating used"
    assert len(metrics) == len(LADDER) and (metrics.note == STAGE_A_FALLBACK_NOTE).all()
    sens = pd.read_csv(out / "sensitivity_metrics.csv")
    run = sens[sens.sensitivity != "non_development"]
    assert set(run.sensitivity) == set(SETTINGS[1:]) - {"non_development"} and (run.note == STAGE_A_FALLBACK_NOTE).all()
    # final review I2: no development videos anywhere, so that sensitivity is not run and says so everywhere
    not_run = ap.NOT_RUN_IDENTICAL
    assert not_run == "not run: identical to main (no development video among the analysed videos)"
    assert list(sens[sens.sensitivity == "non_development"].note) == [not_run]
    deltas = pd.read_csv(out / "sensitivity_deltas.csv")
    assert list(deltas[deltas.sensitivity == "non_development"].note) == [not_run]
    verdicts = pd.read_csv(out / "verdicts.csv")
    assert set(verdicts[verdicts.setting == "non_development"].verdict) == {not_run}
    assert not_run in _section((out / "report.md").read_text(), 4)
    info = manifest["sensitivities"]["non_development"]
    assert info["status"] == not_run and info["excluded_videos"] == [] and info["selection_files"] == []
    assert "figure_1_model_ladder.png" in manifest["figures"]["files"]
    assert not_run in (out / FIGURES / "figure_1_caption.md").read_text()


def test_analyze_panel_writes_the_exchange_sensitivity_only_with_the_flag(tmp_path, capsys):
    """The exchange-based H9 outcomes are computed only with --h9-exchange-outcomes, at --h9-sensitivity-draws
    (499 by default), into their own table; report.md points at it in one sentence and states no direction."""
    from analysis.answer_level import H9_COLUMNS, SENSITIVITY_OUTCOMES

    defaults = ap.parser().parse_args(["--run", "r", "--analysis-id", "x"])
    assert defaults.h9_exchange_outcomes is False and defaults.h9_sensitivity_draws == 499
    root = write_run(tmp_path, 8, seed=3, planted="own", strength=4.0)
    common = ["--run", str(root), "--n-boot", "9", "--n-perm", "0", "--single-draws", "19", *QUICK]
    assert ap.main([*common, "--analysis-id", "plain"]) == 0
    plain = root / "analysis" / "plain"
    assert not (plain / "h9_exchange_sensitivity.csv").exists()
    assert sorted(p.name for p in plain.iterdir()) == sorted(LISTED + [FIGURES])
    manifest = json.loads((plain / "manifest.json").read_text())
    assert manifest["h9_exchange_sensitivity"] is None
    assert "h9_exchange_sensitivity.csv" not in _section((plain / "report.md").read_text(), 5)

    assert ap.main([*common, "--analysis-id", "flagged", "--h9-exchange-outcomes", "--h9-sensitivity-draws", "7"]) == 0
    out = root / "analysis" / "flagged"
    table = pd.read_csv(out / "h9_exchange_sensitivity.csv")
    assert list(table.columns) == ["questions", *H9_COLUMNS]
    assert set(table.questions) == {"all", "hard"} and set(table.outcome) == set(SENSITIVITY_OUTCOMES)
    assert set(table.model) == set(H9_MODELS)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["h9_exchange_sensitivity"]["draws"] == 7
    assert manifest["h9_exchange_sensitivity"]["outcomes"] == list(SENSITIVITY_OUTCOMES)
    assert "h9_exchange_sensitivity.csv" in manifest["outputs"]
    # the main H9 tables hold the frame outcomes only, and the summary states no direction for the sensitivity
    for name in ("h9_all.csv", "h9_hard.csv", "h9_responsiveness.csv", "h9_summary.csv"):
        assert not set(pd.read_csv(out / name).outcome) & set(SENSITIVITY_OUTCOMES), name
        pd.testing.assert_frame_equal(pd.read_csv(out / name), pd.read_csv(plain / name))   # the flag changes nothing else
    section = _section((out / "report.md").read_text(), 5)
    pointing = [line for line in section.splitlines() if "h9_exchange_sensitivity.csv" in line]
    sentence = pointing[0].strip() if len(pointing) == 1 else ""
    assert sentence.endswith(".") and ". " not in sentence                  # one line, one sentence
    assert not any(outcome in line for outcome in SENSITIVITY_OUTCOMES for line in section.splitlines()
                   if "h9_exchange_sensitivity.csv" not in line)          # no row, no direction, no verdict
    assert all(word not in sentence for word in ("positive", "negative", "supported", "direction"))
    # without H9 there is nothing to run the sensitivity on: the combination is refused, nothing is written
    assert ap.main([*common, "--analysis-id", "both", "--skip-h9", "--h9-exchange-outcomes"]) == 2
    assert "--h9-exchange-outcomes" in capsys.readouterr().err and not (root / "analysis" / "both").exists()
    # fix round 1, M5: a sensitivity without draws is refused
    for draws in ("0", "-5"):
        assert ap.main([*common, "--analysis-id", "nodraws", "--h9-exchange-outcomes",
                        "--h9-sensitivity-draws", draws]) == 2
        assert "--h9-sensitivity-draws must be at least 1" in capsys.readouterr().err
        assert not (root / "analysis" / "nodraws").exists()


def test_development_videos_come_from_a_selection_file_outside_the_run_and_it_is_hashed(tmp_path):
    """Final review I2: `--selection` given as a path outside the run names the development videos."""
    root = write_run(tmp_path, 8, seed=7)
    outside = tmp_path / "elsewhere" / "selection_all100.json"
    outside.parent.mkdir()
    outside.write_text(json.dumps({"video_ids": [f"v{i:03d}" for i in range(8)],
                                   "pilot_development_videos": ["v002", "v009"]}))
    assert ap.main(["--run", str(root), "--analysis-id", "sel", "--selection", str(outside), "--n-boot", "9",
                    "--n-perm", "0", "--skip-h9", "--single-draws", "19", *QUICK]) == 0
    out = root / "analysis" / "sel"
    manifest = json.loads((out / "manifest.json").read_text())
    info = manifest["sensitivities"]["non_development"]
    assert info["excluded_videos"] == ["v002"] and info["status"] == "estimated"
    assert info["selection_files"] == [str(outside)]
    assert manifest["inputs"][str(outside.resolve())] == _sha(outside)
    sens = pd.read_csv(out / "sensitivity_metrics.csv").set_index(["sensitivity", "model"])
    assert by_pitches(sens, "non_development") == 7 and by_pitches(sens, "rates") == 8


def by_pitches(table: pd.DataFrame, setting: str) -> int:
    return int(table.loc[(setting, "C1"), "n_pitches"])


def _drop_ranks(root: Path, videos: tuple[str, ...]) -> None:
    path = root / "ranking" / "scores.json"
    scores = json.loads(path.read_text())
    for block in scores.values():
        if isinstance(block, dict) and "pl_score" in block:
            block["pl_score"] = {v: s for v, s in block["pl_score"].items() if v not in videos}
    path.write_text(json.dumps(scores))


def test_a_partial_ranking_is_named_in_the_report_and_the_manifest(tmp_path):
    root = write_run(tmp_path, 8, seed=4, ranking=True)
    _drop_ranks(root, ("v001", "v002", "v003"))
    a = ap.compute(root, None, 10.0, 9, n_perm=0, skip_h9=True, single_draws=19)
    assert a.stage_a_ranking == "partial: 5 of 8"
    # with two or more pitches lacking a rank, every training fold lacks one: C2's columns are C1's
    assert all(not [c for c in f["models"]["C2"]["columns"] if c.startswith("sa_rank_")] for f in a.result["folds"])
    out = ap.write_analysis(a, "partial")
    assert json.loads((out / "manifest.json").read_text())["stage_a_ranking"] == "partial: 5 of 8"
    section = _section((out / "report.md").read_text(), 2)
    assert "partial: 5 of 8" in section and "C2 equals C1" in section and "one fold only" not in section


def test_one_pitch_without_a_rank_leaves_the_rank_column_in_one_fold_only(tmp_path):
    root = write_run(tmp_path, 8, seed=4, ranking=True)
    _drop_ranks(root, ("v004",))
    a = ap.compute(root, None, 10.0, 9, n_perm=0, skip_h9=True, single_draws=19)
    assert a.stage_a_ranking == "partial: 7 of 8"
    used = [f["held_out"] for f in a.result["folds"]
            if [c for c in f["models"]["C2"]["columns"] if c.startswith("sa_rank_")]]
    assert used == ["v004"]                                     # only the fold that holds that pitch out
    section = _section(pr.report_markdown(a), 2)
    assert "used in one fold only" in section and "C2 equals C1" not in section


def test_with_shark_level_the_archived_run_is_written_under_its_folder(tmp_path):
    root = write_run(tmp_path, 12, seed=5, planted="own", strength=4.0)
    assert ap.main(["--run", str(root), "--analysis-id", "sharks", "--n-boot", "9", "--n-perm", "0", "--skip-h9",
                    "--single-draws", "19", "--with-shark-level", *QUICK]) == 0
    out = root / "analysis" / "sharks"
    assert sorted(p.name for p in out.iterdir()) == sorted(LISTED + [FIGURES, SHARK_LEVEL])
    metrics = pd.read_csv(out / SHARK_LEVEL / "metrics.csv")
    assert list(metrics.model) == ["B0", "B1", "B2", "B3", "B4a", "B4", "B5", "B5d"] and "within_auc" in metrics
    f1 = pd.read_csv(out / SHARK_LEVEL / "f1.csv")
    assert f1.model_or_comparison.tolist() == ["I0", "I3 minus I2", "IR minus I4", "I5 minus I2", "I5 minus I2"]
    assert {"identity_metrics.csv", "identity_deltas.csv", "deltas.csv"} <= {p.name for p in (out / SHARK_LEVEL).iterdir()}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["with_shark_level"] is True and manifest["shark_level"]["folder"] == SHARK_LEVEL
    assert manifest["figures_skipped"] == {"9": "H9 skipped (--skip-h9)"}                # S1 is drawn
    assert {"figure_S1_shark_level_ladder.png", "figure_S1_caption.md"} <= set(manifest["figures"]["files"])
    assert sorted(manifest["shark_level"]["tables"]) == sorted(p.name for p in (out / SHARK_LEVEL).iterdir())
    assert all(f"{SHARK_LEVEL}/{p.name}" in manifest["outputs"] for p in (out / SHARK_LEVEL).iterdir())


def test_workers_default_to_all_cores_but_one():
    parser = ap.parser()
    assert parser.parse_args(["--run", "x", "--analysis-id", "y"]).workers == ap.default_workers()
    assert ap.default_workers() >= 1


# ---- the generic behaviour of the writer ------------------------------------------------------------

def test_an_existing_analysis_is_not_overwritten_without_force(tmp_path, capsys):
    root = write_run(tmp_path, 5, seed=6)
    args = ["--run", str(root), "--analysis-id", "a1", "--n-boot", "9", "--n-perm", "0", "--skip-h9",
            "--single-draws", "19", *QUICK]
    assert ap.main(args) == 0
    out = root / "analysis" / "a1"
    before = _snapshot(out)
    assert ap.main(args) == 2 and "--force" in capsys.readouterr().err and _snapshot(out) == before
    (out / "stale.csv").write_text("left over")
    assert ap.main([*args, "--force"]) == 0 and not (out / "stale.csv").exists()


@pytest.mark.parametrize("bad", ["../escape", "a/b", "", ".", ".."])
def test_an_analysis_id_cannot_leave_the_analysis_folder(tmp_path, bad, capsys):
    root = write_run(tmp_path, 2, seed=6)
    before = _snapshot(root)
    assert ap.main(["--run", str(root), "--analysis-id", bad]) == 2
    assert "analysis id" in capsys.readouterr().err and _snapshot(root) == before


def test_missing_run_is_reported_not_created(tmp_path, capsys):
    assert ap.main(["--run", str(tmp_path / "nope"), "--analysis-id", "a1"]) == 2
    assert "not found" in capsys.readouterr().err and not (tmp_path / "nope").exists()


def test_compute_reads_only_and_one_video_still_writes_every_file(tmp_path):
    root = write_run(tmp_path, 3, seed=7)
    before = _snapshot(root)
    a = ap.compute(root, ["v000"], 10.0, 9, n_perm=0, skip_h9=True, single_draws=19)
    assert _snapshot(root) == before and a.result["status"].startswith("not estimable")
    out = ap.write_analysis(a, "one")
    assert sorted(p.name for p in out.iterdir()) == sorted(LISTED + [FIGURES])
    assert (out / "metrics.csv").read_text().startswith("model,")
    assert len(pd.read_csv(out / "verdicts.csv")) == 2 * len(SETTINGS)
    with pytest.raises(ap.ConfigError, match="already exists"):
        ap.write_analysis(a, "one")
    with pytest.raises(ap.ConfigError, match="plain folder name"):
        ap.write_analysis(a, "../outside", force=True)


def test_reviewed_videos_with_open_issues_are_printed_and_listed_but_not_excluded(tmp_path, capsys):
    root = write_run(tmp_path, 4, seed=3)
    primary = json.loads((root / "v001" / "results" / "primary.json").read_text())["record"]
    write_video(root, "v001", primary, reviewed=primary)
    (root / "v001" / "results" / "reviewed_validation.json").write_text(
        json.dumps({"errors": ["[decision.offer] decision 3: no such offer"], "flags": []}))
    assert ap.main(["--run", str(root), "--analysis-id", "open", "--n-boot", "9", "--n-perm", "0", "--skip-h9",
                    "--single-draws", "19", *QUICK]) == 0
    printed = capsys.readouterr().out
    assert "reviewed with open issues: v001 (1 validation errors, 0 structural judgments on decisions)" in printed
    out = root / "analysis" / "open"
    d = json.loads((out / "descriptives.json").read_text())
    assert d["reviewed_with_open_issues"]["videos"] == {
        "v001": {"reviewed_validation_errors": 1, "review_structural_on_decisions": 0}}
    assert set(pd.read_csv(out / "features.csv").video_id) == {"v000", "v001", "v002", "v003"}


def test_a_run_without_agreement_reports_still_exports(tmp_path):
    root = tmp_path / "bare"
    rng = np.random.default_rng(5)
    for i in range(3):
        write_video(root, f"v{i}", make_record(rng))
        (root / f"v{i}" / "agreement.json").unlink()
    assert ap.main(["--run", str(root), "--analysis-id", "x", "--n-boot", "9", "--n-perm", "0", "--skip-h9",
                    "--single-draws", "19", *QUICK]) == 0
    d = json.loads((root / "analysis" / "x" / "descriptives.json").read_text())
    assert d["self_consistency"]["n_videos"] == 0 and d["self_consistency"]["mean"] == {}


def test_frame_table_uses_the_first_linked_question_that_has_a_frame():
    turns = pd.DataFrame({
        "video_id": ["v"] * 6, "turn_id": [1, 2, 3, 4, 5, 6],
        "function": ["question", "question", "answer", "answer", "answer", "accepts"],
        "frame": [None, "prevention", "promotion", "prevention", None, None],
        "responds_to": [(), (), (1, 2), (1,), (), (3,)]})
    assert ap._frame_table(turns) == {"prevention": {"promotion": 1}, "no_frame": {"prevention": 1},
                                      "no_linked_question": {"no_frame": 1}}


def test_the_documented_leakage_rule_names_the_pitch_cut_only():
    assert "first decision of any kind by any investor" in ap.LEAKAGE_RULE and "investor's first offer" not in ap.LEAKAGE_RULE
    assert "first decision of any kind by any investor" in ap.__doc__


def test_the_report_names_the_h9_models_without_an_interval(exported):
    _, out, _, _ = exported
    hard = pd.read_csv(out / "h9_hard.csv")
    section = _section((out / "report.md").read_text(), 5)
    for r in hard[hard.estimable != "yes"].drop_duplicates(["model", "outcome"]).itertuples():
        assert f"hard questions / {r.outcome} ({r.estimable})" in section
    claimed = pd.read_csv(out / "h9_summary.csv")
    claimed = claimed[claimed.hard_and_all_agree]
    every = pd.read_csv(out / "h9_all.csv")
    for r in claimed.itertuples():          # every direction claimed rests on estimable models with >= 2 pitches
        for frame in (every, hard):
            rows = frame[(frame.model == r.model) & (frame.outcome == r.outcome) & (frame.predictor == r.predictor)]
            assert (rows.estimable == "yes").all() and (rows.n_pitches_informative >= 2).all()
