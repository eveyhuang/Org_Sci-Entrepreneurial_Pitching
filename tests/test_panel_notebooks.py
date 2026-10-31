"""The two panel notebooks: what is committed, and that they run on a synthetic run without writing.

What is committed. The notebooks are pinned to the selection file of the real run and were executed on it. The checks
below do not hard-code the results: the selection file and the analysis id are read from the notebook's own
configuration cell, and the numbers the saved outputs must show are derived from the selection file, from the files of
the export `runs/<run>/analysis/<ANALYSIS_ID>/` (written by the command line with the same settings) and from the
report functions applied to those files. The reported run is tracked in the repository. Optional video media and
PDF duplicates are not required. A checkout missing the run skips the empirical checks with an explanation.

Synthetic runs. Each notebook is also executed in a temporary copy whose configuration cell is re-pointed at a
synthetic run (papermill-style: the assignments of the committed cell are replaced, so a renamed setting fails here).
Nothing empirical is asserted there.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from html import escape
from pathlib import Path

import pandas as pd
import pytest

import analysis.panel_explorer as px
from analysis import panel_report as pr
from analysis.panel_bottleneck import LADDER
from analysis import pitch_figures as pf
from tests.panel_analysis_fixtures import write_ranking, write_run, write_video

REPO = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO / "notebooks" / "stage_B"
EXPLORER = NOTEBOOKS / "panel_run_explorer.ipynb"
ANALYSIS = NOTEBOOKS / "project_v3_panel_analysis.ipynb"
SETTINGS = {
    EXPLORER: ("RUN_NAME", "VIDEO_IDS", "VIDEO_ID", "PASS", "SHOW_FULL_PROMPT", "FREQUENCY_PASS", "INPUT_USD_PER_M",
               "OUTPUT_USD_PER_M", "EXPORT"),
    ANALYSIS: ("RUN_NAME", "VIDEO_IDS", "ANALYSIS_ID", "PENALTY", "BOOTSTRAP_REPS", "PERMUTATIONS",
               "SINGLE_FEATURE_DRAWS", "RANDOM_SEED", "RUN_H9", "H9_EXCHANGE_OUTCOMES", "H9_SENSITIVITY_DRAWS",
               "WITH_SHARK_LEVEL", "WORKERS", "EXPORT", "SAVE_FIGURES", "FIGURE_WIDTH", "SINGLE_FEATURE_TOP_N",
               "SINGLE_FEATURE_SHOW_ALL"),
}
SECTIONS = ("## 1. Provenance", "## 2. Sample and outcomes", "## 3. The cut, made visible", "## 4. Features",
            "## 5. Descriptives", "## H4. What the panel does after an answer",
            "## 6. Self-consistency and human agreement", "## 7. The ladder", "## 8. Verdicts", "## 9. Sensitivities",
            "## 10. Each concept on its own", "## 11. H9: a prepared answer and what the panel asks next",
            "## 12. Inspect one prediction", "## 13. Limits", "## 14. Exports", "## 15. Figures")
RUN_NAME = "full-corpus-100"
RUN_DIR = REPO / "runs" / RUN_NAME
SELECTION = "selection_all100.json"
# the export the committed notebook corresponds to and the selection file it is pinned to (relative to the run)
EXPORT_DIR = RUN_DIR / "analysis" / "all100"
needs_run = pytest.mark.skipif(not ((EXPORT_DIR / "manifest.json").is_file() and (RUN_DIR / SELECTION).is_file()),
                               reason=f"runs/{RUN_NAME}/analysis/all100 or the selection file {SELECTION} is not present "
                                      "(the released run must be present to compare its saved outputs)")
PROBE = ("import sys\n"
         "print('PROBE sdk loaded:', any(m == 'google' or m.startswith(('google.genai', 'google.generativeai')) "
         "for m in sys.modules))")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _source(cell) -> str:
    return "".join(cell["source"])


def _code(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _setting(path: Path, name: str):
    """The value a committed notebook's configuration cell gives a setting (a plain literal)."""
    found = re.search(rf"^{name} = ([^#\n]*)", _source(_code(_read(path))[0]), flags=re.M)
    assert found, f"{name} is not assigned in the configuration cell of {path.name}"
    return json.loads(found.group(1).strip().replace("False", "false").replace("True", "true").replace("None", "null"))


def _selection(path: Path = ANALYSIS) -> dict:
    """The selection file the committed notebook is pinned to, read from its own configuration cell."""
    return _read(RUN_DIR / _setting(path, "VIDEO_IDS"))


def _selected(path: Path = ANALYSIS) -> set[str]:
    return {v["video_id"] for v in _selection(path)["videos"]}


def _pilot() -> set[str]:
    return set(_selection()["pilot_development_videos"])


def _annotated() -> set[str]:
    """Every video folder of the run with two complete passes."""
    return {p.name for p in RUN_DIR.glob("st_*")
            if (p / "results" / "primary.json").exists() and (p / "results" / "replicate.json").exists()}


class Export:
    """The export the analysis notebook corresponds to: runs/<run>/analysis/<ANALYSIS_ID>/."""

    def __init__(self):
        self.root = RUN_DIR / "analysis" / _setting(ANALYSIS, "ANALYSIS_ID")
        self.manifest = _read(self.root / "manifest.json")

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / name)

    def json(self, name: str) -> dict:
        return _read(self.root / name)


@pytest.fixture(scope="module")
def export() -> Export:
    return Export()


def _printed(code: list[dict]) -> str:
    return "\n".join("".join(o.get("text", "")) for c in code for o in c["outputs"] if o.get("output_type") == "stream")


def _html(code: list[dict]) -> str:
    return "\n".join("".join(o["data"].get("text/html", "")) for c in code for o in c["outputs"] if "data" in o)


def _video_ids_shown(code: list[dict], run_level: tuple[str, ...] = ()) -> set[str]:
    """Every video id in the saved outputs, leaving out the cells whose source contains one of `run_level`."""
    cells = [c for c in code if not any(marker in _source(c) for marker in run_level)]
    return set(re.findall(r"st_[0-9a-f]{10}", _printed(cells) + "\n" + _html(cells)))


def _cells_after(nb: dict, heading: str) -> list[dict]:
    """The code cells of one section: from its heading to the next `## ` heading."""
    cells = nb["cells"]
    start = next(i for i, c in enumerate(cells) if _source(c).startswith(heading))
    end = next((i for i in range(start + 1, len(cells)) if _source(cells[i]).startswith("## ")), len(cells))
    return [c for c in cells[start:end] if c["cell_type"] == "code"]


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def configure(nb: dict, **settings) -> dict:
    """The notebook with assignments of its configuration cell replaced; every name must be assigned there once.

    The committed notebooks are pinned to a selection file of the real run, which a synthetic run does not have:
    `VIDEO_IDS` is set to None unless a test names videos. The analysis notebook gets a few draws and permutations
    instead of the committed 1999, 500 and 9999, and one worker, to keep these runs short.
    """
    cell = _code(nb)[0]
    source = _source(cell)
    quick = ({"BOOTSTRAP_REPS": 19, "PERMUTATIONS": 9, "SINGLE_FEATURE_DRAWS": 49, "H9_SENSITIVITY_DRAWS": 9,
              "WORKERS": 1} if re.search(r"^PERMUTATIONS = ", source, flags=re.M) else {})
    for name, value in {"VIDEO_IDS": None, **quick, **settings}.items():
        source, n = re.subn(rf"^{name} = [^#\n]*", lambda _: f"{name} = {value!r}  ", source, flags=re.M)
        assert n == 1, f"{name} is assigned {n} times in the configuration cell"
    cell["source"] = source
    return nb


def execute(nb: dict, tmp_path: Path, name: str = "copy.ipynb") -> tuple[list[dict], str, str]:
    """Run a notebook copy with a probe cell appended: (code cells, everything printed, all HTML shown)."""
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": PROBE, "outputs": [], "execution_count": None})
    copy = tmp_path / name
    copy.write_text(json.dumps(nb))
    done = subprocess.run([sys.executable, str(REPO / "scripts/analysis/execute_notebook.py"), str(copy),
                           "--timeout", "600"], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-4000:]
    code = _code(_read(copy))
    assert all(c["execution_count"] for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
    printed, html = _printed(code), _html(code)
    assert "PROBE sdk loaded: False" in printed            # no provider SDK is ever imported
    return code, printed, html


def _images(code: list[dict]) -> int:
    return sum(1 for c in code for o in c["outputs"] if "image/png" in o.get("data", {}))


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Twenty-four pitches with rank scores (both outcomes occur, so every model figure can be drawn); v002 is
    human-reviewed."""
    root = write_run(tmp_path_factory.mktemp("notebooks"), 24, seed=1, planted="own", ranking=True, strength=4.0)
    primary = _read(root / "v002" / "results" / "primary.json")["record"]
    write_video(root, "v002", primary, reviewed=primary)
    return root


# ---- what is committed ----------------------------------------------------------------------------

@pytest.mark.parametrize("path", [EXPLORER, ANALYSIS], ids=["explorer", "analysis"])
def test_notebook_is_committed_with_the_configuration_cell_first(path):
    nb = _read(path)
    cells, code = nb["cells"], _code(nb)
    config = code[0]
    assert "parameters" in config["metadata"].get("tags", [])
    assert sum("parameters" in c["metadata"].get("tags", []) for c in code) == 1
    text = _source(config)
    assert text.startswith("# ===== CONFIGURATION") and "the only cell you need to edit" in text
    for name in SETTINGS[path]:
        assert re.search(rf"^{name} = ", text, flags=re.M), name
        assert re.search(rf"^{name} = .*#|^# .*{name}", text, flags=re.M), f"{name} has no explanation"
    assert re.search(r"^EXPORT = False\b", text, flags=re.M)                    # opening and running it writes nothing
    assert re.search(rf'^RUN_NAME = "{RUN_NAME}"', text, flags=re.M)
    assert _setting(path, "VIDEO_IDS") == SELECTION
    comments = " ".join(line.lstrip("# ") for line in text.splitlines() if line.startswith("#"))
    assert f'"{SELECTION}" is all 100 annotated videos' in comments and "development (pilot) videos" in comments
    assert "re-annotated them with release v3-dev9" in comments and "a list of video ids" in comments
    for video_id in ("st_38f8bf18e0", "st_8642720802", "st_aebb7e3764", "st_d218d68cff", "st_ea076612ce"):
        assert video_id in comments
    first = " ".join(_source(cells[0]).split())
    assert cells[0]["cell_type"] == "markdown" and "How to use this notebook" in first
    for phrase in (".venv", "configuration cell", "Restart & Run All", "no model calls", "never changes annotation files"):
        assert phrase in first, phrase
    assert "closing" not in " ".join(_source(c).lower() for c in cells)        # no closing state since v3-dev9


@needs_run
@pytest.mark.parametrize("path", [EXPLORER, ANALYSIS], ids=["explorer", "analysis"])
def test_notebook_was_executed_on_its_selection_without_errors(path):
    code = _code(_read(path))
    assert all(c["execution_count"] for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
    assert sum(bool(c["outputs"]) for c in code) >= len(code) - 2
    selection = _selection(path)
    ids = [v["video_id"] for v in selection["videos"]]
    assert len(ids) == len(set(ids)) == 100 and set(ids) == _annotated()
    assert len(_pilot()) == 5 and _pilot() <= set(ids)


@pytest.mark.parametrize("path", [EXPLORER, ANALYSIS], ids=["explorer", "analysis"])
def test_notebook_code_is_thin_and_every_section_is_introduced_in_plain_language(path):
    nb = _read(path)
    cells, code = nb["cells"], _code(nb)
    everything = "\n".join(_source(c) for c in code)
    for forbidden in ("def ", "lambda", "class ", "providers", "--execute", ".to_csv", "write_text", "open(", "savefig",
                      "import pipeline", "from pipeline", "subprocess", "os.system", "requests"):
        assert forbidden not in everything, forbidden
    assert not any(line.lstrip().startswith(("%", "!")) for line in everything.splitlines())
    for cell in code[1:]:
        lines = [line for line in _source(cell).splitlines() if line.strip() and not line.lstrip().startswith("#")]
        assert len(lines) <= 12, _source(cell)
    sections = [i for i, c in enumerate(cells) if c["cell_type"] == "markdown" and _source(c).startswith("## ")]
    assert len(sections) >= 7
    for i in sections:
        text = _source(cells[i])
        if "Limits" in text.split("\n")[0] or "export" in text.split("\n")[0].lower():
            continue
        assert "What this shows" in text or "The idea" in text, text[:80]
        assert re.search(r"What it does not (show|establish)", text), text[:80]
    assert len(_source(cells[0]).split()) > 120


def test_analysis_notebook_has_the_pitch_level_sections_in_order():
    nb = _read(ANALYSIS)
    cells = nb["cells"]
    headings = [_source(c).split("\n")[0] for c in cells if c["cell_type"] == "markdown" and _source(c).startswith("## ")]
    assert tuple(headings) == SECTIONS
    for c in cells:
        text = _source(c)
        if c["cell_type"] == "markdown" and text.startswith("## ") and not text.startswith(("## 13.", "## 14.")):
            assert "**Unit and denominator.**" in text, text[:60]
    code = "\n".join(_source(c) for c in _code(nb))
    for call in ("pr.compute(RUN, VIDEO_IDS, PENALTY, BOOTSTRAP_REPS, RANDOM_SEED, PERMUTATIONS, SINGLE_FEATURE_DRAWS, "
                 "WORKERS,\n               RUN_H9, H9_EXCHANGE_OUTCOMES, H9_SENSITIVITY_DRAWS, WITH_SHARK_LEVEL)",
                 "pr.provenance_table(A)", "pr.settings_table(A)", "pr.outcome_table(A)", "pr.history_view(A, video)",
                 "pr.feature_row(A, video)", "pr.descriptive_tables(A)", 'D["prepared_ratings"]', 'D["ask"]',
                 "pr.h4_tables(A)", "pr.h4_shares_view(A)", "pr.self_consistency_table(A)", "pr.ladder_guide(A)",
                 "pr.metrics_table(A.result)", "pr.delta_table(A.result)", "pr.verdict_view(A)", "pr.verdict_text(A)",
                 "pr.sensitivity_view(A)", "pr.single_feature_view(A, SINGLE_FEATURE_TOP_N, SINGLE_FEATURE_SHOW_ALL)",
                 "pr.coefficient_view(A)", "pr.by_offer_view(A)", "pr.h9_directions(A)", "pr.h9_view(A)",
                 "pr.h9_sensitivity_note(A)", "pr.prediction_view(A, model, video)", "pr.limits_table(A)",
                 "pr.export(A, ANALYSIS_ID, FIGURE_WIDTH)", "if SAVE_FIGURES and not EXPORT:",
                 "pr.save_figures(A, ANALYSIS_ID, FIGURE_WIDTH)", "pr.files_note(A, ANALYSIS_ID, EXPORT or SAVE_FIGURES)",
                 *(f"pf.figure(A, {json.dumps(n)}, FIGURE_WIDTH)" for n in pf.FIGURES),
                 *(f"pf.caption(A, {json.dumps(n)}, FIGURE_WIDTH)" for n in pf.FIGURES)):
        assert call in code, call
    words = " ".join(" ".join(_source(c).split()) for c in cells if c["cell_type"] == "markdown")
    for phrase in ("first decision of any kind by any investor", "strictly before", "model sensitivity, not a causal effect",
                   "**H9 is reported as an exploratory pattern, not a set of findings**",
                   "no multiple-test correction", "**descriptive ranking, not a test**", "not a second test",
                   "is not evidence of no effect", "not comparable with the AUCs of the v3-dev8 analyses",
                   "*in turn order, whatever exchange the model put each turn in*", "**not a finding**",
                   "it measures stability, not correctness", "`h4_support_shares.csv`", "`h9_exchange_sensitivity.csv`",
                   "**This table uses all ranked videos of the run, not only those selected in `VIDEO_IDS`**",
                   "every finite AUC is drawn", "C2p (C2′)", "Gemini only, not human-coded"):
        assert phrase in words, phrase


@needs_run
def test_the_export_was_made_with_the_notebooks_settings(export):
    """The export the notebook's numbers are checked against is the notebook's own analysis: same videos, draws,
    permutations, seed, penalty, H9 settings and shark-level run. (The command line wrote it; the notebook writes
    nothing.)"""
    m = export.manifest
    assert m["video_ids"] == sorted(_selected()) and m["analysis_id"] == _setting(ANALYSIS, "ANALYSIS_ID")
    assert m["n_boot"] == _setting(ANALYSIS, "BOOTSTRAP_REPS") == 1999
    assert m["n_perm"] == _setting(ANALYSIS, "PERMUTATIONS") == 500
    assert m["single_draws"] == _setting(ANALYSIS, "SINGLE_FEATURE_DRAWS") == 9999
    assert m["seed"] == _setting(ANALYSIS, "RANDOM_SEED") and m["penalty"] == _setting(ANALYSIS, "PENALTY") == "inner_cv"
    assert m["h9"] == "run" and _setting(ANALYSIS, "RUN_H9") is True
    assert m["h9_exchange_sensitivity"]["draws"] == _setting(ANALYSIS, "H9_SENSITIVITY_DRAWS")
    assert (m["shark_level"] is not None) == _setting(ANALYSIS, "WITH_SHARK_LEVEL") is True
    assert m["release_ids"] == ["v3-dev9"] and m["stage_a_ranking"] == "complete"
    assert m["figures"]["width"] == _setting(ANALYSIS, "FIGURE_WIDTH") and _setting(ANALYSIS, "SAVE_FIGURES") is False
    assert m["counts"]["pitches"] == len(_selected()) == len(export.csv("videos.csv"))
    # The release retains PNGs/captions; PDF duplicates may be omitted by .gitignore.
    # If PDFs exist locally they must still belong to the recorded export.
    declared = set(m["figures"]["files"])
    present = {p.name for p in (export.root / "figures").iterdir()}
    required = {name for name in declared if not name.endswith(".pdf")}
    assert required <= present <= declared
    assert m["figures_skipped"] == {} and len(m["figures"]["files"]) == 3 * len(pf.FIGURES)   # PNG, PDF and caption
    assert m["sensitivities"]["non_development"]["excluded_videos"] == sorted(_pilot())


@needs_run
def test_analysis_notebook_shows_the_sample_of_its_selection(export):
    nb = _read(ANALYSIS)
    code = _code(nb)
    printed, html = _printed(code), _html(code)
    c, n = export.manifest["counts"], len(_selected())
    caption = (f"n pitches = {n} ({c['pitches_with_offer']} with an offer, {c['pitches_without_offer']} without, "
               f"{c['pitches_unknown']} unknown)  [pitch-held-out estimate]")
    assert f"Run: {RUN_NAME}\n{caption}" in printed
    sources = export.csv("videos.csv").source.value_counts()
    assert f"{n} videos: {n} model-coded only, 0 human-reviewed." in printed and sources.sum() == n
    replicate = int(sources.get("replicate", 0))
    assert (f"For {replicate} of them the replicate pass is read" in printed) == bool(replicate)
    assert pr.EXPORT_NOTE in printed
    shown = _video_ids_shown(code)
    assert shown and shown <= _selected()                                      # nothing but the selected videos
    # H4: the caption and the reading from the exported counts
    h4 = export.json("h4_tables.json")
    assert h4["n_pitches"] == n
    assert (f"n pitches = {n}, n answers to an investor question = {h4['n_rows']} ({h4['n_rows_hard']} of them to a "
            "hard question") in printed
    tables = {name: pd.DataFrame(h4["tables"][name]) for name in ("asker", "panel")}
    for line in pr.h4_reading(tables).split("\n"):
        assert line in printed, line
    challenged = int(tables["asker"].set_index("next_move")["all"]["challenge"])
    assert f"{challenged} answers match (the asker's next move: asker challenges; all questions; all videos)." in printed
    # self-consistency and the stability of the ranking, over the selected videos and over the whole run
    compared = int(export.csv("self_consistency.csv").n_videos.iloc[0])
    assert f"n videos with two passes compared = {compared} of {n}" in printed and compared == n
    for field in ("Whether the pitch got any offer", "Answer substance (PREP-SUBS)", "Matched with the same function"):
        assert field in html, field
    stability = export.csv("stage_a_ranking_agreement.csv")
    column = stability.batch_to_batch_correlation_of_standing
    low, high = stability.loc[column.idxmin()], stability.loc[column.idxmax()]
    assert f"between {column.min():.2f} ({low.dimension}) and {column.max():.2f} ({high.dimension})" in printed


@needs_run
def test_analysis_notebook_shows_the_ladder_verdicts_and_sensitivities_of_its_export(export):
    nb = _read(ANALYSIS)
    code = _code(nb)
    printed, html = _printed(code), _html(code)
    metrics = export.csv("metrics.csv").set_index("model")
    assert list(metrics.index) == list(LADDER)
    for model in metrics.index:
        assert f">{round(float(metrics.skill_pct[model]), 1)}<" in html, model
        for column in ("perm_p_log_loss", "perm_p_auc"):
            text = escape(pr.p_text(metrics[column][model]))
            assert text == "" or f">{text}<" in html, (model, column)
    best = metrics.log_loss.idxmin()
    assert f"Lowest log loss: {best} ({metrics.log_loss[best]:.3f}, skill {metrics.skill_pct[best]:+.1f}%)." in printed
    assert "HOW TO READ THESE NUMBERS" in printed
    deltas = export.csv("deltas.csv")
    for r in deltas[deltas.metric == "skill"].itertuples():
        assert pr.delta_reading(r.comparison, r.metric, r.delta, r.ci_low, r.ci_high, r.p_value) in printed, r.comparison
    # section 8 is the verdicts of the export, with their numbers and power sentence, as report.md states them
    section = _printed(_cells_after(nb, "## 8. Verdicts"))
    verdicts = export.csv("verdicts.csv")
    main = verdicts[verdicts.setting == "main"]
    report = (export.root / "report.md").read_text()
    lines = report.splitlines()
    for r in main.itertuples():
        at = next(i for i, x in enumerate(lines) if x.startswith("**Verdict: ") and f"{r.auc_delta:+.3f}" in x)
        notes = lines[at + 2]                                       # which route passed, and the gain from 0.5
        assert lines[at] in section and notes in section and r.power in section, r.hypothesis
        assert notes.startswith(("Passes on ", "Neither route passes.")) and "cannot be told from zero" in r.power
        assert f">{r.verdict}<" in _html(_cells_after(nb, "## 8. Verdicts"))
    # section 9: one row per setting with each verdict, the main analysis first
    sens_html = _html(_cells_after(nb, "## 9. Sensitivities"))
    for setting in verdicts.setting.unique():
        assert f">{setting}<" in sens_html, setting
    assert "not a second test" in _printed(_cells_after(nb, "## 9. Sensitivities"))


@needs_run
def test_analysis_notebook_shows_each_concept_h9_and_the_limits_of_its_export(export):
    nb = _read(ANALYSIS)
    single = export.csv("single_feature.csv")
    single = single.assign(note=single.note.fillna(""))
    concepts = _printed(_cells_after(nb, "## 10. Each concept on its own"))
    assert pr.fdr_sentence(single) in concepts and f"{len(single)} features were ranked." in concepts
    coefficients = export.csv("coefficients.csv")
    assert f"Model C5 refitted on all {int(coefficients.n_rows.iloc[0])} pitches with a known outcome" in concepts
    # H9: every direction of h9_summary.csv and the count of rows without one; the sensitivity is only pointed at
    summary = export.csv("h9_summary.csv")
    h9 = _printed(_cells_after(nb, "## 11. H9: a prepared answer and what the panel asks next"))
    claimed = summary[summary.hard_and_all_agree]
    assert (f"{len(claimed)} of {len(summary)} predictor rows" in h9
            and f"{int((summary.direction == 'not distinguishable').sum())} are not distinguishable." in h9)
    h9_html = _html(_cells_after(nb, "## 11. H9: a prepared answer and what the panel asks next"))
    for r in claimed.itertuples():                  # each claimed direction, set against the hypothesis
        head = f"{r.model} / {r.outcome} / {pr.H9_PREDICTOR_WORDS.get(r.predictor, r.predictor)}: {r.direction}"
        line = next(x for x in h9.splitlines() if head in x)
        assert any(words in line for words in ("against the hypothesis", "in the hypothesised direction",
                                               "states no direction")), line
    assert "Not claimed: in " in h9 and "H9 is an exploratory pattern, not a finding" in h9
    assert "ctx (Gemini only, not human-coded)" in h9_html
    ladder = _html(_cells_after(nb, "## 7. The ladder"))
    assert "pres_ctx (Gemini only, not human-coded)" in ladder and "a_ctx_mean (Gemini only, not human-coded)" in ladder
    draws = export.manifest["h9_exchange_sensitivity"]["draws"]
    assert f"unreported sensitivity ({draws} draws)" in h9 and "h9_exchange_sensitivity.csv" in h9
    assert "next_question_same_exchange" not in h9 + h9_html and "further_questions_in_exchange" not in h9 + h9_html
    # limits: plan v4 section 11 with this run's count, then the method's
    limits = _html(_cells_after(nb, "## 13. Limits"))
    without = export.manifest["counts"]["pitches_without_offer"]
    assert f"{without} pitches without an offer" in limits and "residual leave-one-out bias" in limits


@needs_run
def test_analysis_notebook_embeds_the_figures_with_the_captions_of_the_export(export):
    nb = _read(ANALYSIS)
    cells = nb["cells"]
    start = next(i for i, c in enumerate(cells) if _source(c).startswith("## 15. Figures"))
    assert not any(_source(c).startswith("## ") for c in cells[start + 1:])
    section = [c for c in cells[start:] if c["cell_type"] == "code"]
    assert len(section) == len(pf.FIGURES) + 1
    width = _setting(ANALYSIS, "FIGURE_WIDTH")
    for cell, number in zip(section, pf.FIGURES):
        key = json.dumps(number)
        assert _source(cell) == f"fig = pf.figure(A, {key}, FIGURE_WIDTH)\nprint(pf.caption(A, {key}, FIGURE_WIDTH))"
        images = [o for o in cell["outputs"] if "image/png" in o.get("data", {})]
        assert len(images) == 1 and len(images[0]["data"]["image/png"]) > 20000, number
        caption = (export.root / "figures" / pf.file_names(number)["caption"]).read_text().strip()
        assert caption in _printed([cell]), number                # the notebook and the export are the same analysis
    assert width == export.manifest["figures"]["width"] and _images(_code(nb)) == len(pf.FIGURES)
    assert "Gemini only, not human-coded" in (export.root / "figures" / pf.file_names(9)["caption"]).read_text()
    last = section[-1]
    out = _printed([last])
    assert out.startswith("EXPORT and SAVE_FIGURES are False: nothing was written.")
    assert f"The {len(pf.FIGURES)} figures listed below are already in runs/{RUN_NAME}/analysis/{export.root.name}/figures/" in out
    for name in export.manifest["figures"]["files"]:
        if name.endswith(".png"):
            assert name in _html([last]), name
    assert str(REPO) not in _html(_code(nb)) + _printed(_code(nb))           # no absolute path of this machine


@needs_run
def test_explorer_notebook_has_its_sections_and_real_outputs(export):
    nb = _read(EXPLORER)
    headings = [_source(c).split("\n")[0] for c in nb["cells"]
                if c["cell_type"] == "markdown" and _source(c).startswith("## ")]
    for letter, word in zip("ABCDEFGH", ("Run overview", "instrument as the model saw it", "One video's record",
                                         "Agreement between the two passes", "Human review", "Stage A ranking",
                                         "Comparison", "export")):
        assert any(h.startswith(f"## {letter}. ") and word.lower() in h.lower() for h in headings), word
    code = _code(nb)
    printed, html = _printed(code), _html(code)
    assert _setting(EXPLORER, "VIDEO_IDS") == _setting(ANALYSIS, "VIDEO_IDS")
    n = len(_selected(EXPLORER))
    sources = export.csv("videos.csv").source.value_counts()
    assert f"Run: {RUN_NAME}" in printed and f"{n} videos; instrument release v3-dev9" in printed
    reads = ", ".join(f"the {source} pass of {int(count)}" for source, count in sources.items())
    assert f"The analysis reads: {reads}" in printed
    assert "matches the hash recorded for the run" in printed and "review_panel.py export --run" in printed
    assert pr.EXPORT_NOTE in printed
    assert _images(code) == 1 + len(px.VALUE_GROUPS)                    # the timeline and section A2's charts
    video = _setting(EXPLORER, "VIDEO_ID")
    assert f"Selected video: {video}" in printed and video in _selected(EXPLORER)
    # the prepared presentation's content and the ask, per pass (plan B task 9)
    sources_ = "\n".join(_source(c) for c in code)
    assert "px.prepared_table(VIDEO, inst.books)" in sources_ and "px.ask_table(VIDEO)" in sources_
    for text in ("PRES-SUBS", "Presentation substance", "Amount asked (USD)", "Equity offered (%)"):
        assert text in html, text
    assert "How an investor left an exchange" not in printed + html            # the closing-state table is gone
    compared = int(export.csv("self_consistency.csv").n_videos.iloc[0])
    assert f"n videos with two passes compared = {compared} of {n}" in printed
    shown = _video_ids_shown(code, run_level=("px.ranking_orders(", "px.comparison_tables("))
    assert video in shown and len(shown) >= 10 and shown <= _selected(EXPLORER)
    words = " ".join(" ".join(_source(c).split()) for c in nb["cells"] if c["cell_type"] == "markdown")
    for phrase in ("never from today's codebooks", "does **not** mean the codes are correct",
                   "this notebook does not run them", "### C3b. The prepared presentation's content", "### C3c. The ask",
                   "### C10. What happens after each answer", "no measure of the analysis reads the exchange grouping"):
        assert phrase in words, phrase


# ---- they run on a synthetic run, and write nothing unless asked ----------------------------------

def test_analysis_notebook_executes_on_a_synthetic_run_and_writes_nothing(run, tmp_path):
    nb = configure(_read(ANALYSIS), RUN_NAME=str(run), PENALTY=10, EXPORT=False)
    before = _snapshot(run)
    code, printed, html = execute(nb, tmp_path)
    assert _snapshot(run) == before                                             # reading only
    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy.ipynb"]
    assert "n pitches = 24 (" in printed and "[pitch-held-out estimate]" in printed
    assert "24 videos: 23 model-coded only, 1 human-reviewed." in printed and pr.EXPORT_NOTE in printed
    assert "History cut for the pitch v000: the first decision of any investor (" in printed
    assert "**Verdict: " in printed and "HOW TO READ THESE NUMBERS" in printed and "main verdict '" in printed
    assert "predictor rows (model × outcome × predictor" in printed and "h9_exchange_sensitivity.csv" in printed
    assert "Model C3, trained without v000, gives it a " in printed
    for number in pf.FIGURES:
        name = f"Supplementary Figure {number}" if number in pf.SUPPLEMENTARY else f"Figure {number}"
        assert f"\n{name}. {pf.FIGURES[number][1]}." in "\n" + printed, number
    assert "EXPORT and SAVE_FIGURES are False: nothing was written, and " in printed and "No figure file to list." in printed
    assert _images(code) >= len(pf.FIGURES) - 1


def test_analysis_notebook_export_writes_only_the_analysis_folder(run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run, root)
    nb = configure(_read(ANALYSIS), RUN_NAME=str(root), VIDEO_IDS=[f"v{i:03d}" for i in range(6)], ANALYSIS_ID="nb1",
                   PENALTY=10, RANDOM_SEED=5, RUN_H9=False, WITH_SHARK_LEVEL=False, EXPORT=True)
    before = _snapshot(root)
    _, printed, html = execute(nb, tmp_path)
    out = root / "analysis" / "nb1"
    assert f"files to {out.resolve()}" in printed
    assert {k: v for k, v in _snapshot(root).items() if not k.startswith("analysis/nb1/")} == before
    names = {p.name for p in out.iterdir()}
    assert {"manifest.json", "report.md", "metrics.csv", "verdicts.csv", "h9_summary.csv", "figures"} <= names
    manifest = _read(out / "manifest.json")
    assert manifest["video_ids"] == [f"v{i:03d}" for i in range(6)] and manifest["n_boot"] == 19 and manifest["seed"] == 5
    assert manifest["h9"] == "skipped" and manifest["shark_level"] is None and manifest["n_perm"] == 9
    figures = sorted(p.name for p in (out / "figures").iterdir())
    assert figures == sorted(manifest["figures"]["files"])
    assert f"Saved {len(figures) // 3} figures (PNG at 300 dpi, vector PDF and a caption file each) to " in printed


def test_analysis_notebook_saves_only_the_figures_when_asked(run, tmp_path):
    root = tmp_path / "run"
    shutil.copytree(run, root)
    nb = configure(_read(ANALYSIS), RUN_NAME=str(root), PENALTY=10, PERMUTATIONS=0, RUN_H9=False, ANALYSIS_ID="figs",
                   SAVE_FIGURES=True, FIGURE_WIDTH="single", EXPORT=False)
    before = _snapshot(root)
    _, printed, html = execute(nb, tmp_path)
    folder = root / "analysis" / "figs" / "figures"
    assert {k: v for k, v in _snapshot(root).items() if not k.startswith("analysis/figs/figures/")} == before
    assert sorted(p.name for p in (root / "analysis" / "figs").iterdir()) == ["figures"]
    written = sorted(p.name for p in folder.iterdir())
    assert pr.EXPORT_NOTE in printed and "figure_1_caption.md" in written
    assert f"Saved {len(written) // 3} figures (PNG at 300 dpi, vector PDF and a caption file each) to " in printed
    assert (folder / "figure_1_caption.md").read_text().strip() in printed
    from PIL import Image
    with Image.open(folder / "figure_1_model_ladder.png") as image:
        assert image.size[0] == 1050                                               # 3.5 inches at 300 dpi


def test_analysis_notebook_explains_a_single_video_instead_of_failing(run, tmp_path):
    nb = configure(_read(ANALYSIS), RUN_NAME=str(run), VIDEO_IDS=["v001"], RUN_H9=False, H9_EXCHANGE_OUTCOMES=False,
                   WITH_SHARK_LEVEL=False)
    _, printed, _ = execute(nb, tmp_path)
    assert "n pitches = 1 (" in printed and "fewer than two pitches with a known outcome" in printed
    assert "H9 was not run in this execution (RUN_H9 = False)." in printed
    for number in (1, 2, 3, 6):
        assert f"Figure {number}. {pf.FIGURES[number][1]}. Not drawn: " in printed, number


def test_explorer_notebook_executes_and_writes_nothing(run, tmp_path):
    nb = configure(_read(EXPLORER), RUN_NAME=str(run), VIDEO_IDS=["v000", "v002", "v005"], VIDEO_ID="v002",
                   PASS="reviewed", SHOW_FULL_PROMPT=True)
    before = _snapshot(run)
    code, printed, html = execute(nb, tmp_path)
    assert "3 videos; instrument release v3-dev9; 1 human-reviewed" in printed
    assert "The analysis reads: the primary pass of 2, the reviewed pass of 1" in printed
    assert "n videos with two passes compared = 3 of 3" in printed and "v001" not in html
    assert "Asker's next move" in html and "Next turn summary" in html
    assert "Amount asked (USD)" in html and "Presentation substance" in html
    assert _snapshot(run) == before and sorted(p.name for p in tmp_path.iterdir()) == ["copy.ipynb"]
    assert "Selected video: v002" in printed and "Showing the reviewed pass of v002." in printed
    assert "LIVE release" in printed and "This video kept no prompt file." in printed
    assert pr.EXPORT_NOTE in printed
    assert 2 <= _images(code) <= 1 + len(px.VALUE_GROUPS)


def test_explorer_notebook_copes_with_one_video_one_pass_and_exports_on_request(tmp_path):
    root = write_run(tmp_path, 1, seed=3, name="single")
    (root / "v000" / "results" / "replicate.json").unlink()
    (root / "v000" / "agreement.json").unlink()
    write_ranking(root, ["v000"], __import__("numpy").random.default_rng(0))
    (root / "selection_one.json").write_text(json.dumps({"videos": [{"video_id": "v000"}]}))
    nb = configure(_read(EXPLORER), RUN_NAME=str(root), VIDEO_IDS="selection_one.json", VIDEO_ID=None, PASS="replicate",
                   EXPORT=True)
    before = _snapshot(root)
    _, printed, _ = execute(nb, tmp_path)
    assert "v000 has no replicate pass (it has: primary); showing the primary pass instead." in printed
    assert "No video in this run has a second pass to compare with." in printed
    out = root / "analysis" / "explorer"
    assert f"CSV files to {out}" in printed
    assert {"run_overview.csv", "turns.csv", "decisions.csv", "answers_and_next_moves.csv"} <= {p.name for p in out.iterdir()}
    assert {k: v for k, v in _snapshot(root).items() if not k.startswith("analysis/explorer/")} == before


@needs_run
def test_explorer_notebook_shows_how_often_each_value_of_each_code_is_used():
    nb = _read(EXPLORER)
    cells = [c for c in _code(nb) if "value_frequency_figure" in _source(c) or "value_usage_summary" in _source(c)]
    assert len(cells) == 2
    table, charts = cells
    html = "".join("".join(o.get("data", {}).get("text/html", "")) for o in table["outputs"])
    assert "Values never used" in html and "Dominant (80% or more)" in html and "TURN-FUNC" in html
    assert len([o for o in charts["outputs"] if "image/png" in o.get("data", {})]) == len(px.VALUE_GROUPS)
