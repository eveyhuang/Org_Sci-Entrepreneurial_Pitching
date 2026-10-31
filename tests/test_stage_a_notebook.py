"""The Stage A Hu & Ma notebook: what is committed, and that it runs on synthetic runs without writing."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_hu_ma_funding import _annotations, _hu_ma_csv, _measures, _outcomes_csv, _scores_csv

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "stage_A" / "stage_a_hu_ma_analysis.ipynb"
SETTINGS = ("STAGE_A_RUN_NAME", "RANK_RUN_NAME", "OUTCOMES_CSV", "HU_MA_PITCH_FACTOR_CSV", "ANALYSIS_ID", "EXPORT")
PROBE = ("import sys\n"
         "print('PROBE sdk loaded:', any(m == 'google' or m.startswith(('google.genai', 'google.generativeai')) "
         "for m in sys.modules))")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _source(cell) -> str:
    return "".join(cell["source"])


def _code(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def configure(nb: dict, **settings) -> dict:
    cell = _code(nb)[0]
    source = _source(cell)
    for name, value in settings.items():
        source, n = re.subn(rf"^{name} = [^#\n]*", lambda _: f"{name} = {value!r}  ", source, flags=re.M)
        assert n == 1, f"{name} is assigned {n} times in the configuration cell"
    cell["source"] = source
    return nb


def execute(nb: dict, tmp_path: Path, name: str = "copy.ipynb") -> tuple[list[dict], str, str]:
    pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb["cells"].append({"cell_type": "code", "metadata": {}, "source": PROBE, "outputs": [], "execution_count": None})
    copy = tmp_path / name
    copy.write_text(json.dumps(nb))
    done = subprocess.run([sys.executable, str(REPO / "scripts/analysis/execute_notebook.py"), str(copy),
                           "--timeout", "300"], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-4000:]
    code = _code(_read(copy))
    assert all(c["execution_count"] for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
    printed = "\n".join("".join(o.get("text", "")) for c in code for o in c["outputs"] if o.get("output_type") == "stream")
    html = "\n".join("".join(o["data"].get("text/html", "")) for c in code for o in c["outputs"] if "data" in o)
    assert "PROBE sdk loaded: False" in printed
    return code, printed, html


def _images(code: list[dict]) -> int:
    return sum(1 for c in code for o in c["outputs"] if "image/png" in o.get("data", {}))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A synthetic rating run (with manifest and tables), a ranking run and an outcomes file, 40 videos."""
    base = tmp_path_factory.mktemp("stage_a_nb")
    frame, factor, accepted = _measures(40, seed=21)
    stage_a = base / "runs" / "stage_a_synthetic"
    (stage_a / "tables").mkdir(parents=True)
    _annotations(frame, k=3).to_parquet(stage_a / "tables" / "annotations.parquet", index=False)
    (stage_a / "manifest.json").write_text(json.dumps({"model_requested": "fake-model", "git_sha": "abc",
                                                       "passes": {"holistic": {"done": 120, "failed": 0, "prompt_hash": "h1"}}}))
    rank = base / "runs" / "rank_synthetic"
    (rank / "ranking").mkdir(parents=True)
    _scores_csv(frame, rank / "ranking" / "scores.csv")
    (rank / "ranking" / "design.json").write_text(json.dumps({"settings": {"batch_size": 5, "appearances": 2, "seed": 0},
                                                              "windows_from": "whole_videos", "release_id": "test",
                                                              "summary": {"n_videos": 40, "repeated_pairs": 0}}))
    (rank / "ranking" / "completion.json").write_text(json.dumps({"status": "complete", "batches_ok": 16}))
    outcomes = _outcomes_csv(frame, accepted, base / "outcomes.csv")
    hu_ma = _hu_ma_csv(frame, base / "pitch_factor.csv")
    return {"stage_a": stage_a, "rank": rank, "outcomes": outcomes, "hu_ma": hu_ma,
            "accepted": int(accepted[np.arange(40) % 5 != 3].sum())}


# ---- what is committed ----------------------------------------------------------------------------

def test_notebook_is_committed_executed_with_the_configuration_cell_first():
    nb = _read(NOTEBOOK)
    cells, code = nb["cells"], _code(nb)
    assert all(c["execution_count"] for c in code)
    assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"])
    assert sum(bool(c["outputs"]) for c in code) >= len(code) - 1
    config = code[0]
    assert "parameters" in config["metadata"].get("tags", [])
    assert sum("parameters" in c["metadata"].get("tags", []) for c in code) == 1
    text = _source(config)
    assert text.startswith("# ===== CONFIGURATION") and "the only cell you need to edit" in text
    for name in SETTINGS:
        assert re.search(rf"^{name} = .*#", text, flags=re.M), f"{name} missing or unexplained"
    assert re.search(r"^EXPORT = False\b", text, flags=re.M)
    assert re.search(r'^STAGE_A_RUN_NAME = "stage_a_pitch_factor_20261008_004807"', text, flags=re.M)
    assert re.search(r'^HU_MA_PITCH_FACTOR_CSV = "data/hu-ma/pitch_factor.csv"', text, flags=re.M)
    first = " ".join(_source(cells[0]).split())
    assert cells[0]["cell_type"] == "markdown" and "How to use this notebook" in first
    for phrase in (".venv", "configuration cell", "Restart & Run All", "no model calls", "never changes annotation files",
                   "not human-validated"):
        assert phrase in first, phrase


def test_notebook_code_is_thin_and_every_section_is_introduced_in_plain_language():
    nb = _read(NOTEBOOK)
    cells, code = nb["cells"], _code(nb)
    everything = "\n".join(_source(c) for c in code)
    for forbidden in ("def ", "lambda", "class ", "providers", "--execute", ".to_csv", "write_text", "open(", "savefig"):
        assert forbidden not in everything, forbidden
    headings = [_source(c) for c in cells if c["cell_type"] == "markdown" and _source(c).startswith("## ")]
    assert [h.split("\n")[0] for h in headings] == [
        "## 1. Provenance", "## 2. Ratings: how far the replicates agree (ICC)",
        "## 3. The Pitch Factor from the ratings and from the rank scores", "## 4. Do the two methods agree? (Spearman)",
        "## 5. Against Hu & Ma's own Pitch Factor (convergent validity)",
        "## 6. The funding decision: Hu & Ma Table III and Table IV analogues", "## 7. Export"]
    for h in headings[:-1]:
        assert "**What this shows.**" in h and "**What it does not show.**" in h, h.split("\n")[0]
    # the committed outputs carry the real headline numbers
    printed = "\n".join("".join(o.get("text", "")) for c in code for o in c["outputs"] if o.get("output_type") == "stream")
    assert "accepted: 3; rated: 50; ranked: 50" in printed
    assert "Spearman between the two Pitch Factors" in printed and "Pitch Factor (ratings): a one-SD increase" in printed
    assert "Hu & Ma's file: 50 videos, 50 matched" in printed and "Pitch Factor (ratings) against Hu & Ma's: Pearson" in printed
    assert "Pitch Factor (Hu & Ma): a one-SD increase" in printed
    assert "EXPORT is False: nothing was written." in printed
    assert _images(code) == 3


# ---- execution on synthetic runs ------------------------------------------------------------------

def test_notebook_executes_on_synthetic_runs_and_writes_nothing(runs, tmp_path):
    nb = configure(_read(NOTEBOOK), STAGE_A_RUN_NAME=str(runs["stage_a"]), RANK_RUN_NAME=str(runs["rank"]),
                   OUTCOMES_CSV=str(runs["outcomes"]), HU_MA_PITCH_FACTOR_CSV=str(runs["hu_ma"]), EXPORT=False)
    before = {k: _snapshot(runs[k]) for k in ("stage_a", "rank")}
    code, printed, html = execute(nb, tmp_path)
    assert {k: _snapshot(runs[k]) for k in ("stage_a", "rank")} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["copy.ipynb"]
    assert f"accepted: {runs['accepted']}; rated: 40; ranked: 40" in printed
    assert "Spearman between the two Pitch Factors" in printed
    assert "Hu & Ma's file: 40 videos, 40 matched" in printed
    assert "Pitch Factor (ratings) against Hu & Ma's: Pearson +0." in printed and "the interval excludes zero" in printed
    assert "Dimensions where the mean rating and Hu & Ma's feature agree" in printed
    assert "Correlation between loading vectors over the ten codes: ratings vs ranks" in printed
    assert "Pitch Factor (ratings): a one-SD increase" in printed and "Pitch Factor (ranks): a one-SD increase" in printed
    assert "Pitch Factor (Hu & Ma): a one-SD increase" in printed
    assert "Feature-level rows with exact p below 0.05:" in printed
    assert "stage_a_passes.holistic.prompt_hash" in html and "h1" in html
    assert "EXPORT is False: nothing was written." in printed
    assert _images(code) == 3


def test_notebook_ratings_only_and_export_writes_only_the_analysis_folder(runs, tmp_path):
    stage_a = tmp_path / "stage_a_copy"
    shutil.copytree(runs["stage_a"], stage_a)
    nb = configure(_read(NOTEBOOK), STAGE_A_RUN_NAME=str(stage_a), RANK_RUN_NAME=None,
                   OUTCOMES_CSV=str(runs["outcomes"]), HU_MA_PITCH_FACTOR_CSV=None, ANALYSIS_ID="nb_test", EXPORT=True)
    before = _snapshot(stage_a)
    code, printed, html = execute(nb, tmp_path)
    assert "ranking run: none" in printed and "No ranking run was given" in printed
    assert "Hu & Ma Pitch Factor file: none" in printed and "No Hu & Ma Pitch Factor file was given" in printed
    assert "Pitch Factor (ranks)" not in printed and "Pitch Factor (Hu & Ma)" not in printed and _images(code) == 0
    after = _snapshot(stage_a)
    assert {k for k in after if k not in before} and all(k.startswith("analysis/nb_test/") for k in after if k not in before)
    assert all(after[k] == before[k] for k in before)
    assert (stage_a / "analysis" / "nb_test" / "summary.md").exists()
    assert not (stage_a / "analysis" / "nb_test" / "rank_scores_wide.csv").exists()
    assert not (stage_a / "analysis" / "nb_test" / "hu_ma_agreement.csv").exists()
    assert "Written:" in printed and "summary.md" in printed
