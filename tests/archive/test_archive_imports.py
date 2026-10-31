"""The archive stays usable: every module under analysis/archive/ imports, the two experiment scripts start, and the
archived shark-level run still produces the B ladder."""
from __future__ import annotations

import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import analysis.archive as archive
from tests.panel_analysis_fixtures import write_run

REPO = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "scripts" / "experiments"
MODULES = sorted(m.name for m in pkgutil.iter_modules(archive.__path__))


def test_the_archive_holds_the_shark_level_modules():
    assert {"shark_level_dataset", "shark_level_features", "shark_level_models", "within_pitch_eval",
            "shark_level"} <= set(MODULES)
    assert "legacy_panel_analysis" not in MODULES                     # deleted in plan B task 9


@pytest.mark.parametrize("name", MODULES)
def test_every_archive_module_imports(name):
    importlib.import_module(f"analysis.archive.{name}")


def test_the_within_pitch_metric_lives_in_the_archive_only():
    importlib.import_module("analysis.archive.within_pitch_eval")
    assert not (REPO / "analysis" / "within_pitch_eval.py").exists()


@pytest.mark.parametrize("script", ["reaction_trajectory_test.py", "identity_baseline_test.py"])
def test_the_experiment_scripts_start(script):
    done = subprocess.run([sys.executable, str(EXPERIMENTS / script), "--help"], capture_output=True, text=True,
                          cwd=REPO, timeout=300)
    assert done.returncode == 0, done.stderr
    assert "--run" in done.stdout


def test_the_archived_shark_level_run_returns_the_b_ladder(tmp_path):
    from analysis.archive import shark_level
    from analysis.panel_dataset import load_run
    root = write_run(tmp_path, 24, seed=3, planted="own", strength=4.0)
    data = load_run(root)
    out = shark_level.run(data, root, n_boot=5, seed=1, n_perm=0, penalty=10.0)
    assert set(out) == set(shark_level.TABLES)
    models = list(out["metrics"].model)
    assert {"B0", "B1", "B2", "B3", "B4", "B5"} <= set(models)
    assert models == ["B0", "B1", "B2", "B3", "B4a", "B4", "B5", "B5d"]
    assert {"auc", "within_auc"} <= set(out["metrics"].columns)
