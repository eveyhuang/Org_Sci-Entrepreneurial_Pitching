"""scripts/experiments/reaction_trajectory_test.py: the experiment, its verdict rule and its sensitivities."""
from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.panel_analysis_fixtures import write_run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "reaction_trajectory_test.py"
spec = importlib.util.spec_from_file_location("reaction_trajectory_test", SCRIPT)
rtt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtt)

FILES = ("manifest.json", "reaction_events.csv", "reaction_descriptives.json", "features_o2.csv", "features_o1.csv",
         "metrics_o2.csv", "deltas_o2.csv", "metrics_o1.csv", "deltas_o1.csv", "clogit_o2.csv", "sensitivity.csv",
         "figure_within_pitch_ladder.png", "figure_within_pitch_ladder.pdf", "report.md")


# ---- the decision rule, on hand-made deltas ----------------------------------------------------------

def deltas(rows):
    return pd.DataFrame([{"comparison": c, "metric": m, "delta": d, "ci_low": lo, "ci_high": hi} for c, m, d, lo, hi in rows])


GOOD, SMALL, WIDE = (0.06, 0.01, 0.11), (0.02, 0.005, 0.04), (0.06, -0.01, 0.12)


def o2(primary, asker=SMALL, both=SMALL):
    return deltas([("R3 minus B4", "within_auc", *primary), ("R1 minus B3", "within_auc", *asker),
                   ("R2 minus B4a", "within_auc", *both)])


def o1(pair):
    return deltas([("R-C4 minus C4", "auc", *pair)])


@pytest.mark.parametrize("primary, asker, both, any_offer, expected", [
    (GOOD, GOOD, SMALL, SMALL, "justified"),
    (GOOD, SMALL, GOOD, GOOD, "justified"),
    (GOOD, SMALL, SMALL, SMALL, "mixed"),          # the primary alone, without asker or panel support
    (SMALL, GOOD, GOOD, SMALL, "not justified"),
    (WIDE, GOOD, GOOD, WIDE, "not justified"),     # an interval that includes 0 never holds
    (SMALL, SMALL, SMALL, GOOD, "mixed"),          # only any offer (O1) gains
])
def test_the_verdict_follows_section_zero(primary, asker, both, any_offer, expected):
    assert rtt.verdict(o2(primary, asker, both), o1(any_offer)) == expected


def test_a_missing_comparison_never_holds():
    assert rtt.verdict(o2((math.nan, math.nan, math.nan), GOOD, GOOD), deltas([])) == "not justified"


def test_a_verdict_that_flips_under_a_sensitivity_is_fragile():
    assert rtt.final_verdict("justified", "justified", "justified") == "justified"
    assert rtt.final_verdict("justified", "not justified", "justified") == "justified (fragile: possibly a decision tell)"
    assert rtt.final_verdict("justified", "justified", "mixed") == "justified (fragile: possibly shark disposition)"


# ---- the investor prior --------------------------------------------------------------------------------

def test_the_investor_prior_reads_only_other_pitches():
    rows = pd.DataFrame([
        ("v1", "S1", "Mark", 1.0), ("v1", "S2", "Lori", 0.0),
        ("v2", "S1", "Mark", 0.0), ("v2", "S2", "Lori", 1.0),
        ("v3", "S1", "Mark", 1.0), ("v3", "S2", "Kevin", 0.0),
        ("v4", "S1", "Mark", 1.0), ("v4", "S2", "Lori", np.nan)], columns=["video_id", "investor", "investor_label", "ever_offer"])
    prior = rtt.investor_prior(rows)
    assert prior.iloc[0] == pytest.approx(2 / 3)       # Mark on v2, v3, v4
    assert prior.iloc[2] == pytest.approx(1.0)         # Mark on v1, v3, v4
    assert math.isnan(prior.iloc[1])                   # Lori: only v2 has a known outcome besides v1
    assert math.isnan(prior.iloc[5])                   # Kevin is on no other pitch


def test_the_prior_is_declared_only_while_it_is_used():
    from analysis.archive.shark_level_features import ARCHIVE_FEATURES as FEATURES     # an investor-row column
    with rtt.prior_declared():
        assert FEATURES[rtt.PRIOR]["family"] == rtt.PRIOR_FAMILY
    assert rtt.PRIOR not in FEATURES and f"{rtt.PRIOR}_missing" not in FEATURES


# ---- the whole experiment on synthetic runs ---------------------------------------------------------------

@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    root = write_run(tmp_path_factory.mktemp("rx"), 24, planted="reaction_events", strength=4.0)   # 12 give ~9 pairs
    assert rtt.main(["--run", str(root), "--analysis-id", "t", "--n-boot", "49"]) == 0
    return root


def test_the_experiment_writes_every_file_and_the_planted_verdict(planted):
    out = planted / "analysis" / "t"
    assert sorted(p.name for p in out.iterdir()) == sorted(FILES)
    text = (out / "report.md").read_text()
    assert text.startswith("## 0. The pre-registered test and its decision rule")
    assert text.rstrip().splitlines()[-1] == "Verdict: justified"
    manifest = json.loads((out / "manifest.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
    assert manifest["decision_rule"]["min_gain"] == 0.03


def test_the_tables_hold_both_ladders_both_variants_and_the_sensitivities(planted):
    out = planted / "analysis" / "t"
    metrics = pd.read_csv(out / "metrics_o2.csv")
    assert list(metrics.model) == ["B1", "B3", "R1", "B4a", "R2", "B4", "R3", "B5", "R4", "R1c", "R2c", "R3c", "R4c"]
    assert not metrics.constant_within.any()
    d2 = pd.read_csv(out / "deltas_o2.csv")
    assert {"R3 minus B4", "R3c minus B4"} <= set(d2.comparison)
    assert {"within_auc", "auc", "log_loss"} <= set(d2.metric)
    assert list(pd.read_csv(out / "metrics_o1.csv").model) == ["C1", "C3", "R-C3", "C4", "R-C4", "R-C3c", "R-C4c"]
    sensitivity = pd.read_csv(out / "sensitivity.csv")
    assert {"main", "guard_turns=1", "investor_prior", "counts_only"} <= set(sensitivity.setting)
    assert np.isfinite(sensitivity.set_index("setting").loc["guard_turns=1", "delta"])
    clogit = pd.read_csv(out / "clogit_o2.csv")
    assert {"a_existing", "b_reactions", "c_both", "c_vs_a"} == set(clogit.set)
    descr = json.loads((out / "reaction_descriptives.json").read_text())
    assert descr["answers"] > 0 and set(descr["by_asker_offer"]) == {"asker_offered", "asker_did_not_offer"}


def test_the_experiment_refuses_to_overwrite(planted, capsys):
    assert rtt.main(["--run", str(planted), "--analysis-id", "t", "--n-boot", "49"]) == 2
    assert "already exists" in capsys.readouterr().err


def test_without_a_planted_signal_the_verdict_is_never_justified(tmp_path):
    root = write_run(tmp_path, 24, planted=None)
    assert rtt.main(["--run", str(root), "--analysis-id", "null", "--n-boot", "49"]) == 0
    last = (root / "analysis" / "null" / "report.md").read_text().rstrip().splitlines()[-1]
    assert last.split(" (")[0] in ("Verdict: not justified", "Verdict: mixed")
