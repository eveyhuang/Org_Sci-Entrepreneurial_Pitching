import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import render_panel_record as rpr  # noqa: E402
from tests.panel_fixtures import synthetic_record, turn  # noqa: E402


def build(tmp_path, record=None, experiment=True, validation=None, name="primary"):
    vdir = tmp_path / "vid1"
    (vdir / "results").mkdir(parents=True, exist_ok=True)
    if experiment:
        (vdir / "experiment.json").write_text(json.dumps({"video_id": "vid1", "release_id": "v3-dev9"}))
    path = vdir / "results" / f"{name}.json"
    path.write_text(json.dumps({"pass": name, "unit_key": f"panel__vid1__{name}", "prepared_window": ["0:05-1:00"],
                                "record": record or synthetic_record()}))
    if validation is not None:
        (vdir / "results" / f"{name}_validation.json").write_text(json.dumps(validation))
    return path


def rendered(tmp_path, **kw):
    path = build(tmp_path, **kw)
    out = tmp_path / "view.md"
    assert rpr.main(["--record", str(path), "--out", str(out)]) == 0
    return out.read_text()


def test_header_names_video_release_pass_and_prepared_window(tmp_path):
    md = rendered(tmp_path)
    head = md[:md.index("## Actors")]
    assert head.startswith("# Panel record: vid1 (primary)\n")
    for line in ("- Video: vid1", "- Release: v3-dev9", "- Pass: primary", "- Prepared window: 0:05-1:00"):
        assert line in head, line
    assert "not human-validated" in head


def test_header_without_an_experiment_file_falls_back_to_the_unit_key(tmp_path):
    md = rendered(tmp_path, experiment=False, name="replicate")
    assert "- Video: vid1" in md and "- Release: unknown" in md and "- Pass: replicate" in md


def test_actor_stage_a_and_venture_tables(tmp_path):
    md = rendered(tmp_path)
    assert "| S2 | investor | Investor B |" in md
    assert "| pitch_positivity | 3 | 0:20 | Steady delivery. |" in md
    assert "| expressed_enthusiasm | 3 | — | Steady delivery. |" in md
    assert "Reusable food containers sold online to households." in md and "home_household" in md
    record = synthetic_record()
    record["stage_a"][0].update(value=None, uncodable=True, moment_s=None)
    assert "| pitch_positivity | uncodable | — | Steady delivery. |" in rendered(tmp_path, record=record)


def test_exchange_headings_and_investor_lines(tmp_path):
    md = rendered(tmp_path)
    headings = [ln for ln in md.splitlines() if ln.startswith("### ")]
    assert headings == [
        "### (no exchange)",
        '### Exchange 1 · substantive · operating_efficiency · "customer acquisition cost"',
        "### Exchange 2 · social",
        "### (no exchange)",
        '### Exchange 3 · substantive · customer_retention · "repeat purchase rate"',
        '### Exchange 4 · negotiation · "joint offer"',
    ]
    assert "Investors:" not in md and "satisfied" not in md


def test_turn_rows_resolve_names_and_show_codes(tmp_path):
    md = rendered(tmp_path)
    assert "| turn | time | who | function | codes | summary |" in md
    assert ("| 2 | 1:02 | S1 (Investor A) → F1 | question | PR-CONCEPT=operating_efficiency (prevention) PR-SEV=1 | "
            "Asks what it costs to acquire one customer. |") in md
    assert ("| 3 | 1:06 | F1 (Founder) → S1 | answer ↩ 2 | RS-RESP=addresses RS-CONCEPT=operating_efficiency (prevention) RS-EVID=true RS-UNCERT=false "
            "PREP-SUBS=3 PREP-COH=4 PREP-CTX=∅(not_applicable) | "
            "States a blended acquisition figure and names the channels. |") in md
    assert "| 1 | 0:05 | F1 (Founder) | presents |  | Prepared presentation of the reusable containers and the ask. |" in md
    assert "| 19 | 2:49 | F1 (Founder) → PANEL | decides |  | Accepts the joint offer. |" in md
    assert "| 7 | 1:40 | S1 (Investor A) → F1 | accepts ↩ 6 |" in md
    record = synthetic_record()
    record["turns"][6]["responds_to"] = [6, 2]
    assert "| accepts ↩ 6, 2 |" in rendered(tmp_path, record=record)


def test_reopened_exchange_is_printed_once_with_a_pointer_where_it_resumes(tmp_path):
    md = rendered(tmp_path)
    assert md.count("### Exchange 1 ·") == 1
    first, second, third = (md.index(f"### Exchange {n} ·") for n in (1, 2, 3))
    block = md[first:second]
    for clock in ("1:02", "1:06", "1:20", "1:26", "1:34", "1:40", "2:14", "2:18", "2:30"):
        assert f"| {clock} |" in block, clock
    pointers = [ln for ln in md.splitlines() if ln.startswith("↪")]
    assert pointers == ['↪ 2:14: Exchange 1 ("customer acquisition cost") resumes here for 3 turns, shown above.']
    assert third < md.index(pointers[0]) < md.index("### Exchange 4 ·")
    assert "| 2:14 |" not in md[third:]


def test_decisions_are_shown_inline_at_their_turn(tmp_path):
    md = rendered(tmp_path)
    lines = md.splitlines()
    exit_turn = lines.index("| 10 | 1:51 | S3 (Investor C) → F1 | decides |  | Declares herself out. |")
    assert lines[exit_turn + 1] == "| 10 | 1:51 | S3 (Investor C) | **DECISION** exit |  | Out. |"
    offer = lines.index("| 17 | 2:38 | S1 (Investor A) → F1 | decides |  | Offers the amount asked for a larger stake. |")
    assert lines[offer + 1] == "| 17 | 2:39 | S1 (Investor A) | **DECISION** offer O1 |  | 200k for 20%. States terms. |"
    assert md.count("**DECISION**") == 4
    assert "Decisions without a turn" not in md


def test_decision_without_a_known_turn_is_listed_after_the_timeline(tmp_path):
    record = synthetic_record()
    record["decisions"][0]["turn_id"] = None
    record["decisions"][1]["turn_id"] = 99
    md = rendered(tmp_path, record=record)
    tail = md[md.index("### Decisions without a turn"):md.index("## Claim conflicts")]
    assert "|  | 1:51 | S3 (Investor C) | **DECISION** exit |  | Out. |" in tail
    assert "| 99 | 2:39 | S1 (Investor A) | **DECISION** offer O1 |" in tail
    assert "**DECISION** offer O1" in tail and md.count("**DECISION**") == 4


def test_claim_conflicts_section(tmp_path):
    assert "## Claim conflicts\n\nNone recorded." in rendered(tmp_path)
    record = synthetic_record()
    record["claim_conflicts"] = [
        {"turn_a": 1, "turn_b": 3, "metric": "acquisition cost", "note": "Two figures.", "reconciled": True,
         "reconciled_turn": 6},
        {"turn_a": 3, "turn_b": 15, "metric": "paid cost", "note": "", "reconciled": False, "reconciled_turn": None}]
    md = rendered(tmp_path, record=record)
    assert "- acquisition cost: turn 1 (0:05) against turn 3 (1:06); reconciled in turn 6 (1:34). Two figures." in md
    assert "- paid cost: turn 3 (1:06) against turn 15 (2:18); not reconciled." in md


def test_validation_is_shown_only_when_the_sibling_file_exists(tmp_path):
    assert "## Validation" not in rendered(tmp_path)
    md = rendered(tmp_path, validation={"errors": ["[turn.role] turn 7: wrong"], "flags": ["[flag.gap] gap at 90.0s"]})
    tail = md[md.index("## Validation"):]
    assert "- [turn.role] turn 7: wrong" in tail and "- [flag.gap] gap at 90.0s" in tail
    assert "Errors (1)" in tail and "Flags (1)" in tail
    clean = rendered(tmp_path, validation={"errors": [], "flags": []})
    assert "Errors (0)" in clean and "Flags (0)" in clean


def test_turn_pointing_at_an_unrecorded_exchange_and_cells_with_pipes(tmp_path):
    record = synthetic_record()
    record["turns"].insert(10, turn(20, "S2", 114, 114.5, "NONE", "Says a|b under her breath.", exchange_id=9))
    md = rendered(tmp_path, record=record)
    assert "### Exchange 9 · (not recorded)" in md
    assert "| 20 | 1:54 | S2 (Investor B) | NONE |  | Says a\\|b under her breath. |" in md


def test_unknown_speaker_is_shown_by_id(tmp_path):
    record = synthetic_record()
    record["turns"][7]["speaker"] = "X9"
    assert "| 1:45 | X9 → F1 | social |" in rendered(tmp_path, record=record)


def test_record_file_is_not_modified_and_missing_record_is_an_error(tmp_path):
    path = build(tmp_path)
    before = path.read_bytes()
    rpr.main(["--record", str(path), "--out", str(tmp_path / "o.md")])
    assert path.read_bytes() == before
    with pytest.raises(SystemExit):
        rpr.main(["--record", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o2.md")])
    assert not (tmp_path / "o2.md").exists()


def test_labels_are_shown_in_their_canonical_form(tmp_path):
    record = synthetic_record()
    record["turns"][1]["function"] = "Question"
    record["exchanges"][0].update(kind="Substantive", topic="Operating_Efficiency")
    record["exchanges"][1]["topic"] = "none"
    record["decisions"][0]["kind"] = "EXIT"
    assert rendered(tmp_path, record=record) == rendered(tmp_path)


def test_concepts_are_shown_with_their_derived_frame(tmp_path):
    record = synthetic_record()
    record["turns"][1]["codes"][0]["value"] = "NONE"
    record["turns"][3]["codes"][0].update(value=None, missing="ambiguous")
    record["turns"][10]["codes"][0]["value"] = "pricing"                      # not a concept: shown without a frame
    md = rendered(tmp_path, record=record)
    assert "| question | PR-CONCEPT=NONE (NEUTRAL) PR-SEV=1 |" in md
    assert "| question | PR-CONCEPT=∅(ambiguous) PR-SEV=3 |" in md
    assert "| question | PR-CONCEPT=pricing PR-SEV=2 |" in md
    assert "RS-CONCEPT=forecast (promotion) RS-EVID=false" in md
    assert "PR-SEV=1 (" not in md and "RS-RESP=addresses (" not in md


def test_other_concepts_are_shown_with_their_name_and_listed(tmp_path):
    from tests.panel_fixtures import set_other
    assert "## Other concepts named\n\nNone.\n" in rendered(tmp_path)
    record = synthetic_record()
    set_other(record["turns"][10], "founder's own investment")
    set_other(record["turns"][11], None)
    record["exchanges"][2].update(topic="other", label="money the founder put in")
    md = rendered(tmp_path, record=record)
    assert "| question | PR-CONCEPT=other (\"founder's own investment\") (unclassified) PR-SEV=2 |" in md
    assert "RS-RESP=addresses RS-CONCEPT=other (unnamed) (unclassified) RS-EVID=true" in md
    assert '### Exchange 3 · substantive · other · "money the founder put in"' in md
    section = md[md.index("## Other concepts named"):md.index("## Claim conflicts")]
    assert section.splitlines()[2:5] == [
        "- 1:55 turn 11 PR-CONCEPT: founder's own investment",
        "- 1:59 turn 12 RS-CONCEPT: (unnamed)",
        "- 1:55 exchange 3 EX-TOPIC: money the founder put in"]
    assert md.index("## Timeline") < md.index("## Other concepts named") < md.index("## Claim conflicts")


def test_venture_shows_the_presenting_team_gender_composition(tmp_path):
    assert "- Presenting team: all_men" in rendered(tmp_path)


def test_prepared_presentation_codes_are_one_line_after_stage_a(tmp_path):
    md = rendered(tmp_path)
    line = "prepared: SUBS 3 · COH 4 · CTX 2 · EVID 5"
    assert line in md and md.index("| interpersonal_warmth |") < md.index(line) < md.index("## Venture")
    record = synthetic_record()
    record["prepared"][2].update(value=None, missing="edited_gap")
    assert "prepared: SUBS 3 · COH 4 · CTX ∅(edited_gap) · EVID 5" in rendered(tmp_path, record=record)


def test_venture_shows_the_ask(tmp_path):
    assert "category: reusable food container; ask: $200,000 for 10%\n" in rendered(tmp_path)
    record = synthetic_record()
    record["venture"].update(ask_amount_usd=None, ask_equity_pct=None, ask_missing="not_observable")
    assert "category: reusable food container; ask: not_observable\n" in rendered(tmp_path, record=record)
    record["venture"].update(ask_amount_usd=150000, ask_equity_pct=12.5, ask_missing="none")
    assert "; ask: $150,000 for 12.5%\n" in rendered(tmp_path, record=record)
