import pytest

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.review_apply import (
    CLUSTER_BOOTSTRAP, OVERRIDE_NOTE, RESOLUTIONS, WILSON_METHOD, Correction, ReviewError, Tolerances, agrees,
    apply_corrections,
    alpha_level, cluster_interval, human_agreement, human_human, level_of, parse_sheet, pooled_summary, structural_rate,
    value_kind, variable_of,
    wilson, within_one,
)
from tests.test_review_items import get_value
from tests.panel_fixtures import synthetic_record


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def primary(mutate=None) -> PanelRecord:
    d = synthetic_record()
    if mutate:
        mutate(d)
    return PanelRecord.model_validate(d)


def sheet_row(path, primary="", replicate="", human_value="", resolution="", note="", row=""):
    return {"row": row, "path": path, "primary": primary, "replicate": replicate, "human_value": human_value,
            "resolution": resolution, "note": note}


def fix(path, value, resolution="correct", note=""):
    return Correction(path, value, resolution, note, "ana")


# ---- parse_sheet --------------------------------------------------------------------------

def test_a_row_counts_only_when_a_value_or_a_resolution_is_filled():
    rows = [sheet_row("turn:2/function", "question", "question"),
            sheet_row("turn:3/function", "answer", "answer", human_value="  "),
            sheet_row("turn:4/function", "question", "answer", human_value="challenges")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert corrections == [Correction("turn:4/function", "challenges", "correct", "", "ana")]


def test_resolution_defaults_from_the_value():
    rows = [sheet_row("turn:2/function", "question", "answer", human_value=" Question ", note="clear"),
            sheet_row("turn:4/function", "question", "answer", human_value="challenges"),
            sheet_row("turn:10/exchange", "", "2", human_value="2")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert corrections == [Correction("turn:2/function", None, "confirm_primary", "clear", "ana"),
                           Correction("turn:4/function", "challenges", "correct", "", "ana"),
                           Correction("turn:10/exchange", "2", "correct", "", "ana")]


def test_each_explicit_resolution():
    rows = [sheet_row("turn:2/function", "question", "challenges", resolution="confirm_primary"),
            sheet_row("turn:4/function", "question", "challenges", resolution="Confirm_Replicate"),
            sheet_row("turn:9/exchange", "2", "", resolution="correct"),                     # corrected to empty
            sheet_row("turn:6/function", "repair", "answer", resolution="unclear", note="inaudible"),
            sheet_row("turn:7/function", "accepts", "", human_value="two turns", resolution="structural",
                      note="split at 1:42")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert [(c.path, c.human_value, c.resolution) for c in corrections] == [
        ("turn:2/function", None, "confirm_primary"), ("turn:4/function", "challenges", "confirm_replicate"),
        ("turn:9/exchange", None, "correct"), ("turn:6/function", None, "unclear"),
        ("turn:7/function", "two turns", "structural")]
    assert corrections[3].note == "inaudible" and all(c.reviewer == "ana" for c in corrections)
    assert set(RESOLUTIONS) == {"confirm_primary", "confirm_replicate", "correct", "unclear", "structural"}


def test_problems_name_the_row_and_the_reason():
    rows = [sheet_row("turn:2/function", "question", "", resolution="confirm_replicate", row="7"),
            sheet_row("turn:3/function", "answer", "", resolution="approve"),
            sheet_row("turn:4/function", "question", "answer", human_value="answer", resolution="confirm_primary"),
            sheet_row("turn:5/function", "answer", "accepts", human_value="answer", resolution="confirm_replicate"),
            sheet_row("turn:6/function", "repair", "", note="not sure what this is"),
            sheet_row("", "x", "", human_value="y"),
            sheet_row("turn:7/function", "accepts", "", human_value="challenges"),
            sheet_row("turn:7/function", "accepts", "", human_value="social")]
    corrections, problems = parse_sheet(rows, "ana")
    assert [c.path for c in corrections] == ["turn:7/function"]
    assert len(problems) == 7
    assert problems[0].startswith("row 7:") and "no replicate value" in problems[0]
    assert problems[1].startswith("row 2:") and "'approve'" in problems[1]
    assert "confirm_primary" in problems[2] and "'answer'" in problems[2]
    assert "confirm_replicate" in problems[3]
    assert problems[4].startswith("row 5:") and "note" in problems[4]
    assert "no path" in problems[5]
    assert problems[6].startswith("row 8:") and "more than once" in problems[6]


def test_a_unit_presence_row_confirms_or_becomes_a_structural_judgment():
    rows = [sheet_row("turn:13/present", "present", "absent", human_value="Present"),
            sheet_row("exchange:2/present", "present", "absent", human_value="absent"),
            sheet_row("decision:1/present", "", "", resolution="confirm_replicate", note="never said"),   # blind cells
            sheet_row("replicate_only:turn@2:10", "absent", "present", human_value="present"),
            sheet_row("replicate_only:decision@1:51", "absent", "present", human_value="absent"),
            sheet_row("turn:7/present", "present", "absent", resolution="unclear", note="off camera"),
            sheet_row("turn:8/present", "present", "absent", resolution="structural", note="two turns in one")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert [(c.path, c.human_value, c.resolution) for c in corrections] == [
        ("turn:13/present", None, "confirm_primary"), ("exchange:2/present", "absent", "structural"),
        ("decision:1/present", "absent", "structural"), ("replicate_only:turn@2:10", "present", "structural"),
        ("replicate_only:decision@1:51", None, "confirm_primary"), ("turn:7/present", None, "unclear"),
        ("turn:8/present", None, "structural")]
    assert "nothing was removed" in corrections[1].note and corrections[2].note == "never said"
    assert "the primary missed" in corrections[3].note and corrections[6].note == "two turns in one"

    bad = [sheet_row("turn:13/present", "present", "absent", human_value="maybe", row="4"),
           sheet_row("turn:12/present", "present", "absent", resolution="correct"),
           sheet_row("replicate_only:turn@2:10", "absent", "present", human_value="yes", resolution="structural")]
    corrections, problems = parse_sheet(bad, "ana")
    assert corrections == [] and len(problems) == 3
    assert problems[0] == "row 4: turn:13/present takes present or absent, not 'maybe'"


def test_an_outcome_row_confirms_or_becomes_an_override():
    cases = [  # (primary, replicate, human_value, resolution, note) -> (human_value, resolution)
        (("yes", "yes", "Yes", "", ""), (None, "confirm_primary")),
        (("yes", "no", "no", "", "he only nods"), ("no", "outcome_override")),
        (("no", "yes", "", "confirm_replicate", ""), ("yes", "outcome_override")),
        (("unknown", "", "unknown", "", ""), (None, "confirm_primary")),
        (("no", "", "unknown", "correct", ""), ("unknown", "outcome_override")),
        (("no", "no", "", "unclear", "cut away"), (None, "unclear")),
        (("yes", "yes", "", "confirm_primary", ""), (None, "confirm_primary"))]
    for (p, r, human, resolution, note), expected in cases:
        corrections, problems = parse_sheet([sheet_row("pitch:any_offer", p, r, human_value=human,
                                                       resolution=resolution, note=note)], "ana")
        assert problems == [] and [(c.human_value, c.resolution) for c in corrections] == [expected], (p, human)
        if expected[1] == "outcome_override":
            assert corrections[0].note == (note or OVERRIDE_NOTE) and ("not edited" in corrections[0].note or note)
    assert "outcome_override" not in RESOLUTIONS                       # set by the import, never typed by a coder

    for row, problem in ((sheet_row("pitch:any_offer", "yes", "yes", human_value="maybe", row="3"),
                          "row 3: pitch:any_offer takes yes, no, unknown, not 'maybe'"),
                         (sheet_row("pitch:any_offer", "yes", "no", resolution="correct", row="4"),
                          "row 4: pitch:any_offer takes yes, no, unknown, not None"),
                         (sheet_row("pitch:any_offer", "yes", "no", resolution="outcome_override", human_value="no"),
                          "unknown resolution")):
        corrections, problems = parse_sheet([row], "ana")
        assert corrections == [] and len(problems) == 1 and problem in problems[0]
    _, problems = parse_sheet([sheet_row("pitch:any_offer", "yes", "", human_value="no", row="1"),
                               sheet_row(" pitch:any_offer", "yes", "", human_value="yes", row="2")], "ana")
    assert problems == ["row 2: pitch:any_offer is reviewed more than once"]       # one outcome per pitch


def test_not_an_exchange_is_a_structural_judgment_on_a_topic_only():
    rows = [sheet_row("exchange:1/topic", "operating_efficiency", "forecast", human_value="Not_An_Exchange"),
            sheet_row("exchange:3/topic", "customer_retention", "", human_value="not_an_exchange", note="one aside"),
            sheet_row("exchange:1/kind", "substantive", "", human_value="not_an_exchange")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert [(c.human_value, c.resolution) for c in corrections] == [
        ("not_an_exchange", "structural"), ("not_an_exchange", "structural"), ("not_an_exchange", "correct")]
    assert "nothing was removed" in corrections[0].note and corrections[1].note == "one aside"


# ---- apply_corrections --------------------------------------------------------------------

def test_each_resolution_is_applied_and_logged(books):
    p = primary()
    before = p.model_dump(mode="json")
    corrections = [
        fix("turn:2/code:PR-SEV", None, "confirm_primary"),
        fix("turn:4/code:PR-SEV", "2", "confirm_replicate"),
        fix("stage_a:interpersonal_warmth", "5", "correct", "very bright"),
        fix("turn:9/exchange", None, "correct"),
        fix("turn:6/function", None, "unclear", "inaudible"),
        fix("turn:7/function", "two turns", "structural", "split at 1:42"),
    ]
    reviewed, log = apply_corrections(p, corrections, books)
    assert p.model_dump(mode="json") == before                         # the primary is untouched
    after = PanelRecord.model_validate(reviewed)
    assert get_value(after, "turn:4/code:PR-SEV") == "2"
    assert get_value(after, "stage_a:interpersonal_warmth") == "5"
    assert get_value(after, "turn:9/exchange") is None
    assert log == [
        {"path": "turn:2/code:PR-SEV", "old": "1", "new": "1", "resolution": "confirm_primary", "note": "",
         "reviewer": "ana"},
        {"path": "turn:4/code:PR-SEV", "old": "3", "new": "2", "resolution": "confirm_replicate", "note": "",
         "reviewer": "ana"},
        {"path": "stage_a:interpersonal_warmth", "old": "3", "new": "5", "resolution": "correct", "note": "very bright",
         "reviewer": "ana"},
        {"path": "turn:9/exchange", "old": "2", "new": None, "resolution": "correct", "note": "", "reviewer": "ana"},
        {"path": "turn:6/function", "old": "repair", "new": "repair",
         "resolution": "unclear", "note": "inaudible", "reviewer": "ana"},
        {"path": "turn:7/function", "old": "accepts", "new": "accepts", "resolution": "structural",
         "note": "split at 1:42", "reviewer": "ana"},
    ]
    # nothing else moved
    changed = {"turn:4/code:PR-SEV": "3", "stage_a:interpersonal_warmth": "3", "turn:9/exchange": "2"}
    restored = reviewed
    from pipeline.stage_b.review_items import set_value
    for path, old in changed.items():
        set_value(restored, path, old, books)
    # ... except the note of the Stage A item, which says its value was changed in review
    assert restored["stage_a"][2]["note"] == "Steady delivery. [value changed in review]"
    restored["stage_a"][2]["note"] = "Steady delivery."
    assert restored == before


def test_no_corrections_return_the_primary_unchanged(books):
    p = primary()
    reviewed, log = apply_corrections(p, [], books)
    assert reviewed == p.model_dump(mode="json") and log == []


def test_a_function_change_with_its_codes_in_the_same_sheet_applies(books):
    corrections = [fix("turn:13/code:PR-SEV", "2"),                    # the codes come before the function
                   fix("turn:13/function", "question"),
                   fix("turn:13/code:PR-CONCEPT", "customer_retention"),
                   fix("turn:2/function", "challenges", note="not a question")]
    reviewed, log = apply_corrections(primary(), corrections, books)
    turn13, turn2 = reviewed["turns"][12], reviewed["turns"][1]
    assert turn13["function"] == "question"
    assert turn13["codes"] == [{"code": "PR-CONCEPT", "value": "customer_retention", "missing": "none"},
                               {"code": "PR-SEV", "value": "2", "missing": "none"}]
    assert turn2["function"] == "challenges" and turn2["codes"] == []
    assert [(e["path"], e["old"], e["new"]) for e in log] == [
        ("turn:13/code:PR-SEV", None, "2"), ("turn:13/function", "accepts", "question"),
        ("turn:13/code:PR-CONCEPT", None, "customer_retention"),
        ("turn:2/function", "question", "challenges"),
        ("turn:2/code:PR-CONCEPT", "operating_efficiency", None), ("turn:2/code:PR-SEV", "1", None)]
    assert log[4]["resolution"] == "correct" and "challenges" in log[4]["note"] and log[4]["reviewer"] == "ana"


def test_a_function_change_without_its_codes_aborts_naming_them(books):
    with pytest.raises(ReviewError) as exc:
        apply_corrections(primary(), [fix("turn:13/function", "question"), fix("turn:13/code:PR-SEV", "2")], books)
    assert len(exc.value.problems) == 1
    assert "turn:13/function" in exc.value.problems[0] and "PR-CONCEPT" in exc.value.problems[0]
    assert "PR-SEV" not in exc.value.problems[0].split("needs codes")[1].split(";")[0]


def test_invalid_values_abort_the_import_with_every_error_collected(books):
    p = primary()
    before = p.model_dump(mode="json")
    corrections = [fix("turn:4/code:PR-SEV", "2"), fix("turn:2/code:PR-SEV", "7"),
                   fix("exchange:1/topic", "pricing"), fix("turn:99/function", None, "confirm_primary"),
                   fix("nonsense", "x")]
    with pytest.raises(ReviewError) as exc:
        apply_corrections(p, corrections, books)
    problems = exc.value.problems
    assert len(problems) == 4
    assert any("turn:2/code:PR-SEV" in x and "outside 1-3" in x for x in problems)
    assert any("exchange:1/topic" in x and "not one of" in x for x in problems)
    assert any("turn:99/function" in x and "no such path" in x for x in problems)
    assert any(x.startswith("nonsense") for x in problems)
    assert "4 problems" in str(exc.value)
    assert p.model_dump(mode="json") == before


def test_dependent_edits_are_applied_in_an_order_that_makes_them_valid(books):
    corrections = [fix("turn:13/code:PR-SEV", "2"),                    # needs the function changed first
                   fix("turn:13/code:PR-CONCEPT", "sales"),
                   fix("exchange:3/topic", "sales"),
                   fix("turn:13/function", "question"),
                   fix("decision:2/kind", "revises_offer")]
    reviewed, log = apply_corrections(primary(), corrections, books)
    assert reviewed["turns"][12]["function"] == "question" and reviewed["exchanges"][2]["topic"] == "sales"
    assert [e["path"] for e in log] == [c.path for c in corrections]      # the log keeps the sheet's order
    assert log[0]["old"] is None and log[0]["new"] == "2"


def test_an_unclear_row_on_an_unknown_path_is_logged_not_refused(books):
    reviewed, log = apply_corrections(primary(), [fix("turn:99/function", None, "structural", "a turn is missing "
                                                                                              "after 2:10")], books)
    assert reviewed == primary().model_dump(mode="json")
    assert log == [{"path": "turn:99/function", "old": None, "new": None, "resolution": "structural",
                    "note": "a turn is missing after 2:10", "reviewer": "ana"}]


def test_a_confirmation_writes_nothing_even_when_the_record_is_spelled_differently(books):
    p = primary(lambda d: d["turns"][1].update(function="Question"))
    reviewed, log = apply_corrections(p, [fix("turn:2/function", None, "confirm_primary")], books)
    assert reviewed == p.model_dump(mode="json")
    assert (log[0]["old"], log[0]["new"]) == ("question", "question")


def test_unit_presence_judgments_are_logged_and_change_nothing(books):
    p = primary()
    corrections = [fix("turn:13/present", None, "confirm_primary"),
                   fix("exchange:2/present", "absent", "structural", "not an exchange"),
                   fix("replicate_only:turn@2:10", "present", "structural", "missed"),
                   fix("replicate_only:decision@1:51", None, "confirm_primary"),
                   fix("turn:7/present", None, "unclear", "off camera"),
                   fix("turn:8/present", None, "structural", "two turns in one")]
    reviewed, log = apply_corrections(p, corrections, books)
    assert reviewed == p.model_dump(mode="json")                        # nothing is deleted, nothing is added
    assert [(e["path"], e["old"], e["new"], e["resolution"]) for e in log] == [
        ("turn:13/present", "present", "present", "confirm_primary"),
        ("exchange:2/present", "present", "absent", "structural"),
        ("replicate_only:turn@2:10", "absent", "present", "structural"),
        ("replicate_only:decision@1:51", "absent", "absent", "confirm_primary"),
        ("turn:7/present", "present", None, "unclear"),
        ("turn:8/present", "present", None, "structural")]
    assert log[1]["note"] == "not an exchange" and log[1]["reviewer"] == "ana"
    with pytest.raises(ReviewError) as exc:
        apply_corrections(p, [fix("turn:99/present", None, "confirm_primary"),
                              fix("turn:13/present", "absent", "correct")], books)
    assert len(exc.value.problems) == 2
    assert "turn 99 is not in the record" in exc.value.problems[0] and "cannot be written" in exc.value.problems[1]


def test_outcome_judgments_are_logged_beside_the_record_which_does_not_change(books):
    p = primary()
    corrections = [fix("pitch:any_offer", "no", "outcome_override", "the offer is a joke"),
                   fix("pitch:first_decision", None, "unclear"),
                   fix("exchange:1/topic", "not_an_exchange", "structural", "an aside")]
    reviewed, log = apply_corrections(p, corrections, books)
    assert reviewed == p.model_dump(mode="json")                        # the decisions are not edited
    assert [(e["path"], e["old"], e["new"], e["resolution"]) for e in log] == [
        ("pitch:any_offer", "yes", "no", "outcome_override"),
        ("pitch:first_decision", "1:51", None, "unclear"),
        ("exchange:1/topic", "operating_efficiency", "operating_efficiency", "structural")]
    assert log[0]["note"] == "the offer is a joke"
    _, log = apply_corrections(p, [fix("pitch:any_offer", None, "confirm_primary")], books)
    assert [(e["old"], e["new"]) for e in log] == [("yes", "yes")]
    with pytest.raises(ReviewError) as exc:
        apply_corrections(p, [fix("investor:S1/ever_offer", None, "confirm_primary"),
                              fix("pitch:any_offer", "no", "correct"),
                              fix("pitch:any_offer", "perhaps", "outcome_override"),
                              fix("turn:2/function", "answer", "outcome_override")], books)
    assert len(exc.value.problems) == 4
    assert "no such path" in exc.value.problems[0] and "cannot be written" in exc.value.problems[1]
    assert "perhaps" in exc.value.problems[2]
    assert "pitch:any_offer and pitch:first_decision only" in exc.value.problems[3]


def agreement_row(path, field_class, primary, replicate, selected_as):
    return {"path": path, "field_class": field_class, "primary": primary, "replicate": replicate,
            "selected_as": selected_as, "reasons": []}


def test_human_agreement_arithmetic():
    rows = [
        agreement_row("turn:2/function", "turn_function", "question", "question", "sampled"),
        agreement_row("turn:4/function", "turn_function", "question", "challenges", "flagged"),
        agreement_row("turn:6/function", "turn_function", "repair", "answer", "flagged"),
        agreement_row("turn:13/function", "turn_function", "accepts", None, "flagged"),
        agreement_row("turn:16/function", "turn_function", "challenges", "challenges", "sampled"),
        agreement_row("turn:2/code:PR-SEV", "turn_code", "1", "1", "sampled"),
        agreement_row("turn:4/code:PR-SEV", "turn_code", "3", "1", "flagged"),
        agreement_row("turn:9/exchange", "turn_exchange", "2", "2", "sampled"),        # not reviewed
    ]
    corrections = [
        fix("turn:2/function", None, "confirm_primary"),        # = primary = replicate
        fix("turn:4/function", "challenges", "confirm_replicate"),   # = replicate
        fix("turn:6/function", "social"),                        # neither
        fix("turn:13/function", "Accepts"),                      # = primary, no replicate value
        fix("turn:16/function", None, "unclear"),                # no definite value
        fix("turn:2/code:PR-SEV", "2"),                          # neither; the passes agree
        fix("turn:4/code:PR-SEV", " 3 "),                        # = primary
    ]
    report = human_agreement(rows, corrections)
    assert report["n_reviewed"] == 6 and report["n_without_a_value"] == 1
    assert "unbiased" in report["note"] and "sampled" in report["note"]
    f = report["by_field_class"]["turn_function"]
    assert f == {"n": 4,
                 "human_vs_primary": {"agree": 0.5, "n": 4},
                 "human_vs_replicate": {"agree": 2 / 3, "n": 3},
                 "primary_vs_replicate": {"agree": 1 / 3, "n": 3},
                 # human against primary, by hand: units [q,q] [ch,q] [sc,ce] [ag,ag]; 4 disagreeing ordered pairs;
                 # expected (8*8 - (9+1+1+1+4)) / 7 = 48/7; alpha = 1 - 4 / (48/7) = 5/12. The other two by function.
                 "alpha": pytest.approx(5 / 12), "alpha_human_vs_replicate": pytest.approx(8 / 13),
                 "alpha_primary_vs_replicate": pytest.approx(1 / 6)}
    c = report["by_field_class"]["turn_code"]
    # PR-SEV is ordinal: exact agreement, and beside it agreement within one point (2 against 1 and 1; 3 against
    # 3 and 1; the passes 1 = 1 and 3 against 1)
    assert c == {"n": 2, "human_vs_primary": {"agree": 0.5, "n": 2},
                 "human_vs_replicate": {"agree": 0.0, "n": 2}, "primary_vs_replicate": {"agree": 0.5, "n": 2},
                 "within_one": {"n": 2, "human_vs_primary": {"agree": 1.0, "n": 2},
                                "human_vs_replicate": {"agree": 0.5, "n": 2},
                                "primary_vs_replicate": {"agree": 0.5, "n": 2}},
                 # ordinal alpha, by hand: human against primary units [2,1] [3,3] give 1 - 2/12 = 5/6; the other two likewise
                 "alpha": pytest.approx(5 / 6), "alpha_human_vs_replicate": pytest.approx(-5 / 12),
                 "alpha_primary_vs_replicate": pytest.approx(0.0, abs=1e-12)}
    assert "alpha" not in c["within_one"]                           # the nested share block carries no alpha
    assert "within_one" not in f                                    # a category has no within-one
    assert set(report["by_field_class"]) == {"turn_function", "turn_code"}
    assert report["all"]["n"] == 6 and report["all"]["human_vs_primary"] == {"agree": 0.5, "n": 6}
    assert report["all"]["alpha"] is None                           # categories and ordinal ratings: no one level
    assert report["by_field_class"]["turn_function"]["alpha"] is not None
    sampled, flagged = report["by_selection"]["sampled"], report["by_selection"]["flagged"]
    assert sampled["all"] == {"n": 2, "human_vs_primary": {"agree": 0.5, "n": 2},
                              "human_vs_replicate": {"agree": 0.5, "n": 2},
                              "primary_vs_replicate": {"agree": 1.0, "n": 2},
                              "alpha": None, "alpha_human_vs_replicate": None,      # a function and a rating: mixed
                              "alpha_primary_vs_replicate": None,
                              "within_one": {"n": 1, "human_vs_primary": {"agree": 1.0, "n": 1},
                                             "human_vs_replicate": {"agree": 1.0, "n": 1},
                                             "primary_vs_replicate": {"agree": 1.0, "n": 1}}}
    assert flagged["all"]["n"] == 4
    assert flagged["all"]["human_vs_primary"] == {"agree": 0.5, "n": 4}
    assert flagged["all"]["human_vs_replicate"] == {"agree": 1 / 3, "n": 3}
    assert flagged["all"]["primary_vs_replicate"] == {"agree": 0.0, "n": 3}
    assert sampled["by_field_class"]["turn_function"]["n"] == 1
    assert flagged["by_field_class"]["turn_code"]["human_vs_primary"] == {"agree": 1.0, "n": 1}


def test_human_agreement_with_nothing_reviewed_has_no_figures():
    report = human_agreement([agreement_row("turn:2/function", "turn_function", "question", "question", "sampled")],
                             [])
    assert report["n_reviewed"] == 0 and report["by_field_class"] == {}
    assert report["all"] == {"n": 0, "human_vs_primary": {"agree": None, "n": 0},
                             "human_vs_replicate": {"agree": None, "n": 0},
                             "primary_vs_replicate": {"agree": None, "n": 0},
                             "alpha": None, "alpha_human_vs_replicate": None, "alpha_primary_vs_replicate": None}
    assert report["by_selection"]["sampled"]["all"]["n"] == 0 and report["by_selection"]["flagged"]["all"]["n"] == 0


def test_a_correction_to_empty_counts_as_a_definite_value():
    rows = [agreement_row("turn:9/exchange", "turn_exchange", "2", None, "flagged"),
            agreement_row("turn:10/exchange", "turn_exchange", None, None, "sampled")]
    report = human_agreement(rows, [fix("turn:9/exchange", None), fix("turn:10/exchange", None, "confirm_primary")])
    assert report["by_field_class"]["turn_exchange"]["human_vs_primary"] == {"agree": 0.5, "n": 2}
    assert report["by_field_class"]["turn_exchange"]["human_vs_replicate"] == {"agree": None, "n": 0}


def test_unit_presence_is_reported_apart_with_the_share_judged_present():
    rows = [agreement_row("turn:13/present", "unit_presence", "present", "absent", "flagged"),
            agreement_row("turn:14/present", "unit_presence", "present", "absent", "flagged"),
            agreement_row("exchange:2/present", "unit_presence", "present", "absent", "flagged"),
            agreement_row("decision:1/present", "unit_presence", "present", "absent", "flagged"),
            agreement_row("replicate_only:turn@2:10", "unit_presence", "absent", "present", "flagged"),
            agreement_row("replicate_only:decision@1:51", "unit_presence", "absent", "present", "flagged"),
            agreement_row("turn:2/function", "turn_function", "question", "question", "sampled")]
    corrections = [fix("turn:13/present", None, "confirm_primary"), fix("turn:14/present", None, "confirm_primary"),
                   fix("exchange:2/present", "absent", "structural"), fix("decision:1/present", None, "unclear"),
                   fix("replicate_only:turn@2:10", "present", "structural"),
                   fix("replicate_only:decision@1:51", None, "confirm_primary"),
                   fix("turn:2/function", None, "confirm_primary"), fix("turn:4/function", "x", "structural")]
    report = human_agreement(rows, corrections)
    assert report["by_field_class"]["unit_presence"] == {
        "n": 5, "primary_only": {"judged_present": 2 / 3, "n": 3}, "replicate_only": {"judged_present": 0.5, "n": 2}}
    assert list(report["by_field_class"]) == ["unit_presence", "turn_function"]
    assert report["n_reviewed"] == 6 and report["n_without_a_value"] == 2
    # the judgments enter no agreement figure
    assert report["all"]["n"] == 1 and report["by_selection"]["flagged"]["all"]["n"] == 0
    assert "unit_presence" in report["note"]
    assert report["flagged_left_out"] is None and report["flagged_note"] is None


def test_the_flagged_rows_left_out_are_passed_through_with_a_note():
    rows = [agreement_row("turn:2/function", "turn_function", "question", "answer", "flagged")]
    left = {"n": 7, "by_field_class": {"turn_code": 5, "stage_a": 2}, "by_reason": {"passes_differ": 5,
                                                                                     "stage_a_gap": 2}}
    report = human_agreement(rows, [fix("turn:2/function", None, "confirm_primary")], left)
    assert report["flagged_left_out"] == left
    assert "reviewed subset only" in report["flagged_note"]
    nothing = {"n": 0, "by_field_class": {}, "by_reason": {}}
    assert human_agreement(rows, [], nothing)["flagged_left_out"] is None


def test_wilson_interval():
    assert wilson(0, 0) is None
    low, high = wilson(8, 10)
    assert round(low, 3) == 0.490 and round(high, 3) == 0.943
    assert wilson(10, 10)[1] == 1.0 and round(wilson(10, 10)[0], 3) == 0.722
    assert wilson(0, 10)[0] == 0.0 and round(wilson(0, 10)[1], 3) == 0.278


def test_sampled_estimates_carry_an_interval_the_population_and_the_outcome_class():
    rows = [agreement_row("pitch:any_offer", "outcome", "yes", "yes", "sampled"),
            agreement_row("exchange:1/topic", "exchange_topic", "sales", "sales", "sampled"),
            agreement_row("exchange:2/topic", "exchange_topic", "sales", "usage", "sampled"),
            agreement_row("exchange:3/topic", "exchange_topic", "usage", None, "sampled"),
            agreement_row("exchange:4/topic", "exchange_topic", "usage", "usage", "sampled"),
            agreement_row("turn:4/function", "turn_function", "question", "answer", "flagged")]
    corrections = [fix("pitch:any_offer", "no", "outcome_override"),
                   fix("exchange:1/topic", None, "confirm_primary"),
                   fix("exchange:2/topic", "usage", "confirm_replicate"),
                   fix("exchange:3/topic", "pricing"),
                   fix("exchange:4/topic", "not_an_exchange", "structural"),
                   fix("turn:4/function", None, "confirm_primary")]
    report = human_agreement(rows, corrections, population={"outcome": 1, "exchange_topic": 7})
    est = report["sampled_estimates"]
    assert list(est) == ["outcome", "exchange_topic"]                   # the flagged row is no estimate
    assert est["exchange_topic"] == {"n": 3, "human_vs_primary": {"agree": 1 / 3, "n": 3},
                                     "human_vs_replicate": {"agree": 1.0, "n": 2},
                                     "primary_vs_replicate": {"agree": 0.5, "n": 2},
                                     # human against primary, by hand: units [sales,sales] [usage,sales]
                                     # [pricing,usage]; observed 4, expected (36 - (9+4+1)) / 5 = 4.4;
                                     # alpha = 1 - 4/4.4 = 1/11
                                     "alpha": pytest.approx(1 / 11), "alpha_human_vs_replicate": pytest.approx(1.0),
                                     "alpha_primary_vs_replicate": pytest.approx(0.0, abs=1e-12),
                                     "human_vs_primary_ci95": wilson(1, 3), "human_vs_primary_wilson95": wilson(1, 3),
                                     "ci_method": WILSON_METHOD, "population": 7}
    assert WILSON_METHOD == "Wilson; treats rows as independent; optimistic"
    assert est["outcome"]["n"] == 1 and est["outcome"]["population"] == 1
    assert est["outcome"]["human_vs_primary"] == {"agree": 0.0, "n": 1}
    assert est["outcome"]["human_vs_primary_ci95"] == wilson(0, 1)
    assert (report["n_structural"], report["n_outcome_overrides"], report["n_without_a_value"]) == (1, 1, 1)
    assert report["n_cut_overrides"] == 0
    assert report["by_field_class"]["outcome"]["human_vs_primary"] == {"agree": 0.0, "n": 1}
    assert human_agreement(rows, corrections)["sampled_estimates"]["outcome"]["population"] is None


def test_pooled_summary_counts_videos_rows_and_notes():
    def line(path, cls, primary, replicate, new, resolution, selected_as="sampled"):
        return {"path": path, "field_class": cls, "primary": primary, "replicate": replicate, "new": new,
                "resolution": resolution, "selected_as": selected_as, "note": ""}
    videos = {
        "a": [line("pitch:any_offer", "outcome", "no", "no", "yes", "outcome_override"),
              line("exchange:1/topic", "exchange_topic", "sales", "usage", "usage", "confirm_replicate"),
              line("exchange:2/topic", "exchange_topic", "sales", "sales", "sales", "structural"),
              line("turn:4/function", "turn_function", "question", "answer", "answer", "correct", "flagged"),
              line("turn:9/present", "unit_presence", "present", "absent", "absent", "structural", "flagged")],
        "b": [line("pitch:any_offer", "outcome", "yes", None, "yes", "confirm_primary"),
              line("exchange:1/topic", "exchange_topic", "sales", "sales", "sales", "confirm_primary"),
              line("turn:5/code:RS-RESP", "turn_code", "addresses", "partial", "partial", "correct", "added")],
    }
    out = pooled_summary(videos)
    assert (out["n_videos"], out["n_rows"], out["n_structural"], out["n_outcome_overrides"]) == (2, 8, 2, 1)
    assert out["n_cut_overrides"] == 0
    assert out["n_without_a_value"] == 1
    assert list(out["sampled"]) == ["outcome", "exchange_topic"]
    outcome = out["sampled"]["outcome"]
    assert (outcome["videos"], outcome["n"]) == (2, 2) and outcome["human_vs_primary"] == {"agree": 0.5, "n": 2}
    assert outcome["primary_vs_replicate"] == {"agree": 1.0, "n": 1}
    assert outcome["human_vs_primary_ci95"] == wilson(1, 2)
    topic = out["sampled"]["exchange_topic"]
    assert (topic["videos"], topic["n"]) == (2, 2) and topic["human_vs_primary"] == {"agree": 0.5, "n": 2}
    assert set(out["flagged"]) == {"turn_function", "turn_code"}        # flagged and added rows are no estimate
    assert out["unit_presence"]["primary_only"] == {"judged_present": 0.0, "n": 1}
    empty = pooled_summary({})
    assert (empty["n_videos"], empty["sampled"], empty["flagged"], empty["unit_presence"]) == (0, {}, {}, None)


# ---- one path per field, tolerances, decisions ----------------------------------------------------

def test_two_spellings_of_one_path_are_one_field_reviewed_twice():
    rows = [sheet_row("turn:4/function", "question", "", human_value="challenges", row="3"),
            sheet_row("turn:0004/function", "question", "", human_value="accepts", row="9"),
            sheet_row(" pitch:any_offer ", "yes", "", human_value="no", row="10"),
            sheet_row("decision:02/present", "", "", human_value="absent", row="11")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == ["row 9: turn:4/function is reviewed more than once"]
    assert [c.path for c in corrections] == ["turn:4/function", "pitch:any_offer", "decision:2/present"]


def test_value_kinds_and_tolerant_agreement():
    assert value_kind("decision:1/time") == "clock" and value_kind("prepared_window") == "window"
    for path in ("stage_a:interpersonal_warmth", "turn:4/code:PR-SEV", "turn:3/code:PREP-SUBS", "turn:3/code:PREP-CTX"):
        assert value_kind(path) == "ordinal", path
    for path in ("turn:3/code:RS-RESP", "turn:3/code:RS-EVID", "decision:1/kind", "exchange:1/topic",
                 "pitch:any_offer", "turn:4/present", "investor:S1/ever_offer", "nonsense"):
        assert value_kind(path) == "exact", path
    tol = Tolerances()
    assert (tol.decision_s, tol.window_edge_s) == (30.0, 2.0)
    assert Tolerances.of({"settings": {"decision_tolerance_s": 5, "event_tolerance_s": 10}}) == Tolerances(5.0, 2.0)
    assert Tolerances.of(None) == tol and Tolerances.of({}) == tol
    # a decision time agrees within the agreement settings' tolerance, the same one selection flags by
    assert agrees("decision:1/time", "1:51", "2:21", tol) and not agrees("decision:1/time", "1:51", "2:22", tol)
    assert not agrees("decision:1/time", "1:51", "1:55", Tolerances(decision_s=3.0))
    assert not agrees("decision:1/time", "1:51", None, tol) and agrees("decision:1/time", None, "", tol)
    # the prepared window: every edge within two seconds, the same number of spans
    assert agrees("prepared_window", "0:05-1:00", "0:07-0:58", tol)
    assert not agrees("prepared_window", "0:05-1:00", "0:05-1:03", tol)
    assert not agrees("prepared_window", "0:05-1:00", "0:05-0:30;0:40-1:00", tol)
    # ordinals agree exactly; within one is a second figure
    assert not agrees("stage_a:interpersonal_warmth", "3", "4", tol) and agrees("stage_a:interpersonal_warmth", " 3", "3", tol)
    assert within_one("3", "4") and within_one("4", "3") and not within_one("3", "5")
    assert within_one(None, "") and not within_one("3", None) and not within_one("missing:not_applicable", "3")
    assert agrees("turn:4/function", "Question", "question", tol) and not agrees("turn:4/function", "a", "b", tol)


def test_a_clock_within_tolerance_confirms_the_primary_instead_of_correcting_it():
    rows = [sheet_row("decision:1/time", "1:51", "2:40", human_value="1:55"),             # 4 s off: the same moment
            sheet_row("decision:2/time", "2:39", "", human_value="3:30"),                 # 51 s off: a correction
            sheet_row("prepared_window", "0:05-1:00", "", human_value="0:06-1:01"),
            sheet_row("decision:3/time", "2:44", "", human_value="2:50", resolution="confirm_primary"),
            sheet_row("decision:4/time", "2:50", "", human_value="2:52", resolution="correct"),   # asked for: written
            sheet_row("stage_a:interpersonal_warmth", "3", "", human_value="4")]                 # an ordinal is exact
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == []
    assert [(c.path, c.human_value, c.resolution) for c in corrections] == [
        ("decision:1/time", None, "confirm_primary"), ("decision:2/time", "3:30", "correct"),
        ("prepared_window", None, "confirm_primary"), ("decision:3/time", None, "confirm_primary"),
        ("decision:4/time", "2:52", "correct"), ("stage_a:interpersonal_warmth", "4", "correct")]
    tight, _ = parse_sheet(rows[:1], "ana", Tolerances(decision_s=2.0))
    assert tight[0].resolution == "correct"


def test_agreement_counts_clocks_within_tolerance_and_reports_ordinals_exact_and_within_one():
    rows = [agreement_row("decision:1/time", "decision", "1:51", "2:40", "sampled"),
            agreement_row("decision:2/time", "decision", "2:39", "2:41", "sampled"),
            agreement_row("decision:2/kind", "decision", "offer", "offer", "sampled"),
            agreement_row("prepared_window", "prepared_window", "0:05-1:00", "0:06-1:01", "sampled"),
            agreement_row("stage_a:interpersonal_warmth", "stage_a", "3", "5", "sampled"),
            agreement_row("stage_a:pitch_positivity", "stage_a", "3", "3", "sampled"),
            agreement_row("stage_a:expressed_enthusiasm", "stage_a", "2", None, "sampled")]
    corrections = [fix("decision:1/time", "1:55"),                  # within 30 s of the primary, not of the replicate
                   fix("decision:2/time", "3:30"),                  # of neither; the passes agree with each other
                   fix("decision:2/kind", None, "confirm_primary"),
                   fix("prepared_window", "0:07-1:02"),             # within 2 s per edge of both
                   fix("stage_a:interpersonal_warmth", "4"),               # one point from each pass, which are two apart
                   fix("stage_a:pitch_positivity", None, "confirm_primary"),
                   fix("stage_a:expressed_enthusiasm", "4")]               # two points off, no replicate value
    report = human_agreement(rows, corrections)
    d = report["by_field_class"]["decision"]
    assert d == {"n": 3, "human_vs_primary": {"agree": 2 / 3, "n": 3},
                 "human_vs_replicate": {"agree": 1 / 3, "n": 3}, "primary_vs_replicate": {"agree": 2 / 3, "n": 3},
                 "alpha": None, "alpha_human_vs_replicate": None,          # a clock and a kind: no single level
                 "alpha_primary_vs_replicate": None}
    assert report["by_field_class"]["prepared_window"]["human_vs_primary"] == {"agree": 1.0, "n": 1}
    assert report["by_field_class"]["prepared_window"]["alpha"] is None     # two clocks per value: no alpha
    a = report["by_field_class"]["stage_a"]
    assert a["human_vs_primary"] == {"agree": 1 / 3, "n": 3} and a["human_vs_replicate"] == {"agree": 0.5, "n": 2}
    # three items (variables) with one unit each: none can estimate its own expected disagreement, so alpha is None
    # (pooled into one matrix it would have been -17/36: the plain alpha mixes the items' own levels)
    assert a["alpha"] is None
    assert a["within_one"] == {"n": 3, "human_vs_primary": {"agree": 2 / 3, "n": 3},
                               "human_vs_replicate": {"agree": 1.0, "n": 2},
                               "primary_vs_replicate": {"agree": 0.5, "n": 2}}
    # the sampled estimate and its interval count what agrees within tolerance
    est = report["sampled_estimates"]["decision"]
    assert est["human_vs_primary"] == {"agree": 2 / 3, "n": 3} and est["human_vs_primary_ci95"] == wilson(2, 3)
    assert report["sampled_estimates"]["stage_a"]["within_one"]["human_vs_primary"] == {"agree": 2 / 3, "n": 3}
    assert "within one" in report["note"] and "tolerance" in report["note"]
    # a tighter tolerance from the agreement settings
    strict = human_agreement(rows, corrections, tolerances=Tolerances(decision_s=3.0))
    assert strict["by_field_class"]["decision"]["human_vs_primary"] == {"agree": 1 / 3, "n": 3}
    assert strict["tolerances"] == {"decision_s": 3.0, "window_edge_s": 2.0, "cut_s": 5.0}


def test_decision_times_get_an_interval_alpha_over_seconds():
    rows = [agreement_row("decision:1/time", "decision", "1:51", "2:40", "sampled"),
            agreement_row("decision:2/time", "decision", "2:39", "2:41", "sampled")]
    report = human_agreement(rows, [fix("decision:1/time", "1:55"), fix("decision:2/time", "3:30")])
    # human against primary, in seconds: units [115,111] [210,159]; observed 2*16 + 2*2601 = 5234;
    # expected 2 * (16 + 9025 + 1936 + 9801 + 2304 + 2601) / 3 = 17122; alpha = 1 - 5234/17122
    assert report["by_field_class"]["decision"]["alpha"] == pytest.approx(1 - 5234 / 17122)
    assert report["all"]["alpha"] == pytest.approx(1 - 5234 / 17122)       # one level across the rows


def test_a_closing_correction_is_refused_on_a_record_without_closing_states(books):
    """Closing states were recorded up to v3-dev8 only; a sheet that still names one cannot change a v3-dev9 record."""
    with pytest.raises(ReviewError) as exc:
        apply_corrections(primary(), [fix("exchange:3/investor:S2/closing", "satisfied")], books)
    assert len(exc.value.problems) == 1 and "no such path" in exc.value.problems[0]


def test_a_decisions_kind_actor_and_offer_id_are_corrected_together(books):
    reviewed, log = apply_corrections(primary(), [fix("decision:1/offer_id", "O2"), fix("decision:1/kind", "offer"),
                                                  fix("decision:4/actor", "S3"), fix("decision:4/kind", "exit"),
                                                  fix("decision:4/offer_id", None)], books)
    assert (reviewed["decisions"][0]["kind"], reviewed["decisions"][0]["offer_id"]) == ("offer", "O2")
    assert (reviewed["decisions"][3]["kind"], reviewed["decisions"][3]["actor"]) == ("exit", "S3")
    assert [(e["path"], e["old"], e["new"]) for e in log] == [
        ("decision:1/offer_id", None, "O2"), ("decision:1/kind", "exit", "offer"),
        ("decision:4/actor", "F1", "S3"), ("decision:4/kind", "founder_accepts", "exit"),
        ("decision:4/offer_id", "O1", None)]
    with pytest.raises(ReviewError) as exc:
        apply_corrections(primary(), [fix("decision:1/kind", "offer"), fix("decision:2/actor", "F1"),
                                      fix("decision:4/kind", "exit"), fix("decision:4/offer_id", None)], books)
    assert len(exc.value.problems) == 3
    assert "decision:1/kind: offer needs an offer id" in exc.value.problems[0]
    assert "needs an investor, and F1 is a founder" in exc.value.problems[1]
    assert exc.value.problems[2].startswith("decision:4/kind: exit needs an investor")


def test_clock_corrections_are_checked_against_the_videos_duration(books):
    corrections = [fix("decision:1/time", "0:01:55"), fix("prepared_window", "0:05-3:30")]
    with pytest.raises(ReviewError) as exc:
        apply_corrections(primary(), corrections, books, duration_s=180.0)
    assert "H:MM:SS" in exc.value.problems[0] and "beyond the video's duration (3:00)" in exc.value.problems[1]
    reviewed, _ = apply_corrections(primary(), [fix("decision:1/time", "1:55")], books, duration_s=180.0)
    assert reviewed["decisions"][0]["time_s"] == 115.0


# ---- intervals over videos ------------------------------------------------------------------------

def _line(path, cls, primary, new, resolution="correct", selected_as="sampled", replicate=None):
    return {"path": path, "field_class": cls, "primary": primary, "replicate": replicate, "new": new,
            "resolution": resolution, "selected_as": selected_as, "note": ""}


def test_cluster_interval_resamples_videos_and_is_seeded():
    per_video = [(4, 4), (4, 4), (4, 4), (0, 4), (0, 4), (2, 4)]         # agreement clusters within videos
    low, high = cluster_interval(per_video, n_boot=1999, seed=0)
    assert cluster_interval(per_video, n_boot=1999, seed=0) == [low, high]
    uneven = [(4, 4), (3, 5), (1, 6), (0, 3), (5, 5), (2, 4), (6, 7), (1, 2), (7, 9)]
    assert cluster_interval(uneven, seed=0) != cluster_interval(uneven, seed=1)
    assert cluster_interval(uneven, seed=0) == cluster_interval(uneven)       # 1,999 draws, seed 0 by default
    assert 0.0 <= low < 14 / 24 < high <= 1.0
    w_low, w_high = wilson(14, 24)
    assert high - low > w_high - w_low                                    # rows of one video are not independent
    assert cluster_interval([(3, 3)] * 6, n_boot=200, seed=0) == [1.0, 1.0]
    assert cluster_interval([], n_boot=200, seed=0) is None


def test_pooled_summary_uses_a_cluster_bootstrap_from_five_videos_and_labels_wilson_below():
    def video(agreeing: int):
        return [_line(f"exchange:{i}/topic", "exchange_topic", "sales", "sales" if i < agreeing else "usage",
                      "confirm_primary" if i < agreeing else "correct") for i in range(4)]
    six = {f"v{i}": video(n) for i, n in enumerate((4, 4, 4, 0, 0, 2))}
    out = pooled_summary(six)
    topic = out["sampled"]["exchange_topic"]
    assert (topic["videos"], topic["n"]) == (6, 24) and topic["human_vs_primary"] == {"agree": 14 / 24, "n": 24}
    assert topic["ci_method"] == CLUSTER_BOOTSTRAP == "cluster bootstrap over videos (1,999 draws, seed 0)"
    assert topic["human_vs_primary_ci95"] == cluster_interval([(4, 4), (4, 4), (4, 4), (0, 4), (0, 4), (2, 4)])
    assert topic["human_vs_primary_wilson95"] == wilson(14, 24)
    four = pooled_summary({k: v for k, v in list(six.items())[:4]})
    topic = four["sampled"]["exchange_topic"]
    assert topic["videos"] == 4 and topic["ci_method"] == WILSON_METHOD
    assert topic["human_vs_primary_ci95"] == wilson(12, 16) == topic["human_vs_primary_wilson95"]
    # five videos reviewed, but a class only three of them carry falls back to Wilson
    five = {k: list(v) for k, v in list(six.items())[:5]}
    for k in ("v0", "v1", "v2"):
        five[k].append(_line("stage_a:interpersonal_warmth", "stage_a", "3", "4"))
    out = pooled_summary(five)
    assert out["sampled"]["exchange_topic"]["ci_method"] == CLUSTER_BOOTSTRAP
    assert out["sampled"]["stage_a"]["ci_method"] == WILSON_METHOD
    assert out["sampled"]["stage_a"]["within_one"]["human_vs_primary"] == {"agree": 1.0, "n": 3}


def test_pooled_summary_counts_agreement_within_each_videos_tolerance():
    videos = {"a": [_line("decision:1/time", "decision", "1:51", "1:58")],
              "b": [_line("decision:1/time", "decision", "1:51", "1:58")]}
    assert pooled_summary(videos)["sampled"]["decision"]["human_vs_primary"] == {"agree": 1.0, "n": 2}
    out = pooled_summary(videos, tolerances={"b": Tolerances(decision_s=5.0)})
    assert out["sampled"]["decision"]["human_vs_primary"] == {"agree": 0.5, "n": 2}


# ---- coder against coder and the structural rate ----------------------------------------------------

def test_human_human_compares_two_coders_on_rows_both_judged():
    ana = {"stage_a:interpersonal_warmth": _line("stage_a:interpersonal_warmth", "stage_a", None, "4"),
           "turn:2/code:PR-SEV": _line("turn:2/code:PR-SEV", "turn_code", None, "2"),
           "exchange:1/topic": _line("exchange:1/topic", "exchange_topic", None, None, "unclear"),
           "pitch:any_offer": _line("pitch:any_offer", "outcome", "yes", None, "confirm_primary")}
    ben = {"stage_a:interpersonal_warmth": _line("stage_a:interpersonal_warmth", "stage_a", None, "5"),
           "turn:2/code:PR-SEV": _line("turn:2/code:PR-SEV", "turn_code", None, "2"),
           "exchange:1/topic": _line("exchange:1/topic", "exchange_topic", None, "sales"),
           "pitch:any_offer": _line("pitch:any_offer", "outcome", "yes", "no")}
    out = human_human({"ana": ana, "ben": ben})
    assert out["reviewers"] == ["ana", "ben"] and out["n_both"] == 3 and out["n_dropped_unclear"] == 1
    assert out["all"] == {"agree": pytest.approx(1 / 3), "n": 3}
    assert out["by_field_class"]["stage_a"]["within_one"] == {"agree": 1.0, "n": 1}
    assert out["by_field_class"]["outcome"]["agree"] == 0.0


def test_human_human_needs_exactly_two_coders_and_handles_none():
    assert human_human({"ana": {}}) == {"reviewers": ["ana"], "n_both": 0, "n_dropped_unclear": 0,
                                        "n_dropped_structural": 0, "all": {"agree": None, "n": 0}, "by_field_class": {}}
    with pytest.raises(ValueError, match="two"):
        human_human({"a": {}, "b": {}, "c": {}})


def test_structural_rate_per_field_class():
    lines = [_line("exchange:1/topic", "exchange_topic", None, None, "structural"),
             _line("exchange:3/topic", "exchange_topic", None, "sales"),
             _line("turn:2/code:PR-SEV", "turn_code", None, "2")]
    assert structural_rate(lines) == {"exchange_topic": {"n_rows": 2, "n_structural": 1, "share": 0.5},
                                      "turn_code": {"n_rows": 1, "n_structural": 0, "share": 0.0}}


def test_pooled_summary_adds_coders_the_pair_and_the_structural_rate_without_changing_the_rest():
    def coder(value):
        return [_line("stage_a:interpersonal_warmth", "stage_a", "4", value),
                _line("exchange:1/topic", "exchange_topic", None, None, "structural")]
    videos = {"v1": coder("4"), "v2": coder("5")}
    plain = pooled_summary(videos)
    both = pooled_summary(videos, by_reviewer={"ana": {"v1": coder("4"), "v2": coder("4")},
                                               "ben": {"v1": coder("4"), "v2": coder("5")}})
    added = {"by_reviewer", "human_human", "structural", "structural_by_reviewer"}
    assert {k: both[k] for k in plain if k not in added} == {k: v for k, v in plain.items() if k not in added}
    assert set(both["by_reviewer"]) == {"ana", "ben"}
    assert both["by_reviewer"]["ben"]["sampled"]["stage_a"]["human_vs_primary"] == {"agree": 0.5, "n": 2}
    hh = both["human_human"]["stage_a"]
    assert (hh["videos"], hh["n"], hh["agree"]) == (2, 2, 0.5) and hh["ci_method"] == WILSON_METHOD
    assert hh["within_one"] == {"agree": 1.0, "n": 2}
    assert both["structural"]["exchange_topic"] == {"n_rows": 2, "n_structural": 2, "share": 1.0}
    assert both["structural_by_reviewer"]["ana"]["exchange_topic"]["n_rows"] == 2
    assert pooled_summary(videos)["structural"] == both["structural"]


def test_pooled_human_human_skips_a_video_without_exactly_two_coders():
    one = {"ana": {"v1": [_line("stage_a:interpersonal_warmth", "stage_a", None, "4")]}}
    assert pooled_summary({"v1": one["ana"]["v1"]}, by_reviewer=one)["human_human"] == {"by_variable": {}}


# ---- unit_presence, typed clocks, pooled coder against coder ----------------------------------------

def test_human_human_and_structural_rate_leave_unit_presence_rows_out():
    ordinary = "exchange:1/topic"
    def coder(*judgments):
        lines = {f"turn:{i}/present": _line(f"turn:{i}/present", "unit_presence", "present", j,
                                           "confirm_primary" if j is None else "structural")
                 for i, j in zip((1, 2, 3), judgments)}
        lines[ordinary] = _line(ordinary, "exchange_topic", None, "sales")
        return lines
    ana, ben = coder(None, "absent", "absent"), coder(None, "present", "absent")
    out = human_human({"ana": ana, "ben": ben})
    assert out["n_both"] == 1 and out["n_dropped_structural"] == 0 and out["n_dropped_unclear"] == 0
    assert set(out["by_field_class"]) == {"exchange_topic"}
    assert set(structural_rate(list(ana.values()))) == {"exchange_topic"}


def test_human_human_compares_what_each_coder_typed_for_a_clock():
    path, tol = "decision:1/time", Tolerances(decision_s=5.0)
    assert value_kind(path) == "clock"
    def coder(typed):
        line = _line(path, "decision", "1:51", "1:51", "confirm_primary")
        return {path: {**line, "typed": typed}}
    close = human_human({"a": coder("1:53"), "b": coder("1:58")}, tol)       # 5 s apart
    far = human_human({"a": coder("1:47"), "b": coder("1:56")}, tol)         # 9 s apart
    assert close["all"] == {"agree": 1.0, "n": 1} and far["all"] == {"agree": 0.0, "n": 1}
    bare = {path: _line(path, "decision", "1:51", "1:51", "confirm_primary")}          # no `typed`: the primary
    assert human_human({"a": bare, "b": coder("1:58")}, tol)["all"]["agree"] == 0.0
    assert human_human({"a": bare, "b": bare}, tol)["all"]["agree"] == 1.0


def test_pooled_human_human_over_six_videos_is_a_cluster_bootstrap():
    def coder(second_agrees):
        return [_line("exchange:1/topic", "exchange_topic", None, "sales"),
                _line("exchange:2/topic", "exchange_topic", None, "sales" if second_agrees else "pricing")]
    by = {"ana": {}, "ben": {}}
    for i in range(6):
        by["ana"][f"v{i}"] = coder(True)
        by["ben"][f"v{i}"] = coder(i < 3)
    hh = pooled_summary({}, by_reviewer=by)["human_human"]["exchange_topic"]
    assert (hh["videos"], hh["n"], hh["agree"]) == (6, 12, 0.75) and hh["ci_method"] == CLUSTER_BOOTSTRAP


# ---- Krippendorff's alpha beside the shares ----------------------------------------------------------

def test_pooled_alpha_is_pooled_on_the_rows_with_a_video_cluster_interval_from_five_videos():
    from pipeline.reliability.agreement import bootstrap_alpha, krippendorff_alpha

    def video(agreeing: int):
        return [_line(f"exchange:{i}/topic", "exchange_topic", "sales" if i % 2 else "usage",
                      ("sales" if i % 2 else "usage") if i < agreeing else "pricing",
                      "confirm_primary" if i < agreeing else "correct") for i in range(4)]
    six = {f"v{i}": video(n) for i, n in enumerate((4, 4, 4, 0, 1, 2))}
    topic = pooled_summary(six)["sampled"]["exchange_topic"]
    units = {v: [[x["primary"], x["new"] if x["resolution"] != "confirm_primary" else x["primary"]] for x in lines]
             for v, lines in six.items()}
    pooled = krippendorff_alpha([u for us in units.values() for u in us], "nominal")
    assert topic["alpha"] == pytest.approx(pooled) and 0 < pooled < 1
    assert topic["alpha_ci95"] == bootstrap_alpha({v: [[u[1], u[0]] for u in us] for v, us in units.items()},
                                                   "nominal", 1999, 0)
    assert topic["alpha_human_vs_replicate"] is None and topic["alpha_primary_vs_replicate"] is None   # no replicate
    assert "alpha" not in topic["human_vs_primary"]
    four = pooled_summary({k: v for k, v in list(six.items())[:4]})["sampled"]["exchange_topic"]
    assert four["alpha_ci95"] is None and four["alpha"] is not None       # fewer than five videos: no interval
    by_coder = pooled_summary(six, by_reviewer={"ana": six})["by_reviewer"]["ana"]["sampled"]["exchange_topic"]
    assert by_coder["alpha"] == topic["alpha"] and by_coder["alpha_ci95"] == topic["alpha_ci95"]


def test_human_human_alpha_per_video_and_pooled():
    def coder(values):
        return {f"exchange:{i}/topic": _line(f"exchange:{i}/topic", "exchange_topic", None, v)
                for i, v in enumerate(values, 1)}
    per_video = human_human({"ana": coder(["a", "a", "b", "b"]), "ben": coder(["a", "b", "b", "b"])})
    cls = per_video["by_field_class"]["exchange_topic"]
    # units [a,a] [a,b] [b,b] [b,b]: observed 2, n_a = 3, n_b = 5, expected 2*3*5/7; alpha = 1 - 2/(30/7) = 8/15
    assert cls == {"agree": 0.75, "n": 4, "alpha": pytest.approx(8 / 15)}
    assert per_video["all"] == {"agree": 0.75, "n": 4}                     # the overall share stays a plain share
    by = {"ana": {}, "ben": {}}
    for i in range(6):
        by["ana"][f"v{i}"] = list(coder(["a", "a", "b", "b"]).values())
        by["ben"][f"v{i}"] = list(coder(["a", "b" if i < 3 else "a", "b", "b"]).values())
    hh = pooled_summary({}, by_reviewer=by)["human_human"]["exchange_topic"]
    # 24 units: 21 a's and 27 b's, three units split; observed 2*3, expected 2*21*27/47; alpha = 1 - 6*47/1134
    assert hh["agree"] == 0.875 and hh["alpha"] == pytest.approx(852 / 1134)
    assert isinstance(hh["alpha_ci95"], list) and 0 < hh["alpha_ci95_defined"] <= 1
    few = {name: {v: lines for v, lines in per.items() if v in ("v0", "v1")} for name, per in by.items()}
    assert pooled_summary({}, by_reviewer=few)["human_human"]["exchange_topic"]["alpha_ci95"] is None


def test_variable_of_names_the_variable_a_path_measures():
    assert variable_of("turn:4/code:PR-SEV") == "PR-SEV"
    assert variable_of("stage_a:interpersonal_warmth") == "stage_a:interpersonal_warmth"
    assert variable_of("venture:primary_industry") == "venture:primary_industry"
    assert variable_of("actor:S1/role") == "actor:role"
    assert variable_of("turn:4/function") == "turn_function"
    assert variable_of("pitch:any_offer") == "outcome"
    assert variable_of("investor:S1/ever_offer") == "other"                 # no such path since v3-dev9
    assert variable_of("decision:2/kind") == "decision:kind"
    assert variable_of("decision:2/time") == "decision:time"
    assert variable_of("exchange:1/topic") == "exchange_topic:topic"
    assert variable_of("exchange:1/label") == "exchange_topic:label"
    assert variable_of("turn:4/other_concept") == "turn_code:other_concept"
    assert variable_of("decision:2/offer_id") == "decision:offer_id" and variable_of("exchange:1/kind") == "exchange_kind"


def test_alpha_counts_an_empty_value_as_a_category_where_the_share_does():
    # human against primary on five turn_exchange rows: 1/1, 2/2, 1/1, 2/empty, empty/3 -> the share is 3/5.
    # Units [1,1] [2,2] [1,1] [2,E] [E,3]: n_1 = 4, n_2 = 3, n_E = 2, n_3 = 1 (n = 10); observed 4 ordered pairs;
    # expected (100 - (16+9+4+1)) / 9 = 70/9; alpha = 1 - 4*9/70 = 17/35 (dropping the empties gave 1.0).
    rows = [agreement_row(f"turn:{i}/exchange", "turn_exchange", p, None, "sampled")
            for i, p in enumerate(["1", "2", "1", None, "3"], 1)]
    corrections = [fix(f"turn:{i}/exchange", h) for i, h in enumerate(["1", "2", "1", "2", None], 1)]
    f = human_agreement(rows, corrections)["by_field_class"]["turn_exchange"]
    assert f["human_vs_primary"] == {"agree": 0.6, "n": 5}
    assert f["alpha"] == pytest.approx(17 / 35)


def _stage_a_lines(items, videos):
    return {f"v{v}": [_line(f"stage_a:{item}", "stage_a", p, h) for item, pairs in items.items() for p, h in pairs[v]]
            for v in range(videos)}


def test_a_pooled_class_of_several_variables_is_stratified_and_each_variable_is_reported():
    from pipeline.reliability.agreement import krippendorff_alpha, stratified_alpha
    items = {"interpersonal_warmth": [[("1", "2"), ("2", "2")], [("1", "1"), ("2", "1")], [("1", "1"), ("1", "2")]],
             "pitch_positivity": [[("4", "5"), ("5", "5")], [("4", "4"), ("5", "4")], [("4", "4"), ("4", "5")]]}
    out = pooled_summary(_stage_a_lines(items, 3))
    strata = {f"stage_a:{name}": [[float(h), float(p)] for pairs in per for p, h in pairs]
              for name, per in items.items()}
    cls = out["sampled"]["stage_a"]
    assert cls["alpha"] == pytest.approx(stratified_alpha(strata, "ordinal"))
    pooled_plain = krippendorff_alpha([u for us in strata.values() for u in us], "ordinal")
    assert pooled_plain > cls["alpha"] + 0.2                       # pooling the two items inflates alpha
    assert list(out["by_variable"]) == ["stage_a:interpersonal_warmth", "stage_a:pitch_positivity"]
    for name in items:
        row = out["by_variable"][f"stage_a:{name}"]
        assert row["alpha"] == pytest.approx(krippendorff_alpha(strata[f"stage_a:{name}"], "ordinal"))
        assert row["videos"] == 3 and row["n"] == 6 and "human_vs_primary" in row and "within_one" in row
        assert row["alpha_ci95"] is None and row["alpha_ci95_defined"] is None      # three videos: no interval
    assert out["by_reviewer"] == {} and pooled_summary(_stage_a_lines(items, 3), by_reviewer={"ana": _stage_a_lines(
        items, 3)})["by_reviewer"]["ana"]["by_variable"].keys() == out["by_variable"].keys()


def test_a_turn_code_class_of_a_nominal_and_an_ordinal_code_has_no_class_alpha_but_each_code_has_one():
    def lines(v):
        sev = [("1", "1"), ("2", "3"), ("3", "3"), ("1", "2")][v % 2:][:3]
        concept = [("x", "x"), ("y", "x"), ("y", "y")]
        return ([_line(f"turn:{i}/code:PR-SEV", "turn_code", p, h) for i, (p, h) in enumerate(sev, 1)]
                + [_line(f"turn:{i}/code:PR-CONCEPT", "turn_code", p, h) for i, (p, h) in enumerate(concept, 10)])
    out = pooled_summary({f"v{v}": lines(v) for v in range(6)})
    assert out["sampled"]["turn_code"]["alpha"] is None and out["sampled"]["turn_code"]["alpha_ci95"] is None
    assert out["sampled"]["turn_code"]["alpha_level"] == "mixed" and out["by_variable"]["PR-SEV"]["alpha_level"] == "ordinal"
    assert out["by_variable"]["PR-CONCEPT"]["alpha_level"] == "nominal"
    assert out["by_variable"]["PR-CONCEPT"]["alpha"] is not None
    assert out["by_variable"]["PR-SEV"]["alpha"] is not None
    assert out["by_variable"]["PR-SEV"]["within_one"]["n"] == 18 and "within_one" not in out["by_variable"]["PR-CONCEPT"]
    assert out["by_variable"]["PR-CONCEPT"]["ci_method"] == CLUSTER_BOOTSTRAP     # six videos


def test_pooled_human_human_reports_each_variable():
    def coder(values):
        return [_line(f"stage_a:{item}", "stage_a", None, v) for item, v in values]
    by = {"ana": {"v1": coder([("a", "2"), ("b", "5")]), "v2": coder([("a", "3"), ("b", "4")])},
          "ben": {"v1": coder([("a", "2"), ("b", "5")]), "v2": coder([("a", "2"), ("b", "4")])}}
    hh = pooled_summary({}, by_reviewer=by)["human_human"]
    assert sorted(hh["by_variable"]) == ["stage_a:a", "stage_a:b"]
    assert hh["by_variable"]["stage_a:b"]["agree"] == 1.0 and hh["by_variable"]["stage_a:a"]["agree"] == 0.5
    assert {"videos", "n", "agree", "ci95", "ci_method", "alpha", "alpha_ci95", "alpha_ci95_defined",
            "within_one"} <= set(hh["by_variable"]["stage_a:a"])


def test_a_decision_class_of_a_kind_and_a_time_has_no_class_alpha_but_each_variable_has_one():
    def lines(v):
        return [_line("decision:1/kind", "decision", "offer", "offer" if v % 2 else "pass"),
                _line("decision:2/kind", "decision", "pass", "pass"),
                _line("decision:3/kind", "decision", "offer", "offer"),
                _line("decision:1/time", "decision", "1:00", "1:05" if v % 2 else "2:30"),
                _line("decision:2/time", "decision", "2:10", "2:10"),
                _line("decision:3/time", "decision", "3:00", "3:40")]
    out = pooled_summary({f"v{v}": lines(v) for v in range(6)})
    assert out["sampled"]["decision"]["alpha"] is None
    assert out["by_variable"]["decision:kind"]["alpha"] is not None
    assert out["by_variable"]["decision:time"]["alpha"] is not None


# ---- what alpha reads: the level a line records, the units it uses, the passes counted once -----------

def test_a_pooled_figure_says_how_many_rows_its_alpha_uses():
    """Ten rated rows of which three have a side judged not rateable: all ten are rows and count in the share
    (a number against `missing:<reason>` disagrees), seven are the alpha's units."""
    def lines(v):
        rows = [_line(f"turn:{i}/code:PREP-SUBS", "turn_code", p, h, replicate=p)
                for i, (p, h) in enumerate([("1", "1"), ("2", "3"), ("3", "3"), ("4", "4"), ("5", "4")], 1)]
        if v == 0:      # the model, the human, and both: three rows with a side that is no rating
            rows[0] = _line("turn:1/code:PREP-SUBS", "turn_code", "missing:too_short", "2", replicate="2")
            rows[1] = _line("turn:2/code:PREP-SUBS", "turn_code", "3", "missing:too_short", replicate="3")
            rows[2] = _line("turn:3/code:PREP-SUBS", "turn_code", "missing:too_short", "missing:ambiguous",
                            replicate="3")
        return rows
    videos = {"v0": lines(0), "v1": lines(1)}
    out = pooled_summary(videos, by_reviewer={"ana": videos, "ben": videos})
    for figure in (out["sampled"]["turn_code"], out["by_variable"]["PREP-SUBS"]):
        assert figure["n"] == 10 and figure["alpha_units"] == 7 and figure["alpha_level"] == "ordinal"
        # the share counts every row: the 3 not-rateable rows disagree (two reasons are two values), and so do
        # 3 of the 7 rated ones
        assert figure["human_vs_primary"] == {"agree": 0.4, "n": 10}
    pair = out["human_human"]["by_variable"]["PREP-SUBS"]
    # the two coders gave the same answers: ten rows, of which two are `missing:` on both sides and no rating
    assert pair["n"] == 10 and pair["alpha_units"] == 8 and out["human_human"]["turn_code"]["alpha_units"] == 8
    # a class whose rows share no level has no alpha and no units
    mixed = pooled_summary({"v0": lines(0) + [_line("turn:9/code:PR-CONCEPT", "turn_code", "x", "x")]})
    assert mixed["sampled"]["turn_code"]["alpha_level"] == "mixed" and mixed["sampled"]["turn_code"]["alpha_units"] is None


def test_the_passes_are_compared_once_per_field_however_many_coders_judged_it():
    def coder(shift):
        return [_line(f"exchange:{i}/topic", "exchange_topic", p, [p, "z"][shift and i == 1], replicate=r)
                for i, (p, r) in enumerate([("a", "a"), ("b", "c"), ("c", "c")], 1)]
    by = {"ana": {"v1": coder(0), "v2": coder(0)}, "ben": {"v1": coder(1), "v2": coder(1)}}
    videos = {v: by["ana"][v] + by["ben"][v] for v in ("v1", "v2")}
    out = pooled_summary(videos, by_reviewer=by)
    topic = out["sampled"]["exchange_topic"]
    assert topic["n"] == 12 and topic["human_vs_replicate"]["n"] == 12        # one row per coder per field
    assert topic["primary_vs_replicate"] == {"agree": 4 / 6, "n": 6}          # six distinct (video, path) pairs
    assert out["by_variable"]["exchange_topic:topic"]["primary_vs_replicate"]["n"] == 6
    one = pooled_summary({v: by["ana"][v] for v in ("v1", "v2")})["sampled"]["exchange_topic"]
    assert one["primary_vs_replicate"] == topic["primary_vs_replicate"]
    assert one["alpha_primary_vs_replicate"] == topic["alpha_primary_vs_replicate"] is not None
    # the per-video report of a video with two coders counts its fields once too
    rows = [{**l, "reviewer": name} for name in by for l in by[name]["v1"]]
    corrections = [Correction(l["path"], l["new"], "correct", "", l["reviewer"], "coder") for l in rows]
    assert human_agreement(by["ana"]["v1"], corrections)["all"]["primary_vs_replicate"]["n"] == 3


def test_a_lines_level_decides_alpha_and_within_one_and_the_path_rule_is_only_the_fallback():
    def lines(v, level=...):
        values = [("1", "1"), ("2", "3"), ("3", "3"), ("1", "3")][v % 2:][:3]
        extra = {} if level is ... else {"level": level}
        return [{**_line(f"turn:{i}/code:QX-PRESS", "turn_code", p, h), **extra} for i, (p, h) in enumerate(values, 1)]
    # QX-PRESS is neither PR-SEV nor PREP-*: the path rule reads it as nominal, the recorded level as ordinal
    by_path = pooled_summary({f"v{v}": lines(v) for v in range(6)})["by_variable"]["QX-PRESS"]
    by_level = pooled_summary({f"v{v}": lines(v, "ordinal") for v in range(6)})["by_variable"]["QX-PRESS"]
    assert by_path["alpha_level"] == "nominal" and "within_one" not in by_path
    assert by_level["alpha_level"] == "ordinal" and by_level["within_one"]["n"] == 18
    assert by_level["alpha"] is not None and by_level["alpha"] != by_path["alpha"]
    assert by_level["human_vs_primary"] == by_path["human_vs_primary"]          # the share does not depend on it
    # a recorded level wins over the path rule in the other direction too, and None means no alpha
    sev = [{**_line(f"turn:{i}/code:PR-SEV", "turn_code", p, h), "level": "nominal"}
           for i, (p, h) in enumerate([("1", "1"), ("2", "3"), ("3", "3")], 1)]
    assert pooled_summary({"v": sev})["by_variable"]["PR-SEV"]["alpha_level"] == "nominal"
    text = [{**_line(f"exchange:{i}/label", "exchange_topic", "a", "b"), "level": None} for i in (1, 2)]
    label = pooled_summary({"v": text})["by_variable"]["exchange_topic:label"]
    assert label["alpha_level"] is None and label["alpha"] is None and label["alpha_units"] is None
    # coder against coder reads the same recorded level
    by = {"ana": {f"v{v}": lines(v, "ordinal") for v in range(2)}, "ben": {f"v{v}": lines(v + 1, "ordinal") for v in range(2)}}
    pair = pooled_summary({}, by_reviewer=by)["human_human"]["by_variable"]["QX-PRESS"]
    assert pair["alpha_level"] == "ordinal" and "within_one" in pair


def test_level_of_reads_the_codebooks_not_the_code_ids(books):
    from dataclasses import replace
    from pipeline.codebook import Codebook
    sev = books.panel.by_id("PR-SEV")
    concept = books.panel.by_id("PR-CONCEPT")
    renamed = replace(books, panel=Codebook(entries=[
        *books.panel.entries, sev.model_copy(update={"id": "QX-PRESS"}), concept.model_copy(update={"id": "PREP-KIND"})]))
    assert level_of("turn:4/code:QX-PRESS", renamed) == "ordinal"          # ordinal by its entry, whatever its id
    assert level_of("turn:4/code:PREP-KIND", renamed) == "nominal"         # a PREP-* id whose entry is categorical
    assert alpha_level("turn:4/code:PREP-KIND") == "ordinal"               # which the path rule gets wrong
    assert level_of("turn:4/code:PR-SEV", books) == "ordinal" and level_of("turn:4/code:PR-CONCEPT", books) == "nominal"
    assert level_of("turn:5/code:RS-RESP", books) == "nominal" and level_of("turn:5/code:PREP-SUBS", books) == "ordinal"
    assert level_of("turn:5/code:RS-EVID", books) == "nominal"                                     # binary
    assert level_of(f"stage_a:{books.stage_a.entries[0].id}", books) == "ordinal"
    assert level_of("venture:founder_gender", books) == "nominal" and level_of("pitch:any_offer", books) == "nominal"
    assert level_of("exchange:1/topic", books) == "nominal"
    assert level_of("turn:4/exchange", books) == "nominal" and level_of("turn:4/present", books) == "nominal"
    assert level_of("decision:2/time", books) == "interval" and level_of("decision:2/kind", books) == "nominal"
    assert level_of("prepared_window", books) is None
    for free_text in ("exchange:1/label", "actor:S1/label", "turn:4/other_concept"):
        assert level_of(free_text, books) is None


# ---- the pitch-level fields -------------------------------------------------------------------------

def test_cut_agrees_within_five_seconds(books):
    assert Tolerances().cut_s == 5.0
    assert value_kind("pitch:first_decision") == value_kind("decision:1/time") == "clock"
    assert value_kind("venture:ask") == "number"
    count = next(e.id for e in books.prepared_codes() if e.type == "count")
    # the count of checkable claims is answered and compared in its four ordered bands (Task 7c, 2026-10-21)
    assert value_kind(f"prep_pres:{count}", books) == "ordinal"
    assert agrees("pitch:first_decision", "1:51", "1:56") and not agrees("pitch:first_decision", "1:51", "1:57")
    assert not agrees("pitch:first_decision", "1:51", "1:57", Tolerances(decision_s=60.0))  # its own tolerance
    assert agrees("pitch:first_decision", "1:51", "1:58", Tolerances(cut_s=7.0))
    # the ask: the amount within 1%, the equity exactly
    assert agrees("venture:ask", "200000;10", "201000;10") and not agrees("venture:ask", "200000;10", "203000;10")
    assert not agrees("venture:ask", "200000;10", "200000;12") and agrees("venture:ask", "$200,000;10%", "200000; 10")
    assert agrees("venture:ask", "missing:not_observable", "missing:not_observable")
    assert not agrees("venture:ask", "missing:not_observable", "200000;10")

    rows = [sheet_row("pitch:first_decision", "1:51", "1:50", human_value="1:54", row="2")]
    corrections, problems = parse_sheet(rows, "ana")
    assert problems == [] and corrections == [Correction("pitch:first_decision", None, "confirm_primary", "", "ana")]
    corrections, problems = parse_sheet([{**rows[0], "human_value": "2:30"}], "ana")
    assert problems == [] and [(c.human_value, c.resolution) for c in corrections] == [("2:30", "outcome_override")]
    assert "not edited" in corrections[0].note
    _, problems = parse_sheet([{**rows[0], "human_value": "soon"}], "ana")
    assert problems == ["row 2: pitch:first_decision takes a clock time (M:SS), not 'soon'"]

    # judged, never written: the log carries old and new, the record keeps its decisions
    p = primary()
    reviewed, log = apply_corrections(p, [fix("pitch:first_decision", "2:30", "outcome_override")], books,
                                      duration_s=180.0)
    assert reviewed == p.model_dump(mode="json")
    assert [(e["path"], e["old"], e["new"], e["resolution"]) for e in log] == [
        ("pitch:first_decision", "1:51", "2:30", "outcome_override")]
    # a clock beyond a nine-minute video, or written as hours on it, is refused (Review Focus 2)
    for clock, reason in (("12:05", "beyond the video's duration"), ("1:02:30", "H:MM:SS")):
        with pytest.raises(ReviewError, match=reason):
            apply_corrections(p, [fix("pitch:first_decision", clock, "outcome_override")], books, duration_s=540.0)
    with pytest.raises(ReviewError, match="cannot be written"):
        apply_corrections(p, [fix("pitch:first_decision", "2:30", "correct")], books)


def test_the_pitch_paths_measure_the_outcome_and_the_cut(books):
    count = next(e.id for e in books.prepared_codes() if e.type == "count")
    rated = next(e.id for e in books.prepared_codes() if e.type == "ordinal")
    assert (variable_of("pitch:any_offer"), variable_of("pitch:first_decision")) == ("outcome", "cut")
    assert variable_of(f"prep_pres:{rated}") == f"prep_pres:{rated}" and variable_of("venture:ask") == "venture:ask"
    assert variable_of("pair:b01/pitch_positivity") == "pair:pitch_positivity"
    assert level_of("pitch:any_offer", books) == "nominal" and level_of("pitch:first_decision", books) == "interval"
    assert level_of(f"prep_pres:{rated}", books) == "ordinal" and level_of(f"prep_pres:{count}", books) == "ordinal"
    assert level_of("venture:ask", books) is None and level_of("pair:b01/pitch_positivity", books) == "nominal"
    assert alpha_level("venture:ask") is None


# ---- pooled figures of the cut, the ask and the pair slot --------------------------------------------

def _coded(path, cls, primary, new, level):
    return {**_line(path, cls, primary, new), "level": level}


def test_pooled_summary_gives_the_cut_same_turn_share_and_retains_timing_alpha():
    def coder(clock, ask):
        return [{**_coded("pitch:first_decision", "cut", "1:51", clock, "interval"),
                 "decision_turn_context": {"pass": "primary", "sha256": "test",
                                           "turns": [[1, 100, 120], [2, 120, 160]]}},
                _coded("venture:ask", "venture", "200000;10", ask, None)]
    # the primary at 1:51 in every video; ana within 5 s in four videos of five, ben within 5 s in all
    ana = {f"v{i}": coder(("1:53", "1:49", "1:56", "1:51", "2:30")[i], "201000;10") for i in range(5)}
    ben = {f"v{i}": coder(("1:52", "1:50", "1:54", "1:55", "1:47")[i], ("200000;10", "200000;12")[i % 2])
           for i in range(5)}
    videos = {v: ana[v] + ben[v] for v in ana}
    out = pooled_summary(videos, by_reviewer={"ana": ana, "ben": ben})
    cut = out["sampled"]["cut"]
    assert cut["share_within"] == pytest.approx(9 / 10) and cut["within_seconds"] is None
    assert cut["alpha_level"] == "interval" and cut["alpha"] is not None
    assert out["cut"] == {"share_within": pytest.approx(0.9), "n": 10, "alpha": cut["alpha"], "within_seconds": None}
    # coder against coder: within 5 s of each other in four videos of five (2:30 against 1:47 is not)
    hh = out["human_human"]["cut"]
    assert hh["share_within"] == pytest.approx(4 / 5) and hh["within_seconds"] is None and hh["alpha"] is not None
    # the ask is judged by its share: amount within 1% and equity exactly; it has no alpha
    ask = out["by_variable"]["venture:ask"]
    assert ask["human_vs_primary"]["agree"] == pytest.approx(8 / 10) and ask["alpha"] is None
    assert ask["alpha_units"] is None and "share_within" not in ask
    hh_ask = out["human_human"]["by_variable"]["venture:ask"]
    assert hh_ask["agree"] == pytest.approx(3 / 5) and hh_ask["alpha"] is None
    assert human_human({"ana": {l["path"]: l for l in ana["v0"]},
                        "ben": {l["path"]: l for l in ben["v0"]}})["by_field_class"]["cut"]["share_within"] == 1.0


def test_pooled_summary_has_an_empty_pair_slot_and_a_cut_entry_without_cut_rows():
    out = pooled_summary({"v1": [_line("exchange:1/topic", "exchange_topic", "sales", "sales")]})
    assert out["pairs"] == {}
    assert out["cut"] == {"share_within": None, "n": 0, "alpha": None, "within_seconds": 5.0}
    assert "share_within" not in out["sampled"]["exchange_topic"]


def test_the_outcome_comes_first_among_the_field_classes_of_a_video():
    rows = [{"path": "exchange:1/topic", "field_class": "exchange_topic", "primary": "sales", "replicate": None,
             "selected_as": "sampled"},
            {"path": "pitch:any_offer", "field_class": "outcome", "primary": "yes", "replicate": None,
             "selected_as": "sampled"}]
    corrections = [Correction("exchange:1/topic", None, "confirm_primary", "", "ana"),
                   Correction("pitch:any_offer", None, "confirm_primary", "", "ana")]
    report = human_agreement(rows, corrections)
    assert list(report["by_field_class"]) == ["outcome", "exchange_topic"]
    assert list(report["sampled_estimates"]) == list(report["by_field_class"])


# ---- plan D Task 3: pooled_summary["reporting"], the statistics of each kind of code --------------------------

@pytest.fixture(scope="module")
def reporting(books):
    from pipeline.stage_b.review_report import load_reporting
    return load_reporting(books)


def _resp_videos(note=lambda v, i: ""):
    """RS-RESP over ten videos of ten rows: 85 both `addresses`, 5 both `partial`, 5 + 5 crossed. 90% identical and
    90% of all answers `addresses`."""
    videos = {}
    for v in range(10):
        pairs = ([("addresses", "addresses")] * 8 + [("addresses", "partial"), ("partial", "addresses")] if v < 5
                 else [("addresses", "addresses")] * 9 + [("partial", "partial")])
        videos[f"v{v}"] = [{**_line(f"turn:{i}/code:RS-RESP", "turn_code", model, coder), "note": note(v, i)}
                           for i, (coder, model) in enumerate(pairs, 1)]
    return videos


def test_reporting_of_a_dominant_category_gives_a_low_alpha_a_high_ac1_and_the_prevalence_flag(reporting):
    import json
    from pipeline.reliability.agreement import gwet_ac1, krippendorff_alpha
    values = reporting.values_of("RS-RESP")
    out = pooled_summary(_resp_videos(), n_boot=199, reporting=reporting)
    row = out["reporting"]["RS-RESP"]
    assert row["kind"] == "categorical" and row["n"] == 100 and row["videos"] == 10
    assert row["coder_values"] == {"addresses": 90, "partial": 10, "deflects": 0, "NONE": 0}
    assert row["model_values"] == row["coder_values"]
    pairs = [(c, m) for v in _resp_videos().values() for c, m in ((l["new"], l["primary"]) for l in v)]
    f = row["figures"]
    assert set(f) == {"exact", "alpha", "ac1", "modal_share", "confusion"}
    assert f["exact"]["value"] == pytest.approx(0.9)
    assert f["alpha"]["value"] == pytest.approx(krippendorff_alpha([list(p) for p in pairs], "nominal"))
    assert f["alpha"]["value"] < 0.5 < 0.85 < f["ac1"]["value"] == pytest.approx(gwet_ac1(pairs, values))
    assert f["modal_share"]["value"] == pytest.approx(0.9) and f["modal_share"]["by_rater"] == [0.9, 0.9]
    assert row["flags"] == ["prevalence"]
    assert row["counts"]["addresses"]["partial"] == 5 and row["counts"]["partial"]["partial"] == 5
    assert f["confusion"]["value"] == row["counts"]
    # every share and chance-corrected statistic carries an interval resampling videos (ten videos)
    for name in ("exact", "alpha", "ac1", "modal_share"):
        assert f[name]["low"] <= f[name]["value"] <= f[name]["high"], name
    assert out["reporting"]["human_human"] == {} and out["reporting"]["pass_to_pass"] == {}     # no replicate
    json.dumps(out["reporting"], allow_nan=False)


def test_reporting_of_the_first_decision_is_in_seconds_and_has_no_alpha(reporting):
    """Times that span minutes and differ by 2-20 s: the 5 s share and the median seconds, no alpha."""
    from pipeline.reliability.agreement import clock_agreement
    model = ["1:00", "3:00", "5:00", "7:30", "9:10", "12:00"]
    coder = ["1:02", "2:56", "5:08", "7:42", "8:50", "12:03"]
    diffs = [2, -4, 8, 12, -20, 3]
    videos = {f"v{i}": [_line("pitch:first_decision", "cut", m, c)] for i, (m, c) in enumerate(zip(model, coder))}
    # a confirmation of the model's clock stands for the clock the coder typed
    videos["v0"] = [{**_line("pitch:first_decision", "cut", "1:00", None, "confirm_primary"), "typed": "1:02"}]
    out = pooled_summary(videos, n_boot=199, reporting=reporting)
    row = out["reporting"]["cut"]
    assert row["kind"] == "clock" and row["n"] == 6 and row["videos"] == 6
    expected = clock_agreement(diffs)
    f = row["figures"]
    assert set(f) == {"same_turn", "share_within_5", "share_within_30", "median_abs", "mean_signed", "loa_low", "loa_high"}
    assert f["share_within_5"]["value"] == pytest.approx(0.5) and f["median_abs"]["value"] == pytest.approx(6.0)
    assert f["mean_signed"]["value"] == pytest.approx(1 / 6)
    for name in ("share_within_30", "loa_low", "loa_high"):
        assert f[name]["value"] == pytest.approx(expected[name])
    assert f["share_within_5"]["low"] <= 0.5 <= f["share_within_5"]["high"]
    assert f["median_abs"]["low"] is None                 # an interval only for the shares
    assert row["counts"] == {"n": 6, "within_5": 3, "within_30": 6,
                             "same_turn_n": 0, "same_turn_agree": 0, "same_turn_undetermined": 6}
    assert f["same_turn"] == {"value": None, "low": None, "high": None}
    assert row["flags"] == ["small_n"]
    assert out["cut"]["alpha"] is not None                # the old key keeps its value


def test_reporting_of_an_ordinal_level_shift_prints_chance_within_one_the_mean_difference_and_rho(reporting):
    """One rater uses only 4-5 and the other 2-5: within-one equals its chance value, the coder sits one point higher
    and rho is 0."""
    combos = [(c, m) for c in ("4", "5") for m in ("2", "3", "4", "5")] * 3
    videos = {f"v{v}": [_line(f"stage_a:pitch_positivity", "stage_a", m, c)] for v, (c, m) in enumerate(combos)}
    out = pooled_summary(videos, n_boot=199, reporting=reporting)
    assert "stage_a:pitch_positivity" not in out["reporting"]
    row = out["reporting"]["pitch_positivity"]
    f = row["figures"]
    assert row["kind"] == "ordinal" and row["n"] == 24
    assert set(f) == {"exact", "chance_exact", "within_one", "chance_within_one", "alpha", "ac2", "modal_share",
                      "mean_difference", "spearman"}
    # the modal share behind a prevalence flag, with its interval and each side's (final review M5)
    assert f["modal_share"]["low"] is not None and len(f["modal_share"]["by_rater"]) == 2
    assert f["within_one"]["value"] == pytest.approx(5 / 8) == f["chance_within_one"]["value"]
    assert f["exact"]["value"] == pytest.approx(2 / 8) == f["chance_exact"]["value"]
    assert f["mean_difference"]["value"] == pytest.approx(1.0)
    assert f["spearman"]["value"] == pytest.approx(0.0, abs=1e-12)
    assert row["coder_values"] == {"1": 0, "2": 0, "3": 0, "4": 12, "5": 12}
    assert f["ac2"]["low"] is not None and f["alpha"]["low"] is not None and f["within_one"]["low"] is not None


def test_reporting_has_the_frame_of_the_question_concept_and_no_concept_row(reporting):
    concepts = [("customer_acquisition", "go_to_market"), ("execution", "execution"), ("competition", "assets"),
                ("deal_terms", "other"), ("NONE", "NONE")]
    videos = {f"v{v}": [_line(f"turn:{i}/code:PR-CONCEPT", "turn_code", m, c) for i, (c, m) in enumerate(concepts, 1)]
              for v in range(5)}
    out = pooled_summary(videos, n_boot=199, reporting=reporting)
    assert "PR-CONCEPT" not in out["reporting"] and "PR-CONCEPT" not in out["reporting"]["pass_to_pass"]
    assert "PR-CONCEPT" in out["by_variable"]                 # the old figures keep the concept
    row = out["reporting"]["PR-FRAME"]
    assert row["kind"] == "categorical" and row["n"] == 25
    # promotion/promotion, prevention/prevention, prevention/promotion, unclassified/unclassified, NEUTRAL/NEUTRAL
    assert row["figures"]["exact"]["value"] == pytest.approx(4 / 5)
    assert row["coder_values"] == {"promotion": 5, "prevention": 10, "unclassified": 5, "NEUTRAL": 5}
    assert row["counts"]["prevention"]["promotion"] == 5


def test_reporting_of_the_ask_is_its_share_under_the_asks_rule(reporting):
    asks = [("200000;10", "201000;10"), ("200000;10", "200000;12"), ("50000;5", "50000;5")] * 2
    videos = {f"v{v}": [_line("venture:ask", "venture", m, c)] for v, (m, c) in enumerate(asks)}
    row = pooled_summary(videos, n_boot=199, reporting=reporting)["reporting"]["venture:ask"]
    assert row["kind"] == "exact" and set(row["figures"]) == {"exact"}
    assert row["figures"]["exact"]["value"] == pytest.approx(4 / 6) and row["counts"] == {"agree": 4, "n": 6}


def test_reporting_of_a_comparison_is_k_of_n_with_its_exact_interval_p_and_gap_terciles(reporting):
    from pipeline.reliability.agreement import binomial_interval, binomial_p_vs_half
    from pipeline.stage_b import review_clips
    videos = [f"w{i}" for i in range(6)]
    lines = [{"reviewer": coder, "batch_id": f"b{i:02d}", "code": "pitch_positivity", "left": videos[i % 6],
              "right": videos[(i + 2) % 6], "value": "left" if (i < 8 or coder == "ben") else "right",
              "close": coder == "ana" and i == 0, "model": "left", "gap": 0.1 * i}
             for i in range(12) for coder in ("ana", "ben")]
    pairs = review_clips.pooled_pairs(review_clips.pair_agreement(lines, videos, n_boot=99))
    out = pooled_summary({}, pairs=pairs, reporting=reporting)
    assert out["pairs"] == pairs
    row = out["reporting"]["pair:pitch_positivity"]
    f = row["figures"]
    assert row["kind"] == "comparison" and row["n"] == 24 and row["counts"] == {"agree": 20, "n": 24, "close": 1}
    assert set(f) == {"agreement", "binomial_interval", "p_vs_half", "by_gap_tercile", "close"}
    low, high = binomial_interval(20, 24)
    # the leading interval resamples videos (pairs share clips); the exact binomial one sits beside it
    resampled = pairs["pitch_positivity"]["ci95"]["coder_model"]
    assert f["agreement"] == {"value": pytest.approx(20 / 24), "low": resampled[0], "high": resampled[1]}
    assert (f["binomial_interval"]["low"], f["binomial_interval"]["high"]) == (low, high)
    assert row["units"] == 12                                   # the twelve pairs, each judged by both coders
    assert f["p_vs_half"]["value"] == pytest.approx(binomial_p_vs_half(20, 24)) and f["p_vs_half"]["against"] == 0.5
    # thirds of the twelve pairs by the model's gap: ana answered against the model on the widest four
    assert [(t["tercile"], t["agree"], t["n"]) for t in f["by_gap_tercile"]["value"]] == [(1, 8, 8), (2, 8, 8),
                                                                                          (3, 4, 8)]
    # twelve pairs of clips are fewer than 20 units, though the two coders give 24 judgements (final review I1)
    assert f["close"]["value"] == 1 and row["flags"] == ["small_n"]
    hh = out["reporting"]["human_human"]["pair:pitch_positivity"]       # the coders chose alike on 8 of 12 pairs
    assert hh["n"] == 12 and hh["figures"]["agreement"]["value"] == pytest.approx(8 / 12)
    assert "pair:pitch_positivity" not in out["reporting"]["pass_to_pass"]


def test_reporting_without_unsure_rows_and_by_outcome_stratum(reporting):
    unsure = lambda v, i: "[unsure]" if v < 5 and i == 9 else ""      # five crossed rows are unsure
    videos = _resp_videos(unsure)
    strata = {"stratify": "any_offer", "of": {f"v{v}": "yes" if v < 5 else "no" for v in range(10)},
              "run": {"yes": 10, "no": 30}}
    out = pooled_summary(videos, n_boot=199, strata=strata, reporting=reporting)
    row = out["reporting"]["RS-RESP"]
    assert row["without_unsure"]["exact"] == pytest.approx(90 / 95) and row["without_unsure"]["n"] == 95
    assert row["strata"]["yes"]["exact"] == pytest.approx(0.8) and row["strata"]["yes"]["n"] == 50
    assert row["strata"]["no"]["exact"] == pytest.approx(1.0)
    # reweighted to the run's mix (a quarter yes): .25 x .8 + .75 x 1
    assert row["strata"]["reweighted"]["exact"] == pytest.approx(0.95)
    assert row["strata"]["reweighted"]["alpha"] is not None
    plain = pooled_summary(_resp_videos(), n_boot=199, reporting=reporting)["reporting"]["RS-RESP"]
    assert plain["strata"] == {} and plain["without_unsure"]["exact"] == plain["figures"]["exact"]["value"]


def test_reporting_of_two_coders_and_of_the_two_passes(reporting):
    def coder(shift):
        return {f"v{v}": [_line(f"turn:{i}/code:RS-EVID", "turn_code", "true" if i % 2 else "false",
                                ("true" if (i + shift * (i == 1)) % 2 else "false"), replicate="true")
                          for i in range(1, 5)] for v in range(6)}
    ana, ben = coder(0), coder(1)            # ben says "false" on turn 1 where ana and the model say "true"
    videos = {v: ana[v] + ben[v] for v in ana}
    out = pooled_summary(videos, n_boot=199, by_reviewer={"ana": ana, "ben": ben}, reporting=reporting)
    rep = out["reporting"]
    row = rep["RS-EVID"]
    assert row["kind"] == "binary" and row["n"] == 48
    assert row["units"] == 24                     # two coders judge each of the 24 question-answer pairs once
    assert set(row["figures"]) == {"counts", "exact", "modal_share", "positive_agreement", "negative_agreement", "ac1",
                                   "prevalence_index", "bias_index", "alpha"}
    # alpha (nominal) beside AC1, with an interval resampling the six videos, as a categorical row has it
    from pipeline.reliability.agreement import krippendorff_alpha
    alpha = row["figures"]["alpha"]
    assert alpha["value"] == pytest.approx(krippendorff_alpha(
        [[l["new"], l["primary"]] for v in videos.values() for l in v], "nominal"))
    assert alpha["low"] is not None and alpha["low"] <= alpha["value"] <= alpha["high"]
    assert "alpha" in row["without_unsure"] and rep["human_human"]["RS-EVID"]["figures"]["alpha"]["value"] is not None
    # coder against model: a both true 18, b coder true model false 0, c coder false model true 6, d both false 24
    assert row["counts"] == {"a": 18, "b": 0, "c": 6, "d": 24}
    assert row["figures"]["positive_agreement"]["value"] == pytest.approx(36 / 42)
    assert row["figures"]["bias_index"]["value"] == pytest.approx(6 / 48)
    assert row["figures"]["bias_index"]["direction"] == "model" and row["raters"] == ["coder", "model"]
    hh = rep["human_human"]["RS-EVID"]
    assert hh["n"] == 24 and hh["raters"] == ["ana", "ben"] and hh["counts"] == {"a": 6, "b": 6, "c": 0, "d": 12}
    p2p = rep["pass_to_pass"]["RS-EVID"]          # one pair of passes per field, however many coders judged it
    assert p2p["n"] == 24 and p2p["raters"] == ["primary", "replicate"]
    assert p2p["counts"] == {"a": 12, "b": 0, "c": 12, "d": 0}


def test_reporting_leaves_the_old_keys_unchanged_and_is_empty_without_a_configuration(reporting):
    import json
    videos = _resp_videos()
    plain = pooled_summary(videos, n_boot=199)
    assert plain["reporting"] == {}
    full = pooled_summary(videos, n_boot=199, reporting=reporting)
    assert {k: v for k, v in full.items() if k != "reporting"} == {k: v for k, v in plain.items() if k != "reporting"}
    # one unit: nothing chance-corrected is defined, and no NaN is written
    one = pooled_summary({"v0": [_line("stage_a:pitch_positivity", "stage_a", "3", "4")]}, reporting=reporting)
    row = one["reporting"]["pitch_positivity"]
    assert row["figures"]["alpha"]["value"] is None and row["figures"]["spearman"]["value"] is None
    assert row["flags"] == ["small_n"]
    json.dumps(one["reporting"], allow_nan=False)
    assert pooled_summary({}, reporting=reporting)["reporting"] == {"human_human": {}, "pass_to_pass": {}}


def test_a_comparison_against_another_level_than_a_coin_flip_is_refused(reporting):
    """The figure is p_vs_half: a configuration testing against another level cannot be written under that name."""
    from dataclasses import replace
    pairs = {"pitch_positivity": {"coder_model": 0.75, "n": 8, "videos": 4, "coder_coder": None,
                                  "by_gap_tercile": [], "n_close": 0}}
    with pytest.raises(ValueError, match="p_vs_half"):
        pooled_summary({}, pairs=pairs, reporting=replace(reporting, comparisons={"against": 0.6}))
    assert pooled_summary({}, pairs=pairs, reporting=reporting)["reporting"]["pair:pitch_positivity"]["n"] == 8


def test_reporting_drops_and_counts_off_scale_pairs_and_reads_bands_and_float_text(reporting):
    """`missing:<reason>`, an empty value and a concept with no frame are off the scale: dropped from the row and
    counted in n_off_scale. "4.0" is the scale's "4"; the model's PRES-EVID count is read in the coder's bands."""
    subs = [("missing:too_short", "3"), ("2", "missing:ambiguous"), ("", "4"), ("4.0", "4"), ("5", "4.0")]
    evid = [("4", "3-5"), ("7", "6+"), ("0", "1-2"), ("2", "1\u20132"), ("missing:not_shown", "3-5")]
    videos = {f"v{v}": [_line("prep_pres:PRES-EVID", "prep_pres", *evid[v]),
                        _line(f"turn:1/code:PREP-SUBS", "turn_code", *subs[v]),
                        _line("turn:2/code:PR-CONCEPT", "turn_code", "execution", "execution" if v else "bogus")]
              for v in range(5)}
    rep = pooled_summary(videos, n_boot=99, reporting=reporting)["reporting"]
    sub = rep["PREP-SUBS"]
    assert (sub["n"], sub["n_off_scale"]) == (2, 3)
    assert sub["coder_values"] == {"1": 0, "2": 0, "3": 0, "4": 2, "5": 0}
    assert sub["model_values"] == {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1}
    assert sub["figures"]["exact"]["value"] == pytest.approx(0.5)
    ev = rep["PRES-EVID"]
    assert (ev["n"], ev["n_off_scale"]) == (4, 1) and ev["figures"]["exact"]["value"] == pytest.approx(0.75)
    assert ev["model_values"] == {"0": 1, "1-2": 1, "3-5": 1, "6+": 1}
    assert ev["coder_values"] == {"0": 0, "1-2": 2, "3-5": 1, "6+": 1}
    assert ev["counts"]["1-2"]["0"] == 1
    frame = rep["PR-FRAME"]
    assert (frame["n"], frame["n_off_scale"]) == (4, 1) and frame["coder_values"]["prevention"] == 4
