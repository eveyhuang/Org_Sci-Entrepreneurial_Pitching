import copy

import pytest

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.review_items import (
    CUT_PATH, FIELD_CLASSES, OUTCOME_PATH, SHEET_CLASSES, ReviewItem, allowed_values, any_offer, canonical_full,
    field_class, first_decision_clock, flatten, is_cut, is_outcome, locator, locator_span, main_path,
    normalize_path, parse_path, presence_path, presence_ref, replicate_only_path, set_decision, set_function, set_value,
)
from pipeline.stage_b.review_items import read_value
from tests.panel_fixtures import set_other, synthetic_record


def get_value(record: PanelRecord, path: str) -> str | None:
    """The record's value at a path, as the record writes it (see canonical_full for the codebook's spelling)."""
    return read_value(record.model_dump(mode="json"), path)


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def record(mutate=None) -> PanelRecord:
    d = synthetic_record()
    if mutate:
        mutate(d)
    return PanelRecord.model_validate(d)


def by_path(items):
    return {i.path: i for i in items}


# ---- flatten ------------------------------------------------------------------------------

def test_flatten_yields_one_item_per_path(books):
    items = flatten(record(), books)
    paths = [i.path for i in items]
    assert len(paths) == len(set(paths))
    assert all(isinstance(i, ReviewItem) and i.field_class in FIELD_CLASSES for i in items)
    # 4 actors x 2; 19 turns x 3 plus 4 questions x (2 codes + other_concept) and 4 answers x (7 codes +
    # other_concept); 4 exchanges x 3 (no investor list or closing since v3-dev9); 4 decisions x 4; 3 Stage A;
    # 4 prepared presentation codes; 4 venture fields and the ask; 1 window
    assert len(items) == 8 + (57 + 12 + 32) + 12 + 16 + 3 + 4 + 5 + 1
    assert not [p for p in paths if "investor" in p and p.startswith("exchange:")]


def test_flatten_classes(books):
    items = by_path(flatten(record(), books))
    expected = {
        "actor:S1/label": "actor", "actor:S1/role": "actor",
        "turn:2/function": "turn_function", "turn:2/exchange": "turn_exchange", "turn:3/responds_to": "turn_link",
        "turn:2/code:PR-CONCEPT": "turn_code", "turn:2/other_concept": "turn_code",
        "exchange:1/kind": "exchange_kind", "exchange:1/topic": "exchange_topic", "exchange:1/label": "exchange_topic",
        "decision:2/kind": "decision", "decision:2/actor": "decision", "decision:2/time": "decision",
        "decision:2/offer_id": "decision", "stage_a:interpersonal_warmth": "stage_a",
        "venture:primary_industry": "venture", "venture:product_form": "venture", "venture:customer_type": "venture",
        "venture:founder_gender": "venture", "prepared_window": "prepared_window",
    }
    for path, cls in expected.items():
        assert items[path].field_class == cls, path
        assert field_class(path) == cls
    # the first-decision clock and the pair judgements are sheet-only
    assert {i.field_class for i in items.values()} == set(FIELD_CLASSES) - {"cut", "pair"}
    assert "exchange_investor" not in FIELD_CLASSES and "closing" not in FIELD_CLASSES


def test_flatten_values_and_clocks(books):
    items = by_path(flatten(record(), books))
    assert items["actor:S1/label"].value == "Investor A"
    assert items["actor:S1/role"].value == "investor"
    assert (items["actor:S1/label"].clock_start, items["actor:S1/label"].clock_end) == ("1:02", None)
    t = items["turn:3/function"]
    assert (t.value, t.clock_start, t.clock_end) == ("answer", "1:06", "1:20")
    assert items["turn:3/exchange"].value == "1"
    assert items["turn:10/exchange"].value is None
    assert items["turn:7/responds_to"].value == "6"
    assert items["turn:1/responds_to"].value is None
    assert items["turn:3/code:PREP-SUBS"].value == "3"
    assert items["turn:3/code:PREP-CTX"].value == "missing:not_applicable"
    assert items["turn:3/code:RS-EVID"].value == "true"
    assert items["turn:2/other_concept"].value is None
    assert "turn:1/other_concept" not in items          # a presents turn has no concept
    e = items["exchange:1/kind"]
    assert (e.value, e.clock_start, e.clock_end) == ("substantive", "1:02", "2:34")
    assert items["exchange:1/topic"].value == "operating_efficiency"
    assert items["exchange:1/label"].value == "customer acquisition cost"
    assert items["exchange:2/label"].value is None
    assert "exchange:1/investors" not in items and "exchange:1/investor:S2/closing" not in items
    d = items["decision:2/time"]
    assert (d.value, d.clock_start) == ("2:39", "2:39")
    assert items["decision:2/offer_id"].value == "O1"
    assert items["decision:1/offer_id"].value is None
    assert items["decision:3/kind"].value == "joins_offer"
    assert items["decision:3/actor"].value == "S2"
    s = items["stage_a:interpersonal_warmth"]
    assert (s.value, s.clock_start, s.clock_end) == ("3", "0:05", "1:00")
    assert items["venture:founder_gender"].value == "all_men"
    assert items["prepared_window"].value == "0:05-1:00"


def test_flatten_context_is_one_line_naming_who_and_what(books):
    items = by_path(flatten(record(), books))
    assert items["turn:4/function"].context == ("turn 4, S2 (Investor B): Disputes the paid-channel assumption "
                                                "behind that figure.")
    assert items["exchange:1/kind"].context == "exchange 1, turns 2-7, 14-16: substantive, customer acquisition cost"
    assert items["decision:2/kind"].context == "decision 2, turn 17: S1 (Investor A) offer, 200k for 20%"
    assert all("\n" not in i.context and i.context for i in items.values())


def test_locator_names_the_unit_without_any_judged_value(books):
    r = record()
    assert locator(r, "turn:4/code:PR-SEV") == "turn 4, S2 (Investor B)"
    assert locator(r, "exchange:1/topic") == "exchange 1, turns 2-7, 14-16"
    assert locator(r, "decision:2/kind") == "decision 2, turn 17"
    assert locator(r, "stage_a:interpersonal_warmth") == "prepared presentation"
    assert locator(r, "venture:product_form") == "the venture"
    # where to look, without a clock that is itself the judged value
    assert locator_span(r, "turn:4/code:PR-SEV") == ("1:20", "1:26")
    assert locator_span(r, "decision:2/time") == ("2:38", "2:43")          # the span of turn 17, not 2:39
    assert locator_span(r, "prepared_window") == (None, None)
    assert locator_span(record(lambda d: d["decisions"][1].update(turn_id=None)), "decision:2/kind") == (None, None)


def test_flatten_lists_a_required_code_the_turn_lacks_and_a_stray_one_it_carries(books):
    def mutate(d):
        d["turns"][1]["codes"] = [c for c in d["turns"][1]["codes"] if c["code"] != "PR-SEV"]   # turn 2, a question
        d["turns"][6]["codes"] = [{"code": "PR-SEV", "value": "2", "missing": "none"}]          # turn 7, accepts
    items = by_path(flatten(record(mutate), books))
    assert items["turn:2/code:PR-SEV"].value is None
    assert items["turn:7/code:PR-SEV"].value == "2"


def test_flatten_reads_labels_and_values_in_the_codebook_spelling(books):
    def mutate(d):
        d["turns"][1]["function"] = "Question"
        d["turns"][1]["codes"][0]["value"] = " Operating_Efficiency "
        d["turns"][2]["codes"][2]["value"] = "TRUE"
        d["exchanges"][0]["kind"] = "SUBSTANTIVE"
        d["venture"]["product_form"] = "Physical_Good"
    r = record(mutate)
    items = by_path(flatten(r, books))
    assert items["turn:2/function"].value == "question"
    assert items["turn:2/code:PR-CONCEPT"].value == "operating_efficiency"
    assert items["turn:3/code:RS-EVID"].value == "true"
    assert items["exchange:1/kind"].value == "substantive"
    assert items["venture:product_form"].value == "physical_good"
    assert get_value(r, "turn:2/function") == "Question"          # get_value reads the record as written
    assert get_value(canonical_full(r, books), "turn:2/function") == "question"


def test_get_value_equals_the_item_value_for_every_item(books):
    r = record()
    for item in flatten(r, books):
        assert get_value(r, item.path) == item.value, item.path


@pytest.mark.parametrize("path", ["turn:99/function", "nothing", "turn:2/colour", "exchange:1/investor:S3/closing",
                                  "stage_a:charisma", "venture:description", "actor:Z9/role", "decision:9/kind",
                                  "turn:x/function", "exchange:1/investor:S1", "prepared_window/x"])
def test_get_value_of_an_unknown_path_raises(path):
    with pytest.raises(ValueError):
        get_value(record(), path)


# ---- set_value: round trips ---------------------------------------------------------------

def test_set_value_round_trips_every_item_to_the_same_record(books):
    r = record()
    d = r.model_dump(mode="json")
    before = copy.deepcopy(d)
    for item in flatten(r, books):
        set_value(d, item.path, item.value, books)
    assert d == before


CHANGES = [
    ("actor:S1/label", "Kevin", "Kevin"), ("actor:H1/role", "other", "other"),
    ("turn:13/function", "Challenges", "challenges"),
    ("turn:10/exchange", "2", "2"), ("turn:9/exchange", "", None),
    ("turn:7/responds_to", "6, 4", "6,4"), ("turn:7/responds_to", "", None),
    ("turn:2/code:PR-CONCEPT", "Customer_Acquisition", "customer_acquisition"),
    ("turn:2/code:PR-SEV", "3", "3"), ("turn:3/code:RS-EVID", "FALSE", "false"),
    ("turn:3/code:PREP-CTX", "4", "4"), ("turn:3/code:PREP-SUBS", "missing:ambiguous", "missing:ambiguous"),
    ("turn:2/other_concept", "unit economics", "unit economics"),
    ("exchange:3/kind", "demonstration", "demonstration"), ("exchange:3/topic", "SALES", "sales"),
    ("exchange:3/label", "reorders", "reorders"), ("exchange:2/label", "lids", "lids"),
    ("decision:1/kind", "re_entry", "re_entry"), ("decision:2/actor", "S2", "S2"),
    ("decision:2/time", "2:41", "2:41"), ("decision:1/offer_id", "O2", "O2"), ("decision:2/offer_id", "", None),
    ("stage_a:interpersonal_warmth", "5", "5"), ("stage_a:interpersonal_warmth", "", None),
    ("venture:primary_industry", "pets", "pets"), ("venture:product_form", "service", "service"),
    ("venture:customer_type", "Both", "both"), ("venture:founder_gender", "mixed", "mixed"),
    ("prepared_window", "0:15-1:43; 2:00-2:10", "0:15-1:43;2:00-2:10"), ("prepared_window", "", None),
]


@pytest.mark.parametrize("path,written,read", CHANGES)
def test_set_value_writes_each_field_class(books, path, written, read):
    r = record(lambda d: d["actors"].append({"actor_id": "H1", "role": "host", "label": "Narrator"}))
    d = r.model_dump(mode="json")
    set_value(d, path, written, books)
    after = PanelRecord.model_validate(d)        # still a record
    assert get_value(after, path) == read


def test_set_value_changes_only_the_addressed_field(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    set_value(d, "turn:4/code:PR-SEV", "2", books)
    assert d["turns"][3]["codes"][1] == {"code": "PR-SEV", "value": "2", "missing": "none"}
    d["turns"][3]["codes"][1]["value"] = "3"
    assert d == before


def test_stage_a_empty_means_uncodable_and_a_number_means_rated(books):
    d = record().model_dump(mode="json")
    set_value(d, "stage_a:interpersonal_warmth", None, books)
    item = next(i for i in d["stage_a"] if i["code"] == "interpersonal_warmth")
    assert (item["value"], item["uncodable"]) == (None, True)
    set_value(d, "stage_a:interpersonal_warmth", "4", books)
    assert (item_of(d, "interpersonal_warmth")["value"], item_of(d, "interpersonal_warmth")["uncodable"]) == (4, False)


def item_of(d, code):
    return next(i for i in d["stage_a"] if i["code"] == code)


# ---- decisions: roles, offer ids, clocks ----------------------------------------------------

def test_a_decisions_actor_must_have_the_role_its_kind_needs(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    with pytest.raises(ValueError, match="offer needs an investor, and F1 is a founder"):
        set_value(d, "decision:2/actor", "F1", books)
    with pytest.raises(ValueError, match="founder_accepts needs a founder, and S1 is an investor"):
        set_value(d, "decision:4/actor", "S1", books)
    with pytest.raises(ValueError, match="founder_rejects needs a founder, and S1 is an investor"):
        set_value(d, "decision:2/kind", "founder_rejects", books)
    with pytest.raises(ValueError, match="exit needs an investor, and F1 is a founder"):
        set_value(d, "decision:4/kind", "exit", books)
    assert d == before
    set_value(d, "decision:2/actor", "s2", books)
    set_value(d, "decision:2/kind", "joins_offer", books)
    assert (d["decisions"][1]["actor"], d["decisions"][1]["kind"]) == ("S2", "joins_offer")


def test_a_kind_that_needs_an_offer_id_is_refused_without_one_naming_what_is_needed(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    with pytest.raises(ValueError, match=r"decision:1/kind: offer needs an offer id.*decision:1/offer_id"):
        set_value(d, "decision:1/kind", "offer", books)                 # S3's exit carries none
    with pytest.raises(ValueError, match="founder_counters needs an offer id"):
        set_decision(d, 1, {"kind": "founder_counters", "actor": "F1"}, books)
    assert d == before
    set_value(d, "decision:2/kind", "revises_offer", books)             # it has one: fine
    # kind, actor and offer id of one decision change together when no single step would be valid
    set_decision(d, 1, {"kind": "offer", "offer_id": "O2"}, books)
    assert (d["decisions"][0]["kind"], d["decisions"][0]["offer_id"]) == ("offer", "O2")
    set_decision(d, 4, {"kind": "exit", "actor": "S3", "offer_id": ""}, books)
    assert (d["decisions"][3]["kind"], d["decisions"][3]["actor"], d["decisions"][3]["offer_id"]) == ("exit", "S3", None)
    with pytest.raises(ValueError, match="needs a founder"):
        set_decision(d, 4, {"kind": "founder_accepts", "offer_id": "O1"}, books)
    with pytest.raises(ValueError, match="no such path"):
        set_decision(d, 99, {"kind": "exit"}, books)
    with pytest.raises(ValueError, match="not a field of a decision"):
        set_decision(d, 1, {"terms": "x"}, books)


def test_clock_values_must_lie_inside_the_video_and_not_be_hours_on_a_short_one(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    for path, value, reason in (
            ("decision:1/time", "3:30", r"3:30 is beyond the video's duration \(3:00\)"),
            ("decision:1/time", "0:01:51", "H:MM:SS"),                 # 1:51 retyped by a spreadsheet as a time of day
            ("decision:1/time", "1:51:00", "H:MM:SS"),
            ("prepared_window", "0:05-3:30", "beyond the video's duration"),
            ("prepared_window", "0:00:05-0:01:00", "H:MM:SS")):
        with pytest.raises(ValueError, match=reason):
            set_value(d, path, value, books, duration_s=180.0)
    assert d == before
    set_value(d, "decision:1/time", "3:00", books, duration_s=180.0)       # the last second is inside
    set_value(d, "prepared_window", "0:05-1:00;1:10-2:59", books, duration_s=180.0)
    assert d["decisions"][0]["time_s"] == 180.0 and len(d["prepared_spans"]) == 2
    # a video of an hour or more has such times; without a duration nothing is checked
    set_value(d, "decision:1/time", "1:00:05", books, duration_s=7200.0)
    set_value(d, "decision:1/time", "0:01:51", books)
    assert d["decisions"][0]["time_s"] == 111.0


def test_path_ids_are_normalised_so_one_field_has_one_path():
    assert normalize_path("turn:0005/function") == "turn:5/function" == normalize_path(" turn:5/function ")
    assert normalize_path("turn:05/code:PR-SEV") == "turn:5/code:PR-SEV"
    assert normalize_path("decision:004/present") == "decision:4/present"
    for path in ("stage_a:interpersonal_warmth", "venture:founder_gender", "prepared_window", "actor:S1/role",
                 "pitch:any_offer", "prep_pres:PRES-EVID", "venture:ask", "replicate_only:turn@1:05", "nonsense",
                 "turn:x/function", ""):
        assert normalize_path(path) == path


def test_a_stage_a_value_changed_in_review_says_so_in_its_note(books):
    d = record().model_dump(mode="json")
    item = d["stage_a"][2]
    assert (item["value"], item["note"]) == (3, "Steady delivery.")
    set_value(d, "stage_a:interpersonal_warmth", "3", books)                   # the same value: the note still fits
    assert d["stage_a"][2]["note"] == "Steady delivery."
    set_value(d, "stage_a:interpersonal_warmth", "5", books)
    assert d["stage_a"][2]["note"] == "Steady delivery. [value changed in review]"
    set_value(d, "stage_a:interpersonal_warmth", "", books)                    # changed again: the mark is not repeated
    assert d["stage_a"][2]["note"] == "Steady delivery. [value changed in review]"
    d["stage_a"][0]["note"] = ""
    set_value(d, "stage_a:pitch_positivity", "1", books)
    assert d["stage_a"][0]["note"] == "[value changed in review]"


def test_a_required_code_the_turn_lacks_can_be_supplied(books):
    def mutate(d):
        d["turns"][1]["codes"] = d["turns"][1]["codes"][:1]
    d = record(mutate).model_dump(mode="json")
    set_value(d, "turn:2/code:PR-SEV", "2", books)
    assert d["turns"][1]["codes"][1] == {"code": "PR-SEV", "value": "2", "missing": "none"}


# ---- set_value: refusals ------------------------------------------------------------------

INVALID = [
    ("turn:2/function", "interrogates", "not one of"),                 # unknown value
    ("turn:2/function", "", "needs one of"),
    ("exchange:1/kind", "NONE", "not one of"),
    ("exchange:1/topic", "pricing", "not one of"),
    ("decision:1/kind", "NONE", "not one of"),
    ("venture:primary_industry", "aerospace", "not one of"),
    ("actor:S1/role", "judge", "not one of"),
    ("turn:2/code:PR-CONCEPT", "pricing", "not one of"),
    ("turn:2/code:PR-SEV", "4", "outside 1-3"),                        # out of scale
    ("turn:2/code:PR-SEV", "high", "not an integer"),
    ("turn:3/code:RS-EVID", "maybe", "not true or false"),
    ("turn:3/code:RS-EVID", "", "missing:<reason>"),
    ("turn:3/code:RS-EVID", "missing:bored", "not a missing reason"),
    ("turn:3/code:RS-EVID", "missing:none", "not a missing reason"),
    ("stage_a:interpersonal_warmth", "6", "outside 1-5"),
    ("stage_a:interpersonal_warmth", "loud", "not an integer"),
    ("turn:2/exchange", "9", "not a recorded exchange"),               # unknown exchange
    ("turn:2/exchange", "one", "not an exchange id"),
    ("turn:3/responds_to", "5", "not an earlier turn"),
    ("turn:3/responds_to", "44", "not an earlier turn"),
    ("exchange:1/investors", "S2", "no such path"),                     # no investor list since v3-dev9
    ("exchange:1/investor:S1/closing", "satisfied", "no such path"),
    ("exchange:1/investor:S2/closing_turn", "16", "no such path"),
    ("decision:1/actor", "S9", "not an actor"),
    ("decision:2/time", "2.41", "not a clock time"),                   # bad clock
    ("prepared_window", "0:15", "not a clock range"),
    ("prepared_window", "1:43-0:15", "ends before it starts"),
    ("prepared_window", "0:15-1:4", "not a clock time"),
    ("actor:S1/label", "", "needs a label"),
    ("exchange:1/label", "x" * 81, "80"),
    ("turn:99/function", "question", "no such path"),                  # unknown path
    ("turn:7/code:PR-SEV", "2", "no such path"),
    ("turn:2/code:XX-NOPE", "2", "no such path"),
    ("exchange:1/investor:S3/closing", "satisfied", "no such path"),
    ("venture:description", "x", "no such path"),
    ("stage_a:charisma", "3", "no such path"),
    ("bogus", "x", "no such path"),
]


@pytest.mark.parametrize("path,value,reason", INVALID)
def test_set_value_refuses_with_a_reason_and_leaves_the_record_alone(books, path, value, reason):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    with pytest.raises(ValueError) as exc:
        set_value(d, path, value, books)
    assert reason in str(exc.value)
    assert path.split("/")[0].split(":")[0] in str(exc.value)      # the message names the path
    assert d == before


def test_function_change_without_its_codes_raises_naming_them(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    with pytest.raises(ValueError) as exc:
        set_value(d, "turn:13/function", "question", books)     # accepts -> question
    assert "PR-CONCEPT" in str(exc.value) and "PR-SEV" in str(exc.value)
    with pytest.raises(ValueError) as exc:
        set_value(d, "turn:2/function", "answer", books)        # question -> answer needs the seven answer codes
    assert "RS-RESP" in str(exc.value) and "PREP-CTX" in str(exc.value)
    assert d == before


def test_set_function_with_the_codes_supplied_adds_them_and_drops_the_ones_that_no_longer_apply(books):
    d = record().model_dump(mode="json")
    removed = set_function(d, "turn:13/function", "question", {"PR-CONCEPT": "sales", "PR-SEV": "missing:ambiguous"},
                           books)
    assert removed == []
    assert d["turns"][12]["function"] == "question"
    assert d["turns"][12]["codes"] == [{"code": "PR-CONCEPT", "value": "sales", "missing": "none"},
                                       {"code": "PR-SEV", "value": None, "missing": "ambiguous"}]
    removed = set_function(d, "turn:2/function", "challenges", {}, books)        # question -> challenges
    assert removed == ["PR-CONCEPT", "PR-SEV"]
    assert d["turns"][1]["function"] == "challenges" and d["turns"][1]["codes"] == []
    PanelRecord.model_validate(d)


def test_set_function_refuses_a_code_the_new_function_does_not_take_and_a_partial_set(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    with pytest.raises(ValueError, match="RS-RESP"):
        set_function(d, "turn:13/function", "question", {"PR-CONCEPT": "sales", "PR-SEV": "1", "RS-RESP": "partial"},
                     books)
    with pytest.raises(ValueError, match="PR-SEV"):
        set_function(d, "turn:13/function", "question", {"PR-CONCEPT": "sales"}, books)
    with pytest.raises(ValueError, match="outside 1-3"):
        set_function(d, "turn:13/function", "question", {"PR-CONCEPT": "sales", "PR-SEV": "9"}, books)
    assert d == before


def test_set_value_of_the_same_function_keeps_the_codes(books):
    d = record().model_dump(mode="json")
    before = copy.deepcopy(d)
    set_value(d, "turn:2/function", "QUESTION", books)
    assert d == before


# ---- allowed values -----------------------------------------------------------------------

def test_allowed_values(books):
    r = record()
    assert allowed_values("turn:2/function", r, books) == books.turn_functions()
    assert allowed_values("turn:2/exchange", r, books) == ("1", "2", "3", "4")
    assert allowed_values("turn:2/code:PR-SEV", r, books) == ("1", "2", "3")
    assert allowed_values("turn:3/code:RS-EVID", r, books) == ("true", "false")
    assert "other" in allowed_values("turn:2/code:PR-CONCEPT", r, books)
    assert allowed_values("exchange:1/kind", r, books) == books.exchange_kinds()
    assert allowed_values("exchange:1/topic", r, books) == books.exchange_topics()
    with pytest.raises(ValueError):
        allowed_values("exchange:1/investor:S1/closing", r, books)       # no closing state since v3-dev9
    assert allowed_values("decision:1/kind", r, books) == books.decision_kinds()
    assert allowed_values("decision:1/actor", r, books) == ("F1", "S1", "S2", "S3")
    assert allowed_values("actor:S1/role", r, books) == ("founder", "investor", "host", "other")
    assert allowed_values("stage_a:interpersonal_warmth", r, books) == ("1", "2", "3", "4", "5")
    assert allowed_values("venture:founder_gender", r, books) == books.venture_values("VC-GENDER")
    for free in ("actor:S1/label", "turn:3/responds_to", "turn:2/other_concept", "exchange:1/label",
                 "decision:1/time", "decision:1/offer_id", "prepared_window"):
        assert allowed_values(free, r, books) is None, free
    with pytest.raises(ValueError):
        allowed_values("turn:2/colour", r, books)


def test_other_concept_item_appears_for_a_turn_that_names_one(books):
    def mutate(d):
        set_other(d["turns"][1], "founder salary")
        d["turns"][6]["other_concept"] = "stray"        # an accepts turn that carries a name anyway
    items = by_path(flatten(record(mutate), books))
    assert items["turn:2/code:PR-CONCEPT"].value == "other"
    assert items["turn:2/other_concept"].value == "founder salary"
    assert items["turn:7/other_concept"].value == "stray"


# ---- unit presence ------------------------------------------------------------------------

def test_unit_presence_paths(books):
    assert SHEET_CLASSES == ("unit_presence", "outcome", *FIELD_CLASSES)
    assert presence_path("turn", 13) == "turn:13/present" and main_path("exchange", 3) == "exchange:3/kind"
    assert replicate_only_path("decision", "1:51") == "replicate_only:decision@1:51"
    assert replicate_only_path("exchange", None) == "replicate_only:exchange@unplaced"
    here, there = presence_ref("turn:13/present"), presence_ref("replicate_only:decision@1:51")
    assert (here.unit, here.unit_id, here.replicate_only, here.primary_value, here.replicate_value) == (
        "turn", 13, False, "present", "absent")
    assert (there.unit, there.unit_id, there.replicate_only, there.primary_value, there.replicate_value) == (
        "decision", None, True, "absent", "present")
    assert presence_ref("replicate_only:turn@2:10#2").unit == "turn"
    for path in ("turn:13/function", "actor:S1/present", "stage_a:present", "replicate_only:actor@1:00", "", None):
        assert presence_ref(path) is None, path
    assert field_class("turn:13/present") == field_class("replicate_only:turn@2:10") == "unit_presence"

    r = record()
    assert allowed_values("turn:13/present", r, books) == ("present", "absent")
    assert allowed_values("replicate_only:exchange@1:02", r, books) == ("present", "absent")
    assert locator(r, "turn:13/present") == "turn 13, S2 (Investor B)"
    assert locator(r, "exchange:3/present") == "exchange 3, turns 11-13"
    assert locator_span(r, "decision:2/present") == ("2:38", "2:43")       # by its turn, not by its judged time
    with pytest.raises(ValueError, match="turn 99 is not in the record"):
        allowed_values("turn:99/present", r, books)
    with pytest.raises(ValueError, match="only the replicate"):
        locator(r, "replicate_only:turn@2:10")
    d = r.model_dump(mode="json")
    with pytest.raises(ValueError, match="no such path"):                  # a presence is judged, never written
        set_value(d, "turn:13/present", "absent", books)
    assert d == r.model_dump(mode="json")


# ---- the pitch-level fields (v3-dev9): the outcome, the first decision, the prepared ratings, the ask ----------

def test_flatten_lists_prep_pres_and_ask_paths(books):
    items = by_path(flatten(record(), books))
    prepared = [p for p in items if p.startswith("prep_pres:")]
    assert prepared == [f"prep_pres:{e.id}" for e in books.prepared_codes()]        # 4, in book order
    assert [items[p].value for p in prepared] == ["3", "4", "2", "5"]
    assert {items[p].field_class for p in prepared} == {"prep_pres"}
    evid = items[prepared[-1]]
    assert (evid.clock_start, evid.clock_end) == ("0:05", "1:00")                    # the prepared presentation
    assert evid.context.startswith("prepared presentation")
    ask = items["venture:ask"]
    assert (ask.field_class, ask.value, ask.context.split(":")[0]) == ("venture", "200000;10", "the venture")
    assert not [p for p in items if p.startswith("pitch:")]              # the pitch paths are sheet-only

    def missing(d):
        d["venture"].update(ask_amount_usd=None, ask_equity_pct=None, ask_missing="not_observable")
        d["prepared"][3] = {"code": d["prepared"][3]["code"], "value": None, "missing": "edited_gap"}
    items = by_path(flatten(record(missing), books))
    assert items["venture:ask"].value == "missing:not_observable"
    assert items[prepared[-1]].value == "missing:edited_gap"
    assert by_path(flatten(record(lambda d: d["venture"].update(ask_equity_pct=12.5)), books))[
        "venture:ask"].value == "200000;12.5"


def test_pitch_paths_parse_and_are_never_written(books):
    assert (OUTCOME_PATH, CUT_PATH) == ("pitch:any_offer", "pitch:first_decision")
    assert field_class(OUTCOME_PATH) == parse_path(OUTCOME_PATH).field_class == "outcome"
    assert field_class(CUT_PATH) == parse_path(CUT_PATH).field_class == "cut"
    assert is_outcome(" pitch:any_offer ") and not is_outcome(CUT_PATH) and not is_outcome(None)
    assert is_cut(CUT_PATH) and not is_cut(OUTCOME_PATH) and not is_cut("decision:1/time")
    assert normalize_path(" pitch:any_offer ") == OUTCOME_PATH
    for path in ("pitch:closing", "pitch:any_offer/x", "pitch:", "pitch/any_offer", "investor:S1/ever_offer"):
        with pytest.raises(ValueError, match="no such path"):
            parse_path(path)
    r = record()
    assert allowed_values(OUTCOME_PATH, r, books) == ("yes", "no", "unknown")
    assert allowed_values(CUT_PATH, r, books) is None                     # a clock
    assert (get_value(r, OUTCOME_PATH), get_value(r, CUT_PATH)) == ("yes", "1:51")
    for path in (OUTCOME_PATH, CUT_PATH):
        assert locator(r, path) == "the pitch"
        assert locator_span(r, path) == (None, None)                        # when the panel decides is judged
        d = r.model_dump(mode="json")
        with pytest.raises(ValueError, match="no such path"):              # judged from the footage, never written
            set_value(d, path, "no" if path == OUTCOME_PATH else "1:40", books)
        assert d == r.model_dump(mode="json")


def test_any_offer_and_first_decision_from_the_fixture(books):
    d = record().model_dump(mode="json")
    assert (any_offer(d), first_decision_clock(d)) == ("yes", "1:51")      # S3 exits at 1:51, S1 offers later

    def exits(*actors):
        return [{"decision_id": i, "actor": a, "kind": "exit", "time_s": 100.0 + i, "turn_id": 10, "offer_id": None,
                 "terms": "", "note": ""} for i, a in enumerate(actors, 1)]
    assert any_offer({**d, "decisions": exits("S1", "S2", "S3")}) == "no"
    assert first_decision_clock({**d, "decisions": exits("S2", "S1", "S3")}) == "1:41"
    assert any_offer({**d, "decisions": exits("S1", "S2")}) == "unknown"     # S3 is never seen deciding
    # an exit and then a re-entry without an offer is still no offer (the rule of analysis.panel_dataset)
    back = exits("S1", "S2", "S3") + [{**exits("S3")[0], "decision_id": 4, "kind": "re_entry", "time_s": 150.0}]
    assert any_offer({**d, "decisions": back}) == "no"
    # a revision or withdrawal shows that an offer was made
    assert any_offer({**d, "decisions": [{**d["decisions"][1], "kind": "withdraws_offer"}]}) == "yes"
    assert (any_offer({**d, "decisions": []}), first_decision_clock({**d, "decisions": []})) == ("unknown", None)
    # a founder's move is no decision of the panel
    founder = [{**d["decisions"][3], "time_s": 30.0}] + d["decisions"][:3]
    assert first_decision_clock({**d, "decisions": founder}) == "1:51"


def test_prep_pres_count_accepts_whole_numbers_only(books):
    count = next(e.id for e in books.prepared_codes() if e.type == "count")
    rated = next(e.id for e in books.prepared_codes() if e.type == "ordinal")
    r = record()
    assert allowed_values(f"prep_pres:{count}", r, books) is None              # a number
    assert allowed_values(f"prep_pres:{rated}", r, books) == ("1", "2", "3", "4", "5")
    d = r.model_dump(mode="json")
    for written, read in ((" 07 ", "7"), ("0", "0"), ("missing:edited_gap", "missing:edited_gap")):
        set_value(d, f"prep_pres:{count}", written, books)
        assert get_value(PanelRecord.model_validate(d), f"prep_pres:{count}") == read
    set_value(d, f"prep_pres:{rated}", "5", books)
    assert get_value(PanelRecord.model_validate(d), f"prep_pres:{rated}") == "5"
    before = copy.deepcopy(d)
    for path, value, reason in ((f"prep_pres:{count}", "-1", "not a whole number"),
                                (f"prep_pres:{count}", "2.5", "not a whole number"),
                                (f"prep_pres:{count}", "many", "not a whole number"),
                                (f"prep_pres:{count}", "", "needs a value"),
                                (f"prep_pres:{rated}", "6", "outside 1-5"),
                                ("prep_pres:PR-SEV", "2", "no such path"),
                                ("prep_pres:XX-NOPE", "2", "no such path")):
        with pytest.raises(ValueError, match=reason) as exc:
            set_value(d, path, value, books)
        assert "prep_pres:" in str(exc.value)
        assert d == before


def test_ask_values_are_normalised_and_junk_is_refused(books):
    d = record().model_dump(mode="json")
    for written, fields in (("$200,000;10%", (200000, 10.0, "none")), ("150000; 12.5", (150000, 12.5, "none")),
                            ("missing:not_observable", (None, None, "not_observable"))):
        set_value(d, "venture:ask", written, books)
        v = PanelRecord.model_validate(d).venture
        assert (v.ask_amount_usd, v.ask_equity_pct, v.ask_missing) == fields, written
    before = copy.deepcopy(d)
    for junk in ("two hundred", "200000", "200000;", "200000;ten", "200000;1.2.3", "", "missing:none",
                 "200000;101"):
        with pytest.raises(ValueError, match="venture:ask") as exc:
            set_value(d, "venture:ask", junk, books)
        assert ("is not amount;equity" in str(exc.value) or "not a missing reason" in str(exc.value)
                or "outside 0-100" in str(exc.value)), junk
        assert d == before


def _decisions(*moves):
    """Decision dicts of (actor, kind, offer_id, turn_id, time_s) moves."""
    return [{"decision_id": i, "actor": a, "kind": k, "time_s": t, "turn_id": turn, "offer_id": o, "terms": "",
             "note": ""} for i, (a, k, o, turn, t) in enumerate(moves, 1)]


OUTCOME_CASES = {
    "exit, re-entry without an offer id, joined offer": _decisions(
        ("S3", "exit", None, 10, 111.0), ("S3", "re_entry", None, 17, 158.0), ("S1", "offer", "O1", 17, 159.0),
        ("S2", "joins_offer", "O1", 18, 164.0)),
    "all exit, one re-enters without an offer id": _decisions(
        ("S3", "exit", None, 10, 111.0), ("S1", "exit", None, 17, 159.0), ("S2", "exit", None, 18, 164.0),
        ("S3", "re_entry", None, 18, 165.0)),
    "one investor never deciding": _decisions(("S3", "exit", None, 10, 111.0), ("S1", "exit", None, 17, 159.0)),
    "re-entry with an offer id": _decisions(
        ("S3", "exit", None, 10, 111.0), ("S1", "exit", None, 17, 159.0), ("S2", "exit", None, 18, 164.0),
        ("S3", "re_entry", "O9", 18, 165.0)),
    "withdraw only": _decisions(("S1", "withdraws_offer", "O1", 17, 159.0)),
    "joins only": _decisions(("S2", "joins_offer", "O1", 18, 164.0)),
    "the fixture": None,
    "no decisions": [],
}


@pytest.mark.parametrize("case", list(OUTCOME_CASES))
def test_any_offer_agrees_with_the_analysis_pitch_outcome(tmp_path, case):
    """review_items.any_offer is the model side a human verdict on pitch:any_offer is compared with; it must stay
    the variable analysis.panel_dataset.pitch_outcome analyses."""
    from analysis.panel_dataset import load_run, pitch_outcome
    from tests.panel_analysis_fixtures import write_video
    d = synthetic_record()
    if OUTCOME_CASES[case] is not None:
        d["decisions"] = OUTCOME_CASES[case]
    write_video(tmp_path / "run", "vid", d)
    analysed = pitch_outcome(load_run(tmp_path / "run")).any_offer.iloc[0]
    expected = "unknown" if analysed != analysed else {1.0: "yes", 0.0: "no"}[analysed]
    assert any_offer(record(lambda r: r.update(decisions=d["decisions"])).model_dump(mode="json")) == expected


def test_an_empty_ask_says_to_give_a_missing_reason(books):
    d = record().model_dump(mode="json")
    with pytest.raises(ValueError) as exc:
        set_value(d, "venture:ask", "  ", books)
    assert str(exc.value) == "venture:ask: an empty value is not amount;equity (use missing:<reason>)"
