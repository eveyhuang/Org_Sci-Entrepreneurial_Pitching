import pytest
from pydantic import ValidationError

from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.stage_b_panel import CodeValue, Exchange, PanelRecord, StageAItem, Turn
from tests.panel_fixtures import synthetic_record


def test_synthetic_record_round_trips():
    record = PanelRecord.model_validate(synthetic_record())
    assert PanelRecord.model_validate_json(record.model_dump_json()) == record
    assert len(record.turns) == 19 and len(record.exchanges) == 4
    assert len(record.decisions) == 4 and len(record.stage_a) == 3


def test_fixture_tells_the_required_story():
    record = PanelRecord.model_validate(synthetic_record())
    turns = {t.turn_id: t for t in record.turns}
    assert [t.function for t in record.turns] == [
        "presents", "question", "answer", "question", "answer", "repair", "accepts",
        "social", "social", "decides", "question", "answer", "accepts",
        "question", "answer", "challenges", "decides", "decides", "decides"]
    assert turns[1].exchange_id is None and turns[10].exchange_id is None
    members = {e.exchange_id: [t.turn_id for t in record.turns if t.exchange_id == e.exchange_id]
               for e in record.exchanges}
    assert members == {1: [2, 3, 4, 5, 6, 7, 14, 15, 16], 2: [8, 9], 3: [11, 12, 13], 4: [17, 18, 19]}
    one, two, three, four = record.exchanges
    assert (one.kind, one.topic) == ("substantive", "operating_efficiency")
    assert (two.kind, two.topic) == ("social", "NONE")
    assert (three.kind, three.topic) == ("substantive", "customer_retention")
    assert four.kind == "negotiation"
    severity = {c.code: c.value for c in turns[4].codes}
    assert severity["PR-SEV"] == "3" and turns[4].speaker == "S2"
    assert [(d.kind, d.turn_id, d.offer_id) for d in record.decisions] == [
        ("exit", 10, None), ("offer", 17, "O1"), ("joins_offer", 18, "O1"), ("founder_accepts", 19, "O1")]


def test_code_value_is_null_exactly_when_missing():
    assert CodeValue(code="PREP-CTX", value=None, missing="not_applicable").value is None
    for value, missing in [(None, "none"), ("3", "too_short")]:
        with pytest.raises(ValidationError):
            CodeValue(code="PREP-SUBS", value=value, missing=missing)


def test_there_is_no_events_list_or_link_basis():
    import pipeline.schemas.stage_b_panel as schema
    assert "events" not in PanelRecord.model_fields and "events" not in schema.PanelWire.model_fields
    assert not hasattr(schema, "Event") and not hasattr(schema, "LinkBasis")
    with pytest.raises(ValidationError):
        Turn.model_validate({**synthetic_record()["turns"][1], "link_basis": "none"})


def test_turn_fields_and_required_but_nullable_exchange():
    base = synthetic_record()["turns"][1]
    assert list(Turn.model_fields) == ["turn_id", "speaker", "start_s", "end_s", "function", "addressed",
                                       "exchange_id", "responds_to", "codes", "other_concept", "summary"]
    assert Turn.model_validate({**base, "exchange_id": None}).exchange_id is None
    assert base["other_concept"] is None
    assert Turn.model_validate({**base, "other_concept": "founder's own investment"}).other_concept
    with pytest.raises(ValidationError):
        Turn.model_validate({**base, "other_concept": "x" * 81})
    assert Turn.model_validate({**base, "addressed": []}).addressed == []
    for field in ("function", "addressed", "exchange_id", "responds_to", "codes", "other_concept", "summary"):
        with pytest.raises(ValidationError):
            Turn.model_validate({k: v for k, v in base.items() if k != field})
    for bad in ({"summary": "x" * 241}, {"summary": ""}, {"exchange_id": 0}, {"turn_id": 0}):
        with pytest.raises(ValidationError):
            Turn.model_validate({**base, **bad})
    wire = gemini_response_schema(PanelRecord)["properties"]["turns"]["items"]
    assert "exchange_id" in wire["required"] and wire["properties"]["exchange_id"]["nullable"] is True


def test_exchange_fields_and_no_investor_list():
    base = synthetic_record()["exchanges"][0]
    assert list(Exchange.model_fields) == ["exchange_id", "kind", "topic", "label"]
    assert Exchange.model_validate({**base, "label": ""}).label == ""
    with pytest.raises(ValidationError):
        Exchange.model_validate({**base, "label": "x" * 81})
    with pytest.raises(ValidationError):
        Exchange.model_validate({**base, "exchange_id": 0})
    with pytest.raises(ValidationError, match="investors"):
        Exchange(exchange_id=1, kind="social", topic="NONE", label="", investors=[])
    wire = gemini_response_schema(PanelRecord)["properties"]["exchanges"]["items"]
    assert "investors" not in wire["properties"]


def test_a_v3_dev8_record_with_per_investor_closings_is_rejected():
    old = synthetic_record()
    for e in old["exchanges"]:
        e["investors"] = [{"actor": "S1", "closing": "NONE", "closing_turn": None}]
    old["exchanges"][0]["investors"] = [{"actor": "S1", "closing": "satisfied", "closing_turn": 7},
                                        {"actor": "S2", "closing": "unsatisfied", "closing_turn": 16}]
    with pytest.raises(ValidationError, match="investors"):
        PanelRecord.model_validate(old)


def test_stage_a_item_value_xor_uncodable_and_bounds():
    assert StageAItem(code="x", value=None, uncodable=True, moment_s=None, note="").value is None
    for value, uncodable in [(None, False), (3, True), (6, False)]:
        with pytest.raises(ValidationError):
            StageAItem(code="x", value=value, uncodable=uncodable, moment_s=None, note="")


def test_stage_a_moment_is_required_but_nullable():
    assert StageAItem(code="x", value=3, uncodable=False, moment_s=20.0, note="").moment_s == 20.0
    assert StageAItem(code="x", value=3, uncodable=False, moment_s=None, note="").moment_s is None
    with pytest.raises(ValidationError):
        StageAItem(code="x", value=3, uncodable=False, note="")
    with pytest.raises(ValidationError):
        StageAItem(code="x", value=3, uncodable=False, moment_s=-1.0, note="")
    with pytest.raises(ValidationError):
        StageAItem(code="x", value=3, uncodable=False, moment_s=None, evidence_s=[20.0], note="")
    wire = gemini_response_schema(PanelRecord)["properties"]["stage_a"]["items"]
    assert "moment_s" in wire["required"] and wire["properties"]["moment_s"]["nullable"] is True


def test_unknown_fields_are_rejected():
    data = synthetic_record(); data["prediction"] = "offer"
    with pytest.raises(ValidationError):
        PanelRecord.model_validate(data)


def test_wire_schema_is_provider_compatible():
    from pipeline.schemas.stage_b_panel import PanelWire
    wire = gemini_response_schema(PanelWire)
    text = str(wire)
    for banned in ("$ref", "additionalProperties", "anyOf", "pattern"):
        assert banned not in text, banned
    assert wire["properties"]["turns"]["items"]["properties"]["codes"]["items"]["properties"]["missing"]["enum"]


def test_record_field_order():
    assert list(PanelRecord.model_fields) == [
        "actors", "prepared_spans", "excluded_spans", "gaps", "source_complete", "stage_a", "prepared", "venture",
        "turns", "exchanges", "decisions", "claim_conflicts", "notes"]
    assert list(gemini_response_schema(PanelRecord)["properties"]) == list(PanelRecord.model_fields)


def test_decision_turn_and_offer_are_required_but_nullable():
    from pipeline.schemas.stage_b_panel import Decision
    base = synthetic_record()["decisions"][0]
    assert Decision.model_validate({**base, "turn_id": None, "offer_id": None}).turn_id is None
    for field in ("turn_id", "offer_id"):
        with pytest.raises(ValidationError):
            Decision.model_validate({k: v for k, v in base.items() if k != field})
    wire = gemini_response_schema(PanelRecord)["properties"]["decisions"]["items"]
    assert "turn_id" in wire["required"] and "offer_id" in wire["required"]
    assert wire["properties"]["turn_id"]["nullable"] is True and wire["properties"]["offer_id"]["nullable"] is True


def test_claim_conflict_names_turns_and_a_reconciling_turn_only_when_reconciled():
    from pipeline.schemas.stage_b_panel import ClaimConflict
    base = {"turn_a": 1, "turn_b": 3, "metric": "cac", "note": "", "reconciled": True, "reconciled_turn": 6}
    assert list(ClaimConflict.model_fields) == list(base)
    assert ClaimConflict.model_validate(base).reconciled_turn == 6
    assert ClaimConflict.model_validate({**base, "reconciled_turn": None}).reconciled_turn is None
    assert ClaimConflict.model_validate({**base, "reconciled": False, "reconciled_turn": None}).reconciled is False
    with pytest.raises(ValidationError):
        ClaimConflict.model_validate({**base, "reconciled": False})
    with pytest.raises(ValidationError):
        ClaimConflict.model_validate({**base, "turn_a": 0})          # the prepared presentation is a turn now
    with pytest.raises(ValidationError):
        ClaimConflict.model_validate({k: v for k, v in base.items() if k != "reconciled_turn"})
    with pytest.raises(ValidationError):
        ClaimConflict.model_validate({**base, "time_a_s": 20.0})
    wire = gemini_response_schema(PanelRecord)["properties"]["claim_conflicts"]["items"]
    assert "reconciled_turn" in wire["required"] and wire["properties"]["reconciled_turn"]["nullable"] is True


def test_canonical_value_and_venture():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import canonical_value, load_panel_books
    books = load_panel_books(load_release())
    frame, sev, evid = (books.panel.by_id(c) for c in ("PR-CONCEPT", "PR-SEV", "RS-EVID"))
    assert canonical_value(frame, "none") == "NONE"
    assert canonical_value(frame, "Sales") == "sales"
    assert canonical_value(frame, "NONE") == "NONE"
    assert canonical_value(frame, "sideways") == "sideways"
    assert canonical_value(frame, None) is None
    assert canonical_value(evid, "True") == "true" and canonical_value(evid, "FALSE") == "false"
    assert canonical_value(evid, "yes") == "yes"
    assert canonical_value(sev, " 2 ") == "2"
    assert canonical_value(sev, None) is None
    assert books.canonical_venture("VC-INDUSTRY", "none") == "NONE"
    assert books.canonical_venture("VC-FORM", "Physical_Good") == "physical_good"
    assert books.canonical_venture("VC-FORM", "thing") == "thing"
    assert books.canonical_venture("VC-FORM", None) is None


# ---- clock times on the wire --------------------------------------------------------------

def test_clock_to_seconds_accepts_clock_times():
    from pipeline.schemas.stage_b_panel import clock_to_seconds
    for text, seconds in (("0:07", 7.0), ("1:48", 108.0), ("12:05", 725.0), ("1:02:03", 3723.0), (" 1:48 ", 108.0),
                          ("1:48.5", 108.5), ("0:00", 0.0), ("75:10", 4510.0)):
        assert clock_to_seconds(text) == seconds, text


@pytest.mark.parametrize("text", ["148", "1:75", "1:2:3:4", "", "abc", "1:60", "1:75:03", "a:05", "1:b", "-1:05",
                                  "1:", ":05", "1:-5", "1:4.", "1::05", "1.5:05", "1:48s", "1:5", "1:5.5", "0:7", "1:048",
                                  "1:02:3"])
def test_clock_to_seconds_rejects_everything_else(text):
    from pipeline.schemas.stage_b_panel import clock_to_seconds
    with pytest.raises(ValueError):
        clock_to_seconds(text)


def test_seconds_to_clock():
    from pipeline.schemas.stage_b_panel import clock_to_seconds, seconds_to_clock
    for seconds, text in ((7, "0:07"), (108.0, "1:48"), (725, "12:05"), (3723, "1:02:03"), (0, "0:00"),
                          (59.6, "1:00"), (108.4, "1:48"), (3599.7, "1:00:00")):
        assert seconds_to_clock(seconds) == text
    for seconds in (0, 7, 59, 60, 61, 599, 3600, 3661):
        assert clock_to_seconds(seconds_to_clock(seconds)) == float(seconds)


def test_wire_round_trips_to_the_normalized_record():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record
    from tests.panel_fixtures import synthetic_wire, to_wire
    assert wire_to_record(PanelWire.model_validate(synthetic_wire())) == PanelRecord.model_validate(synthetic_record())
    rich = synthetic_record()
    rich["gaps"] = [{"at_s": 90.0, "note": "cut"}]
    rich["claim_conflicts"] = [{"turn_a": 1, "turn_b": 3, "metric": "cac", "note": "", "reconciled": True,
                                "reconciled_turn": 6},
                               {"turn_a": 3, "turn_b": 5, "metric": "cac", "note": "", "reconciled": False,
                                "reconciled_turn": None}]
    rich["venture"]["earliest_s"] = None
    assert wire_to_record(PanelWire.model_validate(to_wire(rich))) == PanelRecord.model_validate(rich)


def test_wire_model_mirrors_the_record_with_string_times():
    from pipeline.schemas.stage_b_panel import PanelWire
    assert list(PanelWire.model_fields) == list(PanelRecord.model_fields)
    wire = gemini_response_schema(PanelWire)["properties"]
    turn = wire["turns"]["items"]["properties"]
    assert list(turn) == ["turn_id", "speaker", "start", "end", "function", "addressed", "exchange_id",
                          "responds_to", "codes", "other_concept", "summary"]
    other = wire["turns"]["items"]["properties"]["other_concept"]
    assert other["type"] == "string" and other["nullable"] is True
    assert "other_concept" in wire["turns"]["items"]["required"]
    assert wire["turns"]["items"]["properties"]["exchange_id"]["nullable"] is True
    assert wire["exchanges"] == gemini_response_schema(PanelRecord)["properties"]["exchanges"]
    assert wire["claim_conflicts"] == gemini_response_schema(PanelRecord)["properties"]["claim_conflicts"]
    assert turn["start"]["type"] == "string" and "clock time M:SS" in turn["start"]["description"]
    assert "e.g. 1:48" in turn["start"]["description"]
    assert wire["decisions"]["items"]["properties"]["time"]["type"] == "string"
    assert wire["gaps"]["items"]["properties"]["at"]["type"] == "string"
    moment = wire["stage_a"]["items"]["properties"]["moment"]
    assert moment["type"] == "string" and moment["nullable"] is True
    assert "moment" in wire["stage_a"]["items"]["required"]
    assert "evidence" not in wire["stage_a"]["items"]["properties"]
    assert wire["venture"]["properties"]["earliest"]["nullable"] is True
    text = str(gemini_response_schema(PanelWire))
    assert "pattern" not in text and "_s'" not in text
    for name in ("turn_id", "offer_id"):
        assert name in wire["decisions"]["items"]["required"]


@pytest.mark.parametrize("path, mutate", [
    ("turns[2].start", lambda d: d["turns"][2].update(start="148")),
    ("turns[0].end", lambda d: d["turns"][0].update(end="1:75")),
    ("prepared_spans[0].start", lambda d: d["prepared_spans"][0].update(start="five")),
    ("excluded_spans[0].end", lambda d: d["excluded_spans"][0].update(end="5")),
    ("decisions[1].time", lambda d: d["decisions"][1].update(time="129")),
    ("stage_a[1].moment", lambda d: d["stage_a"][1].update(moment="20")),
    ("turns[4].end", lambda d: d["turns"][4].update(end="1:5")),
    ("venture.earliest", lambda d: d["venture"].update(earliest="8.0")),
])
def test_wire_to_record_names_the_malformed_time(path, mutate):
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record
    from tests.panel_fixtures import synthetic_wire
    data = synthetic_wire()
    mutate(data)
    bad = eval("data" + "".join(f"[{p!r}]" if not p.isdigit() else f"[{p}]"
                                for p in path.replace("]", "").replace("[", ".").split(".")))
    with pytest.raises(ValueError) as excinfo:
        wire_to_record(PanelWire.model_validate(data))
    assert path in str(excinfo.value) and repr(bad) in str(excinfo.value)


def test_wire_keeps_the_record_invariants():
    from pipeline.schemas.stage_b_panel import PanelWire
    from tests.panel_fixtures import synthetic_wire
    data = synthetic_wire()
    data["stage_a"][0]["value"] = None            # rated item with no value
    with pytest.raises(ValidationError):
        PanelWire.model_validate(data)
    data = synthetic_wire()
    data["claim_conflicts"] = [{"turn_a": 1, "turn_b": 3, "metric": "m", "note": "", "reconciled": False,
                                "reconciled_turn": 6}]
    with pytest.raises(ValidationError):
        PanelWire.model_validate(data)
    data = synthetic_wire()
    data["prediction"] = "offer"
    with pytest.raises(ValidationError):
        PanelWire.model_validate(data)


def test_canonical_record_fixes_the_case_of_structure_labels_only():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import load_panel_books
    books = load_panel_books(load_release())
    clean = PanelRecord.model_validate(synthetic_record())
    assert books.canonical_record(clean) == clean
    data = synthetic_record()
    data["turns"][1]["function"] = "Question"
    data["turns"][7]["function"] = "none"
    data["turns"][2]["function"] = "replies"                 # not a value: left as written
    data["turns"][2]["codes"][1]["value"] = "neutral"        # code values are not touched here
    data["exchanges"][0].update(kind="Substantive", topic="Operating_Efficiency")
    data["exchanges"][1].update(topic="none")
    data["decisions"][0]["kind"] = "EXIT"
    out = books.canonical_record(PanelRecord.model_validate(data))
    assert [t.function for t in out.turns[:3]] == ["presents", "question", "replies"] and out.turns[7].function == "NONE"
    assert out.turns[2].codes[1].value == "neutral"
    assert (out.exchanges[0].kind, out.exchanges[0].topic) == ("substantive", "operating_efficiency")
    assert out.exchanges[1].topic == "NONE"
    assert out.decisions[0].kind == "exit"
    as_dict = books.canonical_structure(data)
    assert as_dict == out.model_dump(mode="json") and data["decisions"][0]["kind"] == "EXIT"


# ---- v3-dev6: wire tolerance ----------------------------------------------------------------

def test_wire_accepts_a_value_with_a_missing_reason_and_the_record_does_not():
    from pipeline.schemas.stage_b_panel import PanelWire, WireCodeValue, WireStageAItem
    assert WireCodeValue(code="PREP-COH", value="4", missing="not_applicable").value == "4"
    assert WireCodeValue(code="PREP-COH", value=None, missing="too_short").value is None
    assert WireCodeValue(code="PREP-COH", value="4", missing="none").missing == "none"
    with pytest.raises(ValidationError):
        WireCodeValue(code="PREP-COH", value=None, missing="none")
    with pytest.raises(ValidationError):
        CodeValue(code="PREP-COH", value="4", missing="not_applicable")
    assert WireStageAItem(code="x", value=4, uncodable=True, moment=None, note="").value == 4
    with pytest.raises(ValidationError):
        WireStageAItem(code="x", value=None, uncodable=False, moment=None, note="")
    with pytest.raises(ValidationError):
        StageAItem(code="x", value=4, uncodable=True, moment_s=None, note="")
    wire = gemini_response_schema(PanelWire)["properties"]
    assert wire["turns"]["items"]["properties"]["codes"] == \
        gemini_response_schema(PanelRecord)["properties"]["turns"]["items"]["properties"]["codes"]


def test_wire_to_record_normalizes_and_reports_a_value_given_with_a_missing_reason():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record, wire_to_record_with_notes
    from tests.panel_fixtures import synthetic_wire
    clean = PanelWire.model_validate(synthetic_wire())
    record, notes = wire_to_record_with_notes(clean)
    assert notes == [] and record == wire_to_record(clean) == PanelRecord.model_validate(synthetic_record())
    data = synthetic_wire()
    data["turns"][2]["codes"][5].update(value="4", missing="not_applicable")
    data["turns"][14]["codes"][4].update(missing="too_short")
    data["stage_a"][2].update(value=4, uncodable=True)
    record, notes = wire_to_record_with_notes(PanelWire.model_validate(data))
    assert notes == ["stage_a[2] interpersonal_warmth: value 4 dropped because uncodable is true",
                     "turns[2].codes PREP-COH: value '4' dropped because missing is 'not_applicable'",
                     "turns[14].codes PREP-SUBS: value '3' dropped because missing is 'too_short'"]
    expected = synthetic_record()
    expected["turns"][2]["codes"][5].update(value=None, missing="not_applicable")
    expected["turns"][14]["codes"][4].update(value=None, missing="too_short")
    expected["stage_a"][2].update(value=None, uncodable=True)
    assert record == PanelRecord.model_validate(expected)
    assert wire_to_record(PanelWire.model_validate(data)) == record


# ---- v3-dev7: the frame is derived from the concept --------------------------------------

def test_concept_tuples_are_exactly_the_codebook_values():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import PREVENTION_CONCEPTS, PROMOTION_CONCEPTS, load_panel_books
    books = load_panel_books(load_release())
    assert PROMOTION_CONCEPTS == ("customer_acquisition", "go_to_market", "market_opportunity", "assets",
                                  "self_promotion", "sales", "usage", "vision", "forecast")
    assert PREVENTION_CONCEPTS == ("execution", "safety_security", "liability", "competition", "operating_efficiency",
                                   "customer_retention", "team_capabilities")
    for code_id in ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC"):
        values = tuple(books.panel.by_id(code_id).values)
        assert values == PROMOTION_CONCEPTS + PREVENTION_CONCEPTS + ("deal_terms", "other", "NONE"), code_id
    assert books.exchange_topics() == PROMOTION_CONCEPTS + PREVENTION_CONCEPTS + ("deal_terms", "other", "NONE")
    assert "PR-FRAME" not in books.panel.ids and "RS-FRAME" not in books.panel.ids


def test_frame_of_and_turn_frame():
    from pipeline.stage_b.panel_books import PREVENTION_CONCEPTS, PROMOTION_CONCEPTS, frame_of, turn_frame
    assert all(frame_of(c) == "promotion" for c in PROMOTION_CONCEPTS)
    assert all(frame_of(c) == "prevention" for c in PREVENTION_CONCEPTS)
    assert frame_of("NONE") == "NEUTRAL" and frame_of(None) is None
    assert frame_of("pricing") is None and frame_of("") is None
    assert frame_of(" Sales ") == "promotion" and frame_of("none") == "NEUTRAL"
    assert frame_of("other") == "unclassified" and frame_of(" Other") == "unclassified"
    assert turn_frame("question", [("PR-CONCEPT", "other")]) == "unclassified"
    assert turn_frame("question", [("PR-CONCEPT", "sales"), ("PR-SEV", "1")]) == "promotion"
    assert turn_frame("answer", [("RS-RESP", "addresses"), ("RS-CONCEPT", "liability")]) == "prevention"
    assert turn_frame("answer", [("RS-CONCEPT", None)]) is None and turn_frame("answer", []) is None
    assert turn_frame("question", [("RS-CONCEPT", "sales")]) is None      # the other function's code does not count
    assert turn_frame("accepts", [("PR-CONCEPT", "sales")]) is None


def test_deal_terms_is_a_concept_and_a_topic_without_a_frame():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import NO_FRAME_CONCEPTS, frame_of, load_panel_books, turn_frame
    books = load_panel_books(load_release())
    for code in ("PR-CONCEPT", "RS-CONCEPT", "EX-TOPIC"):
        values = books.panel.by_id(code).values
        assert values.index("deal_terms") == values.index("other") - 1
    assert NO_FRAME_CONCEPTS == ("other", "deal_terms")
    assert frame_of("deal_terms") == "unclassified" and frame_of("DEAL_TERMS ") == "unclassified"
    assert turn_frame("question", [("PR-CONCEPT", "deal_terms")]) == "unclassified"
    assert turn_frame("answer", [("RS-CONCEPT", "deal_terms")]) == "unclassified"


# ---- v3-dev8 ----------------------------------------------------------------------------

def test_other_concept_travels_through_the_wire():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record
    from tests.panel_fixtures import set_other, synthetic_wire
    data = synthetic_wire()
    set_other(data["turns"][1], "founder's own investment")
    record = wire_to_record(PanelWire.model_validate(data))
    assert record.turns[1].other_concept == "founder's own investment" and record.turns[2].other_concept is None
    del data["turns"][2]["other_concept"]
    with pytest.raises(ValidationError):
        PanelWire.model_validate(data)


def test_other_concepts_lists_named_matters_in_turn_then_exchange_order():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import load_panel_books
    from tests.panel_fixtures import set_other
    books = load_panel_books(load_release())
    assert books.other_concepts(synthetic_record()) == []
    data = synthetic_record()
    set_other(data["turns"][10], "founder's own investment")      # turn 11, question
    set_other(data["turns"][11], None)                             # turn 12, answer, unnamed
    data["turns"][1]["codes"][0]["value"] = " Other "              # turn 2, case and space
    data["turns"][1]["other_concept"] = "  shelf life "
    data["exchanges"][2].update(topic="OTHER", label="money the founder put in")
    assert books.other_concepts(data) == [
        {"where": "turn", "id": 2, "code": "PR-CONCEPT", "name": "shelf life", "clock": "1:02"},
        {"where": "turn", "id": 11, "code": "PR-CONCEPT", "name": "founder's own investment", "clock": "1:55"},
        {"where": "turn", "id": 12, "code": "RS-CONCEPT", "name": "", "clock": "1:59"},
        {"where": "exchange", "id": 3, "code": "EX-TOPIC", "name": "money the founder put in", "clock": "1:55"},
    ]


def test_venture_carries_the_presenting_team_gender_composition():
    from pipeline.schemas.stage_b_panel import PanelWire, Venture, WireVenture
    assert list(Venture.model_fields) == ["description", "primary_industry", "secondary_industries", "product_form",
                                          "customer_type", "founder_gender", "product_category", "earliest_s", "note",
                                          "ask_amount_usd", "ask_equity_pct", "ask_missing"]
    assert list(WireVenture.model_fields) == ["description", "primary_industry", "secondary_industries", "product_form",
                                              "customer_type", "founder_gender", "product_category", "earliest", "note",
                                              "ask_amount_usd", "ask_equity_pct", "ask_missing"]
    base = synthetic_record()["venture"]
    assert Venture.model_validate(base).founder_gender == "all_men"
    with pytest.raises(ValidationError):
        Venture.model_validate({k: v for k, v in base.items() if k != "founder_gender"})
    wire = gemini_response_schema(PanelWire)["properties"]["venture"]
    assert "founder_gender" in wire["required"] and wire["properties"]["founder_gender"]["type"] == "string"
    assert {"ask_amount_usd", "ask_equity_pct", "ask_missing"} <= set(wire["required"])
    assert {"ask_amount_usd", "ask_equity_pct", "ask_missing"} <= set(PanelWire.model_json_schema()["$defs"]["WireVenture"]["required"])
    from tests.panel_fixtures import synthetic_wire
    for field in ("ask_amount_usd", "ask_equity_pct", "ask_missing"):
        with pytest.raises(ValidationError):
            WireVenture.model_validate({k: v for k, v in synthetic_wire()["venture"].items() if k != field})


def test_the_ask_passes_from_wire_to_record():
    from pipeline.schemas.stage_b_panel import PanelWire, Venture, wire_to_record
    from tests.panel_fixtures import synthetic_wire
    record = wire_to_record(PanelWire.model_validate(synthetic_wire()))
    assert (record.venture.ask_amount_usd, record.venture.ask_equity_pct, record.venture.ask_missing) == (
        200000, 10.0, "none")
    base = synthetic_record()["venture"]
    for bad in (dict(ask_equity_pct=150), dict(ask_equity_pct=-1), dict(ask_amount_usd=-5)):
        with pytest.raises(ValidationError):
            Venture.model_validate({**base, **bad})


# ---- a code value written as the word "missing" ---------------------------------------------

def _wire_with_missing_words():
    from tests.panel_fixtures import synthetic_wire
    data = synthetic_wire()
    data["turns"][1]["codes"][1].update(value="missing")                    # PR-SEV, ordinal
    data["turns"][2]["codes"][4].update(value=" N/A ")                      # PREP-SUBS, ordinal
    data["turns"][2]["codes"][5].update(value="None")                       # PREP-COH, ordinal
    data["turns"][2]["codes"][2].update(value="NULL")                       # RS-EVID, binary
    data["turns"][2]["codes"][1].update(value="NONE")                       # RS-CONCEPT, categorical residual
    data["turns"][2]["codes"][0].update(value="Not Applicable")             # RS-RESP, categorical
    data["turns"][4]["codes"][4].update(value="na", missing="too_short")    # already has a reason: the older rule
    return data


def test_missing_words_tuple_is_the_documented_one():
    from pipeline.schemas.stage_b_panel import MISSING_WORDS
    assert MISSING_WORDS == ("missing", "null", "n/a", "na", "not applicable")


def test_a_value_written_as_the_word_missing_becomes_a_missing_value():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record_with_notes
    categorical = {"PR-CONCEPT", "RS-CONCEPT", "RS-RESP"}
    record, notes = wire_to_record_with_notes(PanelWire.model_validate(_wire_with_missing_words()),
                                              categorical_codes=categorical)
    by = lambda i: {c.code: (c.value, c.missing) for c in record.turns[i].codes}   # noqa: E731
    assert by(1)["PR-SEV"] == (None, "not_applicable")
    assert by(2)["PREP-SUBS"] == (None, "not_applicable")
    assert by(2)["PREP-COH"] == (None, "not_applicable")          # "none" on an ordinal code
    assert by(2)["RS-EVID"] == (None, "not_applicable")
    assert by(2)["RS-CONCEPT"] == ("NONE", "none")                # the codebook's residual value stays a value
    assert by(2)["RS-RESP"] == (None, "not_applicable")           # an unambiguous word on a categorical code
    assert by(4)["PREP-SUBS"] == (None, "too_short")              # the stated reason is kept
    assert notes == [
        "turns[1].codes PR-SEV: value 'missing' read as missing, reason set to 'not_applicable'",
        "turns[2].codes RS-RESP: value 'Not Applicable' read as missing, reason set to 'not_applicable'",
        "turns[2].codes RS-EVID: value 'NULL' read as missing, reason set to 'not_applicable'",
        "turns[2].codes PREP-SUBS: value ' N/A ' read as missing, reason set to 'not_applicable'",
        "turns[2].codes PREP-COH: value 'None' read as missing, reason set to 'not_applicable'",
        "turns[4].codes PREP-SUBS: value 'na' dropped because missing is 'too_short'"]


def test_without_code_types_only_the_unambiguous_words_are_missing():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record, wire_to_record_with_notes
    data = _wire_with_missing_words()
    record, notes = wire_to_record_with_notes(PanelWire.model_validate(data))
    by = {c.code: (c.value, c.missing) for c in record.turns[2].codes}
    assert by["PREP-COH"] == ("None", "none") and by["RS-CONCEPT"] == ("NONE", "none")
    assert by["PREP-SUBS"] == (None, "not_applicable") and by["RS-EVID"] == (None, "not_applicable")
    assert not any("PREP-COH" in n for n in notes) and len(notes) == 5
    assert wire_to_record(PanelWire.model_validate(data)) == record


def test_normalize_missing_words_on_a_stored_record_is_idempotent():
    from pipeline.schemas.stage_b_panel import normalize_missing_words
    record = synthetic_record()
    assert normalize_missing_words(record, {"RS-RESP"}) == [] and record == synthetic_record()
    record["turns"][2]["codes"][4].update(value="missing")
    record["turns"][2]["codes"][5].update(value="none")
    notes = normalize_missing_words(record, {"RS-RESP"})
    assert notes == ["turns[2].codes PREP-SUBS: value 'missing' read as missing, reason set to 'not_applicable'",
                     "turns[2].codes PREP-COH: value 'none' read as missing, reason set to 'not_applicable'"]
    assert record["turns"][2]["codes"][4] == {"code": "PREP-SUBS", "value": None, "missing": "not_applicable"}
    PanelRecord.model_validate(record)
    assert normalize_missing_words(record, {"RS-RESP"}) == []


# ---- v3-dev9: the prepared presentation is rated once per pitch --------------------------

def test_prepared_items_travel_on_the_wire_and_round_trip():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record_with_notes
    from tests.panel_fixtures import synthetic_wire
    record = PanelRecord.model_validate(synthetic_record())
    assert [(c.code, c.value, c.missing) for c in record.prepared] == [
        ("PRES-SUBS", "3", "none"), ("PRES-COH", "4", "none"), ("PRES-CTX", "2", "none"), ("PRES-EVID", "5", "none")]
    converted, notes = wire_to_record_with_notes(PanelWire.model_validate(synthetic_wire()))
    assert notes == [] and converted == record
    assert PanelRecord.model_validate_json(record.model_dump_json()) == record
    assert "prepared" in gemini_response_schema(PanelWire)["required"]
    with pytest.raises(ValidationError):
        PanelRecord.model_validate({k: v for k, v in synthetic_record().items() if k != "prepared"})


def test_a_prepared_value_given_with_a_missing_reason_is_dropped_with_a_note():
    from pipeline.schemas.stage_b_panel import PanelWire, wire_to_record_with_notes
    from tests.panel_fixtures import synthetic_wire
    data = synthetic_wire()
    data["prepared"][0].update(value="4", missing="edited_gap")
    record, notes = wire_to_record_with_notes(PanelWire.model_validate(data))
    assert notes == ["prepared[0] PRES-SUBS: value '4' dropped because missing is 'edited_gap'"]
    assert (record.prepared[0].value, record.prepared[0].missing) == (None, "edited_gap")
