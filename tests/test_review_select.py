import pytest

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord, clock_to_seconds
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.replicate_agreement import align
from pipeline.stage_b.review_items import FIELD_CLASSES, SHEET_CLASSES, flatten
from pipeline.stage_b.review_select import select_exchange_sheet, select_sheet
from tests.panel_fixtures import set_other, synthetic_record

CLEAN = {"errors": [], "flags": []}


def select_items(primary, replicate, agreement, validation, books, **kw) -> list[dict]:
    """The rows of select_sheet: every flagged item it keeps and a seeded sample of the unflagged ones."""
    return select_sheet(primary, replicate, agreement, validation, books, **kw)["rows"]
# answer coherence joined the question-answer codes on 2026-10-21 (plan C Task 7e)
SIX_CODES = [["question", "PR-CONCEPT"], ["question", "PR-SEV"], ["answer", "RS-RESP"], ["answer", "RS-EVID"],
             ["answer", "PREP-SUBS"], ["answer", "PREP-COH"]]
ROW_KEYS = {"path", "field_class", "clock_start", "clock_end", "context", "primary", "replicate", "reasons",
            "selected_as", "priority"}


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def pair(mutate_replicate=None, mutate_primary=None):
    a, b = synthetic_record(), synthetic_record()
    if mutate_primary:
        mutate_primary(a)
    if mutate_replicate:
        mutate_replicate(b)
    return PanelRecord.model_validate(a), PanelRecord.model_validate(b)


def select(books, primary, replicate, validation=CLEAN, sample=0, seed=0, with_agreement=True, **kw):
    agreement = align(primary, replicate, books) if with_agreement and replicate is not None else None
    return select_items(primary, replicate, agreement, validation, books, sample_unflagged=sample, seed=seed, **kw)


def flagged(rows):
    return {r["path"]: r for r in rows if r["selected_as"] == "flagged"}


def test_identical_passes_with_a_clean_validation_give_only_the_sample(books):
    p, r = pair()
    rows = select(books, p, r, sample=30)
    assert len(rows) == 30
    assert all(row["selected_as"] == "sampled" and row["reasons"] == [] and row["priority"] == 1 for row in rows)
    assert all(set(row) == ROW_KEYS for row in rows)
    assert all(row["primary"] == row["replicate"] for row in rows)
    assert select(books, p, r, sample=0) == []


def test_a_differing_code_is_flagged_with_both_values(books):
    def mutate(d):
        d["turns"][3]["codes"][1]["value"] = "1"        # turn 4 PR-SEV, 3 in the primary
    p, r = pair(mutate)
    rows = flagged(select(books, p, r))
    assert set(rows) == {"turn:4/code:PR-SEV"}
    row = rows["turn:4/code:PR-SEV"]
    assert (row["primary"], row["replicate"], row["reasons"]) == ("3", "1", ["passes_differ"])
    assert (row["field_class"], row["clock_start"], row["clock_end"]) == ("turn_code", "1:20", "1:26")
    assert row["context"].startswith("turn 4, S2 (Investor B)")


def test_a_turn_the_other_pass_lacks_is_one_unit_presence_row(books):
    def mutate(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]
    p, r = pair(mutate)
    rows = flagged(select(books, p, r))
    assert set(rows) == {"turn:13/present"}
    row = rows["turn:13/present"]
    assert (row["field_class"], row["primary"], row["replicate"]) == ("unit_presence", "present", "absent")
    assert (row["reasons"], row["priority"]) == (["no_counterpart"], 2)
    assert (row["clock_start"], row["clock_end"]) == ("2:10", "2:13")
    assert row["context"] == "turn 13, S2 (Investor B): Calls the reorder rate strong."
    # the turn's own fields are neither flagged for it nor sampled: the unit itself is what is judged
    everything = {x["path"]: x for x in select(books, p, r, sample=10_000)}
    assert not [path for path in everything if path.startswith("turn:13/") and path != "turn:13/present"]


def test_a_missing_exchange_and_a_missing_decision_are_one_row_each(books):
    def mutate(d):
        d["exchanges"] = [e for e in d["exchanges"] if e["exchange_id"] != 2]
        for t in d["turns"]:
            if t["exchange_id"] == 2:
                t["exchange_id"] = None
        d["decisions"] = [x for x in d["decisions"] if x["decision_id"] != 1]
    p, r = pair(mutate)
    rows = flagged(select(books, p, r))
    presence = {path: row for path, row in rows.items() if row["field_class"] == "unit_presence"}
    assert set(presence) == {"exchange:2/present", "decision:1/present"}
    assert presence["exchange:2/present"]["context"].startswith("exchange 2, turns 8-9: social")
    assert presence["decision:1/present"]["context"].startswith("decision 1, turn 10: S3 (Investor C) exit")
    assert not [path for path in rows if path.startswith(("exchange:2/", "decision:1/")) and path not in presence]
    assert rows["turn:8/exchange"]["reasons"] == ["passes_differ"]         # flagged for another reason


def test_a_unit_only_the_replicate_has_is_one_row_under_a_replicate_only_path(books):
    # the primary lacks turn 13 and decision 1; the replicate calls Investor B "S1"
    def relabel(d):
        swap = {"S1": "S2", "S2": "S1"}
        d["actors"][1]["label"], d["actors"][2]["label"] = d["actors"][2]["label"], d["actors"][1]["label"]
        for t in d["turns"]:
            t["speaker"] = swap.get(t["speaker"], t["speaker"])
        for x in d["decisions"]:
            x["actor"] = swap.get(x["actor"], x["actor"])
    p, r = pair(mutate_replicate=relabel, mutate_primary=lambda d: d.update(
        turns=[t for t in d["turns"] if t["turn_id"] != 13],
        decisions=[x for x in d["decisions"] if x["decision_id"] != 1]))
    rows = flagged(select(books, p, r))
    presence = {path: row for path, row in rows.items() if row["field_class"] == "unit_presence"}
    assert set(presence) == {"replicate_only:turn@2:10", "replicate_only:decision@1:51"}
    turn = presence["replicate_only:turn@2:10"]
    assert (turn["primary"], turn["replicate"], turn["reasons"], turn["priority"]) == (
        "absent", "present", ["no_counterpart"], 2)
    assert (turn["clock_start"], turn["clock_end"]) == ("2:10", "2:13")
    assert turn["context"] == ("only in the replicate, turn by S2 (Investor B): accepts, Calls the reorder rate "
                               "strong.")
    assert presence["replicate_only:decision@1:51"]["context"] == (
        "only in the replicate, decision by S3 (Investor C): exit")
    assert not [x for x in select(books, p, r, replicate_only=False) if x["path"].startswith("replicate_only:")]


def test_empty_none_and_blank_are_one_value_and_non_substantive_exchanges_have_no_topic(books):
    def mutate(d):
        d["exchanges"][3]["topic"] = ""                          # negotiation: NONE in the primary
        d["venture"]["customer_type"] = ""
    p, r = pair(mutate, lambda d: d["venture"].update(customer_type="NONE"))
    rows = select(books, p, r, sample=10_000)
    assert set(flagged(rows)) == set()
    paths = {row["path"] for row in rows}
    assert not [x for x in paths if x.startswith(("exchange:2/", "exchange:4/")) and x.endswith("topic")]
    assert not [x for x in paths if "investor" in x and x.startswith("exchange:")]       # none since v3-dev9
    assert {"exchange:1/topic", "exchange:2/kind", "exchange:4/kind", "venture:customer_type"} <= paths


def test_a_validation_message_flags_only_the_field_it_is_about(books):
    p, r = pair()
    validation = {
        "errors": ["[exchange.topic] exchange 2: topic is 'sales' but the exchange is not substantive",
                   "[exchange.topic] exchange 1: 'pricing' is not an EX-TOPIC value",
                   "[turn.exchange] turn 9: exchange_id 4 is not a recorded exchange",
                   "[turn.codes] turn 4: PR-SEV 9 is outside 1-3",
                   "[turn.codes] turn 6: RS-RESP applies to 'answer', not 'repair'",
                   "[turn.link] turn 7: responds_to 99 does not exist",
                   "[turn.time] turn 8: 105.0-108.0 is reversed or beyond 100s",
                   "[decision.offer] decision 3: joins_offer has no offer_id",
                   "[stage_a.moment] interpersonal_warmth: 200.0s is outside every prepared span",
                   "[venture.value] product_form: 'gizmo' is not a VC-FORM value",
                   "[claim.turn] conflict 3-5: turn_a is not earlier than turn_b"],
        "flags": ["[flag.unanswered_question] turn 11: no answer responds to it",
                  "[flag.turn_role] turn 2: question is expected from an investor, got founder 'F1'",
                  "[flag.orphan_turn] turn 14: question belongs to no exchange",
                  "[flag.decision_function] decision 2: turn 17 has function 'negotiates', not decides",
                  "[flag.exit_in_exchange] decision 1: exit turn 10 belongs to exchange 2, which is not the "
                  "negotiation",
                  "[flag.multiple_negotiations] exchanges [2, 4] are all negotiation; there is one negotiation "
                  "exchange per encounter",
                  "[flag.decides_without_decision] turn 10: no decision names this decides turn",
                  "[flag.no_prepared] no prepared span",
                  "[flag.no_decision] investor S9 has no decision"]}
    rows = flagged(select(books, p, r, validation=validation))
    assert {path: row["reasons"] for path, row in rows.items()} == {
        # exchange 2 is social, so it has no topic item: the message falls back to its kind
        "exchange:2/kind": ["validation:exchange.topic", "validation:flag.multiple_negotiations"],
        "exchange:1/topic": ["validation:exchange.topic"],
        "turn:9/exchange": ["validation:turn.exchange"],
        "turn:4/code:PR-SEV": ["validation:turn.codes"],
        "turn:6/function": ["validation:turn.codes"],              # the turn has no RS-RESP item
        "turn:7/responds_to": ["validation:turn.link"],
        "turn:8/function": ["validation:turn.time"],               # no field of its own: the unit's main field
        "decision:3/kind": ["validation:decision.offer"],
        "stage_a:interpersonal_warmth": ["validation:stage_a.moment"],
        "venture:product_form": ["validation:venture.value"],
        "turn:11/responds_to": ["validation:flag.unanswered_question"],
        "turn:2/function": ["validation:flag.turn_role"],
        "turn:14/exchange": ["validation:flag.orphan_turn"],
        "decision:2/kind": ["validation:flag.decision_function"],
        "decision:1/kind": ["validation:flag.exit_in_exchange"],
        "exchange:4/kind": ["validation:flag.multiple_negotiations"],
        "turn:10/function": ["validation:flag.decides_without_decision"],
        "prepared_window": ["validation:flag.no_prepared"],
    }
    assert rows["turn:4/code:PR-SEV"]["priority"] == 3 and rows["turn:11/responds_to"]["priority"] == 4
    assert rows["exchange:2/kind"]["priority"] == 3


def test_informational_validation_flags_flag_nothing(books):
    p, r = pair()
    validation = {"errors": [], "flags": [
        "[flag.value_case] turn 2: function 'Question' → 'question'",
        "[flag.value_with_missing] turn 3: RS-EVID value dropped, it came with missing not_applicable",
        "[flag.gap] gap at 12.0s", "[flag.unknown_addressee] turn 4: addressee unknown",
        "[flag.single_turn_exchange] exchange 3: substantive exchange of one turn",
        "[flag.venture_unknown] customer_type is NONE"]}
    assert flagged(select(books, p, r, validation=validation)) == {}


def test_other_concept_messages_point_at_the_concept_code_or_the_topic(books):
    def mutate(d):
        set_other(d["turns"][1], "founder salary")
        d["exchanges"][2].update(topic="other", label="patent dispute")
    p, r = pair(mutate, mutate)
    validation = {"errors": [], "flags": ['[flag.other_concept] turn 2: PR-CONCEPT other is "founder salary"',
                                          '[flag.other_concept] exchange 3: EX-TOPIC other is "patent dispute"',
                                          "[flag.other_concept_unused] turn 4: other_concept is 'x' but no concept "
                                          "of the turn is other"]}
    rows = flagged(select(books, p, r, validation=validation))
    assert rows["turn:2/code:PR-CONCEPT"]["reasons"] == ["validation:flag.other_concept", "other_concept"]
    assert rows["exchange:3/topic"]["reasons"] == ["validation:flag.other_concept", "other_concept"]
    assert rows["turn:2/other_concept"]["reasons"] == ["other_concept"]
    assert rows["turn:4/code:PR-CONCEPT"]["reasons"] == ["validation:flag.other_concept_unused"]


def test_low_value_fields_are_off_unless_their_class_is_asked_for(books):
    def mutate(d):
        d["actors"][3]["label"] = "Lady"                # S3: mapped by order
        d["exchanges"][0]["label"] = "CAC"
        d["turns"][2]["responds_to"] = []               # turn 3: the trivial link in the primary
        d["turns"][15]["responds_to"] = [14]            # turn 16 (challenges): a link that is not trivial
    p, r = pair(mutate)
    rows = select(books, p, r, sample=10_000)
    paths = {row["path"] for row in rows}
    assert not [x for x in paths if x.startswith("actor:") or x.endswith(("/label", "/other_concept"))]
    assert not {"turn:3/responds_to", "turn:5/responds_to", "turn:12/responds_to", "turn:15/responds_to"} & paths
    assert set(flagged(rows)) == {"turn:16/responds_to"}
    assert {"turn:2/responds_to", "turn:6/responds_to", "turn:13/responds_to"} <= paths     # other links stay

    asked = {row["path"]: row for row in select(books, p, r, sample=10_000,
                                                field_classes=("actor", "turn_link", "exchange_topic", "turn_code"))}
    assert asked["actor:S3/role"]["reasons"] == ["actor_mapped_by_order"]
    assert asked["turn:3/responds_to"]["reasons"] == ["passes_differ"]
    assert {"actor:S1/label", "exchange:1/label", "turn:2/other_concept", "turn:5/responds_to"} <= set(asked)
    assert "exchange:2/topic" not in asked                                  # never an item, whatever is asked


def test_a_cap_keeps_unit_presence_and_validator_errors_and_draws_the_rest(books):
    def mutate(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]
        for t in d["turns"]:
            for c in t["codes"]:
                if c["code"] in ("PR-SEV", "PREP-SUBS", "PREP-COH"):
                    c["value"] = "2" if c["value"] != "2" else "1"
        for i in d["stage_a"]:
            i["value"] = 5
        d["exchanges"][0]["topic"] = "forecast"
    p, r = pair(mutate)
    validation = {"errors": ["[turn.codes] turn 4: PR-SEV 9 is outside 1-3",
                             "[decision.offer] decision 3: joins_offer has no offer_id"],
                  "flags": ["[flag.turn_role] turn 2: question is expected from an investor, got founder 'F1'"]}

    def sheet(**kw):
        return select_sheet(p, r, None, validation, books, sample_unflagged=4, seed=1, **kw)

    full = sheet()
    found = full["flagged"]["found"]
    assert found["n"] == 19 and full["flagged"]["left_out"] == {"n": 0, "by_field_class": {}, "by_reason": {}}
    assert full["flagged"]["on_sheet"] == found and len(full["rows"]) == 23
    assert sheet(max_flagged=19) == {**full, "flagged": {**full["flagged"], "max_flagged": 19}}

    capped = sheet(max_flagged=8)
    rows = flagged(capped["rows"])
    assert len(rows) == 8 and len(capped["rows"]) == 12
    assert {"turn:13/present", "turn:4/code:PR-SEV", "decision:3/kind"} <= set(rows)
    drawn = [row for row in rows.values() if row["priority"] == 4]
    assert len(drawn) == 5
    # the draw reaches every field class that has flagged rows before it takes a second row of any
    assert {row["field_class"] for row in drawn} >= {"turn_function", "turn_code", "exchange_topic", "stage_a"}
    left = capped["flagged"]["left_out"]
    assert left["n"] == 11 and capped["flagged"]["found"] == found and capped["flagged"]["on_sheet"]["n"] == 8
    assert sum(left["by_field_class"].values()) == 11 and set(left["by_field_class"]) == {"turn_code", "stage_a"}
    assert left["by_reason"] == {"passes_differ": left["by_field_class"]["turn_code"],
                                 "stage_a_gap": left["by_field_class"]["stage_a"]}
    for cls, n in found["by_field_class"].items():
        assert n == capped["flagged"]["on_sheet"]["by_field_class"].get(cls, 0) + left["by_field_class"].get(cls, 0)
    assert capped == sheet(max_flagged=8)                                   # the draw is seeded
    other = select_sheet(p, r, None, validation, books, sample_unflagged=4, seed=2, max_flagged=8)
    assert set(flagged(other["rows"])) != set(rows)
    # the cap never drops what must be kept, and the sample of unflagged rows does not depend on it
    must = flagged(sheet(max_flagged=1)["rows"])
    assert set(must) == {"turn:13/present", "turn:4/code:PR-SEV", "decision:3/kind"}
    sampled = lambda out: [row["path"] for row in out["rows"] if row["selected_as"] == "sampled"]  # noqa: E731
    assert sampled(capped) == sampled(full)


def test_an_other_concept_is_flagged_even_when_the_passes_agree(books):
    def mutate(d):
        set_other(d["turns"][1], "founder salary")
        d["exchanges"][2].update(topic="other", label="patent dispute")
    p, r = pair(mutate, mutate)
    rows = flagged(select(books, p, r))
    assert set(rows) == {"turn:2/code:PR-CONCEPT", "turn:2/other_concept", "exchange:3/topic", "exchange:3/label"}
    assert all(row["reasons"] == ["other_concept"] for row in rows.values())
    assert rows["turn:2/other_concept"]["primary"] == "founder salary"


def test_a_stage_a_gap_of_two_is_flagged_and_a_gap_of_one_is_not(books):
    def mutate(d):
        d["stage_a"][2]["value"] = 5        # interpersonal_warmth, 3 in the primary
        d["stage_a"][1]["value"] = 4        # expressed_enthusiasm
        d["stage_a"][0].update(value=None, uncodable=True)      # pitch_positivity
    p, r = pair(mutate)
    rows = flagged(select(books, p, r))
    assert set(rows) == {"stage_a:interpersonal_warmth", "stage_a:pitch_positivity"}
    assert rows["stage_a:interpersonal_warmth"]["reasons"] == ["stage_a_gap"]
    assert (rows["stage_a:interpersonal_warmth"]["primary"], rows["stage_a:interpersonal_warmth"]["replicate"]) == ("3", "5")
    assert rows["stage_a:pitch_positivity"]["replicate"] is None


def test_an_actor_mapped_by_order_is_flagged(books):
    def mutate(d):
        d["actors"][3]["label"] = "Lady"        # S3 "Investor C": no naming word in common
    p, r = pair(mutate)
    rows = flagged(select(books, p, r))
    assert rows == {}                               # actors are not on a sheet unless asked for
    rows = flagged(select(books, p, r, field_classes=("actor",)))
    assert set(rows) == {"actor:S3/label", "actor:S3/role"}
    assert rows["actor:S3/role"]["reasons"] == ["actor_mapped_by_order"]


def test_replicate_values_are_read_at_the_aligned_unit_and_in_the_primary_ids(books):
    def mutate(d):
        swap = {"S1": "S2", "S2": "S1"}
        d["actors"][1]["label"], d["actors"][2]["label"] = d["actors"][2]["label"], d["actors"][1]["label"]
        for t in d["turns"]:
            t["turn_id"] += 100
            t["speaker"] = swap.get(t["speaker"], t["speaker"])
            t["responds_to"] = [i + 100 for i in t["responds_to"]]
            if t["exchange_id"] is not None:
                t["exchange_id"] += 10
        for e in d["exchanges"]:
            e["exchange_id"] += 10
        for x in d["decisions"]:
            x["decision_id"] += 20
            x["actor"] = swap.get(x["actor"], x["actor"])
            x["turn_id"] += 100
            if x["offer_id"]:
                x["offer_id"] = "offer-A"
    p, r = pair(mutate)
    rows = {row["path"]: row for row in select(books, p, r, sample=10_000, field_classes=SHEET_CLASSES)}
    assert flagged(rows.values()) == {}
    # no topic of the social and negotiation exchanges; one outcome for the pitch
    assert len(rows) == len(flatten(p, books)) - 2 + 1
    assert (rows["pitch:any_offer"]["primary"], rows["pitch:any_offer"]["replicate"]) == ("yes", "yes")
    assert (rows["prep_pres:PRES-EVID"]["primary"], rows["venture:ask"]["replicate"]) == ("5", "200000;10")
    assert rows["turn:2/exchange"]["replicate"] == "1"
    assert rows["turn:7/responds_to"]["replicate"] == "6"
    assert rows["decision:2/actor"]["replicate"] == "S1"
    assert rows["decision:2/offer_id"]["replicate"] == "O1"
    assert rows["actor:S1/label"]["replicate"] == "Investor A"


def test_times_differ_only_beyond_the_tolerance_and_free_text_never_differs(books):
    def near(d):
        d["decisions"][0]["time_s"] += 20       # decision tolerance is 30 s
        d["prepared_spans"][0]["end_s"] += 8    # event tolerance is 10 s
        d["exchanges"][0]["label"] = "CAC"
        d["actors"][1]["label"] = "Kevin"           # mapped by the shared name, so not a guess
    p, r = pair(near, lambda d: d["actors"][1].update(label="Kevin O'Leary"))
    assert flagged(select(books, p, r)) == {}

    def far(d):
        d["decisions"][1]["time_s"] -= 45
        d["prepared_spans"][0]["end_s"] -= 25
    p, r = pair(far)
    rows = flagged(select(books, p, r))
    assert set(rows) == {"decision:2/time", "prepared_window"}
    assert rows["prepared_window"]["replicate"] == "0:05-0:35"


def test_sampling_is_reproducible_and_never_returns_a_flagged_path(books):
    def mutate(d):
        d["turns"][3]["codes"][1]["value"] = "1"
    p, r = pair(mutate)
    first = select(books, p, r, sample=25, seed=7)
    assert first == select(books, p, r, sample=25, seed=7)
    other = select(books, p, r, sample=25, seed=8)
    assert [row["path"] for row in first] != [row["path"] for row in other]
    for rows in (first, other):
        assert [row["selected_as"] for row in rows].count("sampled") == 25
        assert [row["path"] for row in rows if row["selected_as"] == "flagged"] == ["turn:4/code:PR-SEV"]
        assert len({row["path"] for row in rows}) == 26


def _sheet(books, primary, replicate, sample, seed=0, **kw):
    return select_sheet(primary, replicate, align(primary, replicate, books), CLEAN, books, sample_unflagged=sample,
                        seed=seed, **kw)


def test_the_unflagged_sample_is_proportional_to_each_field_class_with_no_forced_one_per_class(books):
    p, r = pair()
    everything = _sheet(books, p, r, 10_000)
    population = everything["unflagged"]["population"]
    assert sum(population.values()) == len(everything["rows"]) and population["turn_code"] > 4 * population["venture"]
    assert everything["unflagged"]["on_sheet"] == population
    total = sum(population.values())
    for seed in range(5):
        sheet = _sheet(books, p, r, 24, seed=seed)
        drawn = sheet["unflagged"]["on_sheet"]
        assert sum(drawn.values()) == len(sheet["rows"]) == 24 and sheet["unflagged"]["population"] == population
        for cls, n in population.items():
            assert abs(drawn.get(cls, 0) - 24 * n / total) < 1, (cls, drawn)       # its share, to the nearest row
        assert sheet["unflagged"]["requested"] == 24 and sheet["unflagged"]["seed"] == seed
    # a small budget goes where the fields are: no class is given a row it has no share of
    few = _sheet(books, p, r, 3)["unflagged"]["on_sheet"]
    assert sum(few.values()) == 3 and few.get("turn_code", 0) >= 1 and "prepared_window" not in few
    assert _sheet(books, p, r, 0)["unflagged"]["on_sheet"] == {}
    note = everything["unflagged"]["note"]
    assert "proportional" in note and "nothing flagged" in note and "both passes recorded" in note
    assert "not" in note and "flagged fields" in note


def test_fields_of_a_unit_the_other_pass_lacks_are_not_in_the_unflagged_pool(books):
    def mutate(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]          # S2's accepts turn
        d["decisions"] = d["decisions"][:3]
    p, r = pair(mutate)
    whole, _ = pair()
    baseline = _sheet(books, whole, whole, 10_000)["unflagged"]["population"]
    sheet = _sheet(books, p, r, 10_000)
    sampled = {row["path"] for row in sheet["rows"] if row["selected_as"] == "sampled"}
    assert not [path for path in sampled if path.startswith(("turn:13/", "decision:4/"))]
    assert {"turn:13/present", "decision:4/present"} <= set(flagged(sheet["rows"]))
    unflagged = sheet["unflagged"]
    # turn 13 (function, exchange, link) and decision 4 (kind, actor, time, offer id) are judged as units only
    assert unflagged["excluded_no_counterpart"] == {
        "n": 7, "by_field_class": {"turn_function": 1, "turn_exchange": 1, "turn_link": 1, "decision": 4}}
    assert unflagged["population"]["decision"] == baseline["decision"] - 4
    assert unflagged["population"]["turn_function"] == baseline["turn_function"] - 1
    assert sum(unflagged["population"].values()) == len(sampled)
    # without a replicate nothing has a counterpart to lack
    alone = select_sheet(p, None, None, CLEAN, books, sample_unflagged=10_000, seed=0)["unflagged"]
    assert alone["excluded_no_counterpart"] == {"n": 0, "by_field_class": {}}


def test_a_sample_larger_than_what_is_left_takes_everything(books):
    p, r = pair()
    assert len(select(books, p, r, sample=10_000, field_classes=FIELD_CLASSES)) == len(flatten(p, books)) - 2
    # by default also without 8 actor fields, 4 labels, 8 unnamed other_concepts and 4 trivial answer links
    assert len(select(books, p, r, sample=10_000)) == len(flatten(p, books)) - 2 - 24


def test_field_classes_filter_both_the_flagged_and_the_sampled_rows(books):
    def mutate(d):
        d["turns"][3]["codes"][1]["value"] = "1"
        d["stage_a"][2]["value"] = 5
    p, r = pair(mutate)
    rows = select(books, p, r, sample=6, field_classes=("stage_a", "turn_function"))
    assert {row["field_class"] for row in rows} == {"stage_a", "turn_function"}
    assert set(flagged(rows)) == {"stage_a:interpersonal_warmth"}
    assert len(rows) == 7                     # one flagged and a sample of 6 from the unflagged pool
    with pytest.raises(ValueError, match="closing"):          # no closing state since v3-dev9
        select(books, p, r, sample=6, field_classes=("closing",))
    with pytest.raises(ValueError, match="mood"):
        select(books, p, r, field_classes=("mood",))
    # unit_presence is a class like the others: on by default, filtered when it is not named
    p, r = pair(lambda d: d.update(decisions=d["decisions"][:3]))
    assert set(flagged(select(books, p, r))) == {"decision:4/present"}
    assert flagged(select(books, p, r, field_classes=("decision",))) == {}
    assert set(flagged(select(books, p, r, field_classes=("unit_presence",)))) == {"decision:4/present"}


def test_rows_are_ordered_by_clock_then_path(books):
    p, r = pair()
    rows = select(books, p, r, sample=10_000)

    def key(row):
        return (-1.0 if row["clock_start"] is None else clock_to_seconds(row["clock_start"]), row["path"])
    assert [key(row) for row in rows] == sorted(key(row) for row in rows)
    assert rows[0]["clock_start"] in (None, "0:05")


def test_without_a_replicate_nothing_is_compared(books):
    p, _ = pair()
    rows = select_items(p, None, None, {"errors": [], "flags": ["[flag.orphan_turn] turn 2: question belongs to no "
                                                                 "exchange"]}, books, sample_unflagged=3, seed=0)
    assert all(row["replicate"] is None for row in rows)
    assert set(flagged(rows)) == {"turn:2/exchange"}
    assert len(rows) == len(flagged(rows)) + 3


def test_agreement_is_computed_when_it_is_not_given(books):
    def mutate(d):
        d["turns"][3]["codes"][1]["value"] = "1"
    p, r = pair(mutate)
    assert select(books, p, r, with_agreement=False) == select(books, p, r)


# ---- the exchange-level sheet -----------------------------------------------------------------

def exchange_sheet(books, p, r, validation=CLEAN, seed=0, key="vid1", **kw):
    return select_exchange_sheet(p, r, None, validation, books, seed=seed, key=key, **kw)


def test_the_exchange_sheet_is_a_random_sample_of_pitch_rows_exchanges_and_pairs_with_venture_and_stage_a_on_request(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, venture=2, stage_a=2, exchanges=4)
    rows = {row["path"]: row for row in out["rows"]}
    assert len(rows) == len(out["rows"]) == 19
    assert all(row["selected_as"] == "sampled" and row["priority"] == 1 and row["reasons"] == []
               for row in rows.values())
    by_class = {}
    for row in rows.values():
        by_class.setdefault(row["field_class"], []).append(row["path"])
    # the pitch's cut (no outcome or window card since Task 7d); 2 venture; 2 Stage A; both substantive exchanges
    # (topic; no closing since v3-dev9); 3 pairs asked, but only 2 come before the first decision (turn 10): 2 x 6 codes
    assert {c: len(v) for c, v in by_class.items()} == {"cut": 1, "venture": 2, "stage_a": 2, "exchange_topic": 2,
                                                        "turn_code": 12}
    assert by_class["cut"] == ["pitch:first_decision"]
    assert "venture:ask" not in rows                    # the ask is coded in the clip session
    assert "venture:founder_gender" in by_class["venture"]
    assert set(by_class["exchange_topic"]) == {"exchange:1/topic", "exchange:3/topic"}
    sample = out["sample"]
    assert sample["requested"] == {"stage_a": 2, "exchanges": 4, "qa": 3, "venture": 2, "qa_codes": SIX_CODES}
    assert (sample["seed"], sample["key"]) == (0, "vid1")
    assert sample["available"] == {"venture": 4, "stage_a": 3, "exchanges": 2, "qa": 2}
    assert sample["drawn"]["exchanges"] == [1, 3] and len(sample["drawn"]["qa"]) == 2
    assert sample["qa_before_cut"] is True
    assert sample["on_sheet"] == {"cut": 1, "turn_code": 12, "exchange_topic": 2, "stage_a": 2, "venture": 2}
    assert sample["population"] == {"cut": 1, "turn_code": 12, "exchange_topic": 2, "stage_a": 3, "venture": 4}
    assert sample["drawn"]["pitch"] == ["first_decision"]
    for question, answer in sample["drawn"]["qa"]:
        assert {f"turn:{question}/code:PR-CONCEPT", f"turn:{question}/code:PR-SEV", f"turn:{answer}/code:RS-RESP",
                f"turn:{answer}/code:RS-EVID", f"turn:{answer}/code:PREP-SUBS"} <= set(rows)
    assert {(q, a) for q, a in sample["drawn"]["qa"]} == {(2, 3), (4, 5)}           # 11-12 and 14-15 are after it


def test_exchange_sheet_values_and_contexts(books):
    p, r = pair()
    rows = {row["path"]: row for row in exchange_sheet(books, p, r, qa=10, venture=2, exchanges=4,
                                                        field_classes=SHEET_CLASSES)["rows"]}       # the outcome too
    outcome = rows["pitch:any_offer"]
    assert (outcome["primary"], outcome["replicate"], outcome["context"]) == (
        "yes", "yes", "the pitch: S3 exit@1:51, S1 offer@2:39, S2 joins_offer@2:44")
    assert outcome["blind_context"] == "the pitch: Did any investor make an offer? Watch to the end."
    assert (outcome["clock_start"], outcome["clock_end"]) == (None, None)
    topic = rows["exchange:3/topic"]
    assert topic["context"] == ('exchange 3 "repeat purchase rate", 1:55-2:13; investors: S2 (Investor B); questions: '
                                "turn 11 (1:55-1:59) S2 (Investor B): Asks for the repeat purchase rate.")
    assert topic["blind_context"] == ("exchange 3, 1:55-2:13; investors: S2 (Investor B); questions: turn 11 "
                                      "(1:55-1:59) S2 (Investor B): Asks for the repeat purchase rate.")
    assert (topic["primary"], topic["clock_start"], topic["clock_end"]) == ("customer_retention", "1:55", "2:13")
    assert rows["exchange:1/topic"]["context"].count("turn ") == 3              # at most three questions
    assert not [path for path in rows if "/investor:" in path]           # no closing state since v3-dev9
    pair_text = ("question turn 2 (1:02-1:06) S1 (Investor A): Asks what it costs to acquire one customer. -> answer "
                 "turn 3 (1:06-1:20) F1 (Founder): States a blended acquisition figure and names the channels.")
    for path, start in (("turn:2/code:PR-CONCEPT", "1:02"), ("turn:3/code:RS-RESP", "1:06"),
                        ("turn:3/code:RS-EVID", "1:06"), ("turn:3/code:PREP-SUBS", "1:06")):
        assert rows[path]["context"] == rows[path]["blind_context"] == pair_text and rows[path]["clock_start"] == start
    # no blind context prints the value its row asks for, nor (for the outcome) any recorded move
    for row in rows.values():
        shown = row.get("blind_context")
        if shown is not None and row["field_class"] != "turn_code":
            assert str(row["primary"]) not in shown, row["path"]
    assert not [k for row in rows.values() if row["field_class"] in ("outcome", "cut")
                for k in ("offer@", "exit", "joins", "@") if k in row["blind_context"]]
    cut = rows["pitch:first_decision"]
    assert (cut["primary"], cut["replicate"], cut["clock_start"], cut["clock_end"]) == ("1:51", "1:51", None, None)
    assert cut["blind_context"] == "the pitch: When is the first decision spoken?"
    assert cut["context"] == "the pitch: S3 exit@1:51, S1 offer@2:39, S2 joins_offer@2:44"
    assert "customer acquisition cost" not in rows["exchange:1/topic"]["blind_context"]


def test_the_outcome_follows_the_offer_rule_and_is_compared_between_the_passes(books):
    def all_exit(d):
        d["decisions"] = [{**d["decisions"][0], "decision_id": i, "actor": a} for i, a in enumerate(("S1", "S2", "S3"), 1)]

    p, r = pair(mutate_primary=all_exit)
    row = next(row for row in exchange_sheet(books, p, r, field_classes=SHEET_CLASSES)["rows"]
               if row["field_class"] == "outcome")
    assert (row["path"], row["primary"], row["replicate"]) == ("pitch:any_offer", "no", "yes")
    assert (row["reasons"], row["selected_as"]) == (["passes_differ"], "sampled")

    p, r = pair(mutate_primary=lambda d: d.update(decisions=d["decisions"][:1]))     # only S3 is seen deciding
    row = next(row for row in exchange_sheet(books, p, r, field_classes=SHEET_CLASSES)["rows"]
               if row["field_class"] == "outcome")
    assert (row["primary"], row["replicate"], row["context"]) == ("unknown", "yes", "the pitch: S3 exit@1:51")
    assert all(row["replicate"] is None for row in exchange_sheet(books, p, None, field_classes=SHEET_CLASSES)["rows"])


def test_flags_are_shown_on_an_exchange_sheet_but_select_nothing(books):
    def mutate(d):
        for t in d["turns"]:
            for c in t["codes"]:
                if c["code"] == "RS-RESP":
                    c["value"] = "partial"
        d["exchanges"][0]["topic"] = "forecast"
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]
    p, r = pair(mutate)
    validation = {"errors": ["[exchange.topic] exchange 3: 'pricing' is not an EX-TOPIC value"], "flags": []}
    clean = exchange_sheet(books, *pair(), exchanges=4)
    out = exchange_sheet(books, p, r, validation=validation, exchanges=4)
    assert [row["path"] for row in out["rows"]] == [row["path"] for row in clean["rows"]]       # same draw
    rows = {row["path"]: row for row in out["rows"]}
    assert rows["exchange:1/topic"]["reasons"] == ["passes_differ"]
    assert rows["exchange:3/topic"]["reasons"] == ["validation:exchange.topic"]
    answers = [row for row in rows.values() if row["path"].endswith("RS-RESP")]
    assert answers and all(row["reasons"] == ["passes_differ"] and row["replicate"] == "partial" for row in answers)
    assert all(row["selected_as"] == "sampled" and row["priority"] == 1 for row in rows.values())
    assert not [path for path in rows if "present" in path]                    # no unit_presence rows either


def test_default_full_video_sheet_draws_no_exchange_rows(books):
    """Since 2026-10-21 no reported analysis reads the model's exchange grouping, so the sheet does not ask for it."""
    from pipeline.stage_b.review_select import EXCHANGE_DEFAULTS
    assert EXCHANGE_DEFAULTS == {"stage_a": 0, "exchanges": 0, "qa": 3, "venture": 0}
    p, r = pair()
    out = exchange_sheet(books, p, r)
    assert not [row for row in out["rows"] if row["field_class"] == "exchange_topic" or "/topic" in row["path"]]
    sample = out["sample"]
    assert sample["requested"]["exchanges"] == 0 and sample["drawn"]["exchanges"] == []
    assert "exchange_topic" not in sample["on_sheet"]
    # what the record holds is still counted, so a sheet that asks for exchanges can be weighted
    assert sample["available"]["exchanges"] == 2 and sample["population"]["exchange_topic"] == 2


def test_exchanges_can_still_be_requested(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, exchanges=2)
    topics = [row["path"] for row in out["rows"] if row["field_class"] == "exchange_topic"]
    assert topics == ["exchange:1/topic", "exchange:3/topic"]
    assert out["sample"]["requested"]["exchanges"] == 2 and out["sample"]["drawn"]["exchanges"] == [1, 3]
    assert out["sample"]["on_sheet"]["exchange_topic"] == 2


def test_the_exchange_draw_is_seeded_per_video_and_respects_the_counts(books):
    p, r = pair()
    paths = lambda **kw: [row["path"] for row in exchange_sheet(books, p, r, **kw)["rows"]]  # noqa: E731
    assert paths() == paths()
    stage_a = lambda **kw: exchange_sheet(books, p, r, stage_a=2, **kw)["sample"]["drawn"]["stage_a"]  # noqa: E731
    assert len({tuple(stage_a(key=f"vid{i}")) for i in range(8)}) > 1          # two videos do not share a draw
    assert len({tuple(stage_a(seed=i)) for i in range(8)}) > 1
    out = exchange_sheet(books, p, r, stage_a=0, exchanges=1, qa=1)
    assert out["sample"]["requested"]["stage_a"] == 0
    assert out["sample"]["on_sheet"]["exchange_topic"] == 1 and "stage_a" not in out["sample"]["on_sheet"]
    assert out["sample"]["on_sheet"]["turn_code"] == 6 and out["sample"]["population"]["stage_a"] == 3
    assert len(exchange_sheet(books, p, r, stage_a=99)["sample"]["drawn"]["stage_a"]) == 3
    only = exchange_sheet(books, p, r, exchanges=4, field_classes=("outcome", "exchange_topic"))
    assert {row["field_class"] for row in only["rows"]} == {"outcome", "exchange_topic"}
    pitch = exchange_sheet(books, p, r, field_classes=("cut", "prepared_window"))
    assert [row["path"] for row in pitch["rows"]] == ["pitch:first_decision", "prepared_window"]
    with pytest.raises(ValueError, match="mood"):
        exchange_sheet(books, p, r, field_classes=("mood",))


def test_the_draw_can_take_every_stage_a_item_and_no_venture_row(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, stage_a="all", venture=0)
    by_class = {}
    for row in out["rows"]:
        by_class.setdefault(row["field_class"], []).append(row["path"])
    assert len(by_class["stage_a"]) == 3 and "venture" not in by_class
    assert out["sample"]["requested"]["stage_a"] == "all" and out["sample"]["drawn"]["venture"] == []
    assert out["sample"]["on_sheet"].get("venture") is None


def test_qa_codes_choose_which_codes_a_pair_puts_on_the_sheet(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, qa=10, qa_codes=(("question", "PR-SEV"),))
    paths = [row["path"] for row in out["rows"] if row["field_class"] == "turn_code"]
    assert paths and all(path.endswith("/code:PR-SEV") for path in paths)
    assert out["sample"]["population"]["turn_code"] == 2          # one code per pair, two pairs before the cut
    six = exchange_sheet(books, p, r, qa=1)
    assert [list(pair) for pair in six["sample"]["requested"]["qa_codes"]] == SIX_CODES
    with pytest.raises(ValueError, match="qa_codes"):
        exchange_sheet(books, p, r, qa_codes=(("question", "NO-SUCH"),))


def test_two_answers_to_one_question_share_its_concept_row(books):
    def mutate(d):
        d["turns"][4]["responds_to"] = [2]              # turn 5 also answers turn 2
    p, r = pair(mutate, mutate)
    out = exchange_sheet(books, p, r, qa=10)
    paths = [row["path"] for row in out["rows"]]
    assert len(paths) == len(set(paths)) and paths.count("turn:2/code:PR-CONCEPT") == 1
    # both pairs come before the first decision: (2, 3) and (2, 5), turn 2's two codes once, each answer's four
    assert out["sample"]["available"]["qa"] == 2 and out["sample"]["population"]["turn_code"] == 10


def test_the_default_exchange_sheet_has_the_cut_row_and_no_outcome_or_window_row(books):
    """Since 2026-10-21 (Task 7d) the full-video sheet draws neither the outcome (the passes agree on it, and it is a
    public fact of the show) nor the window check (the coder does not check the window): one cut row per pitch."""
    p, r = pair()
    out = exchange_sheet(books, p, r, exchanges=1, qa=1)
    classes = [row["field_class"] for row in out["rows"]]
    assert classes.count("cut") == 1 and "outcome" not in classes and "prepared_window" not in classes
    assert out["rows"][0]["path"] == "pitch:first_decision" and out["sample"]["drawn"]["pitch"] == ["first_decision"]
    assert out["sample"]["population"]["cut"] == 1
    assert "outcome" not in out["sample"]["population"] and "prepared_window" not in out["sample"]["population"]


def test_exchange_sheet_has_one_outcome_row_one_cut_row_and_a_window_row_when_asked(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, exchanges=1, qa=1, field_classes=SHEET_CLASSES)
    rows = out["rows"]
    pitch = [(row["path"], row["field_class"]) for row in rows if row["field_class"] in ("outcome", "cut", "prepared_window")]
    assert sorted(pitch) == [("pitch:any_offer", "outcome"), ("pitch:first_decision", "cut"),
                             ("prepared_window", "prepared_window")]
    by_path = {row["path"]: row for row in rows}
    assert by_path["pitch:any_offer"]["blind_context"] == "the pitch: Did any investor make an offer? Watch to the end."
    assert by_path["pitch:first_decision"]["blind_context"] == "the pitch: When is the first decision spoken?"
    window = by_path["prepared_window"]
    assert window["blind_context"] == "Is the prepared presentation where the page says it is?"
    assert (window["primary"], window["clock_start"], window["clock_end"]) == ("0:05-1:00", "0:05", "1:00")
    # clock order: the pitch rows carry no clock and come first, then the window, then the exchanges
    order = [row["path"] for row in rows]
    assert order[:3] == ["pitch:any_offer", "pitch:first_decision", "prepared_window"]
    assert order[3] == "exchange:1/topic"
    assert out["sample"]["drawn"]["pitch"] == ["any_offer", "first_decision"]
    assert all(out["sample"]["population"][c] == 1 for c in ("outcome", "cut", "prepared_window"))
    assert "closing" not in out["sample"]["population"]


def test_the_cut_rows_passes_differ_beyond_the_decision_tolerance(books):
    def later(d):
        for x in d["decisions"]:
            x["time_s"] += 31
    p, r = pair(mutate_replicate=later)
    row = next(row for row in exchange_sheet(books, p, r)["rows"] if row["field_class"] == "cut")
    assert (row["primary"], row["replicate"], row["reasons"]) == ("1:51", "2:22", ["passes_differ"])
    r_near = pair(mutate_replicate=lambda d: [x.update(time_s=x["time_s"] + 30) for x in d["decisions"]])
    row = next(row for row in exchange_sheet(books, *r_near)["rows"] if row["field_class"] == "cut")
    assert (row["replicate"], row["reasons"]) == ("2:21", [])           # within the 30 s tolerance
    p, r = pair(mutate_replicate=lambda d: d.update(decisions=[]))
    row = next(row for row in exchange_sheet(books, p, r)["rows"] if row["field_class"] == "cut")
    assert (row["primary"], row["replicate"], row["reasons"]) == ("1:51", None, ["passes_differ"])


def test_qa_pairs_carry_six_codes_including_rs_evid_and_answer_coherence(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, qa=10)
    assert [list(pair) for pair in out["sample"]["requested"]["qa_codes"]] == SIX_CODES
    for question, answer in out["sample"]["drawn"]["qa"]:
        paths = {row["path"] for row in out["rows"] if row["field_class"] == "turn_code"}
        assert {f"turn:{question}/code:PR-CONCEPT", f"turn:{question}/code:PR-SEV", f"turn:{answer}/code:RS-RESP",
                f"turn:{answer}/code:RS-EVID", f"turn:{answer}/code:PREP-SUBS",
                f"turn:{answer}/code:PREP-COH"} <= paths


def test_stage_a_and_venture_rows_are_off_by_default(books):
    p, r = pair()
    out = exchange_sheet(books, p, r)
    classes = {row["field_class"] for row in out["rows"]}
    assert not classes & {"stage_a", "venture"}
    assert out["sample"]["requested"]["stage_a"] == out["sample"]["requested"]["venture"] == 0
    assert out["sample"]["drawn"]["stage_a"] == out["sample"]["drawn"]["venture"] == []
    asked = exchange_sheet(books, p, r, stage_a=2, venture=1)
    assert {row["field_class"] for row in asked["rows"]} >= {"stage_a", "venture"}


def test_the_draw_record_lists_only_the_pitch_rows_the_sheet_carries(books):
    p, r = pair()
    out = exchange_sheet(books, p, r, qa=1, field_classes=("outcome", "turn_code"))
    assert [row["path"] for row in out["rows"] if row["field_class"] in ("outcome", "cut")] == ["pitch:any_offer"]
    assert out["sample"]["drawn"]["pitch"] == ["any_offer"]
    assert out["sample"]["population"]["outcome"] == 1 and "cut" not in out["sample"]["population"]
