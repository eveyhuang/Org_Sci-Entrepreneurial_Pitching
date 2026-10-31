import copy
from pathlib import Path

import pytest

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.panel_validate import RESPONSE_FUNCTIONS, validate_record
from tests.panel_fixtures import DURATION_S, question_codes, synthetic_record, turn

INVESTOR_FUNCTIONS = ("question", "challenges", "accepts")
FOUNDER_FUNCTIONS = ("answer", "presents", "repair")
EITHER_ROLE_FUNCTIONS = ("social", "hostile", "demonstrates", "negotiates", "decides", "NONE")


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def check(books, mutate, duration=DURATION_S):
    data = copy.deepcopy(synthetic_record())
    mutate(data)
    return validate_record(PanelRecord.model_validate(data), books, duration)


def has(items, tag):
    return any(i.startswith(f"[{tag}]") for i in items)


def tagged(items, tag):
    return [i for i in items if i.startswith(f"[{tag}]")]


def tn(data, n):
    return next(t for t in data["turns"] if t["turn_id"] == n)


def ex(data, n):
    return next(e for e in data["exchanges"] if e["exchange_id"] == n)


def code(t, name):
    return next(c for c in t["codes"] if c["code"] == name)


def claim(**kw):
    return {"turn_a": 1, "turn_b": 3, "metric": "cac", "note": "", "reconciled": False, "reconciled_turn": None, **kw}


def test_synthetic_record_is_valid_with_no_flags(books):
    report = validate_record(PanelRecord.model_validate(synthetic_record()), books, DURATION_S)
    assert report.errors == () and report.ok
    assert report.flags == ()


def test_every_function_set_is_a_codebook_function(books):
    functions = set(books.turn_functions())
    assert set(INVESTOR_FUNCTIONS + FOUNDER_FUNCTIONS + EITHER_ROLE_FUNCTIONS) == functions
    assert set(RESPONSE_FUNCTIONS) <= functions


# ---- turns: function, addressee, exchange, links ----------------------------------------

def test_unknown_turn_function_is_an_error(books):
    errs = tagged(check(books, lambda d: tn(d, 8).update(function="asks")).errors, "turn.function")
    assert len(errs) == 1 and "turn 8" in errs[0] and "'asks'" in errs[0]


def test_a_function_merged_in_v3_dev9_is_no_longer_a_function(books):
    errs = tagged(check(books, lambda d: tn(d, 13).update(function="agrees")).errors, "turn.function")
    assert errs == ["[turn.function] turn 13: 'agrees' is not a TURN-FUNC value"]


def test_function_none_is_a_valid_function(books):
    def m(d):
        d["turns"].insert(10, turn(20, "S2", 114, 114.5, "NONE", "Asks the crew whether to continue."))
    report = check(books, m)
    assert report.errors == () and report.flags == ()


def test_unknown_addressee_is_an_error_and_tokens_are_not(books):
    errs = tagged(check(books, lambda d: tn(d, 3).update(addressed=["X9", "S1"])).errors, "turn.addressed")
    assert len(errs) == 1 and "turn 3" in errs[0] and "'X9'" in errs[0]
    assert check(books, lambda d: tn(d, 3).update(addressed=["PANEL"])).errors == ()
    report = check(books, lambda d: tn(d, 3).update(addressed=[]))
    assert report.errors == () and report.flags == ()


def test_turn_pointing_at_an_unrecorded_exchange_is_an_error(books):
    errs = tagged(check(books, lambda d: tn(d, 10).update(exchange_id=9)).errors, "turn.exchange")
    assert len(errs) == 1 and "turn 10" in errs[0] and "9" in errs[0]


def test_missing_self_and_forward_links_are_errors(books):
    def m(d):
        tn(d, 3)["responds_to"] = [2, 99, 3, 11]
    errs = tagged(check(books, m).errors, "turn.link")
    assert len(errs) == 3 and all("turn 3" in e for e in errs)
    assert any("99" in e and "does not exist" in e for e in errs)
    assert sum("not an earlier turn" in e for e in errs) == 2


def test_link_order_is_position_in_the_list_not_the_id(books):
    def m(d):
        tn(d, 2)["turn_id"] = 40            # still second in the list, so turn 3 may respond to it
        tn(d, 3)["responds_to"] = [40]
    report = check(books, m)
    assert not has(report.errors, "turn.link") and not has(report.flags, "flag.answer_link")


def test_delayed_cross_investor_link_is_accepted(books):
    def m(d):
        tn(d, 7)["responds_to"] = [5, 2]    # S1 reacts to the answer given to S2 and to his own question
    assert check(books, m).errors == ()


# ---- turns: codes -----------------------------------------------------------------------

def test_question_missing_a_required_code_is_an_error(books):
    def m(d):
        tn(d, 2)["codes"] = [c for c in tn(d, 2)["codes"] if c["code"] != "PR-SEV"]
    errs = tagged(check(books, m).errors, "turn.codes")
    assert len(errs) == 1 and "turn 2" in errs[0] and "PR-SEV" in errs[0] and "absent" in errs[0]


def test_answer_with_no_codes_misses_all_seven(books):
    errs = tagged(check(books, lambda d: tn(d, 3).update(codes=[])).errors, "turn.codes")
    assert len(errs) == 7


def test_code_on_a_turn_of_another_function_is_an_error(books):
    def on_answer(d):
        tn(d, 3)["codes"].append({"code": "PR-CONCEPT", "value": "sales", "missing": "none"})
    errs = tagged(check(books, on_answer).errors, "turn.codes")
    assert len(errs) == 1 and "PR-CONCEPT" in errs[0] and "'question'" in errs[0]

    def on_reaction(d):
        tn(d, 13)["codes"] = question_codes()
    assert len(tagged(check(books, on_reaction).errors, "turn.codes")) == 2

    def exchange_level_code(d):
        tn(d, 13)["codes"] = [{"code": "EX-KIND", "value": "social", "missing": "none"}]
    assert len(tagged(check(books, exchange_level_code).errors, "turn.codes")) == 1


def test_duplicate_and_unknown_codes_are_errors(books):
    def m(d):
        tn(d, 2)["codes"] += [{"code": "PR-SEV", "value": "2", "missing": "none"},
                              {"code": "ZZ-NOPE", "value": "1", "missing": "none"}]
    errs = tagged(check(books, m).errors, "turn.codes")
    assert any("duplicate" in e for e in errs) and any("ZZ-NOPE" in e for e in errs)


def test_invalid_code_values_are_errors(books):
    def m(d):
        code(tn(d, 2), "PR-SEV")["value"] = "4"
        code(tn(d, 3), "RS-EVID")["value"] = "yes"
        code(tn(d, 3), "RS-RESP")["value"] = "evades"
        code(tn(d, 5), "PREP-SUBS")["value"] = "high"
    assert len(tagged(check(books, m).errors, "turn.codes")) == 4


def test_missing_code_value_with_reason_is_accepted(books):
    def m(d):
        c = code(tn(d, 3), "PREP-SUBS")
        c["value"], c["missing"] = None, "too_short"
    report = check(books, m)
    assert report.errors == () and report.flags == ()


def test_value_differing_only_in_case_is_accepted_and_flagged(books):
    def m(d):
        code(tn(d, 5), "RS-CONCEPT")["value"] = "none"
        code(tn(d, 2), "PR-CONCEPT")["value"] = "Operating_Efficiency"
        code(tn(d, 3), "RS-EVID")["value"] = "True"
    report = check(books, m)
    assert report.errors == ()
    assert "[flag.value_case] turn 5: RS-CONCEPT 'none' → 'NONE'" in report.flags
    assert "[flag.value_case] turn 2: PR-CONCEPT 'Operating_Efficiency' → 'operating_efficiency'" in report.flags
    assert "[flag.value_case] turn 3: RS-EVID 'True' → 'true'" in report.flags
    assert len(tagged(report.flags, "flag.value_case")) == 3


def test_wrong_value_is_still_an_error_and_not_a_case_flag(books):
    report = check(books, lambda d: code(tn(d, 5), "RS-CONCEPT").update(value="promotion"))
    assert has(report.errors, "turn.codes") and not has(report.flags, "flag.value_case")


# ---- turns: answers and roles -----------------------------------------------------------

def test_answer_without_a_question_link_is_an_error(books):
    report = check(books, lambda d: tn(d, 3).update(responds_to=[]))
    flagged = tagged(report.flags, "flag.answer_link")
    assert len(flagged) == 1 and "turn 3" in flagged[0]
    assert report.errors == ()


def test_answer_to_a_challenges_turn_is_accepted(books):
    def m(d):
        tn(d, 4).update(function="challenges", codes=[], responds_to=[3])
    report = check(books, m)
    assert synthetic_record()["turns"][4]["responds_to"] == [4]
    assert report.errors == ()


def test_answer_to_another_function_is_an_error(books):
    def m(d):
        tn(d, 4).update(function="accepts", codes=[], responds_to=[3])
    report = check(books, m)
    assert len(tagged(report.flags, "flag.answer_link")) == 1 and report.errors == ()     # turn 5 answers turn 4


def test_answer_linked_to_a_reaction_and_a_question_is_accepted(books):
    def m(d):
        tn(d, 15)["responds_to"] = [13, 14]
    assert not has(check(books, m).flags, "flag.answer_link")


@pytest.mark.parametrize("function", INVESTOR_FUNCTIONS)
def test_investor_functions_need_an_investor_speaker(books, function):
    def m(d):
        tn(d, 7).update(function=function, speaker="F1", codes=[])
    report = check(books, m)
    flagged = tagged(report.flags, "flag.turn_role")
    assert len(flagged) == 1 and "turn 7" in flagged[0] and "investor" in flagged[0]
    assert not any("role" in e for e in report.errors)


@pytest.mark.parametrize("function", FOUNDER_FUNCTIONS)
def test_founder_functions_need_a_founder_speaker(books, function):
    def m(d):
        tn(d, 6).update(function=function, speaker="S2", codes=[])
    report = check(books, m)
    flagged = tagged(report.flags, "flag.turn_role")
    assert len(flagged) == 1 and "turn 6" in flagged[0] and "founder" in flagged[0]
    assert not any("role" in e for e in report.errors)


@pytest.mark.parametrize("function", EITHER_ROLE_FUNCTIONS)
@pytest.mark.parametrize("speaker", ["F1", "S2"])
def test_other_functions_are_open_to_both_roles(books, function, speaker):
    def m(d):
        tn(d, 6).update(function=function, speaker=speaker, codes=[])
    report = check(books, m)
    assert not has(report.flags, "flag.turn_role") and not any("role" in e for e in report.errors)


def test_a_host_is_neither_investor_nor_founder(books):
    def m(d):
        d["actors"].append({"actor_id": "H1", "role": "host", "label": "Host"})
        tn(d, 8).update(function="question", speaker="H1", codes=question_codes())
    assert len(tagged(check(books, m).flags, "flag.turn_role")) == 1


def test_unknown_speaker_is_a_speaker_error_not_a_role_error(books):
    report = check(books, lambda d: tn(d, 2).update(speaker="X9"))
    assert has(report.errors, "turn.speaker") and not has(report.flags, "flag.turn_role")


# ---- exchanges --------------------------------------------------------------------------

def test_duplicate_ids_are_errors(books):
    def m(d):
        d["actors"].append(dict(d["actors"][0]))
        d["turns"][1]["turn_id"] = 1
        d["exchanges"][1]["exchange_id"] = 1
        d["decisions"][1]["decision_id"] = 1
    errs = check(books, m).errors
    for tag in ("actor.id", "turn.id", "exchange.id", "decision.id"):
        assert has(errs, tag), tag


@pytest.mark.parametrize("kind", ["NONE", "chat", "none"])
def test_exchange_kind_must_be_a_real_kind(books, kind):
    errs = tagged(check(books, lambda d: ex(d, 2).update(kind=kind)).errors, "exchange.kind")
    assert len(errs) == 1 and "exchange 2" in errs[0]


def test_exchange_topic_rules(books):
    unknown = tagged(check(books, lambda d: ex(d, 1).update(topic="pricing")).errors, "exchange.topic")
    assert len(unknown) == 1 and "exchange 1" in unknown[0] and "'pricing'" in unknown[0]
    social = tagged(check(books, lambda d: ex(d, 2).update(topic="product")).errors, "exchange.topic")
    assert len(social) == 1 and "exchange 2" in social[0]
    bare = tagged(check(books, lambda d: ex(d, 3).update(topic="NONE")).errors, "exchange.topic")
    assert len(bare) == 1 and "exchange 3" in bare[0]
    assert check(books, lambda d: ex(d, 3).update(topic="other")).errors == ()


def test_exchange_no_turn_points_to_is_an_error(books):
    def m(d):
        d["exchanges"].append({"exchange_id": 5, "kind": "demonstration", "topic": "NONE", "label": "tasting"})
    errs = tagged(check(books, m).errors, "exchange.empty")
    assert len(errs) == 1 and "exchange 5" in errs[0]


def test_exchanges_carry_no_investor_list_and_no_closing_rule(books):
    import pipeline.stage_b.panel_validate as module
    source = Path(module.__file__).read_text()
    for tag in ("exchange.investor", "exchange.missing_investor", "flag.closing_unshown",
                "flag.closing_after_decision"):
        assert tag not in source, tag


# ---- kept rules: ids, times, decisions, venture ------------------------------------------

def test_turn_speaker_and_order_errors(books):
    def m(d):
        d["turns"][1]["speaker"] = "X9"
        d["turns"][3]["start_s"] = 63.0
        d["turns"][3]["end_s"] = 64.0
    errs = check(books, m).errors
    assert has(errs, "turn.speaker") and has(errs, "turn.order")


def test_times_beyond_duration_are_errors(books):
    def m(d):
        d["turns"][-1]["end_s"] = 400.0
        d["prepared_spans"][0]["end_s"] = 400.0
    errs = check(books, m).errors
    assert has(errs, "turn.time") and has(errs, "span.time")


def test_reversed_turn_and_empty_span_are_errors(books):
    def m(d):
        d["turns"][0]["end_s"] = 1.0
        d["excluded_spans"][0]["end_s"] = 0.0
    errs = check(books, m).errors
    assert has(errs, "turn.time") and has(errs, "span.time")


def test_early_exit_does_not_block_later_turns(books):
    assert synthetic_record()["decisions"][0]["kind"] == "exit"
    assert check(books, lambda d: None).errors == ()


def test_founder_decision_kinds_need_a_founder(books):
    def m(d):
        d["decisions"][3]["actor"] = "S1"
        d["decisions"][1]["actor"] = "F1"
    assert len(tagged(check(books, m).errors, "decision.actor")) == 2


def test_decision_by_an_unknown_actor_is_an_error(books):
    errs = tagged(check(books, lambda d: d["decisions"][0].update(actor="X9")).errors, "decision.actor")
    assert len(errs) == 1 and "decision 1" in errs[0] and "unknown actor 'X9'" in errs[0]


def test_decision_kind_time_and_turn_errors(books):
    def m(d):
        d["decisions"][0]["kind"] = "bids"
        d["decisions"][1]["time_s"] = 400.0
        d["decisions"][2]["turn_id"] = 99
    errs = check(books, m).errors
    assert has(errs, "decision.kind") and has(errs, "decision.time") and has(errs, "decision.turn")


def test_offer_moves_need_an_offer_id(books):
    for index, kind in ((1, "offer"), (2, "joins_offer"), (3, "founder_accepts")):
        def m(d, index=index):
            d["decisions"][index]["offer_id"] = None
        errs = tagged(check(books, m).errors, "decision.offer")
        assert len(errs) >= 1 and any(f"decision {index + 1}" in e for e in errs), kind
    for kind, actor in (("revises_offer", "S1"), ("withdraws_offer", "S1"), ("founder_rejects", "F1"),
                        ("founder_counters", "F1")):
        def m(d, kind=kind, actor=actor):
            d["decisions"].append({"decision_id": 5, "actor": actor, "kind": kind, "time_s": 171.0, "turn_id": None,
                                   "offer_id": None, "terms": "", "note": ""})
        assert has(check(books, m).errors, "decision.offer"), kind


def test_offer_id_must_belong_to_an_offer_decision(books):
    errs = tagged(check(books, lambda d: d["decisions"][2].update(offer_id="O9")).errors, "decision.offer")
    assert len(errs) == 1 and "decision 3" in errs[0] and "O9" in errs[0]


def test_exit_with_null_offer_id_is_not_an_offer_error(books):
    assert synthetic_record()["decisions"][0]["offer_id"] is None
    assert not has(check(books, lambda d: None).errors, "decision.offer")


def test_venture_values_are_checked(books):
    def m(d):
        d["venture"]["primary_industry"] = "tech"
        d["venture"]["secondary_industries"] = ["gadgets"]
        d["venture"]["product_form"] = "thing"
        d["venture"]["customer_type"] = "kids"
    assert len(tagged(check(books, m).errors, "venture.value")) == 4


def test_venture_value_differing_only_in_case_is_accepted_and_flagged(books):
    def m(d):
        d["venture"]["primary_industry"] = "Home_Household"
        d["venture"]["customer_type"] = "none"
    report = check(books, m)
    assert report.errors == ()
    assert "[flag.value_case] venture primary_industry: 'Home_Household' → 'home_household'" in report.flags
    assert "[flag.value_case] venture customer_type: 'none' → 'NONE'" in report.flags
    assert has(report.flags, "flag.venture_unknown")


# ---- stage A and claims -----------------------------------------------------------------

def test_stage_a_must_cover_exactly_the_three_codes(books):
    def drop(d):
        d["stage_a"].pop()
    def dup(d):
        d["stage_a"][1] = dict(d["stage_a"][0])
    def extra(d):
        d["stage_a"].append({"code": "made_up", "value": 3, "uncodable": False, "moment_s": None, "note": ""})
    assert has(check(books, drop).errors, "stage_a.codes")
    assert has(check(books, dup).errors, "stage_a.codes")
    errs = tagged(check(books, extra).errors, "stage_a.codes")
    assert len(errs) == 1 and "extra ['made_up']" in errs[0]


def test_stage_a_moment_must_be_inside_a_prepared_span(books):
    errs = tagged(check(books, lambda d: d["stage_a"][0].update(moment_s=90.0)).errors, "stage_a.moment")
    assert len(errs) == 1 and "pitch_positivity" in errs[0] and "90.0" in errs[0]
    assert check(books, lambda d: d["stage_a"][0].update(moment_s=61.5)).errors == ()      # within the 2 s tolerance
    assert has(check(books, lambda d: d["stage_a"][0].update(moment_s=62.5)).errors, "stage_a.moment")
    assert has(check(books, lambda d: d["stage_a"][0].update(moment_s=2.5)).errors, "stage_a.moment")


def test_stage_a_needs_no_moment(books):
    def m(d):
        for item in d["stage_a"]:
            item["moment_s"] = None
    report = check(books, m)
    assert report.errors == () and report.flags == ()


def test_no_prepared_span_requires_uncodable_stage_a(books):
    def rated(d):
        d["prepared_spans"] = []
        for i in d["stage_a"]:
            i["moment_s"] = None
    def uncodable(d):
        d["prepared_spans"] = []
        for i in d["stage_a"]:
            i.update(value=None, uncodable=True, moment_s=None)
        for c in d["prepared"]:
            c.update(value=None, missing="not_applicable")
    assert len(tagged(check(books, rated).errors, "stage_a.no_prepared")) == 3
    report = check(books, uncodable)
    assert report.errors == () and has(report.flags, "flag.no_prepared")


def test_claim_turns_are_checked(books):
    def ok(d):
        d["claim_conflicts"] = [claim(turn_a=1, turn_b=3, reconciled=True, reconciled_turn=6), claim(turn_a=3, turn_b=15)]
    assert check(books, ok).errors == ()
    for name, c, n in (("unknown turns", claim(turn_a=98, turn_b=99), 2),
                       ("same turn", claim(turn_a=3, turn_b=3), 1),
                       ("reversed", claim(turn_a=5, turn_b=3), 1),
                       ("unknown reconciling turn", claim(reconciled=True, reconciled_turn=99), 1),
                       ("reconciled in the later claim itself", claim(reconciled=True, reconciled_turn=3), 1),
                       ("reconciled before the later claim", claim(reconciled=True, reconciled_turn=2), 1)):
        report = check(books, lambda d, c=c: d.update(claim_conflicts=[c]))
        assert len(tagged(report.errors, "claim.turn")) == n, name
        assert len(report.errors) == n, name


# ---- soft flags -------------------------------------------------------------------------

def test_unanswered_question_flag(books):
    def m(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] not in (15, 16)]
    report = check(books, m)
    flagged = tagged(report.flags, "flag.unanswered_question")
    assert report.errors == () and len(flagged) == 1 and "turn 14" in flagged[0]


def test_question_answered_only_by_a_non_answer_turn_is_unanswered(books):
    def m(d):
        tn(d, 5).update(function="repair", codes=[])       # turn 6 still responds to question 4
    flagged = tagged(check(books, m).flags, "flag.unanswered_question")
    assert len(flagged) == 1 and "turn 4" in flagged[0]


@pytest.mark.parametrize("function", RESPONSE_FUNCTIONS)
def test_every_response_function_is_flagged_when_unlinked(books, function):
    assert function in books.turn_functions()
    speaker = "F1" if function in FOUNDER_FUNCTIONS else "S1"

    def m(d):
        tn(d, 7).update(function=function, speaker=speaker, responds_to=[])
    flagged = tagged(check(books, m).flags, "flag.unlinked_response")
    assert len(flagged) == 1 and "turn 7" in flagged[0] and function in flagged[0]


def test_response_functions_are_the_three_named_ones():
    assert RESPONSE_FUNCTIONS == ("accepts", "challenges", "repair")


def test_questions_and_other_functions_are_not_unlinked_responses(books):
    assert synthetic_record()["turns"][1]["responds_to"] == []
    def m(d):
        tn(d, 8)["responds_to"] = []
        tn(d, 17)["responds_to"] = []
    assert not has(check(books, m).flags, "flag.unlinked_response")


def test_orphan_turn_flag(books):
    def m(d):
        tn(d, 11)["exchange_id"] = None
        tn(d, 12)["exchange_id"] = None
    report = check(books, m)
    flagged = tagged(report.flags, "flag.orphan_turn")
    assert len(flagged) == 2 and "turn 11" in flagged[0] and "turn 12" in flagged[1]
    assert not has(check(books, lambda d: None).flags, "flag.orphan_turn")       # turns 1 and 10 have no exchange


def test_kind_function_mismatch_flag(books):
    def question_in_social(d):
        tn(d, 8).update(function="question", codes=question_codes())
        tn(d, 9).update(function="answer", codes=tn(d, 3)["codes"])
    flagged = tagged(check(books, question_in_social).flags, "flag.kind_function_mismatch")
    assert len(flagged) == 2 and "turn 8" in flagged[0] and "turn 9" in flagged[1]

    def question_in_demonstration(d):
        question_in_social(d)
        ex(d, 2)["kind"] = "demonstration"
    assert len(tagged(check(books, question_in_demonstration).flags, "flag.kind_function_mismatch")) == 2

    def joke_in_substantive(d):
        tn(d, 8)["exchange_id"] = 3
        tn(d, 9)["exchange_id"] = 3
        d["exchanges"] = [e for e in d["exchanges"] if e["exchange_id"] != 2]
    report = check(books, joke_in_substantive)
    flagged = tagged(report.flags, "flag.kind_function_mismatch")
    assert report.errors == () and len(flagged) == 2 and "turn 8" in flagged[0] and "social" in flagged[0]

    def question_in_negotiation(d):
        tn(d, 14)["exchange_id"] = 4
        tn(d, 15)["exchange_id"] = 4
        tn(d, 16)["exchange_id"] = 4
    assert not has(check(books, question_in_negotiation).flags, "flag.kind_function_mismatch")


def test_single_turn_exchange_flag(books):
    def m(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] not in (12, 13)]
    report = check(books, m)
    flagged = tagged(report.flags, "flag.single_turn_exchange")
    assert report.errors == () and len(flagged) == 1 and "exchange 3" in flagged[0]

    def social(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 9]
    assert not has(check(books, social).flags, "flag.single_turn_exchange")


def test_unknown_addressee_flag(books):
    flagged = tagged(check(books, lambda d: tn(d, 2).update(addressed=["UNKNOWN"])).flags, "flag.unknown_addressee")
    assert len(flagged) == 1 and "turn 2" in flagged[0]


def test_record_level_flags(books):
    def m(d):
        d["decisions"] = [x for x in d["decisions"] if x["actor"] != "S3"]
        d["gaps"] = [{"at_s": 90.0, "note": "cut"}]
        d["source_complete"] = False
        d["venture"]["customer_type"] = "NONE"
    report = check(books, m)
    for tag in ("no_decision", "gap", "incomplete_source", "venture_unknown"):
        assert has(report.flags, f"flag.{tag}"), tag


def test_decision_turn_flags(books):
    def null_turn(d):
        d["decisions"][0]["turn_id"] = None
    def other_speaker(d):
        d["decisions"][0]["turn_id"] = 11         # S2 speaks turn 11; the decision is S3's
        d["decisions"][0]["time_s"] = 116.0
    def outside(d):
        d["decisions"][0]["time_s"] = 125.0       # turn 10 is 111-114
    def within_tolerance(d):
        d["decisions"][0]["time_s"] = 115.5
    for m in (null_turn, other_speaker, outside):
        report = check(books, m)
        flagged = tagged(report.flags, "flag.decision_turn")
        assert len(flagged) == 1 and "decision 1" in flagged[0], m.__name__
        assert report.errors == ()
    assert not has(check(books, within_tolerance).flags, "flag.decision_turn")


def test_short_coverage_flag(books):
    assert has(check(books, lambda d: None, duration=240.0).flags, "flag.short_coverage")     # 172 < 0.75 * 240
    assert not has(check(books, lambda d: None, duration=229.0).flags, "flag.short_coverage")  # 172 >= 171.75


def test_removed_rules_are_gone(books):
    def m(d):
        for t in d["turns"]:
            t["addressed"] = []
        for i in d["stage_a"]:
            i["moment_s"] = None
    report = check(books, m)
    assert report.errors == () and report.flags == ()
    for tag in ("event.", "flag.empty_addressed", "flag.uncertain_link", "stage_a.evidence", "claim.time", "claim.event"):
        assert not any(tag in line for line in report.errors + report.flags)


# ---- structure labels differing only in case ---------------------------------------------

def _miscased(d):
    tn(d, 2)["function"] = "Question"
    tn(d, 3)["function"] = "ANSWER"
    ex(d, 1).update(kind="Substantive", topic="Operating_Efficiency")
    ex(d, 2)["topic"] = "none"
    d["decisions"][0]["kind"] = "Exit"
    d["decisions"][1]["kind"] = "OFFER"


def test_structure_labels_differing_only_in_case_are_accepted_and_flagged(books):
    report = check(books, _miscased)
    assert report.errors == ()
    assert set(report.flags) == {
        "[flag.value_case] turn 2: function 'Question' → 'question'",
        "[flag.value_case] turn 3: function 'ANSWER' → 'answer'",
        "[flag.value_case] exchange 1: kind 'Substantive' → 'substantive'",
        "[flag.value_case] exchange 1: topic 'Operating_Efficiency' → 'operating_efficiency'",
        "[flag.value_case] exchange 2: topic 'none' → 'NONE'",
        "[flag.value_case] decision 1: kind 'Exit' → 'exit'",
        "[flag.value_case] decision 2: kind 'OFFER' → 'offer'",
    }
    assert len(report.flags) == 7


def test_rules_run_on_the_canonical_labels(books):
    # a miscased question still needs its codes, its investor speaker and an answer
    def m(d):
        tn(d, 2).update(function="Question", codes=[], speaker="F1")
        tn(d, 3)["responds_to"] = []
    report = check(books, m)
    assert len(tagged(report.errors, "turn.codes")) == 2 and has(report.flags, "flag.turn_role")
    assert has(report.flags, "flag.unanswered_question")
    # a miscased offer still needs its offer id; a miscased exit still counts as S3's decision
    def offer(d):
        d["decisions"][1].update(kind="Offer", offer_id=None)
        d["decisions"][0]["kind"] = "EXIT"
    report = check(books, offer)
    assert has(report.errors, "decision.offer") and not has(report.flags, "flag.no_decision")


def test_wrong_structure_labels_are_still_errors_and_not_case_flags(books):
    def m(d):
        tn(d, 8)["function"] = "jokes"
        ex(d, 2)["kind"] = "none"                 # NONE in any case is not a kind a recorded exchange can have
        ex(d, 1)["topic"] = "pricing"
        d["decisions"][0]["kind"] = "bids"
    report = check(books, m)
    for tag in ("turn.function", "exchange.kind", "exchange.topic", "decision.kind"):
        assert has(report.errors, tag), tag
    assert tagged(report.flags, "flag.value_case") == ["[flag.value_case] exchange 2: kind 'none' → 'NONE'"]


# ---- v3-dev5 ----------------------------------------------------------------------------

def test_an_accepts_turn_by_a_founder_is_flagged(books):
    def m(d):
        d["turns"].insert(13, turn(20, "F1", 133, 133.5, "accepts", "Says that is right.", ["S2"], 3, [13]))
    report = check(books, m)
    assert report.errors == ()
    assert report.flags == ("[flag.turn_role] turn 20: accepts is expected from an investor, got founder 'F1'",)


def test_labels_and_values_are_stripped_before_comparison(books):
    def m(d):
        tn(d, 2)["function"] = " question "
        ex(d, 1)["kind"] = "substantive\n"
        ex(d, 1)["topic"] = " Operating_Efficiency"
        d["decisions"][0]["kind"] = " exit"
        code(tn(d, 2), "PR-CONCEPT")["value"] = " sales "
        code(tn(d, 3), "RS-EVID")["value"] = "True "
        d["venture"]["product_form"] = " physical_good"
    report = check(books, m)
    assert report.errors == ()
    assert len(tagged(report.flags, "flag.value_case")) == 7 and len(report.flags) == 7
    assert "[flag.value_case] turn 2: function ' question ' → 'question'" in report.flags
    assert "[flag.value_case] turn 2: PR-CONCEPT ' sales ' → 'sales'" in report.flags


def test_decision_spoken_in_a_turn_that_is_not_decides_is_flagged(books):
    report = check(books, lambda d: tn(d, 18).update(function="negotiates"))
    flagged = tagged(report.flags, "flag.decision_function")
    assert report.errors == () and len(flagged) == 1 and "decision 3" in flagged[0] and "negotiates" in flagged[0]
    assert [f[:f.index("]") + 1] for f in report.flags] == ["[flag.decision_function]"]


def test_decides_turn_named_by_no_decision_is_flagged(books):
    def m(d):
        d["decisions"][2]["turn_id"] = None
    report = check(books, m)
    flagged = tagged(report.flags, "flag.decides_without_decision")
    assert len(flagged) == 1 and "turn 18" in flagged[0]


def test_exit_turn_inside_an_exchange_is_flagged_unless_it_is_the_negotiation(books):
    for exchange_id, n in ((2, 1), (3, 1), (4, 0)):
        def m(d, exchange_id=exchange_id):
            tn(d, 10)["exchange_id"] = exchange_id
        report = check(books, m)
        flagged = tagged(report.flags, "flag.exit_in_exchange")
        assert report.errors == () and len(flagged) == n, exchange_id
        if n:
            assert "decision 1" in flagged[0] and "turn 10" in flagged[0]
            assert len(report.flags) == 1


def test_offer_moves_outside_the_negotiation_are_flagged(books):
    def none(d):
        tn(d, 19)["exchange_id"] = None
    def elsewhere(d):
        tn(d, 19)["exchange_id"] = 3
    for m, decision in ((none, "decision 4"), (elsewhere, "decision 4")):
        report = check(books, m)
        flagged = tagged(report.flags, "flag.offer_outside_negotiation")
        assert report.errors == () and len(flagged) == 1 and decision in flagged[0], decision
    assert not has(check(books, lambda d: None).flags, "flag.offer_outside_negotiation")


def test_every_offer_move_kind_is_checked_for_the_negotiation(books):
    for kind, actor in (("revises_offer", "S1"), ("withdraws_offer", "S1"), ("founder_rejects", "F1"),
                        ("founder_counters", "F1"), ("joins_offer", "S2")):
        def m(d, kind=kind, actor=actor):
            d["turns"].append(turn(20, actor, 173, 175, "decides", "Speaks again.", ["PANEL"]))
            d["decisions"].append({"decision_id": 5, "actor": actor, "kind": kind, "time_s": 173.0, "turn_id": 20,
                                   "offer_id": "O1", "terms": "", "note": ""})
        flagged = tagged(check(books, m).flags, "flag.offer_outside_negotiation")
        assert len(flagged) == 1 and "decision 5" in flagged[0], kind


def test_more_than_one_negotiation_exchange_is_flagged(books):
    def m(d):
        tn(d, 19)["exchange_id"] = 5
        d["exchanges"].append({"exchange_id": 5, "kind": "negotiation", "topic": "NONE", "label": "acceptance"})
    report = check(books, m)
    flagged = tagged(report.flags, "flag.multiple_negotiations")
    assert report.errors == () and len(flagged) == 1 and "4" in flagged[0] and "5" in flagged[0]
    assert len(report.flags) == 1


# ---- v3-dev6 ----------------------------------------------------------------------------

def test_normalization_notes_become_soft_flags(books):
    record = PanelRecord.model_validate(synthetic_record())
    notes = ["turns[9].codes PREP-COH: value '4' dropped because missing is 'not_applicable'", "stage_a[2] x: y"]
    report = validate_record(record, books, DURATION_S, normalizations=notes)
    assert report.errors == ()
    assert report.flags == tuple(f"[flag.value_with_missing] {n}" for n in notes)
    assert validate_record(record, books, DURATION_S).flags == ()
    assert validate_record(record, books, DURATION_S, normalizations=[]).flags == ()


def test_prepared_span_without_a_presents_turn_is_flagged(books):
    report = check(books, lambda d: tn(d, 1).update(function="NONE"))
    assert report.errors == () and tagged(report.flags, "flag.no_presents_turn") == [
        "[flag.no_presents_turn] there is a prepared span but no presents turn"]
    def no_prepared(d):
        tn(d, 1)["function"] = "NONE"
        d["prepared_spans"] = []
        for i in d["stage_a"]:
            i.update(value=None, uncodable=True, moment_s=None)
    assert not has(check(books, no_prepared).flags, "flag.no_presents_turn")
    assert not has(check(books, lambda d: tn(d, 1).update(function="Presents")).flags, "flag.no_presents_turn")


def test_demonstrates_turn_with_no_exchange_is_flagged(books):
    def m(d, exchange_id):
        d["turns"].insert(10, turn(20, "F1", 114, 114.5, "demonstrates", "Hands a container to each investor.",
                                   ["PANEL"], exchange_id))
    report = check(books, lambda d: m(d, None))
    flagged = tagged(report.flags, "flag.demonstration_without_exchange")
    assert report.errors == () and len(flagged) == 1 and "turn 20" in flagged[0] and len(report.flags) == 1
    assert not has(check(books, lambda d: m(d, 2)).flags, "flag.demonstration_without_exchange")


# ---- v3-dev7 ----------------------------------------------------------------------------

def test_framing_codes_are_gone_and_concepts_are_required(books):
    def old_codes(d):
        tn(d, 2)["codes"] = [{"code": "PR-FRAME", "value": "promotion", "missing": "none"},
                             {"code": "PR-SEV", "value": "1", "missing": "none"}]
    errs = tagged(check(books, old_codes).errors, "turn.codes")
    assert len(errs) == 2 and any("PR-FRAME is not a panel code" in e for e in errs)
    assert any("required code PR-CONCEPT is absent" in e for e in errs)
    def no_answer_concept(d):
        tn(d, 3)["codes"] = [c for c in tn(d, 3)["codes"] if c["code"] != "RS-CONCEPT"]
    assert any("RS-CONCEPT" in e for e in tagged(check(books, no_answer_concept).errors, "turn.codes"))
    for concept in books.panel.by_id("PR-CONCEPT").values:
        if concept == "other":
            continue                    # other also needs a name; see the v3-dev8 tests below
        assert check(books, lambda d, concept=concept: code(tn(d, 2), "PR-CONCEPT").update(value=concept)).errors == ()


# ---- v3-dev8: other, named ----------------------------------------------------------------

def test_named_other_concept_is_accepted_and_flagged_for_review(books):
    from tests.panel_fixtures import set_other
    def m(d):
        set_other(tn(d, 11), "founder's own investment")
        set_other(tn(d, 12), "money the founder put in")
        ex(d, 3).update(topic="other", label="founder's own money")
    report = check(books, m)
    assert report.errors == ()
    assert report.flags == (
        "[flag.other_concept] turn 11: PR-CONCEPT other is \"founder's own investment\"",
        "[flag.other_concept] turn 12: RS-CONCEPT other is \"money the founder put in\"",
        "[flag.other_concept] exchange 3: EX-TOPIC other is \"founder's own money\"",
    )


def test_other_concept_without_a_name_is_an_error(books):
    from tests.panel_fixtures import set_other
    for name in (None, "", "   "):
        report = check(books, lambda d, name=name: set_other(tn(d, 11), name))
        errs = tagged(report.errors, "turn.other_concept")
        assert len(errs) == 1 and "turn 11" in errs[0] and "PR-CONCEPT" in errs[0], repr(name)
        assert len(report.errors) == 1 and len(tagged(report.flags, "flag.other_concept")) == 1
    errs = tagged(check(books, lambda d: set_other(tn(d, 12), None)).errors, "turn.other_concept")
    assert len(errs) == 1 and "turn 12" in errs[0] and "RS-CONCEPT" in errs[0]
    assert check(books, lambda d: code(tn(d, 11), "PR-CONCEPT").update(value="Other")).errors != ()   # case-folded first


def test_other_concept_given_without_an_other_code_is_flagged(books):
    for target in (11, 13, 1):          # a question with a listed concept, a reaction, the presentation
        report = check(books, lambda d, target=target: tn(d, target).update(other_concept="shelf life"))
        flagged = tagged(report.flags, "flag.other_concept_unused")
        assert report.errors == () and len(flagged) == 1 and f"turn {target}" in flagged[0] and "shelf life" in flagged[0]
        assert len(report.flags) == 1
    assert len(tagged(check(books, lambda d: tn(d, 11).update(other_concept="")).flags,
                      "flag.other_concept_unused")) == 1


def test_exchange_topic_other_needs_a_label(books):
    for label in ("", "  "):
        report = check(books, lambda d, label=label: ex(d, 3).update(topic="other", label=label))
        errs = tagged(report.errors, "exchange.topic")
        assert len(errs) == 1 and "exchange 3" in errs[0] and "label" in errs[0] and len(report.errors) == 1
    assert check(books, lambda d: ex(d, 3).update(label="")).errors == ()          # a listed topic needs no label
    assert has(check(books, lambda d: ex(d, 2).update(topic="other")).errors, "exchange.topic")   # social: must be NONE


def test_other_and_none_are_valid_concepts_and_the_old_topics_are_not(books):
    assert check(books, lambda d: code(tn(d, 2), "PR-CONCEPT").update(value="NONE")).flags == ()
    for old in ("unit_economics", "sales_traction", "product", "valuation_deal"):
        assert has(check(books, lambda d, old=old: ex(d, 1).update(topic=old)).errors, "exchange.topic"), old
    for concept in books.exchange_topics():
        if concept not in ("NONE", "other"):
            assert check(books, lambda d, c=concept: ex(d, 1).update(topic=c)).errors == (), concept


def test_founder_gender_is_checked_like_the_other_venture_values(books):
    errs = tagged(check(books, lambda d: d["venture"].update(founder_gender="men")).errors, "venture.value")
    assert len(errs) == 1 and "founder_gender" in errs[0] and "VC-GENDER" in errs[0]
    for value in ("all_women", "all_men", "mixed"):
        report = check(books, lambda d, value=value: d["venture"].update(founder_gender=value))
        assert report.errors == () and report.flags == (), value
    unknown = check(books, lambda d: d["venture"].update(founder_gender="NONE"))
    assert unknown.errors == () and unknown.flags == ("[flag.venture_unknown] founder_gender is NONE",)
    cased = check(books, lambda d: d["venture"].update(founder_gender=" All_Women"))
    assert cased.errors == () and cased.flags == (
        "[flag.value_case] venture founder_gender: ' All_Women' → 'all_women'",)


def test_unnamed_other_is_flagged_as_unnamed_and_the_clean_fixture_has_no_other(books):
    from tests.panel_fixtures import set_other
    clean = check(books, lambda d: None)
    assert clean.errors == () and clean.flags == ()
    def m(d):
        set_other(tn(d, 11), None)
        ex(d, 3).update(topic="other", label="")
    report = check(books, m)
    assert tagged(report.flags, "flag.other_concept") == [
        "[flag.other_concept] turn 11: PR-CONCEPT other is unnamed",
        "[flag.other_concept] exchange 3: EX-TOPIC other is unnamed"]
    assert [e.split("]")[0] for e in report.errors] == ["[turn.other_concept", "[exchange.topic"]


def _reenter(d, offer_id="O2"):
    d["decisions"].append({"decision_id": 5, "actor": "S3", "kind": "re_entry", "time_s": 175.0, "turn_id": None,
                           "offer_id": offer_id, "terms": "$200,000 for 25%", "note": "Back with an offer."})
    d["decisions"].append({"decision_id": 6, "actor": "F1", "kind": "founder_accepts", "time_s": 180.0,
                           "turn_id": None, "offer_id": offer_id, "terms": "", "note": "Accepts."})


def test_founder_reply_on_an_offer_id_created_by_a_re_entry_is_accepted(books):
    assert not has(check(books, _reenter).errors, "decision.offer")


def test_founder_reply_on_a_re_entry_without_an_offer_id_is_still_an_error(books):
    errs = tagged(check(books, lambda d: _reenter(d, None)).errors, "decision.offer")
    assert errs and all("founder_accepts" in e for e in errs)


# ---- v3-dev9: the prepared presentation codes -------------------------------------------

def prep(data, name):
    return next(c for c in data["prepared"] if c["code"] == name)


def test_every_prepared_code_is_recorded_exactly_once(books):
    def drop(d):
        d["prepared"] = [c for c in d["prepared"] if c["code"] != "PRES-COH"]
    def dup(d):
        d["prepared"].append(dict(prep(d, "PRES-SUBS")))
    def extra(d):
        d["prepared"].append({"code": "PREP-SUBS", "value": "3", "missing": "none"})
    assert tagged(check(books, drop).errors, "prepared") == ["[prepared] PRES-COH is not recorded"]
    assert tagged(check(books, dup).errors, "prepared") == ["[prepared] PRES-SUBS is recorded twice"]
    assert tagged(check(books, extra).errors, "prepared") == ["[prepared] PREP-SUBS is not a prepared presentation code"]


def test_prepared_values_are_checked(books):
    errs = tagged(check(books, lambda d: prep(d, "PRES-SUBS").update(value="6")).errors, "prepared")
    assert errs == ["[prepared] PRES-SUBS: 6 is outside 1-5"]
    for bad in ("-1", "2.5", "many"):
        errs = tagged(check(books, lambda d, bad=bad: prep(d, "PRES-EVID").update(value=bad)).errors, "prepared")
        assert errs == ["[prepared] PRES-EVID: value must be a whole number of 0 or more"], bad
    assert check(books, lambda d: prep(d, "PRES-EVID").update(value="0")).errors == ()
    assert check(books, lambda d: prep(d, "PRES-CTX").update(value=None, missing="edited_gap")).errors == ()


def test_no_prepared_presentation_needs_every_prepared_item_not_applicable(books):
    def no_presentation(d):
        d["prepared_spans"] = []
        for i in d["stage_a"]:
            i.update(value=None, uncodable=True, moment_s=None)
        for c in d["prepared"]:
            c.update(value=None, missing="not_applicable")
    def rated(d):
        no_presentation(d)
        prep(d, "PRES-SUBS").update(value="3", missing="none")
    def other_reason(d):
        no_presentation(d)
        prep(d, "PRES-EVID").update(missing="edited_gap")
    assert check(books, no_presentation).errors == ()
    assert tagged(check(books, rated).errors, "prepared") == [
        "[prepared] PRES-SUBS: a value with no prepared presentation; use missing not_applicable"]
    assert tagged(check(books, other_reason).errors, "prepared") == [
        "[prepared] PRES-EVID: missing reason edited_gap with no prepared presentation; use not_applicable"]


def test_the_ask_is_two_numbers_or_a_missing_reason(books):
    from pydantic import ValidationError
    given = check(books, lambda d: d["venture"].update(ask_amount_usd=200000, ask_equity_pct=10, ask_missing="none"))
    assert given.errors == () and given.flags == ()
    missing = "[venture.ask] amount and equity must both be given unless ask_missing is not none"
    for field in ("ask_amount_usd", "ask_equity_pct"):
        assert check(books, lambda d, f=field: d["venture"].update({f: None})).errors == (missing,), field
    unseen = check(books, lambda d: d["venture"].update(ask_amount_usd=None, ask_equity_pct=None,
                                                         ask_missing="not_observable"))
    assert unseen.errors == () and unseen.flags == ()
    stray = check(books, lambda d: d["venture"].update(ask_amount_usd=None, ask_missing="edited_gap"))
    assert stray.errors == ("[venture.ask] values are given although ask_missing is 'edited_gap'",)
    with pytest.raises(ValidationError):
        check(books, lambda d: d["venture"].update(ask_equity_pct=150))
    with pytest.raises(ValidationError):
        check(books, lambda d: d["venture"].update(ask_amount_usd=-1))
