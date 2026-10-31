import copy
import json
from pathlib import Path

import pytest

from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.replicate_agreement import AlignmentSettings, align
from tests.panel_fixtures import question_codes, synthetic_record, turn

LABEL = "model self-consistency under sampling; not validity"
ALL = [[i, i] for i in range(1, 20)]
DECISIONS = [[1, 1], [2, 2], [3, 3], [4, 4]]
TIME_KEYS = ("matched_turn_pairs", "median_start_diff_s", "median_abs_start_diff_s", "max_abs_start_diff_s",
             "turn_pairs_beyond_tolerance", "decision_pairs_beyond_tolerance")


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def pair(mutate=None):
    a = synthetic_record()
    b = synthetic_record()
    if mutate:
        mutate(b)
    return PanelRecord.model_validate(a), PanelRecord.model_validate(b)


def tn(data, n):
    return next(t for t in data["turns"] if t["turn_id"] == n)


def ex(data, n):
    return next(e for e in data["exchanges"] if e["exchange_id"] == n)


def code(t, code_id):
    return next(c for c in t["codes"] if c["code"] == code_id)


def drop_turn(data, n):
    data["turns"] = [t for t in data["turns"] if t["turn_id"] != n]


def insert_after(data, n, new_turn):
    data["turns"].insert(next(i for i, t in enumerate(data["turns"]) if t["turn_id"] == n) + 1, new_turn)


def flags_of(report, kind):
    return [f for f in report["flags"] if f["kind"] == kind]


def time_block(pairs, median, median_abs, largest, late_turns, late_decisions):
    return dict(zip(TIME_KEYS, (pairs, median, median_abs, largest, late_turns, late_decisions)))


def _permute_ids(d, m):
    for a in d["actors"]:
        a["actor_id"] = m.get(a["actor_id"], a["actor_id"])
    for t in d["turns"]:
        t["speaker"] = m.get(t["speaker"], t["speaker"])
        t["addressed"] = [m.get(x, x) for x in t["addressed"]]
    for x in d["decisions"]:
        x["actor"] = m.get(x["actor"], x["actor"])


def _shift_turns(d, secs):
    for t in d["turns"]:
        t["start_s"] += secs
        t["end_s"] += secs


def _shift_all(d, secs):
    _shift_turns(d, secs)
    for x in d["decisions"]:
        x["time_s"] += secs
    for sp in d["prepared_spans"] + d["excluded_spans"]:
        sp["start_s"] += secs
        sp["end_s"] += secs


# ---- identical and renamed passes -------------------------------------------------------

def test_identical_passes_agree_perfectly(books):
    p, r = pair()
    rep = align(p, r, books)
    assert rep["label"] == LABEL
    assert set(rep) == {"label", "settings", "settings_note", "actors", "turns", "exchanges", "codes", "links",
                        "decisions", "stage_a", "venture", "prepared", "time", "not_compared", "flags"}
    t = rep["turns"]
    assert set(t) == {"n_primary", "n_replicate", "matched", "function_mismatched", "function_confusion",
                      "unmatched_primary", "unmatched_replicate", "recall", "precision", "recall_any_function",
                      "precision_any_function", "by_function", "speaker_time_agreement", "sequence_agreement", "note"}
    assert t["n_primary"] == t["n_replicate"] == 19 and t["matched"] == ALL
    assert t["recall"] == 1.0 and t["precision"] == 1.0
    assert t["recall_any_function"] == 1.0 and t["precision_any_function"] == 1.0
    assert t["function_mismatched"] == [] and t["function_confusion"] == {}
    assert t["speaker_time_agreement"] == 1.0 and t["sequence_agreement"] == 1.0
    assert t["by_function"]["question"] == {"primary": 4, "replicate": 4, "matched": 4}
    assert t["by_function"]["decides"] == {"primary": 4, "replicate": 4, "matched": 4}
    assert sum(v["primary"] for v in t["by_function"].values()) == 19
    assert set(rep["codes"]) == {"PR-CONCEPT", "PR-SEV", "RS-RESP", "RS-CONCEPT", "RS-EVID", "RS-UNCERT", "PREP-SUBS",
                                 "PREP-COH", "PREP-CTX", "PR-FRAME(derived)", "RS-FRAME(derived)",
                                 "other_concept_names"}
    assert rep["codes"].pop("other_concept_names") == []
    for c in rep["codes"].values():
        assert c["n"] == 4 and c["exact"] == c["both_valued"]
    assert rep["links"]["primary_links"] == rep["links"]["replicate_links"] == rep["links"]["shared"] == 9
    assert rep["flags"] == []


def test_identical_passes_agree_on_every_exchange(books):
    p, r = pair()
    e = align(p, r, books)["exchanges"]
    assert set(e) == {"n_primary", "n_replicate", "by_kind", "matched", "matched_at_half", "mean_jaccard",
                      "unmatched_primary", "unmatched_replicate", "kind_agree", "topic_agree", "topic_n",
                      "kind_confusion", "topic_confusion", "co_membership_agreement", "adjusted_co_membership"}
    assert e["adjusted_co_membership"] == 1.0
    assert e["n_primary"] == e["n_replicate"] == 4
    assert e["matched"] == [[1, 1, 1.0], [2, 2, 1.0], [3, 3, 1.0], [4, 4, 1.0]]
    assert e["matched_at_half"] == 4 and e["mean_jaccard"] == 1.0
    assert e["unmatched_primary"] == [] and e["unmatched_replicate"] == []
    assert e["by_kind"] == {"negotiation": {"primary": 1, "replicate": 1}, "social": {"primary": 1, "replicate": 1},
                            "substantive": {"primary": 2, "replicate": 2}}
    assert e["kind_agree"] == 4
    assert e["kind_confusion"] == {"negotiation|negotiation": 1, "social|social": 1, "substantive|substantive": 2}
    assert e["topic_agree"] == 2 and e["topic_n"] == 2
    assert e["topic_confusion"] == {"customer_retention|customer_retention": 1, "operating_efficiency|operating_efficiency": 1}
    assert e["co_membership_agreement"] == 1.0
    assert "closing" not in e and "investor_sets" not in e


def test_replicate_with_renamed_actors_still_aligns(books):
    p, r = pair(lambda d: _permute_ids(d, {"S1": "S2", "S2": "S1"}))
    rep = align(p, r, books)
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S1", "S3": "S3"}
    assert rep["actors"]["unmapped_primary"] == [] and rep["actors"]["unmapped_replicate"] == []
    assert rep["turns"]["matched"] == ALL
    assert rep["decisions"]["ever_offer_agreement"] == 1.0
    assert rep["exchanges"]["kind_agree"] == 4 and rep["exchanges"]["topic_agree"] == 2
    assert rep["flags"] == []


# ---- turns: who and in what order, not when ----------------------------------------------

def test_shifted_times_do_not_affect_turn_pairing(books):
    p, r = pair(lambda d: _shift_turns(d, 3.0))
    rep = align(p, r, books)
    assert rep["turns"]["matched"] == ALL
    assert rep["time"]["median_start_diff_s"] == 3.0 and rep["time"]["turn_pairs_beyond_tolerance"] == 0
    assert flags_of(rep, "time_drift") == [] and flags_of(rep, "time_outlier") == []
    assert rep["turns"]["speaker_time_agreement"] < 1.0


def test_uniformly_shifted_replicate_matches_everything_and_reports_drift(books):
    p, r = pair(lambda d: _shift_all(d, 40.0))
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["matched"] == ALL
    assert t["unmatched_primary"] == [] and t["unmatched_replicate"] == [] and t["function_mismatched"] == []
    assert rep["exchanges"]["matched"] == [[1, 1, 1.0], [2, 2, 1.0], [3, 3, 1.0], [4, 4, 1.0]]
    assert rep["exchanges"]["co_membership_agreement"] == 1.0
    assert rep["decisions"]["matched"] == DECISIONS
    assert rep["decisions"]["unmatched_primary"] == [] and rep["decisions"]["unmatched_replicate"] == []
    for name, c in rep["codes"].items():
        if name == "other_concept_names":
            continue
        assert c["exact"] == c["both_valued"] and c["missing_agree"] == c["n"]
    assert flags_of(rep, "code_disagreement") == [] and flags_of(rep, "link_disagreement") == []
    assert rep["links"]["recall"] == 1.0 and rep["links"]["precision"] == 1.0
    assert rep["time"] == time_block(19, 40.0, 40.0, 40.0, 19, 4)
    assert len(flags_of(rep, "time_drift")) == 1
    assert len(flags_of(rep, "time_outlier")) == 23
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S1", "S2": "S2", "S3": "S3"}
    assert {f["kind"] for f in rep["flags"]} == {"time_drift", "time_outlier"}
    back = align(r, p, books)
    assert back["time"]["median_start_diff_s"] == -40.0 and back["time"]["median_abs_start_diff_s"] == 40.0


def test_turn_missing_from_the_replicate_is_an_existence_flag(books):
    p, r = pair(lambda d: drop_turn(d, 7))
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["unmatched_primary"] == [7] and t["unmatched_replicate"] == []
    assert t["matched"] == [m for m in ALL if m != [7, 7]]
    assert t["recall"] == pytest.approx(18 / 19) and t["precision"] == 1.0
    fl = flags_of(rep, "unmatched_turn")
    assert [(f["pass"], f["ref"]) for f in fl] == [("primary", 7)]


def test_extra_turn_in_the_replicate_lowers_precision(books):
    def add(d):
        insert_after(d, 12, turn(20, "F1", 129, 130, "repair", "Corrects the month he quoted.", ["S2"], 3, [12]))
    p, r = pair(add)
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["recall"] == 1.0 and t["precision"] == pytest.approx(19 / 20)
    assert t["unmatched_replicate"] == [20] and t["unmatched_primary"] == []
    assert t["by_function"]["repair"] == {"primary": 1, "replicate": 2, "matched": 1}
    assert [(f["pass"], f["ref"]) for f in flags_of(rep, "unmatched_turn")] == [("replicate", 20)]


def test_turn_of_a_different_function_is_a_mismatch_not_a_match(books):
    p, r = pair(lambda d: tn(d, 11).update(function="challenges", codes=[]))
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["function_mismatched"] == [[11, 11]] and t["function_confusion"] == {"question|challenges": 1}
    assert [11, 11] not in t["matched"] and len(t["matched"]) == 18
    assert t["unmatched_primary"] == [] and t["unmatched_replicate"] == []
    assert t["recall"] == pytest.approx(18 / 19) and t["precision"] == pytest.approx(18 / 19)
    assert t["recall_any_function"] == 1.0 and t["precision_any_function"] == 1.0
    fl = flags_of(rep, "function_disagreement")
    assert len(fl) == 1 and fl[0]["ref"] == 11 and "question" in fl[0]["detail"] and "challenges" in fl[0]["detail"]
    assert flags_of(rep, "unmatched_turn") == []
    assert t["by_function"]["question"] == {"primary": 4, "replicate": 3, "matched": 3}
    # the mismatched pair contributes no code comparison
    assert rep["codes"]["PR-CONCEPT"]["n"] == 3 and rep["codes"]["PR-FRAME(derived)"]["n"] == 3
    back = align(r, p, books)["turns"]
    assert back["function_confusion"] == {"challenges|question": 1}


def test_turn_by_a_different_mapped_speaker_never_pairs(books):
    p, r = pair(lambda d: tn(d, 11).update(speaker="S3"))
    t = align(p, r, books)["turns"]
    assert t["unmatched_primary"] == [11] and t["unmatched_replicate"] == [11]
    assert t["function_mismatched"] == []
    assert t["recall_any_function"] == pytest.approx(18 / 19)
    assert t["sequence_agreement"] == pytest.approx(18 / 19)
    assert t["speaker_time_agreement"] < 1.0


def test_function_mismatch_needs_the_same_speaker_but_not_a_nearby_time(books):
    def far(d):
        tn(d, 11).update(function="challenges", codes=[])
        for t in d["turns"]:
            if t["turn_id"] >= 11:
                t["start_s"] += 200.0
                t["end_s"] += 200.0
    p, r = pair(far)
    t = align(p, r, books)["turns"]
    assert t["function_mismatched"] == [[11, 11]] and t["unmatched_primary"] == [] and t["unmatched_replicate"] == []

    p, r = pair(lambda d: tn(d, 11).update(function="challenges", codes=[], speaker="S3"))
    t = align(p, r, books)["turns"]
    assert t["function_mismatched"] == [] and t["unmatched_primary"] == [11] and t["unmatched_replicate"] == [11]


def test_absent_turn_stays_unmatched_beside_a_function_mismatch(books):
    def mut(d):
        tn(d, 11).update(function="challenges", codes=[])
        drop_turn(d, 7)
    p, r = pair(mut)
    t = align(p, r, books)["turns"]
    assert t["function_mismatched"] == [[11, 11]] and t["unmatched_primary"] == [7] and t["unmatched_replicate"] == []
    assert t["recall_any_function"] == pytest.approx(18 / 19) and t["precision_any_function"] == 1.0


def test_extra_candidates_of_the_same_function_match_one_to_one(books):
    def mut(d):
        for n in (21, 20):
            insert_after(d, 11, turn(n, "S2", 118, 119, "question", "Asks for the repeat purchase rate.", ["F1"], 3,
                                     codes=question_codes()))
    p, r = pair(mut)          # replicate order: 11, 20, 21, all indistinguishable S2 questions
    t = align(p, r, books)["turns"]
    assert [11, 11] in t["matched"] and len(t["matched"]) == 19      # the exact tie goes to the earliest
    assert t["unmatched_replicate"] == [20, 21] and t["unmatched_primary"] == []
    pairs = t["matched"] + t["function_mismatched"]
    assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})
    back = align(r, p, books)["turns"]
    assert [11, 11] in back["matched"] and back["unmatched_primary"] == [20, 21] and back["unmatched_replicate"] == []


def test_more_similar_summary_wins_among_candidates_of_the_same_function(books):
    def mut(d):
        tn(d, 11)["summary"] = "Wonders aloud whether the packaging is recyclable."
        insert_after(d, 11, turn(20, "S2", 118, 119, "question", "Asks for the repeat purchase rate.", ["F1"], 3,
                                 codes=question_codes()))
    p, r = pair(mut)
    t = align(p, r, books)["turns"]
    assert [11, 20] in t["matched"]
    assert t["unmatched_replicate"] == [11] and t["unmatched_primary"] == []


def test_same_function_pair_is_preferred_to_a_function_mismatch(books):
    def mut(d):
        # an extra S2 challenge just before the repeat-purchase question, worded like it
        insert_after(d, 10, turn(20, "S2", 114.2, 114.8, "challenges", "Asks for the repeat purchase rate.", ["F1"], 3))
    p, r = pair(mut)
    t = align(p, r, books)["turns"]
    assert [11, 11] in t["matched"] and t["function_mismatched"] == [] and t["unmatched_replicate"] == [20]


TOPICS = ("revenue last year", "customer acquisition cost", "patent filing status", "manufacturing margin",
          "retail distribution partners")


def qa_record(shift=0.0, n=5, starts=None):
    """n question/answer pairs by one investor, 8 s apart, each on its own topic; `shift` moves all of them.

    `starts` gives the question turns' start seconds outright (each answer follows 3 s later).
    """
    d = synthetic_record()
    d["turns"] = d["turns"][:1]
    d["exchanges"], d["decisions"] = [], []
    for k in range(n):
        q0 = (62.0 + 8 * k if starts is None else float(starts[k])) + shift
        topic = TOPICS[k]
        d["turns"] += [
            turn(2 + 2 * k, "S1", q0, q0 + 3, "question", f"Asks about {topic}.", ["F1"]),
            turn(3 + 2 * k, "F1", q0 + 3, q0 + 7, "answer", f"Explains {topic}.", ["S1"], None, [2 + 2 * k])]
    return d


@pytest.mark.parametrize("shift", [5.0, -6.0, 40.0])
def test_dense_same_function_turns_pair_with_their_true_counterparts(books, shift):
    p = PanelRecord.model_validate(qa_record())
    r = PanelRecord.model_validate(qa_record(shift))
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["matched"] == [[i, i] for i in range(1, 12)]
    assert t["unmatched_primary"] == [] and t["unmatched_replicate"] == [] and t["function_mismatched"] == []
    assert rep["links"]["recall"] == 1.0 and rep["links"]["precision"] == 1.0
    assert flags_of(rep, "unmatched_turn") == []


@pytest.mark.parametrize("gone", [2, 6, 10])
@pytest.mark.parametrize("shift", [0.0, 40.0])
def test_absent_question_is_the_one_left_unmatched(books, gone, shift):
    d = qa_record(shift)
    drop_turn(d, gone)                     # one of five questions by the same investor
    tn(d, gone + 1)["responds_to"] = []
    p, r = PanelRecord.model_validate(qa_record()), PanelRecord.model_validate(d)
    t = align(p, r, books)["turns"]
    assert t["unmatched_primary"] == [gone] and t["unmatched_replicate"] == []
    assert t["matched"] == [[i, i] for i in range(1, 12) if i != gone]
    assert t["function_mismatched"] == []


def test_pairing_preserves_order_and_never_crosses(books):
    d = qa_record()
    turns = d["turns"]
    i, j = (next(k for k, t in enumerate(turns) if t["turn_id"] == n) for n in (2, 6))
    turns[i], turns[j] = turns[j], turns[i]          # the replicate asks the first and third questions the other way round
    p, r = PanelRecord.model_validate(qa_record()), PanelRecord.model_validate(d)
    t = align(p, r, books)["turns"]
    pairs = [tuple(m) for m in t["matched"] + t["function_mismatched"]]
    order_p = {x.turn_id: k for k, x in enumerate(p.turns)}
    order_r = {x.turn_id: k for k, x in enumerate(r.turns)}
    ranked = sorted(pairs, key=lambda m: order_p[m[0]])
    assert [order_r[b] for _, b in ranked] == sorted(order_r[b] for _, b in ranked)
    assert not ([2, 2] in t["matched"] and [6, 6] in t["matched"])
    assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs})


def test_turn_order_is_the_list_order_not_the_start_time(books):
    def scramble(d):
        for t in d["turns"]:
            t["start_s"], t["end_s"] = 300.0 - t["end_s"], 300.0 - t["start_s"]      # times now run backwards
    p, r = pair(scramble)
    rep = align(p, r, books)
    assert rep["turns"]["matched"] == ALL and rep["turns"]["sequence_agreement"] == 1.0


def test_times_written_as_concatenated_clock_digits_still_match_every_turn(books):
    # the primary has question turns at 1:07, 1:24, 1:36, 1:48; the replicate wrote them as 107, 124, 136, 148
    p = PanelRecord.model_validate(qa_record(n=4, starts=[67, 84, 96, 108]))
    r = PanelRecord.model_validate(qa_record(n=4, starts=[107, 124, 136, 148]))
    rep = align(p, r, books)
    t = rep["turns"]
    assert t["matched"] == [[i, i] for i in range(1, 10)]
    assert t["unmatched_primary"] == [] and t["unmatched_replicate"] == [] and t["function_mismatched"] == []
    assert rep["actors"]["mapped_by"] == {"F1": "label", "S1": "label", "S2": "label", "S3": "label"}
    assert rep["time"]["median_start_diff_s"] == 40.0 and rep["time"]["turn_pairs_beyond_tolerance"] == 8
    assert len(flags_of(rep, "time_drift")) == 1


def test_garbled_stretch_inside_an_otherwise_aligned_pass(books):
    def garble(d):
        for t in d["turns"]:
            if 2 <= t["turn_id"] <= 16:      # 1:02 -> 102, 1:20 -> 120, 2:14 -> 214
                for k in ("start_s", "end_s"):
                    m, sec = divmod(int(t[k]), 60)
                    t[k] = float(m * 100 + sec)
    p, r = pair(garble)
    rep = align(p, r, books)
    assert rep["turns"]["matched"] == ALL
    assert rep["decisions"]["matched"] == DECISIONS
    assert rep["time"]["median_abs_start_diff_s"] == 40.0
    assert rep["time"]["decision_pairs_beyond_tolerance"] == 0


def test_start_difference_above_tolerance_still_matches_and_is_a_time_outlier(books):
    def mut(d):
        t = tn(d, 12)
        t["start_s"] += 20.0
        t["end_s"] += 20.0
    p, r = pair(mut)
    rep = align(p, r, books)
    assert [12, 12] in rep["turns"]["matched"]
    assert rep["turns"]["unmatched_primary"] == [] and rep["turns"]["unmatched_replicate"] == []
    assert rep["time"]["turn_pairs_beyond_tolerance"] == 1 and rep["time"]["max_abs_start_diff_s"] == 20.0
    assert rep["time"]["median_abs_start_diff_s"] == 0.0
    out = flags_of(rep, "time_outlier")
    assert len(out) == 1 and out[0]["ref"] == 12
    assert flags_of(rep, "time_drift") == []
    wide = align(p, r, books, AlignmentSettings(event_tolerance_s=25.0))
    assert flags_of(wide, "time_outlier") == [] and wide["time"]["turn_pairs_beyond_tolerance"] == 0


def test_time_block_for_identical_passes_and_without_pairs(books):
    p, r = pair()
    rep = align(p, r, books)
    assert rep["time"] == time_block(19, 0.0, 0.0, 0.0, 0, 0)
    assert "diagnostic" in rep["settings_note"]
    assert "event_tolerance_s" in rep["settings_note"] and "decision_tolerance_s" in rep["settings_note"]
    assert "turn_iou" not in rep["settings_note"] and set(rep["settings"]) == {"event_tolerance_s", "decision_tolerance_s"}
    a, b = synthetic_record(), synthetic_record()
    for d in (a, b):
        d["turns"], d["exchanges"], d["decisions"] = [], [], []
    rep = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    assert rep["time"] == time_block(0, None, None, None, 0, 0)
    assert flags_of(rep, "time_drift") == []
    assert rep["turns"]["sequence_agreement"] is None and rep["turns"]["recall"] is None
    assert rep["exchanges"]["co_membership_agreement"] is None and rep["exchanges"]["mean_jaccard"] is None
    assert rep["exchanges"]["adjusted_co_membership"] is None
    assert rep["exchanges"]["matched"] == []


def test_turn_sequence_agreement(books):
    p, r = pair(lambda d: tn(d, 3).update(speaker="S1"))
    assert align(p, r, books)["turns"]["sequence_agreement"] == pytest.approx(18 / 19)
    p, r = pair(lambda d: d["turns"].pop())
    assert align(p, r, books)["turns"]["sequence_agreement"] == pytest.approx(18 / 19)


# ---- codes and links over matched turns -------------------------------------------------

def test_ordinal_disagreement_is_counted_exact_and_within_one(books):
    def mut(d):
        code(tn(d, 3), "PREP-SUBS")["value"] = "4"       # primary 3 vs 4
        code(tn(d, 12), "PREP-SUBS")["value"] = "5"      # primary 3 vs 5
    p, r = pair(mut)
    rep = align(p, r, books)
    c = rep["codes"]["PREP-SUBS"]
    assert c["n"] == 4 and c["both_valued"] == 4
    assert c["exact"] == 2 and c["within_one"] == 3
    assert c["confusion"] == {"3|3": 1, "3|4": 1, "3|5": 1, "4|4": 1}
    fl = flags_of(rep, "code_disagreement")
    assert len(fl) == 2 and all(f["ref"] == "PREP-SUBS" for f in fl) and "turns 3|3" in fl[0]["detail"]
    assert rep["codes"]["RS-RESP"]["within_one"] is None


def test_missing_versus_valued_code_is_not_both_valued(books):
    def mut(d):
        c = code(tn(d, 3), "PREP-COH")
        c["value"] = None
        c["missing"] = "not_applicable"
    p, r = pair(mut)
    rep = align(p, r, books)
    c = rep["codes"]["PREP-COH"]
    assert c["n"] == 4 and c["both_valued"] == 3 and c["exact"] == 3
    assert c["missing_agree"] == 3
    assert c["missing_confusion"] == {}
    assert c["confusion"]["4|∅"] == 1
    assert align(r, p, books)["codes"]["PREP-COH"]["confusion"]["∅|4"] == 1
    assert rep["codes"]["PREP-CTX"]["missing_agree"] == 4


def test_missing_reasons_are_compared(books):
    p, r = pair(lambda d: code(tn(d, 3), "PREP-CTX").update(missing="no_opportunity"))
    c = align(p, r, books)["codes"]["PREP-CTX"]
    assert c["n"] == 4 and c["both_valued"] == 0
    assert c["missing_confusion"] == {"not_applicable|no_opportunity": 1, "not_applicable|not_applicable": 3}
    assert c["missing_agree"] == 3
    assert c["confusion"] == {"∅|∅": 4}


def test_values_are_compared_after_canonicalization(books):
    def mut(d):
        code(tn(d, 3), "RS-CONCEPT")["value"] = "Operating_Efficiency"
        code(tn(d, 5), "RS-EVID")["value"] = "True"
    p, r = pair(mut)
    rep = align(p, r, books)
    c = rep["codes"]["RS-CONCEPT"]
    assert c["exact"] == c["both_valued"] == 4 and c["confusion"] == {
        "customer_retention|customer_retention": 1, "forecast|forecast": 1,
        "operating_efficiency|operating_efficiency": 2}
    assert rep["codes"]["RS-FRAME(derived)"]["confusion"] == {"prevention|prevention": 3, "promotion|promotion": 1}
    assert rep["codes"]["RS-EVID"]["confusion"] == {"false|false": 1, "true|true": 3}
    assert flags_of(rep, "code_disagreement") == []


def test_link_disagreement_is_reported(books):
    p, r = pair(lambda d: tn(d, 7).update(responds_to=[6, 2]))
    rep = align(p, r, books)
    assert rep["links"]["primary_links"] == 9 and rep["links"]["replicate_links"] == 10 and rep["links"]["shared"] == 9
    assert rep["links"]["recall"] == 1.0 and rep["links"]["precision"] == pytest.approx(0.9)
    fl = flags_of(rep, "link_disagreement")
    assert len(fl) == 1 and fl[0]["ref"] == 7
    back = align(r, p, books)["links"]
    assert back["recall"] == pytest.approx(0.9) and back["precision"] == 1.0


def test_links_to_unmatched_targets_are_dropped_and_counted(books):
    def mut(d):
        drop_turn(d, 2)
        tn(d, 3)["responds_to"] = []
    p, r = pair(mut)
    rep = align(p, r, books)
    assert rep["links"]["dropped_unmatched_targets"] == 1      # primary turn 3 points at unmatched turn 2
    assert rep["links"]["primary_links"] == 8 and rep["links"]["shared"] == 8
    assert flags_of(rep, "link_disagreement") == []


def test_links_of_function_mismatched_turns_are_left_out(books):
    p, r = pair(lambda d: tn(d, 13).update(function="challenges"))
    rep = align(p, r, books)
    assert rep["turns"]["function_mismatched"] == [[13, 13]]
    assert rep["links"]["primary_links"] == 8 and rep["links"]["recall"] == 1.0


# ---- exchanges --------------------------------------------------------------------------

def _split_exchange_one(d):
    """The replicate treats the return to acquisition cost as a new exchange 5."""
    for n in (14, 15, 16):
        tn(d, n)["exchange_id"] = 5
    d["exchanges"].append({"exchange_id": 5, "kind": "substantive", "topic": "operating_efficiency",
                           "label": "paid channel cost"})


def test_replicate_that_splits_an_exchange_in_two(books):
    p, r = pair(_split_exchange_one)
    rep = align(p, r, books)
    e = rep["exchanges"]
    assert e["n_primary"] == 4 and e["n_replicate"] == 5
    assert e["matched"] == [[1, 1, pytest.approx(6 / 9)], [2, 2, 1.0], [3, 3, 1.0], [4, 4, 1.0]]
    assert e["unmatched_primary"] == [] and e["unmatched_replicate"] == [5]
    assert e["mean_jaccard"] == pytest.approx((6 / 9 + 3) / 4) and e["matched_at_half"] == 4
    # 171 pairs of aligned turns; the 18 that join turns 2-7 with turns 14-16 are together only in the primary
    assert e["co_membership_agreement"] == pytest.approx(153 / 171)
    # pairs together: 43 in the primary, 25 in the replicate, 25 in both; expected by chance 43 * 25 / 171
    chance = 43 * 25 / 171
    assert e["adjusted_co_membership"] == pytest.approx((25 - chance) / ((43 + 25) / 2 - chance))
    assert e["by_kind"]["substantive"] == {"primary": 2, "replicate": 3}
    assert [(f["pass"], f["ref"]) for f in flags_of(rep, "unmatched_exchange")] == [("replicate", 5)]
    assert rep["turns"]["matched"] == ALL
    back = align(r, p, books)
    assert back["exchanges"]["unmatched_primary"] == [5] and back["exchanges"]["unmatched_replicate"] == []
    assert [(f["pass"], f["ref"]) for f in flags_of(back, "unmatched_exchange")] == [("primary", 5)]


def test_exchange_matched_below_half_is_a_boundary_disagreement(books):
    def mut(d):
        _split_exchange_one(d)
        for n in (4, 5, 6, 7):
            tn(d, n)["exchange_id"] = 3          # the follow-up is filed under the repeat-purchase exchange
    p, r = pair(mut)
    rep = align(p, r, books)
    e = rep["exchanges"]
    # 3|3 overlaps 3 of 7 turns; 1|5 overlaps 3 of 9; 1|1 would overlap only 2 of 9
    assert e["matched"] == [[1, 5, pytest.approx(3 / 9)], [2, 2, 1.0], [3, 3, pytest.approx(3 / 7)], [4, 4, 1.0]]
    assert e["matched_at_half"] == 2 and e["unmatched_replicate"] == [1]
    fl = flags_of(rep, "exchange_boundary_disagreement")
    assert [f["ref"] for f in fl] == [1, 3] and "1|5" in fl[0]["detail"]
    assert e["co_membership_agreement"] < 1.0
    assert flags_of(align(*pair(), books), "exchange_boundary_disagreement") == []


def test_exchange_jaccard_at_exactly_half_is_not_a_boundary_disagreement(books):
    a, b = synthetic_record(), synthetic_record()
    drop_turn(a, 13)                                    # primary exchange 3: turns 11 and 12
    tn(b, 12)["exchange_id"] = None                     # replicate exchange 3: turn 11, and 13 which has no partner
    rep = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    e = rep["exchanges"]
    assert [3, 3, 0.5] in e["matched"] and e["matched_at_half"] == 4
    assert flags_of(rep, "exchange_boundary_disagreement") == []


def test_replicate_with_a_different_topic(books):
    p, r = pair(lambda d: ex(d, 3).update(topic="market_opportunity"))
    rep = align(p, r, books)
    e = rep["exchanges"]
    assert e["matched"] == [[1, 1, 1.0], [2, 2, 1.0], [3, 3, 1.0], [4, 4, 1.0]]
    assert e["topic_agree"] == 1 and e["topic_n"] == 2 and e["kind_agree"] == 4
    assert e["topic_confusion"] == {"customer_retention|market_opportunity": 1, "operating_efficiency|operating_efficiency": 1}
    fl = flags_of(rep, "exchange_topic_disagreement")
    assert len(fl) == 1 and fl[0]["ref"] == 3 and "customer_retention" in fl[0]["detail"]
    assert [f["kind"] for f in rep["flags"]] == ["exchange_topic_disagreement"]


def test_there_is_no_closing_comparison(books):
    p, r = pair(lambda d: ex(d, 1).update(label="another label"))
    rep = align(p, r, books)
    assert "closing" not in rep["exchanges"] and "investor_sets" not in rep["exchanges"]
    assert rep["flags"] == []
    import pipeline.stage_b.replicate_agreement as module
    assert "closing_disagreement" not in Path(module.__file__).read_text()


def test_replicate_that_merges_the_social_exchange_into_a_substantive_one(books):
    def mut(d):
        # the joke and the laugh are filed as the opening of the repeat-purchase questioning
        ex(d, 2).update(kind="substantive", topic="customer_retention", label="repeat purchase rate")
        tn(d, 11)["exchange_id"] = 2
    p, r = pair(mut)
    rep = align(p, r, books)
    e = rep["exchanges"]
    assert e["matched"] == [[1, 1, 1.0], [2, 2, pytest.approx(2 / 3)], [3, 3, pytest.approx(2 / 3)], [4, 4, 1.0]]
    assert e["kind_agree"] == 3
    assert e["kind_confusion"] == {"negotiation|negotiation": 1, "social|substantive": 1, "substantive|substantive": 2}
    fl = flags_of(rep, "exchange_kind_disagreement")
    assert len(fl) == 1 and fl[0]["ref"] == 2 and "social" in fl[0]["detail"] and "substantive" in fl[0]["detail"]
    # the topic is compared only where both passes call the exchange substantive
    assert e["topic_n"] == 2
    assert e["co_membership_agreement"] < 1.0
    assert e["by_kind"] == {"negotiation": {"primary": 1, "replicate": 1}, "social": {"primary": 1, "replicate": 0},
                            "substantive": {"primary": 2, "replicate": 3}}


def test_exchange_absorbed_whole_into_another_is_unmatched(books):
    def mut(d):
        tn(d, 8)["exchange_id"] = 3
        tn(d, 9)["exchange_id"] = 3
        d["exchanges"] = [e for e in d["exchanges"] if e["exchange_id"] != 2]
    p, r = pair(mut)
    rep = align(p, r, books)
    e = rep["exchanges"]
    assert e["matched"] == [[1, 1, 1.0], [3, 3, pytest.approx(3 / 5)], [4, 4, 1.0]]
    assert e["unmatched_primary"] == [2] and e["unmatched_replicate"] == []
    assert [(f["pass"], f["ref"]) for f in flags_of(rep, "unmatched_exchange")] == [("primary", 2)]
    assert flags_of(rep, "exchange_kind_disagreement") == []


def test_exchange_matching_is_one_to_one_and_ties_go_to_the_lower_ids(books):
    def mut(d):
        # exchange 1 is split into two halves that overlap the primary's exchange equally (turns 2-4 and 5-7)
        for t in d["turns"]:
            if t["exchange_id"] == 1:
                t["exchange_id"] = 6 if t["turn_id"] in (2, 3, 4) else (5 if t["turn_id"] in (5, 6, 7) else None)
        d["exchanges"] = [e for e in d["exchanges"] if e["exchange_id"] != 1] + [
            {"exchange_id": n, "kind": "substantive", "topic": "operating_efficiency", "label": ""}
            for n in (5, 6)]
    p, r = pair(mut)
    e = align(p, r, books)["exchanges"]
    assert [1, 5, pytest.approx(3 / 9)] in e["matched"] and e["unmatched_replicate"] == [6]
    assert len({m[0] for m in e["matched"]}) == len(e["matched"]) == len({m[1] for m in e["matched"]})


def test_exchange_overlap_counts_function_mismatched_pairs_but_not_unmatched_turns(books):
    p, r = pair(lambda d: tn(d, 11).update(function="challenges", codes=[]))
    assert align(p, r, books)["exchanges"]["matched"] == [[1, 1, 1.0], [2, 2, 1.0], [3, 3, 1.0], [4, 4, 1.0]]
    def mut(d):
        drop_turn(d, 13)
    p, r = pair(mut)
    e = align(p, r, books)["exchanges"]
    assert [3, 3, 1.0] in e["matched"]                 # the primary's turn 13 has no partner, so it is not counted
    assert e["co_membership_agreement"] == 1.0


def test_turns_with_no_exchange_are_singletons_in_co_membership(books):
    def mut(d):
        tn(d, 10)["exchange_id"] = 2                    # the exit is filed with the joke
    p, r = pair(mut)
    e = align(p, r, books)["exchanges"]
    # turn 10 is alone in the primary and with turns 8 and 9 in the replicate: 2 of 171 pairs differ;
    # turns 1 and 10, both without an exchange in the primary, are not counted as being together
    assert e["co_membership_agreement"] == pytest.approx(169 / 171)
    assert [2, 2, pytest.approx(2 / 3)] in e["matched"]


# ---- decisions --------------------------------------------------------------------------

def test_decision_time_within_tolerance_matches_and_ever_offer_agrees(books):
    def mut(d):
        for x in d["decisions"]:
            x["time_s"] += 20.0
    p, r = pair(mut)
    rep = align(p, r, books)
    dec = rep["decisions"]
    assert dec["matched"] == DECISIONS
    assert dec["unmatched_primary"] == [] and dec["unmatched_replicate"] == []
    assert dec["ever_offer"] == {"S1": {"primary": True, "replicate": True}, "S2": {"primary": True, "replicate": True},
                                 "S3": {"primary": False, "replicate": False}}
    assert dec["ever_offer_agreement"] == 1.0
    assert dec["any_offer"] == {"primary": True, "replicate": True}
    assert rep["time"]["decision_pairs_beyond_tolerance"] == 0 and flags_of(rep, "time_outlier") == []
    # beyond the threshold the pairs still match: time is a diagnostic, not a condition
    rep2 = align(p, r, books, AlignmentSettings(decision_tolerance_s=10.0))
    assert rep2["decisions"]["matched"] == DECISIONS
    assert flags_of(rep2, "unmatched_decision") == []
    assert rep2["time"]["decision_pairs_beyond_tolerance"] == 4
    assert [f["ref"] for f in flags_of(rep2, "time_outlier")] == [1, 2, 3, 4]


def test_missing_join_changes_ever_offer(books):
    p, r = pair(lambda d: d["decisions"].remove(next(x for x in d["decisions"] if x["decision_id"] == 3)))
    rep = align(p, r, books)
    dec = rep["decisions"]
    assert dec["ever_offer"]["S2"] == {"primary": True, "replicate": False}
    assert dec["ever_offer"]["S1"] == {"primary": True, "replicate": True}
    assert dec["ever_offer_agreement"] == pytest.approx(2 / 3)
    assert dec["any_offer"] == {"primary": True, "replicate": True}
    assert dec["unmatched_primary"] == [3]
    assert [f["ref"] for f in flags_of(rep, "ever_offer_disagreement")] == ["S2"]


def test_decision_of_different_kind_does_not_match(books):
    p, r = pair(lambda d: next(x for x in d["decisions"] if x["decision_id"] == 2).update(kind="joins_offer"))
    dec = align(p, r, books)["decisions"]
    assert dec["unmatched_primary"] == [2] and dec["unmatched_replicate"] == [2]
    assert [2, 2] not in dec["matched"]


def test_decision_by_different_mapped_actor_does_not_match(books):
    p, r = pair(lambda d: next(x for x in d["decisions"] if x["decision_id"] == 1).update(actor="S2"))
    dec = align(p, r, books)["decisions"]
    assert dec["unmatched_primary"] == [1] and dec["unmatched_replicate"] == [1]


def test_decisions_pair_by_actor_kind_and_order_only(books):
    def mut(d):
        for x in d["decisions"]:
            x["time_s"] = 10.0 + x["decision_id"]            # far from the primary's times, same order
    p, r = pair(mut)
    assert align(p, r, books)["decisions"]["matched"] == DECISIONS

    def reorder(d):
        d["decisions"][0]["time_s"] = 171.0                  # the exit now comes last in the replicate
    p, r = pair(reorder)
    rep = align(p, r, books)
    dec = rep["decisions"]
    assert dec["matched"] == [[2, 2], [3, 3], [4, 4]]       # keeping three in order beats keeping the exit
    assert dec["unmatched_primary"] == [1] and dec["unmatched_replicate"] == [1]
    assert len(flags_of(rep, "unmatched_decision")) == 2


def test_decisions_pair_in_order_within_actor_and_kind(books):
    def base(d, shift):
        d["decisions"] = [
            {"decision_id": 1, "actor": "S1", "kind": "offer", "time_s": 100.0 + shift, "turn_id": 17, "offer_id": "O1",
             "terms": "a", "note": ""},
            {"decision_id": 2, "actor": "S1", "kind": "offer", "time_s": 120.0 + shift, "turn_id": 17, "offer_id": "O2",
             "terms": "b", "note": ""}]
    a, b = synthetic_record(), synthetic_record()
    base(a, 0.0)
    base(b, 12.0)      # replicate offer 1 at 112 is nearer to primary offer 2 at 120 than to its own counterpart
    dec = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)["decisions"]
    assert dec["matched"] == [[1, 1], [2, 2]]
    assert dec["unmatched_primary"] == [] and dec["unmatched_replicate"] == []
    assert [o["primary_offer_id"] for o in dec["offers"]] == ["O1", "O2"]


def test_offer_membership_agrees_for_identical_passes_and_renamed_offer_ids(books):
    def rename(d):
        for x in d["decisions"]:
            if x["offer_id"] == "O1":
                x["offer_id"] = "O7"
    p, r = pair(rename)
    dec = align(p, r, books)["decisions"]
    assert dec["offers"] == [{"primary_offer_id": "O1", "replicate_offer_id": "O7",
                              "primary_members": ["F1", "S1", "S2"], "replicate_members": ["F1", "S1", "S2"],
                              "agree": True}]
    assert dec["offer_membership_agreement"] == 1.0


def test_offer_membership_disagreement_is_flagged(books):
    def split(d):
        next(x for x in d["decisions"] if x["decision_id"] == 3).update(kind="offer", offer_id="O2")
    p, r = pair(split)
    rep = align(p, r, books)
    dec = rep["decisions"]
    assert dec["offers"] == [{"primary_offer_id": "O1", "replicate_offer_id": "O1",
                              "primary_members": ["F1", "S1", "S2"], "replicate_members": ["F1", "S1"],
                              "agree": False}]
    assert dec["offer_membership_agreement"] == 0.0
    fl = flags_of(rep, "offer_membership_disagreement")
    assert len(fl) == 1 and fl[0]["ref"] == "O1"
    assert dec["ever_offer_agreement"] == 1.0


def test_offer_membership_uses_mapped_actors(books):
    p, r = pair(lambda d: _permute_ids(d, {"S1": "S2", "S2": "S1"}))
    dec = align(p, r, books)["decisions"]
    assert dec["offers"][0]["replicate_members"] == ["F1", "S1", "S2"] and dec["offers"][0]["agree"] is True


def test_no_matched_offer_gives_no_membership_agreement(books):
    p, r = pair(lambda d: d.update(decisions=d["decisions"][:1]))
    dec = align(p, r, books)["decisions"]
    assert dec["offers"] == [] and dec["offer_membership_agreement"] is None


# ---- stage A, venture, prepared ---------------------------------------------------------

def test_stage_a_and_venture_disagreements(books):
    def mut(d):
        d["stage_a"][0]["value"] = 5
        d["stage_a"][1]["value"] = None
        d["stage_a"][1]["uncodable"] = True
        d["stage_a"][2]["moment_s"] = 30.0          # the moment is not compared
        d["venture"]["primary_industry"] = "food_beverage"
    p, r = pair(mut)
    rep = align(p, r, books)
    first, second = books.stage_a_ids()[:2]
    assert rep["stage_a"][first] == {"primary": 3, "replicate": 5, "abs_diff": 2}
    assert rep["stage_a"][second] == {"primary": 3, "replicate": None, "abs_diff": None}
    s = rep["stage_a"]["_summary"]
    assert s["both_rated"] == 2 and s["exact"] == 1 and s["within_one"] == 1
    assert s["mean_abs_diff"] == pytest.approx(2 / 2)
    assert sorted(f["ref"] for f in flags_of(rep, "stage_a_disagreement")) == sorted([first, second])
    assert rep["venture"]["primary_industry"] == {"primary": "home_household", "replicate": "food_beverage", "agree": False}
    assert rep["venture"]["product_form"]["agree"] is True
    assert [f["ref"] for f in flags_of(rep, "venture_disagreement")] == ["primary_industry"]


def test_prepared_overlap(books):
    def mut(d):
        d["prepared_spans"] = [{"start_s": 5.0, "end_s": 30.0}, {"start_s": 40.0, "end_s": 60.0}]
    p, r = pair(mut)
    assert align(p, r, books)["prepared"] == {"primary_s": 55.0, "replicate_s": 45.0, "iou": pytest.approx(45 / 55),
                                              "n": 4, "agree": 4, "within_one": 3}


def test_prepared_codes_agree_exactly_and_within_one_on_the_ordinal_codes(books):
    p, r = pair()
    assert align(p, r, books)["prepared"] == {"primary_s": 55.0, "replicate_s": 55.0, "iou": 1.0,
                                              "n": 4, "agree": 4, "within_one": 3}

    def mut(d):
        values = {"PRES-SUBS": "5", "PRES-COH": "3", "PRES-EVID": "6"}
        for c in d["prepared"]:
            if c["code"] in values:
                c["value"] = values[c["code"]]
            else:
                c.update(value=None, missing="edited_gap")      # valued in one pass only: not compared
    p, r = pair(mut)
    prepared = align(p, r, books)["prepared"]
    assert {k: prepared[k] for k in ("n", "agree", "within_one")} == {"n": 3, "agree": 0, "within_one": 1}


def test_prepared_values_agree_whatever_whitespace_surrounds_them(books):
    def mut(d):
        for c in d["prepared"]:
            c["value"] = f" {c['value']}\n" if c["code"] == "PRES-EVID" else f"{c['value']} "   # a count and ordinals
    p, r = pair(mut)
    assert {c.code: c.value for c in r.prepared}["PRES-EVID"] == " 5\n"
    prepared = align(p, r, books)["prepared"]
    assert {k: prepared[k] for k in ("n", "agree", "within_one")} == {"n": 4, "agree": 4, "within_one": 3}


# ---- actors -----------------------------------------------------------------------------

def test_unmapped_replicate_actor_is_flagged(books):
    p, r = pair(lambda d: d["actors"].append({"actor_id": "S4", "role": "investor", "label": "Investor D"}))
    rep = align(p, r, books)
    assert rep["actors"]["unmapped_replicate"] == ["S4"]
    assert [f["ref"] for f in flags_of(rep, "unmapped_actor")] == ["S4"]
    assert "S4" not in rep["decisions"]["ever_offer"]


def test_permuted_ids_with_kept_labels_map_by_label_even_when_times_are_scrambled(books):
    def mut(d):
        _permute_ids(d, {"S1": "S2", "S2": "S3", "S3": "S1"})
        d["actors"].reverse()
        # move investor turns onto another investor's stretch of the primary: overlap alone would mis-map
        times = {t["turn_id"]: (t["start_s"], t["end_s"]) for t in d["turns"]}
        for a, b in ((2, 4), (7, 8), (10, 11)):
            ta, tb = tn(d, a), tn(d, b)
            (ta["start_s"], ta["end_s"]), (tb["start_s"], tb["end_s"]) = times[b], times[a]
    p, r = pair(mut)
    rep = align(p, r, books)
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S3", "S2": "S1", "S3": "S2"}
    assert rep["actors"]["mapped_by"] == {"F1": "label", "S1": "label", "S2": "label", "S3": "label"}
    assert rep["actors"]["unmapped_primary"] == [] and rep["actors"]["unmapped_replicate"] == []
    assert rep["turns"]["matched"] == ALL
    assert rep["decisions"]["matched"] == DECISIONS
    assert rep["decisions"]["ever_offer_agreement"] == 1.0
    assert rep["decisions"]["offers"][0]["agree"] is True
    assert flags_of(rep, "ever_offer_disagreement") == []


def test_labels_are_normalized_before_comparison(books):
    def mut(d):
        _permute_ids(d, {"S1": "S2", "S2": "S1"})
        for a in d["actors"]:
            a["label"] = "  " + a["label"].upper().replace(" ", "   ")
    p, r = pair(mut)
    rep = align(p, r, books)["actors"]
    assert rep["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S1", "S3": "S3"}
    assert set(rep["mapped_by"].values()) == {"label"}


def test_actors_with_different_labels_map_by_order_of_first_speech(books):
    def mut(d):
        _permute_ids(d, {"S1": "S2", "S2": "S1"})
        for a in d["actors"]:
            if a["role"] == "investor":
                a["label"] = f"Person seated at {a['actor_id']}"
    p, r = pair(mut)
    rep = align(p, r, books)
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S1", "S3": "S3"}
    assert rep["actors"]["mapped_by"] == {"F1": "label", "S1": "order", "S2": "order", "S3": "order"}
    assert rep["turns"]["recall"] == 1.0


def test_a_label_shared_by_two_actors_is_not_used_to_map_them(books):
    def same(d):
        for a in d["actors"]:
            if a["actor_id"] in ("S1", "S2"):
                a["label"] = "Investor"
    a, b = synthetic_record(), synthetic_record()
    same(a)
    same(b)
    _permute_ids(b, {"S1": "S2", "S2": "S1"})
    rep = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)["actors"]
    assert rep["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S1", "S3": "S3"}
    assert rep["mapped_by"] == {"F1": "label", "S1": "order", "S2": "order", "S3": "label"}


def test_label_never_maps_across_roles(books):
    p, r = pair(lambda d: next(a for a in d["actors"] if a["actor_id"] == "S3").update(label="Founder"))
    rep = align(p, r, books)["actors"]
    assert rep["mapping"]["S3"] == "S3" and rep["mapping"]["F1"] == "F1"
    assert rep["mapped_by"]["S3"] == "order"


# ---- the report as a whole --------------------------------------------------------------

def test_report_is_json_serializable_and_deterministic(books):
    def mut(d):
        code(tn(d, 3), "PREP-SUBS")["value"] = "5"
        d["decisions"].pop()
        _split_exchange_one(d)
    p, r = pair(mut)
    one = align(p, r, books)
    json.dumps(one)
    assert one == align(p, r, books)
    assert json.dumps(one) == json.dumps(align(p, r, books))


def test_report_states_its_caveats(books):
    p, r = pair()
    rep = align(p, r, books)
    assert rep["not_compared"] == ["excluded_spans", "source_complete", "claim_conflicts", "addressed",
                                   "exchange label", "stage_a moment"]
    assert "link_basis" not in rep["not_compared"]
    assert "not by time" in rep["turns"]["note"] and "speaker_time_agreement" in rep["turns"]["note"]
    assert rep["links"]["note"] == "computed over links whose both endpoints are matched turns"


def test_alignment_does_not_depend_on_the_order_of_the_unordered_lists(books):
    a, b = synthetic_record(), synthetic_record()
    _split_exchange_one(b)
    tn(b, 11).update(function="challenges", codes=[])
    one = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    for d in (a2, b2):
        d["actors"].reverse()
        d["exchanges"].reverse()
        d["decisions"].reverse()
    two = align(PanelRecord.model_validate(a2), PanelRecord.model_validate(b2), books)
    for block in ("turns", "exchanges", "codes", "links", "decisions", "flags"):
        assert one[block] == two[block], block


def test_structure_labels_are_compared_in_their_canonical_form(books):
    def mut(d):
        tn(d, 2)["function"] = "Question"
        ex(d, 1).update(kind="Substantive", topic="Operating_Efficiency")
        ex(d, 2)["topic"] = "none"
        d["decisions"][0]["kind"] = "Exit"
        d["decisions"][1]["kind"] = "OFFER"
    p, r = pair(mut)
    rep = align(p, r, books)
    assert rep["flags"] == []
    assert rep["turns"]["matched"] == ALL and rep["turns"]["by_function"]["question"]["replicate"] == 4
    assert rep["codes"]["PR-CONCEPT"]["n"] == 4
    assert rep["exchanges"]["kind_agree"] == 4 and rep["exchanges"]["topic_agree"] == 2
    assert rep["decisions"]["matched"] == DECISIONS and rep["decisions"]["ever_offer_agreement"] == 1.0
    assert align(r, p, books)["flags"] == []


# ---- v3-dev5 ----------------------------------------------------------------------------

def test_unlabelled_actors_map_by_order_of_first_speech_not_by_time(books):
    def mut(d):
        _permute_ids(d, {"S1": "S3", "S2": "S1", "S3": "S2"})
        for a in d["actors"]:
            if a["role"] == "investor":
                a["label"] = f"Person seated at {a['actor_id']}"
        d["actors"].reverse()
        # the times of the investors' turns are scrambled across investors; the order of speaking is not
        times = {t["turn_id"]: (t["start_s"], t["end_s"]) for t in d["turns"]}
        for a, b in ((2, 4), (7, 8), (10, 11)):
            ta, tb = tn(d, a), tn(d, b)
            (ta["start_s"], ta["end_s"]), (tb["start_s"], tb["end_s"]) = times[b], times[a]
    p, r = pair(mut)
    rep = align(p, r, books)
    # first to speak: primary S1, S2, S3; replicate S3 (the primary's S1), S1, S2
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S3", "S3": "S1"}
    assert rep["actors"]["mapped_by"] == {"F1": "label", "S1": "order", "S2": "order", "S3": "order"}
    assert rep["turns"]["matched"] == ALL and rep["decisions"]["matched"] == DECISIONS


def test_order_fallback_stays_within_role_and_leaves_extras_unmapped(books):
    def mut(d):
        for a in d["actors"]:
            a["label"] = f"Person {a['actor_id']}"
        d["actors"].append({"actor_id": "S4", "role": "investor", "label": "Person S4"})      # never speaks
        d["actors"].append({"actor_id": "H1", "role": "host", "label": "Narrator"})
    p, r = pair(mut)
    rep = align(p, r, books)["actors"]
    assert rep["mapping"] == {"F1": "F1", "S1": "S1", "S2": "S2", "S3": "S3"}
    assert set(rep["mapped_by"].values()) == {"order"}
    assert rep["unmapped_replicate"] == ["H1", "S4"] and rep["unmapped_primary"] == []


def test_adjusted_co_membership_is_none_when_undefined_and_near_zero_for_unrelated_groupings(books):
    def one_group(d):
        for t in d["turns"]:
            t["exchange_id"] = 1
        d["exchanges"] = d["exchanges"][:1]
    a, b = synthetic_record(), synthetic_record()
    one_group(a)
    one_group(b)
    e = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)["exchanges"]
    assert e["co_membership_agreement"] == 1.0 and e["adjusted_co_membership"] is None    # no grouping to agree on
    p, r = pair(one_group)
    e = align(p, r, books)["exchanges"]
    assert e["adjusted_co_membership"] == 0.0 and 0 < e["co_membership_agreement"] < 1


# ---- actor mapping by shared name tokens -------------------------------------------------

def _relabel(d, labels):
    for a in d["actors"]:
        a["label"] = labels[a["actor_id"]]


FIRST = {"F1": "Manny", "S1": "Rohan", "S2": "Barbara", "S3": "Kevin"}
FULL = {"F1": "Manny Lubin", "S1": "Rohan Oza", "S2": "Barbara Corcoran", "S3": "Kevin O'Leary"}


def _named_pair(primary_labels, replicate_labels, permute=None):
    a, b = synthetic_record(), synthetic_record()
    _relabel(a, primary_labels)
    _relabel(b, replicate_labels)
    if permute:
        _permute_ids(b, permute)
    return PanelRecord.model_validate(a), PanelRecord.model_validate(b)


def test_first_names_map_to_full_names_whatever_the_ids(books):
    # the replicate numbers the investors in another order, and its S2 (the primary's S3) is listed first
    p, r = _named_pair(FIRST, FULL, {"S1": "S3", "S2": "S1", "S3": "S2"})
    rep = align(p, r, books)
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S3", "S3": "S1"}
    assert rep["actors"]["mapped_by"] == {"F1": "label_tokens", "S1": "label_tokens", "S2": "label_tokens",
                                          "S3": "label_tokens"}
    assert rep["decisions"]["matched"] == DECISIONS and rep["decisions"]["ever_offer_agreement"] == 1.0
    assert rep["turns"]["matched"] == ALL and rep["flags"] == []
    back = align(r, p, books)
    assert back["actors"]["mapping"] == {"F1": "F1", "S1": "S3", "S2": "S1", "S3": "S2"} and back["flags"] == []


def test_token_mapping_beats_the_order_of_first_speech(books):
    def mut(d):
        _relabel(d, FULL)
        _permute_ids(d, {"S1": "S3", "S3": "S1"})
        # in the replicate Kevin speaks first: order of first speech alone would map him to Rohan
        d["turns"].insert(1, turn(20, "S1", 60.5, 61.5, "social", "Compliments the packaging.", ["F1"]))
    a = synthetic_record()
    _relabel(a, FIRST)
    b = synthetic_record()
    mut(b)
    rep = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S3", "S2": "S2", "S3": "S1"}
    assert rep["decisions"]["matched"] == DECISIONS
    assert flags_of(rep, "actor_mapped_by_order") == []


def test_apostrophes_and_punctuation_split_labels_into_tokens(books):
    p, r = _named_pair({**FIRST, "S3": "Kevin"}, {**FIRST, "S3": "Mr. Kevin O'Leary"})
    rep = align(p, r, books)["actors"]
    assert rep["mapping"]["S3"] == "S3" and rep["mapped_by"]["S3"] == "label_tokens"
    p, r = _named_pair({**FIRST, "S3": "O'Leary"}, {**FIRST, "S3": "kevin o’leary (investor)"})
    assert align(p, r, books)["actors"]["mapped_by"]["S3"] == "label_tokens"


def test_two_actors_sharing_a_name_token_are_not_token_mapped(books):
    # the primary's "Mark" could be either Mark of the replicate: left to the fallback, and flagged
    p, r = _named_pair({"F1": "Manny", "S1": "Mark", "S2": "Barbara", "S3": "Lori"},
                       {"F1": "Manny", "S1": "Mark Cuban", "S2": "Barbara Corcoran", "S3": "Mark Smith"})
    rep = align(p, r, books)
    assert rep["actors"]["mapped_by"] == {"F1": "label", "S1": "order", "S2": "label_tokens", "S3": "order"}
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S1", "S2": "S2", "S3": "S3"}
    assert [(f["pass"], f["ref"]) for f in flags_of(rep, "actor_mapped_by_order")] == [("both", "S1"), ("both", "S3")]
    # seen from the other side the two Marks tie for the one "Mark" in the same way
    back = align(r, p, books)["actors"]
    assert back["mapped_by"]["S1"] == "order" and back["mapped_by"]["S3"] == "order"


def test_the_fuller_overlap_wins_when_one_token_is_shared_by_two(books):
    # "Mark Cuban" shares two tokens with "Mark Cuban (seated left)" and one with "Mark Smith": not a tie
    p, r = _named_pair({"F1": "Manny", "S1": "Mark Cuban", "S2": "Mark Smith", "S3": "Lori"},
                       {"F1": "Manny", "S1": "Mark Smith, guest", "S2": "Mark Cuban (seated left)", "S3": "Lori"})
    rep = align(p, r, books)["actors"]
    assert rep["mapping"] == {"F1": "F1", "S1": "S2", "S2": "S1", "S3": "S3"}
    assert rep["mapped_by"] == {"F1": "label", "S1": "label_tokens", "S2": "label_tokens", "S3": "label"}


def test_generic_only_labels_fall_through_to_order_with_a_flag(books):
    p, r = _named_pair({"F1": "Founder", "S1": "Investor in blue suit", "S2": "Woman investor", "S3": "The shark"},
                       {"F1": "The founder", "S1": "Investor", "S2": "Lady, host", "S3": "Shark man"})
    rep = align(p, r, books)
    assert set(rep["actors"]["mapped_by"].values()) == {"order"}
    assert rep["actors"]["mapping"] == {"F1": "F1", "S1": "S1", "S2": "S2", "S3": "S3"}
    fl = flags_of(rep, "actor_mapped_by_order")
    assert [f["ref"] for f in fl] == ["F1", "S1", "S2", "S3"] and all(f["pass"] == "both" for f in fl)
    assert "guess" in fl[0]["detail"]
    assert [f["kind"] for f in rep["flags"]] == ["actor_mapped_by_order"] * 4


def test_token_mapping_never_crosses_roles_and_exact_labels_are_unchanged(books):
    p, r = _named_pair({"F1": "Kevin", "S1": "Rohan", "S2": "Barbara", "S3": "Lori"},
                       {"F1": "Manny Lubin", "S1": "Rohan", "S2": "Barbara", "S3": "Kevin O'Leary"})
    rep = align(p, r, books)["actors"]
    assert rep["mapping"] == {"F1": "F1", "S1": "S1", "S2": "S2", "S3": "S3"}
    assert rep["mapped_by"] == {"F1": "order", "S1": "label", "S2": "label", "S3": "order"}
    p, r = pair()
    rep = align(p, r, books)
    assert set(rep["actors"]["mapped_by"].values()) == {"label"} and rep["flags"] == []


# ---- v3-dev7: derived frames -------------------------------------------------------------

def test_derived_frame_agreement_has_the_shape_of_a_code(books):
    p, r = pair()
    codes = align(p, r, books)["codes"]
    assert codes["PR-FRAME(derived)"] == {"n": 4, "both_valued": 4, "exact": 4, "within_one": None, "missing_agree": 4,
                                          "confusion": {"prevention|prevention": 4}, "missing_confusion": {}}
    assert codes["RS-FRAME(derived)"] == {"n": 4, "both_valued": 4, "exact": 4, "within_one": None, "missing_agree": 4,
                                          "confusion": {"prevention|prevention": 3, "promotion|promotion": 1},
                                          "missing_confusion": {}}
    assert set(codes["PR-FRAME(derived)"]) == set(codes["PR-CONCEPT"])


def test_two_concepts_of_one_frame_disagree_on_the_concept_only(books):
    def mut(d):
        code(tn(d, 2), "PR-CONCEPT")["value"] = "competition"        # prevention, like operating_efficiency
        code(tn(d, 11), "PR-CONCEPT")["value"] = "sales"              # promotion against customer_retention
        code(tn(d, 14), "PR-CONCEPT")["value"] = "NONE"
        code(tn(d, 12), "RS-CONCEPT").update(value=None, missing="ambiguous")
    p, r = pair(mut)
    rep = align(p, r, books)
    assert rep["codes"]["PR-CONCEPT"]["exact"] == 1
    frame = rep["codes"]["PR-FRAME(derived)"]
    assert frame["n"] == 4 and frame["both_valued"] == 4 and frame["exact"] == 2
    assert frame["confusion"] == {"prevention|NEUTRAL": 1, "prevention|prevention": 2, "prevention|promotion": 1}
    answers = rep["codes"]["RS-FRAME(derived)"]
    assert answers["both_valued"] == 3 and answers["exact"] == 3 and answers["missing_agree"] == 3
    assert answers["confusion"]["prevention|∅"] == 1
    # the derived rows add no flags of their own: each disagreement is flagged once, on the concept
    assert sorted(f["ref"] for f in flags_of(rep, "code_disagreement")) == ["PR-CONCEPT"] * 3 + ["RS-CONCEPT"]


def test_derived_frames_compare_missing_reasons(books):
    def mut(d):
        code(tn(d, 3), "RS-CONCEPT").update(value=None, missing="ambiguous")
    a, b = synthetic_record(), synthetic_record()
    mut(a)
    mut(b)
    code(tn(b, 3), "RS-CONCEPT")["missing"] = "not_observable"
    row = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)["codes"]["RS-FRAME(derived)"]
    assert row["both_valued"] == 3 and row["missing_agree"] == 3
    assert row["missing_confusion"] == {"ambiguous|not_observable": 1} and row["confusion"]["∅|∅"] == 1


# ---- v3-dev8: other concepts ---------------------------------------------------------------

def test_other_concept_names_are_listed_for_pairs_that_both_say_other(books):
    from tests.panel_fixtures import set_other
    a, b = synthetic_record(), synthetic_record()
    set_other(tn(a, 2), "shelf life")
    set_other(tn(b, 2), "how long the lids last")
    set_other(tn(a, 12), "founder's own investment")
    set_other(tn(b, 12), "Founder's own investment")
    set_other(tn(a, 14), "a patent")                    # only the primary says other here
    set_other(tn(b, 15), "resale")                      # only the replicate says other here
    rep = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    assert rep["codes"]["other_concept_names"] == [["shelf life", "how long the lids last"],
                                                   ["founder's own investment", "Founder's own investment"]]
    assert rep["codes"]["PR-CONCEPT"]["confusion"]["other|other"] == 1
    assert rep["codes"]["PR-FRAME(derived)"]["confusion"] == {
        "prevention|prevention": 2, "unclassified|prevention": 1, "unclassified|unclassified": 1}
    assert rep["codes"]["RS-FRAME(derived)"]["confusion"]["promotion|unclassified"] == 1
    # names are listed, not scored: the two differently worded names raise no flag
    assert sorted(f["ref"] for f in flags_of(rep, "code_disagreement")) == ["PR-CONCEPT", "RS-CONCEPT"]
    json.dumps(rep)


def test_founder_gender_disagreement_is_flagged(books):
    p, r = pair()
    assert align(p, r, books)["venture"]["founder_gender"] == {"primary": "all_men", "replicate": "all_men", "agree": True}
    p, r = pair(lambda d: d["venture"].update(founder_gender="mixed"))
    rep = align(p, r, books)
    assert rep["venture"]["founder_gender"] == {"primary": "all_men", "replicate": "mixed", "agree": False}
    assert [f["ref"] for f in flags_of(rep, "venture_disagreement")] == ["founder_gender"]
    assert list(rep["venture"]) == ["primary_industry", "product_form", "customer_type", "founder_gender", "ask"]


def test_the_ask_agrees_amount_and_equity_separately(books):
    assert align(*pair(), books)["venture"]["ask"] == {"amount_agree": True, "equity_agree": True}
    rep = align(*pair(lambda d: d["venture"].update(ask_amount_usd=250000)), books)
    assert rep["venture"]["ask"] == {"amount_agree": False, "equity_agree": True}
    unseen = dict(ask_amount_usd=None, ask_equity_pct=None, ask_missing="not_observable")
    a, b = synthetic_record(), synthetic_record()
    a["venture"].update(unseen)
    b["venture"].update(unseen)
    both_missing = align(PanelRecord.model_validate(a), PanelRecord.model_validate(b), books)
    assert both_missing["venture"]["ask"] == {"amount_agree": True, "equity_agree": True}
    one_missing = align(*pair(lambda d: d["venture"].update(unseen)), books)
    assert one_missing["venture"]["ask"] == {"amount_agree": False, "equity_agree": False}


def _reenter(d):
    d["decisions"].append({"decision_id": 5, "actor": "S3", "kind": "re_entry", "time_s": 175.0, "turn_id": None,
                           "offer_id": "O2", "terms": "$200,000 for 25%", "note": "Back with an offer."})


def test_a_re_entry_with_an_offer_id_counts_as_offering(books):
    p, r = pair(_reenter)
    rep = align(p, r, books)["decisions"]
    assert rep["ever_offer"]["S3"] == {"primary": False, "replicate": True}
    assert rep["ever_offer_agreement"] < 1.0
    p, r = pair()
    rep = align(p, r, books)["decisions"]
    assert rep["ever_offer"]["S3"] == {"primary": False, "replicate": False}


def test_re_entry_with_an_offer_id_counts_in_any_offer_and_membership(books):
    def only_reentry():
        d = synthetic_record()
        d["decisions"] = [x for x in d["decisions"] if x["kind"] == "exit"]
        _reenter(d)
        return PanelRecord.model_validate(d)
    rep = align(only_reentry(), only_reentry(), books)["decisions"]
    assert rep["any_offer"] == {"primary": True, "replicate": True}
    assert rep["offers"] and rep["offers"][0]["primary_members"] == ["S3"] and rep["offers"][0]["agree"]
