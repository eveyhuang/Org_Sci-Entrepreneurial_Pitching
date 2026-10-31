"""analysis.panel_dataset: a run as tidy tables, the outcomes O1 and O2, and the history cuts."""
from __future__ import annotations

import dataclasses
import inspect
import json
import math

import numpy as np
import pytest

from analysis.panel_dataset import (
    AFTER_CUT, FOLLOW_UP, FOLLOWUP_COLUMNS, H4_TABLES, NEXT_MOVES, NO_REACTION, OTHER_INVESTOR, PANEL_MOVES,
    RATING_CODES, RECEIPT_FUNCTIONS,
    PREPARED_NOT_RECORDED, PanelData, answer_followups, build_data, eligible_turns, h4_tables, load_run,
    panel_next_move, pitch_cut, pitch_outcome,
)
from analysis.archive.shark_level_dataset import investor_cuts, investor_outcomes
from pipeline.errors import ConfigError
from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import load_panel_books
from tests.panel_analysis_fixtures import (
    FORCED_PATTERNS, make_record, talk_record, write_ranking, write_run, write_video,
)
from tests.panel_fixtures import STAGE_A_IDS, prepared_codes, synthetic_record, turn


def one(tmp_path, record, video_id="vid") -> PanelData:
    write_video(tmp_path / "run", video_id, record)
    return load_run(tmp_path / "run")


def row(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    assert len(frame) == 1, f"{where} matches {len(frame)} rows"
    return frame.iloc[0]


# ---- tables ---------------------------------------------------------------------------------

def test_tables_have_the_declared_columns_and_keys(tmp_path):
    data = one(tmp_path, synthetic_record())
    assert list(data.videos.columns) == [
        "video_id", "source", "release_id", "duration_s", "n_turns", "founder_gender", "primary_industry",
        "product_form", "customer_type", "prepared_seconds", "ask_amount_usd", "ask_equity_pct", "ask_missing",
        "reviewed_validation_errors", "review_structural_on_decisions"]
    assert list(data.actors.columns) == ["video_id", "actor_id", "role", "label"]
    assert list(data.turns.columns) == [
        "video_id", "turn_id", "pos", "speaker", "role", "start_s", "end_s", "function", "addressed",
        "exchange_id", "exchange_kind", "responds_to", "concept", "frame", "frame_match", *RATING_CODES]
    assert list(data.exchanges.columns) == [
        "video_id", "exchange_id", "kind", "topic", "label", "first_pos", "last_pos", "n_turns"]
    assert list(data.decisions.columns) == [
        "video_id", "decision_id", "actor", "role", "kind", "time_s", "turn_id", "pos", "offer_id"]
    assert list(data.stage_a.columns) == ["video_id", "code", "value", "source"]
    assert list(data.prepared.columns) == ["video_id", "code", "value", "missing"]

    v = row(data.videos, video_id="vid")
    assert (v.source, v.release_id, v.n_turns, v.founder_gender) == ("primary", "v3-dev9", 19, "all_men")
    assert (v.primary_industry, v.product_form, v.customer_type) == ("home_household", "physical_good", "consumer")
    assert v.prepared_seconds == 55.0 and v.duration_s > 172
    assert (v.ask_amount_usd, v.ask_equity_pct, v.ask_missing) == (200000.0, 10.0, "none")
    assert len(data.turns) == 19 and list(data.turns.pos) == list(range(19))
    assert not data.turns.duplicated(["video_id", "turn_id"]).any()
    assert len(data.actors) == 4 and len(data.exchanges) == 4 and len(data.decisions) == 4
    assert len(data.prepared) == 4 and data.prepared.value.tolist() == [3.0, 4.0, 2.0, 5.0]
    assert len(data.stage_a) == 3 and set(data.stage_a.source) == {"rating"}
    assert set(data.stage_a.code) == set(STAGE_A_IDS)


def _set_release(vdir, release_id):
    path = vdir / "experiment.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "release_id": release_id}))


def test_a_v3_dev9_run_is_read_and_has_no_closings_table(tmp_path):
    _set_release(write_video(tmp_path / "run", "vid", synthetic_record()), "v3-dev9")
    data = load_run(tmp_path / "run")
    assert data.videos.release_id.tolist() == ["v3-dev9"]
    assert not hasattr(data, "closings")
    assert RECEIPT_FUNCTIONS == ("accepts",)


def test_a_v3_dev8_record_is_refused_with_the_release_it_was_annotated_with(tmp_path):
    vdir = write_video(tmp_path / "run", "vid", synthetic_record())
    for name in ("primary", "replicate"):
        path = vdir / "results" / f"{name}.json"
        result = json.loads(path.read_text())
        for e in result["record"]["exchanges"]:
            e["investors"] = [{"actor": "S1", "closing": "NONE", "closing_turn": None}]
        path.write_text(json.dumps(result))
    _set_release(vdir, "v3-dev8")
    with pytest.raises(ValueError, match="annotated with release v3-dev8; this code reads v3-dev9 records"):
        load_run(tmp_path / "run")


def test_a_record_without_the_prepared_presentation_codes_is_refused_with_its_release(tmp_path):
    vdir = write_video(tmp_path / "run", "vid", synthetic_record())
    for name in ("primary", "replicate"):
        path = vdir / "results" / f"{name}.json"
        result = json.loads(path.read_text())
        del result["record"]["prepared"]
        path.write_text(json.dumps(result))
    _set_release(vdir, "v3-dev8")
    with pytest.raises(ValueError, match="annotated with release v3-dev8; this code reads v3-dev9 records"):
        load_run(tmp_path / "run")


def test_turn_columns_carry_role_exchange_kind_links_concept_and_frame(tmp_path):
    data = one(tmp_path, synthetic_record())
    q = row(data.turns, turn_id=2)
    assert (q.speaker, q.role, q.function, q.exchange_kind) == ("S1", "investor", "question", "substantive")
    assert (q.concept, q.frame, q.responds_to, q.addressed) == ("operating_efficiency", "prevention", (), ("F1",))
    a = row(data.turns, turn_id=15)
    assert (a.role, a.responds_to, a.concept, a.frame) == ("founder", (14,), "forecast", "promotion")
    assert a.frame_match == "switch_to_promotion"
    assert row(data.turns, turn_id=3).frame_match == "match"
    plain = row(data.turns, turn_id=10)
    assert plain.exchange_id is None and plain.exchange_kind is None and plain.concept is None and plain.frame is None
    assert row(data.turns, turn_id=17).exchange_kind == "negotiation"
    ex = row(data.exchanges, exchange_id=1)
    assert (ex.kind, ex.topic, ex.first_pos, ex.last_pos, ex.n_turns) == ("substantive", "operating_efficiency", 1, 15, 9)


def test_rating_coding(tmp_path):
    record = synthetic_record()
    by_id = {t["turn_id"]: t for t in record["turns"]}
    codes = {c["code"]: c for c in by_id[5]["codes"]}
    codes["RS-RESP"]["value"] = "partial"
    codes["RS-UNCERT"]["value"] = "TRUE"            # case is repaired with the release's books
    {c["code"]: c for c in by_id[12]["codes"]}["RS-RESP"]["value"] = "deflects"
    {c["code"]: c for c in by_id[15]["codes"]}["RS-RESP"].update(value="NONE")
    data = one(tmp_path, record)
    first = row(data.turns, turn_id=3)
    assert (first["RS-RESP"], first["RS-EVID"], first["RS-UNCERT"]) == (1.0, 1.0, 0.0)
    assert (first["PREP-SUBS"], first["PREP-COH"]) == (3.0, 4.0)
    assert math.isnan(first["PREP-CTX"])                         # missing is not a number
    assert math.isnan(first["PR-SEV"])                           # an answer has no severity
    second = row(data.turns, turn_id=5)
    assert (second["RS-RESP"], second["RS-UNCERT"], second["PREP-SUBS"]) == (0.5, 1.0, 4.0)
    assert row(data.turns, turn_id=12)["RS-RESP"] == 0.0
    assert math.isnan(row(data.turns, turn_id=15)["RS-RESP"])    # NONE is not zero
    assert row(data.turns, turn_id=15)["RS-EVID"] == 0.0
    assert row(data.turns, turn_id=4)["PR-SEV"] == 3.0
    assert data.turns.loc[data.turns.function.eq("presents"), list(RATING_CODES)].isna().all().all()


def test_structure_labels_are_read_in_the_codebook_spelling(tmp_path):
    record = synthetic_record()
    record["turns"][1]["function"] = " Question "
    record["exchanges"][0]["kind"] = "Substantive"
    record["decisions"][0]["kind"] = "EXIT"
    record["venture"]["founder_gender"] = "All_Men"
    data = one(tmp_path, record)
    assert row(data.turns, turn_id=2).function == "question"
    assert row(data.exchanges, exchange_id=1).kind == "substantive"
    assert row(data.decisions, decision_id=1).kind == "exit"
    assert data.videos.founder_gender.iloc[0] == "all_men"


def test_reviewed_record_is_preferred_over_the_primary(tmp_path):
    reviewed = synthetic_record()
    reviewed["venture"]["founder_gender"] = "mixed"
    write_video(tmp_path / "run", "a", synthetic_record(), reviewed=reviewed)
    write_video(tmp_path / "run", "b", synthetic_record())
    data = load_run(tmp_path / "run")
    assert dict(zip(data.videos.video_id, data.videos.source)) == {"a": "reviewed", "b": "primary"}
    assert dict(zip(data.videos.video_id, data.videos.founder_gender)) == {"a": "mixed", "b": "all_men"}


def test_load_run_selects_videos_and_ignores_folders_that_are_not_videos(tmp_path):
    root = write_run(tmp_path, 3, seed=1, ranking=True)
    (root / "analysis" / "x").mkdir(parents=True)
    (root / "unfinished").mkdir()
    (root / "unfinished" / "experiment.json").write_text(json.dumps({"duration_s": 10.0, "release_id": "v3-dev9"}))
    assert list(load_run(root).videos.video_id) == ["v000", "v001", "v002"]
    assert list(load_run(root, ["v002", "v000"]).videos.video_id) == ["v000", "v002"]
    with pytest.raises(ConfigError):
        load_run(root, ["unfinished"])
    with pytest.raises(ConfigError):
        load_run(root, ["nowhere"])


def test_rank_scores_join_stage_a_when_the_run_has_them(tmp_path):
    root = write_run(tmp_path, 2, seed=1)
    assert set(load_run(root).stage_a.source) == {"rating"}
    write_ranking(root, ["v000", "v001", "elsewhere"], np.random.default_rng(0))
    stage_a = load_run(root).stage_a
    ranks = stage_a[stage_a.source == "rank_pl"]
    assert len(ranks) == 6 and set(ranks.video_id) == {"v000", "v001"} and set(ranks.code) == set(STAGE_A_IDS)
    scores = json.loads((root / "ranking" / "scores.json").read_text())
    assert row(ranks, video_id="v001", code="interpersonal_warmth").value == scores["interpersonal_warmth"]["pl_score"]["v001"]


def test_uncodable_stage_a_item_is_missing(tmp_path):
    record = synthetic_record()
    record["stage_a"][0].update(value=None, uncodable=True)
    stage_a = one(tmp_path, record).stage_a
    assert math.isnan(row(stage_a, code=STAGE_A_IDS[0]).value) and row(stage_a, code=STAGE_A_IDS[1]).value == 3


# ---- outcomes and cuts ----------------------------------------------------------------------

def patterns_run(tmp_path) -> PanelData:
    """The first pitch of a synthetic run: S1 offers, S2 joins, S3 exits, S4 exits, re-enters and offers,
    S5 never decides."""
    assert FORCED_PATTERNS == ("offer", "joins", "exit", "exit_reentry_offer", "none")
    return load_run(write_run(tmp_path, 1, seed=3))


def test_o2_for_each_decision_pattern(tmp_path):
    data = patterns_run(tmp_path)
    o2 = investor_outcomes(data)
    assert list(o2.columns) == ["video_id", "investor", "label", "ever_offer", "first_decision_kind",
                                "first_decision_pos", "ever_offer_source"]
    assert set(o2.ever_offer_source) == {"record"} and data.outcome_overrides == {}
    assert list(o2.investor) == ["S1", "S2", "S3", "S4", "S5"]
    got = dict(zip(o2.investor, o2.ever_offer))
    assert (got["S1"], got["S2"], got["S3"], got["S4"]) == (1.0, 1.0, 0.0, 1.0)
    assert math.isnan(got["S5"])
    kinds = dict(zip(o2.investor, o2.first_decision_kind))
    assert (kinds["S1"], kinds["S2"], kinds["S3"], kinds["S4"], kinds["S5"]) == (
        "offer", "joins_offer", "exit", "exit", None)
    d = data.decisions
    assert row(o2, investor="S4").first_decision_pos == d[(d.actor == "S4") & (d.kind == "exit")].pos.iloc[0]
    assert math.isnan(row(o2, investor="S5").first_decision_pos)
    assert set(o2.label) <= {a for a in data.actors.label}


def test_o1_any_offer_all_exit_and_unknown(tmp_path):
    offered = synthetic_record()
    all_exit = synthetic_record()
    all_exit["decisions"] = [
        {"decision_id": i + 1, "actor": a, "kind": "exit", "time_s": 111.0 + i, "turn_id": 10, "offer_id": None,
         "terms": "", "note": ""} for i, a in enumerate(("S1", "S2", "S3"))]
    unknown = synthetic_record()
    unknown["decisions"] = unknown["decisions"][:1]              # S3 exits; S1 and S2 are never seen deciding
    root = tmp_path / "run"
    for video_id, record in (("offered", offered), ("allexit", all_exit), ("unknown", unknown)):
        write_video(root, video_id, record)
    data = load_run(root)
    o1 = pitch_outcome(data)
    assert list(o1.columns) == ["video_id", "any_offer", "first_offer_pos", "any_offer_source"]
    assert row(o1, video_id="offered").any_offer == 1.0 and row(o1, video_id="offered").first_offer_pos == 16
    assert row(o1, video_id="allexit").any_offer == 0.0 and math.isnan(row(o1, video_id="allexit").first_offer_pos)
    assert math.isnan(row(o1, video_id="unknown").any_offer)
    o1_cut = pitch_cut(data)
    # a pitch's history stops at the first decision of any kind (here S3's exit at turn 10), offer or not
    assert dict(zip(o1_cut.video_id, o1_cut.cut_pos)) == {"allexit": 9, "offered": 9, "unknown": 9}


def test_cuts_for_each_decision_pattern(tmp_path):
    data = patterns_run(tmp_path)
    o2_cut, o1_cut = investor_cuts(data), pitch_cut(data)
    assert list(o2_cut.columns) == ["video_id", "investor", "cut_pos"]
    assert list(o1_cut.columns) == ["video_id", "cut_pos"]
    d = data.decisions
    cut = dict(zip(o2_cut.investor, o2_cut.cut_pos))

    def first(actor, kind):
        return int(d[(d.actor == actor) & (d.kind == kind)].pos.min())

    assert cut["S1"] == first("S1", "offer")
    assert cut["S2"] == first("S2", "joins_offer")
    assert cut["S3"] == first("S3", "exit")
    assert cut["S4"] == first("S4", "exit") < first("S4", "offer")     # the exit, not the later offer
    assert cut["S5"] == len(data.turns)                                # no decision: the whole encounter
    # the pitch stops at the first decision of any investor, here S1's offer, which the forced order puts first
    assert o1_cut.cut_pos.iloc[0] == int(d[d.role == "investor"].pos.min()) == first("S1", "offer")
    assert o2_cut.cut_pos.dtype.kind == "i" and o1_cut.cut_pos.dtype.kind == "i"


def test_decision_without_a_turn_takes_the_last_turn_starting_at_or_before_its_time(tmp_path):
    record = synthetic_record()
    record["decisions"][0].update(turn_id=None, time_s=112.5)   # turn 10 runs 111-114; turn 11 starts at 115
    record["decisions"][1].update(turn_id=None, time_s=1.0)     # before any turn starts
    record["decisions"][2].update(turn_id=999)                  # points at no turn: placed by time (164 s)
    data = one(tmp_path, record)
    assert row(data.decisions, decision_id=1).pos == 9
    assert row(data.decisions, decision_id=2).pos == 0
    assert row(data.decisions, decision_id=3).pos == 17
    assert row(data.decisions, decision_id=4).pos == 18
    assert data.decisions.pos.dtype.kind == "i"
    o2_cut = investor_cuts(data)
    assert dict(zip(o2_cut.investor, o2_cut.cut_pos)) == {"S1": 0, "S2": 17, "S3": 9}


# ---- eligible turns -------------------------------------------------------------------------

def test_eligible_turns_stop_before_the_cut_and_leave_out_decisions_and_negotiation(tmp_path):
    record = synthetic_record()
    record["turns"].append(turn(20, "F1", 173, 175, "negotiates", "Asks for better terms.", ["S1"], 4))
    record["turns"].append(turn(21, "S2", 175, 177, "accepts", "Nods at the terms.", ["F1"], 4))
    record["turns"].append(turn(22, "S2", 177, 179, "negotiates", "Names a price in passing.", ["F1"]))
    record["turns"].append(turn(23, "S1", 179, 181, "accepts", "Nods.", ["F1"]))
    data = one(tmp_path, record)
    everything = eligible_turns(data, "vid", 23)
    assert list(everything.columns) == list(data.turns.columns)
    # turn 10 decides (no exchange); 17-19 decide inside the negotiation; 20 negotiates; 21 sits in the
    # negotiation exchange; 22 negotiates outside any exchange
    assert list(everything.turn_id) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 23]
    assert list(eligible_turns(data, "vid", 9).turn_id) == list(range(1, 10))     # pos < cut, strictly
    assert eligible_turns(data, "vid", 0).empty
    assert eligible_turns(data, "elsewhere", 5).empty


def test_eligible_turns_leave_out_no_turn_of_the_focal_investor_since_interest_merged_into_accepts(tmp_path):
    from analysis.archive.shark_level_dataset import FOCAL_EXCLUDED_FUNCTIONS, eligible_turns_for_investor

    data = one(tmp_path, synthetic_record())
    assert FOCAL_EXCLUDED_FUNCTIONS == ()
    everyone = list(eligible_turns(data, "vid", 19).turn_id)
    assert {7, 13} <= set(everyone)                    # S1's and S2's accepts turns
    for focal in ("S1", "S2", "S3"):
        assert list(eligible_turns_for_investor(data, "vid", 19, focal).turn_id) == everyone


def test_build_data_matches_load_run(tmp_path):
    record = synthetic_record()
    from_disk = one(tmp_path, record)
    in_memory = build_data([{"video_id": "vid", "record": record, "source": "primary", "release_id": "v3-dev9",
                             "duration_s": float(from_disk.videos.duration_s.iloc[0])}])
    for name in ("videos", "actors", "turns", "exchanges", "decisions", "stage_a", "prepared"):
        left, right = getattr(from_disk, name), getattr(in_memory, name)
        assert left.astype(object).where(left.notna(), None).equals(right.astype(object).where(right.notna(), None)), name


def _reenter(d, offer_id):
    d["decisions"].append({"decision_id": 5, "actor": "S3", "kind": "re_entry", "time_s": 175.0, "turn_id": None,
                           "offer_id": offer_id, "terms": "$200,000 for 25%", "note": "Back with an offer."})


def test_exit_then_re_entry_with_an_offer_id_is_an_offer(tmp_path):
    record = synthetic_record()
    _reenter(record, "O2")
    data = one(tmp_path, record)
    o2, o1 = investor_outcomes(data), pitch_outcome(data)
    assert row(o2, investor="S3").ever_offer == 1.0
    assert row(o2, investor="S3").first_decision_kind == "exit"
    o2_cut = investor_cuts(data)
    d = data.decisions
    assert row(o2_cut, investor="S3").cut_pos == d[(d.actor == "S3") & (d.kind == "exit")].pos.min()


def test_re_entry_without_an_offer_id_is_not_an_offer(tmp_path):
    record = synthetic_record()
    _reenter(record, None)
    o2 = investor_outcomes(one(tmp_path, record))
    assert row(o2, investor="S3").ever_offer == 0.0


def test_first_offer_position_considers_an_offer_making_re_entry(tmp_path):
    record = synthetic_record()
    record["decisions"] = [x for x in record["decisions"] if x["kind"] not in ("offer", "joins_offer", "founder_accepts")]
    _reenter(record, "O2")
    data = one(tmp_path, record)
    o1 = pitch_outcome(data)
    d = data.decisions
    assert o1.any_offer.iloc[0] == 1.0
    assert o1.first_offer_pos.iloc[0] == d[d.kind == "re_entry"].pos.min()
    o1_cut = pitch_cut(data)
    assert o1_cut.cut_pos.iloc[0] == d[d.kind == "exit"].pos.min() < d[d.kind == "re_entry"].pos.min()


# ---- human outcome overrides ------------------------------------------------------------------

def _override(root, video_id, overrides):
    path = root / video_id / "results" / "reviewed.json"
    result = json.loads(path.read_text())
    result["outcome_overrides"] = overrides
    path.write_text(json.dumps(result))


def test_a_reviewed_outcome_override_replaces_ever_offer_and_any_offer_follows(tmp_path):
    root = tmp_path / "run"
    # in the synthetic record S1 offers, S2 joins and S3 exits
    for video_id in ("none", "flip", "lost", "alone"):
        write_video(root, video_id, synthetic_record(), reviewed=synthetic_record())
    _override(root, "flip", {"S3": "yes", "S1": "No"})
    _override(root, "lost", {"S1": "no", "S2": "no"})
    _override(root, "alone", {"S1": "unknown", "S2": "no"})
    data = load_run(root)
    assert data.outcome_overrides == {"flip": {"S3": "yes", "S1": "no"}, "lost": {"S1": "no", "S2": "no"},
                                      "alone": {"S1": "unknown", "S2": "no"}}
    o2, o1 = investor_outcomes(data), pitch_outcome(data)
    got = {(r.video_id, r.investor): (r.ever_offer, r.ever_offer_source) for r in o2.itertuples()}
    assert [got[("none", a)] for a in ("S1", "S2", "S3")] == [(1.0, "record"), (1.0, "record"), (0.0, "record")]
    assert [got[("flip", a)] for a in ("S1", "S2", "S3")] == [
        (0.0, "human_override"), (1.0, "record"), (1.0, "human_override")]
    assert got[("alone", "S2")] == (0.0, "human_override") and got[("alone", "S1")][1] == "human_override"
    assert math.isnan(got[("alone", "S1")][0])
    any_offer = dict(zip(o1.video_id, o1.any_offer))
    assert (any_offer["none"], any_offer["flip"], any_offer["lost"]) == (1.0, 1.0, 0.0)
    assert math.isnan(any_offer["alone"])                           # one unknown, the rest known 0
    # the decision list, and with it the cuts, still describe the record
    plain = load_run(root, ["none"])
    cut = investor_cuts(data)
    assert list(cut[cut.video_id == "flip"].cut_pos) == list(investor_cuts(plain).cut_pos)
    assert row(o2, video_id="flip", investor="S1").first_decision_kind == "offer"


def test_build_data_takes_overrides_and_refuses_an_unknown_value():
    entry = {"video_id": "v", "record": synthetic_record(), "source": "reviewed"}
    data = build_data([{**entry, "outcome_overrides": {"S3": "yes"}}])
    o2 = investor_outcomes(data)
    assert row(o2, investor="S3").ever_offer == 1.0 and row(o2, investor="S3").ever_offer_source == "human_override"
    assert build_data([entry]).outcome_overrides == {}
    with pytest.raises(ConfigError, match="maybe"):
        build_data([{**entry, "outcome_overrides": {"S3": "maybe"}}])


# ---- the pitch cut does not depend on the outcome ----------------------------------------------

def test_the_pitch_cut_is_the_first_decision_of_any_kind_and_the_end_without_one(tmp_path):
    offered, all_exit, silent = synthetic_record(), synthetic_record(), synthetic_record()
    all_exit["decisions"] = [
        {"decision_id": i + 1, "actor": a, "kind": "exit", "time_s": t, "turn_id": turn_id, "offer_id": None,
         "terms": "", "note": ""} for i, (a, t, turn_id) in enumerate((("S2", 151.0, 16), ("S3", 111.0, 10)))]
    silent["decisions"] = [d for d in silent["decisions"] if d["actor"] == "F1"]     # the founder alone decides
    root = tmp_path / "run"
    for video_id, record in (("offered", offered), ("allexit", all_exit), ("silent", silent)):
        write_video(root, video_id, record)
    data = load_run(root)
    o1 = pitch_outcome(data)
    o1_cut = pitch_cut(data)
    cut = dict(zip(o1_cut.video_id, o1_cut.cut_pos))
    # the same first decision (S3 leaves at turn 10) cuts the pitch with an offer and the one without
    assert cut["offered"] == cut["allexit"] == 9 and cut["silent"] == len(silent["turns"]) == 19
    assert row(o1, video_id="offered").any_offer == 1.0 and row(o1, video_id="offered").first_offer_pos == 16


def test_an_investor_whose_only_move_revises_or_withdraws_an_offer_has_made_one(tmp_path):
    from pipeline.stage_b.panel_books import implies_offer, makes_offer
    assert implies_offer("revises_offer") and implies_offer("withdraws_offer") and implies_offer("offer")
    assert implies_offer("re_entry", "O2") and not implies_offer("re_entry") and not implies_offer("exit")
    assert not makes_offer("revises_offer") and not makes_offer("withdraws_offer")
    for kind in ("revises_offer", "withdraws_offer"):
        record = synthetic_record()
        record["decisions"][1].update(kind=kind)             # S1's only decision, at turn 17
        data = build_data([{"video_id": "v", "record": record, "source": "primary"}])
        o2, o1 = investor_outcomes(data), pitch_outcome(data)
        assert row(o2, investor="S1").ever_offer == 1.0 and o1.any_offer.iloc[0] == 1.0
        o2_cut = investor_cuts(data)
        assert row(o2_cut, investor="S1").cut_pos == 16      # the history stops at that move, not at the end


# ---- reviewed records with open issues ----------------------------------------------------------

def test_a_reviewed_record_with_open_issues_is_read_and_marked(tmp_path):
    root = tmp_path / "run"
    for video_id in ("clean", "errors", "structural", "primary"):
        write_video(root, video_id, synthetic_record(),
                    reviewed=None if video_id == "primary" else synthetic_record())
    (root / "clean" / "results" / "reviewed_validation.json").write_text(json.dumps({"errors": [], "flags": ["x"]}))
    (root / "errors" / "results" / "reviewed_validation.json").write_text(
        json.dumps({"errors": ["[decision.actor] decision 2: ...", "[turn.link] turn 3: ..."], "flags": []}))
    log = [
        {"path": "decision:2/present", "resolution": "structural", "new": "absent", "implied": False},
        {"path": "replicate_only:decision@2:40", "resolution": "structural", "new": "present", "implied": False},
        {"path": "decision:3/kind", "resolution": "structural", "new": None, "implied": False},
        {"path": "decision:3/kind", "resolution": "correct", "new": "offer", "implied": False},   # later, settled
        {"path": "decision:1/present", "resolution": "confirm_primary", "new": "present", "implied": False},
        {"path": "turn:4/present", "resolution": "structural", "new": "absent", "implied": False},
        {"path": "exchange:1/topic", "resolution": "structural", "new": "not_an_exchange", "implied": False},
    ]
    for video_id in ("structural", "primary"):       # a log without a reviewed record marks nothing
        (root / video_id / "review").mkdir()
        (root / video_id / "review" / "review_log.jsonl").write_text("".join(json.dumps(e) + "\n" for e in log))
    videos = load_run(root).videos.set_index("video_id")
    assert dict(videos.source) == {"clean": "reviewed", "errors": "reviewed_with_open_issues",
                                   "structural": "reviewed_with_open_issues", "primary": "primary"}
    assert dict(videos.reviewed_validation_errors) == {"clean": 0, "errors": 2, "structural": 0, "primary": 0}
    assert dict(videos.review_structural_on_decisions) == {"clean": 0, "errors": 0, "structural": 2, "primary": 0}
    assert videos.reviewed_validation_errors.dtype.kind == "i"


# ---- the run's own instrument ---------------------------------------------------------------------

def _bundle(vdir, drop_stage_a=None):
    """Copy the live instrument into a video's bundle folder, as a run does; optionally without one Stage A item."""
    import shutil

    import yaml

    from pipeline.instrument import canonical_files, load_release
    release = load_release()
    (vdir / "bundle").mkdir()
    for name in canonical_files(release):
        shutil.copy2(release.canonical_dir / name, vdir / "bundle" / name)
    if drop_stage_a:
        path = vdir / "bundle" / release.books["stage_a"].yaml.name
        entries = [e for e in yaml.safe_load(path.read_text()) if e["id"] != drop_stage_a]
        path.write_text(yaml.safe_dump(entries))


def test_values_and_stage_a_items_come_from_the_runs_own_bundle_when_it_has_one(tmp_path):
    from analysis.panel_features import pitch_features
    root = tmp_path / "run"
    dropped = STAGE_A_IDS[-1]
    record = synthetic_record()
    record["stage_a"] = [i for i in record["stage_a"] if i["code"] != dropped]
    before = {}
    for video_id in ("a", "b"):
        _bundle(write_video(root, video_id, record), drop_stage_a=dropped)
        before[video_id] = sorted((p.name, p.read_bytes()) for p in (root / video_id / "bundle").iterdir())
    data = load_run(root)
    assert data.books_source == {"a": "bundle", "b": "bundle"}
    assert data.stage_a_codes == tuple(c for c in STAGE_A_IDS if c != dropped)
    table = pitch_features(data)
    assert f"sa_{dropped}" not in table.columns and f"sa_{STAGE_A_IDS[0]}" in table.columns
    for video_id in ("a", "b"):          # reading a bundle never rewrites it
        assert sorted((p.name, p.read_bytes()) for p in (root / video_id / "bundle").iterdir()) == before[video_id]
    # without a bundle the live release is used, and the tables say so
    write_video(tmp_path / "live", "c", synthetic_record())
    live = load_run(tmp_path / "live")
    assert live.books_source == {"c": "live"} and live.stage_a_codes == tuple(STAGE_A_IDS)
    assert f"sa_{dropped}" in pitch_features(live).columns


# ---- the cut, spelled out turn by turn ------------------------------------------------------------

def test_exclusion_reasons_say_why_and_agree_with_eligible_turns(tmp_path):
    from analysis.archive.shark_level_dataset import (
        REASON_OWN_INTEREST, eligible_turns_for_investor, exclusion_reasons_for_investor,
    )
    from analysis.panel_dataset import REASON_AFTER_CUT, REASON_DECISION_TALK, REASON_NEGOTIATION, exclusion_reasons

    data = load_run(write_run(tmp_path, 8, seed=3, planted="pitch_draw"))
    o2_cut, o1_cut = investor_cuts(data), pitch_cut(data)
    seen = set()
    for r in o2_cut.itertuples():
        reasons = exclusion_reasons_for_investor(data, r.video_id, int(r.cut_pos), r.investor)
        turns = data.turns[data.turns.video_id == r.video_id]
        assert list(reasons.index) == list(turns.index)                       # every turn of the video, no other
        eligible = eligible_turns_for_investor(data, r.video_id, int(r.cut_pos), r.investor)
        assert list(reasons.index[reasons == ""]) == list(eligible.index)
        assert (reasons[turns.pos >= r.cut_pos] == REASON_AFTER_CUT).all()    # the cut is named first
        before = turns[turns.pos < r.cut_pos]
        assert (reasons[before.index[before.function.isin(("decides", "negotiates"))]] == REASON_DECISION_TALK).all()
        seen |= set(reasons)
    assert {"", REASON_AFTER_CUT, REASON_DECISION_TALK} <= seen and REASON_OWN_INTEREST not in seen
    for r in o1_cut.itertuples():                                             # a pitch row has no focal investor
        reasons = exclusion_reasons(data, r.video_id, int(r.cut_pos))
        assert list(reasons.index[reasons == ""]) == list(eligible_turns(data, r.video_id, int(r.cut_pos)).index)
        assert REASON_OWN_INTEREST not in set(reasons)
    # a negotiation turn before the cut that is neither decides nor negotiates is named as such
    video = data.turns[(data.turns.exchange_kind == "negotiation")].video_id.iloc[0]
    everything = exclusion_reasons(data, video, 10 ** 6)
    in_negotiation = data.turns[(data.turns.video_id == video) & (data.turns.exchange_kind == "negotiation")]
    assert set(everything[in_negotiation.index]) <= {REASON_NEGOTIATION, REASON_DECISION_TALK}


# ---- the questioner's next move after an answer ---------------------------------------------------

def _talk(turns, **kw):
    return build_data([{"video_id": "vid", "record": talk_record(turns, **kw), "source": "primary"}])


def _moves(data, **kw):
    table = answer_followups(data, **kw)
    return {(int(r.answer_turn), int(r.question_turn)): r for r in table.itertuples()}


def test_answer_followups_has_one_row_per_answer_and_question_with_the_codes():
    data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "accepts", 1, (3,), None)])
    table = answer_followups(data)
    assert list(table.columns) == list(FOLLOWUP_COLUMNS) and len(table) == 1
    r = table.iloc[0]
    assert (r.video_id, r.answer_turn, r.answer_pos, r.question_turn, r.asker, r.asker_label) == (
        "vid", 3, 2, 2, "S1", "Investor S1")
    assert (r.exchange_id, r.question_sev, bool(r.hard_question)) == (1, 2.0, True)
    assert (r.resp, r.evid, r.uncert, r.subs, r.coh) == (1.0, 1.0, 0.0, 4.0, 4.0) and math.isnan(r.ctx)
    assert (r.next_move, r.next_turn, r.next_actor, r.next_function, r.turns_to_next) == (
        "positive_receipt", 4, "S1", "accepts", 1)
    assert math.isnan(r.next_question_sev) and math.isnan(r.escalation)
    assert set(NEXT_MOVES) == {"positive_receipt", "follow_up_question", "challenge", "other_by_asker",
                               "other_investor", "no_reaction"}
    assert NO_REACTION == "no_reaction"


@pytest.mark.parametrize("function, move", [
    ("accepts", "positive_receipt"), ("question", "follow_up_question"), ("challenges", "challenge"),
    ("hostile", "other_by_asker")])
def test_the_askers_own_next_turn_names_the_move(function, move):
    data = _talk([("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("S1", function, 1, (), 3)])
    assert _moves(data)[(3, 2)].next_move == move


def test_another_investor_speaking_first_is_recorded_with_who_and_what():
    data = _talk([("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("S2", "challenges", 1, (), None),
                  ("S1", "accepts", 1, (), None)])
    r = _moves(data)[(3, 2)]
    assert (r.next_move, r.next_actor, r.next_function, r.next_turn) == ("other_investor", "S2", "challenges", 4)


def test_next_move_is_the_first_investor_turn_whatever_the_exchange():
    two = ((1, "substantive"), (2, "substantive"))
    for speaker, move in (("S1", FOLLOW_UP), ("S2", OTHER_INVESTOR)):
        data = _talk([("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), (speaker, "question", 2, (), 2)],
                     exchanges=two)
        r = _moves(data)[(3, 2)]
        assert (r.next_move, r.next_turn, r.next_actor, r.next_function) == (move, 4, speaker, "question"), speaker
    # an answer outside any exchange reads on into an exchange; a founder turn between is passed over
    data = _talk([("S1", "question", None, (), 1), ("F1", "answer", None, (2,), 3), ("F1", "answer", 1, (), 2),
                  ("S1", "accepts", None, (), None)])
    assert _moves(data)[(3, 2)].next_move == "positive_receipt"


def test_no_investor_turn_after_the_answer_is_no_reaction():
    data = _talk([("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("F2", "answer", 1, (), 3)])
    r = _moves(data)[(3, 2)]
    assert r.next_move == NO_REACTION and r.next_actor is None and r.next_function is None
    assert math.isnan(r.next_turn) and math.isnan(r.turns_to_next)


def test_next_move_passes_over_social_and_none_turns_and_founder_continuations():
    data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3),
                  ("F2", "answer", 1, (2,), 4),                    # the co-founder adds to the answer
                  ("S2", "social", 1, (), None), ("F1", "social", 1, (), None),
                  ("S2", "demonstrates", 1, (), None), ("S2", "NONE", 1, (), None),
                  ("S1", "accepts", 1, (), None)])
    moves = _moves(data)
    assert (moves[(3, 2)].next_move, moves[(3, 2)].next_turn, moves[(3, 2)].turns_to_next) == ("positive_receipt", 9, 6)
    assert (moves[(4, 2)].next_move, moves[(4, 2)].turns_to_next) == ("positive_receipt", 5)


def test_a_reopened_exchange_is_not_followed_past_the_turns_of_other_exchanges():
    data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3),
                  ("S2", "question", 2, (), 1), ("F1", "answer", 2, (4,), 2), ("S2", "challenges", None, (), None),
                  ("S1", "question", 1, (), 3), ("F1", "answer", 1, (7,), 2)],
                 exchanges=((1, "substantive"), (2, "substantive")))
    moves = _moves(data)
    r = moves[(3, 2)]
    assert (r.next_move, r.next_turn, r.next_actor, r.turns_to_next) == (OTHER_INVESTOR, 4, "S2", 1)
    assert math.isnan(r.next_question_sev) and math.isnan(r.escalation)
    assert (moves[(5, 4)].next_move, moves[(5, 4)].next_turn) == ("challenge", 6)   # whatever its exchange
    assert moves[(8, 7)].next_move == NO_REACTION


def test_escalation_is_the_follow_up_severity_minus_the_question_severity():
    data = _talk([("S1", "question", 1, (), 3), ("F1", "answer", 1, (2,), 3), ("S1", "question", 1, (), 1),
                  ("F1", "answer", 1, (4,), 3), ("S1", "question", 1, (), 1), ("F1", "answer", 1, (6,), 3),
                  ("S1", "challenges", 1, (), None)])
    moves = _moves(data)
    assert (moves[(3, 2)].escalation, moves[(5, 4)].escalation) == (-2.0, 0.0)
    assert math.isnan(moves[(7, 6)].escalation) and moves[(7, 6)].next_move == "challenge"
    assert [bool(moves[k].hard_question) for k in ((3, 2), (5, 4))] == [True, False]


def test_decision_as_next_turn_is_no_reaction_and_at_the_cut_is_after_cut():
    for function in ("decides", "negotiates"):
        data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3), ("S1", function, None, (), None),
                      ("S1", "accepts", 1, (), None)])                  # turn 4, the decision talk, at position 3
        r = _moves(data)[(3, 2)]
        assert (r.next_move, r.next_actor, r.next_function) == (NO_REACTION, None, None), function
        assert math.isnan(r.next_turn) and math.isnan(r.turns_to_next)
        assert _moves(data, cut={"vid": {"S1": 3}})[(3, 2)].next_move == AFTER_CUT, function   # the decision is the cut
        assert _moves(data, cut={"vid": {"S1": 4}})[(3, 2)].next_move == NO_REACTION, function
    data = _talk([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3), ("S1", "accepts", 2, (), None)],
                 exchanges=((1, "substantive"), (2, "negotiation")))
    assert _moves(data)[(3, 2)].next_move == NO_REACTION                # a turn of a negotiation exchange
    assert _moves(data, cut={"vid": {"S1": 3}})[(3, 2)].next_move == AFTER_CUT


def test_an_answer_linked_to_several_questions_gives_a_row_per_question_and_other_links_give_none():
    data = _talk([("S1", "question", 1, (), 1), ("S2", "question", 1, (), 3), ("F1", "answer", 1, (2, 3), 4),
                  ("S2", "accepts", 1, (), None), ("S1", "challenges", 1, (), None), ("F1", "answer", 1, (6,), 2)])
    moves = _moves(data)
    assert set(moves) == {(4, 2), (4, 3)}                                # the answer to the challenge has no question
    assert (moves[(4, 2)].asker, moves[(4, 2)].next_move, moves[(4, 2)].next_actor) == ("S1", "other_investor", "S2")
    assert (moves[(4, 3)].asker, moves[(4, 3)].next_move) == ("S2", "positive_receipt")


def test_with_a_cut_a_next_move_at_or_after_the_askers_cut_is_after_cut_and_carries_nothing():
    turns = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3), ("S1", "question", 1, (), 3),
             ("F1", "answer", 1, (4,), 3), ("S1", "accepts", 1, (), None)]
    data = _talk(turns)
    assert [r.next_move for r in answer_followups(data).itertuples()] == ["follow_up_question", "positive_receipt"]
    cut = _moves(data, cut={"vid": {"S1": 5}})                           # the agreement is at position 5
    assert cut[(3, 2)].next_move == "follow_up_question" and cut[(3, 2)].escalation == 1.0
    late = cut[(5, 4)]
    assert late.next_move == "after_cut" and late.next_actor is None and late.next_function is None
    assert all(math.isnan(v) for v in (late.next_turn, late.next_pos, late.turns_to_next, late.escalation))
    assert _moves(data, cut={"vid": {"S1": 6}})[(5, 4)].next_move == "positive_receipt"   # strictly before the cut
    assert _moves(data, cut={"vid": {"S1": 3}})[(3, 2)].next_move == "after_cut"
    assert _moves(data, cut={"vid": {"S2": 0}})[(3, 2)].next_move == "follow_up_question"  # another investor's cut
    assert len(answer_followups(data, video_ids=["other"])) == 0
    assert list(answer_followups(data, video_ids=["other"]).columns) == list(FOLLOWUP_COLUMNS)


def test_exchange_columns_are_kept_as_before():
    turns = [("S1", "question", None, (), 1), ("F1", "answer", None, (2,), 3),   # 3: outside any exchange
             ("S1", "question", 1, (), 2), ("F1", "answer", 1, (4,), 3),         # 5: two more questions in 1
             ("S2", "question", 1, (), 1), ("F1", "answer", 1, (6,), 3),         # 7: the next question is in 2
             ("S1", "question", 2, (), 3), ("F1", "answer", 2, (8,), 3),         # 9: none more in 2; next in 1
             ("S2", "question", 1, (), 2), ("F1", "answer", 1, (10,), 3),        # 11: no later question at all
             ("S1", "challenges", 1, (), None)]                                  # a challenge is no question
    data = _talk(turns, exchanges=((1, "substantive"), (2, "substantive")))
    moves = _moves(data)
    same = {k[0]: r.next_question_same_exchange for k, r in moves.items()}
    further = {k[0]: r.further_questions_in_exchange for k, r in moves.items()}
    assert math.isnan(same[3]) and math.isnan(same[11])
    assert (same[5], same[7], same[9]) == (1.0, 0.0, 0.0)
    assert further == {3: 0, 5: 2, 7: 1, 9: 0, 11: 0}
    assert {"next_question_same_exchange", "further_questions_in_exchange"} <= set(FOLLOWUP_COLUMNS)
    assert answer_followups(data).further_questions_in_exchange.dtype == np.int64
    cut = _moves(data, cut={"vid": {"S1": 9, "S2": 9}})                   # turn 10 sits at position 9
    assert {k[0]: r.further_questions_in_exchange for k, r in cut.items()} == {3: 0, 5: 1, 7: 0, 9: 0, 11: 0}
    assert math.isnan(cut[(9, 8)].next_question_same_exchange)            # its next question is at the cut


def _talk_concepts(turns, concepts: dict[int, str], **kw):
    """`_talk` with the question concepts set by turn id (talk_record gives every question `sales`)."""
    record = talk_record(turns, **kw)
    for t in record["turns"]:
        for c in t["codes"]:
            if c["code"] == "PR-CONCEPT" and t["turn_id"] in concepts:
                c["value"] = concepts[t["turn_id"]]
    return build_data([{"video_id": "vid", "record": record, "source": "primary"}])


def _qa(n):
    """n question-answer pairs by S1: questions are turns 2, 4, 6, ..., answers 3, 5, 7, ..."""
    out = []
    for i in range(n):
        out += [("S1", "question", 1, (), 1), ("F1", "answer", 1, (2 + 2 * i,), 3)]
    return out


def test_same_frame_columns():
    new = ("next_question_same_frame", "further_questions_same_frame", "next_question_same_concept")
    at = FOLLOWUP_COLUMNS.index("further_questions_in_exchange")
    assert FOLLOWUP_COLUMNS[at + 1:at + 4] == new
    # promotion answered; prevention next, promotion after: switched, no run
    data = _talk_concepts(_qa(3), {2: "sales", 4: "execution", 6: "sales"})
    assert data.turns.set_index("turn_id").frame[[2, 4, 6]].tolist() == ["promotion", "prevention", "promotion"]
    r = _moves(data)[(3, 2)]
    assert (r.next_question_same_frame, r.further_questions_same_frame) == (0.0, 0)
    # promotion answered; promotion, promotion, prevention next: a run of two
    data = _talk_concepts(_qa(4), {2: "sales", 4: "vision", 6: "sales", 8: "competition"})
    moves = _moves(data)
    assert (moves[(3, 2)].next_question_same_frame, moves[(3, 2)].further_questions_same_frame) == (1.0, 2)
    assert (moves[(5, 4)].next_question_same_frame, moves[(5, 4)].further_questions_same_frame) == (1.0, 1)
    assert (moves[(7, 6)].next_question_same_frame, moves[(7, 6)].further_questions_same_frame) == (0.0, 0)
    last = moves[(9, 8)]                                                   # no later question
    assert math.isnan(last.next_question_same_frame) and last.further_questions_same_frame == 0
    assert answer_followups(data).further_questions_same_frame.dtype == np.int64
    # with a cut before the later questions (turn 4 sits at position 3)
    cut = _moves(data, cut={"vid": {"S1": 3}})[(3, 2)]
    assert math.isnan(cut.next_question_same_frame) and cut.further_questions_same_frame == 0
    # an unclassified frame answered, or next, is not a switch
    data = _talk_concepts(_qa(3), {2: "deal_terms", 4: "sales", 6: "sales"})
    assert data.turns.set_index("turn_id").frame[2] == "unclassified"
    r = _moves(data)[(3, 2)]
    assert math.isnan(r.next_question_same_frame) and r.further_questions_same_frame == 0
    data = _talk_concepts(_qa(3), {2: "sales", 4: "deal_terms", 6: "sales"})
    r = _moves(data)[(3, 2)]
    assert math.isnan(r.next_question_same_frame) and r.further_questions_same_frame == 0
    # a NEUTRAL frame (concept NONE), answered or next, is not a switch either
    data = _talk_concepts(_qa(3), {2: "NONE", 4: "sales", 6: "execution"})
    assert data.turns.set_index("turn_id").frame[2] == "NEUTRAL"
    moves = _moves(data)
    assert math.isnan(moves[(3, 2)].next_question_same_frame) and moves[(3, 2)].further_questions_same_frame == 0
    data = _talk_concepts(_qa(3), {2: "execution", 4: "NONE", 6: "execution"})
    r = _moves(data)[(3, 2)]
    assert math.isnan(r.next_question_same_frame) and r.further_questions_same_frame == 0
    # a cut inside a run: only the questions before the cut count (turn 6 sits at position 5)
    data = _talk_concepts(_qa(3), {2: "sales", 4: "sales", 6: "vision"})
    assert _moves(data)[(3, 2)].further_questions_same_frame == 2
    r = _moves(data, cut={"vid": {"S1": 5}})[(3, 2)]
    assert (r.next_question_same_frame, r.further_questions_same_frame) == (1.0, 1)


def test_same_concept_column():
    data = _talk_concepts(_qa(3), {2: "sales", 4: "sales", 6: "vision"})   # sales and vision: both promotion
    moves = _moves(data)
    assert moves[(3, 2)].next_question_same_concept == 1.0
    assert (moves[(5, 4)].next_question_same_concept, moves[(5, 4)].next_question_same_frame) == (0.0, 1.0)
    assert math.isnan(moves[(7, 6)].next_question_same_concept)                # no later question
    assert math.isnan(_moves(data, cut={"vid": {"S1": 3}})[(3, 2)].next_question_same_concept)
    # `other` is never the same concept, not even as another `other`; nor is NONE compared
    data = _talk_concepts(_qa(3), {2: "other", 4: "other", 6: "sales"})
    moves = _moves(data)
    assert math.isnan(moves[(3, 2)].next_question_same_concept) and math.isnan(moves[(5, 4)].next_question_same_concept)


# ---- H4: the next move, counted --------------------------------------------------------------------

H4_TALK = [
    ("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "accepts", 1, (), None),          # 3: receipt
    ("S1", "question", 1, (), 1), ("F1", "answer", 1, (5,), 2), ("S2", "accepts", 1, (), None),         # 6: other investor
    ("S2", "question", 1, (), 3), ("F1", "answer", 1, (8,), 2), ("S2", "challenges", 1, (), None),      # 9: challenge
    ("S1", "question", 1, (), 1), ("F1", "answer", 1, (11,), 3), ("S2", "question", 1, (), 1),          # 12: other investor
    ("F1", "answer", 1, (13,), 3), ("S2", "hostile", 1, (), None),                                      # 14: other by asker
    ("S1", "question", 2, (), 2), ("F1", "answer", 2, (16,), 1),                                        # 17: ends
]


def _h4_data():
    record = talk_record(H4_TALK, exchanges=((1, "substantive"), (2, "substantive")))
    for t in record["turns"]:
        if t["turn_id"] == 17:                                   # one answer without a substance or support rating
            for c in t["codes"]:
                if c["code"] in ("PREP-SUBS", "RS-EVID"):
                    c.update(value=None, missing="not_applicable")
    other = talk_record([("S1", "question", 1, (), 3), ("F1", "answer", 1, (2,), 5), ("S1", "question", 1, (), 1)])
    return build_data([{"video_id": "vid", "record": record, "source": "primary"},
                       {"video_id": "wid", "record": other, "source": "primary"}])


def test_panel_next_move_reads_the_first_investor_reaction_whoever_gives_it():
    table = answer_followups(_h4_data())
    by_answer = dict(zip(zip(table.video_id, table.answer_turn.astype(int)), panel_next_move(table)))
    assert by_answer == {("vid", 3): "receipt", ("vid", 6): "receipt", ("vid", 9): "challenge", ("vid", 12): "question",
                         ("vid", 14): "other", ("vid", 17): "none", ("wid", 3): "question"}
    assert PANEL_MOVES == ("receipt", "question", "challenge", "other", "none")
    cut = answer_followups(_h4_data(), cut={"vid": {"S1": 0}})
    assert "after_cut" in set(panel_next_move(cut))


def test_h4_tables_count_the_askers_and_the_panels_next_move_for_all_and_hard_questions():
    tables = h4_tables(answer_followups(_h4_data()))
    assert tuple(tables) == H4_TABLES
    asker = tables["asker"].set_index("next_move")
    assert list(asker.index) == list(NEXT_MOVES) and list(asker.columns) == ["all", "hard"]
    assert asker["all"].to_dict() == {"positive_receipt": 1, "follow_up_question": 1, "challenge": 1,
                                      "other_by_asker": 1, "other_investor": 2, "no_reaction": 1}
    assert asker["hard"].to_dict() == {"positive_receipt": 1, "follow_up_question": 1, "challenge": 1,
                                       "other_by_asker": 0, "other_investor": 0, "no_reaction": 1}
    panel = tables["panel"].set_index("next_move")
    assert list(panel.index) == list(PANEL_MOVES)
    assert panel["all"].to_dict() == {"receipt": 2, "question": 2, "challenge": 1, "other": 1, "none": 1}
    assert panel["hard"].to_dict() == {"receipt": 1, "question": 1, "challenge": 1, "other": 0, "none": 1}
    assert asker["all"].sum() == panel["all"].sum() == 7 and asker["hard"].sum() == panel["hard"].sum() == 4


def test_h4_tables_split_by_substance_support_and_video_with_the_rows_behind_each_level():
    tables = h4_tables(answer_followups(_h4_data()))
    subs = tables["panel_by_substance"]
    assert list(subs.columns) == ["questions", "substance", *PANEL_MOVES, "n"]
    assert list(subs.substance[subs.questions == "all"]) == ["1", "2", "3", "4", "5", "not_rated"]
    every = subs[subs.questions == "all"].set_index("substance")
    assert every.n.to_dict() == {"1": 0, "2": 2, "3": 2, "4": 1, "5": 1, "not_rated": 1}
    assert every.loc["2", list(PANEL_MOVES)].to_dict() == {"receipt": 1, "question": 0, "challenge": 1, "other": 0, "none": 0}
    assert every.loc["not_rated", "none"] == 1 and every.loc["5", "question"] == 1
    hard = subs[subs.questions == "hard"].set_index("substance")
    assert hard.n.to_dict() == {"1": 0, "2": 1, "3": 0, "4": 1, "5": 1, "not_rated": 1}
    support = tables["asker_by_support"]
    assert list(support.columns) == ["questions", "support", *NEXT_MOVES, "n"]
    every = support[support.questions == "all"].set_index("support")
    assert every.n.to_dict() == {"yes": 4, "no": 2, "not_rated": 1}          # support when substance is 3 or more
    assert every.loc["no", ["other_investor", "challenge"]].to_dict() == {"other_investor": 1, "challenge": 1}
    assert every.loc["yes", "other_by_asker"] == 1 and every.loc["not_rated", "no_reaction"] == 1
    videos = tables["asker_by_video"]
    assert list(videos.video_id) == ["vid", "wid", "vid", "wid"] and list(videos.n) == [6, 1, 3, 1]
    assert videos.loc[1, "follow_up_question"] == 1
    for name in H4_TABLES[1:4] + H4_TABLES[5:]:                             # every split adds up to its n
        t = tables[name]
        moves = [c for c in t.columns if c not in ("questions", "n", t.columns[1])]
        assert (t[moves].sum(axis=1) == t.n).all(), name
        assert t.n[t.questions == "all"].sum() == 7 and t.n[t.questions == "hard"].sum() == 4, name


def test_h4_tables_refuse_rows_cut_by_a_history_cut_and_cope_with_no_rows():
    with pytest.raises(ValueError, match="without a cut"):
        h4_tables(answer_followups(_h4_data(), cut={"vid": {"S1": 0}}))
    empty = h4_tables(answer_followups(_h4_data(), video_ids=["nobody"]))
    assert tuple(empty) == H4_TABLES and empty["asker"]["all"].sum() == 0 and empty["panel"].hard.sum() == 0
    assert empty["panel_by_video"].empty and list(empty["panel_by_video"].columns) == ["questions", "video_id", *PANEL_MOVES, "n"]
    assert (empty["asker_by_substance"].n == 0).all() and len(empty["asker_by_substance"]) == 12


def test_a_refused_record_without_a_release_id_names_an_earlier_release(tmp_path):
    vdir = write_video(tmp_path / "run", "vid", synthetic_record())
    for name in ("primary", "replicate"):
        path = vdir / "results" / f"{name}.json"
        result = json.loads(path.read_text())
        del result["record"]["prepared"]
        path.write_text(json.dumps(result))
    path = vdir / "experiment.json"
    path.write_text(json.dumps({k: v for k, v in json.loads(path.read_text()).items() if k != "release_id"}))
    with pytest.raises(ValueError, match="annotated with an earlier release; this code reads v3-dev9 records") as exc:
        load_run(tmp_path / "run")
    assert "None" not in str(exc.value)


# ---- the prepared presentation, the ask and the pitch override (v3-dev9) ----------------------------

def test_prepared_table_holds_one_row_per_prepared_code(tmp_path):
    run = write_run(tmp_path, 3, seed=1)
    data = load_run(run)
    codes = [e.id for e in load_panel_books(load_release()).prepared_codes()]
    assert list(data.prepared.columns) == ["video_id", "code", "value", "missing"]
    assert set(data.prepared.code) == set(codes)
    assert len(data.prepared) == 3 * len(codes)
    assert data.prepared.loc[data.prepared.code == "PRES-EVID", "value"].ge(0).all()
    ordinal = data.prepared[data.prepared.code != "PRES-EVID"].value
    assert ordinal.between(1, 5).all() and set(data.prepared.missing) == {"none"}
    assert data.prepared.value.nunique() > 1                     # drawn per pitch, not one constant


def test_a_missing_prepared_rating_is_nan_with_its_reason(tmp_path):
    record = synthetic_record()
    record["prepared"] = [*prepared_codes()[:2], {"code": "PRES-CTX", "value": None, "missing": "not_observable"},
                          {"code": "PRES-EVID", "value": "0", "missing": "none"}]
    prepared = one(tmp_path, record).prepared
    assert math.isnan(row(prepared, code="PRES-CTX").value) and row(prepared, code="PRES-CTX").missing == "not_observable"
    assert row(prepared, code="PRES-EVID").value == 0.0                    # zero is a count, not missing
    assert (row(prepared, code="PRES-SUBS").value, row(prepared, code="PRES-COH").value) == (3.0, 4.0)


def test_the_prepared_rows_follow_the_book_whatever_the_record_lists():
    """One row per prepared code of the book, in book order: a code the record leaves out is NaN with the reason
    not_recorded, a code it rates twice keeps its first rating, and the table still pivots one value per code."""
    codes = [e.id for e in load_panel_books(load_release()).prepared_codes()]
    record = synthetic_record()
    record["prepared"] = [{"code": "PRES-COH", "value": "2", "missing": "none"},
                          {"code": "PRES-SUBS", "value": "5", "missing": "none"},
                          {"code": "PRES-SUBS", "value": "1", "missing": "none"},
                          {"code": "PRES-EVID", "value": "0", "missing": "none"}]
    data = build_data([{"video_id": "v", "record": record, "source": "primary"},
                       {"video_id": "w", "record": synthetic_record(), "source": "primary"}])
    prepared = data.prepared[data.prepared.video_id == "v"]
    assert list(prepared.code) == codes and PREPARED_NOT_RECORDED == "not_recorded"
    assert (row(prepared, code="PRES-SUBS").value, row(prepared, code="PRES-COH").value) == (5.0, 2.0)
    assert math.isnan(row(prepared, code="PRES-CTX").value) and row(prepared, code="PRES-CTX").missing == "not_recorded"
    assert row(prepared, code="PRES-EVID").value == 0.0
    wide = data.prepared.pivot(index="video_id", columns="code", values="value")
    assert wide.shape == (2, len(codes)) and wide.loc["w", "PRES-SUBS"] == 3.0


def test_ask_columns_and_missing_flag(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "run"
    write_video(root, "v000", make_record(rng))
    write_video(root, "v001", make_record(rng, ask_missing="not_observable"))
    videos = load_run(root).videos.set_index("video_id")
    assert {"ask_amount_usd", "ask_equity_pct", "ask_missing"} <= set(videos.columns)
    assert videos.ask_amount_usd["v000"] > 0 and 0 < videos.ask_equity_pct["v000"] <= 100
    assert videos.ask_missing["v000"] == "none"
    assert np.isnan(videos.ask_amount_usd["v001"]) and np.isnan(videos.ask_equity_pct["v001"])
    assert videos.ask_missing["v001"] == "not_observable"


def _pitch_override(root, video_id, value):
    path = root / video_id / "results" / "reviewed.json"
    result = json.loads(path.read_text())
    result["pitch_override"] = {"any_offer": value}
    path.write_text(json.dumps(result))


def test_pitch_override_is_read_and_wins(tmp_path):
    root = tmp_path / "run"
    # in the synthetic record S1 offers: the record says any_offer 1
    for video_id in ("v000", "v001", "v002"):
        write_video(root, video_id, synthetic_record(), reviewed=synthetic_record())
    _pitch_override(root, "v000", "no")
    _pitch_override(root, "v001", "Unknown")
    data = load_run(root)
    assert data.pitch_overrides.keys() == {"v000", "v001"}
    assert data.pitch_overrides["v000"] == 0.0 and math.isnan(data.pitch_overrides["v001"])
    o1 = pitch_outcome(data).set_index("video_id")
    assert list(pitch_outcome(data).columns) == ["video_id", "any_offer", "first_offer_pos", "any_offer_source"]
    assert o1.loc["v000", ["any_offer", "any_offer_source"]].tolist() == [0.0, "human_override"]
    assert math.isnan(o1.loc["v001", "any_offer"]) and o1.loc["v001", "any_offer_source"] == "human_override"
    assert o1.loc["v002", ["any_offer", "any_offer_source"]].tolist() == [1.0, "record"]
    # the decision list, and with it the offer position and the cut, still describe the record
    assert o1.loc["v000", "first_offer_pos"] == o1.loc["v002", "first_offer_pos"]
    assert pitch_cut(data).cut_pos.nunique() == 1


def test_build_data_refuses_an_unknown_pitch_override():
    entry = {"video_id": "v", "record": synthetic_record(), "source": "reviewed"}
    assert build_data([{**entry, "pitch_override": {"any_offer": "yes"}}]).pitch_overrides == {"v": 1.0}
    assert build_data([entry]).pitch_overrides == {}
    with pytest.raises(ConfigError, match="maybe"):
        build_data([{**entry, "pitch_override": {"any_offer": "maybe"}}])


def test_any_offer_is_the_any_and_all_of_the_investors_ever_offer_without_a_pitch_override(tmp_path):
    """Without a pitch override, a pitch has an offer when any investor ever offered, has none when every
    investor's outcome is known and none offered, and is unknown otherwise."""
    data = load_run(write_run(tmp_path, 12, seed=3))
    assert not data.pitch_overrides
    expected = {}
    for video_id, ever in investor_outcomes(data).groupby("video_id").ever_offer:
        expected[video_id] = 1.0 if (ever == 1.0).any() else 0.0 if (ever == 0.0).all() else math.nan
    pitch = pitch_outcome(data)
    assert set(pitch.video_id) == set(expected) and set(pitch.any_offer_source) == {"record"}
    for r in pitch.itertuples():
        assert r.any_offer == expected[r.video_id] or (math.isnan(r.any_offer) and math.isnan(expected[r.video_id]))
    assert {0.0, 1.0} <= set(pitch.any_offer.dropna())                   # the corpus has both kinds of pitch
    assert list(pitch_cut(data).columns) == ["video_id", "cut_pos"]


def test_the_tuple_api_of_outcomes_and_cuts_is_gone():
    import analysis.panel_dataset as pd_module

    assert not hasattr(pd_module, "outcomes") and not hasattr(pd_module, "cuts")


def test_the_shark_level_pieces_live_in_the_archive(tmp_path):
    from analysis.archive import shark_level_dataset as archive

    data = load_run(write_run(tmp_path, 6, seed=3))
    assert archive.FOCAL_EXCLUDED_FUNCTIONS == () and "own expression of interest" in archive.REASON_OWN_INTEREST
    for r in archive.investor_cuts(data).itertuples():
        assert archive.eligible_turns_for_investor(data, r.video_id, int(r.cut_pos), r.investor).equals(
            eligible_turns(data, r.video_id, int(r.cut_pos)))


def test_eligible_turns_has_no_focal_parameter():
    from analysis.panel_dataset import exclusion_reasons

    assert "focal" not in inspect.signature(eligible_turns).parameters
    assert "focal" not in inspect.signature(exclusion_reasons).parameters


def test_the_closings_table_is_gone():
    import analysis.panel_dataset as pd_module

    assert "closings" not in pd_module._COLUMNS
    assert "closings" not in {f.name for f in dataclasses.fields(PanelData)}


# ---- sensitivities: both passes, the development videos ---------------------------------------

def _write_pass(vdir, name, record):
    """Replace results/<name>.json of a video written by write_video with `record`."""
    from pipeline.schemas.stage_b_panel import PanelRecord

    path = vdir / "results" / f"{name}.json"
    result = json.loads(path.read_text())
    result["record"] = PanelRecord.model_validate(record).model_dump(mode="json")
    path.write_text(json.dumps(result))


def _validation(vdir, name, n_errors):
    (vdir / "results" / f"{name}_validation.json").write_text(
        json.dumps({"errors": [f"error {i}" for i in range(n_errors)], "flags": []}))


def test_missing_replicate_uses_primary_and_is_listed(tmp_path):
    """Review Focus 5: a video whose other pass is missing or does not load as a record is read from its best
    record alone and listed, with why, in books_source["single_pass_videos"]. A pass whose validation report
    lists errors is still read (the main analysis reads such records too), and so is the primary when the best
    record is the replicate."""
    root = tmp_path / "run"
    for video_id in ("both", "gone", "errors", "malformed", "not_json", "rep_best", "reviewed"):
        write_video(root, video_id, synthetic_record(),
                    reviewed=synthetic_record() if video_id == "reviewed" else None)
    (root / "gone" / "results" / "replicate.json").unlink()
    _validation(root / "errors", "primary", 0)
    _validation(root / "errors", "replicate", 3)
    path = root / "malformed" / "results" / "replicate.json"
    result = json.loads(path.read_text())
    result["record"]["turns"] = "not a list"
    path.write_text(json.dumps(result))
    (root / "not_json" / "results" / "replicate.json").write_text("{ cut off")
    _validation(root / "rep_best", "primary", 2)
    _validation(root / "rep_best", "replicate", 1)

    best = load_run(root)
    assert best.replicate is None and "single_pass_videos" not in best.books_source
    data = load_run(root, passes="both")
    assert data.books_source["single_pass_videos"] == {
        "gone": "missing", "malformed": "unreadable: ValidationError", "not_json": "unreadable: JSONDecodeError"}
    assert list(data.books_source["single_pass_videos"]) == ["gone", "malformed", "not_json"]
    assert list(data.replicate.videos.video_id) == ["both", "errors", "rep_best", "reviewed"]
    assert set(data.replicate.videos.source) == {"replicate"}
    assert data.replicate.replicate is None
    # the first set of tables is the one passes="best" reads
    assert dict(zip(data.videos.video_id, data.videos.source)) == dict(zip(best.videos.video_id, best.videos.source))
    assert dict(zip(data.videos.video_id, data.videos.source))["rep_best"] == "replicate"
    assert {k: v for k, v in data.books_source.items() if k != "single_pass_videos"} == best.books_source
    assert load_run(root, ["gone", "both"], passes="both").books_source["single_pass_videos"] == {"gone": "missing"}
    with pytest.raises(ValueError, match="passes"):
        load_run(root, passes="all")


def test_a_replicate_whose_validation_report_lists_errors_is_averaged(tmp_path):
    from analysis.panel_features import pitch_features

    root = tmp_path / "run"
    vdir = write_video(root, "vid", synthetic_record())
    replicate = synthetic_record()
    next(c for c in replicate["turns"][2]["codes"] if c["code"] == "PREP-SUBS").update(value="5")   # turn 3: 3 -> 5
    _write_pass(vdir, "replicate", replicate)
    _validation(vdir, "replicate", 4)
    data = load_run(root, passes="both")
    assert data.books_source["single_pass_videos"] == {} and list(data.replicate.videos.video_id) == ["vid"]
    row = pitch_features(data).iloc[0]
    assert (row.a_subs_mean, row.a_subs_mean_n) == (4.0, 4)       # (3.5 + 4.5) / 2, 2 + 2 answers


def test_development_ids_read_from_the_selection(tmp_path):
    from analysis.panel_dataset import development_video_ids

    root = tmp_path / "run"
    root.mkdir()
    assert development_video_ids(root) == set()                          # no selection file: none known
    (root / "selection_sample.json").write_text(json.dumps({"videos": ["a", "b"]}))
    assert development_video_ids(root) == set()                          # a selection without the key
    (root / "selection_all48.json").write_text(json.dumps({"videos": ["a", "b"], "pilot_development_videos": ["b"]}))
    assert development_video_ids(root) == {"b"}
    (root / "selection_all100.json").write_text(json.dumps(
        {"videos": ["a", "b", "c", "d"], "pilot_development_videos": ["c", "d"]}))
    assert development_video_ids(str(root)) == {"c", "d"}                # selection_all100.json comes first


def test_development_ids_come_from_the_given_selection_file_first(tmp_path):
    """Final review I2: a selection file outside the run root (`--selection`) names the development videos; the run
    root is read only when it does not, and every file opened is listed for the manifest."""
    from analysis.panel_dataset import development_video_ids, development_videos

    root, elsewhere = tmp_path / "run", tmp_path / "main_checkout"
    root.mkdir()
    elsewhere.mkdir()
    given = elsewhere / "selection_all100.json"
    given.write_text(json.dumps({"video_ids": ["a", "b", "c"], "pilot_development_videos": ["a"]}))
    assert development_videos(root, given) == ({"a"}, [given])            # the run root has none: no matter
    assert development_video_ids(root) == set() and development_videos(root) == (set(), [])
    (root / "selection_all100.json").write_text(json.dumps({"videos": ["a", "b"], "pilot_development_videos": ["b"]}))
    assert development_video_ids(root, given) == {"a"}                       # the given file wins
    plain = elsewhere / "selection_plain.json"
    plain.write_text(json.dumps({"video_ids": ["a", "b"]}))                  # without the key: the run root's
    assert development_videos(root, plain) == ({"b"}, [plain, root / "selection_all100.json"])
