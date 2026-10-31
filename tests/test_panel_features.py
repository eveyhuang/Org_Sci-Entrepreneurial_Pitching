"""analysis.panel_features: one row per pitch, named concept features that stop at the pitch's cut.

The leakage tests come first: they are the reason this layer exists. The shark-level rows (one per pitch
and investor) are tested in tests/archive/test_shark_level_features.py.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

import analysis.panel_features as pf
from analysis.panel_dataset import build_data, pitch_cut, with_replicate
from analysis.panel_features import FAMILIES, FEATURES, OUTCOME_COLUMNS, feature_dictionary, pitch_features
from tests.panel_analysis_fixtures import FORCED_PATTERNS, make_record, talk_record
from tests.panel_fixtures import STAGE_A_IDS, set_other, synthetic_record, turn

KEYS = ("video_id", "source", "cut_pos")


def data_of(record, video_id="vid", rank_scores=None, source="primary"):
    return build_data([{"video_id": video_id, "record": record, "source": source, "release_id": "v3-dev9",
                        "duration_s": 200.0}], rank_scores)


def both_of(primary, replicate, video_id="vid", rank_scores=None):
    """The data of one pitch read with both passes: `primary` as the best record, `replicate` as the second."""
    return with_replicate(data_of(primary, video_id, rank_scores),
                          data_of(replicate, video_id, rank_scores, source="replicate"))


def same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b or (pd.isna(a) and pd.isna(b))


def stat(r, name):
    return r[name], r[f"{name}_n"], r[f"{name}_missing"]


# ---- the leakage test -----------------------------------------------------------------------

def truncated(record: dict, data, cut_pos: int, drop: str) -> dict:
    """`record` without any turn at or after `cut_pos`.

    drop="later": also without every decision at or after the cut, which is everything a pitch row must
    not see (the cut is the first decision, so that is every decision). drop="all": without any decision.
    """
    out = copy.deepcopy(record)
    out["turns"] = out["turns"][:cut_pos]
    pos = dict(zip(data.decisions.decision_id, data.decisions.pos))
    out["decisions"] = [] if drop == "all" else [d for d in out["decisions"] if pos[d["decision_id"]] < cut_pos]
    return out


def leaks(records: dict[str, dict], drop: str = "later", exposure: str = "counts",
          replicates: dict[str, dict] | None = None) -> list[str]:
    """Every feature of every pitch row that changes when the row's future is deleted from its record.

    With `replicates` (video id -> a second pass), the row averages both passes and each pass loses the
    future of its own cut (each pass is cut at its own first decision)."""
    found = []
    for video_id, record in records.items():
        data = data_of(record, video_id)
        replicate = (replicates or {}).get(video_id)
        second = None if replicate is None else data_of(replicate, video_id, source="replicate")
        full = pitch_features(data if second is None else with_replicate(data, second), exposure=exposure)
        assert len(full) == 1, "the fixture must give one row"
        original = full.iloc[0]
        cut_data = data_of(truncated(record, data, int(original["cut_pos"]), drop), video_id)
        if second is not None:
            own_cut = int(pitch_cut(second).cut_pos.iloc[0])
            cut_data = with_replicate(cut_data, data_of(truncated(replicate, second, own_cut, drop), video_id,
                                                        source="replicate"))
        again = pitch_features(cut_data, exposure=exposure).iloc[0]
        for column in full.columns:
            if column in OUTCOME_COLUMNS or column in KEYS:
                continue
            if not same(original[column], again[column]):
                found.append(f"{video_id} pitch {column}: {original[column]!r} -> {again[column]!r}")
    return found


# the options of the sensitivities (spec section 6.4) the deletion test runs under: (exposure, both passes)
OPTIONS = [pytest.param("counts", False, id="counts"), pytest.param("rates", False, id="rates"),
           pytest.param("counts", True, id="both_passes"), pytest.param("rates", True, id="both_passes_rates")]


def second_passes(records: dict[str, dict]) -> dict[str, dict]:
    """A second pass for every pitch that differs from its first: the record of the next pitch, so that the two
    passes are cut at different turns."""
    names = list(records)
    return {name: records[names[(i + 1) % len(names)]] for i, name in enumerate(names)}


@pytest.fixture(scope="module")
def pitches() -> dict[str, dict]:
    rng = np.random.default_rng(11)
    records = {f"p{i}": make_record(rng, forced=FORCED_PATTERNS if i == 0 else None) for i in range(8)}
    records["story"] = synthetic_record()
    records["talk"] = talk_record(TALK, decisions=TALK_DECISIONS)
    records["late"] = talk_record(LATE_RECEIPT, decisions=(("S1", "exit", 4), ("S2", "exit", 6)))
    return records


@pytest.mark.parametrize("exposure, both", OPTIONS)
def test_no_pitch_feature_changes_when_the_rows_future_is_deleted(pitches, exposure, both):
    assert leaks(pitches, exposure=exposure, replicates=second_passes(pitches) if both else None) == []


def test_the_second_passes_are_cut_elsewhere(pitches):
    """The both-passes deletion test means something only if a second pass's own cut differs from its first's."""
    cuts = {name: int(pitch_cut(data_of(record, name)).cut_pos.iloc[0]) for name, record in pitches.items()}
    names = list(pitches)
    assert len(set(cuts.values())) > 1
    assert any(cuts[a] != cuts[names[(i + 1) % len(names)]] for i, a in enumerate(names))


def test_the_deletion_test_covers_every_column(pitches):
    """`leaks` compares every column of the pitch table except the keys and the outcome columns, so every column
    a pitch-level model set can read is one the deletion test above has checked."""
    from analysis.panel_bottleneck import LADDER, LADDER_NOT_FITTED, family_columns, stage_a_fallback

    table = pitch_features(data_of(pitches["story"], "story"))
    checked = set(table.columns) - set(OUTCOME_COLUMNS) - set(KEYS)
    assert checked == {c for c in table.columns if FEATURES.get(c, {}).get("family") is not None}
    every = {c for columns in family_columns(table, FAMILIES).values() for c in columns}
    assert every == checked
    for family in (LADDER, stage_a_fallback(LADDER, table)[0]):      # this table has no rank scores
        for name, spec in family.items():
            for not_fitted in ((), LADDER_NOT_FITTED):                 # the ladder fits fewer (ruling I4), never more
                read = {c for columns in family_columns(table, spec, not_fitted).values() for c in columns}
                assert read <= checked and bool(read) == bool(spec), name
    assert set(LADDER_NOT_FITTED) <= checked                          # not fitted, but still in the table and checked
    assert not every & set(OUTCOME_COLUMNS)


@pytest.mark.parametrize("exposure, both", OPTIONS)
def test_deleting_every_decision_changes_nothing_on_a_pitch_row(pitches, exposure, both):
    """A pitch row stops at the first decision of anyone, so it reads no decision at all."""
    assert leaks(pitches, drop="all", exposure=exposure, replicates=second_passes(pitches) if both else None) == []


@pytest.mark.parametrize("exposure, both", OPTIONS)
def test_the_leakage_check_catches_features_that_read_past_the_cut(pitches, monkeypatch, exposure, both):
    real = pf.eligible_turns
    monkeypatch.setattr(pf, "eligible_turns", lambda data, video_id, cut_pos: real(data, video_id, 10 ** 9))
    assert any(" hist_turns:" in line
               for line in leaks(pitches, exposure=exposure, replicates=second_passes(pitches) if both else None))


def test_the_leakage_check_catches_a_second_pass_read_past_its_own_cut(pitches, monkeypatch):
    """Only the second pass reads past its cut here: the both-passes deletion test sees it, so it checks the
    second pass, not only the first."""
    real = pf.pitch_cut

    def past_the_cut_of_the_second_pass(data):
        cut = real(data)
        if set(data.videos.source) == {"replicate"}:
            cut["cut_pos"] = 10 ** 6
        return cut
    monkeypatch.setattr(pf, "pitch_cut", past_the_cut_of_the_second_pass)
    assert leaks(pitches) == []
    assert any(" hist_turns:" in line for line in leaks(pitches, replicates=second_passes(pitches)))


# ---- the families and their columns ------------------------------------------------------------

def test_families_are_the_ladders():
    assert FAMILIES == ("control", "exposure", "stage_a", "prepared", "questions", "answers", "reactions",
                        "rapport_repair")
    assert {spec["family"] for spec in FEATURES.values()} - {None} == set(FAMILIES)
    assert OUTCOME_COLUMNS == ("any_offer", "first_offer_pos", "any_offer_source")


def _with_twins(*names):
    return {c for n in names for c in (n, f"{n}_n", f"{n}_missing")}


EXPECTED = {
    "control": {"founder_gender", "primary_industry", "product_form", "customer_type", "ask_log_amount",
                "ask_equity_pct", "ask_missing"},
    "exposure": {"hist_turns", "hist_seconds", "hist_questions", "demo_seconds", "prepared_seconds"},
    "stage_a": {c for code in STAGE_A_IDS for c in (f"sa_{code}", f"sa_{code}_missing", f"sa_rank_{code}")},
    "prepared": {c for s in ("subs", "coh", "ctx", "evid") for c in (f"pres_{s}", f"pres_{s}_missing")},
    "questions": {"q_askers"} | _with_twins(
        "q_sev_mean", "q_hard_share", "q_frame_promotion_share", "q_frame_prevention_share", "q_deal_terms_share",
        "q_other_share", "q_sev_trend", "q_pursuit_share", "q_frame_switch_share", "q_concept_entropy"),
    "answers": _with_twins("a_resp_mean", "a_evid_share", "a_uncert_share", "a_subs_mean", "a_coh_mean",
                           "a_ctx_mean", "a_subs_hard_mean", "a_evid_hard_share", "a_frame_match_share"),
    "reactions": {"rx_answers", "accepts_turns", "challenge_turns"} | _with_twins(
        "rx_receipt_share", "rx_followup_share", "rx_challenge_share", "rx_reaction_rate", "nm_receipt_share",
        "nm_followup_share", "nm_challenge_share"),
    "rapport_repair": {"social_turns", "founder_social", "hostile_at_founder", "founder_repairs"},
}


def test_pitch_row_columns_by_family():
    scores = {code: {"pl_score": {"vid": 0.25}} for code in STAGE_A_IDS}
    table = pitch_features(data_of(synthetic_record(), rank_scores=scores))
    by_family = {}
    for column in table.columns:
        by_family.setdefault(FEATURES[column]["family"] if column in FEATURES else "outcome", set()).add(column)
    assert by_family.pop(None) == set(KEYS) and by_family.pop("outcome") == set(OUTCOME_COLUMNS)
    assert by_family == EXPECTED
    assert {name for name, spec in FEATURES.items() if spec["family"] is not None} == set().union(*EXPECTED.values())
    for gone in ("n_social_exchanges", "q_count", "all_questions", "all_satisfied", "all_no_signal", "rx_all_last_receipt", "rx_all_trend_receipt",
                 "rx_all_reactors", "hist_questions_all", "investor", "investor_label", "oth_exits_before",
                 "own_questions", "rx_own_answers", "rx_pan_answers"):
        assert gone not in FEATURES and gone not in table.columns, gone


# ---- hand-computed values on the story pitch ------------------------------------------------
# tests/panel_fixtures.py: S1 asks (turn 2) and is answered (3); S2 presses (4), is answered (5), the
# founder repairs the explanation (6) and S1 accepts it (7); S3 jokes (8), the founder joins in (9),
# S3 exits (10); the pitch stops there, at the first decision of anyone, so turns 1-9 are its history.

@pytest.fixture(scope="module")
def story():
    return pitch_features(data_of(synthetic_record())).iloc[0]


def test_keys_outcome_and_cut(story):
    assert (story.video_id, story.source, story.cut_pos) == ("vid", "primary", 9)
    assert (story.any_offer, story.any_offer_source) == (1.0, "record")


def test_controls_exposure_stage_a_and_prepared(story):
    assert (story.founder_gender, story.primary_industry, story.product_form, story.customer_type) == (
        "all_men", "home_household", "physical_good", "consumer")
    assert story.ask_log_amount == pytest.approx(math.log10(200000)) and story.ask_equity_pct == 10.0
    assert story.ask_missing == 0
    assert (story.hist_turns, story.hist_seconds, story.hist_questions) == (9, 102.0, 2)
    assert (story.demo_seconds, story.prepared_seconds) == (0.0, 55.0)
    for code in STAGE_A_IDS:
        assert story[f"sa_{code}"] == 3.0 and story[f"sa_{code}_missing"] == 0
    # tests/panel_fixtures.prepared_codes: PRES-SUBS 3, PRES-COH 4, PRES-CTX 2, PRES-EVID 5
    assert (story.pres_subs, story.pres_coh, story.pres_ctx, story.pres_evid) == (3.0, 4.0, 2.0, 5.0)
    assert not story[[f"pres_{s}_missing" for s in ("subs", "coh", "ctx", "evid")]].any()


def test_questions_answers_reactions_and_rapport(story):
    assert story.q_askers == 2                                              # turns 2 (S1) and 4 (S2)
    assert stat(story, "q_sev_mean") == (2.0, 2, 0)
    assert stat(story, "q_hard_share") == (0.5, 2, 0)                      # turn 4 has severity 3
    assert stat(story, "q_frame_prevention_share") == (1.0, 2, 0)
    assert stat(story, "q_frame_promotion_share") == (0.0, 2, 0)
    assert stat(story, "q_deal_terms_share") == (0.0, 2, 0) and stat(story, "q_other_share") == (0.0, 2, 0)
    assert stat(story, "a_resp_mean") == (1.0, 2, 0)
    assert stat(story, "a_evid_share") == (1.0, 2, 0)
    assert stat(story, "a_uncert_share") == (0.0, 2, 0)
    assert stat(story, "a_subs_mean") == (3.5, 2, 0)
    assert stat(story, "a_coh_mean") == (4.0, 2, 0)
    assert story.a_ctx_mean_missing == 1 and math.isnan(story.a_ctx_mean)
    assert stat(story, "a_frame_match_share") == (1.0, 2, 0)
    assert stat(story, "a_subs_hard_mean") == (4.0, 1, 0) and stat(story, "a_evid_hard_share") == (1.0, 1, 0)
    # answer 3 drew S2's question (turn 4), answer 5 S1's accepts (turn 7)
    assert story.rx_answers == 2 and stat(story, "rx_reaction_rate") == (1.0, 2, 0)
    assert stat(story, "rx_receipt_share") == (0.5, 2, 0) and stat(story, "rx_followup_share") == (0.5, 2, 0)
    assert stat(story, "rx_challenge_share") == (0.0, 2, 0)
    assert stat(story, "nm_receipt_share") == (0.0, 2, 0)
    assert (story.accepts_turns, story.challenge_turns) == (1, 0)          # S1's accepts (turn 7)
    assert (story.social_turns, story.founder_social) == (1, 1)
    assert (story.hostile_at_founder, story.founder_repairs) == (0, 1)


def test_question_shares_count_deal_terms_and_other_separately():
    """Two deal_terms questions, one other and one promotion (sales): each share is of all four questions."""
    script = [("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("S2", "question", 1, (), 2),
              ("F1", "answer", 1, (4,), 3), ("S1", "question", 1, (), 1), ("F1", "answer", 1, (6,), 3),
              ("S2", "question", 1, (), 3), ("F1", "answer", 1, (8,), 3), ("S1", "decides", None, (), None)]
    record = talk_record(script, decisions=(("S1", "exit", 10),))
    by_id = {t["turn_id"]: t for t in record["turns"]}
    for turn_id in (2, 4):
        next(c for c in by_id[turn_id]["codes"] if c["code"] == "PR-CONCEPT")["value"] = "deal_terms"
    set_other(by_id[6], "a licensing question")                      # turn 8 keeps the fixture's `sales`
    row = pitch_features(data_of(record)).iloc[0]
    assert row.hist_questions == 4
    assert stat(row, "q_deal_terms_share") == (0.5, 4, 0)
    assert stat(row, "q_other_share") == (0.25, 4, 0)
    assert stat(row, "q_frame_promotion_share") == (0.25, 4, 0)
    assert stat(row, "q_frame_prevention_share") == (0.0, 4, 0)
    assert stat(row, "q_hard_share") == (0.5, 4, 0)
    for name in ("q_deal_terms_share", "q_other_share", "q_frame_promotion_share", "q_frame_prevention_share"):
        assert "need not sum to 1" in FEATURES[name]["description"], name


# ---- the sequence of the questions (plan B2): read in turn order, never from the exchange grouping -----

SEQUENCE = [("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3),     # 2 A sev 1 promotion (sales)
            ("S1", "question", 1, (), 3), ("F1", "answer", 1, (4,), 3),     # 4 A sev 3 promotion (vision)
            ("S2", "question", 1, (), 2), ("F1", "answer", 1, (6,), 3),     # 6 B sev 2 prevention (execution)
            ("S1", "question", 1, (), 2), ("F1", "answer", 1, (8,), 3),     # 8 A sev 2 prevention (execution)
            ("S1", "decides", None, (), None)]                              # 10: S1's exit, the cut


def _with_concepts(record: dict, concepts: dict[int, str]) -> dict:
    """`record` with the question concepts set by turn id (talk_record gives every question `sales`)."""
    for t in record["turns"]:
        for c in t["codes"]:
            if c["code"] == "PR-CONCEPT" and t["turn_id"] in concepts:
                c["value"] = concepts[t["turn_id"]]
    return record


def test_question_sequence_features_on_a_hand_built_pitch():
    record = _with_concepts(talk_record(SEQUENCE, decisions=(("S1", "exit", 10),)),
                            {2: "sales", 4: "vision", 6: "execution", 8: "execution"})
    data = data_of(record)
    assert data.turns.set_index("turn_id").frame[[2, 4, 6, 8]].tolist() == [
        "promotion", "promotion", "prevention", "prevention"]
    row = pitch_features(data).iloc[0]
    assert row.hist_questions == 4
    assert stat(row, "q_sev_trend") == (0.0, 4, 0)                          # halves [1, 3] and [2, 2]
    assert stat(row, "q_pursuit_share") == (pytest.approx(1 / 3), 3, 0)     # pairs AA (rises), AB, BA
    assert stat(row, "q_frame_switch_share") == (pytest.approx(1 / 3), 3, 0)   # prom-prom, prom-prev, prev-prev
    # concepts sales, vision, execution, execution: shares 1/4, 1/4, 1/2 -> 1.5 bits
    assert stat(row, "q_concept_entropy") == (pytest.approx(1.5), 4, 0)
    # the trend is later minus earlier: one question of severity 1, then one of 3, by the same asker
    rising = [("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("S1", "question", 1, (), 3),
              ("F1", "answer", 1, (4,), 3), ("S1", "decides", None, (), None)]
    row = pitch_features(data_of(talk_record(rising, decisions=(("S1", "exit", 6),)))).iloc[0]
    assert stat(row, "q_sev_trend") == (2.0, 2, 0) and stat(row, "q_pursuit_share") == (1.0, 1, 0)
    assert stat(row, "q_concept_entropy") == (0.0, 2, 0)                   # both `sales`
    # an unclassified or NEUTRAL question breaks a pair for the frame switch: neither pair around it is compared
    record = _with_concepts(talk_record(SEQUENCE, decisions=(("S1", "exit", 10),)),
                            {2: "sales", 4: "deal_terms", 6: "execution", 8: "NONE"})
    row = pitch_features(data_of(record)).iloc[0]
    assert stat(row, "q_frame_switch_share")[1:] == (0, 1) and math.isnan(row.q_frame_switch_share)
    # NONE is left out of the entropy; deal_terms is a concept like any other: sales, deal_terms, execution
    assert stat(row, "q_concept_entropy") == (pytest.approx(math.log2(3)), 3, 0)
    for name in ("q_sev_trend", "q_pursuit_share", "q_frame_switch_share", "q_concept_entropy"):
        assert "before the cut" in FEATURES[name]["description"], name
        assert "exchange" not in FEATURES[name]["description"], name


def test_one_question_gives_nan_shares_and_zero_trend_and_entropy():
    """Review Focus 2: one investor question has no consecutive pair. The pair shares are missing with `_n` 0,
    the trend and the entropy are 0.0, and nothing divides by zero; with no question at all the entropy is
    missing too."""
    script = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3), ("S1", "decides", None, (), None)]
    row = pitch_features(data_of(talk_record(script, decisions=(("S1", "exit", 4),)))).iloc[0]
    assert row.hist_questions == 1
    for name in ("q_pursuit_share", "q_frame_switch_share"):
        assert math.isnan(row[name]) and row[f"{name}_n"] == 0 and row[f"{name}_missing"] == 1, name
    assert stat(row, "q_sev_trend") == (0.0, 1, 0)
    assert stat(row, "q_concept_entropy") == (0.0, 1, 0)
    script = [("S1", "accepts", 1, (), None), ("S1", "decides", None, (), None)]
    row = pitch_features(data_of(talk_record(script, decisions=(("S1", "exit", 3),)))).iloc[0]
    assert row.hist_questions == 0
    for name in ("q_pursuit_share", "q_frame_switch_share", "q_concept_entropy"):
        assert math.isnan(row[name]) and row[f"{name}_n"] == 0 and row[f"{name}_missing"] == 1, name
    assert stat(row, "q_sev_trend") == (0.0, 0, 1)                          # no rated question: flagged, not NaN


def test_demo_seconds_sums_demonstrates_turns():
    """demo_seconds reads the turn function, not the exchange kind: two `demonstrates` turns (3 s each) of a
    substantive exchange count, a social turn of a demonstration exchange does not, and a `demonstrates` turn
    after the cut is not read."""
    script = [("F1", "demonstrates", 1, (), None), ("S1", "question", 1, (), 2), ("F1", "answer", 1, (3,), 3),
              ("F1", "demonstrates", 1, (), None), ("S2", "social", 2, (), None), ("S1", "decides", None, (), None),
              ("F1", "demonstrates", 1, (), None)]
    record = talk_record(script, exchanges=((1, "substantive"), (2, "demonstration")),
                         decisions=(("S1", "exit", 7),))
    row = pitch_features(data_of(record)).iloc[0]
    assert row.cut_pos == 6 and row.demo_seconds == 6.0
    assert "n_social_exchanges" not in row.index and "n_social_exchanges" not in FEATURES
    assert "demonstrates" in FEATURES["demo_seconds"]["description"]


def test_new_features_are_fitted_by_c3q_and_later():
    from analysis.panel_bottleneck import LADDER, LADDER_NOT_FITTED, feature_status, ladder_columns

    new = ("q_sev_trend", "q_pursuit_share", "q_frame_switch_share", "q_concept_entropy")
    table = pitch_features(data_of(_with_concepts(talk_record(SEQUENCE, decisions=(("S1", "exit", 10),)),
                                                  {4: "vision", 6: "execution"})))
    for name in new:
        assert FEATURES[name]["family"] == "questions" and FEATURES[name]["unit"] == "o1", name
        assert name not in LADDER_NOT_FITTED and feature_status(name) == "primary", name
        assert f"{name}_n" in LADDER_NOT_FITTED and f"{name}_missing" not in LADDER_NOT_FITTED, name
        for model in ("C3q", "C3", "C4", "C5"):
            assert "questions" in LADDER[model], model
        assert "questions" not in LADDER["C2p"]
    fitted = set(ladder_columns(table, LADDER["C3q"]))
    assert set(new) <= fitted and not {f"{n}_n" for n in new} & fitted
    assert not set(new) & set(ladder_columns(table, LADDER["C2p"]))


def test_early_cut_leaves_nan_not_zero():
    """A pitch whose first decision is at position 2 (one accepts turn after the presentation): nothing to
    average, so every mean and share is missing with 0 observations, never zero; counts are 0."""
    script = [("S1", "accepts", 1, (), None), ("S1", "decides", None, (), None), ("S1", "question", 1, (), 2),
              ("F1", "answer", 1, (4,), 4)]
    row = pitch_features(data_of(talk_record(script, decisions=(("S1", "exit", 3),)))).iloc[0]
    assert row.cut_pos == 2 and row.q_askers == 0 and row.hist_questions == 0 and row.rx_answers == 0
    assert math.isnan(row.a_subs_mean) and row.a_subs_mean_n == 0 and row.a_subs_mean_missing == 1
    for name in ("a_subs_mean", "a_resp_mean", "q_sev_mean", "q_deal_terms_share", "rx_receipt_share",
                 "nm_receipt_share", "a_frame_match_share"):
        assert math.isnan(row[name]) and row[f"{name}_n"] == 0 and row[f"{name}_missing"] == 1, name
    script = [("S1", "question", 1, (), 2), ("S1", "decides", None, (), None), ("F1", "answer", 1, (2,), 4)]
    row = pitch_features(data_of(talk_record(script, decisions=(("S1", "exit", 3),)))).iloc[0]
    assert row.cut_pos == 2 and row.hist_questions == 1     # one question before the cut and no answer
    assert math.isnan(row.a_subs_mean) and row.a_subs_mean_n == 0 and row.a_subs_mean_missing == 1
    assert stat(row, "q_sev_mean") == (2.0, 1, 0)


def test_ask_columns_when_missing():
    record = synthetic_record()
    record["venture"].update(ask_amount_usd=None, ask_equity_pct=None, ask_missing="not_observable")
    row = pitch_features(data_of(record)).iloc[0]
    assert math.isnan(row.ask_log_amount) and math.isnan(row.ask_equity_pct) and row.ask_missing == 1
    record["venture"].update(ask_amount_usd=0, ask_equity_pct=5.0, ask_missing="none")
    row = pitch_features(data_of(record)).iloc[0]
    assert math.isnan(row.ask_log_amount) and row.ask_equity_pct == 5.0 and row.ask_missing == 0   # log of 0: missing
    record["venture"].update(ask_amount_usd=1000000, ask_equity_pct=20.0)
    row = pitch_features(data_of(record)).iloc[0]
    assert row.ask_log_amount == pytest.approx(6.0) and row.ask_equity_pct == 20.0
    assert FEATURES["ask_log_amount"]["family"] == FEATURES["ask_missing"]["family"] == "control"
    assert FEATURES["ask_missing"]["kind"] == "indicator" and FEATURES["ask_log_amount"]["kind"] != "control"


def test_a_prepared_rating_not_given_is_missing_with_its_indicator():
    record = synthetic_record()
    record["prepared"] = [p for p in record["prepared"] if p["code"] != "PRES-COH"]
    next(p for p in record["prepared"] if p["code"] == "PRES-EVID").update(value="0")
    row = pitch_features(data_of(record)).iloc[0]
    assert math.isnan(row.pres_coh) and row.pres_coh_missing == 1
    assert row.pres_evid == 0.0 and row.pres_evid_missing == 0            # a count of 0 is a value


def test_a_pitch_row_reads_no_decision_and_its_cut_ignores_the_outcome():
    """No column counts exits or offers: with the cut at the first decision they would be zero by
    construction, and under a cut at the first offer they would be the outcome itself."""
    offered = synthetic_record()
    no_offer = synthetic_record()
    no_offer["decisions"] = [d for d in no_offer["decisions"] if d["kind"] == "exit"] + [
        {"decision_id": 8 + i, "actor": a, "kind": "exit", "time_s": 160.0 + i, "turn_id": 17 + i, "offer_id": None,
         "terms": "", "note": ""} for i, a in enumerate(("S1", "S2"))]
    a, b = pitch_features(data_of(offered)).iloc[0], pitch_features(data_of(no_offer)).iloc[0]
    assert (a.any_offer, b.any_offer) == (1.0, 0.0) and a.cut_pos == b.cut_pos == 9
    for column in a.index:
        assert "exit" not in column and "offers_before" not in column, column
        if column not in OUTCOME_COLUMNS:
            assert same(a[column], b[column]), column          # same history, whatever came after


def test_the_history_stops_at_the_first_decision_and_counts_everyones_accepts():
    record = synthetic_record()
    record["turns"].insert(8, turn(20, "S1", 104, 105, "accepts", "Says he likes it.", ["F1"]))
    record["turns"].append(turn(21, "S2", 180, 182, "accepts", "Agrees after the deal.", ["F1"]))
    row = pitch_features(data_of(record)).iloc[0]
    # S3's exit (turn 10) is now at position 10: S1's accepts turns 7 and 20 lie before it, S2's 13 and 21 beyond
    assert row.cut_pos == 10 and row.accepts_turns == 2


def test_history_questions_count_investor_questions_only():
    record = synthetic_record()
    record["turns"][5] = turn(6, "F1", 94, 100, "question", "Asks the investor what margin she expects.", ["S2"], 1,
                              codes=[{"code": "PR-CONCEPT", "value": "sales", "missing": "none"},
                                     {"code": "PR-SEV", "value": "1", "missing": "none"}])
    row = pitch_features(data_of(record)).iloc[0]
    assert row.hist_questions == 2 and row.q_askers == 2


def test_rank_columns_appear_only_when_the_run_has_rank_scores():
    record = synthetic_record()
    assert not [c for c in pitch_features(data_of(record)).columns if c.startswith("sa_rank_")]
    scores = {code: {"pl_score": {"vid": 0.25}} for code in STAGE_A_IDS}
    ranked = pitch_features(data_of(record, rank_scores=scores))
    assert [c for c in ranked.columns if c.startswith("sa_rank_")] == [f"sa_rank_{code}" for code in STAGE_A_IDS]
    assert (ranked["sa_rank_interpersonal_warmth"] == 0.25).all()


def test_uncodable_stage_a_rating_is_missing_with_its_indicator():
    record = synthetic_record()
    record["stage_a"][0].update(value=None, uncodable=True)
    table = pitch_features(data_of(record))
    assert table[f"sa_{STAGE_A_IDS[0]}"].isna().all() and (table[f"sa_{STAGE_A_IDS[0]}_missing"] == 1).all()


# ---- the feature dictionary -----------------------------------------------------------------

def test_feature_dictionary_covers_every_non_outcome_column(pitches):
    dictionary = feature_dictionary()
    assert list(dictionary.columns) == ["name", "family", "label", "unit", "kind", "status", "description"]
    assert not dictionary.name.duplicated().any()
    assert set(dictionary.name) == set(FEATURES)
    scores = {code: {"pl_score": {"p0": 0.5}} for code in STAGE_A_IDS}
    table = pitch_features(data_of(pitches["p0"], "p0", rank_scores=scores))
    for column in table.columns:
        if column in OUTCOME_COLUMNS:
            continue
        assert column in FEATURES, column
        assert FEATURES[column]["unit"] in ("o1", "both"), column
    for name, spec in FEATURES.items():
        assert spec["kind"] in ("key", "control", "concept", "exposure", "indicator"), name
        assert (spec["family"] in FAMILIES) or spec["kind"] == "key", name
        assert spec["label"] and spec["description"], name
        assert not any(name == outcome for outcome in OUTCOME_COLUMNS)
    # every mean or share has its denominator and its missing indicator, in the same family
    for name, spec in FEATURES.items():
        if name.endswith(("_mean", "_share", "_rate")):
            assert FEATURES[f"{name}_n"]["family"] == spec["family"] == FEATURES[f"{name}_missing"]["family"]
            assert FEATURES[f"{name}_missing"]["kind"] == "indicator"


def test_no_free_text_column_and_every_family_is_represented(story):
    text_allowed = {"video_id", "source", "founder_gender", "primary_industry", "product_form", "customer_type",
                    "any_offer_source"}
    table = pitch_features(data_of(synthetic_record()))
    for column in table.columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            assert column in text_allowed, column
    assert {FEATURES[c]["family"] for c in table.columns if c in FEATURES} - {None} == set(FAMILIES)


def test_the_live_module_never_imports_the_archive():
    import ast

    tree = ast.parse(open(pf.__file__).read())
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert imported and not [name for name in imported if name and name.startswith("analysis.archive")]


# ---- the same rows, made inspectable --------------------------------------------------------------

def test_history_table_marks_exactly_the_turns_the_features_read():
    from analysis.panel_dataset import eligible_turns
    from analysis.panel_features import history_table

    data = data_of(make_record(np.random.default_rng(5), forced=FORCED_PATTERNS))
    cut = int(pitch_cut(data).cut_pos.iloc[0])
    table = history_table(data, "vid")
    assert len(table) == len(data.turns) and (table.cut_pos == cut).all()
    assert list(table.index[table.used]) == list(eligible_turns(data, "vid", cut).index)
    assert (table.why_excluded[table.used] == "").all() and (table.why_excluded[~table.used] != "").all()
    assert not table.used[table.pos >= cut].any()
    assert int(table.used.sum()) == int(pitch_features(data).hist_turns.iloc[0])
    with pytest.raises(ValueError, match="no row for video 'other'"):
        history_table(data, "other")


def test_describe_row_reads_one_row_downwards_with_its_denominators():
    from analysis.panel_features import describe_row

    data = data_of(synthetic_record())
    features = pitch_features(data)
    table = describe_row(features, "vid").set_index("feature")
    assert list(table.columns) == ["family", "label", "value", "denominator", "description"]
    # a mean comes with the observations behind it; its _n and _missing companions are folded in, not listed
    assert table.denominator["a_resp_mean"] == 2 and table.value["a_resp_mean"] == 1.0
    assert [name for name in table.index if name.endswith(("_n", "_missing"))] == ["ask_missing"]   # its own feature
    assert table.denominator["q_askers"] is None and table.value["q_askers"] == 2
    assert table.label["a_resp_mean"] == FEATURES["a_resp_mean"]["label"] == "answer directness (mean)"
    assert {"video_id", "cut_pos", "any_offer"}.isdisjoint(table.index)       # keys and outcomes are not features
    assert math.isnan(table.value["a_ctx_mean"]) and table.denominator["a_ctx_mean"] == 0
    with pytest.raises(ValueError, match="expected one row"):
        describe_row(features, "other")


# ---- the questioner's next move and hard questions, as features -----------------------------------
# S1 asks (turn 2, severity 1), is answered thinly (3), asks again harder (4, severity 2), is answered well (5)
# and accepts (6); S2 contests (7, severity 3), is answered (8); S1 exits (9); S2 accepts (10), asks again
# (11, severity 2), is answered (12) and offers (13).
TALK = [("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 2), ("S1", "question", 1, (), 2),
        ("F1", "answer", 1, (4,), 4), ("S1", "accepts", 1, (), None), ("S2", "question", 1, (), 3),
        ("F1", "answer", 1, (7,), 3), ("S1", "decides", None, (), None), ("S2", "accepts", 1, (), None),
        ("S2", "question", 1, (), 2), ("F1", "answer", 1, (11,), 5), ("S2", "decides", None, (), None)]
TALK_DECISIONS = (("S1", "exit", 9), ("S2", "offer", 13))
# S1 asks, is answered and accepts at the very position of S1's exit (the decision record points at the accepts
# turn): no decision talk comes before the receipt, so only the cut keeps a row from reading it
RECEIPT_AT_CUT = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "accepts", 1, (), None),
                  ("S2", "decides", None, (), None)]
RECEIPT_AT_CUT_DECISIONS = (("S1", "exit", 4), ("S2", "exit", 5))
# S1 asks, is answered, exits, and only then says the answer was fine: the receipt lies after S1's cut, and in
# turn order the decision talk is the answer's next investor turn, so even the whole encounter has no reaction
LATE_RECEIPT = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 4), ("S1", "decides", None, (), None),
                ("S1", "accepts", 1, (), None), ("S2", "decides", None, (), None)]


@pytest.fixture(scope="module")
def talk():
    data = data_of(talk_record(TALK, decisions=TALK_DECISIONS))
    return data, pitch_features(data)


def test_next_move_features_stop_at_the_first_decision(talk):
    row = talk[1].iloc[0]
    assert row.cut_pos == 8
    # three answers before S1's exit; S2's agreement with the third comes after it and is not read
    assert stat(row, "nm_receipt_share") == (pytest.approx(1 / 3), 3, 0)
    assert stat(row, "nm_followup_share") == (pytest.approx(1 / 3), 3, 0)
    assert stat(row, "nm_challenge_share") == (0.0, 3, 0)
    assert stat(row, "q_hard_share") == (pytest.approx(2 / 3), 3, 0)
    assert stat(row, "a_subs_hard_mean") == (3.5, 2, 0) and stat(row, "a_evid_hard_share") == (1.0, 2, 0)


def test_a_receipt_after_the_first_decision_is_not_read():
    data = data_of(talk_record(LATE_RECEIPT, decisions=(("S1", "exit", 4), ("S2", "exit", 6))))
    from analysis.panel_dataset import answer_followups
    assert answer_followups(data).next_move.tolist() == ["no_reaction"]   # turn order: the decision talk comes first
    assert stat(pitch_features(data).iloc[0], "nm_receipt_share") == (0.0, 1, 0)


def test_the_features_ask_for_next_moves_with_the_pitch_cut(talk, monkeypatch):
    data = talk[0]
    asked = []
    real = pf.answer_followups

    def spy(data, cut=None, video_ids=None):
        asked.append((cut, tuple(video_ids)))
        return real(data, cut, video_ids)
    monkeypatch.setattr(pf, "answer_followups", spy)
    pitch_features(data)
    assert asked == [({"vid": {"S1": 8, "S2": 8}}, ("vid",))]


def test_the_leakage_check_catches_a_next_move_read_past_the_cut(monkeypatch):
    records = {"late": talk_record(RECEIPT_AT_CUT, decisions=RECEIPT_AT_CUT_DECISIONS)}
    from analysis.panel_dataset import answer_followups
    assert answer_followups(data_of(records["late"])).next_move.tolist() == ["positive_receipt"]   # without a cut
    assert leaks(records) == []
    real = pf.answer_followups
    monkeypatch.setattr(pf, "answer_followups", lambda data, cut=None, video_ids=None: real(data, None, video_ids))
    monkeypatch.setattr(pf, "_next_in_history", lambda row, positions: True)
    assert any("pitch nm_receipt_share:" in line for line in leaks(records))


# ---- the reactions family: every investor's reactions in each answer's window ----------------

from tests.test_reaction_events import EXCHANGES as RX_EXCHANGES, INVESTORS as RX_INVESTORS  # noqa: E402
from tests.test_reaction_events import SCRIPT as RX_SCRIPT  # noqa: E402

RX_DECIDED = RX_SCRIPT + [("S1", "decides", None, (), None), ("S2", "decides", None, (), None)]   # turns 11, 12


def rx_row(script=RX_DECIDED, decisions=(("S1", "offer", 11), ("S2", "exit", 12))):
    data = data_of(talk_record(script, exchanges=RX_EXCHANGES, investors=RX_INVESTORS, decisions=decisions))
    return pitch_features(data).iloc[0]


def test_reactions_count_every_investor():
    row = rx_row()                         # the pitch's cut is S1's offer: turns 2..10 are history
    assert row.rx_answers == 3
    assert row.rx_reaction_rate == pytest.approx(5 / 3)          # turn order: S2's question (turn 9) is in a window
    assert row.rx_challenge_share == pytest.approx(1 / 3)


def test_reactions_stop_at_the_cut():
    early = rx_row(RX_SCRIPT, decisions=(("S1", "offer", 5),))     # the cut: the position of turn 5
    assert early.cut_pos < rx_row().cut_pos and early.rx_answers < rx_row().rx_answers


# ---- sensitivities: both passes averaged, exposure as rates -----------------------------------

def _answer(record, turn_id):
    return next(t for t in record["turns"] if t["turn_id"] == turn_id)


def _set_code(t, code, value):
    next(c for c in t["codes"] if c["code"] == code).update(value=value, missing="none")


def test_both_passes_average_numeric_features():
    """The story pitch, its replicate differing in one answer's substance (turn 3: 3 -> 5) and context (turn 3: not
    rated -> 2), in one Stage A rating (3 -> 5) and in the founders' gender. Every numeric feature is the mean of
    the two passes, a `_n` is their sum and a `_missing` their mean; a categorical control comes from the first."""
    replicate = synthetic_record()
    _set_code(_answer(replicate, 3), "PREP-SUBS", "5")
    _set_code(_answer(replicate, 3), "PREP-CTX", "2")
    replicate["stage_a"][0]["value"] = 5
    replicate["venture"]["founder_gender"] = "mixed"
    one = pitch_features(data_of(synthetic_record())).iloc[0]
    row = pitch_features(both_of(synthetic_record(), replicate)).iloc[0]
    # first pass: answers 3 and 5 have substance 3 and 4; the replicate 5 and 4
    assert stat(one, "a_subs_mean") == (3.5, 2, 0)
    assert stat(row, "a_subs_mean") == (pytest.approx((3.5 + 4.5) / 2), 4, 0)
    # no context rating in the first pass, one (2) in the replicate: the mean of what was observed
    assert math.isnan(one.a_ctx_mean) and stat(row, "a_ctx_mean") == (2.0, 1, 0.5)
    # turn 5, the answer to the hard question, is the same in both passes
    assert stat(row, "a_subs_hard_mean") == (4.0, 2, 0)
    assert row[f"sa_{STAGE_A_IDS[0]}"] == 4.0 and row[f"sa_{STAGE_A_IDS[1]}"] == 3.0
    assert row.founder_gender == "all_men" and (row.any_offer, row.cut_pos) == (1.0, 9)
    for column in ("hist_turns", "hist_seconds", "q_askers", "pres_subs", "a_resp_mean"):
        assert row[column] == one[column], column
    # both passes the same: the table of a single pass, but every `_n` counts the observations of both
    same_twice = pitch_features(both_of(synthetic_record(), synthetic_record()))
    single = pitch_features(data_of(synthetic_record()))
    twins = [c for c in single.columns if c.endswith("_n") and f"{c[:-2]}_missing" in single.columns]
    assert "a_subs_mean_n" in twins and "hist_turns" not in twins
    pd.testing.assert_frame_equal(same_twice.drop(columns=twins), single.drop(columns=twins), check_dtype=False)
    assert (same_twice[twins] == 2 * single[twins]).all().all()


def test_both_passes_read_the_trend_placeholder_as_missing():
    """Fix round 1, M1: a pass with no rated question has q_sev_trend 0.0 as a placeholder (its `_missing` 1). The
    average takes the other pass's trend alone, not the mean with 0.0; with neither pass rated it stays 0.0."""
    none = talk_record([("S1", "accepts", 1, (), None), ("S1", "decides", None, (), None)],
                       decisions=(("S1", "exit", 3),))
    rising = talk_record([("S1", "question", 1, (), 1), ("F1", "answer", 1, (2,), 3), ("S1", "question", 1, (), 3),
                          ("F1", "answer", 1, (4,), 3), ("S1", "decides", None, (), None)],
                         decisions=(("S1", "exit", 6),))
    assert stat(pitch_features(data_of(none)).iloc[0], "q_sev_trend") == (0.0, 0, 1)
    assert stat(pitch_features(data_of(rising)).iloc[0], "q_sev_trend") == (2.0, 2, 0)
    for first, second in ((none, rising), (rising, none)):
        row = pitch_features(both_of(first, second)).iloc[0]
        assert stat(row, "q_sev_trend") == (2.0, 2, 0.5)
    assert stat(pitch_features(both_of(none, none)).iloc[0], "q_sev_trend") == (0.0, 0, 1)
    assert pf.PLACEHOLDER_FEATURES == ("q_sev_trend",)


def test_each_pass_is_cut_at_its_own_first_decision():
    """The replicate misses S3's exit (turn 10), so its first decision is S1's offer (turn 17, position 16): its
    history is positions 0-15 less the decides turn 10, 15 turns, against 9 in the first pass. The row's cut and
    outcome are the first pass's."""
    replicate = synthetic_record()
    replicate["decisions"] = [d for d in replicate["decisions"] if d["decision_id"] != 1]
    second = data_of(replicate, source="replicate")
    assert int(pitch_cut(second).cut_pos.iloc[0]) == 16
    assert pitch_features(second).hist_turns.iloc[0] == 15
    row = pitch_features(both_of(synthetic_record(), replicate)).iloc[0]
    assert row.hist_turns == 12.0 and row.cut_pos == 9 and row.any_offer == 1.0


def test_a_pitch_without_a_second_pass_keeps_its_first():
    data = with_replicate(build_data([
        {"video_id": v, "record": synthetic_record(), "source": "primary", "release_id": "v3-dev9",
         "duration_s": 200.0} for v in ("a", "b")]), data_of(talk_record(TALK, decisions=TALK_DECISIONS), "b",
                                                              source="replicate"))
    assert data.books_source["single_pass_videos"] == {"a": "missing"}
    table = pitch_features(data).set_index("video_id")
    alone = pitch_features(data_of(synthetic_record(), "a")).set_index("video_id")
    pd.testing.assert_series_equal(table.loc["a", alone.columns], alone.loc["a"], check_dtype=False,
                                   check_names=False)
    assert table.loc["b", "hist_turns"] != alone.loc["a", "hist_turns"]


RATE_COLUMNS = ("hist_turns", "hist_questions", "accepts_turns", "challenge_turns", "social_turns", "founder_social",
                "hostile_at_founder", "founder_repairs")


def test_rates_divide_counts_by_minutes(story):
    """The story pitch's history is 102 seconds (1.7 minutes): each count of RATE_COLUMNS is per minute of it;
    hist_seconds, the shares, the means and the other counts are unchanged."""
    assert pf.RATE_COLUMNS == RATE_COLUMNS
    rates = pitch_features(data_of(synthetic_record()), exposure="rates").iloc[0]
    for column in RATE_COLUMNS:
        assert rates[column] == pytest.approx(story[column] / 1.7), column
    assert (rates.hist_turns, rates.hist_questions) == (pytest.approx(9 / 1.7), pytest.approx(2 / 1.7))
    for column in story.index:
        if column not in RATE_COLUMNS:
            assert same(rates[column], story[column]), column
    counts, per_minute = feature_dictionary().set_index("name"), feature_dictionary(exposure="rates").set_index("name")
    for column in RATE_COLUMNS:
        assert "per minute" in per_minute.label[column] and "per minute" in per_minute.description[column]
        assert "per minute" not in counts.label[column]
        assert per_minute.family[column] == counts.family[column]
    pd.testing.assert_frame_equal(per_minute.drop(list(RATE_COLUMNS)), counts.drop(list(RATE_COLUMNS)))
    shown = pf.describe_row(pitch_features(data_of(synthetic_record()), exposure="rates"), "vid").set_index("feature")
    assert "per minute" in shown.label["hist_turns"] and "per minute" not in shown.label["hist_seconds"]
    # the rates sensitivity of the ladder builds exactly this table
    from analysis.panel_bottleneck import SENSITIVITY_LADDERS
    variant = pitch_features(data_of(synthetic_record()), **SENSITIVITY_LADDERS["rates"]["features"])
    pd.testing.assert_frame_equal(variant, pitch_features(data_of(synthetic_record()), exposure="rates"))
    assert variant.attrs["exposure"] == "rates"
    with pytest.raises(ValueError, match="exposure"):
        pitch_features(data_of(synthetic_record()), exposure="share")


def test_a_history_without_seconds_has_no_rate():
    """A pitch whose first decision is spoken in the prepared presentation's turn: no history, so no rate."""
    row = pitch_features(data_of(talk_record([("S1", "question", 1, (), 2)], decisions=(("S1", "exit", 1),))),
                         exposure="rates").iloc[0]
    assert row.cut_pos == 0 and row.hist_seconds == 0.0
    for column in RATE_COLUMNS:
        assert math.isnan(row[column]), column
