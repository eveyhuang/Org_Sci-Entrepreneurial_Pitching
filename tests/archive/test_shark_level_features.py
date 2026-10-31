# Expectations re-pinned 2026-10-21: the next move and the reaction window are read on turn order (plan B2); the committed v3-dev8 shark-level record is unchanged and its pin test (test_shark_level_models.py) still passes.
"""analysis.archive.shark_level_features: the retired shark-level rows (one per pitch and investor), each
stopping at that investor's own cut. Moved from tests/test_panel_features.py with the code (plan B task 2);
the expected values are those of the shark-level rows before the move."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

import analysis.archive.shark_level_features as sl
import analysis.panel_features as pf
from analysis.archive.shark_level_dataset import investor_cuts
from analysis.archive.shark_level_features import (
    ARCHIVE_FEATURES, COMBINED_FEATURES, DECISION_HISTORY_FEATURES, INVESTOR_OUTCOME_COLUMNS, SHARK_FAMILIES,
    o2_features,
)
from analysis.panel_dataset import pitch_cut
from tests.panel_analysis_fixtures import FORCED_PATTERNS, make_record, talk_record, write_run
from tests.panel_fixtures import STAGE_A_IDS, synthetic_record, turn
from tests.test_panel_features import (
    LATE_RECEIPT, RECEIPT_AT_CUT, RECEIPT_AT_CUT_DECISIONS, TALK, TALK_DECISIONS, data_of, same, stat,
)

KEYS = ["video_id", "investor"]


def row(frame, **where):
    for key, value in where.items():
        frame = frame[frame[key] == value]
    assert len(frame) == 1, f"{where} matches {len(frame)} rows"
    return frame.iloc[0]


# ---- the leakage test -----------------------------------------------------------------------

def truncated(record: dict, data, cut_pos: int, focal: str, drop: str) -> dict:
    """`record` without any turn at or after `cut_pos`.

    drop="later": also without every decision at or after the cut and every decision of the focal
    investor, which is everything a row must not see; earlier decisions of the others are history.
    drop="all": without any decision at all.
    """
    out = copy.deepcopy(record)
    out["turns"] = out["turns"][:cut_pos]
    pos = dict(zip(data.decisions.decision_id, data.decisions.pos))
    out["decisions"] = [] if drop == "all" else [
        d for d in out["decisions"] if pos[d["decision_id"]] < cut_pos and d["actor"] != focal]
    return out


def leaks(records: dict[str, dict], drop: str = "later") -> list[str]:
    """Every feature of every investor row that changes when the row's future is deleted from its record."""
    found = []
    for video_id, record in records.items():
        data = data_of(record, video_id)
        full = o2_features(data)
        assert len(full), "the fixture must give rows"
        for _, original in full.iterrows():
            focal = original["investor"]
            again = o2_features(data_of(truncated(record, data, int(original["cut_pos"]), focal, drop), video_id))
            recomputed = again.merge(pd.DataFrame([original[KEYS]]), on=KEYS).iloc[0]
            for column in full.columns:
                if column in INVESTOR_OUTCOME_COLUMNS or column in KEYS:
                    continue
                if not same(original[column], recomputed[column]):
                    found.append(f"{video_id} {focal} {column}: {original[column]!r} -> {recomputed[column]!r}")
    return found


@pytest.fixture(scope="module")
def pitches() -> dict[str, dict]:
    rng = np.random.default_rng(11)
    records = {f"p{i}": make_record(rng, forced=FORCED_PATTERNS if i == 0 else None) for i in range(8)}
    records["story"] = synthetic_record()
    records["talk"] = talk_record(TALK, decisions=TALK_DECISIONS)
    records["late"] = talk_record(LATE_RECEIPT, decisions=(("S1", "exit", 4), ("S2", "exit", 6)))
    return records


def test_no_o2_feature_changes_when_the_rows_future_is_deleted(pitches):
    assert leaks(pitches) == []


def test_deleting_every_decision_changes_only_the_counts_of_earlier_decisions_by_others(pitches):
    """Earlier exits and offers of other investors are history a row may use, and they are the only
    thing any feature reads from the decision list: with no decisions at all those counts are zero
    and nothing else moves."""
    changed = {line.split()[2].rstrip(":") for line in leaks(pitches, drop="all")}
    assert changed == set(DECISION_HISTORY_FEATURES)
    for video_id, record in pitches.items():
        table = o2_features(data_of({**record, "decisions": []}, video_id))
        for name in DECISION_HISTORY_FEATURES:
            assert (table[name] == 0).all()


def test_the_leakage_check_catches_features_that_read_past_the_cut(pitches, monkeypatch):
    real = sl.eligible_turns_for_investor
    monkeypatch.setattr(sl, "eligible_turns_for_investor", lambda data, video_id, cut_pos, focal: real(
        data, video_id, 10 ** 9, focal))
    assert any(" hist_turns:" in line for line in leaks(pitches))


def test_the_leakage_check_catches_the_focal_investors_own_decisions(pitches, monkeypatch):
    monkeypatch.setattr(sl, "_decisions_before", lambda decisions, cut_pos, focal: decisions)
    found = leaks({"story": pitches["story"]})
    assert any("S1 oth_offers_before" in line for line in found)     # S1's own offer counted for S1
    assert any("S3 oth_exits_before" in line for line in found)


# ---- the registry -----------------------------------------------------------------------------

def test_the_archive_registry_is_apart_from_the_live_one(pitches):
    assert not set(ARCHIVE_FEATURES) & set(pf.FEATURES)
    assert set(COMBINED_FEATURES) == set(ARCHIVE_FEATURES) | set(pf.FEATURES)
    assert list(COMBINED_FEATURES)[:len(pf.FEATURES)] == list(pf.FEATURES)
    assert COMBINED_FEATURES["own_questions"] is ARCHIVE_FEATURES["own_questions"]
    assert COMBINED_FEATURES["hist_turns"] is pf.FEATURES["hist_turns"]
    scores = {code: {"pl_score": {"p0": 0.5}} for code in STAGE_A_IDS}
    table = o2_features(data_of(pitches["p0"], "p0", rank_scores=scores))
    for column in table.columns:
        if column in INVESTOR_OUTCOME_COLUMNS:
            continue
        assert column in COMBINED_FEATURES, column
        assert COMBINED_FEATURES[column]["unit"] in ("o2", "both"), column
        assert COMBINED_FEATURES[column]["family"] in SHARK_FAMILIES or COMBINED_FEATURES[column]["kind"] == "key"
    for name, spec in ARCHIVE_FEATURES.items():
        if name.endswith(("_mean", "_share")):
            assert ARCHIVE_FEATURES[f"{name}_n"]["family"] == spec["family"] == ARCHIVE_FEATURES[f"{name}_missing"]["family"]


# ---- hand-computed values on the story pitch ------------------------------------------------
# tests/panel_fixtures.py: S1 asks (turn 2) and is answered (3); S2 presses (4), is answered (5), the
# founder repairs the explanation (6) and S1 accepts it (7); S3 jokes (8), the founder joins in (9),
# S3 exits (10); S2 asks about repeat purchases (11), is answered (12), accepts (13); S2 asks again (14), is
# answered with a forecast (15) and challenges (16); S1 offers (17), S2 joins (18), the founder accepts (19).

@pytest.fixture(scope="module")
def story():
    return o2_features(data_of(synthetic_record()))


def test_rows_keys_outcomes_and_cuts(story):
    o2 = story
    assert list(o2.investor) == ["S1", "S2", "S3"]
    assert dict(zip(o2.investor, o2.cut_pos)) == {"S1": 16, "S2": 17, "S3": 9}
    assert dict(zip(o2.investor, o2.ever_offer)) == {"S1": 1.0, "S2": 1.0, "S3": 0.0}
    assert set(o2.source) == {"primary"}


def test_control_exposure_and_stage_a(story):
    s1, s3 = row(story, investor="S1"), row(story, investor="S3")
    for r in (s1, s3):
        assert (r.founder_gender, r.primary_industry, r.product_form, r.customer_type) == (
            "all_men", "home_household", "physical_good", "consumer")
        assert r.prepared_seconds == 55.0
        for code in STAGE_A_IDS:
            assert r[f"sa_{code}"] == 3.0 and r[f"sa_{code}_missing"] == 0
    assert s1.investor_label == "Investor A"
    # S1's history is turns 1-16 without S3's exit (turn 10)
    assert (s1.hist_turns, s1.hist_seconds, s1.hist_questions_all) == (15, 140.0, 4)
    assert (s3.hist_turns, s3.hist_seconds, s3.hist_questions_all) == (9, 102.0, 2)
    assert (s1.n_social_exchanges, s1.demo_seconds) == (1, 0.0)
    # S2's cut is turn 18: S1's offer (turn 17) lies before it but is never history
    assert row(story, investor="S2").hist_turns == 15


def test_own_family(story):
    s1, s2 = row(story, investor="S1"), row(story, investor="S2")
    assert s1.own_questions == 1
    assert stat(s1, "own_sev_mean") == (1.0, 1, 0)
    assert stat(s1, "own_prevention_share") == (1.0, 1, 0)
    assert stat(s1, "own_resp_mean") == (1.0, 1, 0)
    assert stat(s1, "own_evid_share") == (1.0, 1, 0)
    assert stat(s1, "own_uncert_share") == (0.0, 1, 0)
    assert stat(s1, "own_subs_mean") == (3.0, 1, 0)
    assert stat(s1, "own_coh_mean") == (4.0, 1, 0)
    assert math.isnan(s1.own_ctx_mean) and (s1.own_ctx_mean_n, s1.own_ctx_mean_missing) == (0, 1)
    assert stat(s1, "own_frame_match_share") == (1.0, 1, 0)
    # v3-dev9 records carry no closing state: every own substantive exchange counts as no signal
    assert (s1.own_exchanges, s1.own_satisfied, s1.own_unsatisfied, s1.own_no_signal) == (1, 0, 0, 1)
    assert (s1.own_challenges, s1.own_agrees) == (0, 1)                 # turn 7 accepts

    assert s2.own_questions == 3
    assert stat(s2, "own_sev_mean") == (pytest.approx(7 / 3), 3, 0)
    assert stat(s2, "own_evid_share") == (pytest.approx(2 / 3), 3, 0)
    assert stat(s2, "own_subs_mean") == (pytest.approx(10 / 3), 3, 0)
    assert stat(s2, "own_frame_match_share") == (pytest.approx(2 / 3), 3, 0)    # the forecast answer switches
    assert (s2.own_exchanges, s2.own_satisfied, s2.own_unsatisfied, s2.own_no_signal) == (2, 0, 0, 2)
    assert (s2.own_challenges, s2.own_agrees) == (1, 1)


def test_an_investor_who_asked_nothing_still_has_a_row_with_missing_means(story):
    s3 = row(story, investor="S3")
    assert s3.own_questions == 0 and s3.own_exchanges == 0
    for name in ("own_sev_mean", "own_prevention_share", "own_resp_mean", "own_evid_share", "own_uncert_share",
                 "own_subs_mean", "own_coh_mean", "own_ctx_mean", "own_frame_match_share"):
        assert math.isnan(s3[name]) and s3[f"{name}_n"] == 0 and s3[f"{name}_missing"] == 1, name
    assert (s3.own_satisfied, s3.own_unsatisfied, s3.own_no_signal) == (0, 0, 0)


def test_panel_content_family(story):
    s1, s3 = row(story, investor="S1"), row(story, investor="S3")
    assert (s1.oth_questions, s1.oth_questioners) == (3, 1)
    assert stat(s1, "oth_resp_mean") == (1.0, 3, 0)
    assert stat(s1, "oth_evid_share") == (pytest.approx(2 / 3), 3, 0)
    assert stat(s1, "oth_uncert_share") == (0.0, 3, 0)
    assert stat(s1, "oth_subs_mean") == (pytest.approx(10 / 3), 3, 0)
    assert stat(s1, "oth_coh_mean") == (4.0, 3, 0)
    assert s1.oth_ctx_mean_missing == 1
    assert stat(s1, "oth_prevention_share") == (1.0, 3, 0)
    assert (s3.oth_questions, s3.oth_questioners) == (2, 2)          # only turns 2 and 4 precede S3's exit
    assert stat(s3, "oth_subs_mean") == (3.5, 2, 0)


def test_panel_reactions_family(story):
    s1, s2, s3 = row(story, investor="S1"), row(story, investor="S2"), row(story, investor="S3")
    assert (s1.oth_agrees, s1.oth_challenges) == (1, 1)                # S2's accepts (13) and challenge (16)
    assert (s1.oth_satisfied, s1.oth_unsatisfied) == (0, 0)
    assert (s1.oth_exits_before, s1.oth_offers_before) == (1, 0)
    assert (s2.oth_agrees, s2.oth_challenges) == (1, 0)                # S1's accepts (7)
    assert (s2.oth_satisfied, s2.oth_unsatisfied) == (0, 0)
    assert (s3.oth_agrees, s3.oth_challenges, s3.oth_satisfied, s3.oth_unsatisfied) == (1, 0, 0, 0)
    for gone in ("oth_acknowledges", "oth_interest", "all_interest", "own_social_recip", "all_social_recip",
                 "own_trouble_signals", "trouble_signals"):
        assert gone not in COMBINED_FEATURES, gone


def test_other_investors_decisions_count_only_before_the_cut(story):
    s1, s2, s3 = row(story, investor="S1"), row(story, investor="S2"), row(story, investor="S3")
    assert (s3.oth_exits_before, s3.oth_offers_before) == (0, 0)     # S3 left before anyone decided
    assert (s1.oth_exits_before, s1.oth_offers_before) == (1, 0)     # S2 joined after S1 offered
    assert (s2.oth_exits_before, s2.oth_offers_before) == (1, 1)     # S3's exit and S1's offer came first


def test_rapport_and_repair_family(story):
    s1, s2, s3 = row(story, investor="S1"), row(story, investor="S2"), row(story, investor="S3")
    assert (s3.own_social, s3.founder_social_to_own) == (1, 1)
    assert (s1.own_social, s1.founder_social_to_own) == (0, 0)
    assert (s1.founder_repairs, s2.founder_repairs, s3.founder_repairs) == (1, 1, 1)
    assert (s1.own_hostile, s1.hostile_at_founder) == (0, 0)


def test_rapport_counts_hostile_repair_and_social_turns():
    record = synthetic_record()
    by_id = {t["turn_id"]: t for t in record["turns"]}
    by_id[16]["function"] = "hostile"              # S2
    by_id[7]["function"] = "social"                # S1, responding to the founder
    record["turns"].insert(16, turn(20, "F1", 154, 156, "repair", "Corrects a figure.", ["PANEL"], 1, [15]))
    record["turns"].insert(17, turn(21, "F1", 156, 157, "social", "Thanks the second investor.", ["S2"]))
    o2 = o2_features(data_of(record))
    s1, s2 = row(o2, investor="S1"), row(o2, investor="S2")
    assert (s2.own_hostile, s2.hostile_at_founder) == (1, 1)
    assert (s1.own_hostile, s1.hostile_at_founder) == (0, 1)
    assert s1.founder_repairs == 2 and s1.own_social == 1
    assert (s2.founder_social_to_own, s1.founder_social_to_own) == (1, 0)


def test_an_accepts_turn_by_the_focal_investor_is_read_like_any_other_turn():
    """Until v3-dev8 a focal investor's own expressed interest was left out of that investor's row; since the
    function merged into accepts (v3-dev9) it counts as the investor's own accepts turn."""
    base = synthetic_record()
    extra = copy.deepcopy(base)
    extra["turns"].insert(16, turn(20, "S2", 154, 156, "accepts", "Says she likes the product.", ["F1"]))
    before, after = o2_features(data_of(base)), o2_features(data_of(extra))
    assert row(after, investor="S2").own_agrees == row(before, investor="S2").own_agrees + 1
    assert row(after, investor="S2").cut_pos == row(before, investor="S2").cut_pos + 1
    assert row(after, investor="S1").oth_agrees == row(before, investor="S1").oth_agrees + 1


def test_focal_accepts_before_own_decision_is_history_until_plan_b(monkeypatch):
    """A known shark-level leak, pinned: S2 says she likes the product right before joining the offer. Until v3-dev8
    that was expresses_interest and FOCAL_EXCLUDED_FUNCTIONS kept it out of S2's own row; since v3-dev9 it is an
    accepts turn, FOCAL_EXCLUDED_FUNCTIONS is empty, and S2's row reads it, a tell of her decision. Plan B retires
    the shark-level rows; the pitch-level cut has no focal exclusion. If this test fails, the leak was closed or moved."""
    import analysis.archive.shark_level_dataset as pd_module

    record = synthetic_record()
    record["turns"].insert(17, turn(20, "S2", 163, 163.5, "accepts", "Says she likes the product.", ["F1"]))
    data = data_of(record)
    s2 = row(o2_features(data), investor="S2")
    assert pd_module.FOCAL_EXCLUDED_FUNCTIONS == ()
    assert 20 in set(pd_module.eligible_turns_for_investor(data, "vid", int(s2.cut_pos), "S2").turn_id)
    assert s2.own_agrees == 2                                   # turn 13 and the near-outcome turn 20
    monkeypatch.setattr(pd_module, "FOCAL_EXCLUDED_FUNCTIONS", ("accepts",))
    excluded = row(o2_features(data), investor="S2")
    assert excluded.own_agrees == 0 and excluded.cut_pos == s2.cut_pos


def test_decision_history_is_a_family_of_its_own():
    assert SHARK_FAMILIES[-1] == "decision_history"
    assert {n for n, spec in ARCHIVE_FEATURES.items() if spec["family"] == "decision_history"} == set(
        DECISION_HISTORY_FEATURES) == {"oth_exits_before", "oth_offers_before"}
    assert not [n for n, spec in ARCHIVE_FEATURES.items() if spec["family"] == "panel_reactions" and "before" in n]
    assert "exits_before" not in COMBINED_FEATURES


def test_history_questions_count_investor_questions_only():
    record = synthetic_record()
    record["turns"][5] = turn(6, "F1", 94, 100, "question", "Asks the investor what margin she expects.", ["S2"], 1,
                              codes=[{"code": "PR-CONCEPT", "value": "sales", "missing": "none"},
                                     {"code": "PR-SEV", "value": "1", "missing": "none"}])
    assert row(o2_features(data_of(record)), investor="S1").hist_questions_all == 4


# ---- the guard band ---------------------------------------------------------------------------

def test_guard_band_drops_the_focal_investors_last_turns_before_the_cut(story):
    data = data_of(synthetic_record())
    plain, guarded = story, o2_features(data, guard_turns=2)
    assert list(guarded.columns) == list(plain.columns) and list(guarded.cut_pos) == list(plain.cut_pos)
    s1, g1 = row(plain, investor="S1"), row(guarded, investor="S1")
    # S1's eligible turns are 2 (question) and 7 (accepts): both go
    assert (s1.hist_turns, g1.hist_turns) == (15, 13) and (s1.own_questions, g1.own_questions) == (1, 0)
    assert (s1.own_agrees, g1.own_agrees) == (1, 0) and g1.own_exchanges == 0
    assert g1.own_resp_mean_missing == 1 and g1.oth_questions == s1.oth_questions      # others are untouched
    s2, g2 = row(plain, investor="S2"), row(guarded, investor="S2")
    # S2's last two eligible turns are 14 (question) and 16 (challenge)
    assert (s2.own_questions, g2.own_questions) == (3, 2) and (s2.own_challenges, g2.own_challenges) == (1, 0)
    assert (s2.own_agrees, g2.own_agrees) == (1, 1)
    one = row(o2_features(data, guard_turns=1), investor="S2")
    assert (one.own_questions, one.own_challenges) == (3, 0)
    assert o2_features(data, guard_turns=0).equals(plain)
    with pytest.raises(ValueError, match="guard_turns"):
        o2_features(data, guard_turns=-1)


def test_rank_columns_appear_only_when_the_run_has_rank_scores():
    record = synthetic_record()
    assert not [c for c in o2_features(data_of(record)).columns if c.startswith("sa_rank_")]
    scores = {code: {"pl_score": {"vid": 0.25}} for code in STAGE_A_IDS}
    ranked = o2_features(data_of(record, rank_scores=scores))
    assert [c for c in ranked.columns if c.startswith("sa_rank_")] == [f"sa_rank_{code}" for code in STAGE_A_IDS]
    assert (ranked["sa_rank_interpersonal_warmth"] == 0.25).all()


def test_uncodable_stage_a_rating_is_missing_with_its_indicator():
    record = synthetic_record()
    record["stage_a"][0].update(value=None, uncodable=True)
    o2 = o2_features(data_of(record))
    assert o2[f"sa_{STAGE_A_IDS[0]}"].isna().all() and (o2[f"sa_{STAGE_A_IDS[0]}_missing"] == 1).all()


def test_no_free_text_column_and_every_family_is_represented(story):
    text_allowed = {"video_id", "investor", "source", "investor_label", "founder_gender", "primary_industry",
                    "product_form", "customer_type", "first_decision_kind"}
    for column in story.columns:
        if not pd.api.types.is_numeric_dtype(story[column]):
            assert column in text_allowed, column
    assert {COMBINED_FEATURES[c]["family"] for c in story.columns if c in COMBINED_FEATURES} - {None} == set(
        SHARK_FAMILIES)


# ---- the same rows, made inspectable --------------------------------------------------------------

def test_history_table_marks_exactly_the_turns_the_features_read():
    from analysis.archive.shark_level_dataset import eligible_turns_for_investor
    from analysis.archive.shark_level_features import REASON_GUARD, history_table

    data = data_of(make_record(np.random.default_rng(5), forced=FORCED_PATTERNS))
    features = o2_features(data)
    for r in investor_cuts(data).itertuples():
        table = history_table(data, "vid", r.investor)
        assert len(table) == len(data.turns) and (table.cut_pos == r.cut_pos).all()
        eligible = eligible_turns_for_investor(data, "vid", int(r.cut_pos), r.investor)
        assert list(table.index[table.used]) == list(eligible.index)
        assert (table.why_excluded[table.used] == "").all() and (table.why_excluded[~table.used] != "").all()
        assert not table.used[table.pos >= r.cut_pos].any()                    # never the decision or after it
        assert not table.used[table.function.isin(("decides", "negotiates"))].any()
        assert int(table.used.sum()) == int(row(features, investor=r.investor).hist_turns)
    pitch = history_table(data, "vid")
    assert int(pitch.used.sum()) == int(pf.pitch_features(data).hist_turns.iloc[0])
    assert (pitch.cut_pos == int(pitch_cut(data).cut_pos.iloc[0])).all()

    # the guard band also drops the focal investor's own last turns, and says so
    asker = features.sort_values("own_questions", ascending=False).investor.iloc[0]
    plain, guarded = history_table(data, "vid", asker), history_table(data, "vid", asker, guard_turns=2)
    dropped = guarded[plain.used & ~guarded.used]
    assert len(dropped) == 2 and set(dropped.speaker) == {asker} and set(dropped.why_excluded) == {REASON_GUARD}
    guarded_features = o2_features(data, guard_turns=2)
    assert int(guarded.used.sum()) == int(row(guarded_features, investor=asker).hist_turns)

    with pytest.raises(ValueError, match="Investors of this video: S1"):
        history_table(data, "vid", "S99")
    with pytest.raises(ValueError, match="no row for video 'other'"):
        history_table(data, "other")


def test_describe_row_reads_one_row_downwards_with_its_denominators():
    from analysis.archive.shark_level_features import describe_row

    data = data_of(make_record(np.random.default_rng(5), forced=FORCED_PATTERNS))
    features = o2_features(data)
    investor = features.sort_values("own_questions", ascending=False).investor.iloc[0]
    source = row(features, investor=investor)
    table = describe_row(features, "vid", investor).set_index("feature")
    assert list(table.columns) == ["family", "label", "value", "denominator", "description"]
    # a mean comes with the observations behind it; its _n and _missing companions are folded in, not listed
    assert table.denominator["own_resp_mean"] == int(source["own_resp_mean_n"]) > 0
    assert table.value["own_resp_mean"] == source["own_resp_mean"]
    assert not [name for name in table.index if name.endswith(("_n", "_missing"))]
    assert table.denominator["own_questions"] is None and table.value["own_questions"] == source["own_questions"]
    assert table.label["own_resp_mean"] == COMBINED_FEATURES["own_resp_mean"]["label"] == "answer directness (mean), own questions"
    assert {"video_id", "investor", "cut_pos", "ever_offer"}.isdisjoint(table.index)       # keys and outcomes are not features
    # an investor who asked nothing: the mean is missing with a denominator of 0, never zero
    for seed in range(40):
        other = o2_features(data_of(make_record(np.random.default_rng(seed), forced=FORCED_PATTERNS)))
        if (other.own_questions == 0).any():
            break
    quiet = other.investor[other.own_questions == 0].iloc[0]
    silent = describe_row(other, "vid", quiet).set_index("feature")
    assert math.isnan(silent.value["own_resp_mean"]) and silent.denominator["own_resp_mean"] == 0
    with pytest.raises(ValueError, match="expected one row"):
        describe_row(features, "vid")


# ---- the questioner's next move and hard questions, as features -----------------------------------

@pytest.fixture(scope="module")
def talk():
    data = data_of(talk_record(TALK, decisions=TALK_DECISIONS))
    return data, o2_features(data)


def test_next_move_features_of_the_focal_investor(talk):
    _, o2 = talk
    s1, s2 = row(o2, investor="S1"), row(o2, investor="S2")
    assert (s1.cut_pos, s2.cut_pos) == (8, 12)
    # S1: the first answer is followed by S1's harder question, the second by S1's agreement
    assert stat(s1, "own_receipt_share") == (0.5, 2, 0)
    assert stat(s1, "own_followup_share") == (0.5, 2, 0)
    assert stat(s1, "own_challenge_share") == (0.0, 2, 0)
    assert (s1.own_escalations, s1.own_hard_questions) == (1, 1)
    assert stat(s1, "own_subs_hard_mean") == (4.0, 1, 0) and stat(s1, "own_evid_hard_share") == (1.0, 1, 0)
    # S2: S1's decision talk follows the first answer (no reaction, in turn order), S2's own decision the second
    assert stat(s2, "own_receipt_share") == (0.0, 2, 0) and stat(s2, "own_followup_share") == (0.0, 2, 0)
    assert (s2.own_escalations, s2.own_hard_questions) == (0, 2)
    assert stat(s2, "own_subs_hard_mean") == (4.0, 2, 0) and stat(s2, "own_evid_hard_share") == (1.0, 2, 0)


def test_hard_questions_of_the_other_investors_and_the_answers_to_them(talk):
    _, o2 = talk
    s1, s2 = row(o2, investor="S1"), row(o2, investor="S2")
    assert s1.oth_hard_questions == 1 and stat(s1, "oth_subs_hard_mean") == (3.0, 1, 0)     # S2's turn 7 only
    assert stat(s1, "oth_evid_hard_share") == (1.0, 1, 0)
    assert s2.oth_hard_questions == 1 and stat(s2, "oth_subs_hard_mean") == (4.0, 1, 0)     # S1's turn 4
    for name in ("own_receipt_share", "own_escalations", "own_hard_questions", "own_subs_hard_mean"):
        assert ARCHIVE_FEATURES[name]["family"] == "own" and ARCHIVE_FEATURES[name]["unit"] == "o2", name
    for name in ("oth_hard_questions", "oth_subs_hard_mean", "oth_evid_hard_share"):
        assert ARCHIVE_FEATURES[name]["family"] == "panel_content", name


def test_next_move_features_on_the_story_pitch(story):
    s1, s2, s3 = row(story, investor="S1"), row(story, investor="S2"), row(story, investor="S3")
    assert stat(s1, "own_receipt_share") == (0.0, 1, 0)           # S2 spoke before S1 came back to it
    assert stat(s2, "own_receipt_share") == (pytest.approx(1 / 3), 3, 0)
    assert stat(s2, "own_challenge_share") == (pytest.approx(1 / 3), 3, 0)
    assert stat(s2, "own_followup_share") == (0.0, 3, 0) and s2.own_escalations == 0
    assert (s1.own_hard_questions, s2.own_hard_questions, s3.own_hard_questions) == (0, 3, 0)
    assert math.isnan(s1.own_subs_hard_mean) and (s1.own_subs_hard_mean_n, s1.own_subs_hard_mean_missing) == (0, 1)
    assert stat(s2, "own_subs_hard_mean") == (pytest.approx(10 / 3), 3, 0)
    assert stat(s2, "own_evid_hard_share") == (pytest.approx(2 / 3), 3, 0)
    assert s1.oth_hard_questions == 3 and stat(s1, "oth_subs_hard_mean") == (pytest.approx(10 / 3), 3, 0)
    assert s3.oth_hard_questions == 1 and stat(s3, "oth_subs_hard_mean") == (4.0, 1, 0)
    assert math.isnan(s3.own_receipt_share) and (s3.own_receipt_share_n, s3.own_receipt_share_missing) == (0, 1)


def test_a_receipt_after_the_askers_own_decision_is_not_read():
    data = data_of(talk_record(LATE_RECEIPT, decisions=(("S1", "exit", 4), ("S2", "exit", 6))))
    s1 = row(o2_features(data), investor="S1")
    assert stat(s1, "own_receipt_share") == (0.0, 1, 0)


def test_the_features_ask_for_next_moves_with_the_rows_own_cut(talk, monkeypatch):
    data = talk[0]
    asked = []
    real = pf.answer_followups

    def spy(data, cut=None, video_ids=None):
        asked.append((cut, tuple(video_ids)))
        return real(data, cut, video_ids)
    monkeypatch.setattr(pf, "answer_followups", spy)          # the shared `_followups` of the live module asks
    o2_features(data)
    assert asked == [({"vid": {"S1": 8}}, ("vid",)), ({"vid": {"S2": 12}}, ("vid",))]


def test_the_leakage_check_catches_a_next_move_read_past_the_cut(monkeypatch):
    records = {"late": talk_record(RECEIPT_AT_CUT, decisions=RECEIPT_AT_CUT_DECISIONS)}
    assert leaks(records) == []
    real = pf.answer_followups
    monkeypatch.setattr(pf, "answer_followups", lambda data, cut=None, video_ids=None: real(data, None, video_ids))
    monkeypatch.setattr(pf, "_next_in_history", lambda row, positions: True)
    assert any("S1 own_receipt_share:" in line for line in leaks(records))


def test_guard_band_also_hides_a_next_move_it_drops(talk):
    data = talk[0]
    one, two = (row(o2_features(data, guard_turns=k), investor="S1") for k in (1, 2))
    # S1's agreement (turn 6) is dropped: the second answer is still an answer to S1, with no receipt seen
    assert stat(one, "own_receipt_share") == (0.0, 2, 0) and stat(one, "own_followup_share") == (0.5, 2, 0)
    # S1's second question (turn 4) is dropped as well: one own question, one answer, nothing after it
    assert stat(two, "own_followup_share") == (0.0, 1, 0) and (two.own_escalations, two.own_hard_questions) == (0, 0)


# ---- the reaction_events family: the asker's and the panel's reactions, apart ----------------

from tests.test_reaction_events import EXCHANGES as RX_EXCHANGES, INVESTORS as RX_INVESTORS  # noqa: E402
from tests.test_reaction_events import SCRIPT as RX_SCRIPT  # noqa: E402

RX_DECIDED = RX_SCRIPT + [("S1", "decides", None, (), None), ("S2", "decides", None, (), None)]   # turns 11, 12


def rx_table(script=RX_DECIDED, decisions=(("S1", "offer", 11), ("S2", "exit", 12)), guard_turns=0):
    data = data_of(talk_record(script, exchanges=RX_EXCHANGES, investors=RX_INVESTORS, decisions=decisions))
    return o2_features(data, guard_turns=guard_turns)


def test_reaction_events_family_own_trajectory():
    o2 = rx_table()
    s1 = row(o2, investor="S1")
    assert s1.rx_own_answers == 2
    assert (s1.rx_own_receipt_share, s1.rx_own_followup_share, s1.rx_own_challenge_share) == (0.5, 0.5, 0.0)
    assert not [c for c in o2.columns if c.startswith("rx_") and "interest" in c]
    assert s1.rx_own_followup_sev_mean == 3.0
    # the last own answer drew a receipt; over the answers the asker moved from a follow-up to a receipt
    assert (s1.rx_own_last_receipt, s1.rx_own_last_followup, s1.rx_own_last_challenge) == (1.0, 0.0, 0.0)
    assert (s1.rx_own_trend_receipt, s1.rx_own_trend_followup, s1.rx_own_trend_challenge) == (1.0, -1.0, 0.0)
    assert s1.rx_own_other_first_share == 0.5
    assert not [c for c in o2.columns if c.startswith("rx_") and "valence" in c]
    s2 = row(o2, investor="S2")
    assert (s2.rx_own_answers, s2.rx_own_last_receipt, s2.rx_own_last_followup, s2.rx_own_trend_receipt) == (1, 0.0, 0.0, 0.0)
    assert s2.rx_own_followup_sev_mean_missing == 1


def test_reaction_events_family_panel_excludes_the_focal_investor():
    o2 = rx_table()
    s1 = row(o2, investor="S1")
    assert (s1.rx_pan_answers, s1.rx_pan_reactors) == (3, 2)
    assert s1.rx_pan_reaction_rate == 1.0          # turn order: S2's question (turn 9) is in the second window
    # S3's accepts after the second answer is a receipt (it was "interest" until v3-dev8)
    assert (s1.rx_pan_challenge_share, s1.rx_pan_receipt_share) == (pytest.approx(1 / 3), pytest.approx(1 / 3))
    # the last answer (turn 10) drew nothing from the others; challenges came early, none late
    assert (s1.rx_pan_last_challenge, s1.rx_pan_last_receipt, s1.rx_pan_last_challenge_missing) == (0.0, 0.0, 0)
    assert (s1.rx_pan_trend_challenge, s1.rx_pan_trend_receipt) == (-1.0, 0.0)
    s2 = row(o2, investor="S2")             # S2's own challenge is never a panel reaction in S2's row
    assert (s2.rx_pan_challenge_share, s2.rx_pan_reaction_rate) == (0.0, 1.0)
    assert s2.rx_pan_followup_share == pytest.approx(1 / 3)


def test_reaction_events_family_stops_at_the_cut():
    o2 = rx_table(RX_SCRIPT, decisions=(("S1", "offer", 5),))     # S1's cut: the position of turn 5
    s1 = row(o2, investor="S1")
    assert (s1.rx_own_answers, s1.rx_own_followup_share, s1.rx_own_followup_share_n) == (1, 0.0, 1)
    assert (s1.rx_own_last_followup, s1.rx_own_other_first_share) == (0.0, 1.0)


def test_reaction_events_family_respects_the_guard_band():
    o2 = rx_table(guard_turns=1)       # hides S1's last own turn, the agreement (turn 7)
    s1 = row(o2, investor="S1")
    assert (s1.rx_own_answers, s1.rx_own_receipt_share, s1.rx_own_last_receipt) == (2, 0.0, 0.0)


def test_planted_reaction_signal_shows_in_the_course_of_the_own_reactions(tmp_path):
    from analysis.panel_dataset import load_run
    o2 = o2_features(load_run(write_run(tmp_path, 12, planted="reaction_events", strength=4.0)))
    known = o2[o2.ever_offer.notna() & (o2.rx_own_answers > 0)]
    by_outcome = known.groupby("ever_offer")[["rx_own_last_receipt", "rx_own_trend_receipt", "own_receipt_share"]].mean()
    assert by_outcome.rx_own_last_receipt[1.0] - by_outcome.rx_own_last_receipt[0.0] > 0.5
    assert by_outcome.rx_own_trend_receipt[1.0] - by_outcome.rx_own_trend_receipt[0.0] > 0.5
    # the plant leaves the first-move share alike, so only the new family can see it
    assert abs(by_outcome.own_receipt_share[1.0] - by_outcome.own_receipt_share[0.0]) < 0.15
