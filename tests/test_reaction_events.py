"""analysis.reaction_events: every investor reaction after an answer, the asker's apart from the panel's."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from analysis.reaction_events import (
    EVENT_COLUMNS, REACTION_KINDS, TURN_COLUMNS, reaction_events, reaction_kind, reaction_turns, reaction_window,
)
from tests.panel_analysis_fixtures import talk_record
from analysis.panel_dataset import build_data


def data_of(record, video_id="vid"):
    """The record as a one-video PanelData (as tests.test_panel_features.data_of; kept here to avoid a cycle)."""
    return build_data([{"video_id": video_id, "record": record, "source": "primary", "release_id": "v3-dev9",
                        "duration_s": 200.0}])

# (speaker, function, exchange id, responds_to, severity of a question / substance of an answer); ids start at 2
SCRIPT = [
    ("S1", "question", 1, (), 2),               # 2: S1 asks, severity 2
    ("F1", "answer", 1, (2,), 4),               # 3: the answer
    ("S2", "challenges", 1, (3,), None),        # 4: another investor reacts first, with a challenge
    ("S1", "question", 1, (3,), 3),             # 5: the asker follows up, severity 3
    ("F1", "answer", 1, (5,), 2),               # 6: the next answer closes the first window
    ("S1", "accepts", 1, (6,), None),           # 7: the asker accepts the second answer
    ("S3", "accepts", 1, (6,), None),           # 8: another investor shows interest
    ("S2", "question", 2, (), 1),               # 9: a new exchange, still in the second window
    ("F1", "answer", 2, (9,), 3),               # 10: closes the second window; nobody reacts
]
EXCHANGES = ((1, "substantive"), (2, "substantive"))
INVESTORS = ("S1", "S2", "S3")


def scripted(script=SCRIPT, exchanges=EXCHANGES, investors=INVESTORS, decisions=()):
    return data_of(talk_record(script, exchanges=exchanges, investors=investors, decisions=decisions))


def pos_of(data, turn_id: int) -> int:
    return int(data.turns.loc[data.turns.turn_id == turn_id, "pos"].iloc[0])


def event(table, answer_turn: int, asker: str | None = None):
    rows = table[table.answer_turn == answer_turn]
    if asker is not None:
        rows = rows[rows.asker == asker]
    assert len(rows) == 1, f"answer {answer_turn} asker {asker}: {len(rows)} rows"
    return rows.iloc[0]


def test_columns_and_kinds():
    data = scripted()
    assert list(reaction_events(data).columns) == list(EVENT_COLUMNS)
    assert list(reaction_turns(data).columns) == list(TURN_COLUMNS)
    assert REACTION_KINDS == ("receipt", "follow_up", "challenge", "other")
    assert not [c for c in EVENT_COLUMNS if "interest" in c]


def test_the_window_holds_the_asker_and_the_panel_apart():
    data = scripted()
    table = reaction_events(data)
    first = event(table, 3)
    assert first.window_end_pos == pos_of(data, 6)
    assert (first.asker_reaction, first.asker_followup_sev) == ("follow_up", 3.0)
    assert first.first_reactor == "other"
    assert (first.pan_reactions, first.pan_challenges, first.pan_first_kind, first.pan_first_actor) == (1, 1, "challenge", "S2")
    assert first.pan_actors == 1
    second = event(table, 6)
    assert second.window_end_pos == pos_of(data, 10)
    assert (second.asker_reaction, second.first_reactor) == ("receipt", "asker")
    assert (second.pan_receipts, second.pan_follow_ups, second.pan_actors) == (1, 1, 2)   # S2's question in exchange 2
    third = event(table, 10)
    assert (third.asker_reaction, third.pan_reactions, third.first_reactor) == ("none", 0, "none")
    assert math.isnan(third.asker_followup_sev) and pd.isna(third.pan_first_kind)
    # the answer's own ratings come along
    assert (first.question_sev, first.subs, bool(first.hard_question)) == (2.0, 4.0, True)


def test_reaction_turns_are_one_row_per_reacting_turn():
    data = scripted()
    turns = reaction_turns(data)
    first = turns[turns.answer_turn == 3].sort_values("pos")
    assert list(first.actor) == ["S2", "S1"] and list(first.kind) == ["challenge", "follow_up"]
    assert list(first.is_asker) == [False, True] and first.severity.tolist()[1] == 3.0
    assert turns[turns.answer_turn == 10].empty


def test_two_askers_give_two_rows_and_each_panel_excludes_only_its_own_asker():
    script = [("S1", "question", 1, (), 1), ("S2", "question", 1, (), 2), ("F1", "answer", 1, (2, 3), 3),
              ("S2", "accepts", 1, (4,), None), ("S1", "challenges", 1, (4,), None)]
    table = reaction_events(scripted(script, exchanges=((1, "substantive"),)))
    s1, s2 = event(table, 4, "S1"), event(table, 4, "S2")
    assert (s1.asker_reaction, s1.pan_receipts, s1.pan_challenges) == ("challenge", 1, 0)
    assert (s2.asker_reaction, s2.pan_receipts, s2.pan_challenges) == ("receipt", 0, 1)
    assert (s1.first_reactor, s2.first_reactor) == ("other", "asker")


def test_a_decision_ends_the_window_and_other_exchanges_are_read_in_turn_order():
    script = [("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3),
              ("S2", "question", 2, (), 1),         # 4: another exchange, interleaved: in the window all the same
              ("S1", "accepts", 1, (3,), None),     # 5: still the first window
              ("S2", "decides", None, (), None),    # 6: decision talk ends the window
              ("S3", "challenges", 1, (3,), None)]  # 7: after the decision: not a reaction
    data = scripted(script)
    first = event(reaction_events(data), 3)
    assert first.window_end_pos == pos_of(data, 6)
    assert (first.asker_reaction, first.pan_reactions, first.pan_follow_ups) == ("receipt", 1, 1)


def _window(data, answer_turn: int):
    ordered = data.turns.sort_values("pos").to_dict("records")
    return reaction_window(ordered, ordered[pos_of(data, answer_turn)])


def test_reaction_window_runs_to_the_next_founder_answer_or_decision_talk():
    two = ((1, "substantive"), (2, "substantive"))
    data = scripted([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3),
                     ("S2", "accepts", 2, (), None),          # 4: another exchange
                     ("S1", "question", None, (), 1),         # 5: outside any exchange
                     ("F1", "answer", 1, (5,), 3),            # 6: the next founder answer closes the window
                     ("S3", "accepts", 1, (), None)], exchanges=two)
    end, window = _window(data, 3)
    assert end == pos_of(data, 6) and [t["turn_id"] for t in window] == [4, 5]
    data = scripted([("S1", "question", 1, (), 2), ("F1", "answer", 1, (2,), 3),
                     ("S2", "challenges", 2, (), None),       # 4
                     ("S2", "decides", None, (), None),       # 5: decision talk closes the window
                     ("S1", "accepts", 1, (), None)], exchanges=two)
    end, window = _window(data, 3)
    assert end == pos_of(data, 5) and [t["turn_id"] for t in window] == [4]


def test_with_a_cut_a_reaction_at_or_after_the_askers_cut_is_unknown():
    data = scripted()
    late = event(reaction_events(data, cut={"vid": {"S1": pos_of(data, 5)}}), 3)
    assert late.asker_reaction == "after_cut" and late.first_reactor == "other"
    assert (late.pan_reactions, late.pan_challenges, bool(late.pan_truncated)) == (1, 1, True)
    early = event(reaction_events(data, cut={"vid": {"S1": pos_of(data, 4)}}), 3)
    assert early.asker_reaction == "after_cut" and early.first_reactor == "after_cut"
    assert (early.pan_reactions, bool(early.pan_truncated)) == (0, True)
    cut_turns = reaction_turns(data, cut={"vid": {"S1": pos_of(data, 4)}})
    assert cut_turns[cut_turns.answer_turn == 3].empty


def test_no_reaction_is_scored_on_a_single_warmth_scale():
    # every kind stays its own category; no unsupported numeric weighting of kinds against each other
    assert not [c for c in (*EVENT_COLUMNS, *TURN_COLUMNS) if "valence" in c]
    import analysis.reaction_events as rx
    assert not hasattr(rx, "valence") and not hasattr(rx, "VALENCE")


def test_reaction_kind_of_each_function():
    assert [reaction_kind({"function": f}) for f in ("accepts", "question", "challenges", "hostile", "social")] == \
        ["receipt", "follow_up", "challenge", "other", "other"]


def test_an_accepts_turn_after_an_answer_is_a_receipt():
    turns = reaction_turns(scripted())
    second = turns[turns.answer_turn == 6].sort_values("pos")
    assert list(second.function) == ["accepts", "accepts", "question"]          # S2's question: turn order
    assert list(second.kind) == ["receipt", "receipt", "follow_up"]
