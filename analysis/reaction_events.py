"""Every investor reaction after a founder's answer, the asker's apart from the rest of the panel's.

`answer_followups` keeps one move per answer: the first investor turn after it. This module keeps the
whole reaction window instead, from the turn after the answer to the turn before the founder's next
answer (or the first decision talk or turn of a negotiation exchange, or the end of the video), in turn
order whatever exchange the model put each turn in, and splits it in two: the asker's own first
reaction, and every reaction by the other investors. The exchange grouping is not read (the
researcher's decision of 2026-10-21: it is unstable between passes).

A documented limit of turn order: a reaction to an earlier answer that comes after a later answer (an
investor returning to a claim two question-answer pairs on) falls in the window of the nearest
preceding answer, not of the answer it refers to.

`reaction_events` is one row per (answer, asker) with the window summarised; `reaction_turns` is the
long companion, one row per reacting turn, which the feature family filters by each model row's
eligible positions so that the guard band and the focal investor's excluded turns hide reactions
exactly as they hide next moves.

Each kind of reaction stays its own category: no kind is weighed against another on a single numeric
scale (an earlier "valence" score did so without support in the literature and was removed). A follow-up
question keeps its severity. Read-only; everything comes from `data.turns`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.panel_dataset import (
    AFTER_CUT, EXCLUDED_FUNCTIONS, NEGOTIATION, RECEIPT_FUNCTIONS, SKIPPED_NEXT_FUNCTIONS, PanelData,
    answer_followups,
)

REACTION_KINDS = ("receipt", "follow_up", "challenge", "other")
NONE = "none"
ASKER, OTHER = "asker", "other"
EVENT_COLUMNS = (
    "video_id", "answer_turn", "answer_pos", "answer_start_s", "exchange_id", "question_turn", "asker", "question_sev",
    "hard_question", "subs", "evid", "resp",
    "window_end_pos",                                              # the window: answer_pos+1 .. window_end_pos-1
    "asker_reaction", "asker_reaction_pos", "asker_followup_sev",
    "pan_reactions", "pan_receipts", "pan_follow_ups", "pan_challenges", "pan_other",
    "pan_max_followup_sev", "pan_first_kind", "pan_first_actor", "pan_first_pos", "pan_actors",
    "first_reactor",                                               # "asker" | "other" | "none" | "after_cut"
    "pan_truncated",                                               # the asker's cut falls inside the window
)
TURN_COLUMNS = ("video_id", "answer_turn", "question_turn", "asker", "pos", "turn_id", "actor", "function", "kind",
                "severity", "is_asker")
_PAN_COUNTS = {"receipt": "pan_receipts", "follow_up": "pan_follow_ups", "challenge": "pan_challenges",
               "other": "pan_other"}


def _ends_window(t: dict) -> bool:
    return t["function"] in EXCLUDED_FUNCTIONS or t["exchange_kind"] == NEGOTIATION


def reaction_window(turns: list[dict], answer: dict) -> tuple[int, list[dict]]:
    """(window_end_pos, investor turns in the window), `turns` being the video's turns in `pos` order.

    The window runs in turn order from the turn after `answer` to the turn before the next founder
    answer, or to the first `decides`/`negotiates` turn or turn of a negotiation exchange, whichever
    comes first, or past the video's last turn; exchange membership is not read. Investor turns whose
    function is in SKIPPED_NEXT_FUNCTIONS are no reaction and are left out.

    The window and `answer_followups`' next move are not the same rule: the next move passes over every
    founder turn (a co-founder adding to the answer), while the window closes at the next founder answer,
    so after two founder answers in a row the next move may be a reaction the first answer's window does
    not hold. With a cut at a decision that closes the window, the window is empty (`first_reactor`
    "none"), while the next move is `after_cut`.
    """
    end = answer["pos"] + 1
    window = []
    for t in turns[answer["pos"] + 1:]:
        if _ends_window(t):
            return t["pos"], window
        if t["role"] == "founder" and t["function"] == "answer":
            return t["pos"], window
        end = t["pos"] + 1
        if t["role"] == "investor" and t["function"] not in SKIPPED_NEXT_FUNCTIONS:
            window.append(t)
    return end, window


def reaction_kind(turn: dict) -> str:
    """receipt (RECEIPT_FUNCTIONS) | follow_up (question) | challenge | other."""
    function = turn["function"]
    if function in RECEIPT_FUNCTIONS:
        return "receipt"
    return {"question": "follow_up", "challenges": "challenge"}.get(function, "other")


def _reacting(window: list[dict], asker: str, cut_pos) -> list[dict]:
    """The window's reactions as turn rows, without those at or after the asker's cut."""
    rows = []
    for t in window:
        if cut_pos is not None and t["pos"] >= cut_pos:
            break
        kind = reaction_kind(t)
        severity = float(t["PR-SEV"]) if kind == "follow_up" and t["PR-SEV"] == t["PR-SEV"] else np.nan
        rows.append({"pos": t["pos"], "turn_id": t["turn_id"], "actor": t["speaker"], "function": t["function"],
                     "kind": kind, "severity": severity,
                     "is_asker": t["speaker"] == asker})
    return rows


def _walk(data: PanelData, cut: dict | None, video_ids):
    """(answer_followups row, window_end_pos, reacting turn rows, cut_pos) for every (answer, asker)."""
    followups = answer_followups(data, None, video_ids)
    turns = data.turns if video_ids is None else data.turns[data.turns.video_id.isin(list(video_ids))]
    ordered = {v: t.sort_values("pos").to_dict("records") for v, t in turns.groupby("video_id", sort=True)}
    windows: dict[tuple, tuple] = {}
    for row in followups.to_dict("records"):
        video = ordered[row["video_id"]]
        key = (row["video_id"], row["answer_turn"])
        if key not in windows:
            windows[key] = reaction_window(video, video[int(row["answer_pos"])])
        end, window = windows[key]
        cut_pos = ((cut or {}).get(row["video_id"]) or {}).get(row["asker"])
        yield row, end, _reacting(window, row["asker"], cut_pos), cut_pos


def reaction_turns(data: PanelData, cut: dict | None = None, video_ids=None) -> pd.DataFrame:
    """One row per (answer, asker, reacting turn) of `reaction_events`, at most up to the asker's cut."""
    rows = []
    for row, _, reacting, _ in _walk(data, cut, video_ids):
        head = {"video_id": row["video_id"], "answer_turn": row["answer_turn"],
                "question_turn": row["question_turn"], "asker": row["asker"]}
        rows.extend({**head, **r} for r in reacting)
    table = pd.DataFrame(rows, columns=list(TURN_COLUMNS))
    table["severity"] = table["severity"].astype(float)
    table["is_asker"] = table["is_asker"].astype(bool)
    return table


def reaction_events(data: PanelData, cut: dict | None = None, video_ids=None) -> pd.DataFrame:
    """One row per (answer, asker) of answer_followups(data, video_ids=video_ids), with the window's reactions.

    The asker's reaction is the asker's first investor turn in the window; panel reactions are every
    turn in it by another investor (so with two askers, the other asker's reaction is a panel reaction
    in this row). With `cut` ({video_id: {investor: cut_pos}}), reactions at or after the ASKER's cut
    are unknown to the row: they are dropped from every count, `pan_truncated` is True when the cut
    falls inside the window, and the asker's reaction (and `first_reactor`) is `after_cut` when the
    asker has not reacted before the cut and the window runs past it. Without `cut`, the whole window.
    """
    rows = []
    for row, end, reacting, cut_pos in _walk(data, cut, video_ids):
        truncated = cut_pos is not None and cut_pos < end
        own = [r for r in reacting if r["is_asker"]]
        panel = [r for r in reacting if not r["is_asker"]]
        first_own = own[0] if own else None
        first_pan = panel[0] if panel else None
        if first_own is not None:
            asker_reaction = first_own["kind"]
        else:
            asker_reaction = AFTER_CUT if truncated else NONE
        if reacting:
            first_reactor = ASKER if reacting[0]["is_asker"] else OTHER
        else:
            first_reactor = AFTER_CUT if truncated else NONE
        follow_sevs = [r["severity"] for r in panel if r["kind"] == "follow_up" and r["severity"] == r["severity"]]
        counts = {name: sum(r["kind"] == kind for r in panel) for kind, name in _PAN_COUNTS.items()}
        rows.append({
            "video_id": row["video_id"], "answer_turn": row["answer_turn"], "answer_pos": row["answer_pos"],
            "answer_start_s": row["answer_start_s"], "exchange_id": row["exchange_id"],
            "question_turn": row["question_turn"], "asker": row["asker"], "question_sev": row["question_sev"],
            "hard_question": row["hard_question"], "subs": row["subs"], "evid": row["evid"], "resp": row["resp"],
            "window_end_pos": end,
            "asker_reaction": asker_reaction,
            "asker_reaction_pos": first_own["pos"] if first_own else np.nan,
            "asker_followup_sev": first_own["severity"] if first_own else np.nan,
            "pan_reactions": len(panel), **counts,
            "pan_max_followup_sev": max(follow_sevs) if follow_sevs else np.nan,
            "pan_first_kind": first_pan["kind"] if first_pan else None,
            "pan_first_actor": first_pan["actor"] if first_pan else None,
            "pan_first_pos": first_pan["pos"] if first_pan else np.nan,
            "pan_actors": len({r["actor"] for r in panel}),
            "first_reactor": first_reactor, "pan_truncated": truncated})
    table = pd.DataFrame(rows, columns=list(EVENT_COLUMNS))
    for name in ("asker_reaction_pos", "asker_followup_sev", "pan_max_followup_sev", "pan_first_pos", "question_sev", "subs", "evid", "resp"):
        table[name] = table[name].astype(float)
    for name in ("hard_question", "pan_truncated"):
        table[name] = table[name].astype(bool)
    return table
