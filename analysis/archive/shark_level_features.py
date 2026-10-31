"""The shark-level feature rows (the retired O2 design): one row per pitch and investor, each stopping at
that investor's own cut.

Moved from `analysis.panel_features`, which now builds one row per pitch (`pitch_features`); the shared
building blocks are imported from there, never the reverse. An investor's row is computed from
`eligible_turns_for_investor` (strictly before the investor's own first offer, joined offer or exit, no
`decides` or `negotiates` turn, no negotiation exchange). Beyond turns it reads the exits and offers of
*other* investors strictly before the cut (the family `decision_history`, kept apart from what the panel
said because it encodes the order in which investors decided) and the facts fixed before the questioning
starts (venture controls, Stage A ratings). A closing state counts only when its turn is eligible;
records of v3-dev9 have none, so every exchange counts as no_signal.

"own" means the questions the focal investor asked and the answers responding to them; "others" the same
for the other investors. `rx_own_*` is the focal investor's own reaction to the answers to their own
questions, `rx_pan_*` every reaction by an investor other than the focal one. `guard_turns = k` is a
sensitivity setting: it additionally drops the focal investor's own last k eligible turns before the cut.

The declarations of these columns live in ARCHIVE_FEATURES, apart from the live pitch-level FEATURES.
COMBINED_FEATURES and COMBINED_FAMILIES read both, for the shark-level model sets
(`analysis.archive.shark_level_models`) and for the report and figures while they still show shark rows.

`n_social_exchanges` (social exchanges with an eligible turn) is computed and declared here since plan B2
(2026-10-21) dropped it from the live pitch rows, which no longer read the exchange grouping; the archived
primary set keeps it at its rapport step. `demo_seconds` comes with the live exposure and so sums the
`demonstrates` turns.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import pandas as pd

from analysis import panel_features as pf
from analysis.archive.shark_level_dataset import (
    eligible_turns_for_investor, exclusion_reasons_for_investor, investor_cuts, investor_outcomes,
)
from analysis.panel_dataset import FOLLOW_UP, RECEIPT_FUNCTIONS, PanelData, exclusion_reasons, pitch_cut
from analysis.panel_features import (
    ANSWER_STATS, CONTROLS, REPAIR_FUNCTIONS, RX_SHARES, SOCIAL_FUNCTIONS, SUBSTANTIVE, _answer_stats,
    _answers_to, _count, _declare_stat, _describe, _dictionary, _exposure, _followups, _hard, _next_move_stats, _per_answer,
    _rx_windows, _stage_a, _stat, _trend, _with_ranks, declare_stage_a,
)
from pipeline.stage_b.panel_books import makes_offer

SOCIAL_EXCHANGE = "social"       # the exchange kind n_social_exchanges counts
# the shark-level families in their order; control, exposure, stage_a and rapport_repair are shared with the pitch rows
SHARK_FAMILIES = ("control", "exposure", "stage_a", "own", "panel_content", "panel_reactions", "rapport_repair",
                  "reaction_events", "decision_history")
INVESTOR_OUTCOME_COLUMNS = ("ever_offer", "first_decision_kind", "first_decision_pos")
# the only features read from the decision list: earlier exits and offers of investors other than the focal one
DECISION_HISTORY_FEATURES = ("oth_exits_before", "oth_offers_before")
SHOWN_CLOSINGS = ("satisfied", "unsatisfied")
_WHOSE = {"own": "to the focal investor's own questions", "oth": "to the other investors' questions"}
_WHOSE_LABELS = {"own": "own questions", "oth": "others' questions"}
_RX_COURSE = "Each kind is its own column: no kind is weighed against another on one scale."

ARCHIVE_FEATURES: dict[str, dict] = {}


def _declare(name: str, family: str | None, label: str, unit: str, kind: str, description: str) -> None:
    ARCHIVE_FEATURES[name] = {"family": family, "label": label, "unit": unit, "kind": kind, "description": description}


def _stat_declared(name: str, family: str, label: str, description: str) -> None:
    _declare_stat(name, family, label, "o2", description, registry=ARCHIVE_FEATURES)


def _declare_counts(family: str, items: tuple[tuple[str, str, str], ...]) -> None:
    for name, label, description in items:
        _declare(name, family, label, "o2", "concept", description)


def _declare_answers(prefix: str, family: str) -> None:
    for suffix, _, description in ANSWER_STATS:
        _stat_declared(f"{prefix}_{suffix}", family, f"{pf.ANSWER_LABELS[suffix]}, {_WHOSE_LABELS[prefix]}",
                       f"{description[0].upper()}{description[1:]} {_WHOSE[prefix]}, before the cut.")


def _declare_hard(prefix: str, family: str) -> None:
    whose = {"own": "the focal investor's", "oth": "the other investors'"}[prefix]
    label = {"own": "own hard questions", "oth": "others' hard questions"}[prefix]
    _declare(f"{prefix}_hard_questions", family, label, "o2", "exposure",
             f"Of {whose} questions, those that ask for substantiation or contest a claim (severity 2 or 3).")
    _stat_declared(f"{prefix}_subs_hard_mean", family, f"answer substance (mean), {label}",
                   f"Mean substance rating of the answers to {whose} hard questions (severity 2 or 3), before the cut.")
    _stat_declared(f"{prefix}_evid_hard_share", family, f"answers with checkable support (share), {label}",
                   f"Share of the answers to {whose} hard questions (severity 2 or 3) that give evidence, before "
                   "the cut.")


def _declare_all() -> None:
    _declare("investor", None, "investor", "o2", "key", "Actor id of the focal investor within the pitch.")
    _declare("investor_label", "control", "investor name", "o2", "control",
             "Investor identity, kept for identity shrinkage later; not a predictor of the B-family heads.")
    _declare("hist_questions_all", "exposure", "questions in the history", "o2", "exposure",
             "Question turns by investors before the cut.")
    _declare("n_social_exchanges", "exposure", "social exchanges", "o2", "exposure",
             "Social exchanges with an eligible turn before the cut.")

    # ---- the focal investor's own questions and the answers to them
    _declare("own_questions", "own", "own questions", "o2", "exposure", "Questions the focal investor asked.")
    _stat_declared("own_sev_mean", "own", "severity of own questions", "Mean severity of the focal investor's questions.")
    _stat_declared("own_prevention_share", "own", "prevention share of own questions",
                   "Share of the focal investor's questions framed as prevention, among those framed as promotion "
                   "or prevention.")
    _declare_answers("own", "own")
    _stat_declared("own_frame_match_share", "own", "frame match of own answers",
                   "Share of answers to the focal investor's questions whose frame matches the question's, among "
                   "answers where both frames are promotion or prevention.")
    _declare("own_exchanges", "own", "own substantive exchanges", "o2", "exposure",
             "Substantive exchanges in which the focal investor has an eligible turn.")
    _declare_counts("own", (
        ("own_satisfied", "own exchanges left satisfied",
         "Of own_exchanges, those the investor left satisfied, shown by an eligible turn before the cut."),
        ("own_unsatisfied", "own exchanges left unsatisfied",
         "Of own_exchanges, those the investor left unsatisfied, shown by an eligible turn before the cut."),
        ("own_no_signal", "own exchanges without a closing signal",
         "Of own_exchanges, those with no closing state shown by an eligible turn before the cut."),
        ("own_challenges", "own challenges", "Challenge turns by the focal investor."),
        ("own_agrees", "own accepts turns", "Accepts turns by the focal investor.")))
    for suffix, _, label, does in pf.NEXT_MOVE_STATS:
        _stat_declared(f"own_{suffix}", "own", f"{label}, own questions",
                       f"Share of answers {_WHOSE['own']} after which, as the first investor turn of the same "
                       f"exchange and before the cut, the questioner {does}.")
    _declare("own_escalations", "own", "own follow-up questions that press harder", "o2", "concept",
             "Follow-up questions by the focal investor, each the first investor turn after an answer to one of "
             "their questions, whose severity is higher than that of the question answered.")
    _declare_hard("own", "own")
    # ---- what the rest of the panel asked and was told
    _declare("oth_questions", "panel_content", "others' questions", "o2", "exposure",
             "Questions the other investors asked.")
    _declare("oth_questioners", "panel_content", "other questioners", "o2", "exposure",
             "Distinct other investors who asked a question.")
    _declare_answers("oth", "panel_content")
    _stat_declared("oth_prevention_share", "panel_content", "prevention share of others' questions",
                   "Share of the other investors' questions framed as prevention, among promotion or prevention.")
    _declare_hard("oth", "panel_content")
    # ---- how the rest of the panel reacted
    _declare_counts("panel_reactions", (
        ("oth_agrees", "others' accepts turns", "Accepts turns by other investors."),
        ("oth_challenges", "others' challenges", "Challenge turns by other investors."),
        ("oth_satisfied", "others left satisfied",
         "Closing states of other investors in substantive exchanges that are satisfied, shown before the cut."),
        ("oth_unsatisfied", "others left unsatisfied",
         "Closing states of other investors in substantive exchanges that are unsatisfied, shown before the cut.")))
    # ---- what the other investors had already decided (the order of decisions, not what was said)
    _declare_counts("decision_history", (
        ("oth_exits_before", "others' earlier exits", "Exit decisions of other investors before the cut."),
        ("oth_offers_before", "others' earlier offers",
         "Offers made or joined by other investors before the cut.")))
    # ---- rapport and repair around the focal investor (hostile_at_founder and founder_repairs are shared)
    _declare_counts("rapport_repair", (
        ("own_social", "own social turns", "Social turns by the focal investor."),
        ("founder_social_to_own", "founder social turns to the investor",
         "Founder social turns addressed to the focal investor or responding to one of that investor's turns."),
        ("own_hostile", "own hostile turns", "Hostile turns by the focal investor.")))
    _declare_reactions()


def _rx_kind(suffix: str) -> str:
    return suffix.removesuffix("_share")


def _declare_reactions() -> None:
    family = "reaction_events"
    _declare("rx_own_answers", family, "answers to own questions (reaction windows)", "o2", "exposure",
             "Answers to the focal investor's own questions, answer and question before the cut.")
    for suffix, _, label in RX_SHARES:
        _stat_declared(f"rx_own_{suffix}", family, f"own reaction to the answer: {label} (share)",
                       f"Share of the answers to the focal investor's own questions to which that investor's first "
                       f"reaction in the answer's window, before the cut, is {label}; an answer the investor did "
                       "not react to (or not visibly before the cut) stays in the denominator.")
    _stat_declared("rx_own_followup_sev_mean", family, "own follow-up questions: severity (mean)",
                   "Mean severity of the focal investor's follow-up questions that were their first reaction to an "
                   "answer to their own question.")
    for suffix, _, label in RX_SHARES:
        kind = _rx_kind(suffix)
        _stat_declared(f"rx_own_last_{kind}", family, f"own reaction to the last answer: {label}",
                       f"1 when the focal investor's first reaction to the last answer to their own questions before "
                       f"the cut is {label}, else 0: where the investor stood when deciding. {_RX_COURSE}")
        _declare(f"rx_own_trend_{kind}", family, f"own reactions: {label}, later minus earlier", "o2", "concept",
                 f"Share of the later half of the answers to the focal investor's own questions whose first own "
                 f"reaction is {label}, minus that share in the earlier half (0 with fewer than two answers).")
    _stat_declared("rx_own_other_first_share", family, "answers to own questions where another investor reacted first",
                   "Share of the answers to the focal investor's own questions in whose window another "
                   "investor reacted before the focal investor did (before the cut).")
    whose = "investors other than the focal one (the focal investor's own reactions never count here)"
    _declare("rx_pan_answers", family, "answers (reaction windows)", "o2", "exposure",
             "Answers to an investor question, answer and question before the cut, whoever asked.")
    _stat_declared("rx_pan_reaction_rate", family, "reactions per answer",
                   f"Mean number of reactions per answer by {whose}, in the answer's window and before the cut.")
    for suffix, _, label in RX_SHARES:
        _stat_declared(f"rx_pan_{suffix}", family, f"answers drawing {label} (share)",
                       f"Share of the answers with at least one reaction that is {label}, by {whose}.")
    for suffix, _, label in RX_SHARES:
        kind = _rx_kind(suffix)
        _stat_declared(f"rx_pan_last_{kind}", family, f"last answer drew {label}",
                       f"1 when the last answer before the cut drew at least one reaction that is {label}, by "
                       f"{whose}, else 0. {_RX_COURSE}")
        _declare(f"rx_pan_trend_{kind}", family, f"answers drawing {label}, later minus earlier", "o2", "concept",
                 f"Share of the later half of the answers drawing at least one reaction that is {label} by {whose}, "
                 "minus that share in the earlier half (0 with fewer than two answers).")
    _declare("rx_pan_reactors", family, "investors reacting", "o2", "exposure",
             f"Distinct {whose} who reacted to at least one answer.")


_declare_all()


class _Combined(Mapping):
    """The live FEATURES and ARCHIVE_FEATURES as one read-only mapping, live first; both stay live, so a
    column declared later (a Stage A item of another instrument) shows up here too."""

    def __getitem__(self, name: str) -> dict:
        return pf.FEATURES[name] if name in pf.FEATURES else ARCHIVE_FEATURES[name]

    def __iter__(self) -> Iterator[str]:
        yield from pf.FEATURES
        yield from (name for name in ARCHIVE_FEATURES if name not in pf.FEATURES)

    def __len__(self) -> int:
        return len(pf.FEATURES) + sum(1 for name in ARCHIVE_FEATURES if name not in pf.FEATURES)


# read by analysis.archive.shark_level_models (the legacy notebook figures that also read it were removed in plan B task 9)
COMBINED_FEATURES = _Combined()
# the pitch families and the shark-level ones, interleaved so a later family is always a later rung of either ladder
COMBINED_FAMILIES = ("control", "exposure", "stage_a", "prepared", "own", "questions", "answers", "panel_content",
                     "reactions", "panel_reactions", "rapport_repair", "reaction_events", "decision_history")
assert set(COMBINED_FAMILIES) == set(pf.FAMILIES) | set(SHARK_FAMILIES)


def combined_dictionary() -> pd.DataFrame:
    """`analysis.panel_features.feature_dictionary` over COMBINED_FEATURES: the pitch columns and the shark-level
    ones, for the report and the export that still show investor rows. A shark-level column has its shark-level
    status (`analysis.archive.shark_level_models.feature_status`), a pitch column its pitch-level one."""
    from analysis.archive.shark_level_models import shark_level_rows     # deferred: that module imports this one

    with shark_level_rows():
        return _dictionary(COMBINED_FEATURES)


# ---- the rows ---------------------------------------------------------------------------------

def _question_stats(out: dict, prefix: str, questions: list[dict], severity: bool) -> None:
    if severity:
        _stat(out, f"{prefix}_sev_mean", [q["PR-SEV"] for q in questions])
    _stat(out, f"{prefix}_prevention_share",
          [float(q["frame"] == "prevention") for q in questions if q["frame"] in ("promotion", "prevention")])


def _hard_stats(out: dict, prefix: str, questions: list[dict], answers: list[dict]) -> None:
    hard = _hard(questions)
    to_hard = _answers_to(answers, hard)
    out[f"{prefix}_hard_questions"] = len(hard)
    _stat(out, f"{prefix}_subs_hard_mean", [a["PREP-SUBS"] for a in to_hard])
    _stat(out, f"{prefix}_evid_hard_share", [a["RS-EVID"] for a in to_hard])


def _shown_before_cut(closing_pos: float, eligible_pos: set[int]) -> bool:
    """Whether the turn showing a closing state is part of the row's history."""
    return closing_pos == closing_pos and int(closing_pos) in eligible_pos


def _closing_states(closings: list[dict], eligible_pos: set[int]) -> dict[tuple[int, str], str]:
    """(substantive exchange, investor) -> satisfied | unsatisfied | no_signal, as the row may know it."""
    states: dict[tuple[int, str], str] = {}
    for c in closings:
        shown = c["closing"] in SHOWN_CLOSINGS and _shown_before_cut(c["closing_pos"], eligible_pos)
        states.setdefault((c["exchange_id"], c["actor"]), c["closing"] if shown else "no_signal")
    return states


def _substantive_pairs(history: list[dict], investors: set[str]) -> set[tuple[int, str]]:
    """(exchange, investor) for every substantive exchange in which that investor has an eligible turn."""
    return {(t["exchange_id"], t["speaker"]) for t in history
            if t["exchange_kind"] == SUBSTANTIVE and t["speaker"] in investors}


def _rx_own_stats(out: dict, windows: list[dict], focal: str) -> None:
    own = [w for w in windows if w["asker"] == focal]
    firsts = [next((r for r in w["reactions"] if r["actor"] == focal), None) for w in own]
    out["rx_own_answers"] = len(own)
    for suffix, kind, _ in RX_SHARES:
        _stat(out, f"rx_own_{suffix}", [float(f is not None and f["kind"] == kind) for f in firsts])
    _stat(out, "rx_own_followup_sev_mean", [f["severity"] for f in firsts if f is not None and f["kind"] == "follow_up"])
    for suffix, kind, _ in RX_SHARES:
        drew = [float(f is not None and f["kind"] == kind) for f in firsts]
        _stat(out, f"rx_own_last_{_rx_kind(suffix)}", drew[-1:])
        out[f"rx_own_trend_{_rx_kind(suffix)}"] = _trend(drew)
    _stat(out, "rx_own_other_first_share",
          [float(bool(w["reactions"]) and w["reactions"][0]["actor"] != focal) for w in own])


def _rx_panel_stats(out: dict, prefix: str, windows: list[dict], reactors: set[str]) -> None:
    """Reactions by `reactors` to every answer, an answer of two askers counted once."""
    per_answer = _per_answer(windows, reactors)
    out[f"rx_{prefix}_answers"] = len(per_answer)
    _stat(out, f"rx_{prefix}_reaction_rate", [len(rs) for rs in per_answer])
    for suffix, kind, _ in RX_SHARES:
        drew = [float(any(r["kind"] == kind for r in rs)) for rs in per_answer]
        _stat(out, f"rx_{prefix}_{suffix}", drew)
        _stat(out, f"rx_{prefix}_last_{_rx_kind(suffix)}", drew[-1:])
        out[f"rx_{prefix}_trend_{_rx_kind(suffix)}"] = _trend(drew)
    out[f"rx_{prefix}_reactors"] = len({r["actor"] for rs in per_answer for r in rs})


def _decisions_before(decisions: list[dict], cut_pos: int, focal: str | None) -> list[dict]:
    """Investor decisions a row may read: strictly before its cut, never the focal investor's own."""
    return [d for d in decisions if d["pos"] < cut_pos and d["actor"] != focal]


class _Pitch(pf._Pitch):
    """One video's rows of the tables, with the investor decisions an investor's row may count."""

    def __init__(self, data: PanelData, video_id: str):
        super().__init__(data, video_id)
        # how an investor left an exchange: no record holds it since v3-dev9, so no row reads a closing state
        self.closings: list[dict] = []
        decisions = data.decisions[(data.decisions.video_id == video_id) & (data.decisions.role == "investor")]
        self.decisions = decisions.to_dict("records")


def _shared(pitch: _Pitch, history: list[dict], with_ranks: bool) -> dict:
    """Control, exposure and Stage A columns of an investor row (the exposure count of questions under its
    shark-level name, hist_questions_all)."""
    out = {name: pitch.video[name] for name in CONTROLS}
    out.update({("hist_questions_all" if k == "hist_questions" else k): v for k, v in _exposure(pitch, history).items()})
    out["n_social_exchanges"] = len({t["exchange_id"] for t in history if t["exchange_kind"] == SOCIAL_EXCHANGE})
    out.update(_stage_a(pitch, with_ranks))
    return out


def _guarded(history: list[dict], speakers: set[str], guard_turns: int) -> list[dict]:
    """`history` without the last `guard_turns` turns spoken by `speakers`."""
    if guard_turns <= 0:
        return history
    last = [t["pos"] for t in history if t["speaker"] in speakers][-guard_turns:]
    return [t for t in history if t["pos"] not in set(last)]


def _guard(guard_turns: int) -> int:
    if int(guard_turns) != guard_turns or guard_turns < 0:
        raise ValueError(f"guard_turns must be a whole number of turns, 0 or more, not {guard_turns!r}")
    return int(guard_turns)


def _o2_row(data: PanelData, pitch: _Pitch, focal: str, cut_pos: int, with_ranks: bool,
            guard_turns: int = 0) -> dict:
    history = eligible_turns_for_investor(data, pitch.video["video_id"], cut_pos, focal).to_dict("records")
    history = _guarded(history, {focal}, guard_turns)
    eligible_pos = {t["pos"] for t in history}
    others = set(pitch.investors) - {focal}
    out = _shared(pitch, history, with_ranks)

    questions = [t for t in history if t["function"] == "question"]
    answers = [t for t in history if t["function"] == "answer"]
    own_q = [q for q in questions if q["speaker"] == focal]
    oth_q = [q for q in questions if q["speaker"] in others]
    states = _closing_states(pitch.closings, eligible_pos)

    out["own_questions"] = len(own_q)
    _question_stats(out, "own", own_q, severity=True)
    _answer_stats(out, "own", _answers_to(answers, own_q), frame_match=True)
    own_pairs = _substantive_pairs(history, {focal})
    own_states = [states.get(pair, "no_signal") for pair in own_pairs]
    out["own_exchanges"] = len(own_pairs)
    for state in ("satisfied", "unsatisfied", "no_signal"):
        out[f"own_{state}"] = own_states.count(state)
    out["own_challenges"] = _count(history, "challenges", speakers={focal})
    out["own_agrees"] = _count(history, *RECEIPT_FUNCTIONS, speakers={focal})
    followed = _followups(data, pitch.video["video_id"], {focal: cut_pos}, history, {focal})
    _next_move_stats(out, "own", followed)
    out["own_escalations"] = sum(1 for r in followed if r["move"] == FOLLOW_UP and r["escalation"] > 0)
    _hard_stats(out, "own", own_q, answers)

    out["oth_questions"] = len(oth_q)
    out["oth_questioners"] = len({q["speaker"] for q in oth_q})
    _answer_stats(out, "oth", _answers_to(answers, oth_q), frame_match=False)
    _question_stats(out, "oth", oth_q, severity=False)
    _hard_stats(out, "oth", oth_q, answers)

    out["oth_agrees"] = _count(history, *RECEIPT_FUNCTIONS, speakers=others)
    out["oth_challenges"] = _count(history, "challenges", speakers=others)
    for state in SHOWN_CLOSINGS:
        out[f"oth_{state}"] = sum(1 for (_, actor), s in states.items() if actor in others and s == state)
    earlier = _decisions_before(pitch.decisions, cut_pos, focal)
    out["oth_exits_before"] = sum(1 for d in earlier if d["kind"] == "exit")
    out["oth_offers_before"] = sum(1 for d in earlier if makes_offer(d["kind"], d["offer_id"]))

    focal_turns = {t["turn_id"] for t in history if t["speaker"] == focal}
    out["own_social"] = _count(history, *SOCIAL_FUNCTIONS, speakers={focal})
    out["founder_social_to_own"] = sum(
        1 for t in history if t["function"] in SOCIAL_FUNCTIONS and t["speaker"] in pitch.founders
        and (focal in t["addressed"] or focal_turns.intersection(t["responds_to"])))
    out["own_hostile"] = _count(history, "hostile", speakers={focal})
    out["hostile_at_founder"] = _count(history, "hostile", speakers=set(pitch.investors))
    out["founder_repairs"] = _count(history, *REPAIR_FUNCTIONS, speakers=pitch.founders)

    windows = _rx_windows(pitch, data, history)
    _rx_own_stats(out, windows, focal)
    _rx_panel_stats(out, "pan", windows, others)
    return out


def o2_features(data: PanelData, guard_turns: int = 0) -> pd.DataFrame:
    """One row per pitch x investor, whether or not that investor's outcome is known.

    Columns: keys (video_id, investor, source, cut_pos), the outcome columns of `investor_outcomes`, then
    every shark-level feature. `guard_turns = k > 0` additionally drops the focal investor's own last k
    eligible turns before the cut (a sensitivity table, not the default).
    """
    guard_turns = _guard(guard_turns)
    declare_stage_a(data.stage_a_codes)
    table = investor_outcomes(data).merge(investor_cuts(data), on=["video_id", "investor"], validate="one_to_one")
    with_ranks = _with_ranks(data)
    pitches: dict[str, _Pitch] = {}
    rows = []
    for r in table.itertuples():
        pitch = pitches.setdefault(r.video_id, _Pitch(data, r.video_id))
        rows.append({
            "video_id": r.video_id, "investor": r.investor, "source": pitch.video["source"],
            "cut_pos": int(r.cut_pos), "ever_offer": r.ever_offer, "first_decision_kind": r.first_decision_kind,
            "first_decision_pos": r.first_decision_pos, "investor_label": r.label,
            **_o2_row(data, pitch, r.investor, int(r.cut_pos), with_ranks, guard_turns)})
    return pd.DataFrame(rows)


# ---- the same rows, made inspectable -------------------------------------------------------

REASON_GUARD = "guard band: one of the last turns this row's speaker(s) took before the cut"


def history_table(data: PanelData, video_id: str, investor: str | None = None, guard_turns: int = 0) -> pd.DataFrame:
    """Every turn of a video with whether a row's features read it, and why not when they do not.

    `investor` names the focal investor of an O2 row; None is the pitch row (with the guard band, if any,
    over the turns of every investor). Columns: the turn columns of the data, then `cut_pos`, `used` and
    `why_excluded` (empty when used). The used turns of an investor are exactly the history `o2_features`
    computes that row from, guard band included.
    """
    guard_turns = _guard(guard_turns)
    o2_cut, o1_cut = investor_cuts(data), pitch_cut(data)
    if investor is None:
        found = o1_cut[o1_cut.video_id == video_id]
    else:
        found = o2_cut[(o2_cut.video_id == video_id) & (o2_cut.investor == investor)]
    if len(found) != 1:
        investors = sorted(o2_cut.investor[o2_cut.video_id == video_id])
        raise ValueError(f"no row for video {video_id!r}" + (f", investor {investor!r}" if investor else "")
                         + f". Videos: {', '.join(sorted(o1_cut.video_id))}."
                         + (f" Investors of this video: {', '.join(investors)}." if investors else ""))
    cut_pos = int(found.cut_pos.iloc[0])
    table = data.turns[data.turns.video_id == video_id].copy()
    reason = (exclusion_reasons(data, video_id, cut_pos) if investor is None
              else exclusion_reasons_for_investor(data, video_id, cut_pos, investor))
    history = table[reason == ""].to_dict("records")
    speakers = {investor} if investor is not None else set(pf._Pitch(data, video_id).investors)
    kept = {t["pos"] for t in _guarded(history, speakers, guard_turns)}
    reason[(reason == "") & ~table.pos.isin(kept)] = REASON_GUARD
    table["cut_pos"] = cut_pos
    table["used"] = reason == ""
    table["why_excluded"] = reason
    return table


def describe_row(features: pd.DataFrame, video_id: str, investor: str | None = None) -> pd.DataFrame:
    """One row of a shark-level (or pitch) feature table read downwards: feature, family, label, value,
    denominator, description, over the columns of COMBINED_FEATURES."""
    keep = features.video_id == video_id
    if investor is not None and "investor" in features.columns:
        keep &= features.investor == investor
    found = features[keep]
    if len(found) != 1:
        raise ValueError(f"expected one row for video {video_id!r}"
                         + (f", investor {investor!r}" if investor else "") + f", found {len(found)}")
    return _describe(found.iloc[0], features.columns, COMBINED_FEATURES)
