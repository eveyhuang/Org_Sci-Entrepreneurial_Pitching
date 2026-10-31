"""Named concept features of a pitch: one row per pitch, from the turns before the pitch's first decision.

Every feature of a row is computed from `eligible_turns` of the pitch (strictly before the cut, no
`decides` or `negotiates` turn, no negotiation exchange). The cut is the first decision of any kind by
any investor (`pitch_cut`), so a row reads no decision at all and its history does not depend on the
outcome. Beyond turns a row reads only facts fixed before the questioning starts: the venture controls
and the ask, the Stage A ratings and rank scores of the prepared presentation, and its prepared ratings
(PRES-*).

The columns come in the families of the model ladder (FAMILIES): control, exposure, stage_a, prepared,
questions, answers, reactions, rapport_repair. FEATURES declares every column a pitch row carries.

Missing is not zero: every mean or share comes with `<name>_n`, the number of observations behind it,
and `<name>_missing`, which is 1 when there are none and the value is NaN.

What a questioner does after an answer (`nm_*`, the next move) comes from
`analysis.panel_dataset.answer_followups`, asked for with the pitch's cut, so a next move at or after
the cut is never read. An answer whose next move the row cannot see stays in the denominator: the
share is "followed, before the cut, by ...". The reaction shares (`rx_*`) read every reaction in an
answer's window, not only the first (`analysis.reaction_events`), and a reaction counts only when its
turn is one of the pitch's eligible turns. A *hard* question has severity 2 or 3.

The questions family describes the investor questions before the cut: how many investors asked, their mean
severity and share of hard questions, the shares by frame and by concept (deal terms, other), and four features
of their sequence in turn order (plan B2, 2026-10-21): `q_sev_trend` (mean severity of the later half minus the
earlier half), `q_pursuit_share` (consecutive questions by the same investor, the second more severe),
`q_frame_switch_share` (consecutive questions whose promotion/prevention frames differ) and `q_concept_entropy`
(the diversity of the concepts asked about, in bits). Nothing in the family reads the exchange grouping: a
sequence is the order of the questions, whichever exchange the model put them in. Exposure counts no exchange
either: `demo_seconds` sums the `demonstrates` turns.

Two options build the tables of sensitivities (spec section 6.4). `exposure="rates"` gives the counts of
RATE_COLUMNS per minute of the history (`hist_seconds / 60`), missing when the history has no seconds.
A PanelData read with both passes (`load_run(..., passes="both")`) gives every numeric feature as the
mean of the two passes' values, each pass cut at its own first decision.

The retired shark-level rows (one per pitch and investor) live in
`analysis.archive.shark_level_features`, which imports its shared helpers from here; this module never
imports the archive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.reaction_events import reaction_events, reaction_turns
from analysis.panel_dataset import (
    CHALLENGE, FOLLOW_UP, HARD_SEVERITY, POSITIVE_RECEIPT, RECEIPT_FUNCTIONS, PanelData, answer_followups,
    eligible_turns, exclusion_reasons, pitch_cut, pitch_outcome,
)
from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import OTHER, load_panel_books

FAMILIES = ("control", "exposure", "stage_a", "prepared", "questions", "answers", "reactions", "rapport_repair")
EXPOSURE_COUNTS, EXPOSURE_RATES = "counts", "rates"
EXPOSURES = (EXPOSURE_COUNTS, EXPOSURE_RATES)
# the counts that exposure="rates" divides by the minutes of the history (shares and means stay as they are)
RATE_COLUMNS = ("hist_turns", "hist_questions", "accepts_turns", "challenge_turns", "social_turns", "founder_social",
                "hostile_at_founder", "founder_repairs")
OUTCOME_COLUMNS = ("any_offer", "first_offer_pos", "any_offer_source")       # pitch_outcome's, on every pitch row
CONTROLS = ("founder_gender", "primary_industry", "product_form", "customer_type")
_BOOKS = load_panel_books(load_release())
STAGE_A_CODES = _BOOKS.stage_a_ids()
PREPARED_CODES = tuple(entry.id for entry in _BOOKS.prepared_codes())
PREPARED_PREFIX = "PRES-"
SUBSTANTIVE = "substantive"
DEMONSTRATES = "demonstrates"      # the turn function demo_seconds sums
SOCIAL_FUNCTIONS = ("social",)
REPAIR_FUNCTIONS = ("repair",)
DEAL_TERMS = "deal_terms"
# (feature suffix, rating code, what one observation is)
ANSWER_STATS = (
    ("resp_mean", "RS-RESP", "mean responsiveness of the answers (addresses 1, partial 0.5, deflects 0)"),
    ("evid_share", "RS-EVID", "share of the answers that give evidence"),
    ("uncert_share", "RS-UNCERT", "share of the answers that voice uncertainty"),
    ("subs_mean", "PREP-SUBS", "mean substance rating of the answers"),
    ("coh_mean", "PREP-COH", "mean coherence rating of the answers"),
    ("ctx_mean", "PREP-CTX", "mean context rating of the answers"),
)
ANSWER_LABELS = {"resp_mean": "answer directness (mean)", "evid_share": "answers with checkable support (share)",
                 "uncert_share": "answers admitting uncertainty (share)", "subs_mean": "answer substance (mean)",
                 "coh_mean": "answer coherence (mean)", "ctx_mean": "answer contextualisation (mean)"}
# (feature suffix, the next move it counts, label, what the questioner does)
NEXT_MOVE_STATS = (
    ("receipt_share", POSITIVE_RECEIPT, "answers the questioner then accepts (share)", "accepts"),
    ("followup_share", FOLLOW_UP, "answers the questioner follows with another question (share)", "asks again"),
    ("challenge_share", CHALLENGE, "answers the questioner then challenges (share)", "challenges"),
)
# (suffix, reaction kind, label)
RX_SHARES = (("receipt_share", "receipt", "receipt (accepts)"),
             ("followup_share", "follow_up", "another question"),
             ("challenge_share", "challenge", "challenge"))
# (suffix, frame, label) and (suffix, concept, label) of the question shares
QUESTION_FRAMES = (("frame_promotion_share", "promotion", "promotion-framed"),
                   ("frame_prevention_share", "prevention", "prevention-framed"))
QUESTION_CONCEPTS = (("deal_terms_share", DEAL_TERMS, "about the deal terms"),
                     ("other_share", OTHER, "about a matter outside the sixteen concepts (other)"))
# the frames a frame switch compares (unclassified: deal_terms, other; NEUTRAL: no concept) and the concept that
# carries none, left out of the concept entropy
SWITCH_FRAMES = ("promotion", "prevention")
NO_CONCEPT = "NONE"
# features whose value is a placeholder, not an observation, when their `_missing` is 1 (q_sev_trend is 0.0 with no
# rated question): averaging two passes reads such a value as missing
PLACEHOLDER_FEATURES = ("q_sev_trend",)
_OF_ALL_QUESTIONS = ("The denominator is every investor question before the cut, whatever its concept or frame, so "
                     "the four question shares need not sum to 1 (a question with no concept counts only in the "
                     "denominator).")

FEATURES: dict[str, dict] = {}


def _declare(name: str, family: str | None, label: str, unit: str, kind: str, description: str) -> None:
    FEATURES[name] = {"family": family, "label": label, "unit": unit, "kind": kind, "description": description}


def _declare_stat(name: str, family: str, label: str, unit: str, description: str,
                  registry: dict | None = None) -> None:
    """A mean or share, with its `_n` (observations) and `_missing` (none observed) companions, in `registry`
    (FEATURES by default)."""
    registry = FEATURES if registry is None else registry
    for column, column_label, kind, text in (
            (name, label, "concept", description),
            (f"{name}_n", f"{label}: observations", "exposure", f"Number of rated observations behind {name}."),
            (f"{name}_missing", f"{label}: none observed", "indicator",
             f"1 when {name} has no observation and is missing, else 0.")):
        registry[column] = {"family": family, "label": column_label, "unit": unit, "kind": kind, "description": text}


def _declare_all() -> None:
    _declare("video_id", None, "pitch", "both", "key", "Video id of the pitch; the unit all rows of a pitch share.")
    _declare("source", None, "record source", "both", "key",
             "reviewed when the features come from the human-reviewed record, primary when model-coded.")
    _declare("cut_pos", None, "history cut", "both", "key",
             "Turn position at which this row's history stops; never a predictor.")

    # ---- control: the venture and its ask, fixed before the questioning starts
    for name, label in zip(CONTROLS, ("founder gender", "primary industry", "product form", "customer type")):
        _declare(name, "control", label, "both", "control", f"Venture control: {label} (categorical).")
    # numeric controls are declared with kind `exposure`: the heads read that kind as a number
    _declare("ask_log_amount", "control", "ask amount (log10 USD)", "o1", "exposure",
             "Base-10 logarithm of the amount in US dollars the founders asked for; missing when the ask is not "
             "recorded or is 0.")
    _declare("ask_equity_pct", "control", "ask equity (%)", "o1", "exposure",
             "Equity in percent the founders offered for the amount asked; missing when the ask is not recorded.")
    _declare("ask_missing", "control", "ask not recorded", "o1", "indicator",
             "1 when the record gives no ask (the venture's ask_missing is not 'none'), else 0.")

    # ---- exposure: how much was said before the cut
    for name, label, description in (
            ("hist_turns", "turns in the history", "Eligible turns before the cut."),
            ("hist_seconds", "seconds in the history", "Summed duration of the eligible turns before the cut."),
            ("hist_questions", "questions in the history",
             "Question turns by investors before the cut (the questions family adds no second count of them)."),
            ("demo_seconds", "demonstration seconds",
             "Summed duration of the eligible `demonstrates` turns before the cut, whoever speaks them (read from "
             "the turn function, not the exchange kind)."),
            ("prepared_seconds", "prepared presentation seconds", "Length of the prepared presentation.")):
        # the investor rows of the archive carry the same exposure, with their own name for the question count
        _declare(name, "exposure", label, "o1" if name == "hist_questions" else "both", "exposure", description)

    declare_stage_a(STAGE_A_CODES)
    declare_prepared(PREPARED_CODES)

    # ---- questions: what the investors asked before the cut
    _declare("q_askers", "questions", "questioners", "o1", "exposure", "Distinct investors who asked a question.")
    _declare_stat("q_sev_mean", "questions", "severity of questions", "o1", "Mean severity of investor questions.")
    _declare_stat("q_hard_share", "questions", "hard questions (share)", "o1",
                  "Share of the investor questions with a severity rating that ask for substantiation or contest a "
                  "claim (severity 2 or 3); a question without a severity rating is left out of the denominator.")
    for suffix, frame, label in QUESTION_FRAMES:
        _declare_stat(f"q_{suffix}", "questions", f"{label} questions (share)", "o1",
                      f"Share of investor questions whose concept is {frame}-framed. {_OF_ALL_QUESTIONS}")
    for suffix, concept, label in QUESTION_CONCEPTS:
        _declare_stat(f"q_{suffix}", "questions", f"questions {label} (share)", "o1",
                      f"Share of investor questions whose concept is {concept}. {_OF_ALL_QUESTIONS}")
    # the sequence of the questions, in turn order (plan B2)
    _declare_stat("q_sev_trend", "questions", "severity trend of questions", "o1",
                  "Mean severity of the later half of the investor questions before the cut, in turn order, minus the "
                  "mean of the earlier half (with an odd number the middle question is in neither half); questions "
                  "without a severity rating are left out. 0.0 with fewer than two rated questions; `_n` counts the "
                  "rated questions and `_missing` is 1 when there is none (the value is then 0.0, not missing).")
    _declare_stat("q_pursuit_share", "questions", "same-asker pursuit (share)", "o1",
                  "Share of consecutive pairs of investor questions before the cut, in turn order, in which the same "
                  "investor asks both and the second has a strictly higher severity; pairs where either question has "
                  "no severity rating are left out. Missing with no such pair (one question or none).")
    _declare_stat("q_frame_switch_share", "questions", "frame switches (share)", "o1",
                  "Share of consecutive pairs of investor questions before the cut, in turn order, whose frames "
                  "differ, among the pairs where both frames are promotion or prevention (a question about deal "
                  "terms, about another matter or with no concept is never counted as a switch, and its pairs are "
                  "left out). Missing with no such pair.")
    _declare_stat("q_concept_entropy", "questions", "concept diversity (entropy, bits)", "o1",
                  "Shannon entropy, in bits, of the concepts of the investor questions before the cut, over the "
                  "questions with a concept (deal_terms and other count as concepts; a question with no concept, "
                  "NONE, is left out). 0.0 when every such question has the same concept; missing with none.")
    # what the companions of the sequence features count (the generic text says "rated observations")
    for name, text in (
            ("q_sev_trend_n", "Number of investor questions with a severity rating behind q_sev_trend."),
            ("q_sev_trend_missing", "1 when no investor question before the cut has a severity rating (q_sev_trend "
                                    "is then 0.0, not missing), else 0."),
            ("q_pursuit_share_n", "Number of consecutive question pairs, both with a severity rating, behind "
                                  "q_pursuit_share."),
            ("q_frame_switch_share_n", "Number of consecutive question pairs, both promotion or prevention, behind "
                                       "q_frame_switch_share."),
            ("q_concept_entropy_n", "Number of investor questions with a concept behind q_concept_entropy.")):
        FEATURES[name]["description"] = text

    # ---- answers: how the founders answered before the cut
    for suffix, _, description in ANSWER_STATS:
        _declare_stat(f"a_{suffix}", "answers", ANSWER_LABELS[suffix], "o1",
                      f"{description[0].upper()}{description[1:]}, before the cut.")
    _declare_stat("a_subs_hard_mean", "answers", "answer substance (mean), hard questions", "o1",
                  "Mean substance rating of the answers to investor hard questions (severity 2 or 3), before the cut.")
    _declare_stat("a_evid_hard_share", "answers", "answers with checkable support (share), hard questions", "o1",
                  "Share of the answers to investor hard questions (severity 2 or 3) that give evidence, before the "
                  "cut.")
    _declare_stat("a_frame_match_share", "answers", "frame match of answers", "o1",
                  "Share of answers whose frame matches the question's, among answers where both frames are "
                  "promotion or prevention.")

    # ---- reactions: what the investors did with the answers
    _declare("rx_answers", "reactions", "answers (reaction windows)", "o1", "exposure",
             "Answers to an investor question, answer and question before the cut, whoever asked.")
    _declare_stat("rx_reaction_rate", "reactions", "reactions per answer", "o1",
                  "Mean number of reactions per answer by any investor, in the answer's window and before the cut.")
    for suffix, _, label in RX_SHARES:
        _declare_stat(f"rx_{suffix}", "reactions", f"answers drawing {label} (share)", "o1",
                      f"Share of the answers with at least one reaction that is {label}, by any investor, in the "
                      "answer's window and before the cut. Each kind is its own column.")
    for suffix, _, label, does in NEXT_MOVE_STATS:
        _declare_stat(f"nm_{suffix}", "reactions", label, "o1",
                      f"Share of answers to any question after which, as the next investor turn in turn order and "
                      f"before the cut, the questioner {does}.")
    _declare("accepts_turns", "reactions", "accepts turns", "o1", "concept", "Accepts turns by investors.")
    _declare("challenge_turns", "reactions", "challenges", "o1", "concept", "Challenge turns by investors.")

    # ---- rapport and repair
    for name, label, description in (
            ("social_turns", "investor social turns", "Social turns by investors."),
            ("founder_social", "founder social turns", "Founder social turns."),
            ("hostile_at_founder", "hostile turns", "Hostile turns by any investor."),
            ("founder_repairs", "founder repairs", "Founder repair turns.")):
        _declare(name, "rapport_repair", label, "both" if name in ("hostile_at_founder", "founder_repairs") else "o1",
                 "concept", description)


def declare_stage_a(codes) -> None:
    """Declare the Stage A columns of these items; a run annotated with another instrument may bring items
    the live release does not have."""
    for code in codes:
        if f"sa_{code}" in FEATURES:
            continue
        _declare(f"sa_{code}", "stage_a", f"Stage A {code}", "both", "concept",
                 f"Stage A rating of the prepared presentation on {code} (1-5); missing when uncodable.")
        _declare(f"sa_{code}_missing", "stage_a", f"Stage A {code}: not rated", "both", "indicator",
                 f"1 when sa_{code} is missing, else 0.")
    for code in codes:
        if f"sa_rank_{code}" not in FEATURES:
            _declare(f"sa_rank_{code}", "stage_a", f"Stage A rank score {code}", "both", "concept",
                     f"Plackett-Luce score of the pitch on {code} from the batch ranking; present only when the "
                     "run has rank scores.")


def prepared_column(code: str) -> str:
    """`pres_<suffix>` of a prepared code: PRES-SUBS -> pres_subs."""
    return "pres_" + code.removeprefix(PREPARED_PREFIX).lower()


def declare_prepared(codes) -> None:
    """Declare the prepared-presentation columns of these codes (PanelBooks.prepared_codes)."""
    for code in codes:
        name = prepared_column(code)
        if name in FEATURES:
            continue
        _declare(name, "prepared", f"prepared presentation {code}", "o1", "concept",
                 f"The prepared presentation's {code} rating, rated once per pitch; missing when not rated.")
        _declare(f"{name}_missing", "prepared", f"prepared presentation {code}: not rated", "o1", "indicator",
                 f"1 when {name} is missing, else 0.")


_declare_all()


def _check_exposure(exposure: str) -> None:
    if exposure not in EXPOSURES:
        raise ValueError(f"exposure takes {' or '.join(EXPOSURES)}, not {exposure!r}")


def _registry(exposure: str = EXPOSURE_COUNTS) -> dict[str, dict]:
    """FEATURES as the table of `exposure` reads: with rates, the columns of RATE_COLUMNS are labelled and
    described per minute."""
    _check_exposure(exposure)
    if exposure == EXPOSURE_COUNTS:
        return FEATURES
    out = dict(FEATURES)
    for name in RATE_COLUMNS:
        spec = FEATURES[name]
        out[name] = {**spec, "label": f"{spec['label']} per minute",
                     "description": f"{spec['description'].rstrip('.')}, per minute of the history (divided by "
                                    "hist_seconds / 60); missing when the history has no seconds."}
    return out


def feature_dictionary(exposure: str = EXPOSURE_COUNTS) -> pd.DataFrame:
    """One row per column a pitch feature table can carry, outcomes excluded; with `exposure="rates"` the
    columns of RATE_COLUMNS are labelled and described per minute.

    `status` (analysis.panel_bottleneck.feature_status) is `primary` for a column the pre-registered ladder fits,
    `descriptive` for a column it never fits (LADDER_NOT_FITTED: the `_n` counts behind a mean or share, the H5
    frame-match share, the exposure columns beyond the spec's C1 list and `rx_answers`), `exploratory` for the rest
    (the other Stage A items, read by the all-Stage-A sensitivity only); a key column has none.
    """
    return _dictionary(_registry(exposure))


def _dictionary(registry) -> pd.DataFrame:
    """The dictionary table of the columns `registry` declares (the archive passes its own registry)."""
    from analysis.panel_bottleneck import feature_status      # deferred: that module imports this one

    table = pd.DataFrame([{"name": name, **{k: spec[k] for k in ("family", "label", "unit", "kind", "description")}}
                          for name, spec in registry.items()],
                         columns=["name", "family", "label", "unit", "kind", "status", "description"])
    table["status"] = pd.Series([feature_status(name) for name in table.name], dtype=object)
    return table


# ---- building blocks (the archived shark-level rows use them too) ----------------------------

def _stat(out: dict, name: str, values) -> None:
    observed = [float(v) for v in values if v == v and v is not None]
    out[name] = float(np.mean(observed)) if observed else np.nan
    out[f"{name}_n"] = len(observed)
    out[f"{name}_missing"] = int(not observed)


def _count(turns: list[dict], *functions: str, speakers=None) -> int:
    return sum(1 for t in turns if t["function"] in functions and (speakers is None or t["speaker"] in speakers))


def _answer_stats(out: dict, prefix: str, answers: list[dict], frame_match: bool) -> None:
    for suffix, code, _ in ANSWER_STATS:
        _stat(out, f"{prefix}_{suffix}", [a[code] for a in answers])
    if frame_match:
        _stat(out, f"{prefix}_frame_match_share",
              [float(a["frame_match"] == "match") for a in answers
               if isinstance(a["frame_match"], str) and (a["frame_match"] == "match"
                                                         or a["frame_match"].startswith("switch_to_"))])


def _answers_to(answers: list[dict], questions: list[dict]) -> list[dict]:
    ids = {q["turn_id"] for q in questions}
    return [a for a in answers if ids.intersection(a["responds_to"])]


def _hard(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["PR-SEV"] >= HARD_SEVERITY]      # a missing severity is not hard


def _hard_answer_stats(out: dict, prefix: str, questions: list[dict], answers: list[dict]) -> None:
    to_hard = _answers_to(answers, _hard(questions))
    _stat(out, f"{prefix}_subs_hard_mean", [a["PREP-SUBS"] for a in to_hard])
    _stat(out, f"{prefix}_evid_hard_share", [a["RS-EVID"] for a in to_hard])


def _next_in_history(row: dict, positions: set[int]) -> bool:
    """Whether the turn that makes a next move is one of the row's eligible turns."""
    return row["next_pos"] == row["next_pos"] and int(row["next_pos"]) in positions


def _followups(data: PanelData, video_id: str, cut: dict, history: list[dict], askers: set[str]) -> list[dict]:
    """The (answer, question) pairs of `askers` a row may read, each with the next move as the row knows it.

    Answer and question must both be eligible turns of the row. The next move is asked for with the
    askers' cut, so one at or after it comes back as `after_cut`; one made by a turn that is not
    eligible for another reason is hidden the same way.
    """
    positions = {t["pos"] for t in history}
    rows = []
    for r in answer_followups(data, cut={video_id: cut}, video_ids=[video_id]).to_dict("records"):
        if r["asker"] not in askers or r["answer_pos"] not in positions or r["question_pos"] not in positions:
            continue
        known = _next_in_history(r, positions)
        rows.append({"move": r["next_move"] if known else None,
                     "escalation": r["escalation"] if known else np.nan})
    return rows


def _next_move_stats(out: dict, prefix: str, rows: list[dict]) -> None:
    for suffix, move, _, _ in NEXT_MOVE_STATS:
        _stat(out, f"{prefix}_{suffix}", [float(r["move"] == move) for r in rows])


def _trend(values: list[float]) -> float:
    """Mean of the later half minus mean of the earlier half (0 with fewer than two values)."""
    values = [v for v in values if v == v]
    if len(values) < 2:
        return 0.0
    half = len(values) // 2
    return float(np.mean(values[len(values) - half:]) - np.mean(values[:half]))


def _rx_windows(pitch, data: PanelData, history: list[dict]) -> list[dict]:
    """The reaction windows a row may read: (answer, asker) pairs whose answer and question are eligible,
    each with its reacting turns that are eligible, in turn order."""
    if pitch.reaction_turns is None:
        video_id = pitch.video["video_id"]
        pitch.reaction_turns = reaction_turns(data, video_ids=[video_id])
        events = reaction_events(data, video_ids=[video_id])
        in_video = data.turns[data.turns.video_id == video_id]
        pos_of = dict(zip(in_video.turn_id, in_video.pos))
        pitch.reaction_pairs = [{"answer_turn": e["answer_turn"], "answer_pos": e["answer_pos"],
                                 "question_pos": pos_of[e["question_turn"]], "asker": e["asker"]}
                                for e in events.to_dict("records")]
    positions = {t["pos"] for t in history}
    reacting = pitch.reaction_turns[pitch.reaction_turns.pos.isin(positions)]
    # an answer to two questions of one asker is one window: its reactions once each
    by_pair = {key: g.drop_duplicates("turn_id").sort_values("pos").to_dict("records")
               for key, g in reacting.groupby(["answer_turn", "asker"], sort=False)}
    windows: dict[tuple, dict] = {}
    for pair in pitch.reaction_pairs:
        key = (pair["answer_turn"], pair["asker"])
        if key not in windows and pair["answer_pos"] in positions and pair["question_pos"] in positions:
            windows[key] = {**pair, "reactions": by_pair.get(key, [])}
    return sorted(windows.values(), key=lambda w: w["answer_pos"])


def _per_answer(windows: list[dict], reactors: set[str]) -> list[list[dict]]:
    """The reactions by `reactors` to every answer, an answer of two askers counted once."""
    answers: dict[int, dict] = {}
    for w in windows:
        seen = answers.setdefault(w["answer_turn"], {})
        for r in w["reactions"]:
            if r["actor"] in reactors:
                seen[r["turn_id"]] = r
    return [list(seen.values()) for seen in answers.values()]


class _Pitch:
    """One video's rows of the tables, gathered once."""

    def __init__(self, data: PanelData, video_id: str):
        self.stage_a_codes = tuple(data.stage_a_codes) or STAGE_A_CODES
        self.video = data.videos[data.videos.video_id == video_id].iloc[0]
        actors = data.actors[data.actors.video_id == video_id]
        self.investors = list(actors.actor_id[actors.role == "investor"])
        self.founders = set(actors.actor_id[actors.role == "founder"])
        stage_a = data.stage_a[data.stage_a.video_id == video_id]
        self.ratings = dict(zip(stage_a.code[stage_a.source == "rating"], stage_a.value[stage_a.source == "rating"]))
        self.ranks = dict(zip(stage_a.code[stage_a.source == "rank_pl"], stage_a.value[stage_a.source == "rank_pl"]))
        prepared = data.prepared[data.prepared.video_id == video_id]
        self.prepared = dict(zip(prepared.code, prepared.value))
        self.reaction_turns = None          # read once, on first use, by _rx_windows
        self.reaction_pairs: list[dict] = []


def _exposure(pitch: _Pitch, history: list[dict]) -> dict:
    return {"hist_turns": len(history),
            "hist_seconds": float(sum(max(0.0, t["end_s"] - t["start_s"]) for t in history)),
            "hist_questions": _count(history, "question", speakers=set(pitch.investors)),
            "demo_seconds": float(sum(max(0.0, t["end_s"] - t["start_s"]) for t in history
                                      if t["function"] == DEMONSTRATES)),
            "prepared_seconds": float(pitch.video["prepared_seconds"])}


def _stage_a(pitch: _Pitch, with_ranks: bool) -> dict:
    out = {}
    for code in pitch.stage_a_codes:
        value = pitch.ratings.get(code, np.nan)
        out[f"sa_{code}"] = value
        out[f"sa_{code}_missing"] = int(value != value)
    if with_ranks:
        for code in pitch.stage_a_codes:
            out[f"sa_rank_{code}"] = pitch.ranks.get(code, np.nan)
    return out


def _ask(video) -> dict:
    amount = float(video["ask_amount_usd"])
    return {"ask_log_amount": float(np.log10(amount)) if amount == amount and amount > 0 else np.nan,
            "ask_equity_pct": float(video["ask_equity_pct"]),
            "ask_missing": int(video["ask_missing"] != "none")}


def _question_shares(out: dict, questions: list[dict]) -> None:
    _stat(out, "q_sev_mean", [q["PR-SEV"] for q in questions])
    _stat(out, "q_hard_share", [float(q["PR-SEV"] >= HARD_SEVERITY) for q in questions if q["PR-SEV"] == q["PR-SEV"]])
    for suffix, frame, _ in QUESTION_FRAMES:
        _stat(out, f"q_{suffix}", [float(q["frame"] == frame) for q in questions])
    for suffix, concept, _ in QUESTION_CONCEPTS:
        _stat(out, f"q_{suffix}", [float(q["concept"] == concept) for q in questions])


def _entropy_bits(values: list) -> float:
    """Shannon entropy in bits of the distribution of `values` (0.0 for one distinct value; NaN for none)."""
    if not values:
        return np.nan
    p = pd.Series(values).value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(p * np.log2(p)).sum()) + 0.0       # + 0.0: a single value gives 0.0, never -0.0


def _question_sequence(out: dict, questions: list[dict]) -> None:
    """The sequence features of the investor questions, which must come in turn order: severity trend, same-asker
    pursuit, frame switches and concept entropy. None of them reads the exchange grouping."""
    rated = [q["PR-SEV"] for q in questions if q["PR-SEV"] == q["PR-SEV"]]
    out["q_sev_trend"] = _trend(rated)
    out["q_sev_trend_n"] = len(rated)
    out["q_sev_trend_missing"] = int(not rated)
    pairs = list(zip(questions, questions[1:]))
    _stat(out, "q_pursuit_share", [float(a["speaker"] == b["speaker"] and b["PR-SEV"] > a["PR-SEV"])
                                   for a, b in pairs if a["PR-SEV"] == a["PR-SEV"] and b["PR-SEV"] == b["PR-SEV"]])
    _stat(out, "q_frame_switch_share", [float(a["frame"] != b["frame"]) for a, b in pairs
                                        if a["frame"] in SWITCH_FRAMES and b["frame"] in SWITCH_FRAMES])
    concepts = [q["concept"] for q in questions if isinstance(q["concept"], str) and q["concept"] != NO_CONCEPT]
    out["q_concept_entropy"] = _entropy_bits(concepts)
    out["q_concept_entropy_n"] = len(concepts)
    out["q_concept_entropy_missing"] = int(not concepts)


def _rx_stats(out: dict, windows: list[dict], reactors: set[str]) -> None:
    per_answer = _per_answer(windows, reactors)
    out["rx_answers"] = len(per_answer)
    _stat(out, "rx_reaction_rate", [len(rs) for rs in per_answer])
    for suffix, kind, _ in RX_SHARES:
        _stat(out, f"rx_{suffix}", [float(any(r["kind"] == kind for r in rs)) for rs in per_answer])


def _pitch_row(data: PanelData, pitch: _Pitch, cut_pos: int, with_ranks: bool) -> dict:
    """Every feature of one pitch, from its eligible turns before `cut_pos`, family by family."""
    investors = set(pitch.investors)
    video_id = pitch.video["video_id"]
    history = eligible_turns(data, video_id, cut_pos).to_dict("records")

    out = {name: pitch.video[name] for name in CONTROLS}
    out.update(_ask(pitch.video))
    out.update(_exposure(pitch, history))
    out.update(_stage_a(pitch, with_ranks))
    codes = PREPARED_CODES + tuple(c for c in pitch.prepared if c not in PREPARED_CODES)
    for code in codes:
        value = float(pitch.prepared.get(code, np.nan))
        out[prepared_column(code)] = value
        out[f"{prepared_column(code)}_missing"] = int(value != value)

    questions = sorted((t for t in history if t["function"] == "question" and t["speaker"] in investors),
                       key=lambda t: t["pos"])
    answers = [t for t in history if t["function"] == "answer"]
    out["q_askers"] = len({q["speaker"] for q in questions})
    _question_shares(out, questions)
    _question_sequence(out, questions)

    _answer_stats(out, "a", answers, frame_match=True)
    _hard_answer_stats(out, "a", questions, answers)

    _rx_stats(out, _rx_windows(pitch, data, history), investors)
    _next_move_stats(out, "nm", _followups(data, video_id, dict.fromkeys(pitch.investors, cut_pos), history,
                                           investors))
    out["accepts_turns"] = _count(history, *RECEIPT_FUNCTIONS, speakers=investors)
    out["challenge_turns"] = _count(history, "challenges", speakers=investors)

    out["social_turns"] = _count(history, *SOCIAL_FUNCTIONS, speakers=investors)
    out["founder_social"] = _count(history, *SOCIAL_FUNCTIONS, speakers=pitch.founders)
    out["hostile_at_founder"] = _count(history, "hostile", speakers=investors)
    out["founder_repairs"] = _count(history, *REPAIR_FUNCTIONS, speakers=pitch.founders)
    return out


def _with_ranks(data: PanelData) -> bool:
    return bool((data.stage_a.source == "rank_pl").any())


def _as_rates(table: pd.DataFrame) -> pd.DataFrame:
    """`table` with the counts of RATE_COLUMNS per minute of the history; NaN where the history has no seconds."""
    out = table.copy()
    minutes = out["hist_seconds"].astype(float) / 60.0
    minutes = minutes.where(minutes > 0)
    for name in RATE_COLUMNS:
        out[name] = out[name].astype(float) / minutes
    return out


def _count_twin(name: str) -> bool:
    """A `<statistic>_n` column: the observations behind a mean or share."""
    return name.endswith("_n") and name[:-2] in FEATURES and f"{name[:-2]}_missing" in FEATURES


def _average_passes(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """`first` with every numeric feature column averaged with `second` on the pitches `second` has.

    A `_n` column is the sum of the two, every other numeric feature the mean of the values observed (a
    `_missing` indicator thus 0.5 when one pass has none); a pitch `second` lacks keeps its first-pass row.
    A feature of PLACEHOLDER_FEATURES counts as not observed in a pass whose `_missing` is 1, so the other
    pass's value stands alone, and it keeps its placeholder 0.0 when neither pass observes it.
    Keys, outcomes and the categorical controls come from `first`, and so does the set of columns.
    """
    other = second.set_index("video_id").reindex(first.video_id)
    has = other.index.isin(second.video_id)
    out = first.copy()
    for name in first.columns:
        if FEATURES.get(name, {}).get("family") is None or not pd.api.types.is_numeric_dtype(first[name]):
            continue
        a = first[name].to_numpy(dtype=float)
        b = other[name].to_numpy(dtype=float) if name in other.columns else np.full(len(first), np.nan)
        b = np.where(has, b, np.nan)
        if name in PLACEHOLDER_FEATURES:
            a = np.where(first[f"{name}_missing"].to_numpy(dtype=float) == 1, np.nan, a)
            b = np.where(other[f"{name}_missing"].to_numpy(dtype=float) == 1, np.nan, b)
        if _count_twin(name):
            out[name] = np.where(has, a + np.nan_to_num(b), a).astype(int)
        else:
            mean = np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2.0))
            out[name] = np.where(np.isnan(mean), 0.0, mean) if name in PLACEHOLDER_FEATURES else mean
    return out


def pitch_features(data: PanelData, exposure: str = EXPOSURE_COUNTS) -> pd.DataFrame:
    """One row per pitch: keys (video_id, source, cut_pos), the outcome columns of `pitch_outcome`
    (any_offer, first_offer_pos, any_offer_source), then every feature of FEATURES the pitch carries.

    The history stops at the first decision of any investor, whatever its kind, so no decision is
    read and the cut does not depend on the outcome.

    `exposure="rates"` divides the counts of RATE_COLUMNS by the minutes of the history (`hist_seconds /
    60`); a pitch whose history has no seconds has NaN there. The table's `attrs["exposure"]` says which,
    and `describe_row` labels those columns per minute.

    With a second pass (`data.replicate`, from `load_run(..., passes="both")`) every numeric feature is
    the mean of the two passes, a `_n` their sum and a `_missing` their mean. The cut is a property of
    each pass's record: each pass is cut at its own first decision, and its features read only its own
    history. The keys, `cut_pos`, the outcome and the categorical controls are those of the first pass,
    the record `passes="best"` reads; a pitch without a second pass keeps its first-pass row.
    """
    _check_exposure(exposure)
    declare_stage_a(data.stage_a_codes)
    declare_prepared(tuple(dict.fromkeys(data.prepared.code)))
    table = pitch_outcome(data).merge(pitch_cut(data), on="video_id", validate="one_to_one")
    with_ranks = _with_ranks(data)
    rows = []
    for r in table.itertuples():
        pitch = _Pitch(data, r.video_id)
        rows.append({"video_id": r.video_id, "source": pitch.video["source"], "cut_pos": int(r.cut_pos),
                     **{column: getattr(r, column) for column in OUTCOME_COLUMNS},
                     **_pitch_row(data, pitch, int(r.cut_pos), with_ranks)})
    out = pd.DataFrame(rows)
    if exposure == EXPOSURE_RATES and len(out):
        out = _as_rates(out)
    if data.replicate is not None and len(out) and len(data.replicate.videos):
        out = _average_passes(out, pitch_features(data.replicate, exposure=exposure))
    out.attrs["exposure"] = exposure
    return out


# ---- the same rows, made inspectable -------------------------------------------------------

def history_table(data: PanelData, video_id: str) -> pd.DataFrame:
    """Every turn of a video with whether the pitch row's features read it, and why not when they do not.

    Columns: the turn columns of the data, then `cut_pos`, `used` and `why_excluded` (empty when used).
    The used turns are exactly the history `pitch_features` computes the pitch's row from.
    """
    cut = pitch_cut(data)
    found = cut[cut.video_id == video_id]
    if len(found) != 1:
        raise ValueError(f"no row for video {video_id!r}. Videos: {', '.join(sorted(cut.video_id))}.")
    cut_pos = int(found.cut_pos.iloc[0])
    table = data.turns[data.turns.video_id == video_id].copy()
    reason = exclusion_reasons(data, video_id, cut_pos)
    table["cut_pos"] = cut_pos
    table["used"] = reason == ""
    table["why_excluded"] = reason
    return table


def describe_row(features: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """One pitch row of a feature table read downwards: feature, family, label, value, denominator, description.

    A mean or share is shown with the number of observations behind it (its `_n` column) as the
    denominator; its `_n` and `_missing` companions are folded into that line rather than listed.
    A missing value stays missing (it is not zero). A table built with `exposure="rates"` (its
    `attrs["exposure"]`) is labelled per minute.
    """
    found = features[features.video_id == video_id]
    if len(found) != 1:
        raise ValueError(f"expected one row for video {video_id!r}, found {len(found)}")
    return _describe(found.iloc[0], features.columns, _registry(features.attrs.get("exposure", EXPOSURE_COUNTS)))


def _describe(row: pd.Series, columns, registry) -> pd.DataFrame:
    """`row` read downwards over the columns `registry` declares (the archive passes its own registry)."""
    out = []
    for name, spec in registry.items():
        if name not in columns or spec["family"] is None:
            continue
        stem = name.removesuffix("_n").removesuffix("_missing")
        if stem != name and stem in registry:
            continue                      # a companion column (observations, none observed) of another feature
        denominator = int(row[f"{name}_n"]) if f"{name}_n" in columns else None
        out.append({"feature": name, "family": spec["family"], "label": spec["label"], "value": row[name],
                    "denominator": denominator, "description": spec["description"]})
    table = pd.DataFrame(out, columns=["feature", "family", "label", "value", "denominator", "description"])
    table["denominator"] = pd.Series([r["denominator"] for r in out], dtype=object)
    return table
