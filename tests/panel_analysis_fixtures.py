"""Synthetic multi-pitch panel runs for the analysis layer. Illustrative only; never empirical records.

`write_run` builds a run directory in the layout the panel pipeline leaves behind
(`<run>/<video_id>/experiment.json`, `results/primary.json`, `results/replicate.json`, `agreement.json`,
optionally `results/reviewed.json` and `<run>/ranking/scores.json`). Every record is validated through
`PanelRecord`. Each pitch has a founder and three to five investors who ask questions, get rated answers,
react, sometimes joke, and then decide one after another.

`planted` makes the offers depend on one thing only (besides a pitch-level appeal that also drives the
Stage A ratings), so a test can check that the matching model step picks it up:

- ``"own"``: an investor offers when the answers to that investor's own questions were good;
- ``"panel_reactions"``: an investor offers when the other investors accepted rather than challenged;
- ``"reaction_events"``: an investor offers by a disposition that only the course of their reactions shows.
  Every investor asks two or three questions; right after the first answer and right after the last, with
  probability sigmoid(strength), the asker accepts once and presses once with a severity-3 question, an
  investor inclined to offer pressing first and accepting last, one inclined not to the other way round.
  Counts and first-move shares are the same for both, so only the trend and the last reaction differ;
- ``None``: offers depend on the pitch-level appeal only;
- ``"pitch_draw"``: offers depend on a pitch-level draw that nothing before a decision shows: the Stage A
  ratings follow a second, independent draw, and an investor voices interest whatever they then decide.
  No feature should predict the outcome of such a run, so it is the null against which a leak shows;
- ``"prepared"``: as ``None``, and the prepared presentation's substance (PRES-SUBS) is 5 whenever the
  pitch's appeal is high, so the prepared ratings carry the appeal beyond the Stage A rank scores; and every
  investor's offer logit gains `strength` when PRES-SUBS is 4 or 5, so the substance moves the offers beyond
  the appeal too.

The prepared presentation ratings and the ask are drawn from a stream of their own, spawned from the run's
generator without drawing from it, so every other draw of every pitch is what it was before they existed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from pipeline.schemas.stage_b_panel import PanelRecord
from tests.panel_fixtures import STAGE_A_IDS, prepared_codes, synthetic_record, turn

RELEASE_ID = "v3-dev9"
INVESTOR_LABELS = ("Investor A", "Investor B", "Investor C", "Investor D", "Investor E", "Investor F")
PROMOTION = ("market_opportunity", "sales", "vision")
PREVENTION = ("competition", "execution", "customer_retention")
INDUSTRIES = ("food_beverage", "home_household", "general_technology", "pets")
FORMS = ("physical_good", "software_digital")
CUSTOMERS = ("consumer", "business_institution", "both")
GENDERS = ("all_women", "all_men", "mixed")
# the decision patterns the first pitch of every run shows, one investor each
FORCED_PATTERNS = ("offer", "joins", "exit", "exit_reentry_offer", "none")
PLANTED = (None, "own", "panel_reactions", "pitch_draw", "reaction_events", "prepared")
PITCH_DRAW = "pitch_draw"
PREPARED_ORDINAL = ("PRES-SUBS", "PRES-COH", "PRES-CTX")       # 1-5; PRES-EVID is a count
ASK_AMOUNTS = (50_000, 100_000, 200_000, 500_000)
ASK_EQUITY = (5, 10, 15, 20)


def _cv(code, value, missing="none"):
    return {"code": code, "value": value, "missing": missing}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def make_record(rng: np.random.Generator, planted: str | None = None, forced: tuple[str, ...] | None = None,
                strength: float = 2.5, ask_missing: str = "none") -> dict:
    """One synthetic pitch as a record dict. `forced` fixes each investor's decision pattern; `ask_missing`
    other than "none" leaves the ask amount and equity out with that reason."""
    if planted not in PLANTED:
        raise ValueError(f"planted must be one of {PLANTED}")
    n_inv = len(forced) if forced else int(rng.integers(3, 6))
    appeal = int(rng.random() < (0.45 if planted == PITCH_DRAW else 0.6))
    shown = int(rng.random() < 0.6) if planted == PITCH_DRAW else appeal     # what the Stage A ratings follow
    # The prepared presentation and the ask, from a child stream: spawning draws nothing from `rng`, so the
    # stream of `rng` and every later pitch of the run stay what they were (one spawn per pitch, so drawing them
    # here or after the decisions gives the same values). All of them are drawn whatever the plant or the ask's
    # missing reason, so that neither changes the other values.
    own = rng.spawn(1)[0]
    prepared = {code: str(int(own.integers(1, 6))) for code in PREPARED_ORDINAL}
    prepared["PRES-EVID"] = str(int(own.integers(0, 9)))
    if planted == "prepared" and appeal:
        prepared["PRES-SUBS"] = "5"
    ask_amount, ask_equity = int(own.choice(ASK_AMOUNTS)), float(own.choice(ASK_EQUITY))
    if ask_missing != "none":
        ask_amount, ask_equity = None, None
    investors = [f"S{i + 1}" for i in range(n_inv)]
    labels = list(rng.choice(INVESTOR_LABELS, size=n_inv, replace=False))
    actors = [{"actor_id": "F1", "role": "founder", "label": "Founder"}] + [
        {"actor_id": a, "role": "investor", "label": str(lab)} for a, lab in zip(investors, labels)]

    turns: list[dict] = []
    exchanges: list[dict] = []
    decisions: list[dict] = []
    clock = [5.0]

    def add(speaker, seconds, function, summary, **kw) -> int:
        tid = len(turns) + 1
        turns.append(turn(tid, speaker, clock[0], clock[0] + seconds, function, summary, **kw))
        clock[0] += seconds + 1.0
        return tid

    def exchange(kind, topic="NONE", label="") -> dict:
        ex = {"exchange_id": len(exchanges) + 1, "kind": kind, "topic": topic, "label": label}
        exchanges.append(ex)
        return ex

    add("F1", 55.0, "presents", "Prepared presentation of the product and the ask.")

    quality = {a: int(rng.choice((-1, 1))) for a in investors}      # how well this investor's questions get answered
    reaction = {a: int(rng.choice((-1, 1))) for a in investors}     # +1 accepts, -1 challenges
    asked = {a: int(rng.choice((0, 1, 2, 3), p=(0.1, 0.35, 0.35, 0.2))) for a in investors}
    reacted: list[str] = []                                         # investors whose reaction the panel has heard
    # drawn only for this plant, so the other plants' random streams stay as they were
    disposition = {a: int(rng.choice((-1, 1))) for a in investors} if planted == "reaction_events" else {}
    if disposition:
        asked = {a: max(2, n) for a, n in asked.items()}

    def questioning(a: str) -> None:
        """Investor `a` asks, is answered, and reacts."""
        if not asked[a]:
            return
        good = quality[a] > 0
        first_concept = None
        ex = exchange("substantive")
        last_answer = None
        for i in range(asked[a]):
            concept = str(rng.choice(PREVENTION if rng.random() < 0.5 else PROMOTION))
            first_concept = first_concept or concept
            q = add(a, 4.0, "question", "Asks about one part of the business.", addressed=["F1"],
                    exchange_id=ex["exchange_id"],
                    codes=[_cv("PR-CONCEPT", concept), _cv("PR-SEV", str(int(rng.integers(1, 4))))])
            resp = (("addresses" if rng.random() < 0.85 else "partial") if good
                    else ("deflects" if rng.random() < 0.7 else "partial"))
            evid = rng.random() < (0.8 if good else 0.2)
            subs = int(rng.integers(4, 6)) if good else int(rng.integers(1, 3))
            answer_concept = concept if rng.random() < 0.7 else str(rng.choice(PROMOTION + PREVENTION))
            ctx = _cv("PREP-CTX", None, "not_applicable") if rng.random() < 0.5 else _cv(
                "PREP-CTX", str(int(rng.integers(2, 5))))
            last_answer = add(
                "F1", float(rng.integers(8, 15)), "answer", "Answers with the figures at hand.", addressed=[a],
                exchange_id=ex["exchange_id"], responds_to=[q],
                codes=[_cv("RS-RESP", resp), _cv("RS-CONCEPT", answer_concept),
                       _cv("RS-EVID", "true" if evid else "false"),
                       _cv("RS-UNCERT", "true" if rng.random() < 0.2 else "false"),
                       _cv("PREP-SUBS", str(subs)), _cv("PREP-COH", str(int(rng.integers(3, 6)))), ctx])
            ends = (i == 0) or (i == asked[a] - 1)
            if disposition and ends and rng.random() < _sigmoid(strength):
                if (disposition[a] > 0) == (i == asked[a] - 1):
                    add(a, 2.0, "accepts", "Accepts the answer.", addressed=["F1"], exchange_id=ex["exchange_id"],
                        responds_to=[last_answer])
                else:
                    add(a, 4.0, "question", "Presses the same point harder.", addressed=["F1"],
                        exchange_id=ex["exchange_id"], codes=[_cv("PR-CONCEPT", concept), _cv("PR-SEV", "3")])
            if not good and rng.random() < 0.2:
                last_answer = add("F1", 5.0, "repair", "Recasts the earlier figure another way.",
                                  addressed=[a], exchange_id=ex["exchange_id"], responds_to=[q])
            if rng.random() < 0.1:
                # an investor's confusion that asks for nothing is NONE (trouble_signal until v3-dev8)
                add(a, 3.0, "NONE", "Says the last figure was hard to follow.", addressed=["F1"],
                    exchange_id=ex["exchange_id"])
        ex["topic"] = first_concept
        ex["label"] = "one business matter"
        add(a, 3.0, "accepts" if reaction[a] > 0 else "challenges", "Reacts to the answer.",
            addressed=["F1"], exchange_id=ex["exchange_id"], responds_to=[last_answer])
        reacted.append(a)
        # the draws that set the closing state up to v3-dev8, kept so that the random stream (and so every
        # synthetic corpus the analysis tests were tuned on) stays what it was
        if rng.random() < 0.85:
            rng.random()
        if rng.random() < 0.05:
            add(a, 3.0, "hostile", "Mocks the founder rather than the claim.", addressed=["F1"])

    def social() -> None:
        joker = str(rng.choice(investors))
        ex = exchange("social")
        bid = add(joker, 3.0, "social", "Makes a friendly joke about the product.", addressed=["F1"],
                  exchange_id=ex["exchange_id"])
        add("F1", 2.0, "social", "Laughs and returns the joke.", addressed=[joker],
            exchange_id=ex["exchange_id"], responds_to=[bid])

    def demonstration() -> None:
        watcher = str(rng.choice(investors))
        ex = exchange("demonstration", label="product demonstration")
        demo = add("F1", 12.0, "demonstrates", "Shows the product working.", addressed=["PANEL"],
                   exchange_id=ex["exchange_id"])
        add(watcher, 2.0, "accepts", "Nods at the demonstration.", addressed=["F1"],
            exchange_id=ex["exchange_id"], responds_to=[demo])

    # ---- decisions: an investor decides on what the room has heard so far ------------------------
    def signal(a: str) -> int:
        if planted == "own":
            return quality[a] if asked[a] else 0
        if planted == "panel_reactions":
            return int(np.sign(sum(reaction[b] for b in reacted if b != a)))
        if planted == "reaction_events":
            return disposition[a] if asked[a] else 0
        if planted == "prepared":           # the prepared substance moves every investor, beyond the appeal
            return int(int(prepared["PRES-SUBS"]) >= 4)
        return 0

    negotiation = None
    open_offer: str | None = None
    n_offers = 0
    returning: list[str] = []

    def decide(actor, kind, summary, offer_id=None, in_negotiation=False, spoken=True) -> None:
        nonlocal negotiation
        ex_id = None
        if in_negotiation:
            if negotiation is None:
                negotiation = exchange("negotiation", label="the offers")
            ex_id = negotiation["exchange_id"]
        start = clock[0]
        tid = add(actor, 4.0, "decides", summary, addressed=["F1"] if actor != "F1" else ["PANEL"],
                  exchange_id=ex_id)
        decisions.append({"decision_id": len(decisions) + 1, "actor": actor, "kind": kind, "time_s": start + 1.0,
                          "turn_id": tid if spoken else None, "offer_id": offer_id, "terms": "", "note": ""})

    def decision(a: str, pattern: str | None = None) -> None:
        nonlocal open_offer, n_offers
        if pattern is None:
            if rng.random() < 0.06:
                pattern = "none"
            else:
                base = (-3.0 + 4.0 * appeal) if planted == PITCH_DRAW else (-1.6 + 1.8 * appeal)
                offers = rng.random() < _sigmoid(base + strength * signal(a))
                pattern = ("exit_reentry_offer" if rng.random() < 0.15 else "offer") if offers else "exit"
            if pattern == "offer" and open_offer and rng.random() < 0.4:
                pattern = "joins"
        if pattern == "none":
            return
        if ((rng.random() < 0.3) if planted == PITCH_DRAW else (pattern != "exit" and rng.random() < 0.5)) and (
                planted == PITCH_DRAW):
            # Near-outcome talk, an accepts turn since v3-dev9 (expresses_interest before, which the focal investor's
            # own features left out). Only the null run voices it, whatever the investor then decides; in the other
            # runs it would be a tell of the offer that nothing now removes, so it is drawn (the random stream stays
            # what it was) but not spoken.
            add(a, 3.0, "accepts", "Says the product is appealing.", addressed=["F1"])
        if pattern == "exit":
            # one exit in ten is recorded without the turn that speaks it
            decide(a, "exit", "Declares out.", spoken=bool(forced) or rng.random() >= 0.1)
        elif pattern == "exit_reentry_offer":
            decide(a, "exit", "Declares out.")
            returning.append(a)
        elif pattern == "joins" and open_offer:
            decide(a, "joins_offer", "Joins the standing offer.", offer_id=open_offer, in_negotiation=True)
        else:
            n_offers += 1
            open_offer = f"O{n_offers}"
            decide(a, "offer", "Makes an offer.", offer_id=open_offer, in_negotiation=True)
            if rng.random() < 0.5:
                add("F1", 4.0, "negotiates", "Asks for a smaller stake.", addressed=[a],
                    exchange_id=negotiation["exchange_id"])

    # ---- the encounter: some investors decide right after their own questions, the rest at the end,
    # so later questions, answers and reactions lie beyond an early decider's cut
    early = set() if forced else {a for a in investors if rng.random() < 0.35}
    social_after = int(rng.integers(0, n_inv)) if rng.random() < 0.5 else None
    demo_after = int(rng.integers(0, n_inv)) if rng.random() < 0.3 else None
    for i, a in enumerate(investors):
        questioning(a)
        if a in early:
            decision(a)
        if i == social_after:
            social()
        if i == demo_after:
            demonstration()
    late = [a for a in investors if a not in early]
    for i, a in enumerate(late if forced else [str(a) for a in rng.permutation(late)]):
        decision(a, forced[i] if forced else None)
    for a in returning:
        decide(a, "re_entry", "Comes back in.", in_negotiation=True)
        n_offers += 1
        open_offer = f"O{n_offers}"
        decide(a, "offer", "Makes an offer after coming back.", offer_id=open_offer, in_negotiation=True)
    if open_offer:
        decide("F1", "founder_accepts", "Accepts an offer.", offer_id=open_offer, in_negotiation=True)

    # Ten noise draws whatever the size of the Stage A book: the random stream after them (and so every
    # synthetic corpus the analysis tests were tuned on) stays what it was under the ten-item book.
    noise = rng.normal(0, 0.6, size=10)
    record = {
        "actors": actors,
        "prepared_spans": [{"start_s": 5.0, "end_s": 60.0}],
        "excluded_spans": [],
        "gaps": [],
        "source_complete": True,
        "stage_a": [{"code": code, "value": int(np.clip(round(2.5 + 1.5 * shown + noise[i]), 1, 5)),
                     "uncodable": False, "moment_s": None, "note": ""} for i, code in enumerate(STAGE_A_IDS)],
        "prepared": prepared_codes(subs=prepared["PRES-SUBS"], coh=prepared["PRES-COH"],
                                   ctx=prepared["PRES-CTX"], evid=prepared["PRES-EVID"]),
        "venture": {"description": "A synthetic venture.", "primary_industry": str(rng.choice(INDUSTRIES)),
                    "secondary_industries": [], "product_form": str(rng.choice(FORMS)),
                    "customer_type": str(rng.choice(CUSTOMERS)), "founder_gender": str(rng.choice(GENDERS)),
                    "product_category": "synthetic", "earliest_s": 8.0, "note": "",
                    "ask_amount_usd": ask_amount, "ask_equity_pct": ask_equity, "ask_missing": ask_missing},
        "turns": turns,
        "exchanges": exchanges,
        "decisions": decisions,
        "claim_conflicts": [],
        "notes": "",
    }
    return PanelRecord.model_validate(record).model_dump(mode="json")


def talk_record(turns, exchanges=((1, "substantive"),), investors=("S1", "S2"), decisions=()):
    """A record whose questioning is `turns`: (speaker, function, exchange id or None, responds_to, severity
    for a question / substance for an answer; an answer gives checkable support when its substance is 3 or
    more). Turn ids start at 2, after the prepared presentation; `decisions` are (actor, kind, turn id)."""
    record = synthetic_record()
    record["actors"] = [{"actor_id": "F1", "role": "founder", "label": "Founder"},
                        {"actor_id": "F2", "role": "founder", "label": "Co-founder"}] + [
        {"actor_id": a, "role": "investor", "label": f"Investor {a}"} for a in investors]
    built = [turn(1, "F1", 5, 60, "presents", "Prepared presentation.")]
    for i, (speaker, function, exchange_id, responds_to, rating) in enumerate(turns, start=2):
        codes = []
        if function == "question":
            codes = [{"code": "PR-CONCEPT", "value": "sales", "missing": "none"},
                     {"code": "PR-SEV", "value": str(rating), "missing": "none"}]
        elif function == "answer":
            codes = [{"code": "RS-RESP", "value": "addresses", "missing": "none"},
                     {"code": "RS-CONCEPT", "value": "sales", "missing": "none"},
                     {"code": "RS-EVID", "value": "true" if rating >= 3 else "false", "missing": "none"},
                     {"code": "RS-UNCERT", "value": "false", "missing": "none"},
                     {"code": "PREP-SUBS", "value": str(rating), "missing": "none"},
                     {"code": "PREP-COH", "value": "4", "missing": "none"},
                     {"code": "PREP-CTX", "value": None, "missing": "not_applicable"}]
        built.append(turn(i, speaker, 60 + 4 * i, 63 + 4 * i, function, "Says something.", ["PANEL"], exchange_id,
                          list(responds_to), codes))
    record["turns"] = built
    record["exchanges"] = [{"exchange_id": e, "kind": kind, "topic": "sales" if kind == "substantive" else "NONE",
                            "label": ""} for e, kind in exchanges]
    record["decisions"] = [{"decision_id": n, "actor": actor, "kind": kind, "time_s": 61.0 + 4 * turn_id,
                            "turn_id": turn_id, "offer_id": "O1" if kind == "offer" else None, "terms": "", "note": ""}
                           for n, (actor, kind, turn_id) in enumerate(decisions, start=1)]
    return PanelRecord.model_validate(record).model_dump(mode="json")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def agreement_stub(video_id: str) -> dict:
    """The headline part of an agreement report, with the values two identical passes give."""
    return {"video_id": video_id, "release_id": RELEASE_ID,
            "label": "model self-consistency under sampling; not validity",
            "turns": {"recall": 1.0, "precision": 1.0, "recall_any_function": 1.0, "precision_any_function": 1.0,
                      "speaker_time_agreement": 1.0, "sequence_agreement": 1.0},
            "exchanges": {"mean_jaccard": 1.0, "co_membership_agreement": 1.0},
            "links": {"recall": 1.0, "precision": 1.0},
            "decisions": {"ever_offer_agreement": 1.0, "offer_membership_agreement": 1.0}}


def write_video(root: Path, video_id: str, record: dict, reviewed: dict | None = None) -> Path:
    """One video directory holding `record` as both passes; `reviewed` adds results/reviewed.json."""
    record = PanelRecord.model_validate(record).model_dump(mode="json")
    vdir = Path(root) / video_id
    duration = max([t["end_s"] for t in record["turns"]] + [60.0]) + 5.0
    _dump(vdir / "experiment.json", {"video_id": video_id, "duration_s": duration, "release_id": RELEASE_ID,
                                     "annotation_context": "descriptive_full_context"})
    for name in ("primary", "replicate"):
        _dump(vdir / "results" / f"{name}.json", {"pass": name, "annotation_context": "descriptive_full_context",
                                                  "record": copy.deepcopy(record)})
    if reviewed is not None:
        _dump(vdir / "results" / "reviewed.json",
              {"pass": "reviewed", "annotation_context": "descriptive_full_context",
               "record": PanelRecord.model_validate(reviewed).model_dump(mode="json")})
    _dump(vdir / "agreement.json", agreement_stub(video_id))
    return vdir


def write_ranking(root: Path, video_ids: list[str], rng: np.random.Generator) -> None:
    scores = {code: {"pl_score": {v: float(rng.normal()) for v in video_ids},
                     "mean_rank": {v: float(rng.random()) for v in video_ids}, "n_orders": 4,
                     "order": list(video_ids)} for code in STAGE_A_IDS}
    _dump(Path(root) / "ranking" / "scores.json", {**scores, "batches_ok": 4, "batches_failed": 0})


def write_run(tmp_path: Path, n_pitches: int, seed: int = 0, planted: str | None = None, ranking: bool = False,
              strength: float = 2.5, name: str = "run") -> Path:
    """A run of `n_pitches` videos v000, v001, ... under `tmp_path/name`. The first pitch shows every
    decision pattern in FORCED_PATTERNS; the rest draw theirs."""
    rng = np.random.default_rng(seed)
    root = Path(tmp_path) / name
    ids = []
    for i in range(n_pitches):
        video_id = f"v{i:03d}"
        record = make_record(rng, planted=planted, forced=FORCED_PATTERNS if i == 0 else None, strength=strength)
        write_video(root, video_id, record)
        ids.append(video_id)
    if ranking:
        write_ranking(root, ids, rng)
    return root
