"""Synthetic panel conversation in the v3-dev4 shape (concept codes since v3-dev7; exchange topics are concepts
and nothing is `other` since v3-dev8; twelve turn functions, no closing state and the prepared presentation
codes since v3-dev9): turns carry the function, exchanges group them.

The story: the founder presents. Investor A asks about acquisition cost and the founder answers;
Investor B presses the same topic, the founder answers and then changes the explanation, and A
says the concern is settled. Investor C jokes, the founder joins in, and C exits. B asks about
repeat purchases, is answered and agrees. The panel then returns to acquisition cost: B asks
again, is answered and says it still does not add up. A makes an offer, B joins, the founder
accepts. Illustrative only; never an empirical record."""
from __future__ import annotations

import copy

from pipeline.schemas.stage_b_panel import seconds_to_clock

DURATION_S = 180.0
STAGE_A_IDS = ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")
STAGE_A_WITH_MOMENT = ("pitch_positivity",)


def _cv(code, value, missing="none"):
    return {"code": code, "value": value, "missing": missing}


def question_codes(concept="operating_efficiency", severity="1"):
    return [_cv("PR-CONCEPT", concept), _cv("PR-SEV", severity)]


def answer_codes(resp="addresses", evid="true", subs="3", concept="operating_efficiency"):
    return [_cv("RS-RESP", resp), _cv("RS-CONCEPT", concept), _cv("RS-EVID", evid), _cv("RS-UNCERT", "false"),
            _cv("PREP-SUBS", subs), _cv("PREP-COH", "4"), _cv("PREP-CTX", None, "not_applicable")]


def prepared_codes(subs="3", coh="4", ctx="2", evid="5"):
    """The four prepared presentation codes (since v3-dev9), rated once per pitch."""
    return [_cv("PRES-SUBS", subs), _cv("PRES-COH", coh), _cv("PRES-CTX", ctx), _cv("PRES-EVID", evid)]


def turn(turn_id, speaker, start_s, end_s, function, summary, addressed=(), exchange_id=None, responds_to=(),
         codes=(), other_concept=None):
    return {"turn_id": turn_id, "speaker": speaker, "start_s": float(start_s), "end_s": float(end_s),
            "function": function, "addressed": list(addressed), "exchange_id": exchange_id,
            "responds_to": list(responds_to), "codes": list(codes), "other_concept": other_concept,
            "summary": summary}


def set_other(t, name):
    """Make a question or answer turn's concept `other`, named `name` (None leaves other_concept null)."""
    next(c for c in t["codes"] if c["code"].endswith("-CONCEPT")).update(value="other", missing="none")
    t["other_concept"] = name


_RECORD = {
    "actors": [
        {"actor_id": "F1", "role": "founder", "label": "Founder"},
        {"actor_id": "S1", "role": "investor", "label": "Investor A"},
        {"actor_id": "S2", "role": "investor", "label": "Investor B"},
        {"actor_id": "S3", "role": "investor", "label": "Investor C"},
    ],
    "prepared_spans": [{"start_s": 5.0, "end_s": 60.0}],
    "excluded_spans": [{"start_s": 0.0, "end_s": 5.0, "kind": "narration"}],
    "gaps": [],
    "source_complete": True,
    "stage_a": [{"code": code, "value": 3, "uncodable": False,
                 "moment_s": 20.0 if code in STAGE_A_WITH_MOMENT else None, "note": "Steady delivery."}
                for code in STAGE_A_IDS],
    "prepared": prepared_codes(),
    "venture": {"description": "Reusable food containers sold online to households.", "primary_industry": "home_household",
                "secondary_industries": [], "product_form": "physical_good", "customer_type": "consumer",
                "founder_gender": "all_men",
                "product_category": "reusable food container", "earliest_s": 8.0, "note": "Stated in the opening.",
                "ask_amount_usd": 200000, "ask_equity_pct": 10.0, "ask_missing": "none"},
    "turns": [
        turn(1, "F1", 5, 60, "presents", "Prepared presentation of the reusable containers and the ask."),
        # exchange 1: acquisition cost, two investors
        turn(2, "S1", 62, 66, "question", "Asks what it costs to acquire one customer.", ["F1"], 1,
             codes=question_codes()),
        turn(3, "F1", 66, 80, "answer", "States a blended acquisition figure and names the channels.", ["S1"], 1, [2],
             answer_codes()),
        turn(4, "S2", 80, 86, "question", "Disputes the paid-channel assumption behind that figure.", ["F1"], 1,
             codes=question_codes(severity="3")),
        turn(5, "F1", 86, 94, "answer", "Defends the figure by separating paid from organic customers.", ["S2"], 1, [4],
             answer_codes(subs="4")),
        turn(6, "F1", 94, 100, "repair", "Recasts the acquisition figure per channel instead of blended.",
             ["S2"], 1, [4]),
        turn(7, "S1", 100, 104, "accepts", "Says the per-channel breakdown settles his acquisition concern.",
             ["F1"], 1, [6]),
        # exchange 2: social
        turn(8, "S3", 105, 108, "social", "Jokes that her kitchen drawers are already overflowing with lids.",
             ["F1"], 2),
        turn(9, "F1", 108, 110, "social", "Laughs and promises these lids stack flat.", ["S3"], 2, [8]),
        turn(10, "S3", 111, 114, "decides", "Declares herself out.", ["F1"]),
        # exchange 3: repeat purchases
        turn(11, "S2", 115, 119, "question", "Asks for the repeat purchase rate.", ["F1"], 3,
             codes=question_codes("customer_retention", "2")),
        turn(12, "F1", 119, 130, "answer", "Gives the share of buyers who reorder within six months.", ["S2"], 3, [11],
             answer_codes(concept="customer_retention")),
        turn(13, "S2", 130, 133, "accepts", "Calls the reorder rate strong.", ["F1"], 3, [12]),
        # the panel returns to exchange 1
        turn(14, "S2", 134, 138, "question", "Returns to acquisition and asks what the paid channel costs alone.",
             ["F1"], 1, codes=question_codes(severity="2")),
        turn(15, "F1", 138, 150, "answer", "Quotes the paid-channel cost and expects it to fall.", ["S2"], 1, [14],
             answer_codes(evid="false", concept="forecast")),
        turn(16, "S2", 150, 154, "challenges", "Says the paid cost still exceeds the margin.", ["F1"], 1, [15]),
        # exchange 4: negotiation
        turn(17, "S1", 158, 163, "decides", "Offers the amount asked for a larger stake.", ["F1"], 4),
        turn(18, "S2", 163, 167, "decides", "Joins the offer and splits it.", ["F1"], 4),
        turn(19, "F1", 169, 172, "decides", "Accepts the joint offer.", ["PANEL"], 4),
    ],
    "exchanges": [
        {"exchange_id": 1, "kind": "substantive", "topic": "operating_efficiency", "label": "customer acquisition cost"},
        {"exchange_id": 2, "kind": "social", "topic": "NONE", "label": ""},
        {"exchange_id": 3, "kind": "substantive", "topic": "customer_retention", "label": "repeat purchase rate"},
        {"exchange_id": 4, "kind": "negotiation", "topic": "NONE", "label": "joint offer"},
    ],
    "decisions": [
        {"decision_id": 1, "actor": "S3", "kind": "exit", "time_s": 111.0, "turn_id": 10, "offer_id": None, "terms": "", "note": "Out."},
        {"decision_id": 2, "actor": "S1", "kind": "offer", "time_s": 159.0, "turn_id": 17, "offer_id": "O1", "terms": "200k for 20%", "note": "States terms."},
        {"decision_id": 3, "actor": "S2", "kind": "joins_offer", "time_s": 164.0, "turn_id": 18, "offer_id": "O1", "terms": "splits 200k for 20%", "note": "Joins."},
        {"decision_id": 4, "actor": "F1", "kind": "founder_accepts", "time_s": 170.0, "turn_id": 19, "offer_id": "O1", "terms": "", "note": "Accepts joint offer."},
    ],
    "claim_conflicts": [],
    "notes": "",
}


def synthetic_record() -> dict:
    return copy.deepcopy(_RECORD)


def _clock(value):
    return None if value is None else seconds_to_clock(value)


def to_wire(record: dict) -> dict:
    """A normalized record dict in the form the provider returns: times as M:SS strings."""
    d = copy.deepcopy(record)
    d["prepared_spans"] = [{"start": _clock(s["start_s"]), "end": _clock(s["end_s"])} for s in d["prepared_spans"]]
    d["excluded_spans"] = [{"start": _clock(s["start_s"]), "end": _clock(s["end_s"]), "kind": s["kind"]}
                           for s in d["excluded_spans"]]
    d["gaps"] = [{"at": _clock(g["at_s"]), "note": g["note"]} for g in d["gaps"]]
    for t in d["turns"]:
        t["start"], t["end"] = _clock(t.pop("start_s")), _clock(t.pop("end_s"))
    for x in d["decisions"]:
        x["time"] = _clock(x.pop("time_s"))
    for i in d["stage_a"]:
        i["moment"] = _clock(i.pop("moment_s"))
    d["venture"]["earliest"] = _clock(d["venture"].pop("earliest_s"))
    return d


def synthetic_wire() -> dict:
    return to_wire(synthetic_record())
