#!/usr/bin/env python3
"""Compare a v3 panel run with the archived five-video structured pilot, video by video.

Writes comparison_vs_structured.json and .md. The two instruments differ, so this is a coverage,
consistency and cost comparison, not an accuracy test. All values are model-coded.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_b.panel_books import (  # noqa: E402
    CONCEPT_CODE_OF, PanelBooks, canonical_value, frame_of, load_panel_books, makes_offer,
)
from pipeline.stage_b.panel_validate import RESPONSE_FUNCTIONS  # noqa: E402

CODE_IDS = ("PR-FRAME", "PR-SEV", "RS-RESP", "RS-FRAME", "RS-EVID", "RS-UNCERT", "PREP-SUBS", "PREP-COH", "PREP-CTX")
# The structured pilot recorded the frame; the panel run records the concept, and its frame is derived.
FRAME_FROM_CONCEPT = {"PR-FRAME": "PR-CONCEPT", "RS-FRAME": "RS-CONCEPT"}
MISSING = "∅"
OLD_STAGE_A = ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")
OLD_TO_NEW_KIND = {"offer": "offer", "joins_offer": "joins_offer", "out": "exit", "withdraws_offer": "withdraws_offer"}
MATCH_WINDOW_S = 20.0
EPISODE_FILE = re.compile(r"accepted_ep[_-]?\d+\.json$")  # old runs name them ep_1, ep-1 and ep1


def _warn(path: Path, why: str) -> None:
    print(f"warning: {path}: {why}; treated as missing", file=sys.stderr)


def _read(path: Path):
    try:
        text = path.read_text()
    except OSError:
        return None
    try:
        return json.loads(text)
    except ValueError:
        _warn(path, "cannot be parsed")
        return None


def _calls(path: Path) -> list[dict] | None:
    try:
        text = path.read_text()
    except OSError:
        return None
    calls = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict):
            calls.append(row)
        else:
            print(f"warning: {path}: line {n} skipped (not a JSON object)", file=sys.stderr)
    return calls


def _cost(prefix: str, calls: list[dict] | None) -> dict:
    keys = ("input_tokens", "output_tokens", "thinking_tokens")
    if calls is None:
        return {f"{prefix}_calls": None, **{f"{prefix}_{k}": None for k in keys}}
    return {f"{prefix}_calls": len(calls),
            **{f"{prefix}_{k}": sum(c.get(k) or 0 for c in calls) for k in keys}}


def _old_value(item) -> str:
    if not isinstance(item, dict) or item.get("uncodable") or item.get("value") is None:
        return MISSING
    value = item["value"]
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _merge(spans: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _length(spans) -> float:
    return sum(b - a for a, b in spans)


def _iou(a: list[list[float]], b: list[list[float]]) -> float | None:
    a, b = _merge(a), _merge(b)
    inter = sum(max(0.0, min(x1, y1) - max(x0, y0)) for x0, x1 in a for y0, y1 in b)
    union = _length(a) + _length(b) - inter
    return round(inter / union, 4) if union > 0 else None


def _load_new_record(new_dir: Path, books: PanelBooks) -> dict | None:
    """The primary record with its structure labels in the codebook's spelling."""
    data = _read(new_dir / "results" / "primary.json")
    record = data.get("record") if isinstance(data, dict) else None
    return books.canonical_structure(record) if isinstance(record, dict) else None


def _old_episodes(old_dir: Path) -> list[dict]:
    eps = []
    for path in sorted((old_dir / "results").glob("accepted_ep*.json")):
        if EPISODE_FILE.match(path.name):
            data = _read(path)
            if isinstance(data, dict):
                eps.append(data.get("record") or {})
    return eps


def _old_prepared(old_dir: Path) -> dict | None:
    data = _read(old_dir / "results" / "accepted_prepared.json")
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("record"), dict):
        data = data["record"]
    return {k: {"value": v.get("value"), "uncodable": bool(v.get("uncodable"))}
            for k, v in data.items() if isinstance(v, dict) and k in OLD_STAGE_A} or None


COVERAGE_KEYS = (
    "old_cutoff_s", "new_last_turn_end_s", "new_turns", "new_turns_after_cutoff", "new_questions",
    "new_questions_after_cutoff", "new_answers", "new_answers_after_cutoff", "old_episodes", "new_exchanges",
    "new_substantive_exchanges_after_cutoff", "new_multi_investor_exchanges", "new_reopened_exchanges",
    "new_social_exchanges", "new_demonstration_seconds", "new_closing_counts", "new_topic_counts",
    "new_function_counts", "new_unlinked_responses", "new_concept_counts")
DISTRIBUTIONS = ("new_exchanges", "new_closing_counts", "new_topic_counts", "new_function_counts")


def _concept_counts(turns: list[dict], books: PanelBooks) -> dict:
    """{"question": {concept: n}, "answer": {concept: n}} from the concept code of each such turn."""
    out = {}
    for function, code in CONCEPT_CODE_OF.items():
        values = [MISSING if cv["value"] is None else _canonical(books, code, cv["value"])
                  for t in turns if t["function"] == function for cv in t["codes"] if cv["code"] == code]
        out[function] = dict(sorted(_count(values).items()))
    return out


def _concepts_text(counts: dict | None) -> str:
    if counts is None:
        return "n/a"
    return "; ".join(f"{function}s: {_dist(counts[function])}" for function in CONCEPT_CODE_OF)


def _reopened(turns: list[dict], exchange_id: int) -> bool:
    """True when a turn of another exchange lies between the first and last turn of this one.

    Turns that belong to no exchange (an exit, an aside) do not interrupt it.
    """
    places = [i for i, t in enumerate(turns) if t["exchange_id"] == exchange_id]
    if not places:
        return False
    return any(t["exchange_id"] not in (None, exchange_id) for t in turns[places[0]:places[-1] + 1])


def _coverage(record: dict | None, boundaries: dict | None, n_old_episodes: int, books: PanelBooks) -> dict:
    cutoff = boundaries.get("cutoff_s") if boundaries else None
    out = {key: None for key in COVERAGE_KEYS}
    out.update(old_cutoff_s=cutoff, old_episodes=n_old_episodes)
    if record is None:
        return out
    turns, exchanges = record["turns"], record["exchanges"]
    questions = [t for t in turns if t["function"] == "question"]
    answers = [t for t in turns if t["function"] == "answer"]
    substantive = [e for e in exchanges if e["kind"] == "substantive"]
    first_start: dict[int, float] = {}
    for t in turns:
        if t["exchange_id"] is not None:
            first_start.setdefault(t["exchange_id"], t["start_s"])
    kind_of = {e["exchange_id"]: e["kind"] for e in exchanges}
    investor_ids = {a["actor_id"] for a in record["actors"] if a["role"] == "investor"}
    speakers: dict[int, set] = {}
    for t in turns:
        if t["speaker"] in investor_ids:
            speakers.setdefault(t["exchange_id"], set()).add(t["speaker"])
    out.update(
        new_last_turn_end_s=max((t["end_s"] for t in turns), default=None),
        new_turns=len(turns), new_questions=len(questions), new_answers=len(answers),
        new_exchanges=dict(sorted(_count(e["kind"] for e in exchanges).items())),
        new_multi_investor_exchanges=sum(len(speakers.get(e["exchange_id"], ())) >= 2 for e in substantive),
        new_reopened_exchanges=sum(_reopened(turns, e["exchange_id"]) for e in substantive),
        new_social_exchanges=sum(e["kind"] == "social" for e in exchanges),
        new_demonstration_seconds=sum(t["end_s"] - t["start_s"] for t in turns
                                      if kind_of.get(t["exchange_id"]) == "demonstration"),
        # closing states were recorded up to v3-dev8 only; a v3-dev9 record gives an empty count
        new_closing_counts=dict(sorted(_count(i["closing"] for e in substantive
                                              for i in e.get("investors", [])).items())),
        new_topic_counts=dict(sorted(_count(e["topic"] for e in substantive).items())),
        new_function_counts=dict(sorted(_count(t["function"] for t in turns).items())),
        new_unlinked_responses=sum(t["function"] in RESPONSE_FUNCTIONS and not t["responds_to"] for t in turns),
        new_concept_counts=_concept_counts(turns, books),
    )
    if cutoff is not None:
        out.update(
            new_turns_after_cutoff=sum(t["start_s"] >= cutoff for t in turns),
            new_questions_after_cutoff=sum(t["start_s"] >= cutoff for t in questions),
            new_answers_after_cutoff=sum(t["start_s"] >= cutoff for t in answers),
            new_substantive_exchanges_after_cutoff=sum(
                e["exchange_id"] in first_start and first_start[e["exchange_id"]] >= cutoff for e in substantive),
        )
    return out


def _decisions(record: dict | None, boundaries: dict | None) -> dict:
    out = {k: None for k in ("old_n_investors", "old_n_ever_offer", "new_n_investors", "new_n_ever_offer",
                             "any_offer_old", "any_offer_new", "any_offer_agree", "matched_events")}
    sharks = None
    if boundaries:
        sharks = (boundaries.get("outcomes_model_reviewed_not_gold") or {}).get("sharks")
    if sharks is not None:
        ever = [s for s in sharks if s.get("ever_offer") is True]
        out.update(old_n_investors=len(sharks), old_n_ever_offer=len(ever), any_offer_old=bool(ever))
    if record is not None:
        investors = {a["actor_id"] for a in record["actors"] if a["role"] == "investor"}
        ever_new = {d["actor"] for d in record["decisions"] if makes_offer(d["kind"], d.get("offer_id"))} & investors
        out.update(new_n_investors=len(investors), new_n_ever_offer=len(ever_new), any_offer_new=bool(ever_new))
    if out["any_offer_old"] is not None and out["any_offer_new"] is not None:
        out["any_offer_agree"] = out["any_offer_old"] == out["any_offer_new"]
    if sharks is not None and record is not None:
        old_events = [(OLD_TO_NEW_KIND[e["kind"]], e["source_s"]) for s in sharks for e in s.get("events", [])
                      if e.get("kind") in OLD_TO_NEW_KIND and e.get("source_s") is not None]
        new_events = [(d["kind"], d["time_s"]) for d in record["decisions"] if d["kind"] in OLD_TO_NEW_KIND.values()]
        pairs = sorted((abs(ot - nt), i, j) for i, (ok, ot) in enumerate(old_events)
                       for j, (nk, nt) in enumerate(new_events) if ok == nk and abs(ot - nt) <= MATCH_WINDOW_S)
        used_old, used_new = set(), set()
        for _, i, j in pairs:
            if i not in used_old and j not in used_new:
                used_old.add(i)
                used_new.add(j)
        out["matched_events"] = {"old": len(old_events), "matched": len(used_old), "new": len(new_events)}
    return out


def _prepared(record: dict | None, boundaries: dict | None) -> dict:
    old = None
    if boundaries and boundaries.get("prepared_start_s") is not None and boundaries.get("prepared_end_s") is not None:
        old = [boundaries["prepared_start_s"], boundaries["prepared_end_s"]]
    new = [[s["start_s"], s["end_s"]] for s in record["prepared_spans"]] if record is not None else None
    return {"old": old, "new": new, "iou": _iou([old], new) if old and new else None}


def _canonical(books: PanelBooks, code: str, value: str) -> str:
    try:
        return canonical_value(books.panel.by_id(code), value)
    except KeyError:
        return value


def _codes(record: dict | None, boundaries: dict | None, episodes: list[dict], books: PanelBooks) -> dict:
    cutoff = boundaries.get("cutoff_s") if boundaries else None
    out = {}
    for code in CODE_IDS:
        old = _count(_old_value(ep[code]) for ep in episodes if code in ep)
        before = after = None
        if record is not None and cutoff is not None:
            b, a = [], []
            recorded = FRAME_FROM_CONCEPT.get(code, code)
            for t in record["turns"]:
                for cv in t["codes"]:
                    if cv["code"] != recorded:
                        continue
                    if recorded == code:
                        text = MISSING if cv["value"] is None else _canonical(books, code, cv["value"])
                    else:
                        text = frame_of(cv["value"]) or MISSING
                    (a if t["start_s"] >= cutoff else b).append(text)
            before, after = _count(b), _count(a)
        out[code] = {"old": old, "new_before_cutoff": before, "new_after_cutoff": after}
    return out


def _stage_a(new_dir: Path, record: dict | None, old_prepared: dict | None) -> dict:
    new = None
    if record is not None:
        replicate = _read(new_dir / "results" / "replicate.json")
        rep = {i["code"]: i["value"] for i in replicate["record"]["stage_a"]} if isinstance(replicate, dict) else {}
        new = {i["code"]: {"primary": i["value"], "replicate": rep.get(i["code"])} for i in record["stage_a"]}
    return {"old": old_prepared, "new": new}


def _self_consistency(new_dir: Path) -> dict | None:
    ag = _read(new_dir / "agreement.json")
    if not isinstance(ag, dict):
        return None
    turns, decisions, exchanges = ag.get("turns") or {}, ag.get("decisions") or {}, ag.get("exchanges") or {}
    closing = exchanges.get("closing") or {}
    codes = {c: {"exact": v.get("exact"), "both_valued": v.get("both_valued")}
             for c, v in (ag.get("codes") or {}).items() if isinstance(v, dict) and "both_valued" in v}
    return {"turn_recall": turns.get("recall"), "turn_precision": turns.get("precision"),
            "turn_recall_any_function": turns.get("recall_any_function"),
            "turn_precision_any_function": turns.get("precision_any_function"),
            "exchanges_primary": exchanges.get("n_primary"), "exchanges_replicate": exchanges.get("n_replicate"),
            "exchanges_matched_at_half": exchanges.get("matched_at_half"),
            "exchange_mean_jaccard": exchanges.get("mean_jaccard"),
            "co_membership_agreement": exchanges.get("co_membership_agreement"),
            "adjusted_co_membership": exchanges.get("adjusted_co_membership"),
            "exchange_topic_agree": exchanges.get("topic_agree"), "exchange_topic_n": exchanges.get("topic_n"),
            "closing_agree": closing.get("agree"), "closing_n": closing.get("n"),
            "ever_offer_agreement": decisions.get("ever_offer_agreement"),
            "offer_membership_agreement": decisions.get("offer_membership_agreement"),
            "speaker_time_agreement": turns.get("speaker_time_agreement"),
            "turn_sequence_agreement": turns.get("sequence_agreement"),
            "time": ag.get("time") if isinstance(ag.get("time"), dict) else None,
            "stage_a_summary": (ag.get("stage_a") or {}).get("_summary"), "codes": codes}


def compare_video(new_dir: Path, old_dir: Path, books: PanelBooks | None = None) -> dict:
    new_dir, old_dir = Path(new_dir), Path(old_dir)
    if books is None:
        books = load_panel_books(load_release())
    record = _load_new_record(new_dir, books)
    boundaries = _read(old_dir / "boundaries.json")
    boundaries = boundaries if isinstance(boundaries, dict) else None
    episodes = _old_episodes(old_dir)
    return {
        "coverage": _coverage(record, boundaries, len(episodes), books),
        "decisions": _decisions(record, boundaries),
        "prepared": _prepared(record, boundaries),
        "codes": _codes(record, boundaries, episodes, books),
        "stage_a": _stage_a(new_dir, record, _old_prepared(old_dir)),
        "cost": {**_cost("old", _calls(old_dir / "calls.jsonl")), **_cost("new", _calls(new_dir / "calls.jsonl"))},
        "self_consistency": _self_consistency(new_dir),
    }


# ---- markdown -------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:,}" if isinstance(v, int) else str(v)


def _dist(d: dict | None) -> str:
    if d is None:
        return "n/a"
    return ", ".join(f"{k}: {n}" for k, n in sorted(d.items())) or "none"


LABELS = {
    "old_cutoff_s": "Old cutoff (s)", "new_last_turn_end_s": "New last turn end (s)", "new_turns": "New turns",
    "new_turns_after_cutoff": "New turns after cutoff", "new_questions": "New questions",
    "new_questions_after_cutoff": "New questions after cutoff", "new_answers": "New answers",
    "new_answers_after_cutoff": "New answers after cutoff", "old_episodes": "Old episodes",
    "new_exchanges": "New exchanges by kind",
    "new_substantive_exchanges_after_cutoff": "New substantive exchanges starting at or after the cutoff",
    "new_multi_investor_exchanges": "New substantive exchanges with two or more investors",
    "new_reopened_exchanges": "New reopened substantive exchanges",
    "new_social_exchanges": "New social exchanges", "new_demonstration_seconds": "New demonstration seconds",
    "new_closing_counts": "New closing states (substantive exchanges)",
    "new_topic_counts": "New substantive exchanges by topic",
    "new_function_counts": "New turns by function", "new_unlinked_responses": "New unlinked responses",
    "new_concept_counts": "New concepts",
    "old_n_investors": "Old investors", "old_n_ever_offer": "Old investors who ever offered",
    "new_n_investors": "New investors", "new_n_ever_offer": "New investors who ever offered (primary)",
    "any_offer_old": "Any offer, old", "any_offer_new": "Any offer, new", "any_offer_agree": "Any offer agrees",
}


TIME_LABELS = (
    ("matched_turn_pairs", "matched turn pairs with times"),
    ("median_start_diff_s", "median start difference (s), replicate minus primary"),
    ("median_abs_start_diff_s", "median absolute start difference (s)"),
    ("max_abs_start_diff_s", "largest absolute start difference (s)"),
    ("turn_pairs_beyond_tolerance", "turn pairs beyond the time threshold"),
    ("decision_pairs_beyond_tolerance", "decision pairs beyond the time threshold"),
)


def _of(agree, n, joiner: str = "of") -> str | None:
    return None if agree is None or n is None else f"{_fmt(agree)} {joiner} {_fmt(n)}"


def _tokens(cost: dict, prefix: str):
    parts = [cost[f"{prefix}_{k}"] for k in ("input_tokens", "output_tokens", "thinking_tokens")]
    return None if any(p is None for p in parts) else sum(parts)


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(_fmt(c) for c in row) + " |" for row in rows]
    return lines


def render_markdown(report: dict) -> str:
    videos = report["videos"]
    L = ["# Panel v3 compared with the structured pilot", "",
         "The two instruments differ. The structured pilot coded everything up to the first investor decision "
         "(the cutoff), while the v3 panel run covers the whole encounter. The old units are issues and episodes; the new "
         "units are turns and the exchanges that group them. The old Stage A has three items; the new one has ten. The old pipeline made many scoped "
         "calls per video; the new one makes two full-video passes per video plus any retries (the table gives the "
         "counts). This is therefore a coverage, consistency and cost comparison, not an "
         "accuracy test, and every value in it is model-coded and not human-validated.", ""]
    for key, label in (("new_run", "New run"), ("old_run", "Old run")):
        if report.get(key):
            L.append(f"{label}: `{report[key]}`  ")
    L += ["", "## Summary across videos", ""]
    rows = []
    for vid, r in videos.items():
        c, d, k, s = r["coverage"], r["decisions"], r["cost"], r["self_consistency"] or {}
        n_exchanges = None if c["new_exchanges"] is None else sum(c["new_exchanges"].values())
        rows.append([vid, c["old_cutoff_s"], c["new_last_turn_end_s"], c["new_turns"], c["new_turns_after_cutoff"],
                     c["new_questions_after_cutoff"], n_exchanges, c["old_episodes"], d["old_n_ever_offer"],
                     d["new_n_ever_offer"], d["any_offer_agree"], k["old_calls"], k["new_calls"], _tokens(k, "old"),
                     _tokens(k, "new"), s.get("turn_recall"), s.get("turn_precision"),
                     s.get("exchanges_matched_at_half"), s.get("co_membership_agreement"),
                     (s.get("time") or {}).get("median_abs_start_diff_s")])
    L += _table(["video", "old cutoff s", "new last turn end s", "new turns", "turns after cutoff",
                 "questions after cutoff", "new exchanges", "old episodes", "old investors offering",
                 "new investors offering", "any offer agrees", "old calls", "new calls", "old tokens", "new tokens",
                 "turn recall", "turn precision", "exchanges matched at half", "co-membership",
                 "median abs start diff s"], rows)
    L += ["", "Tokens are input plus output plus thinking, summed over all attempts. Turn recall and precision are "
          "primary against replicate, and are computed from the speaker, order and wording of turns, not from their "
          "times. Exchanges matched at half counts primary exchanges whose best replicate counterpart shares at "
          "least half of their aligned turns; co-membership is the share of turn pairs that both passes put "
          "together or both keep apart. The median absolute start difference is over matched turns and is a check "
          "on the times.", ""]
    for vid, r in videos.items():
        c, d, p, k = r["coverage"], r["decisions"], r["prepared"], r["cost"]
        L += [f"## {vid}", "", "### Coverage", ""]
        L += _table(["measure", "value"],
                    [[LABELS[n], _concepts_text(v) if n == "new_concept_counts" else _dist(v) if n in DISTRIBUTIONS else v]
                     for n, v in c.items()])
        L += ["", "### Decisions", ""]
        L += _table(["measure", "value"], [[LABELS[n], v] for n, v in d.items() if n != "matched_events"])
        m = d["matched_events"]
        L += ["", "Old decision events matched to new decisions of the same kind within 20 s: "
              + ("n/a" if m is None else f"{m['matched']} of {m['old']} old events; {m['new']} new decisions in those kinds.")]
        L += ["", "### Prepared presentation", "",
              f"Old span: {_fmt(p['old'])}. New spans: {_fmt(p['new'])}. Overlap (IoU): {_fmt(p['iou'])}.", "",
              "### Code values", ""]
        L += _table(["code", "old (all episodes)", "new before cutoff", "new after cutoff"],
                    [[code + (" (new: derived from the concept)" if code in FRAME_FROM_CONCEPT else ""),
                      _dist(v["old"]), _dist(v["new_before_cutoff"]), _dist(v["new_after_cutoff"])]
                     for code, v in r["codes"].items()])
        L += ["", "∅ means uncodable or missing.", "", "### Stage A", ""]
        sa = r["stage_a"]
        if sa["old"] is None:
            L += ["Old three-item Stage A: n/a.", ""]
        else:
            L += _table(["old item", "value"], [[n, "uncodable" if v["uncodable"] else v["value"]] for n, v in sa["old"].items()])
            L.append("")
        if sa["new"] is None:
            L += ["New ten-item Stage A: n/a.", ""]
        else:
            L += _table(["new item", "primary", "replicate"], [[n, v["primary"], v["replicate"]] for n, v in sa["new"].items()])
            L.append("")
        L += ["### Cost", ""]
        L += _table(["", "calls", "input tokens", "output tokens", "thinking tokens"],
                    [[pre, k[f"{pre}_calls"], k[f"{pre}_input_tokens"], k[f"{pre}_output_tokens"],
                      k[f"{pre}_thinking_tokens"]] for pre in ("old", "new")])
        L += ["", "### Self-consistency of the new run (primary against replicate)", ""]
        sc = r["self_consistency"]
        if sc is None:
            L += ["n/a (no agreement report).", ""]
        else:
            L += _table(["measure", "value"], [
                ["turn recall", sc["turn_recall"]], ["turn precision", sc["turn_precision"]],
                ["turn recall, any function", sc["turn_recall_any_function"]],
                ["turn precision, any function", sc["turn_precision_any_function"]],
                ["exchanges, primary and replicate", _of(sc["exchanges_primary"], sc["exchanges_replicate"], "and")],
                ["exchanges matched on half their turns or more", sc["exchanges_matched_at_half"]],
                ["mean overlap of matched exchanges (Jaccard)", sc["exchange_mean_jaccard"]],
                ["co-membership agreement", sc["co_membership_agreement"]],
                ["co-membership agreement, adjusted for chance", sc["adjusted_co_membership"]],
                ["topic agreement, matched substantive exchanges",
                 _of(sc["exchange_topic_agree"], sc["exchange_topic_n"])],
                ["closing-state agreement, investors in both passes", _of(sc["closing_agree"], sc["closing_n"])],
                ["ever-offer agreement", sc["ever_offer_agreement"]],
                ["offer membership agreement", sc["offer_membership_agreement"]],
                ["speaker-time agreement", sc["speaker_time_agreement"]],
                ["turn sequence agreement", sc["turn_sequence_agreement"]]]
                + [[label, (sc["time"] or {}).get(key)] for key, label in TIME_LABELS])
            summ = sc["stage_a_summary"]
            if summ:
                L += ["", "Stage A: " + ", ".join(f"{n} {_fmt(v)}" for n, v in summ.items()) + "."]
            if sc["codes"]:
                L.append("")
                L += _table(["code", "exact", "both valued"], [[cd, v["exact"], v["both_valued"]] for cd, v in sc["codes"].items()])
            L.append("")
    return "\n".join(L).rstrip() + "\n"


# ---- cli ------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--new-run", type=Path, required=True)
    # Earlier runs may have been moved under runs/old_version_runs/; prefer that location when present.
    moved = ROOT / "runs" / "old_version_runs" / "stage_b_structured_pilot5"
    ap.add_argument("--old-run", type=Path,
                    default=moved if moved.is_dir() else ROOT / "runs" / "stage_b_structured_pilot5")
    ap.add_argument("--video-ids", nargs="+")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or args.new_run
    old_root = args.old_run.resolve()
    if out_dir.resolve() == old_root or old_root in out_dir.resolve().parents:
        ap.error(f"output directory {out_dir} is the old run directory or inside it; the archive must not be written to")
    if args.video_ids:
        ids = list(args.video_ids)
    else:
        ids = sorted(p.name for p in args.new_run.iterdir() if p.is_dir() and (args.old_run / p.name).is_dir())
    books = load_panel_books(load_release())
    report = {"new_run": str(args.new_run), "old_run": str(args.old_run),
              "videos": {v: compare_video(args.new_run / v, args.old_run / v, books) for v in ids}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison_vs_structured.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "comparison_vs_structured.md").write_text(render_markdown(report))
    print(f"compared {len(ids)} video(s); wrote {out_dir / 'comparison_vs_structured.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
