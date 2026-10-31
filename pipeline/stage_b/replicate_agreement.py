"""Deterministic alignment of a primary and a replicate panel record of the same video.

The report measures model self-consistency under sampling. It is not a validity measure and
nothing here calls a model or reads a file. Ties are always broken by the lower id.

Alignment rests on who and in what order, not on when: the two passes of one video can agree
on content while one of them writes its times wrongly. Actors are mapped by label, then by
the name tokens two labels share, then by order of first speech.
Turns are paired by one order-preserving alignment of the two turn lists in which a pair
needs the same mapped speaker and is scored by function and by the wording of its summary;
decisions likewise by actor and kind. Pairs of different functions are reported as function
disagreements, so "unmatched" means the other pass lacks the turn. Exchanges are then matched
by how many paired turns they share. Time differences of the pairs are reported afterwards as
a diagnostic.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import asdict, dataclass
from fractions import Fraction

from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import (CONCEPT_CODE_OF, OTHER, PanelBooks, canonical_value, creates_offer_id, frame_of,
                                          makes_offer, other_name)

LABEL = "model self-consistency under sampling; not validity"
MISSING_KEY = "∅"
DERIVED_FRAME = {"PR-CONCEPT": "PR-FRAME(derived)", "RS-CONCEPT": "RS-FRAME(derived)"}  # concept code -> its row
ABSENT = "absent"  # stands in for the missing reason when a turn does not carry the code at all
NOT_COMPARED = ["excluded_spans", "source_complete", "claim_conflicts", "addressed", "exchange label",
                "stage_a moment"]
TURNS_NOTE = ("turns are paired by speaker, order and wording, not by time; recall and precision count pairs of the "
              "same function and are sensitive to segmentation; speaker_time_agreement compares who speaks second "
              "by second and so depends on the times")
LINKS_NOTE = "computed over links whose both endpoints are matched turns"
SETTINGS_NOTE = ("event_tolerance_s (applied to the starts of matched turns) and decision_tolerance_s are diagnostic "
                 "thresholds for the time block and the time_drift and time_outlier flags; they are not conditions "
                 "for pairing turns or decisions.")
SAME_FUNCTION_BASE, OTHER_FUNCTION_BASE, OTHER_FUNCTION_WEIGHT = 1.0, 0.4, 0.5
SUBSTANTIVE = "substantive"
BOUNDARY_JACCARD = 0.5  # a matched exchange pair sharing less than this is a boundary disagreement
_SCORE_UNIT = 1_000_000  # scores are summed as whole millionths so equal totals compare equal
VENTURE_FIELDS = ("primary_industry", "product_form", "customer_type", "founder_gender")
OTHER_NAMES = "other_concept_names"  # in the codes block: the names both passes gave where both said other


@dataclass(frozen=True)
class AlignmentSettings:
    event_tolerance_s: float = 10.0
    decision_tolerance_s: float = 30.0


def _ratio(num: int | float, den: int | float):
    return num / den if den else None


def _flag(kind: str, side: str, ref, detail: str) -> dict:
    return {"kind": kind, "pass": side, "ref": ref, "detail": detail}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _greedy(candidates: list[tuple]) -> list[tuple]:
    """candidates: (sort_key, p_id, r_id). Smallest key first, lower ids first, one to one."""
    used_p, used_r, out = set(), set(), []
    for _key, p, r in sorted(candidates):
        if p in used_p or r in used_r:
            continue
        used_p.add(p)
        used_r.add(r)
        out.append((p, r))
    return sorted(out)


def _best_alignment(n_p: int, n_r: int, score) -> list[tuple[int, int]]:
    """Order-preserving one-to-one alignment of two sequences that maximizes the total score.

    score(i, j) is the positive value of pairing primary item i with replicate item j, or None
    when they may not pair. Unpaired items cost nothing. Returns index pairs (i, j) increasing
    in both. Among alignments of equal total the one that pairs the earlier primary items first
    is returned.
    """
    value: list[list[int | None]] = [[None] * n_r for _ in range(n_p)]
    best = [[0] * (n_r + 1) for _ in range(n_p + 1)]  # best total over the suffixes i.., j..
    for i in range(n_p - 1, -1, -1):
        for j in range(n_r - 1, -1, -1):
            sc = score(i, j)
            top = max(best[i + 1][j], best[i][j + 1])
            if sc is not None:
                value[i][j] = round(sc * _SCORE_UNIT)
                top = max(top, best[i + 1][j + 1] + value[i][j])
            best[i][j] = top
    out = []
    i = j = 0
    while i < n_p and j < n_r:
        if value[i][j] is not None and best[i + 1][j + 1] + value[i][j] == best[i][j]:
            out.append((i, j))
            i, j = i + 1, j + 1
        elif best[i][j + 1] == best[i][j]:
            j += 1
        else:
            i += 1
    return out


def _tokens(*texts: str) -> frozenset[str]:
    return frozenset(w for text in texts for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3)


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _lcs(a: list, b: list) -> int:
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], (prev + 1 if x == y else max(row[j], row[j - 1]))
    return row[-1]


# ---- actors -----------------------------------------------------------------------------

def _norm_label(label: str) -> str:
    return " ".join(label.casefold().split())


def _by_first_speech(record: PanelRecord, role: str, skip: set[str]) -> list[str]:
    """Actors of a role not in `skip`, in order of their first turn; those who never speak last, by id."""
    first: dict[str, int] = {}
    for i, t in enumerate(record.turns):
        first.setdefault(t.speaker, i)
    ids = sorted({a.actor_id for a in record.actors if a.role == role and a.actor_id not in skip})
    return sorted(ids, key=lambda a: (a not in first, first.get(a, 0), a))


# words in a label that describe a role or a kind of person, not a name
GENERIC_LABEL_WORDS = frozenset({"the", "a", "an", "founder", "cofounder", "co", "investor", "shark", "host", "narrator",
                                 "man", "woman", "guy", "lady", "mr", "mrs", "ms", "dr"})


def _label_tokens(label: str) -> frozenset[str]:
    """The naming words of a label: "Kevin O'Leary" gives kevin and leary; "The investor" gives nothing."""
    return frozenset(w for w in re.findall(r"[a-z0-9]+", label.casefold())
                     if len(w) >= 2 and w not in GENERIC_LABEL_WORDS)


def _map_by_tokens(left_p: dict[str, frozenset], left_r: dict[str, frozenset]) -> list[tuple[str, str]]:
    """(primary id, replicate id) pairs of actors whose labels share a naming word, best pairs first.

    The score of a pair is the share of the shorter label's words found in the other, then the
    share of all their words they have in common. An actor whose best score is reached with two
    different partners (one "Mark", two candidates) is left out: that choice would be a guess.
    """
    scores: dict[tuple[str, str], tuple[Fraction, Fraction]] = {}
    for a, ta in left_p.items():
        for b, tb in left_r.items():
            shared = len(ta & tb)
            if shared:
                scores[(a, b)] = (Fraction(shared, min(len(ta), len(tb))), Fraction(shared, len(ta | tb)))
    ambiguous: set[tuple[int, str]] = set()
    for side in (0, 1):
        best: dict[str, list] = {}
        for pair, score in scores.items():
            best.setdefault(pair[side], []).append(score)
        for actor, found in best.items():
            if found.count(max(found)) > 1:
                ambiguous.add((side, actor))
    cands = [((-score[0], -score[1]), a, b) for (a, b), score in scores.items()
             if (0, a) not in ambiguous and (1, b) not in ambiguous]
    return _greedy(cands)


def _map_actors(primary: PanelRecord, replicate: PanelRecord) -> tuple[dict[str, str], dict[str, str]]:
    """(replicate actor id -> primary actor id, replicate id -> "label" | "label_tokens" | "order"), same role only.

    A label that exactly one actor of a role carries in each pass maps those two. Of the rest,
    actors whose labels share a naming word are paired (first name against full name). The
    actors of a role still left are paired in order of first speech: the first of them to
    speak in one pass with the first to speak in the other. Times play no part.
    """
    mapping: dict[str, str] = {}
    how: dict[str, str] = {}
    roles = sorted({a.role for a in primary.actors} | {a.role for a in replicate.actors})
    for role in roles:
        by_label: dict[str, tuple[list, list]] = {}
        for side, record in enumerate((primary, replicate)):
            for a in record.actors:
                if a.role == role:
                    by_label.setdefault(_norm_label(a.label), ([], []))[side].append(a.actor_id)
        for label in sorted(by_label):
            ps, rs = by_label[label]
            if len(ps) == 1 and len(rs) == 1:
                mapping[rs[0]] = ps[0]
                how[rs[0]] = "label"
    for role in roles:
        left_p = {a.actor_id: _label_tokens(a.label) for a in primary.actors
                  if a.role == role and a.actor_id not in set(mapping.values())}
        left_r = {a.actor_id: _label_tokens(a.label) for a in replicate.actors
                  if a.role == role and a.actor_id not in mapping}
        for p, r in _map_by_tokens(left_p, left_r):
            mapping[r] = p
            how[r] = "label_tokens"
    for role in roles:
        left_p = _by_first_speech(primary, role, set(mapping.values()))
        left_r = _by_first_speech(replicate, role, set(mapping))
        for p, r in zip(left_p, left_r):
            mapping[r] = p
            how[r] = "order"
    return mapping, how


def _mapped(actor: str, mapping: dict[str, str]) -> str:
    # an unmapped replicate actor gets an id no primary actor can carry, so it never matches
    return mapping.get(actor, f"~{actor}")


# ---- turns ------------------------------------------------------------------------------

def _speakers_at(turns, sec: int, mapping: dict[str, str] | None) -> frozenset:
    mid = sec + 0.5
    return frozenset(
        (_mapped(t.speaker, mapping) if mapping is not None else t.speaker)
        for t in turns if t.start_s <= mid < t.end_s
    )


def _speaker_time_agreement(primary, replicate, mapping):
    both = either = 0
    all_turns = list(primary.turns) + list(replicate.turns)
    if all_turns:
        lo = math.floor(min(t.start_s for t in all_turns))
        hi = math.ceil(max(t.end_s for t in all_turns))
        for sec in range(lo, hi):
            ps = _speakers_at(primary.turns, sec, None)
            rs = _speakers_at(replicate.turns, sec, mapping)
            if ps or rs:
                either += 1
                if ps and ps == rs:
                    both += 1
    return _ratio(both, either)


def _align_turns(primary, replicate, mapping):
    """Pairs are positions in the two turn lists, which are the order of the encounter.

    Returns (report, flags, same-function pairs, all pairs).
    """
    pt, rt = list(primary.turns), list(replicate.turns)
    p_speaker = [t.speaker for t in pt]
    r_speaker = [_mapped(t.speaker, mapping) for t in rt]
    p_tokens = [_tokens(t.summary) for t in pt]
    r_tokens = [_tokens(t.summary) for t in rt]

    def score(i, j):
        if p_speaker[i] != r_speaker[j]:
            return None
        sim = _jaccard(p_tokens[i], r_tokens[j])
        if pt[i].function == rt[j].function:
            return SAME_FUNCTION_BASE + sim
        return OTHER_FUNCTION_BASE + OTHER_FUNCTION_WEIGHT * sim

    pairs = _best_alignment(len(pt), len(rt), score)
    matched = [(i, j) for i, j in pairs if pt[i].function == rt[j].function]
    mismatched = [(i, j) for i, j in pairs if pt[i].function != rt[j].function]
    flags = []
    confusion: dict[str, int] = {}
    for i, j in mismatched:
        key = f"{pt[i].function}|{rt[j].function}"
        confusion[key] = confusion.get(key, 0) + 1
        flags.append(_flag("function_disagreement", "both", pt[i].turn_id,
                           f"turns {pt[i].turn_id}|{rt[j].turn_id}: same speaker and place in the sequence, "
                           f"primary {pt[i].function} vs replicate {rt[j].function}"))
    paired_p = {i for i, _ in pairs}
    paired_r = {j for _, j in pairs}
    un_p = [t.turn_id for i, t in enumerate(pt) if i not in paired_p]
    un_r = [t.turn_id for j, t in enumerate(rt) if j not in paired_r]
    flags += [_flag("unmatched_turn", "primary", i, "no replicate turn by the same speaker at this place in the sequence")
              for i in un_p]
    flags += [_flag("unmatched_turn", "replicate", i, "no primary turn by the same speaker at this place in the sequence")
              for i in un_r]
    functions = sorted({t.function for t in pt} | {t.function for t in rt})
    by_function = {
        f: {"primary": sum(t.function == f for t in pt), "replicate": sum(t.function == f for t in rt),
            "matched": sum(pt[i].function == f for i, _ in matched)}
        for f in functions
    }
    report = {
        "n_primary": len(pt), "n_replicate": len(rt),
        "matched": [[pt[i].turn_id, rt[j].turn_id] for i, j in matched],
        "function_mismatched": [[pt[i].turn_id, rt[j].turn_id] for i, j in mismatched],
        "function_confusion": dict(sorted(confusion.items())),
        "unmatched_primary": un_p, "unmatched_replicate": un_r,
        "recall": _ratio(len(matched), len(pt)), "precision": _ratio(len(matched), len(rt)),
        "recall_any_function": _ratio(len(pairs), len(pt)), "precision_any_function": _ratio(len(pairs), len(rt)),
        "by_function": by_function,
        "speaker_time_agreement": _speaker_time_agreement(primary, replicate, mapping),
        "sequence_agreement": _ratio(_lcs(p_speaker, r_speaker), max(len(pt), len(rt))),
        "note": TURNS_NOTE,
    }
    return report, flags, matched, pairs


# ---- exchanges --------------------------------------------------------------------------

def _first_by_id(exchanges) -> dict:
    out: dict = {}
    for e in exchanges:
        out.setdefault(e.exchange_id, e)
    return dict(sorted(out.items()))


def _count_into(table: dict[str, int], key: str) -> None:
    table[key] = table.get(key, 0) + 1


def _align_exchanges(primary, replicate, pairs):
    """Exchanges matched by the aligned turn pairs they share; `pairs` are positions in the turn lists."""
    pt, rt = primary.turns, replicate.turns
    pe, re_ = _first_by_id(primary.exchanges), _first_by_id(replicate.exchanges)
    sides = [(pt[i].exchange_id, rt[j].exchange_id) for i, j in pairs]
    in_p: dict[int, int] = {}
    in_r: dict[int, int] = {}
    shared: dict[tuple[int, int], int] = {}
    for a, b in sides:
        if a is not None:
            in_p[a] = in_p.get(a, 0) + 1
        if b is not None:
            in_r[b] = in_r.get(b, 0) + 1
        if a is not None and b is not None:
            shared[(a, b)] = shared.get((a, b), 0) + 1
    cands = []
    for (a, b), n in shared.items():
        if a in pe and b in re_:
            cands.append((-Fraction(n, in_p[a] + in_r[b] - n), a, b))
    jaccard = {(a, b): -key for key, a, b in cands}
    matched = _greedy(cands)

    flags = []
    un_p = [a for a in pe if a not in {p for p, _ in matched}]
    un_r = [b for b in re_ if b not in {r for _, r in matched}]
    flags += [_flag("unmatched_exchange", "primary", a, f"{pe[a].kind} exchange shares turns with no free replicate exchange")
              for a in un_p]
    flags += [_flag("unmatched_exchange", "replicate", b, f"{re_[b].kind} exchange shares turns with no free primary exchange")
              for b in un_r]

    kind_confusion: dict[str, int] = {}
    topic_confusion: dict[str, int] = {}
    kind_agree = topic_agree = topic_n = 0
    for a, b in matched:
        p, r = pe[a], re_[b]
        ref = f"exchanges {a}|{b}"
        if jaccard[(a, b)] < BOUNDARY_JACCARD:
            flags.append(_flag("exchange_boundary_disagreement", "both", a,
                               f"{ref}: share {float(jaccard[(a, b)]):.2f} of their aligned turns"))
        _count_into(kind_confusion, f"{p.kind}|{r.kind}")
        if p.kind == r.kind:
            kind_agree += 1
        else:
            flags.append(_flag("exchange_kind_disagreement", "both", a, f"{ref}: primary {p.kind} vs replicate {r.kind}"))
        if p.kind != SUBSTANTIVE or r.kind != SUBSTANTIVE:
            continue
        topic_n += 1
        _count_into(topic_confusion, f"{p.topic}|{r.topic}")
        if p.topic == r.topic:
            topic_agree += 1
        else:
            flags.append(_flag("exchange_topic_disagreement", "both", a, f"{ref}: primary {p.topic} vs replicate {r.topic}"))

    same = total = together_p = together_r = together_both = 0
    for x in range(len(sides)):
        for y in range(x + 1, len(sides)):
            (a1, b1), (a2, b2) = sides[x], sides[y]
            total += 1
            # a turn with no exchange is a group of its own, so two such turns are not together
            in_primary, in_replicate = a1 is not None and a1 == a2, b1 is not None and b1 == b2
            same += in_primary == in_replicate
            together_p += in_primary
            together_r += in_replicate
            together_both += in_primary and in_replicate
    # adjusted Rand index: agreement on which turn pairs are together, corrected for chance
    adjusted = None
    if total:
        chance = Fraction(together_p * together_r, total)
        ceiling = Fraction(together_p + together_r, 2) - chance
        if ceiling:
            adjusted = float((together_both - chance) / ceiling)
    kinds = sorted({e.kind for e in pe.values()} | {e.kind for e in re_.values()})
    values = [jaccard[m] for m in matched]
    report = {
        "n_primary": len(pe), "n_replicate": len(re_),
        "by_kind": {k: {"primary": sum(e.kind == k for e in pe.values()),
                        "replicate": sum(e.kind == k for e in re_.values())} for k in kinds},
        "matched": [[a, b, float(jaccard[(a, b)])] for a, b in matched],
        "matched_at_half": sum(v >= BOUNDARY_JACCARD for v in values),
        "mean_jaccard": float(sum(values) / len(values)) if values else None,
        "unmatched_primary": un_p, "unmatched_replicate": un_r,
        "kind_agree": kind_agree, "topic_agree": topic_agree, "topic_n": topic_n,
        "kind_confusion": dict(sorted(kind_confusion.items())),
        "topic_confusion": dict(sorted(topic_confusion.items())),
        "co_membership_agreement": _ratio(same, total),
        "adjusted_co_membership": adjusted,
    }
    return report, flags


# ---- codes ------------------------------------------------------------------------------

def _entry(books: PanelBooks, code_id: str):
    try:
        return books.panel.by_id(code_id)
    except KeyError:
        return None


def _values(turn, books: PanelBooks) -> dict[str, tuple[str | None, str]]:
    """code id -> (value in the codebook's spelling, missing reason); the first item of a code wins."""
    out: dict[str, tuple[str | None, str]] = {}
    for c in turn.codes:
        entry = _entry(books, c.code)
        out.setdefault(c.code, (canonical_value(entry, c.value) if entry is not None else c.value, c.missing))
    # the frame is not recorded: it is read off the concept and compared like a code of its own
    concept = CONCEPT_CODE_OF.get(turn.function)
    if concept in out:
        value, reason = out[concept]
        out[DERIVED_FRAME[concept]] = (frame_of(value), reason)
    return out


def _as_int(text: str):
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _align_codes(matched, pt, rt, books):
    """One row per code over the matched turns, and under OTHER_NAMES the [primary, replicate] names of
    every pair in which both passes say `other` on the same concept code. The names are listed for a
    reader to compare; they are not scored."""
    table: dict[str, dict] = {}
    names: list[list[str]] = []
    flags = []
    for p, r in matched:
        pv, rv = _values(pt[p], books), _values(rt[r], books)
        for code in CONCEPT_CODE_OF.values():
            if pv.get(code, (None,))[0] == OTHER and rv.get(code, (None,))[0] == OTHER:
                names.append([other_name(pt[p].other_concept), other_name(rt[r].other_concept)])
        for code in sorted(set(pv) | set(rv)):
            (a, a_reason), (b, b_reason) = pv.get(code, (None, ABSENT)), rv.get(code, (None, ABSENT))
            row = table.setdefault(code, {"n": 0, "both_valued": 0, "exact": 0, "within_one": None,
                                          "missing_agree": 0, "confusion": {}, "missing_confusion": {}})
            entry = _entry(books, code)
            if entry is not None and entry.type == "ordinal" and row["within_one"] is None:
                row["within_one"] = 0
            row["n"] += 1
            key = f"{MISSING_KEY if a is None else a}|{MISSING_KEY if b is None else b}"
            row["confusion"][key] = row["confusion"].get(key, 0) + 1
            if a is None and b is None:
                reasons = f"{a_reason}|{b_reason}"
                row["missing_confusion"][reasons] = row["missing_confusion"].get(reasons, 0) + 1
                if a_reason == b_reason:
                    row["missing_agree"] += 1
            elif a is not None and b is not None:
                row["missing_agree"] += 1
                row["both_valued"] += 1
                if a == b:
                    row["exact"] += 1
                if row["within_one"] is not None:
                    ia, ib = _as_int(a), _as_int(b)
                    if ia is not None and ib is not None and abs(ia - ib) <= 1:
                        row["within_one"] += 1
            if a != b and code not in DERIVED_FRAME.values():      # a frame disagreement is already flagged on its concept
                flags.append(_flag("code_disagreement", "both", code,
                                   f"turns {pt[p].turn_id}|{rt[r].turn_id}: {MISSING_KEY if a is None else a} vs "
                                   f"{MISSING_KEY if b is None else b}"))
    for row in table.values():
        row["confusion"] = dict(sorted(row["confusion"].items()))
        row["missing_confusion"] = dict(sorted(row["missing_confusion"].items()))
    table[OTHER_NAMES] = names
    return table, flags


# ---- links ------------------------------------------------------------------------------

def _align_links(matched, pt, rt):
    """responds_to compared over matched pairs; a link is (index of the pair, index of the pair it points to)."""
    p_index = {pt[p].turn_id: i for i, (p, _) in enumerate(matched)}
    r_index = {rt[r].turn_id: i for i, (_, r) in enumerate(matched)}
    p_links: set[tuple[int, int]] = set()
    r_links: set[tuple[int, int]] = set()
    dropped = 0
    flags = []
    for i, (p, r) in enumerate(matched):
        mine, theirs = set(), set()
        for t in pt[p].responds_to:
            if t in p_index:
                mine.add((i, p_index[t]))
            else:
                dropped += 1
        for t in rt[r].responds_to:
            if t in r_index:
                theirs.add((i, r_index[t]))
            else:
                dropped += 1
        p_links |= mine
        r_links |= theirs
        if mine != theirs:
            flags.append(_flag("link_disagreement", "both", pt[p].turn_id,
                               f"primary {sorted(t for _, t in mine)} vs replicate {sorted(t for _, t in theirs)} (matched-pair indices)"))
    shared = p_links & r_links
    report = {"primary_links": len(p_links), "replicate_links": len(r_links), "shared": len(shared),
              "recall": _ratio(len(shared), len(p_links)), "precision": _ratio(len(shared), len(r_links)),
              "dropped_unmatched_targets": dropped, "note": LINKS_NOTE}
    return report, flags


# ---- decisions --------------------------------------------------------------------------

def _align_decisions(primary, replicate, mapping, settings):
    pd = {d.decision_id: d for d in primary.decisions}
    rd = {d.decision_id: d for d in replicate.decisions}
    ps = sorted((d.time_s, d.decision_id, d.actor, d.kind) for d in primary.decisions)
    rs = sorted((d.time_s, d.decision_id, _mapped(d.actor, mapping), d.kind) for d in replicate.decisions)
    matched = sorted((ps[i][1], rs[j][1]) for i, j in _best_alignment(
        len(ps), len(rs), lambda i, j: 1.0 if ps[i][2:] == rs[j][2:] else None))
    mp = {p for p, _ in matched}
    mr = {r for _, r in matched}
    un_p = sorted(i for i in pd if i not in mp)
    un_r = sorted(i for i in rd if i not in mr)
    flags = [_flag("unmatched_decision", "primary", i, f"{pd[i].kind} by {pd[i].actor} has no replicate counterpart") for i in un_p]
    flags += [_flag("unmatched_decision", "replicate", i, f"{rd[i].kind} by {rd[i].actor} has no primary counterpart") for i in un_r]

    investors = sorted(a.actor_id for a in primary.actors if a.role == "investor")
    p_offer = {d.actor for d in primary.decisions if makes_offer(d.kind, d.offer_id)}
    r_offer = {_mapped(d.actor, mapping) for d in replicate.decisions if makes_offer(d.kind, d.offer_id)}
    ever = {a: {"primary": a in p_offer, "replicate": a in r_offer} for a in investors}
    agree = sum(v["primary"] == v["replicate"] for v in ever.values())
    for a, v in ever.items():
        if v["primary"] != v["replicate"]:
            flags.append(_flag("ever_offer_disagreement", "both", a,
                               f"offer, joins_offer or offer-making re_entry: primary {v['primary']}, replicate {v['replicate']}"))
    offers = []
    for p, r in matched:
        if not creates_offer_id(pd[p].kind, pd[p].offer_id):
            continue
        p_members = {d.actor for d in primary.decisions
                     if d.decision_id == p or (d.offer_id is not None and d.offer_id == pd[p].offer_id)}
        r_members = {_mapped(d.actor, mapping) for d in replicate.decisions
                     if d.decision_id == r or (d.offer_id is not None and d.offer_id == rd[r].offer_id)}
        offers.append({"primary_offer_id": pd[p].offer_id, "replicate_offer_id": rd[r].offer_id,
                       "primary_members": sorted(p_members), "replicate_members": sorted(r_members),
                       "agree": p_members == r_members})
        if p_members != r_members:
            flags.append(_flag("offer_membership_disagreement", "both", pd[p].offer_id,
                               f"decisions {p}|{r}: primary {sorted(p_members)} vs replicate {sorted(r_members)}"))
    report = {
        "matched": [list(m) for m in matched], "unmatched_primary": un_p, "unmatched_replicate": un_r,
        "offers": offers,
        "offer_membership_agreement": _ratio(sum(o["agree"] for o in offers), len(offers)),
        "ever_offer": ever, "ever_offer_agreement": _ratio(agree, len(ever)),
        "any_offer": {"primary": any(makes_offer(d.kind, d.offer_id) for d in primary.decisions),
                      "replicate": any(makes_offer(d.kind, d.offer_id) for d in replicate.decisions)},
    }
    return report, flags, matched, pd, rd


# ---- time as a diagnostic ---------------------------------------------------------------

def _time_block(matched_turns, pt, rt, matched_decisions, pd, rd, settings):
    """Start differences (replicate minus primary) of the pairs that order and content produced."""
    diffs = [(pt[p].turn_id, rt[r].start_s - pt[p].start_s) for p, r in matched_turns]
    values = [d for _, d in diffs]
    flags = []
    median_abs = statistics.median(abs(d) for d in values) if values else None
    if median_abs is not None and median_abs > settings.event_tolerance_s:
        flags.append(_flag("time_drift", "both", None,
                           f"median absolute start difference of matched turns is {median_abs}s"))
    late_turns = [(p, d) for p, d in diffs if abs(d) > settings.event_tolerance_s]
    flags += [_flag("time_outlier", "both", p, f"turn starts differ by {d}s") for p, d in late_turns]
    late_decisions = [(p, rd[r].time_s - pd[p].time_s) for p, r in matched_decisions
                      if abs(rd[r].time_s - pd[p].time_s) > settings.decision_tolerance_s]
    flags += [_flag("time_outlier", "both", p, f"decision times differ by {d}s") for p, d in late_decisions]
    report = {
        "matched_turn_pairs": len(values),
        "median_start_diff_s": statistics.median(values) if values else None,
        "median_abs_start_diff_s": median_abs,
        "max_abs_start_diff_s": max(abs(d) for d in values) if values else None,
        "turn_pairs_beyond_tolerance": len(late_turns),
        "decision_pairs_beyond_tolerance": len(late_decisions),
    }
    return report, flags


# ---- stage A, venture, prepared ---------------------------------------------------------

def _stage_a_values(record: PanelRecord) -> dict[str, int | None]:
    out: dict[str, int | None] = {}
    for item in record.stage_a:
        out.setdefault(item.code, item.value)
    return out


def _align_stage_a(primary, replicate, books):
    pv, rv = _stage_a_values(primary), _stage_a_values(replicate)
    ids = list(books.stage_a_ids()) + sorted((set(pv) | set(rv)) - set(books.stage_a_ids()))
    out: dict = {}
    flags = []
    diffs = []
    for code in ids:
        if code not in pv and code not in rv:
            continue
        a, b = pv.get(code), rv.get(code)
        diff = abs(a - b) if a is not None and b is not None else None
        out[code] = {"primary": a, "replicate": b, "abs_diff": diff}
        if diff is not None:
            diffs.append(diff)
        if (diff is not None and diff >= 2) or ((a is None) != (b is None)):
            flags.append(_flag("stage_a_disagreement", "both", code, f"primary {a}, replicate {b}"))
    out["_summary"] = {"both_rated": len(diffs), "exact": sum(d == 0 for d in diffs),
                       "within_one": sum(d <= 1 for d in diffs),
                       "mean_abs_diff": _ratio(sum(diffs), len(diffs))}
    return out, flags


def _align_venture(primary, replicate):
    out, flags = {}, []
    for field in VENTURE_FIELDS:
        a, b = getattr(primary.venture, field), getattr(replicate.venture, field)
        out[field] = {"primary": a, "replicate": b, "agree": a == b}
        if a != b:
            flags.append(_flag("venture_disagreement", "both", field, f"primary {a!r}, replicate {b!r}"))
    # the ask: two numbers, each agreeing when equal (both absent counts as agreeing)
    out["ask"] = {"amount_agree": primary.venture.ask_amount_usd == replicate.venture.ask_amount_usd,
                  "equity_agree": primary.venture.ask_equity_pct == replicate.venture.ask_equity_pct}
    return out, flags


def _union(spans) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted((s.start_s, s.end_s) for s in spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _prepared_values(record: PanelRecord, books: PanelBooks) -> dict[str, str | None]:
    """code id -> value in the codebook's spelling, surrounding whitespace stripped (" 4" is 4, also for a count); the
    first item of a code wins."""
    out: dict[str, str | None] = {}
    for c in record.prepared:
        entry = _entry(books, c.code)
        value = c.value.strip() if isinstance(c.value, str) else c.value
        out.setdefault(c.code, canonical_value(entry, value) if entry is not None else value)
    return out


def _prepared(primary, replicate, books):
    """The overlap of the two prepared windows, and agreement on the prepared presentation codes: n codes valued in
    both passes, agree of them with the same value, within_one of the ordinal ones at most one point apart."""
    pu, ru = _union(primary.prepared_spans), _union(replicate.prepared_spans)
    ps = sum(e - s for s, e in pu)
    rs = sum(e - s for s, e in ru)
    inter = sum(_overlap(a0, a1, b0, b1) for a0, a1 in pu for b0, b1 in ru)
    pv, rv = _prepared_values(primary, books), _prepared_values(replicate, books)
    n = agree = within_one = 0
    for entry in books.prepared_codes():
        a, b = pv.get(entry.id), rv.get(entry.id)
        if a is None or b is None:
            continue
        n += 1
        agree += a == b
        if entry.type == "ordinal":
            ia, ib = _as_int(a), _as_int(b)
            within_one += ia is not None and ib is not None and abs(ia - ib) <= 1
    return {"primary_s": ps, "replicate_s": rs, "iou": _ratio(inter, ps + rs - inter),
            "n": n, "agree": agree, "within_one": within_one}


# ---- entry point ------------------------------------------------------------------------

def align(primary: PanelRecord, replicate: PanelRecord, books: PanelBooks,
          settings: AlignmentSettings = AlignmentSettings()) -> dict:
    # a label that differs from the codebook only in case is the same label
    primary, replicate = books.canonical_record(primary), books.canonical_record(replicate)
    mapping, mapped_by = _map_actors(primary, replicate)
    mapped_p = set(mapping.values())
    un_p = sorted(a.actor_id for a in primary.actors if a.actor_id not in mapped_p)
    un_r = sorted(a.actor_id for a in replicate.actors if a.actor_id not in mapping)
    flags = [_flag("unmapped_actor", "primary", a, "no replicate actor maps to this actor") for a in un_p]
    flags += [_flag("unmapped_actor", "replicate", a, "maps to no primary actor; left out of comparisons") for a in un_r]
    flags += [_flag("actor_mapped_by_order", "both", p,
                    f"actors {p}|{r}: no shared label or name, paired by order of first speech; this mapping is a guess")
              for p, r in sorted((p, r) for r, p in mapping.items() if mapped_by[r] == "order")]

    pt, rt = primary.turns, replicate.turns
    turns, f, matched, pairs = _align_turns(primary, replicate, mapping)
    flags += f
    exchanges, f = _align_exchanges(primary, replicate, pairs)
    flags += f
    codes, f = _align_codes(matched, pt, rt, books)
    flags += f
    links, f = _align_links(matched, pt, rt)
    flags += f
    decisions, f, matched_decisions, pd, rd = _align_decisions(primary, replicate, mapping, settings)
    flags += f
    time_block, time_flags = _time_block(matched, pt, rt, matched_decisions, pd, rd, settings)
    stage_a, f = _align_stage_a(primary, replicate, books)
    flags += f
    venture, f = _align_venture(primary, replicate)
    flags += f
    flags += time_flags

    return {
        "label": LABEL,
        "settings": asdict(settings),
        "settings_note": SETTINGS_NOTE,
        "actors": {"mapping": dict(sorted(mapping.items())), "mapped_by": dict(sorted(mapped_by.items())),
                   "unmapped_primary": un_p, "unmapped_replicate": un_r},
        "turns": turns, "exchanges": exchanges, "codes": codes, "links": links, "decisions": decisions,
        "stage_a": stage_a, "venture": venture, "prepared": _prepared(primary, replicate, books),
        "time": time_block,
        "not_compared": list(NOT_COMPARED),
        "flags": flags,
    }
