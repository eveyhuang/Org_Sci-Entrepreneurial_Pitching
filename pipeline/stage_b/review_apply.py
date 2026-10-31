"""A coder's filled sheet turned into corrections, a reviewed record and human-model agreement.

The model's passes are inputs only: corrections are applied to a copy of the primary record, and an
import either applies every row or nothing. No I/O: rows and records in, dicts out.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from pipeline.reliability import agreement as ag
from pipeline.reliability.agreement import (
    MIN_DEFINED_SHARE, Level, alpha_units, bootstrap_alpha_detail, stratified_alpha,
)
from pipeline.schemas.stage_b_panel import PanelRecord, clock_to_seconds
from pipeline.stage_b.panel_books import PanelBooks, is_other
from pipeline.stage_b.replicate_agreement import AlignmentSettings
from pipeline.stage_b.review_items import (
    ASK, ASK_RE, CUT, EVID_BANDS, FIELD_CLASSES, NOT_AN_EXCHANGE, OUTCOME, OUTCOME_VALUES, PAIR, PRESENCE_VALUES, PRESENT,
    UNIT_PRESENCE, _clock, as_band, band_rank, banded, canonical_full, is_banded, field_class, is_cut, is_outcome, main_path, normalize_path, parse_path, presence_ref,
    read_value, set_decision, set_function, set_value,
)
from pipeline.stage_b.review_labels import entry_for
from pipeline.stage_b.review_select import _FREE_TEXT

if TYPE_CHECKING:       # review_report imports this module
    from pipeline.stage_b.review_report import Reporting

RESOLUTIONS = ("confirm_primary", "confirm_replicate", "correct", "unclear", "structural")
WRITES = ("confirm_replicate", "correct")           # resolutions that put a value into the record
# set by parse_sheet, not by a coder: a pitch path (any_offer, first_decision) judged otherwise than the record gives
OUTCOME_OVERRIDE = "outcome_override"
DEFINITE = ("confirm_primary", *WRITES, OUTCOME_OVERRIDE)   # resolutions that say what the value is
LOGGED = (*RESOLUTIONS, OUTCOME_OVERRIDE)           # every resolution a correction can carry
WINDOW_EDGE_TOLERANCE_S = 2.0       # each edge of the prepared window, for two windows to count as the same
CUT_TOLERANCE_S = 5.0               # two clocks of the first decision within this are one moment
ASK_AMOUNT_SHARE = 0.01             # two asks agree when the amounts are within 1% and the equity is the same
MIN_CLUSTER_VIDEOS = 5              # videos from which an interval resamples videos instead of rows
N_BOOT = 1999
SEED = 0
CLUSTER_BOOTSTRAP = f"cluster bootstrap over videos ({N_BOOT:,} draws, seed {SEED})"
WILSON_METHOD = "Wilson; treats rows as independent; optimistic"
AGREEMENT_NOTE = (
    "Each figure is the share of reviewed items on which two sources give the same value, with the number of items "
    "behind it. First-decision agreement uses the same reference turn; missing mappings are undetermined. "
    "Other clock values are the same within the tolerance the selection flags by (a decision time within "
    "the agreement settings' decision tolerance, each edge of the prepared window within two seconds; see "
    "tolerances); ordinal fields (the codes and items whose codebook entry is ordinal) are compared exactly, and "
    "within_one gives beside it the share within one scale point. On a flag-driven sheet the sampled rows are a seeded draw, proportional to each "
    "field class's share, from the fields that nothing flagged and whose unit both passes recorded: they are the "
    "unbiased estimate of human-model agreement on such fields, not on flagged fields nor on units one pass lacks. "
    "On an exchange-level sheet every row is a random draw from the populations the manifest records. The flagged "
    "rows were chosen because something looked wrong and understate agreement. human_vs_replicate and primary_vs_replicate cover only the rows that have a replicate "
    "value; primary_vs_replicate counts a field once however many coders judged it. unit_presence rows (a turn, exchange or decision only one "
    "pass recorded) enter none of these figures; by_field_class.unit_presence gives the share of such units the human "
    "judged present, separately for those only the primary and those only the replicate recorded.")
LEFT_OUT_NOTE = (
    "The export left some flagged rows off the sheet (flagged_left_out, by field class and by reason), so every "
    "flagged-row figure describes the reviewed subset only; weight by these counts before reading it as a rate over "
    "all flagged rows. The sampled rows are not affected.")
NO_EXCHANGE_NOTE = ("judged not an exchange: these turns are no line of questioning; nothing was removed from the "
                    "record")
OVERRIDE_NOTE = "the pitch's outcome judged from the footage; the record's decisions were not edited"
CUT_NOTE = "the first decision's clock judged from the footage; the record's decisions were not edited"
NOT_PRESENT_NOTE = "judged absent: the footage does not show this unit; nothing was removed from the record"
MISSED_NOTE = "judged present: a unit the primary missed; nothing was added to the record"


@dataclass(frozen=True)
class Correction:
    path: str
    human_value: str | None      # None with resolution "confirm_primary" means the primary value stands
    resolution: str              # one of RESOLUTIONS
    note: str
    reviewer: str
    role: str = "adjudicator"    # "coder" (logged and measured, never applied to the record) or "adjudicator"


@dataclass(frozen=True)
class Tolerances:
    """How far apart two clock values may be and still be one value: the tolerances selection flags by, and that of
    the first decision's clock (a human judgment of one moment, which no pass comparison sets)."""
    decision_s: float = AlignmentSettings().decision_tolerance_s
    window_edge_s: float = WINDOW_EDGE_TOLERANCE_S
    cut_s: float = CUT_TOLERANCE_S

    @classmethod
    def of(cls, agreement: dict | None) -> "Tolerances":
        """From a stored agreement report (its "settings"); the defaults where it gives none."""
        settings = (agreement or {}).get("settings") or {}
        return cls(decision_s=float(settings.get("decision_tolerance_s", cls().decision_s)))


class ReviewError(ValueError):
    """An import that cannot be applied; `problems` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        count = f"{len(self.problems)} problem" + ("" if len(self.problems) == 1 else "s")
        super().__init__(f"{count}: " + "; ".join(self.problems))


def _text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _fold(value) -> str | None:
    """A value for comparison: case, surrounding blanks and the blanks inside a comma list do not count."""
    text = _text(value)
    return None if text is None else ",".join(part.strip() for part in text.split(",")).casefold()


def same_value(a, b) -> bool:
    return _fold(a) == _fold(b)


def _count_code(ref, books: PanelBooks | None) -> bool:
    """Whether a path names a code whose codebook entry is a count (with the books; never without them)."""
    if books is None:
        return False
    code = ref.unit_id if ref.unit == "prep_pres" else ref.sub if ref.unit == "turn" and ref.field == "code" else None
    entry = next((e for e in books.panel.entries if e.id == code), None) if code else None
    return entry is not None and entry.type == "count"


def value_kind(path: str | None, books: PanelBooks | None = None) -> str:
    """How the values at a path are compared, from the path: "clock" (a decision time, the first decision),
    "window" (the prepared window), "number" (the ask; with `books`, a count code), "ordinal" or "exact". `agrees`
    reads clock, window and number from it. The ordinal rule here names code ids (Stage A, PREP-*, PR-SEV) and is
    the fallback for log lines written before an import recorded each line's `level` from the codebooks
    (`level_of`); nothing else should rely on it. `prep_pres:PRES-EVID` is "ordinal": a count the record keeps, but
    answered and compared in its four ordered bands (review_items.EVID_BANDS)."""
    if path == "prepared_window":
        return "window"
    if is_banded(path):
        return "ordinal"
    try:
        ref = parse_path(path or "")
    except ValueError:
        return "exact"
    if (ref.unit == "decision" and ref.field == "time") or is_cut(path):
        return "clock"
    if (ref.unit == "venture" and ref.unit_id == ASK) or _count_code(ref, books):
        return "number"
    if ref.unit == "stage_a":
        return "ordinal"
    if ref.unit == "turn" and ref.field == "code" and (ref.sub == "PR-SEV" or ref.sub.startswith("PREP-")):
        return "ordinal"
    return "exact"


def _edges(window: str) -> list[float]:
    return [clock_to_seconds(end) for span in window.split(";") for end in span.split("-")]


def _ask(text: str) -> tuple[int, float] | None:
    found = ASK_RE.match(text)
    try:
        return (int(found.group(1).replace(",", "")), float(found.group(2))) if found else None
    except ValueError:
        return None


def _numbers_agree(a: str, b: str) -> bool | None:
    """Two asks: the amounts within 1% of the larger and the equity the same; two numbers: equal. None when the
    values are not both of that form (a `missing:<reason>`, say): they are then compared as written."""
    if ";" in a or ";" in b:
        x, y = _ask(a), _ask(b)
        if x is None or y is None:
            return None
        return abs(x[0] - y[0]) <= ASK_AMOUNT_SHARE * max(x[0], y[0]) and x[1] == y[1]
    try:
        return float(a) == float(b)
    except ValueError:
        return None


def agrees(path: str | None, a, b, tolerances: Tolerances | None = None, bands: bool = True) -> bool:
    """Whether two values at a path are the same value: clocks within tolerance (the first decision within
    `cut_s`), an ask by amount within 1% and equity exactly, a count of checkable claims (`prep_pres:PRES-EVID`) by
    its band (a coder's band against the model's count; two counts in one band agree), everything else as written.
    `bands=False` compares that count as written: what an import writes into a record is the count, not its band."""
    if bands and is_banded(path):
        a, b = banded(path, a), banded(path, b)
    kind = value_kind(path)
    a_text, b_text = _text(a), _text(b)
    if kind == "number" and a_text is not None and b_text is not None:
        same = _numbers_agree(a_text, b_text)
        if same is not None:
            return same
    if kind in ("clock", "window") and a_text is not None and b_text is not None:
        tolerances = tolerances or Tolerances()
        try:
            if kind == "clock":
                limit = tolerances.cut_s if is_cut(path) else tolerances.decision_s
                return abs(clock_to_seconds(a_text) - clock_to_seconds(b_text)) <= limit
            ends_a, ends_b = _edges(a_text), _edges(b_text)
            return len(ends_a) == len(ends_b) and all(
                abs(x - y) <= tolerances.window_edge_s for x, y in zip(ends_a, ends_b))
        except ValueError:
            pass        # not clocks after all: compared as written
    return same_value(a, b)


def within_one(a, b, path: str | None = None) -> bool:
    """Two ordinal values at most one scale point apart; two empty values agree, an empty and a rated one do not.
    At a banded `path` (`prep_pres:PRES-EVID`) the points are the bands."""
    a_text, b_text = _text(band_rank(path, a)), _text(band_rank(path, b))
    try:
        return abs(int(a_text) - int(b_text)) <= 1
    except (TypeError, ValueError):
        return same_value(a, b)


SUMMARY_SHOWN = "[summary shown]"      # what the coding page adds to a note when the turn summaries were opened
# what the coding page writes at the start of a note when the coder ticked "not sure" beside a best guess (plan C Task
# 7f, 2026-10-21): the row is a review like any other and is imported as one; agreement is reported with every row
# and again without these rows. The "I can't tell from the footage" tick keeps its meaning: no value, the row missing.
UNSURE = "[unsure]"


def is_unsure(line: dict) -> bool:
    """Whether a log line (or a sheet row) carries the coder's "not sure" tick in its note."""
    return UNSURE in str(line.get("note") or "")


def unnamed_others(rows: list[dict]) -> list[str]:
    """A warning per row answered `other` whose note names no matter (the coding page asks for a few words naming
    it, as the model gives other_concept). The import keeps such a row as it keeps any other: a warning, not a
    refusal, so older sheets still import. A row with a ticked box (`unclear`, `structural`) is not counted."""
    out = []
    for index, row in enumerate(rows, 1):
        if not is_other(_text(row.get("human_value"))):
            continue
        if (_text(row.get("resolution")) or "").casefold() in ("unclear", "structural"):
            continue
        note = (_text(row.get("note")) or "").replace(SUMMARY_SHOWN, "").replace(UNSURE, "").strip()
        if not note:
            out.append(f"row {_text(row.get('row')) or index}: {row.get('path')} is `other` but the note does not name "
                       "the matter")
    return out


def parse_sheet(rows: list[dict], reviewer: str, tolerances: Tolerances | None = None,
                role: str = "adjudicator") -> tuple[list[Correction], list[str]]:
    """(corrections, problems) of a filled sheet; each row carries path, primary, replicate, human_value, resolution, note.

    A row is reviewed when human_value or resolution is filled. With no resolution, a value equal to
    the primary confirms it and any other value corrects it; a clock value within `tolerances` of the
    primary is the same value, so it confirms and is no correction. `correct` with an empty value
    corrects the field to empty. A problem names the row (its `row` cell, else its position). Paths
    are normalised (`turn:0005/...` is `turn:5/...`), so one field cannot be reviewed twice under two
    spellings.

    A unit_presence row takes `present` or `absent` and never writes: the model's own answer
    confirms the primary, the other one becomes a `structural` correction that carries the judgment.
    An exchange topic answered `not_an_exchange` is `structural` too. The pitch rows are judged and
    never written: the outcome (pitch:any_offer) takes `yes`, `no` or `unknown`, the first decision
    (pitch:first_decision) a clock (or empty for none); the record's own value (a clock within
    `tolerances.cut_s`) confirms it, another one becomes an `outcome_override`, which import stores
    beside the record.
    """
    corrections: list[Correction] = []
    problems: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        where = f"row {_text(row.get('row')) or index}"
        human, resolution, note = _text(row.get("human_value")), _text(row.get("resolution")), _text(row.get("note"))
        if human is None and resolution is None:
            if note is not None:
                problems.append(f"{where}: a note without human_value or resolution is not a review; set resolution "
                                "to unclear or structural to record it")
            continue
        path = _text(row.get("path"))
        if path is None:
            problems.append(f"{where}: no path")
            continue
        path = normalize_path(path)
        if path in seen:
            problems.append(f"{where}: {path} is reviewed more than once")
            continue
        primary, replicate = _text(row.get("primary")), _text(row.get("replicate"))
        presence = presence_ref(path)
        if presence is not None:    # what the passes say of a unit follows from its path, whatever the cells hold
            primary, replicate = presence.primary_value, presence.replicate_value
        if resolution is None:
            # a count is compared as written here: whether to write it, not whether it agrees in its band
            resolution = "confirm_primary" if agrees(path, human, primary, tolerances, bands=False) else "correct"
        resolution = resolution.casefold()
        if resolution not in RESOLUTIONS:
            problems.append(f"{where}: unknown resolution {resolution!r}; use one of {', '.join(RESOLUTIONS)}")
            continue
        if resolution == "confirm_primary":
            if human is not None and not agrees(path, human, primary, tolerances, bands=False):
                problems.append(f"{where}: confirm_primary with human_value {human!r}, but the primary value is "
                                f"{primary!r}")
                continue
            human = None
        elif resolution == "confirm_replicate":
            if replicate is None:
                problems.append(f"{where}: confirm_replicate but the sheet has no replicate value for {path}")
                continue
            if human is not None and not agrees(path, human, replicate, tolerances, bands=False):
                problems.append(f"{where}: confirm_replicate with human_value {human!r}, but the replicate value is "
                                f"{replicate!r}")
                continue
            human = replicate
        if is_outcome(path) and (resolution in DEFINITE or human is not None):
            judged = (primary or "" if resolution == "confirm_primary" else human or "").casefold()
            if judged not in OUTCOME_VALUES:
                problems.append(f"{where}: {path} takes {', '.join(OUTCOME_VALUES)}, not {human!r}")
                continue
            if resolution not in DEFINITE:
                human = judged
            elif judged == (primary or "").casefold():
                human, resolution = None, "confirm_primary"
            else:
                human, resolution, note = judged, OUTCOME_OVERRIDE, note or OVERRIDE_NOTE
        if is_cut(path) and resolution != "confirm_primary" and (resolution in DEFINITE or human is not None):
            if human is not None:
                try:
                    clock_to_seconds(human)
                except ValueError:
                    problems.append(f"{where}: {path} takes a clock time (M:SS), not {human!r}")
                    continue
            if resolution in DEFINITE:
                if agrees(path, human, primary, tolerances):
                    human, resolution = None, "confirm_primary"
                else:
                    resolution, note = OUTCOME_OVERRIDE, note or CUT_NOTE
        if (resolution in WRITES and (human or "").casefold() == NOT_AN_EXCHANGE
                and path.startswith("exchange:") and path.endswith("/topic")):
            human, resolution, note = NOT_AN_EXCHANGE, "structural", note or NO_EXCHANGE_NOTE
        if presence is not None and (resolution in DEFINITE or human is not None):
            judged = (primary if resolution == "confirm_primary" else human or "").casefold()
            if judged not in PRESENCE_VALUES:
                problems.append(f"{where}: {path} takes {' or '.join(PRESENCE_VALUES)}, not {human!r}")
                continue
            if resolution in DEFINITE:
                if judged == primary:
                    human, resolution = None, "confirm_primary"
                else:
                    human, resolution = judged, "structural"
                    note = note or (MISSED_NOTE if presence.replicate_only else NOT_PRESENT_NOTE)
            else:
                human = judged
        seen.add(path)
        corrections.append(Correction(path, human, resolution, note or "", reviewer, role))
    return corrections, problems


EXCHANGE_STAGE, TURN_STAGE, DECISION_STAGE, REST_STAGE = range(4)


def _stage(path: str) -> tuple[int, int | None]:
    """When a correction is applied: exchange fields, then turns (grouped), then decisions (grouped), then the
    rest."""
    ref = parse_path(path)
    if ref.unit == "exchange":
        return EXCHANGE_STAGE, None
    if ref.unit == "turn":
        return TURN_STAGE, ref.unit_id
    if ref.unit == "decision":
        return DECISION_STAGE, ref.unit_id
    return REST_STAGE, None


def _off_band_but_readable(value) -> bool:
    """A coder's answer at the banded path that is no band but can still be read: a whole-number count (banded for
    agreement, and counted by the import as off the sheet's scale) or `missing:<reason>`."""
    text = (value or "").strip()
    return bool(re.fullmatch(r"\d+", text)) or text.casefold().startswith("missing:")


def counted_not_banded(corrections: list[Correction]) -> int:
    """How many of a coder's answers at the banded path (PRES-EVID) are a count rather than a band: accepted and
    banded for agreement, but off the coder sheet's scale, so the import reports them."""
    return sum(1 for c in corrections if c.role == "coder" and is_banded(c.path) and c.resolution in WRITES
               and re.fullmatch(r"\d+", (c.human_value or "").strip()) is not None)


def apply_corrections(primary: PanelRecord, corrections: list[Correction], books: PanelBooks,
                      duration_s: float | None = None) -> tuple[dict, list[dict]]:
    """(reviewed record dict, log entries) of a copy of `primary` with the corrections applied.

    Exchange edits go first, then each turn's function together with the codes the sheet supplies
    for that turn, then everything else, so that dependent edits are valid. `unclear` and
    `structural` change nothing and are logged; so is a unit_presence judgment, whose entry reads
    old = what the primary says of the unit, new = what the human judged (None for no judgment), and
    a pitch judgment (the outcome, the first decision), whose entry reads old = what the record's
    decisions give, new = the human judgment. A turn's codes that its new function does not take
    are removed, each with a log entry of its own. Any value the record cannot take raises
    ReviewError with every problem collected; nothing is returned then.

    The corrections of one decision (kind, actor, time, offer id) are written together, so a kind
    and the actor or offer id it needs can change in one import. `duration_s` is the video's length,
    against which clock values (the first decision's among them) are checked.

    Log entries are {"path", "old", "new", "resolution", "note", "reviewer"} in the sheet's order;
    old is the value before this import (None for a field that was not there).
    """
    record = primary.model_dump(mode="json")
    before = canonical_full(primary, books).model_dump(mode="json")
    problems: list[str] = []
    entries: dict[int, list[dict]] = {}
    plan: list[tuple[tuple[int, int | None], int, Correction]] = []

    def old_value(path: str) -> str | None:
        try:
            return read_value(before, path)
        except ValueError:
            return None

    def entry(c: Correction, new: str | None) -> dict:
        return {"path": c.path, "old": old_value(c.path), "new": new, "resolution": c.resolution, "note": c.note,
                "reviewer": c.reviewer}

    def presence_entry(c: Correction) -> dict:
        """A unit_presence row never writes: old is what the primary says of the unit, new the human judgment."""
        presence = presence_ref(c.path)
        if c.resolution in WRITES:
            raise ValueError(f"{c.path}: a unit is judged {' or '.join(PRESENCE_VALUES)}; it cannot be written")
        if not presence.replicate_only:
            read_value(before, main_path(presence.unit, presence.unit_id))      # raises when the unit is not there
        old = presence.primary_value
        if c.resolution == "confirm_primary":
            judged = old
        else:       # no judgment (unclear, or a structural note without a value) is logged as None
            judged = c.human_value if c.resolution == "structural" and c.human_value in PRESENCE_VALUES else None
        return {**entry(c, judged), "old": old}

    def pitch_entry(c: Correction) -> dict:
        """A pitch row never writes: old is what the record's decisions give, new the human judgment."""
        if c.resolution in WRITES:
            what = f"judged {', '.join(OUTCOME_VALUES)}" if is_outcome(c.path) else "a judged clock"
            raise ValueError(f"{c.path}: the pitch's {'outcome' if is_outcome(c.path) else 'first decision'} is "
                             f"{what}; it cannot be written")
        old = read_value(before, c.path)
        if c.resolution == "confirm_primary":
            judged = old
        elif c.resolution == OUTCOME_OVERRIDE:
            judged = c.human_value
            if is_outcome(c.path) and judged not in OUTCOME_VALUES:
                raise ValueError(f"{c.path}: takes {', '.join(OUTCOME_VALUES)}, not {judged!r}")
            if is_cut(c.path) and judged is not None:
                _clock(judged, duration_s)          # a moment of this video, written M:SS on a short one
        else:
            judged = None
        return {**entry(c, judged), "old": old}

    for index, c in enumerate(corrections):
        if c.resolution not in LOGGED:
            problems.append(f"{c.path}: unknown resolution {c.resolution!r}")
        elif is_outcome(c.path) or is_cut(c.path):
            try:
                entries[index] = [pitch_entry(c)]
            except ValueError as exc:
                text = str(exc)
                problems.append(text if text.startswith(f"{c.path}:") else f"{c.path}: {text}")
        elif c.resolution == OUTCOME_OVERRIDE:
            problems.append(f"{c.path}: {OUTCOME_OVERRIDE} applies to pitch:any_offer and pitch:first_decision "
                            "only")
        elif presence_ref(c.path) is not None:
            try:
                entries[index] = [presence_entry(c)]
            except ValueError as exc:
                problems.append(str(exc))
        elif c.resolution in WRITES and is_banded(c.path) and as_band(c.human_value):
            if c.role == "coder":       # a coder's band is measured against the model's count, never written
                entries[index] = [entry(c, as_band(c.human_value))]
            else:
                problems.append(f"{c.path}: the record holds a count of checkable claims; {c.human_value.strip()!r} "
                                f"is a band of a coder's sheet ({', '.join(EVID_BANDS)}): write the count")
        elif (c.resolution in WRITES and is_banded(c.path) and c.role == "coder"
              and not _off_band_but_readable(c.human_value)):
            problems.append(f"{c.path}: takes a band: {', '.join(EVID_BANDS)} (not {c.human_value!r})")
        elif c.resolution in WRITES:
            try:
                plan.append((_stage(c.path), index, c))
            except ValueError as exc:
                problems.append(str(exc))
        elif c.resolution == "confirm_primary":
            try:
                entries[index] = [entry(c, read_value(before, c.path))]
            except ValueError as exc:
                problems.append(str(exc))
        else:
            entries[index] = [entry(c, old_value(c.path))]

    def write(index: int, c: Correction) -> None:
        try:
            set_value(record, c.path, c.human_value, books, duration_s)
            entries[index] = [entry(c, read_value(record, c.path))]
        except ValueError as exc:
            problems.append(str(exc))

    plan.sort(key=lambda item: (item[0][0], item[1]))
    turns: dict[int, list[tuple[int, Correction]]] = {}
    decisions: dict[int, list[tuple[int, Correction]]] = {}
    for (stage, unit_id), index, c in plan:
        if stage == TURN_STAGE:
            turns.setdefault(unit_id, []).append((index, c))
        elif stage == DECISION_STAGE:
            decisions.setdefault(unit_id, []).append((index, c))
        elif stage < TURN_STAGE:
            write(index, c)
    for turn_id, group in turns.items():
        function = next(((i, c) for i, c in group if parse_path(c.path).field == "function"), None)
        codes = [(i, c) for i, c in group if parse_path(c.path).field == "code"]
        if function is not None:
            index, c = function
            try:
                removed = set_function(record, c.path, c.human_value,
                                       {parse_path(x.path).sub: x.human_value for _, x in codes}, books)
            except ValueError as exc:
                problems.append(str(exc))
                continue        # the turn's other rows depend on the function the coder gave
            new_function = read_value(record, c.path)
            entries[index] = [entry(c, new_function)] + [
                {"path": f"turn:{turn_id}/code:{code}", "old": old_value(f"turn:{turn_id}/code:{code}"), "new": None,
                 "resolution": "correct", "note": f"removed: {code} does not apply to function {new_function}",
                 "reviewer": c.reviewer}
                for code in removed]
            for i, x in codes:
                entries[i] = [entry(x, read_value(record, x.path))]
        for i, x in group:
            if i not in entries:        # everything the function change did not already take with it
                write(i, x)
    for decision_id, group in decisions.items():
        if len(group) == 1:
            write(*group[0])
            continue
        try:        # several fields of one decision: valid together or not at all
            set_decision(record, decision_id, {parse_path(c.path).field: c.human_value for _, c in group}, books,
                         duration_s)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for index, c in group:
            entries[index] = [entry(c, read_value(record, c.path))]
    for (stage, _), index, c in plan:
        if stage == REST_STAGE:
            write(index, c)

    if problems:
        raise ReviewError(problems)
    return record, [e for index in sorted(entries) for e in entries[index]]


def _share(agreeing: list[bool | None]) -> dict:
    known = [a for a in agreeing if a is not None]
    n = len(known)
    return {"agree": sum(known) / n if n else None, "n": n}


def alpha_level(path: str | None) -> Level | None:
    """The measurement level of the values at a path by the path rule (see `value_kind`): the fallback for a log
    line that records no `level`. Ordinal for Stage A, PREP-* and PR-SEV, interval for a decision time and the
    first decision (read as seconds), nominal for the rest; None for the prepared window (two clocks, no alpha)
    and the ask (two numbers, judged by its share)."""
    kind = value_kind(path)
    if kind == "window" or kind == "number":
        return None
    return "ordinal" if kind == "ordinal" else "interval" if kind == "clock" else "nominal"


def level_of(path: str, books: PanelBooks) -> Level | None:
    """The measurement level of the values at a path, from the codebooks: what an import records on a log line.

    "ordinal" for a turn code, a prepared presentation code or a Stage A item whose codebook entry is ordinal;
    "interval" for a count code and for a single clock value (a decision time, the first decision, read as
    seconds); "nominal" for a code or item of another type, a pair judgement and every other field that takes one
    of a set of values or ids; None for the prepared window (several clocks), the ask (judged by its share) and
    free text, which have no alpha."""
    if presence_ref(path) is not None or is_outcome(path):
        return "nominal"
    kind = value_kind(path, books)
    if kind == "window":
        return None
    if kind == "clock":
        return "interval"
    try:
        ref = parse_path(path)
    except ValueError:
        return "nominal"
    if is_banded(path):
        return "ordinal"                # answered and compared in four ordered bands (review_items.EVID_BANDS)
    if kind == "number":
        return None if ref.unit == "venture" else "interval"        # the ask has no alpha; a count is a number
    if (ref.unit, ref.field) in _FREE_TEXT:
        return None
    if ref.unit in ("stage_a", "prep_pres") or (ref.unit == "turn" and ref.field == "code"):
        entry = entry_for(path, books)
        return "ordinal" if entry is not None and entry.type == "ordinal" else "nominal"
    return "nominal"


BY_PATH = "by_path"     # the level of an item whose log line records none: read from the path (older logs)


def _level(x: tuple) -> Level | None:
    """The measurement level of an item (human, primary, replicate, path[, level]): the level its log line
    recorded, else the path rule; always ordinal for the banded count (a line logged before 2026-10-21 recorded
    it as interval)."""
    if is_banded(x[3]):
        return "ordinal"
    return alpha_level(x[3]) if len(x) < 5 or x[4] == BY_PATH else x[4]


def _level_in(line: dict):
    return line["level"] if "level" in line else BY_PATH


EMPTY = "∅"          # the nominal category of an empty value that a share counts as definite ("in no exchange")


def variable_of(path: str | None) -> str:
    """The variable a path measures, which alpha keeps apart: a turn code's code id, `stage_a:<item>`,
    `prep_pres:<code>`, `venture:<field>` (`venture:ask`), `actor:<field>`, `pair:<stage_a item>`; for the classes
    that hold several variables always `<class>:<field>` (`decision:kind`, `exchange_topic:topic`,
    `turn_code:other_concept`); else the path's field class (`outcome` and `cut` for the pitch paths)."""
    try:
        ref = parse_path(path or "")
    except ValueError:
        ref = None
    if ref is not None:
        if ref.unit == "turn" and ref.field == "code":
            return ref.sub
        if ref.unit == "turn" and ref.field == "other_concept":
            return "turn_code:other_concept"
        if ref.unit in ("stage_a", "prep_pres", "venture"):
            return f"{ref.unit}:{ref.unit_id}"
        if ref.unit == "pair":
            return f"pair:{ref.sub}"
        if ref.unit == "actor":
            return f"actor:{ref.field}"
        if ref.unit == "decision":
            return f"decision:{ref.field}"
        if ref.unit == "exchange" and ref.field in ("topic", "label"):
            return f"exchange_topic:{ref.field}"
    try:
        return field_class(path or "")
    except ValueError:
        return "other"


def alpha_level_of(rows: list[tuple]) -> str | None:
    """The level alpha reads `rows` at: "nominal", "ordinal" or "interval"; "mixed" when the rows do not share one
    (alpha is then not defined); None when no row has a level (the prepared window)."""
    levels = {_level(x) for x in rows}
    return "mixed" if len(levels) > 1 else levels.pop() if levels else None


def _alpha_value(level: Level, value):
    """A value as alpha reads it. Nominal: an empty value is the category EMPTY, as the share counts it. Ordinal
    and interval: an empty or unreadable value is None and drops its unit."""
    if level == "nominal":
        return EMPTY if _text(value) is None else _fold(value)
    if _text(value) is None:
        return None
    if level == "interval":        # a clock as seconds, or a count
        try:
            return clock_to_seconds(_text(value))
        except ValueError:
            pass
        try:
            return float(_text(value))
        except ValueError:
            return None
    return _text(value)


def _alpha_strata(rows: list[tuple], i: int, j: int) -> tuple[Level | None, dict[str, list[list]]]:
    """The two-coder units of `rows` ((human, primary, replicate, path) items) for columns i and j, grouped by
    variable, and their level. The level is None when the rows do not share one (alpha over mixed kinds means
    nothing)."""
    levels = {_level(x) for x in rows}
    if len(levels) != 1 or None in levels:
        return None, {}
    level = levels.pop()
    strata: dict[str, list[list]] = {}
    for x in rows:
        # a banded count is read as its band's place, so alpha runs over the four ordered bands
        strata.setdefault(variable_of(x[3]), []).append([_alpha_value(level, band_rank(x[3], x[i])),
                                                         _alpha_value(level, band_rank(x[3], x[j]))])
    return level, strata


def _alpha(rows: list[tuple], i: int, j: int) -> float | None:
    level, strata = _alpha_strata(rows, i, j)
    return stratified_alpha(strata, level) if level else None


def _alpha_units(rows: list[tuple], i: int, j: int) -> int | None:
    """How many of `rows` the alpha between columns i and j is computed from; None when the rows share no level.
    At the ordinal and interval levels a row where either side is no number (an empty value, `missing:<reason>`)
    is not among them, though it counts in the rows and in the share."""
    level, strata = _alpha_strata(rows, i, j)
    return alpha_units(strata, level) if level else None


def _once(rows: list[tuple]) -> list[tuple]:
    """One row per path: the passes' two values at a field are the same whichever coder's row carries them."""
    return list({x[3]: x for x in rows}.values())


def _alpha_ci95(by_video: dict[str, list[tuple]], i: int, j: int, n_boot: int, seed: int) -> tuple[list | None, float | None]:
    """Cluster bootstrap over videos of the stratified alpha between columns i and j of per-video rows:
    (interval or None, share of the draws that had an alpha)."""
    level, _ = _alpha_strata([x for items in by_video.values() for x in items], i, j)
    if not level:
        return None, None
    detail = bootstrap_alpha_detail({v: _alpha_strata(items, i, j)[1] for v, items in by_video.items()},
                                    level, n_boot, seed)
    if detail is None:
        return None, None
    return detail["ci95"], detail["n_defined"] / detail["n_boot"] if detail["n_boot"] else None


def _figures(items: list[tuple], tolerances: Tolerances | None = None, with_alpha: bool = True) -> dict:
    """items are (human, primary, replicate, path) of reviewed rows with a definite human value.

    The first decision agrees by reference turn; other values agree as `agrees` says. When ordinal items are among them,
    "within_one" repeats the three figures over those items alone, two values agreeing when at most
    one scale point apart. "alpha", "alpha_human_vs_replicate" and "alpha_primary_vs_replicate" are
    Krippendorff's alpha of the same pairs (None when the rows share no single level or show no variation).
    The pass-to-pass figures count each path once: with two coders a path has two rows and one pair of passes.
    """
    def figures(rows: list[tuple], same, turn_rule=True) -> dict:
        def compare(x, i, j):
            if turn_rule and is_cut(x[3]):
                return decision_turn_comparison(x[i], x[j], x[5] if len(x) > 5 else None)['agreed']
            return same(x[3], x[i], x[j])
        with_replicate = [x for x in rows if _text(x[2]) is not None]
        return {"n": len(rows),
                "human_vs_primary": _share([compare(x, 0, 1) for x in rows]),
                "human_vs_replicate": _share([compare(x, 0, 2) for x in with_replicate]),
                "primary_vs_replicate": _share([compare(x, 1, 2) for x in _once(with_replicate)])}

    out = figures(items, lambda path, a, b: agrees(path, a, b, tolerances))
    clocks = [x for x in items if is_cut(x[3])]
    if clocks:
        out['decision_rule'] = 'same_reference_turn'
        out['clock_diagnostics'] = figures(clocks, lambda path, a, b: agrees(path, a, b, tolerances), False)
        out['decision_undetermined'] = len(clocks) - _share([
            decision_turn_comparison(x[0], x[1], x[5] if len(x) > 5 else None)['agreed'] for x in clocks])['n']
    if with_alpha:
        with_replicate = [x for x in items if _text(x[2]) is not None]
        out["alpha"] = _alpha(items, 0, 1)
        out["alpha_human_vs_replicate"] = _alpha(with_replicate, 0, 2)
        out["alpha_primary_vs_replicate"] = _alpha(_once(with_replicate), 1, 2)
    ordinal = [x for x in items if _level(x) == "ordinal"]
    if ordinal:
        out["within_one"] = figures(ordinal, lambda path, a, b: within_one(a, b, path))
    return out


def _class_order(present) -> list[str]:
    """Field classes in report order: the outcome, then FIELD_CLASSES, then any other class by name."""
    known = (OUTCOME, *FIELD_CLASSES)
    return [c for c in known if c in present] + sorted(set(present) - set(known))


def _by_class(items: list[tuple[str, tuple]], tolerances: Tolerances | None = None) -> dict:
    return {cls: _figures([x for c, x in items if c == cls], tolerances) for cls in _class_order({c for c, _ in items})}


def _cut_share(agree: float | None, cut_seconds: set[float]) -> dict:
    """Same-turn share; `share_within` is a compatibility alias, with no seconds-based agreement threshold."""
    return {"share_within": agree, "within_seconds": None, 'same_turn': agree,
            'rule': 'same_reference_turn'}


def _presence_figures(judged: list[tuple[bool, str]]) -> dict:
    """judged are (only the replicate has the unit, human judgment) of the unit_presence rows with a judgment."""
    def side(replicate_only: bool) -> dict:
        values = [v for r, v in judged if r == replicate_only]
        return {"judged_present": sum(v == PRESENT for v in values) / len(values) if values else None,
                "n": len(values)}
    return {"n": len(judged), "primary_only": side(False), "replicate_only": side(True)}


def wilson(successes: float, n: int, z: float = 1.959964) -> list[float] | None:
    """The 95% Wilson score interval [low, high] of a share of n items; None without items."""
    if not n:
        return None
    share = successes / n
    centre = share + z * z / (2 * n)
    spread = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    low, high = (centre - spread) / (1 + z * z / n), (centre + spread) / (1 + z * z / n)
    return [0.0 if successes == 0 else max(0.0, low), 1.0 if successes == n else min(1.0, high)]


def _judgment(c: Correction, row: dict):
    """What one reviewed row says: ("presence", only the replicate has the unit, judgment),
    ("item", selection, field class, (human, primary, replicate)) or ("undecided",)."""
    unit = presence_ref(c.path)
    if unit is not None:
        judged = unit.primary_value if c.resolution == "confirm_primary" else (
            c.human_value if c.resolution == "structural" else None)
        return ("presence", unit.replicate_only, judged) if judged in PRESENCE_VALUES else ("undecided",)
    if c.resolution not in DEFINITE:
        return ("undecided",)
    primary, replicate = _text(row.get("primary")), _text(row.get("replicate"))
    human = primary if c.resolution == "confirm_primary" else _text(c.human_value)
    if is_cut(c.path) and c.resolution == 'confirm_primary':
        human = _text(row.get('typed')) or human
    try:
        cls = row.get("field_class") or field_class(c.path)
    except ValueError:
        cls = "unknown"
    item = (human, primary, replicate, c.path, _level_in(row))
    if is_cut(c.path):
        item += (row.get('decision_turn_context'),)
    return ("item", row.get("selected_as") or "added", cls, item)


def cluster_interval(per_video: list[tuple[int, int]], n_boot: int = N_BOOT, seed: int = SEED) -> list[float] | None:
    """The 95% interval [low, high] of a pooled share, resampling whole videos with replacement.

    `per_video` gives (agreeing rows, rows) of each video. Rows of one video stand or fall together,
    which an interval over rows ignores. Seeded; None without rows.
    """
    counts = np.array([c for c in per_video if c[1]], dtype=float).reshape(-1, 2)
    if not len(counts):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(counts), size=(n_boot, len(counts)))
    shares = counts[draws, 0].sum(axis=1) / counts[draws, 1].sum(axis=1)
    low, high = np.quantile(shares, [0.025, 0.975])
    return [float(low), float(high)]


def _estimate(items: list[tuple], population: int | None = None, tolerances: Tolerances | None = None) -> dict:
    """The figures of one field class over sampled rows of one video, with the 95% Wilson interval of human
    against primary (rows of one video are not independent, so the interval is optimistic and says so)."""
    figures = _figures(items, tolerances)
    measured = figures['human_vs_primary']
    interval = wilson((measured['agree'] or 0) * measured['n'], measured['n'])
    return {**figures, "human_vs_primary_ci95": interval, "human_vs_primary_wilson95": interval,
            "ci_method": WILSON_METHOD, "population": population}


def _estimates(items: list[tuple[str, tuple]], population: dict | None = None,
               tolerances: Tolerances | None = None) -> dict:
    return {cls: _estimate([x for c, x in items if c == cls], (population or {}).get(cls), tolerances)
            for cls in _class_order({cls for cls, _ in items})}


def validity_lines(lines: list[dict]) -> list[dict]:
    """The log lines of one video that measure the model's validity: its coder lines (the last per reviewer and
    path) when it has any coder line, else its adjudicator lines (the last per path). Implied lines are left
    out. A line without a `role` is an adjudicator's. With two coders a path gives two lines."""
    real = [line for line in lines if not line.get("implied")]
    coders = [line for line in real if line.get("role") == "coder"]
    if coders:
        last = {(line["reviewer"], line["path"]): line for line in coders}
    else:
        last = {line["path"]: line for line in real}
    return list(last.values())


def structural_rate(lines: list[dict]) -> dict[str, dict]:
    """Per field class, how many reviewed rows were marked structural: {"n_rows", "n_structural", "share"}.
    unit_presence rows are left out: their `structural` resolution carries a present/absent judgment."""
    out: dict[str, dict] = {}
    for line in lines:
        if presence_ref(line["path"]) is not None:
            continue
        cls = line.get("field_class") or "unknown"
        f = out.setdefault(cls, {"n_rows": 0, "n_structural": 0})
        f["n_rows"] += 1
        f["n_structural"] += line.get("resolution") == "structural"
    for f in out.values():
        f["share"] = f["n_structural"] / f["n_rows"] if f["n_rows"] else None
    return dict(sorted(out.items()))


def _coder_pairs(a: dict[str, dict], b: dict[str, dict]) -> tuple[list[tuple[str, tuple]], dict[str, int]]:
    """(field class, (coder A, coder B, None, path)) of the paths both coders gave a definite value, and the
    counts of paths dropped as unclear or structural. unit_presence paths are left out."""
    items: list[tuple[str, tuple]] = []
    dropped = {"unclear": 0, "structural": 0}
    for path in sorted(p for p in set(a) & set(b) if presence_ref(p) is None):
        pair = []
        for line in (a[path], b[path]):
            if line["resolution"] in ("unclear", "structural"):
                dropped[line["resolution"]] += 1
                break
            if line["resolution"] == "confirm_primary":
                pair.append(_text(line.get("typed")) or _text(line["primary"]))
            else:
                pair.append(_text(line.get("new")))
        if len(pair) == 2:
            context = a[path].get('decision_turn_context')
            if context != b[path].get('decision_turn_context'):
                context = None
            items.append((a[path].get("field_class") or field_class(path),
                          (pair[0], pair[1], None, path, _level_in(a[path]), context)))
    return items, dropped


def human_human(by_reviewer: dict[str, dict[str, dict]], tolerances: Tolerances | None = None) -> dict:
    """Agreement between two coders on the rows both gave a definite value; `unclear` and `structural` rows
    are dropped and counted. Values compare as `agrees` (clocks within tolerance); ordinal classes add within_one;
    the cut's share is also given as `share_within` the first decision's tolerance (`within_seconds`).
    unit_presence rows are left out altogether (their `structural` resolution is a present/absent judgment, not a
    dropped row). A coder's confirmation of the primary stands for the value the coder typed when the log line
    carries it as `typed` (parse_sheet discards a clock typed within tolerance of the primary), else for the primary."""
    names = sorted(by_reviewer)
    empty = {"reviewers": names, "n_both": 0, "n_dropped_unclear": 0, "n_dropped_structural": 0,
             "all": {"agree": None, "n": 0}, "by_field_class": {}}
    if len(names) < 2:
        return empty
    if len(names) > 2:
        raise ValueError(f"human_human compares two coders; got {names}")
    items, dropped = _coder_pairs(*(by_reviewer[n] for n in names))
    by_class = _by_class(items, tolerances)
    figures = {cls: {"agree": f["human_vs_primary"]["agree"], "n": f["n"], "alpha": f["alpha"],
                     **({"within_one": {"agree": f["within_one"]["human_vs_primary"]["agree"],
                                        "n": f["within_one"]["human_vs_primary"]["n"]}} if "within_one" in f else {}),
                     **(_cut_share(f["human_vs_primary"]["agree"], {(tolerances or Tolerances()).cut_s})
                        if cls == CUT else {})}
               for cls, f in by_class.items()}
    all_ = _figures([x for _, x in items], tolerances)["human_vs_primary"]
    return {**empty, "n_both": len(items), "n_dropped_unclear": dropped["unclear"],
            "n_dropped_structural": dropped["structural"], "all": all_, "by_field_class": figures}


def human_agreement(rows: list[dict], corrections: list[Correction], left_out: dict | None = None,
                    population: dict | None = None, tolerances: Tolerances | None = None) -> dict:
    """Human-versus-model agreement over the reviewed rows that carry a definite human value.

    `rows` are the sheet's rows (path, field_class, primary, replicate, selected_as). A confirmation
    takes the confirmed pass's value; `unclear` and `structural` rows are counted apart and enter no
    figure. Every figure is {"agree": share or None, "n": items}; per field class the report gives
    human against each pass and, on the same rows, the passes against each other, overall and
    separately for the flagged and the sampled rows.

    Two values agree as selection compares them: a decision time within `tolerances.decision_s`
    (the agreement settings' tolerance), each edge of the prepared window within two seconds,
    everything else as written. Ordinal fields (a row whose `level` is ordinal; without one, the path
    rule of `alpha_level`) are exact, with a `within_one` block beside the exact figures.

    "sampled_estimates" is the part to quote: per field class over the sampled rows only (every row of
    an exchange-level sheet), the same figures with a 95% Wilson interval for human against primary
    (labelled: it treats rows as independent) and, from `population` (the export manifest's count
    per field class), how many such fields the sample stands for. An investor's outcome is the field
    class `outcome`.

    unit_presence rows are reported apart, as by_field_class["unit_presence"]: of the units only the
    primary recorded and of those only the replicate recorded, the share the human judged present.
    `left_out` is the export's count of flagged rows the sheet did not take ({"n", "by_field_class",
    "by_reason"}); it is passed through with a note on what the flagged figures then describe.
    """
    tolerances = tolerances or Tolerances()
    by_path = {normalize_path(row["path"]): row for row in rows}
    items: list[tuple[str, str, tuple]] = []        # (selection, field class, (human, primary, replicate, path))
    presence: list[tuple[bool, str]] = []
    undecided = 0
    for c in corrections:
        judged = _judgment(c, by_path.get(c.path) or {})
        if judged[0] == "presence":
            presence.append(judged[1:])
        elif judged[0] == "item":
            items.append(judged[1:])
        else:
            undecided += 1

    selections = ["flagged", "sampled"] + sorted({s for s, _, _ in items} - {"flagged", "sampled"})
    by_class = _by_class([(cls, x) for _, cls, x in items], tolerances)
    if presence:
        by_class = {UNIT_PRESENCE: _presence_figures(presence), **by_class}
    left = left_out if left_out and left_out.get("n") else None
    return {
        "note": AGREEMENT_NOTE,
        "tolerances": asdict(tolerances),
        "n_reviewed": len(items) + len(presence),
        "n_without_a_value": undecided,
        "n_structural": sum(c.resolution == "structural" for c in corrections),
        "n_outcome_overrides": sum(c.resolution == OUTCOME_OVERRIDE and is_outcome(c.path) for c in corrections),
        "n_cut_overrides": sum(c.resolution == OUTCOME_OVERRIDE and is_cut(c.path) for c in corrections),
        "flagged_left_out": left,
        "flagged_note": LEFT_OUT_NOTE if left else None,
        "all": _figures([x for _, _, x in items], tolerances),
        "by_field_class": by_class,
        "sampled_estimates": _estimates([(cls, x) for sel, cls, x in items if sel == "sampled"], population,
                                        tolerances),
        "by_selection": {
            s: {"all": _figures([x for sel, _, x in items if sel == s], tolerances),
                "by_field_class": _by_class([(cls, x) for sel, cls, x in items if sel == s], tolerances)}
            for s in selections},
    }


def pooled_summary(videos: dict[str, list[dict]], tolerances: dict[str, Tolerances] | None = None,
                   n_boot: int = N_BOOT, seed: int = SEED,
                   by_reviewer: dict[str, dict[str, list[dict]]] | None = None, pairs: dict | None = None,
                   strata: dict | None = None, reporting: Reporting | None = None) -> dict:
    """Agreement pooled over videos. `videos` maps a video id to its reviewed rows: the last log line of
    every reviewed path (path, field_class, primary, replicate, selected_as, new, resolution).

    "sampled" and "flagged" give, per field class, the number of videos and rows and the figures of
    human_agreement's sampled_estimates (the population is not pooled); only "sampled" is an estimate.
    First-decision values agree by the sheet’s pinned turn context, with unmapped times undetermined.
    Other values use each video’s tolerances (`tolerances`, by video id; the defaults else).

    The interval of human against primary resamples whole videos (`ci_method` says "cluster
    bootstrap over videos", seeded) when at least five videos carry rows of the class; with fewer it
    is the Wilson interval, labelled as treating rows as independent and optimistic. The Wilson
    interval is always given too, as human_vs_primary_wilson95.

    `by_reviewer` (reviewer -> video -> that coder's lines) adds "by_reviewer" (the same tables per coder),
    "human_human" (coder against coder per field class, over the videos on which exactly two coders judged;
    each coder's lines compared within the video's tolerances, the interval resampling videos) and
    "structural_by_reviewer"; "structural" gives the structural rate of `videos` per field class.

    "cut" gives the first decision's older figures: {"share_within", "n", "alpha",
    "within_seconds"}. `share_within` now aliases same-turn agreement and `within_seconds` is None;
    alpha on seconds remains only as a legacy diagnostic (the same figures as sampled["cut"]). "pairs" holds the clip session's pair judgements per Stage A
    dimension as given (`pairs`, from review_clips.pooled_pairs: {dimension: {"coder_model", "coder_coder", "n",
    "videos", ...}}); empty when none were imported.

    "without_unsure" repeats the sampled figures without the rows a coder ticked "not sure" ([unsure] in the note):
    {"n_unsure", "sampled", "by_variable"}, the same figures when no row is unsure.

    `strata` describes a validation sample stratified by outcome ({"stratify", "of": {video: stratum}, "run":
    {stratum: eligible pitches in the run}}, from review_panel.sample_strata) and adds "strata" when some of
    `videos` is in that sample: per stratum its pooled figures, and over the stratified videos present each class's
    and variable's share and alpha reweighted to the run's mix beside the unweighted ones (`_strata`, weights from
    the videos present, not the planned sample). The other figures stay unweighted.

    `reporting` (review_report.Reporting, validation_reporting.yaml) adds "reporting": every reported code with the
    statistics of its kind (`reporting_summary`; plan D). Without it "reporting" is {}. Every other key is the same
    with or without it.
    """
    tolerances = tolerances or {}
    core = _pool(videos, tolerances, n_boot, seed)
    sure = {v: [line for line in lines if not is_unsure(line)] for v, lines in videos.items()}
    n_unsure = sum(len(lines) - len(sure[v]) for v, lines in videos.items())
    without = _pool({v: lines for v, lines in sure.items() if lines}, tolerances, n_boot, seed) if n_unsure else core
    by_reviewer = by_reviewer or {}
    flat = {name: {v: lines for v, lines in per.items() if lines} for name, per in by_reviewer.items()}
    cut = core["sampled"].get(CUT)
    return {**core,
            "cut": ({key: cut[key] for key in ("share_within", "n", "alpha", "within_seconds")} if cut else
                    {"share_within": None, "n": 0, "alpha": None, "within_seconds": CUT_TOLERANCE_S}),
            "pairs": dict(pairs or {}),
            "without_unsure": {"n_unsure": n_unsure, "sampled": without["sampled"],
                               "by_variable": without["by_variable"]},
            # only when some pooled video belongs to the stratified sample (the training videos never do)
            **({"strata": _strata(videos, tolerances, strata)}
               if strata and any(v in strata["of"] for v in videos) else {}),
            "by_reviewer": {name: _pool(per, tolerances, n_boot, seed) for name, per in sorted(flat.items())},
            "human_human": _pool_human_human(flat, tolerances, n_boot, seed),
            "structural": structural_rate([line for lines in videos.values() for line in lines]),
            "structural_by_reviewer": {name: structural_rate([line for lines in per.values() for line in lines])
                                       for name, per in sorted(flat.items())},
            "reporting": (reporting_summary(videos, flat, pairs or {}, reporting, n_boot, seed, strata)
                          if reporting is not None else {})}


def _pool_human_human(by_reviewer: dict[str, dict[str, list[dict]]], tolerances: dict[str, Tolerances],
                      n_boot: int, seed: int) -> dict:
    """Coder against coder per field class (and, under "by_variable", per variable) over the videos on which
    exactly two coders judged."""
    coders: dict[str, dict[str, dict[str, dict]]] = {}
    for name, per in by_reviewer.items():
        for video_id, lines in per.items():
            coders.setdefault(video_id, {})[name] = {line["path"]: line for line in lines}
    pairs = {video_id: _coder_pairs(*(got[n] for n in sorted(got)))[0]
             for video_id, got in sorted(coders.items()) if len(got) == 2}

    def pooled(group_of, order: list[str]) -> dict:
        per_group: dict[str, list[tuple[int, int, dict | None]]] = {}
        pair_rows: dict[str, dict[str, list[tuple]]] = {}
        for video_id, items in pairs.items():
            groups: dict[str, list[tuple]] = {}
            for cls, x in items:
                groups.setdefault(group_of(cls, x), []).append(x)
            for group, rows in groups.items():
                f = _figures(rows, tolerances.get(video_id), with_alpha=False)
                measured = f['human_vs_primary']
                per_group.setdefault(group, []).append((round((measured['agree'] or 0) * measured['n']), measured['n'],
                                                        f.get("within_one", {}).get("human_vs_primary")))
                pair_rows.setdefault(group, {})[video_id] = rows
        out = {}
        for group in [g for g in order if g in per_group] + sorted(set(per_group) - set(order)):
            parts = [p for p in per_group[group] if p[1]]
            n, agree = sum(p[1] for p in parts), sum(p[0] for p in parts)
            clustered = len(parts) >= MIN_CLUSTER_VIDEOS
            by_video = pair_rows[group]
            ci, defined = _alpha_ci95(by_video, 0, 1, n_boot, seed) if clustered else (None, None)
            everything = [x for rows in by_video.values() for x in rows]
            entry = {"videos": len(parts), "n": n, "agree": agree / n if n else None,
                     "alpha_level": alpha_level_of(everything),
                     "alpha": _alpha(everything, 0, 1), "alpha_units": _alpha_units(everything, 0, 1),
                     "alpha_ci95": ci, "alpha_ci95_defined": defined,
                     "ci95": (cluster_interval([(p[0], p[1]) for p in parts], n_boot, seed) if clustered
                              else wilson(agree, n)),
                     "ci_method": (CLUSTER_BOOTSTRAP if (n_boot, seed) == (N_BOOT, SEED) else
                                   f"cluster bootstrap over videos ({n_boot:,} draws, seed {seed})") if clustered
                     else WILSON_METHOD}
            near = [(round(p[2]["agree"] * p[2]["n"]), p[2]["n"]) for p in parts if p[2] and p[2]["n"]]
            if near:
                entry["within_one"] = {"agree": sum(a for a, _ in near) / sum(m for _, m in near),
                                       "n": sum(m for _, m in near)}
            if everything and all(is_cut(x[3]) for x in everything):
                entry.update(_cut_share(entry["agree"], {(tolerances.get(v) or Tolerances()).cut_s for v in by_video}))
            out[group] = entry
        return out

    out = pooled(lambda cls, x: cls, [OUTCOME, *FIELD_CLASSES])
    out["by_variable"] = pooled(lambda cls, x: variable_of(x[3]), [])
    return out


def _pool(videos: dict[str, list[dict]], tolerances: dict[str, Tolerances], n_boot: int, seed: int,
          intervals: bool = True) -> dict:
    """The pooled figures of `pooled_summary` over `videos` alone. `intervals=False` skips the cluster bootstraps
    (every interval is then the Wilson one, or none for alpha): for figures that are reported as points only."""
    pooled: dict[str, dict[str, list[tuple[str, tuple]]]] = {"sampled": {}, "flagged": {}}
    presence: list[tuple[bool, str]] = []
    counts = {"rows": 0, "structural": 0, "outcome_overrides": 0, "cut_overrides": 0, "without_a_value": 0}
    for video_id, lines in sorted(videos.items()):
        for line in lines:
            c = Correction(line["path"], line.get("new"), line["resolution"], line.get("note") or "", "")
            counts["rows"] += 1
            counts["structural"] += c.resolution == "structural"
            counts["outcome_overrides"] += c.resolution == OUTCOME_OVERRIDE and is_outcome(c.path)
            counts["cut_overrides"] += c.resolution == OUTCOME_OVERRIDE and is_cut(c.path)
            judged = _judgment(c, line)
            if judged[0] == "presence":
                presence.append(judged[1:])
            elif judged[0] == "item":
                selection = "sampled" if judged[1] == "sampled" else "flagged"      # an added row is no sample
                pooled[selection].setdefault(judged[2], []).append((video_id, judged[3]))
            else:
                counts["without_a_value"] += 1

    def figures(rows: list[tuple[str, tuple]]) -> dict:
        """One class over videos: each video's rows compared within that video's tolerances, then pooled."""
        by_video: dict[str, list[tuple]] = {}
        for video_id, item in rows:
            by_video.setdefault(video_id, []).append(item)
        parts = {v: _figures(items, tolerances.get(v)) for v, items in by_video.items()}

        def pool(block: str | None, key: str) -> dict:
            got = [(p[block] if block else p)[key] for p in parts.values() if not block or block in p]
            n = sum(f["n"] for f in got)
            return {"agree": sum(f["agree"] * f["n"] for f in got if f["n"]) / n if n else None, "n": n}

        keys = ("human_vs_primary", "human_vs_replicate", "primary_vs_replicate")
        out = {"n": len(rows), **{key: pool(None, key) for key in keys}}
        if any("within_one" in p for p in parts.values()):
            out["within_one"] = {"n": sum(p["within_one"]["n"] for p in parts.values() if "within_one" in p),
                                 **{key: pool("within_one", key) for key in keys}}
        per_video = [(round(p['human_vs_primary']['agree'] * p['human_vs_primary']['n']),
                      p['human_vs_primary']['n']) for p in parts.values() if p['human_vs_primary']['n']]
        agree = sum(a for a, _ in per_video)
        plain = wilson(agree, sum(n for _, n in per_video))
        clustered = intervals and len(per_video) >= MIN_CLUSTER_VIDEOS
        everything = [x for items in by_video.values() for x in items]
        replicated = {v: [x for x in items if _text(x[2]) is not None] for v, items in by_video.items()}
        out["alpha_level"] = alpha_level_of(everything)
        out["alpha"] = _alpha(everything, 0, 1)
        out["alpha_units"] = _alpha_units(everything, 0, 1)
        out["alpha_human_vs_replicate"] = _alpha([x for r in replicated.values() for x in r], 0, 2)
        # the passes' pair of a field is one unit however many coders judged the field
        out["alpha_primary_vs_replicate"] = _alpha([x for r in replicated.values() for x in _once(r)], 1, 2)
        out["alpha_ci95"], out["alpha_ci95_defined"] = (_alpha_ci95(by_video, 0, 1, n_boot, seed) if clustered
                                                        else (None, None))
        if everything and all(is_cut(x[3]) for x in everything):
            out.update(_cut_share(out["human_vs_primary"]["agree"],
                                  {(tolerances.get(v) or Tolerances()).cut_s for v in by_video}))
            out['decision_undetermined'] = len(rows) - out['human_vs_primary']['n']
        return {"videos": len(by_video), **out,
                "human_vs_primary_ci95": cluster_interval(per_video, n_boot, seed) if clustered else plain,
                "human_vs_primary_wilson95": plain,
                "ci_method": (CLUSTER_BOOTSTRAP if (n_boot, seed) == (N_BOOT, SEED) else
                              f"cluster bootstrap over videos ({n_boot:,} draws, seed {seed})") if clustered
                else WILSON_METHOD,
                "population": None}

    def table(by_class: dict[str, list[tuple[str, tuple]]]) -> dict:
        return {cls: figures(by_class[cls]) for cls in _class_order(by_class)}

    by_variable: dict[str, list[tuple[str, tuple]]] = {}
    for rows in pooled["sampled"].values():
        for video_id, item in rows:
            by_variable.setdefault(variable_of(item[3]), []).append((video_id, item))
    return {"n_videos": len(videos), **{f"n_{k}": v for k, v in counts.items()},
            "sampled": table(pooled["sampled"]), "flagged": table(pooled["flagged"]),
            "by_variable": {var: figures(by_variable[var]) for var in sorted(by_variable)},
            "unit_presence": _presence_figures(presence) if presence else None}


# ---- a sample stratified by outcome (plan C Task 7f, 2026-10-21) ---------------------------------------------------

def stratum_weights(of: dict[str, str], run: dict[str, int]) -> dict[str, float | None]:
    """Each stratum's weight: its share of the eligible pitches in the run over its share of the sample's videos
    (None for a stratum the sample does not hold)."""
    total, n = sum(run.values()), len(of)
    sample = {s: sum(1 for t in of.values() if t == s) for s in run}
    return {s: (run[s] / total) / (sample[s] / n) if sample[s] and total and n else None for s in run}


def _sampled_items(videos: dict[str, list[dict]]) -> list[tuple[str, str, tuple]]:
    """(video, field class, item) of the sampled rows with a definite value, as `_pool` counts them."""
    out = []
    for video_id, lines in sorted(videos.items()):
        for line in lines:
            judged = _judgment(Correction(line["path"], line.get("new"), line["resolution"], line.get("note") or "", ""),
                               line)
            if judged[0] == "item" and judged[1] == "sampled":
                out.append((video_id, judged[2], judged[3]))
    return out


def _weighted_alpha(rows: list[tuple[float, tuple]]) -> float | None:
    """The stratified alpha (human against primary) of (weight, item) rows, each unit weighted; the weights scaled to
    a mean of 1 over the group's rows, so the alpha's small-sample term counts about as many units as the group has
    (within one variable whose rows lean to one stratum the weighted count differs somewhat from its unit count)."""
    level, _ = _alpha_strata([x for _, x in rows], 0, 1)
    if not level:
        return None
    mean = sum(w for w, _ in rows) / len(rows)
    units: dict[str, list[list]] = {}
    weights: dict[str, list[float]] = {}
    for w, x in rows:
        variable = variable_of(x[3])
        units.setdefault(variable, []).append([_alpha_value(level, band_rank(x[3], x[0])),
                                               _alpha_value(level, band_rank(x[3], x[1]))])
        weights.setdefault(variable, []).append(w / mean)
    return stratified_alpha(units, level, weights)


def _reweighted(rows: list[tuple[str, tuple]], weight: dict[str, float], tolerances: dict[str, Tolerances]) -> dict:
    """One group's human-against-primary share and alpha with each row weighted by its video's stratum weight, beside
    the unweighted ones over the same rows."""
    agree = [(weight[v], decision_turn_comparison(x[0], x[1], x[5] if len(x) > 5 else None)['agreed']
              if is_cut(x[3]) else agrees(x[3], x[0], x[1], tolerances.get(v))) for v, x in rows]
    agree = [(w, a) for w, a in agree if a is not None]
    total = sum(w for w, _ in agree)
    return {"n": len(rows), "videos": len({v for v, _ in rows}),
            "agree": sum(w for w, a in agree if a) / total if total else None,
            "agree_unweighted": sum(a for _, a in agree) / len(agree) if agree else None,
            "alpha": _weighted_alpha([(weight[v], x) for v, x in rows]),
            "alpha_unweighted": _alpha([x for _, x in rows], 0, 1)}


def _strata(videos: dict[str, list[dict]], tolerances: dict[str, Tolerances], strata: dict) -> dict:
    """pooled_summary's "strata": {"stratify", "strata": {stratum: {"run", "planned", "sample", "weight", "videos"}},
    "by_stratum": {stratum: {"n_videos", "sampled", "by_variable"}}, "reweighted": {"sampled": {class: figures},
    "by_variable": {variable: figures}}}, the figures {"n", "videos", "agree", "agree_unweighted", "alpha",
    "alpha_unweighted"}. A video outside the stratified sample enters none of them. The weights: the stratum's share
    of the run's eligible pitches over its share of the stratified videos present in `videos` (`sample`, `videos`;
    `planned` is the stratum's size in the drawn sample), so a summary run before every video is imported still
    weights what it pools to the run's mix. The per-stratum figures are points: no bootstrap is run for them."""
    of, run = strata["of"], strata["run"]
    in_sample = {v: lines for v, lines in videos.items() if v in of}
    present = {v: of[v] for v in in_sample}
    weights = stratum_weights(present, run)
    info = {s: {"run": run[s], "planned": sum(1 for t in of.values() if t == s),
                "sample": sum(1 for t in present.values() if t == s), "weight": weights[s],
                "videos": sorted(v for v, t in present.items() if t == s)} for s in run}
    by_stratum = {}
    for s in run:
        pooled = _pool({v: lines for v, lines in in_sample.items() if of[v] == s}, tolerances, 0, 0,
                       intervals=False)
        by_stratum[s] = {"n_videos": pooled["n_videos"], "sampled": pooled["sampled"],
                         "by_variable": pooled["by_variable"]}
    weight = {v: weights[of[v]] for v in in_sample}
    items = _sampled_items(in_sample)
    classes: dict[str, list[tuple[str, tuple]]] = {}
    variables: dict[str, list[tuple[str, tuple]]] = {}
    for video_id, cls, x in items:
        classes.setdefault(cls, []).append((video_id, x))
        variables.setdefault(variable_of(x[3]), []).append((video_id, x))
    return {"stratify": strata.get("stratify"), "strata": info, "by_stratum": by_stratum,
            "reweighted": {"sampled": {cls: _reweighted(classes[cls], weight, tolerances) for cls in _class_order(classes)},
                           "by_variable": {var: _reweighted(variables[var], weight, tolerances)
                                           for var in sorted(variables)}}}


# ---- the reporting of plan D (2026-10-30): each code with the statistics of its kind -------------------------------

SCALE_KINDS = ("categorical", "binary", "ordinal")      # the kinds whose values are a scale: values, counts, prevalence
# the figures of each kind that carry a 95% interval resampling videos: the shares and the chance-corrected statistics
# (a comparison's interval resamples videos in review_clips, beside the exact binomial one; the chance values, indices,
# differences and seconds are points)
REPORTING_INTERVALS = {
    "categorical": ("exact", "alpha", "ac1", "modal_share"),
    "binary": ("exact", "positive_agreement", "negative_agreement", "ac1", "alpha", "modal_share"),
    "ordinal": ("exact", "within_one", "alpha", "ac2", "modal_share"),
    "clock": ("same_turn", "share_within_5", "share_within_30"),
    "exact": ("exact",),
}
# the shares a stratified sample's figures are reweighted to the run's mix (beside alpha where the kind has one)
REWEIGHTED_SHARES = {"categorical": ("exact",), "binary": ("exact",), "ordinal": ("exact", "within_one"),
                     "clock": ("same_turn", "share_within_5", "share_within_30"), "exact": ("exact",)}
# figures a kind's row carries beyond the statistics validation_reporting.yaml lists for it: a binary row's nominal
# alpha, printed in parentheses beside AC1 as the other kinds print theirs (the coordinator's ruling, 2026-10-30)
REPORTING_EXTRA = {"binary": ("alpha",)}
ALPHA_LEVEL = {"categorical": "nominal", "binary": "nominal", "ordinal": "ordinal"}


def _finite(value):
    """A statistic as JSON may hold it: NaN and infinities (an empty input) are None."""
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _on_scale(values: list[str], path: str | None, value, via=None) -> str | None:
    """A value as one of the scale's `values`: a banded count as its band, case and blanks folded, a whole number
    written as a float ("4.0") as its whole number; `via` (a derivation such as frame_of) applied first. None for a
    value off the scale (empty, `missing:<reason>`, a concept that has no frame)."""
    if via is not None:
        value = via(_text(value))
    text = _text(banded(path, value))
    if text is None:
        return None
    if text in values:
        return text
    folded = {v.casefold(): v for v in values}
    if text.casefold() in folded:
        return folded[text.casefold()]
    try:
        number = float(text)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() and str(int(number)) in values else None


def _seconds(value) -> float | None:
    text = _text(value)
    if text is None:
        return None
    try:
        return clock_to_seconds(text)
    except ValueError:
        pass
    try:
        seconds = float(text)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) else None


def decision_turn_context(record: dict, pass_name: str, digest: str) -> dict:
    """Freeze the sheet's reference segmentation, independently of later pass selection or adjudication."""
    return {'pass': pass_name, 'sha256': digest,
            'turns': [[t['turn_id'], t['start_s'], t['end_s']] for t in record['turns']]}


def decision_turn_comparison(human_clock, model_clock, context: dict | None) -> dict:
    """Same-turn judgment using the sheet's reference record. No context or ambiguous mappings stay unknown."""
    context = context or {}
    spans = context.get('turns') or []
    result = ag.same_turn_agreement(_seconds(human_clock), _seconds(model_clock), spans)
    matched = next((t for t in spans if t[0] == result['b_turn_id']), None)
    return {**result, 'human_clock': _text(human_clock), 'model_clock': _text(model_clock),
            'reference_pass': context.get('pass'), 'reference_sha256': context.get('sha256'),
            'turn_start_s': matched[1] if matched else None, 'turn_end_s': matched[2] if matched else None}


class _TurnClock(float):
    """Seconds with the reference turn attached; subtraction still supplies unchanged timing diagnostics."""
    def __new__(cls, seconds, context):
        value = super().__new__(cls, seconds)
        result = decision_turn_comparison(seconds, seconds, context)
        value.turn_id = result['a_turn_id']
        value.reference = (context or {}).get('sha256')
        return value


def _same_clock_turn(a, b):
    if (getattr(a, 'turn_id', None) is None or getattr(b, 'turn_id', None) is None
            or getattr(a, 'reference', None) != getattr(b, 'reference', None)):
        return None
    return a.turn_id == b.turn_id


def _kind_statistics(kind: str, pairs: list[tuple], values: list[str] | None, positive: str | None,
                     path: str | None, raters: tuple[str, str], only: tuple[str, ...] | None = None) -> dict:
    """The statistics of `kind` over `pairs` ((first rater, second rater), each on the scale; a clock's pair the two
    readings in seconds; the ask's pair its two texts). `only` restricts them to those names (a bootstrap draw)."""
    want = (lambda name: only is None or name in only)
    n = len(pairs)
    exact = (sum(a == b for a, b in pairs) / n) if n else float("nan")
    if kind == "categorical":
        out = {"exact": exact}
        if want("ac1"):
            out["ac1"] = ag.gwet_ac1(pairs, values)
        if want("modal_share"):
            out["modal_share"] = ag.modal_share(pairs)
        if only is None:
            out["alpha"] = ag.krippendorff_alpha([[a, b] for a, b in pairs], "nominal")
            out["confusion"] = {x: {y: sum(1 for a, b in pairs if (a, b) == (x, y)) for y in values} for x in values}
            out["by_rater"] = list(ag.modal_shares_by_rater(pairs))
        return out
    if kind == "binary":
        negative = next(v for v in values if v != positive)
        pos, neg = ag.specific_agreement(pairs, positive, negative) if n else (None, None)
        out = {"exact": exact, "positive_agreement": pos, "negative_agreement": neg}
        if want("ac1"):
            out["ac1"] = ag.gwet_ac1(pairs, values)
        if want("modal_share"):
            out["modal_share"] = ag.modal_share(pairs)
        if only is None:
            counts = {"a": sum(1 for a, b in pairs if a == positive and b == positive),
                      "b": sum(1 for a, b in pairs if a == positive and b != positive),
                      "c": sum(1 for a, b in pairs if a != positive and b == positive)}
            counts["d"] = n - counts["a"] - counts["b"] - counts["c"]
            index, who = ag.bias_index(pairs, positive, negative)
            out.update(counts=counts, prevalence_index=ag.prevalence_index(pairs, positive, negative),
                       bias_index=index, direction={"a": raters[0], "b": raters[1]}.get(who, "none"),
                       by_rater=list(ag.modal_shares_by_rater(pairs)),
                       alpha=ag.krippendorff_alpha([[a, b] for a, b in pairs], "nominal"))
        return out
    if kind == "ordinal":
        out = {"exact": exact}
        if want("within_one"):
            out["within_one"] = ag.within_one_share(pairs, values)
        if want("ac2"):
            out["ac2"] = ag.gwet_ac2(pairs, values)
        if want("modal_share"):
            out["modal_share"] = ag.modal_share(pairs)
        if only is None:
            place = {v: i for i, v in enumerate(values)}
            out.update(chance_exact=ag.chance_exact(pairs, values), chance_within_one=ag.chance_within_one(pairs, values),
                       alpha=ag.krippendorff_alpha([[place[a], place[b]] for a, b in pairs], "ordinal"),
                       mean_difference=ag.mean_difference(pairs, values), spearman=ag.spearman(pairs, values),
                       by_rater=list(ag.modal_shares_by_rater(pairs)),
                       confusion={x: {y: sum(1 for a, b in pairs if (a, b) == (x, y)) for y in values}
                                  for x in values})
        return out
    if kind == "clock":
        figures = ag.clock_agreement([a - b for a, b in pairs])
        out = {k: v for k, v in figures.items() if k != "n"}
        turn_stats = ag.turn_agreement_summary([_same_clock_turn(a, b) for a, b in pairs])
        out['same_turn'] = turn_stats.pop('same_turn')
        out.update(turn_stats)
        if only is None:
            out["counts"] = {"n": n, "within_5": sum(abs(a - b) <= 5 for a, b in pairs),
                             "within_30": sum(abs(a - b) <= 30 for a, b in pairs), **turn_stats}
        return out
    if kind == "exact":
        same = [agrees(path, a, b) for a, b in pairs]
        out = {"exact": (sum(same) / n) if n else float("nan")}
        if only is None:
            out["counts"] = {"agree": sum(same), "n": n}
        return out
    raise ValueError(f"no statistics for kind {kind!r}")


def _resampled(kind: str, by_video: dict[str, list[tuple]], values, positive, path, raters,
               n_boot: int, seed: int) -> dict[str, list[float] | None]:
    """The 95% percentile intervals of the kind's interval figures, resampling whole videos with replacement (the
    draws of `cluster_interval`, seeded); a figure undefined in more than 1 - MIN_DEFINED_SHARE of the draws has none.
    Alpha's interval is `bootstrap_alpha_detail`'s, over the same videos."""
    names = tuple(n for n in REPORTING_INTERVALS[kind] if n != "alpha")
    videos = [pairs for _, pairs in sorted(by_video.items()) if pairs]
    out: dict[str, list[float] | None] = {}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(videos), size=(n_boot, len(videos)))
    got: dict[str, list[float]] = {name: [] for name in names}
    for row in draws:
        pairs = [p for i in row for p in videos[i]]
        stats = _kind_statistics(kind, pairs, values, positive, path, raters, only=names)
        for name in names:
            value = _finite(stats.get(name))
            if value is not None:
                got[name].append(value)
    for name in names:
        if name == 'same_turn' and sum(any(_same_clock_turn(a, b) is not None for a, b in pairs)
                                       for pairs in videos) < MIN_CLUSTER_VIDEOS:
            out[name] = None
            continue
        if len(got[name]) >= MIN_DEFINED_SHARE * n_boot:
            low, high = np.quantile(got[name], [0.025, 0.975])
            out[name] = [float(low), float(high)]
        else:
            out[name] = None
    if "alpha" in REPORTING_INTERVALS[kind]:
        level = ALPHA_LEVEL[kind]
        place = {v: i for i, v in enumerate(values)}
        units = {v: {"": [[a, b] if level == "nominal" else [place[a], place[b]] for a, b in pairs]}
                 for v, pairs in sorted(by_video.items()) if pairs}
        detail = bootstrap_alpha_detail(units, level, n_boot, seed)
        out["alpha"] = detail["ci95"] if detail else None
    return out


def _points(kind: str, stats: dict, statistics: list[str]) -> dict:
    """The scalar figures of `stats` among the kind's statistics, NaN as None."""
    out = {}
    for name in statistics:
        value = stats.get(name)
        if value is None or isinstance(value, (int, float, np.floating)):
            out[name] = _finite(value)
    if kind == 'clock':
        out.update({name: stats[name] for name in ('same_turn_n', 'same_turn_undetermined')})
    return out


def _weighted(kind: str, units: list[tuple], weight: dict[str, float], values, path) -> dict:
    """The kind's shares (and alpha where it has one) with each unit weighted by its video's stratum weight, the
    weights scaled to a mean of 1 (as `_weighted_alpha`)."""
    rows = [(weight[v], a, b) for v, a, b, _ in units if weight.get(v) is not None]
    total = sum(w for w, _, _ in rows)
    out: dict = {"n": len(rows)}

    def share(ok) -> float | None:
        return sum(w for w, a, b in rows if ok(a, b)) / total if total else None

    for name in REWEIGHTED_SHARES[kind]:
        if name == "exact":
            out[name] = share((lambda a, b: agrees(path, a, b)) if kind == "exact" else (lambda a, b: a == b))
        elif name == "within_one":
            place = {v: i for i, v in enumerate(values)}
            out[name] = share(lambda a, b: abs(place[a] - place[b]) <= 1)
        elif name == 'same_turn':
            mapped = [(w, _same_clock_turn(a, b)) for w, a, b in rows]
            mapped = [(w, ok) for w, ok in mapped if ok is not None]
            out[name] = sum(w * ok for w, ok in mapped) / sum(w for w, _ in mapped) if mapped else None
            out['same_turn_n'] = len(mapped)
            out['same_turn_undetermined'] = len(rows) - len(mapped)
        else:
            limit = 5 if name == "share_within_5" else 30
            out[name] = share(lambda a, b: abs(a - b) <= limit)
    if kind in ALPHA_LEVEL:
        if rows:
            mean = total / len(rows)
            place = {v: i for i, v in enumerate(values)}
            level = ALPHA_LEVEL[kind]
            units_ = [[a, b] if level == "nominal" else [place[a], place[b]] for _, a, b in rows]
            out["alpha"] = stratified_alpha({"": units_}, level, {"": [w / mean for w, _, _ in rows]})
        else:
            out["alpha"] = None
    return out


def _report_row(kind: str, units: list[tuple], n_off_scale: int, values, positive, path, raters: tuple[str, str],
                reporting: Reporting, n_boot: int, seed: int, strata: dict | None, n_units: int | None = None) -> dict:
    """One reported variable: `units` are (video, first, second, unsure) on the scale, one per judgement. The shape
    is plan D's (Task 3): {"kind", "n" (the judgements), "units" (`n_units`, the distinct units, else n), "videos",
    "raters", "n_off_scale", "coder_values", "model_values", "figures": {name: {"value", "low", "high"}}, "flags",
    "counts", "without_unsure": {name: value, "n"}, "strata": {stratum: {name: value, "n"}, "reweighted": {share:
    value, "alpha", "n"}}}. "coder_values" are the first rater's, "model_values" the second's (a second coder's, or
    the replicate's, as "raters" names them), counted per judgement. The small-n flag counts the distinct units."""
    statistics = reporting.statistics_of(kind)
    statistics += [name for name in REPORTING_EXTRA.get(kind, ()) if name not in statistics]
    pairs = [(a, b) for _, a, b, _ in units]
    by_video: dict[str, list[tuple]] = {}
    for v, a, b, _ in units:
        by_video.setdefault(v, []).append((a, b))
    stats = _kind_statistics(kind, pairs, values, positive, path, raters)
    clustered = n_boot > 0 and len(by_video) >= MIN_CLUSTER_VIDEOS
    intervals = _resampled(kind, by_video, values, positive, path, raters, n_boot, seed) if clustered else {}
    figures = {}
    for name in statistics:
        if name in ("confusion", "counts"):
            figures[name] = {"value": stats[name], "low": None, "high": None}
            continue
        ci = intervals.get(name)
        figures[name] = {"value": _finite(stats.get(name)), "low": ci[0] if ci else None,
                         "high": ci[1] if ci else None}
    if "modal_share" in figures and "by_rater" in stats:
        figures["modal_share"]["by_rater"] = [_finite(x) for x in stats["by_rater"]]
    if "bias_index" in figures:
        figures["bias_index"]["direction"] = stats["direction"]
    flags = []
    if kind in SCALE_KINDS and pairs and stats["modal_share"] >= reporting.flags["prevalence"]:
        flags.append("prevalence")
    # the small-n flag counts units (a pitch, a question-answer pair), not judgements: two coders judge one unit twice
    distinct = len(pairs) if n_units is None else n_units
    if distinct < reporting.flags["small_n"]:
        flags.append("small_n")
    sure = [(a, b) for _, a, b, unsure in units if not unsure]
    without = {**_points(kind, _kind_statistics(kind, sure, values, positive, path, raters), statistics),
               "n": len(sure)}
    by_stratum: dict = {}
    if strata:
        of, weights = strata["of"], strata["weights"]
        for s in strata["run"]:
            mine = [(a, b) for v, a, b, _ in units if of.get(v) == s]
            if mine:
                by_stratum[s] = {**_points(kind, _kind_statistics(kind, mine, values, positive, path, raters),
                                           statistics), "n": len(mine)}
        inside = [u for u in units if u[0] in of]
        if inside:
            by_stratum["reweighted"] = _weighted(kind, inside, {v: weights[of[v]] for v, *_ in inside}, values, path)
    return {"kind": kind, "n": len(pairs), "units": distinct, "videos": len(by_video), "raters": list(raters),
            "n_off_scale": n_off_scale,
            "coder_values": {v: sum(a == v for a, _ in pairs) for v in values} if kind in SCALE_KINDS else {},
            "model_values": {v: sum(b == v for _, b in pairs) for v in values} if kind in SCALE_KINDS else {},
            "figures": figures, "flags": flags,
            "counts": stats.get("confusion", stats.get("counts", {})),
            "without_unsure": without, "strata": by_stratum}


def _reported_as(variable: str, reporting: Reporting) -> list[tuple[str, str, list | None, str | None, object]]:
    """(name, kind, values, positive, derivation) of each row a pooled variable gives: its own when the reporting
    names its kind (keyed by the YAML's name: a Stage A item and a prepared code by bare id, a pair dimension as
    `pair:<item>`), and each derived row computed from it (PR-FRAME from PR-CONCEPT, via frame_of)."""
    from pipeline.stage_b.review_report import REPORTING_DERIVATIONS
    out = []
    kind = reporting.kind_of(variable)
    if kind is not None:
        name = variable if variable.startswith("pair:") else reporting.aliases.get(variable, variable)
        out.append((name, kind, reporting.values_of(variable), reporting.positive_of(variable), None))
    code = reporting.aliases.get(variable, variable)
    for name, entry in reporting.derived.items():
        if entry["from"] == code:
            out.append((name, entry["kind"], list(entry["values"]), None, REPORTING_DERIVATIONS[entry["via"]]))
    return out


def _report_rows(groups: dict[str, list[tuple]], reporting: Reporting, raters: tuple[str, str], n_boot: int, seed: int,
                 strata: dict | None) -> dict:
    """{name: row} of the reported variables among `groups` ({variable: [(video, first, second, unsure, path)]}, the
    raw values). A row's units are the pairs whose two values are on its scale (a readable clock, a non-empty ask)."""
    out = {}
    for variable in sorted(groups):
        for name, kind, values, positive, via in _reported_as(variable, reporting):
            if kind == "comparison":
                continue
            units, off, distinct = [], 0, set()
            for video, a, b, unsure, path, *context in groups[variable]:
                if kind in SCALE_KINDS:
                    x, y = _on_scale(values, path, a, via), _on_scale(values, path, b, via)
                elif kind == "clock":
                    x, y = _seconds(a), _seconds(b)
                    if x is not None and y is not None:
                        reference = context[0] if context else None
                        x, y = _TurnClock(x, reference), _TurnClock(y, reference)
                else:
                    x, y = _text(a), _text(b)
                if x is None or y is None:
                    off += 1
                else:
                    units.append((video, x, y, unsure))
                    distinct.add((video, path))
            path = groups[variable][0][4] if groups[variable] else None
            out[name] = _report_row(kind, units, off, values, positive, path, raters, reporting, n_boot, seed, strata,
                                    n_units=len(distinct))
    return out


def _comparison_row(f: dict, who: str, raters: tuple[str, str], reporting: Reporting) -> dict | None:
    """A pair dimension's row from review_clips.pooled_pairs' figures: coder against model (`who` "coder_model") or
    coder against coder ("coder_coder"): k of n judgements with the 95% interval resampling videos (review_clips'
    `ci95`: pairs share clips, so it is the interval that respects their dependence), the Clopper-Pearson interval
    beside it ("binomial_interval", which treats the judgements as independent), the exact two-sided p against 0.5
    (`p_vs_half`, which records the level as "against"), the same counts per third of the pairs by the model's gap,
    the "close" ticks. "units" are the pairs of clips (each judged once per coder; against the model, the pairs the
    model ranked); the small-n flag counts them. A `comparisons.against` other than 0.5 is refused (ValueError; the
    loader refuses it first): the figure is named for a coin flip, and a test against another level would need a
    statistic of its own name."""
    share, n = f.get(who), (f.get("n") if who == "coder_model" else f.get("n_both"))
    if n is None or (who == "coder_coder" and share is None):
        return None
    k = round(share * n) if share is not None and n else 0
    against = reporting.comparisons["against"]
    if against != 0.5:
        raise ValueError(f"comparisons.against is {against}: the comparison's p is reported as p_vs_half, against 0.5 "
                         "only")
    if n:
        low, high = ag.binomial_interval(k, n)
        p = ag.binomial_p_vs_half(k, n)
        value = k / n
    else:
        low = high = p = value = None
    resampled = (f.get("ci95") or {}).get(who) if n else None
    # the pairs of clips behind the judgements: against the model, the pairs anyone answered that the model ranked
    units = (f.get("n_pairs", n) - (f.get("n_model_ties") or 0)) if who == "coder_model" else n
    keys = ("agree", "n_judged") if who == "coder_model" else ("coder_coder_agree", "n_both")
    terciles = [{"tercile": t.get("tercile"), "agree": t.get(keys[0]), "n": t.get(keys[1]),
                 "gap_range": t.get("gap_range")} for t in f.get("by_gap_tercile") or []]
    close = f.get("n_close")
    stats = {"agreement": {"value": value, "low": resampled[0] if resampled else None,
                           "high": resampled[1] if resampled else None},
             "binomial_interval": {"value": [low, high] if n else None, "low": low, "high": high},
             "p_vs_half": {"value": p, "low": None, "high": None, "against": against},
             "by_gap_tercile": {"value": terciles, "low": None, "high": None},
             "close": {"value": close, "low": None, "high": None}}
    figures = {name: stats[name] for name in reporting.statistics_of("comparison")}
    return {"kind": "comparison", "n": n, "units": units, "videos": f.get("videos"), "raters": list(raters),
            "n_off_scale": 0, "coder_values": {}, "model_values": {}, "figures": figures,
            "flags": ["small_n"] if units < reporting.flags["small_n"] else [],
            "counts": {"agree": k, "n": n, "close": close}, "without_unsure": {}, "strata": {}}


def reporting_summary(videos: dict[str, list[dict]], by_reviewer: dict[str, dict[str, list[dict]]], pairs: dict,
                      reporting: Reporting, n_boot: int = N_BOOT, seed: int = SEED, strata: dict | None = None) -> dict:
    """pooled_summary["reporting"] (plan D): {name: row} coder against model for every reported variable of the
    sampled rows (`_report_row`'s shape), "human_human" the same between two coders (the videos exactly two coders
    judged, the rows both gave a definite value) and "pass_to_pass" the model's primary against its replicate (each
    field once). A coder's value is the line's (a confirmation the primary's; for the first decision the clock the
    coder typed, when the line carries it); the model's the line's `primary`. PR-CONCEPT gives the frame's row
    (PR-FRAME), never its own; a pair dimension's row (`pair:<item>`) comes from `pairs`. In "figures", every share
    and chance-corrected statistic carries a 95% interval resampling videos (from MIN_CLUSTER_VIDEOS videos); the
    figures without unsure units and per stratum are points. "n" counts judgements and "units" the distinct units
    (pitches, question-answer pairs), which differ when two coders judged the same unit. NaN is written as None."""
    of = (strata or {}).get("of") or {}
    stratified = None
    if strata and any(v in of for v in videos):
        present = {v: of[v] for v in videos if v in of}
        stratified = {"of": present, "run": strata["run"], "weights": stratum_weights(present, strata["run"])}
    coder_model: dict[str, list[tuple]] = {}
    passes: dict[str, dict[tuple[str, str], tuple]] = {}
    for video_id, lines in sorted(videos.items()):
        for line in lines:
            judged = _judgment(Correction(line["path"], line.get("new"), line["resolution"], line.get("note") or "",
                                          ""), line)
            if judged[0] != "item" or judged[1] != "sampled":
                continue
            human, primary, replicate, path = judged[3][:4]
            if is_cut(path) and line["resolution"] == "confirm_primary" and _text(line.get("typed")) is not None:
                human = _text(line.get("typed"))
            variable, unsure = variable_of(path), is_unsure(line)
            context = line.get('decision_turn_context')
            coder_model.setdefault(variable, []).append((video_id, human, primary, unsure, path, context))
            if _text(replicate) is not None:
                before = passes.setdefault(variable, {}).get((video_id, path))
                passes[variable][(video_id, path)] = (video_id, primary, replicate,
                                                      unsure or bool(before and before[3]), path, context)
    coders: dict[str, dict[str, dict[str, dict]]] = {}
    for name, per in by_reviewer.items():
        for video_id, lines in per.items():
            coders.setdefault(video_id, {})[name] = {line["path"]: line for line in lines}
    two = {v: got for v, got in coders.items() if len(got) == 2}
    names = {tuple(sorted(got)) for got in two.values()}
    hh_raters = names.pop() if len(names) == 1 else ("first coder", "second coder")
    coder_coder: dict[str, list[tuple]] = {}
    for video_id, got in sorted(two.items()):
        a, b = (got[n] for n in sorted(got))
        items, _ = _coder_pairs(a, b)
        for _, x in items:
            path = x[3]
            coder_coder.setdefault(variable_of(path), []).append(
                (video_id, x[0], x[1], is_unsure(a[path]) or is_unsure(b[path]), path,
                 a[path].get('decision_turn_context') if a[path].get('decision_turn_context') ==
                 b[path].get('decision_turn_context') else None))
    out = _report_rows(coder_model, reporting, ("coder", "model"), n_boot, seed, stratified)
    human = _report_rows(coder_coder, reporting, hh_raters, n_boot, seed, stratified)
    pass_to_pass = _report_rows({v: list(per.values()) for v, per in passes.items()}, reporting,
                                ("primary", "replicate"), n_boot, seed, stratified)
    for item, f in sorted((pairs or {}).items()):
        name = f"{PAIR}:{item}"
        if reporting.kind_of(name) != "comparison":
            continue
        row = _comparison_row(f, "coder_model", ("coder", "model"), reporting)
        if row is not None:
            out[name] = row
        named = sorted(f.get("coder_model_by_coder") or {})
        row = _comparison_row(f, "coder_coder", tuple(named) if len(named) == 2 else ("first coder", "second coder"),
                              reporting)
        if row is not None:
            human[name] = row
    return {**out, "human_human": human, "pass_to_pass": pass_to_pass}
