"""A panel run as tidy tables, the pitch outcome, and the history cut of every pitch.

The model annotated each whole video, decisions included. Nothing is withheld here either: these
tables hold everything, and leakage is prevented by the cut. `pitch_cut` gives, per pitch, the turn
position at which its history stops (the first decision of any kind by any investor, so that the cut
never depends on the outcome), and `eligible_turns` is the only door through which features read
turns: strictly before the cut, never a `decides` or `negotiates` turn, and never a turn of a
negotiation exchange. `pitch_outcome` reads whether any investor offered, a human pitch override
winning over the record. The tables include the prepared presentation's ratings (`prepared`) and the
ask (on `videos`), both rated once per pitch since v3-dev9.

The per-investor outcomes, cuts and history (the retired shark-level design) live in
`analysis.archive.shark_level_dataset`.

`answer_followups` derives what a questioner does after an answer (accepts it, asks again, challenges,
or no investor reacts) from turn order, for the whole encounter or up to a cut. The model's grouping of
turns into exchanges is not read for it (the researcher's decision of 2026-10-21: the grouping is
unstable between passes); only the two `*_exchange` follow-up columns, kept as an unreported
sensitivity, read it. A documented limit of turn order: a delayed reaction, an investor coming back to
an earlier answer two question-answer pairs later, is attributed to the nearest preceding answer.

Read-only and offline: a run is read from disk (the human-reviewed record of a video when there is
one, otherwise its primary pass) and nothing is written. `load_run(..., passes="both")` also reads the
second annotation pass of every video where it loads as a record, as `PanelData.replicate`, for the sensitivity
that averages both passes; `development_video_ids` names the videos read while the instrument was
revised, for the sensitivity without them. Values are read in the spelling of the
instrument the run was annotated with (the copy in each video's bundle folder) when it is there,
otherwise in that of the live release; `PanelData.books_source` says which.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.errors import ConfigError
from pipeline.instrument import DEFAULT_MANIFEST, load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import (
    CONCEPT_CODE_OF, PanelBooks, canonical_value, implies_offer, load_panel_books, makes_offer,
)

RATING_CODES = ("PR-SEV", "RS-RESP", "RS-EVID", "RS-UNCERT", "PREP-SUBS", "PREP-COH", "PREP-CTX")
RESPONSIVENESS = {"addresses": 1.0, "partial": 0.5, "deflects": 0.0}     # NONE and missing stay missing
BINARY = {"true": 1.0, "false": 0.0}
CUT_KINDS = ("offer", "joins_offer", "exit")
SOURCE_PRIMARY, SOURCE_REVIEWED, SOURCE_OPEN = "primary", "reviewed", "reviewed_with_open_issues"
SOURCE_REPLICATE = "replicate"      # the replicate pass, read when the primary fails validation and it fails less
MODEL_SOURCES = (SOURCE_PRIMARY, SOURCE_REPLICATE)
REVIEWED_SOURCES = (SOURCE_REVIEWED, SOURCE_OPEN)
OPEN_ISSUE_COLUMNS = ("reviewed_validation_errors", "review_structural_on_decisions")
BOOKS_BUNDLE, BOOKS_LIVE = "bundle", "live"
BUNDLE_DIR = "bundle"
# a human reviewer's verdict on whether an investor ever offered (scripts/review_panel.py), as ever_offer, and on
# whether a pitch drew any offer (reviewed.json["pitch_override"]["any_offer"]), as any_offer
OVERRIDE_VALUES = {"yes": 1.0, "no": 0.0, "unknown": np.nan}
OUTCOME_RECORD, OUTCOME_OVERRIDE = "record", "human_override"
PREPARED_NOT_RECORDED = "not_recorded"   # the reason of a prepared code of the book the record does not rate


def _makes_offer(decisions: pd.DataFrame) -> pd.Series:
    """Rows of a decisions table that make an offer (the shared rule: offer, joins_offer, or a re-entry with an offer id)."""
    return pd.Series([makes_offer(k, o) for k, o in zip(decisions.kind, decisions.offer_id)],
                     index=decisions.index, dtype=bool)


def _implies_offer(decisions: pd.DataFrame) -> pd.Series:
    """Rows that show their investor made an offer at some point: those that make one, and those that revise
    or withdraw one (the shared rule of pipeline.stage_b.panel_books.implies_offer)."""
    return pd.Series([implies_offer(k, o) for k, o in zip(decisions.kind, decisions.offer_id)],
                     index=decisions.index, dtype=bool)
EXCLUDED_FUNCTIONS = ("decides", "negotiates")       # never history, whoever speaks them
NEGOTIATION = "negotiation"
NOT_VIDEO_DIRS = ("analysis", "ranking")
RANKING_SCORES = Path("ranking") / "scores.json"
PASSES_BEST, PASSES_BOTH = "best", "both"
PASSES = (PASSES_BEST, PASSES_BOTH)
SINGLE_PASS_VIDEOS = "single_pass_videos"     # the key of books_source: {video id read from one pass only: why}
SECOND_PASS_MISSING = "missing"               # why: the other pass has no result file
SECOND_PASS_UNREADABLE = "unreadable"         # why: "unreadable: <exception type>", it is not a v3-dev9 PanelRecord
SELECTION_ALL = "selection_all100.json"
SELECTION_GLOB = "selection*.json"
DEVELOPMENT_KEY = "pilot_development_videos"

_COLUMNS = {
    "videos": {"video_id": object, "source": object, "release_id": object, "duration_s": float, "n_turns": int,
               "founder_gender": object, "primary_industry": object, "product_form": object,
               "customer_type": object, "prepared_seconds": float,
               "ask_amount_usd": float, "ask_equity_pct": float, "ask_missing": object,
               "reviewed_validation_errors": int, "review_structural_on_decisions": int},
    "actors": {"video_id": object, "actor_id": object, "role": object, "label": object},
    "turns": {"video_id": object, "turn_id": int, "pos": int, "speaker": object, "role": object, "start_s": float,
              "end_s": float, "function": object, "addressed": object, "exchange_id": object,
              "exchange_kind": object, "responds_to": object, "concept": object, "frame": object,
              "frame_match": object, **{code: float for code in RATING_CODES}},
    "exchanges": {"video_id": object, "exchange_id": int, "kind": object, "topic": object, "label": object,
                  "first_pos": float, "last_pos": float, "n_turns": int},
    "decisions": {"video_id": object, "decision_id": int, "actor": object, "role": object, "kind": object,
                  "time_s": float, "turn_id": object, "pos": int, "offer_id": object},
    "stage_a": {"video_id": object, "code": object, "value": float, "source": object},
    # the prepared presentation codes of the book (PanelBooks.prepared_codes), one row per pitch and code
    "prepared": {"video_id": object, "code": object, "value": float, "missing": object},
}


@dataclass(frozen=True)
class PanelData:
    videos: pd.DataFrame
    actors: pd.DataFrame
    turns: pd.DataFrame
    exchanges: pd.DataFrame
    decisions: pd.DataFrame
    stage_a: pd.DataFrame
    prepared: pd.DataFrame
    # video_id -> {investor actor id: "yes" | "no" | "unknown"}: human verdicts that replace the outcome read
    # from the decision list (the decisions themselves are not edited)
    outcome_overrides: dict = field(default_factory=dict)
    # video_id -> any_offer (1.0 yes, 0.0 no, NaN unknown): a human verdict on the pitch that replaces the
    # any_offer read from the record; a pitch without one is absent
    pitch_overrides: dict = field(default_factory=dict)
    # the Stage A items of the instrument(s) the records were annotated with, in codebook order
    stage_a_codes: tuple = ()
    # video_id -> "bundle" (the run's own copy of its instrument) | "live" (the current release); with both passes
    # read (load_run(..., passes="both")), also SINGLE_PASS_VIDEOS -> {video id read from one pass only: why}, in id
    # order, why being "missing" or "unreadable: <exception type>"
    books_source: dict = field(default_factory=dict)
    # the second annotation pass of the videos where it reads as a record (load_run(..., passes="both")), as tables of its
    # own; None when one pass is read. Its own `replicate` is None.
    replicate: PanelData | None = None


def _table(name: str, rows: list[dict]) -> pd.DataFrame:
    """A table with its declared columns and dtypes; text columns keep None for a missing value."""
    return pd.DataFrame({column: pd.Series([r[column] for r in rows], dtype=dtype)
                         for column, dtype in _COLUMNS[name].items()})


def _rating(code: str, value: str | None, books: PanelBooks) -> float:
    """A rating as a number: RS-RESP on 0 / 0.5 / 1, a binary as 0 / 1, an ordinal as its integer."""
    if value is None:
        return np.nan
    value = canonical_value(books.panel.by_id(code), value)
    if code == "RS-RESP":
        return RESPONSIVENESS.get(value, np.nan)
    if value in BINARY:
        return BINARY[value]
    try:
        return float(int(value))
    except ValueError:
        return np.nan


def _video_tables(entry: dict, books: PanelBooks) -> dict[str, list[dict]]:
    from pipeline.stage_b.panel_pipeline import derive   # deferred: that module also loads the provider layer

    video_id = entry["video_id"]
    model = books.canonical_record(PanelRecord.model_validate(entry["record"]))
    derived = derive(model, books)
    record = model.model_dump(mode="json")
    role_of = {a["actor_id"]: a["role"] for a in record["actors"]}
    kind_of: dict[int, str] = {}
    for e in record["exchanges"]:
        kind_of.setdefault(e["exchange_id"], e["kind"])

    turns = []
    pos_of: dict[int, int] = {}
    for pos, t in enumerate(record["turns"]):
        pos_of.setdefault(t["turn_id"], pos)
        values: dict[str, str | None] = {}
        for c in t["codes"]:
            values.setdefault(c["code"], c["value"])
        concept_code = CONCEPT_CODE_OF.get(t["function"])
        concept = values.get(concept_code) if concept_code else None
        if concept is not None:
            concept = canonical_value(books.panel.by_id(concept_code), concept)
        turns.append({
            "video_id": video_id, "turn_id": t["turn_id"], "pos": pos, "speaker": t["speaker"],
            "role": role_of.get(t["speaker"]), "start_s": t["start_s"], "end_s": t["end_s"],
            "function": t["function"], "addressed": tuple(t["addressed"]), "exchange_id": t["exchange_id"],
            "exchange_kind": kind_of.get(t["exchange_id"]), "responds_to": tuple(t["responds_to"]),
            "concept": concept, "frame": derived["turn_frame"].get(str(t["turn_id"])),
            "frame_match": derived["frame_match"].get(str(t["turn_id"])),
            **{code: _rating(code, values.get(code), books) for code in RATING_CODES},
        })

    exchanges = []
    for e in record["exchanges"]:
        members = [t["pos"] for t in turns if t["exchange_id"] == e["exchange_id"]]
        exchanges.append({"video_id": video_id, "exchange_id": e["exchange_id"], "kind": e["kind"],
                          "topic": e["topic"], "label": e["label"],
                          "first_pos": min(members) if members else np.nan,
                          "last_pos": max(members) if members else np.nan, "n_turns": len(members)})

    decisions = []
    for d in record["decisions"]:
        pos = pos_of.get(d["turn_id"])
        if pos is None:
            # no turn speaks it: the last turn starting at or before its time, the first turn when none does
            started = [t["pos"] for t in turns if t["start_s"] <= d["time_s"]]
            pos = max(started) if started else 0
        decisions.append({"video_id": video_id, "decision_id": d["decision_id"], "actor": d["actor"],
                          "role": role_of.get(d["actor"]), "kind": d["kind"], "time_s": d["time_s"],
                          "turn_id": d["turn_id"], "pos": pos, "offer_id": d["offer_id"]})

    venture = record["venture"]
    video = {
        "video_id": video_id, "source": entry["source"], "release_id": entry.get("release_id"),
        "duration_s": entry.get("duration_s", np.nan), "n_turns": len(turns),
        "founder_gender": books.canonical_venture("VC-GENDER", venture["founder_gender"]),
        "primary_industry": books.canonical_venture("VC-INDUSTRY", venture["primary_industry"]),
        "product_form": books.canonical_venture("VC-FORM", venture["product_form"]),
        "customer_type": books.canonical_venture("VC-CUSTOMER", venture["customer_type"]),
        "prepared_seconds": float(sum(max(0.0, s["end_s"] - s["start_s"]) for s in record["prepared_spans"])),
        "ask_amount_usd": np.nan if venture["ask_amount_usd"] is None else float(venture["ask_amount_usd"]),
        "ask_equity_pct": np.nan if venture["ask_equity_pct"] is None else float(venture["ask_equity_pct"]),
        "ask_missing": venture["ask_missing"],
        **{name: int(entry.get(name) or 0) for name in OPEN_ISSUE_COLUMNS},
    }
    stage_a = [{"video_id": video_id, "code": item["code"],
                "value": np.nan if item["value"] is None else float(item["value"]), "source": "rating"}
               for item in record["stage_a"]]
    return {"videos": [video],
            "actors": [{"video_id": video_id, **a} for a in record["actors"]],
            "turns": turns, "exchanges": exchanges, "decisions": decisions,
            "stage_a": stage_a, "prepared": _prepared_rows(video_id, record["prepared"], books)}


def _prepared_rows(video_id: str, items: list[dict], books: PanelBooks) -> list[dict]:
    """One row per prepared code of the book, in book order: a code the record does not rate is NaN with the
    reason PREPARED_NOT_RECORDED; a code the record rates twice keeps its first rating; a code the book
    does not have is not read."""
    first: dict[str, dict] = {}
    for item in items:
        first.setdefault(item["code"], item)
    rows = []
    for entry in books.prepared_codes():
        item = first.get(entry.id)
        rows.append({"video_id": video_id, "code": entry.id,
                     "value": np.nan if item is None else _rating(entry.id, item["value"], books),
                     "missing": PREPARED_NOT_RECORDED if item is None else item["missing"]})
    return rows


def build_data(entries: list[dict], rank_scores: dict | None = None, books: PanelBooks | None = None) -> PanelData:
    """The tables of records already in memory.

    Each entry gives `video_id`, `record` (a PanelRecord dict), `source` ("reviewed", "primary" or
    "reviewed_with_open_issues") and optionally `release_id`, `duration_s`, `outcome_overrides` (a
    reviewed result's human verdicts on ever_offer, by investor id), `pitch_override` (a reviewed
    result's {"any_offer": "yes" | "no" | "unknown"}), the two open-issue counts of
    OPEN_ISSUE_COLUMNS, and `books` with `books_source` (the instrument that video was annotated
    with). `rank_scores` is the content of a run's ranking/scores.json. An entry without books of its
    own is read in the spelling of `books` (the live instrument release when not given).
    """
    live = None
    rows: dict[str, list[dict]] = {name: [] for name in _COLUMNS}
    stage_a_codes: dict[str, None] = {}
    books_source: dict[str, str] = {}
    for entry in sorted(entries, key=lambda e: e["video_id"]):
        own = entry.get("books") or books
        if own is None:
            own = live = live or load_panel_books(load_release())
        books_source[entry["video_id"]] = entry.get("books_source") or BOOKS_LIVE
        stage_a_codes.update(dict.fromkeys(own.stage_a_ids()))
        for name, part in _video_tables(entry, own).items():
            rows[name].extend(part)
    video_ids = {v["video_id"] for v in rows["videos"]}
    for code, block in sorted((rank_scores or {}).items()):
        if not isinstance(block, dict) or not isinstance(block.get("pl_score"), dict):
            continue        # scores.json also carries counts and labels beside the per-code blocks
        for video_id, score in sorted(block["pl_score"].items()):
            if video_id in video_ids:
                rows["stage_a"].append({"video_id": video_id, "code": code, "value": float(score),
                                        "source": "rank_pl"})
    overrides = {}
    for entry in entries:
        given = {str(actor): str(value).strip().casefold()
                 for actor, value in (entry.get("outcome_overrides") or {}).items()}
        unknown = sorted(v for v in given.values() if v not in OVERRIDE_VALUES)
        if unknown:
            raise ConfigError(f"{entry['video_id']}: outcome_overrides takes {', '.join(OVERRIDE_VALUES)}, "
                              f"not {', '.join(unknown)}")
        if given:
            overrides[entry["video_id"]] = given
    pitch_overrides = {}
    for entry in entries:
        given = (entry.get("pitch_override") or {}).get("any_offer")
        if given is None:
            continue
        value = str(given).strip().casefold()
        if value not in OVERRIDE_VALUES:
            raise ConfigError(f"{entry['video_id']}: pitch_override any_offer takes {', '.join(OVERRIDE_VALUES)}, "
                              f"not {given}")
        pitch_overrides[entry["video_id"]] = OVERRIDE_VALUES[value]
    return PanelData(**{name: _table(name, part) for name, part in rows.items()}, outcome_overrides=overrides,
                     pitch_overrides=pitch_overrides, stage_a_codes=tuple(stage_a_codes), books_source=books_source)


def _has_result(video_dir: Path) -> bool:
    results = video_dir / "results"
    return (results / "reviewed.json").exists() or (results / "primary.json").exists()


def video_dirs(run_root: Path | str, video_ids: list[str] | None = None) -> list[Path]:
    """The video folders of a run that analysis reads, in id order.

    Without `video_ids`: every folder holding a result (a video still being annotated has none and is
    left out). A named video without a result is an error.
    """
    run_root = Path(run_root)
    if not run_root.is_dir():
        raise ConfigError(f"run folder not found: {run_root}")
    if video_ids is None:
        return sorted(d for d in run_root.iterdir()
                      if d.is_dir() and d.name not in NOT_VIDEO_DIRS and _has_result(d))
    dirs = [run_root / video_id for video_id in sorted(set(video_ids))]
    for d in dirs:
        if not _has_result(d):
            raise ConfigError(f"{d.name}: no primary or reviewed result under {d / 'results'}")
    return dirs


def bundle_books(video_dir: Path | str, cache: dict | None = None) -> PanelBooks | None:
    """The books a video was annotated with, read from its bundle folder without writing; None when the
    folder does not hold a whole instrument (then the live release is the only one there is)."""
    bundle = Path(video_dir) / BUNDLE_DIR
    manifest = bundle / DEFAULT_MANIFEST.name
    if not manifest.exists():
        return None
    try:
        release = load_release(manifest)
        files = [release.books[key].yaml for key in sorted(release.books)]
        if not all(f.exists() for f in files):
            return None
        key = tuple(f.read_bytes() for f in files)
        if cache is not None and key in cache:
            return cache[key]
        books = load_panel_books(release, sync=False)
    except ConfigError:
        return None
    if cache is not None:
        cache[key] = books
    return books


def with_replicate(data: PanelData, replicate: PanelData, reasons: dict[str, str] | None = None) -> PanelData:
    """`data` with `replicate` as its second pass: `books_source` then also maps, under SINGLE_PASS_VIDEOS, every
    video of `data` that `replicate` does not have (read from one pass only) to why, in id order: its entry of
    `reasons`, or "missing" when `reasons` does not give one."""
    single = sorted(set(data.videos.video_id) - set(replicate.videos.video_id))
    why = {video_id: (reasons or {}).get(video_id, SECOND_PASS_MISSING) for video_id in single}
    return dataclasses.replace(data, replicate=replicate, books_source={**data.books_source, SINGLE_PASS_VIDEOS: why})


def _second_pass(video_dir: Path, best_source: str) -> tuple[dict | None, str]:
    """(result, "") for the other model pass of a video, or (None, why not): the replicate, or the primary when
    the best record is the replicate.

    It is read whenever it loads as a PanelRecord of the v3-dev9 shape, whatever its validation report
    lists: the main analysis reads records with validation errors too (`load_best_record` only prefers
    the pass with fewer), so dropping them here would make the both-passes sensitivity another sample.
    Why not: "missing" without a result file, "unreadable: <exception type>" when it does not parse as
    JSON or as such a record.
    """
    name = SOURCE_PRIMARY if best_source == SOURCE_REPLICATE else SOURCE_REPLICATE
    path = video_dir / "results" / f"{name}.json"
    if not path.exists():
        return None, SECOND_PASS_MISSING
    try:
        result = json.loads(path.read_text())
        PanelRecord.model_validate(result["record"])
        _check_record_shape(video_dir.parent, None, result["record"])
    except (KeyError, TypeError, ValueError) as error:   # JSONDecodeError and pydantic's ValidationError are ValueErrors
        return None, f"{SECOND_PASS_UNREADABLE}: {type(error).__name__}"
    return result, ""


def _check_record_shape(run, release_id, record: dict) -> None:
    """Refuse a record of an earlier instrument: one whose exchanges list investors, or one without the prepared
    presentation codes (both as up to v3-dev8). A v3-dev9 record's `prepared` codes and venture ask are what the
    prepared table and the ask columns of the videos table are read from."""
    if any("investors" in e for e in record.get("exchanges") or []) or "prepared" not in record:
        release = f"release {release_id}" if release_id else "an earlier release"
        raise ValueError(f"run {run} was annotated with {release}; this code reads v3-dev9 records")


def load_run(run_root: Path | str, video_ids: list[str] | None = None, passes: str = PASSES_BEST) -> PanelData:
    """The tables of a run on disk: per video the reviewed record when present, otherwise the primary pass, or the
    replicate pass when the primary has validation errors and the replicate has fewer.

    A reviewed record that still fails validation, or whose review log holds a structural or
    present/absent judgment on a decision, is read all the same and marked: `source` is then
    `reviewed_with_open_issues`, and the videos table counts both kinds of open issue. Nothing is
    excluded here. A reviewed result's `pitch_override` (a human verdict on any_offer) is read into
    `pitch_overrides`.

    `passes="both"` (the sensitivity averaging both passes) reads the same tables and, as `replicate`,
    the tables of the other model pass of every video where it loads as a record (`_second_pass`: the
    replicate, or the primary when the record read is the replicate), source "replicate", whatever its
    validation report lists. A video whose other pass is missing or does not load as a record is read
    from its one record, and `books_source["single_pass_videos"]` maps it to why ("missing" or
    "unreadable: <exception type>"). Each pass keeps its own decisions, so each is cut at its own first
    decision.
    """
    from pipeline.stage_b.panel_pipeline import load_best_record, review_open_issues   # deferred, see _video_tables

    if passes not in PASSES:
        raise ValueError(f"passes takes {' or '.join(PASSES)}, not {passes!r}")
    run_root = Path(run_root)
    entries, second, reasons = [], [], {}
    live = None
    cache: dict = {}
    for vdir in video_dirs(run_root, video_ids):
        result, source = load_best_record(vdir)
        issues = review_open_issues(vdir) if source == SOURCE_REVIEWED else dict.fromkeys(OPEN_ISSUE_COLUMNS, 0)
        if any(issues.values()):
            source = SOURCE_OPEN
        books = bundle_books(vdir, cache)
        books_source = BOOKS_BUNDLE
        if books is None:
            books = live = live or load_panel_books(load_release())
            books_source = BOOKS_LIVE
        experiment_path = vdir / "experiment.json"
        experiment = json.loads(experiment_path.read_text()) if experiment_path.exists() else {}
        _check_record_shape(run_root, experiment.get("release_id"), result["record"])
        entry = {"video_id": vdir.name, "record": result["record"], "source": source, **issues,
                 "books": books, "books_source": books_source,
                 "outcome_overrides": result.get("outcome_overrides") or {},
                 "pitch_override": result.get("pitch_override") or {},
                 "release_id": experiment.get("release_id"),
                 "duration_s": float(experiment.get("duration_s", np.nan))}
        entries.append(entry)
        other, why = _second_pass(vdir, source) if passes == PASSES_BOTH else (None, "")
        if why:
            reasons[vdir.name] = why
        if other is not None:
            # the outcome is read from the first pass only, so the second carries no human verdict
            second.append({**entry, "record": other["record"], "source": SOURCE_REPLICATE,
                           **dict.fromkeys(OPEN_ISSUE_COLUMNS, 0), "outcome_overrides": {}, "pitch_override": {}})
    scores_path = run_root / RANKING_SCORES
    rank_scores = json.loads(scores_path.read_text()) if scores_path.exists() else None
    data = build_data(entries, rank_scores)
    return with_replicate(data, build_data(second, rank_scores), reasons) if passes == PASSES_BOTH else data


def development_videos(run_root: Path | str, selection: Path | str | None = None) -> tuple[set[str], list[Path]]:
    """(development video ids, every selection file opened to find them), for the sensitivity without the pilot
    development videos.

    With `selection` (the selection file the analysis was limited to, e.g. `--selection`), its
    `pilot_development_videos` when it holds that key. Otherwise, or when it does not, the run root's:
    `<run_root>/selection_all100.json` when it holds the key, else every other `selection*.json` there that holds it
    (their union). An empty set when no file names any, so nothing is known to exclude. The files are listed in the
    order they were opened (`selection` first), each once, so a manifest can hash every one read."""
    run_root = Path(run_root)
    opened: list[Path] = []

    def read(path: Path):
        if not path.is_file():
            return None
        if path not in opened:
            opened.append(path)
        return json.loads(path.read_text())

    if selection is not None:
        content = read(Path(selection))
        if isinstance(content, dict) and DEVELOPMENT_KEY in content:
            return {str(v) for v in content[DEVELOPMENT_KEY]}, opened
    first = run_root / SELECTION_ALL
    others = [first] + sorted(p for p in run_root.glob(SELECTION_GLOB) if p != first)
    found: set[str] = set()
    for path in others:
        content = read(path)
        if isinstance(content, dict) and DEVELOPMENT_KEY in content:
            found.update(str(v) for v in content[DEVELOPMENT_KEY])
            if path == first:
                break
    return found, opened


def development_video_ids(run_root: Path | str, selection: Path | str | None = None) -> set[str]:
    """The development video ids of `development_videos` (the files read left out)."""
    return development_videos(run_root, selection)[0]


def _investors(data: PanelData) -> pd.DataFrame:
    return data.actors[data.actors.role == "investor"]


def _ordered_decisions(data: PanelData) -> pd.DataFrame:
    return data.decisions.sort_values(["video_id", "pos", "decision_id"], kind="stable")


def _ever_offer(data: PanelData, decisions: pd.DataFrame, video_id: str, actor: str) -> float:
    """Whether one investor ever offered: 1 when a decision implies an offer, 0 after an exit without one,
    missing otherwise; that investor's human verdict in data.outcome_overrides in its place when given."""
    override = data.outcome_overrides.get(video_id, {}).get(actor)
    if override is not None:
        return OVERRIDE_VALUES[override]
    own = decisions[decisions.actor == actor]
    return 1.0 if _implies_offer(own).any() else (0.0 if (own.kind == "exit").any() else np.nan)


def pitch_outcome(data: PanelData) -> pd.DataFrame:
    """One row per pitch: any_offer, first_offer_pos and any_offer_source, read from the decision list.

    any_offer is 1 when any investor made or joined an offer at any time (also after an exit and
    re-entry; also when the record holds only the revision or withdrawal of an offer, which must have
    existed), 0 when every investor exited and never offered, missing otherwise. An investor's human
    verdict in data.outcome_overrides counts in place of what the record says of that investor. A human
    pitch verdict in data.pitch_overrides wins over all of it (yes 1, no 0, unknown missing) and marks the
    row any_offer_source = "human_override" (otherwise "record"). first_offer_pos, like the cut, still
    describes the record.
    """
    decisions = _ordered_decisions(data)
    investors = _investors(data)
    rows = []
    for video_id in data.videos.video_id:
        in_video = decisions[(decisions.video_id == video_id) & (decisions.role == "investor")]
        offers = in_video[_makes_offer(in_video)]
        ever = pd.Series([_ever_offer(data, in_video, video_id, actor)
                          for actor in investors.actor_id[investors.video_id == video_id]], dtype=float)
        if (ever == 1.0).any():
            any_offer = 1.0
        elif len(ever) and (ever == 0.0).all():
            any_offer = 0.0
        else:
            any_offer = np.nan
        source = OUTCOME_RECORD
        if video_id in data.pitch_overrides:
            any_offer, source = data.pitch_overrides[video_id], OUTCOME_OVERRIDE
        rows.append({"video_id": video_id, "any_offer": any_offer,
                     "first_offer_pos": float(offers.pos.min()) if len(offers) else np.nan,
                     "any_offer_source": source})
    return pd.DataFrame({
        "video_id": pd.Series([r["video_id"] for r in rows], dtype=object),
        "any_offer": pd.Series([r["any_offer"] for r in rows], dtype=float),
        "first_offer_pos": pd.Series([r["first_offer_pos"] for r in rows], dtype=float),
        "any_offer_source": pd.Series([r["any_offer_source"] for r in rows], dtype=object),
    })


def pitch_cut(data: PanelData) -> pd.DataFrame:
    """One row per pitch: the turn position at which its history stops.

    The position of the first decision of any kind by any investor, an exit as much as an offer: a
    cut at the first offer would give the pitches without an offer the whole encounter as history and
    let the length and content of the history tell the outcome. Without such a decision the cut is the
    number of turns, so the whole encounter is history.
    """
    n_turns = dict(zip(data.videos.video_id, data.videos.n_turns))
    decisions = data.decisions[data.decisions.role == "investor"]
    rows = []
    for video_id in data.videos.video_id:
        in_video = decisions[decisions.video_id == video_id]
        rows.append((video_id, int(in_video.pos.min()) if len(in_video) else int(n_turns[video_id])))
    return pd.DataFrame({"video_id": pd.Series([r[0] for r in rows], dtype=object),
                         "cut_pos": pd.Series([r[1] for r in rows], dtype=int)})


def eligible_turns(data: PanelData, video_id: str, cut_pos: int) -> pd.DataFrame:
    """The turns a row's features may read: the history before `cut_pos` minus near-outcome talk."""
    t = data.turns
    keep = ((t.video_id == video_id) & (t.pos < cut_pos) & ~t.function.isin(EXCLUDED_FUNCTIONS)
            & (t.exchange_kind != NEGOTIATION))
    return t[keep]


REASON_AFTER_CUT = "at or after the cut (the decision and everything that follows it)"
REASON_DECISION_TALK = "a turn that states or negotiates a decision"
REASON_NEGOTIATION = "part of a negotiation exchange"


def exclusion_reasons(data: PanelData, video_id: str, cut_pos: int) -> pd.Series:
    """For every turn of a video, why a row's features may not read it; empty text when they may.

    The same rule as `eligible_turns`, spelled out turn by turn: the turns with an empty reason are
    exactly the ones that function returns. When several reasons hold the first of these is given:
    the cut, a decides or negotiates turn, a negotiation exchange.
    """
    t = data.turns[data.turns.video_id == video_id]
    reason = pd.Series("", index=t.index, dtype=object)
    for mask, text in ((t.exchange_kind == NEGOTIATION, REASON_NEGOTIATION),
                       (t.function.isin(EXCLUDED_FUNCTIONS), REASON_DECISION_TALK),
                       (t.pos >= cut_pos, REASON_AFTER_CUT)):        # the last assignment wins: the cut first
        reason[mask] = text
    return reason


# ---- what the questioner does after an answer -----------------------------------------------

RECEIPT_FUNCTIONS = ("accepts",)
# turns that are passed over when looking for the next move: they are not a reaction to the answer
SKIPPED_NEXT_FUNCTIONS = ("NONE", "social", "demonstrates")
POSITIVE_RECEIPT, FOLLOW_UP, CHALLENGE = "positive_receipt", "follow_up_question", "challenge"
OTHER_BY_ASKER, OTHER_INVESTOR, NO_REACTION = "other_by_asker", "other_investor", "no_reaction"
NEXT_MOVES = (POSITIVE_RECEIPT, FOLLOW_UP, CHALLENGE, OTHER_BY_ASKER, OTHER_INVESTOR, NO_REACTION)
AFTER_CUT = "after_cut"          # only with a cut: the next move exists but the row may not know it
HARD_SEVERITY = 2.0              # a question that asks for substantiation (2) or contests a claim (3)
ANSWER_CODES = {"resp": "RS-RESP", "evid": "RS-EVID", "uncert": "RS-UNCERT", "subs": "PREP-SUBS",
                "coh": "PREP-COH", "ctx": "PREP-CTX"}
FOLLOWUP_COLUMNS = (
    "video_id", "answer_turn", "answer_pos", "answerer", "answer_start_s", "question_turn", "question_pos",
    "question_start_s", "asker", "asker_label", "exchange_id", "question_sev", "hard_question", *ANSWER_CODES,
    "next_move", "next_turn", "next_pos", "next_start_s", "next_actor", "next_function", "next_question_sev",
    "escalation", "turns_to_next", "next_question_same_exchange", "further_questions_in_exchange",
    "next_question_same_frame", "further_questions_same_frame", "next_question_same_concept")
_FOLLOWUP_FLOATS = ("answer_start_s", "question_start_s", "question_sev", *ANSWER_CODES, "next_turn", "next_pos",
                    "next_start_s", "next_question_sev", "escalation", "turns_to_next", "next_question_same_exchange",
                    "next_question_same_frame", "next_question_same_concept")
_FOLLOWUP_INTS = ("further_questions_in_exchange", "further_questions_same_frame")
COMPARED_FRAMES = ("promotion", "prevention")     # unclassified (deal_terms, other) and NEUTRAL are never compared


def _no_exchange(exchange_id) -> bool:
    return exchange_id is None or exchange_id != exchange_id


def _next_investor_turn(turns: list[dict], answer: dict) -> dict | None:
    """The first investor turn after `answer`, in turn order whatever its exchange, that could be a reaction.

    Founder turns and investor turns whose function is in SKIPPED_NEXT_FUNCTIONS are passed over. A
    `decides`/`negotiates` turn is returned like any other, so that `_move` reports it as no reaction.
    """
    for t in turns[answer["pos"] + 1:]:
        if t["role"] == "investor" and t["function"] not in SKIPPED_NEXT_FUNCTIONS:
            return t
    return None


UNCOMPARED_CONCEPTS = ("NONE", "other")      # two `other` questions are not known to be the same matter


def _comparable_concept(concept) -> bool:
    return isinstance(concept, str) and concept not in UNCOMPARED_CONCEPTS


def _later_questions(turns: list[dict], answer: dict, question: dict, cut_pos) -> tuple[float, int, float, int, float]:
    """(next_question_same_exchange, further_questions_in_exchange, next_question_same_frame,
    further_questions_same_frame, next_question_same_concept) of an answer to `question`; see `answer_followups`."""
    later = [t for t in turns[answer["pos"] + 1:] if t["role"] == "investor" and t["function"] == "question"
             and (cut_pos is None or t["pos"] < cut_pos)]
    frame = question["frame"]
    if later and frame in COMPARED_FRAMES and later[0]["frame"] in COMPARED_FRAMES:
        same_frame = float(later[0]["frame"] == frame)
        run = 0
        while run < len(later) and later[run]["frame"] == frame:
            run += 1
    else:
        same_frame, run = np.nan, 0
    concept = question["concept"]
    if later and _comparable_concept(concept) and _comparable_concept(later[0]["concept"]):
        same_concept = float(later[0]["concept"] == concept)
    else:
        same_concept = np.nan
    exchange = answer["exchange_id"]
    if _no_exchange(exchange):
        return np.nan, 0, same_frame, run, same_concept
    def in_exchange(t: dict) -> bool:
        return not _no_exchange(t["exchange_id"]) and t["exchange_id"] == exchange
    first = float(in_exchange(later[0])) if later else np.nan
    return first, sum(in_exchange(t) for t in later), same_frame, run, same_concept


def _move(found: dict | None, asker: str, cut_pos) -> str:
    if found is None:
        return NO_REACTION
    if cut_pos is not None and found["pos"] >= cut_pos:
        return AFTER_CUT                     # before the decision test: a decision at the cut is the cut
    if found["function"] in EXCLUDED_FUNCTIONS or found["exchange_kind"] == NEGOTIATION:
        return NO_REACTION                   # decision talk and negotiation are never a reaction to an answer
    if found["speaker"] != asker:
        return OTHER_INVESTOR
    if found["function"] in RECEIPT_FUNCTIONS:
        return POSITIVE_RECEIPT
    return {"question": FOLLOW_UP, "challenges": CHALLENGE}.get(found["function"], OTHER_BY_ASKER)


def answer_followups(data: PanelData, cut: dict | None = None, video_ids=None) -> pd.DataFrame:
    """What the questioner does next: one row per answer turn and investor question it responds to.

    Few questions are linked to the answer they follow, so the next move is read from turn order: after
    the answer, the first investor turn, whatever exchange the model put it in, passing over founder
    turns (continuations, a co-founder adding to the answer) and turns that are no reaction (function
    NONE, social turns, demonstrations). `next_move` is then

    - `positive_receipt`: the asker accepts (an `accepts` turn);
    - `follow_up_question`: the asker asks again (`escalation` is that question's severity minus the
      severity of the question answered);
    - `challenge`: the asker challenges;
    - `other_by_asker`: any other turn by the asker;
    - `other_investor`: another investor speaks first (`next_actor`, `next_function` say who and what);
    - `no_reaction`: no further investor turn, or the first one states or negotiates a decision or lies
      in a negotiation exchange, which never counts as a reaction to an answer.

    Turn order ties a reaction to the nearest preceding answer: an investor who comes back to an
    earlier answer two question-answer pairs later (a delayed reaction) is read as reacting to the
    latest answer. This is a documented limit of reading reactions without the exchange grouping.

    An answer that responds to no investor question (to a challenge, say) has no row; one linked to
    several questions has a row per question. `hard_question` is a severity of 2 or 3.

    Two columns look past the next move to the questions that follow the answer anywhere in the pitch:
    `next_question_same_exchange` is 1.0 when the next investor question after the answer (whoever asks
    it, passing over every other turn) belongs to the answer's exchange and 0.0 when it opens or returns
    to another one; NaN when no investor question follows, or the answer lies outside any exchange.
    `further_questions_in_exchange` counts the investor questions of the answer's exchange after the
    answer (0 for an answer outside any exchange). These two read the model's exchange grouping and are
    kept as an unreported sensitivity only.

    Three columns read the same later questions without the grouping. `next_question_same_frame` is 1.0
    when the next investor question has the frame of the question answered and 0.0 when it has the other
    one; NaN when no question follows or either frame is not promotion or prevention (an `unclassified`
    concept, deal_terms or other, or NEUTRAL is never counted as a switch). `further_questions_same_frame`
    is the number of later questions, from the next one on, that keep the answered question's frame
    without a break (0 when `next_question_same_frame` is NaN). `next_question_same_concept` is 1.0 when
    the next investor question has the answered question's concept and 0.0 when it has another; NaN when
    no question follows or either concept is missing, NONE or `other` (two `other` questions are not
    known to be the same matter). With a cut, all five read only questions
    before the asker's cut.

    `cut` maps video_id -> {investor -> cut_pos}. For a row whose asker has a cut, a next move at or
    after that position is reported as `after_cut` with every `next_*` column, `escalation` and
    `turns_to_next` empty (a decision turn at the cut is `after_cut`, not `no_reaction`): this is the
    variant features read, and `after_cut` is never a predictor.
    Without `cut` the whole encounter is read, which is descriptive. `video_ids` limits the videos.
    """
    turns = data.turns if video_ids is None else data.turns[data.turns.video_id.isin(list(video_ids))]
    labels = {(a.video_id, a.actor_id): a.label for a in data.actors.itertuples()}
    rows = []
    for video_id, in_video in turns.groupby("video_id", sort=True):
        ordered = in_video.sort_values("pos").to_dict("records")
        by_id: dict[int, dict] = {}
        for t in ordered:
            by_id.setdefault(t["turn_id"], t)
        video_cut = (cut or {}).get(video_id) or {}
        for answer in ordered:
            if answer["function"] != "answer":
                continue
            found = _next_investor_turn(ordered, answer)
            for question_id in dict.fromkeys(answer["responds_to"]):
                question = by_id.get(question_id)
                if question is None or question["function"] != "question" or question["role"] != "investor":
                    continue
                asker = question["speaker"]
                move = _move(found, asker, video_cut.get(asker))
                same_exchange, further, same_frame, further_frame, same_concept = _later_questions(
                    ordered, answer, question, video_cut.get(asker))
                shown = found if move not in (NO_REACTION, AFTER_CUT) else None
                follow_up_sev = shown["PR-SEV"] if move == FOLLOW_UP else np.nan
                severity = question["PR-SEV"]
                rows.append({
                    "video_id": video_id, "answer_turn": answer["turn_id"], "answer_pos": answer["pos"],
                    "answerer": answer["speaker"], "answer_start_s": answer["start_s"],
                    "question_turn": question["turn_id"], "question_pos": question["pos"],
                    "question_start_s": question["start_s"], "asker": asker,
                    "asker_label": labels.get((video_id, asker)),
                    "exchange_id": None if _no_exchange(answer["exchange_id"]) else answer["exchange_id"],
                    "question_sev": severity, "hard_question": bool(severity >= HARD_SEVERITY),
                    **{name: answer[code] for name, code in ANSWER_CODES.items()},
                    "next_move": move,
                    "next_turn": shown["turn_id"] if shown else np.nan,
                    "next_pos": shown["pos"] if shown else np.nan,
                    "next_start_s": shown["start_s"] if shown else np.nan,
                    "next_actor": shown["speaker"] if shown else None,
                    "next_function": shown["function"] if shown else None,
                    "next_question_sev": follow_up_sev, "escalation": follow_up_sev - severity,
                    "turns_to_next": shown["pos"] - answer["pos"] if shown else np.nan,
                    "next_question_same_exchange": same_exchange, "further_questions_in_exchange": further,
                    "next_question_same_frame": same_frame, "further_questions_same_frame": further_frame,
                    "next_question_same_concept": same_concept})
    table = pd.DataFrame(rows, columns=list(FOLLOWUP_COLUMNS))
    for name in _FOLLOWUP_FLOATS:
        table[name] = table[name].astype(float)
    for name in ("video_id", "answerer", "asker", "asker_label", "exchange_id", "next_move", "next_actor",
                 "next_function"):
        table[name] = table[name].astype(object).where(table[name].notna(), None)
    table["hard_question"] = table["hard_question"].astype(bool)
    for name in _FOLLOWUP_INTS:
        table[name] = table[name].astype(np.int64)
    return table


# ---- H4: the next move after an answer, counted ---------------------------------------------

PANEL_RECEIPT, PANEL_QUESTION, PANEL_CHALLENGE, PANEL_OTHER, PANEL_NONE = "receipt", "question", "challenge", "other", "none"
PANEL_MOVES = (PANEL_RECEIPT, PANEL_QUESTION, PANEL_CHALLENGE, PANEL_OTHER, PANEL_NONE)
ALL_QUESTIONS, HARD_QUESTIONS = "all", "hard"
NOT_RATED = "not_rated"
SUBSTANCE_LEVELS = ("1", "2", "3", "4", "5", NOT_RATED)
SUPPORT_LEVELS = ("yes", "no", NOT_RATED)
H4_TABLES = ("asker", "asker_by_substance", "asker_by_support", "asker_by_video",
             "panel", "panel_by_substance", "panel_by_support", "panel_by_video")


def panel_next_move(followups: pd.DataFrame) -> pd.Series:
    """The panel's next move per row of `answer_followups`: the first investor reaction by anyone.

    The asker's move and another investor's are read alike, from `next_function`: `receipt` (accepts),
    `question`, `challenge`, `other` (any other turn), and `none` when no investor reacts (the asker's
    `no_reaction`). A row cut off by a history cut stays `after_cut`.
    """
    def move(next_move, function) -> str:
        if next_move in (NO_REACTION, AFTER_CUT):
            return PANEL_NONE if next_move == NO_REACTION else AFTER_CUT
        if function in RECEIPT_FUNCTIONS:
            return PANEL_RECEIPT
        return {"question": PANEL_QUESTION, "challenges": PANEL_CHALLENGE}.get(function, PANEL_OTHER)
    return pd.Series([move(m, f) for m, f in zip(followups.next_move, followups.next_function)],
                     index=followups.index, dtype=object)


def substance_level(value) -> str:
    """An answer's substance rating as a table level: "1" to "5", or "not_rated"."""
    return NOT_RATED if value is None or value != value else str(int(value))


def support_level(value) -> str:
    """An answer's checkable support as a table level: "yes", "no", or "not_rated"."""
    return NOT_RATED if value is None or value != value else ("yes" if value == 1.0 else "no")


def _crossed(rows: pd.DataFrame, by: str | None, levels, moves: tuple, move_column: str) -> pd.DataFrame:
    """Counts of `move_column` (columns, in the order of `moves`) by the levels of `by`, for all rows and for
    those after a hard question. Without `by`: one row per move, a column per set of questions."""
    hard = rows[rows.hard_question]
    if by is None:
        return pd.DataFrame({"next_move": list(moves),
                             ALL_QUESTIONS: [int((rows[move_column] == m).sum()) for m in moves],
                             HARD_QUESTIONS: [int((hard[move_column] == m).sum()) for m in moves]})
    out = []
    for name, part in ((ALL_QUESTIONS, rows), (HARD_QUESTIONS, hard)):
        for level in levels:
            cell = part[part[by] == level]
            out.append({"questions": name, by: level, **{m: int((cell[move_column] == m).sum()) for m in moves},
                        "n": int(len(cell))})
    return pd.DataFrame(out, columns=["questions", by, *moves, "n"])


def h4_tables(followups: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The H4 read-out as plain count tables, from `answer_followups` of the whole encounter (no cut).

    One counted unit is an answer and an investor question it responds to. "hard" keeps the rows whose
    question has severity 2 or 3. Eight tables (H4_TABLES):

    - `asker`: the asker's next move (rows, NEXT_MOVES) for all and for hard questions;
    - `asker_by_substance`, `asker_by_support`: the same moves as columns, one row per set of questions
      and level of the answer's substance (1 to 5, not_rated) or checkable support (yes, no, not_rated),
      with `n` the rows of that level;
    - `asker_by_video`: the same per video, with only the videos that have rows;
    - `panel`, `panel_by_*`: likewise for the panel's next move (PANEL_MOVES; see `panel_next_move`).

    Descriptive counts, never predictors. Rows cut off by a history cut are refused: they do not know
    their next move, so the tables would not describe the encounter.
    """
    if (followups.next_move == AFTER_CUT).any():
        raise ValueError("h4_tables describes the whole encounter: call answer_followups without a cut")
    rows = followups.assign(panel_next_move=panel_next_move(followups),
                            substance=followups.subs.map(substance_level),
                            support=followups.evid.map(support_level))
    videos = tuple(sorted(set(rows.video_id)))
    tables = {}
    for who, moves, column in (("asker", NEXT_MOVES, "next_move"), ("panel", PANEL_MOVES, "panel_next_move")):
        tables[who] = _crossed(rows, None, None, moves, column)
        tables[f"{who}_by_substance"] = _crossed(rows, "substance", SUBSTANCE_LEVELS, moves, column)
        tables[f"{who}_by_support"] = _crossed(rows, "support", SUPPORT_LEVELS, moves, column)
        tables[f"{who}_by_video"] = _crossed(rows, "video_id", videos, moves, column)
    return {name: tables[name] for name in H4_TABLES}
