"""Freeze workflow: proposed episode boundaries -> human review -> frozen.

Downstream passes enumerate frozen rows only, so nothing can be coded against
boundaries nobody looked at. A reviewer may edit, delete or add rows in the exported CSV.
On import, `start_s`/`end_s` are ALWAYS recomputed from `start_turn`/`end_turn` against the
video's timeline turns (same rule as `pipeline.stage_b.postprocess._write_episodes`): the
CSV's own `start_s`/`end_s` columns are informational only and are ignored. This keeps a
reviewer-edited turn range from desynchronising the clip from its timeline slice.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import re
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.registry import PREPARED_IDX, EpisodeRow, Registry
from pipeline.schemas.stage_b_timeline import normalised_turns
from pipeline.stage_b.context import load_upstream_record

COLUMNS = ["video_id", "title", "episode_idx", "kind", "start_s", "end_s", "start_turn",
           "end_turn", "probing_speaker", "topic", "resolution", "cut", "frozen"]

_SPEAKER_RE = re.compile(r"^speaker-\d+$")


def export_review(registry: Registry, corpus: str, out_csv: Path | str) -> int:
    count = 0
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for video in registry.list_videos(corpus=corpus):
            title = json.loads(video.meta_json or "{}").get("title", "")
            for row in registry.episodes_for(video.video_id, frozen_only=False, kind=None):
                writer.writerow({
                    "video_id": row.video_id, "title": title, "episode_idx": row.episode_idx,
                    "kind": row.kind, "start_s": row.start_s, "end_s": row.end_s,
                    "start_turn": row.start_turn, "end_turn": row.end_turn,
                    "probing_speaker": row.probing_speaker or "", "topic": row.topic or "",
                    "resolution": row.resolution or "",
                    "cut": "" if row.cut is None else str(row.cut), "frozen": str(row.frozen),
                })
                count += 1
    return count


def _parse_row(line: int, raw: dict[str, str]) -> EpisodeRow:
    try:
        start_turn_text = raw["start_turn"].strip()
        end_turn_text = raw["end_turn"].strip()
        if not start_turn_text or not end_turn_text:
            raise ValueError(
                "start_turn and end_turn are required (start_s/end_s are recomputed from them)"
            )
        row = EpisodeRow(
            video_id=raw["video_id"].strip(), episode_idx=int(raw["episode_idx"]),
            # start_s/end_s are parsed only to build the row; import_review overwrites them
            # from the timeline turns before anything is written to the registry.
            start_s=float(raw["start_s"]), end_s=float(raw["end_s"]),
            kind=raw["kind"].strip(), probing_speaker=raw["probing_speaker"].strip() or None,
            start_turn=int(start_turn_text), end_turn=int(end_turn_text),
            cut=None if not raw["cut"].strip() else raw["cut"].strip().lower() == "true",
            topic=raw["topic"].strip() or None, resolution=raw["resolution"].strip() or None,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"line {line}: {exc}") from exc
    if row.kind not in ("episode", "prepared"):
        raise ValueError(f"line {line}: kind must be 'episode' or 'prepared', got {row.kind!r}")
    if (row.kind == "prepared") != (row.episode_idx == PREPARED_IDX):
        raise ValueError(f"line {line}: the prepared row, and only it, has episode_idx {PREPARED_IDX}")
    if row.start_turn > row.end_turn:
        raise ValueError(f"line {line}: start_turn must be <= end_turn")
    if row.probing_speaker is not None and not _SPEAKER_RE.match(row.probing_speaker):
        raise ValueError(
            f"line {line}: probing_speaker must look like 'speaker-3', got {row.probing_speaker!r}"
        )
    return row


def import_review(registry: Registry, csv_path: Path | str, cfg: PipelineConfig) -> tuple[int, int]:
    """Replace unfrozen rows with the CSV's. The whole file is validated before anything is applied.

    `start_s`/`end_s` are re-derived from each row's `start_turn`/`end_turn` against that
    video's timeline (the pass whose config has `produces: timeline`); the CSV's own
    start_s/end_s columns are ignored.
    """
    producer = cfg.pass_producing("timeline")
    if producer is None:
        raise ValueError("this config has no pass with `produces: timeline`")

    by_video: dict[str, list[tuple[int, EpisodeRow]]] = {}
    with open(csv_path, newline="") as handle:
        for line, raw in enumerate(csv.DictReader(handle), start=2):
            row = _parse_row(line, raw)
            by_video.setdefault(row.video_id, []).append((line, row))

    for video_id, entries in by_video.items():
        seen: dict[int, int] = {}
        for line, row in entries:
            if row.episode_idx in seen:
                raise ValueError(
                    f"{video_id}: duplicate episode_idx {row.episode_idx} at lines "
                    f"{seen[row.episode_idx]} and {line} in {csv_path}"
                )
            seen[row.episode_idx] = line

    # Compute every video's re-derived rows before writing any of them, so a bad row for
    # one video (e.g. a turn past its timeline) cannot leave another video half-applied.
    to_apply: dict[str, list[EpisodeRow]] = {}
    skipped = 0
    for video_id, entries in by_video.items():
        if registry.has_frozen_episodes(video_id):
            skipped += 1
            continue
        turns = normalised_turns(load_upstream_record(registry, video_id, producer.name))
        rows: list[EpisodeRow] = []
        for line, row in entries:
            if row.end_turn >= len(turns):
                raise ValueError(
                    f"line {line}: end_turn {row.end_turn} is past the timeline "
                    f"({len(turns)} turns)"
                )
            inside = turns[row.start_turn: row.end_turn + 1]
            start_s = 0.0 if row.kind == "prepared" else turns[row.start_turn]["start_s"]
            end_s = max(t["end_s"] for t in inside)
            rows.append(dataclasses.replace(row, start_s=start_s, end_s=end_s))
        to_apply[video_id] = sorted(rows, key=lambda r: r.episode_idx)

    for video_id, rows in to_apply.items():
        registry.replace_unfrozen_episodes(video_id, rows)
    return len(to_apply), skipped


def freeze(registry: Registry, corpus: str, video_ids: list[str] | None = None) -> int:
    wanted = set(video_ids) if video_ids else None
    frozen = 0
    for video in registry.list_videos(corpus=corpus):
        if wanted is not None and video.video_id not in wanted:
            continue
        rows = registry.episodes_for(video.video_id, frozen_only=False, kind=None)
        if rows and not all(r.frozen for r in rows):
            registry.set_frozen(video.video_id, True)
            frozen += 1
    return frozen


def unfreeze(registry: Registry, video_ids: list[str]) -> int:
    return sum(1 for video_id in video_ids if registry.set_frozen(video_id, False) > 0)
