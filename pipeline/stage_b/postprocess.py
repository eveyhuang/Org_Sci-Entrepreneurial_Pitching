"""Post-processors: what a pass `produces:` beyond its parsed record.

Run inside the unit's work function after the record validates and is on disk, before
the unit is marked done — so a unit is only ever 'done' once its side effects exist.
Only replicate k=0 produces artifacts; further replicates exist to measure stability.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from pipeline.config import PassConfig, PipelineConfig
from pipeline.registry import DecisionRow, EpisodeRow, PREPARED_IDX, Registry, WorkUnit
from pipeline.runs import Run, timeline_path
from pipeline.schemas.stage_b_timeline import clock_to_seconds, normalised_turns
from pipeline.stage_b.context import load_upstream_record


def _write_timeline(cfg, pass_cfg, run, registry, unit, record) -> None:
    # A readable copy for people. Downstream passes read the parsed record through the
    # registry instead (see pipeline/stage_b/context.py), which survives new run dirs.
    timeline_path(run, unit.video_id).write_text(
        json.dumps(record.model_dump(mode="json"), indent=2, ensure_ascii=False)
    )


def _write_episodes(cfg, pass_cfg, run, registry, unit, record) -> None:
    """Turn a segmentation record into proposed (unfrozen) rows.

    Clip bounds come from the speaking-timeline turns, not from the model's own clock
    strings, so an episode's clip and its timeline slice always cover the same material.
    The model's own times remain in the parsed record for audit.

    Revision R2: investor decisions (`out`/`offer`/`joins_offer`) and the on-air `deal`
    are the study's OUTCOMES, not predictor episodes. They are written to their own table
    here, alongside the episodes, and are NOT subject to freeze (decisions are not
    boundaries — Registry.replace_decisions has no freeze guard).
    """
    producer = cfg.pass_producing("timeline")
    if producer is None:
        raise ValueError("`produces: episodes` needs an earlier pass with `produces: timeline`")

    if record.n_pitches == 0:
        # No venture pitch in this clip: clear any earlier proposal and skip it downstream
        # instead of writing rows nobody should code.
        registry.replace_unfrozen_episodes(unit.video_id, [])
        registry.replace_decisions(unit.video_id, [])
        registry.set_deal(unit.video_id, None)
        registry.set_skip_reason(unit.video_id, "no_pitch")
        return

    video = registry.get_video(unit.video_id)
    if video is not None and video.skip_reason == "no_pitch":
        registry.set_skip_reason(unit.video_id, None)

    turns = normalised_turns(load_upstream_record(registry, unit.video_id, producer.name))

    rows: list[EpisodeRow] = []
    prepared_end_s = clock_to_seconds(record.prepared_end)
    first_probe = record.episodes[0].start_turn if record.episodes else len(turns)
    prepared = [t for t in turns
                if t["start_s"] < prepared_end_s and t["turn_idx"] < first_probe]
    if prepared:
        rows.append(EpisodeRow(
            unit.video_id, PREPARED_IDX, 0.0, max(t["end_s"] for t in prepared),
            source_unit_id=unit.unit_id, kind="prepared",
            start_turn=prepared[0]["turn_idx"], end_turn=prepared[-1]["turn_idx"],
        ))
    for idx, episode in enumerate(record.episodes):
        if episode.end_turn >= len(turns):
            raise ValueError(f"episode {idx} ends at turn {episode.end_turn}, but the timeline "
                             f"has only {len(turns)} turns")
        inside = turns[episode.start_turn: episode.end_turn + 1]
        start_s, end_s = inside[0]["start_s"], max(t["end_s"] for t in inside)
        if end_s <= start_s:
            raise ValueError(f"episode {idx} has no duration (turns {episode.start_turn}–{episode.end_turn})")
        rows.append(EpisodeRow(
            unit.video_id, idx, start_s, end_s, source_unit_id=unit.unit_id, kind="episode",
            probing_speaker=episode.probing_speaker, start_turn=episode.start_turn,
            end_turn=episode.end_turn, cut=episode.cut, topic=episode.topic,
            resolution=episode.resolution,
        ))
    registry.replace_unfrozen_episodes(unit.video_id, rows)  # ValueError if frozen

    decision_rows = [
        DecisionRow(unit.video_id, d.speaker, d.decision, turn=d.turn, terms=d.terms,
                    source_unit_id=unit.unit_id)
        for d in record.decisions
    ]
    registry.replace_decisions(unit.video_id, decision_rows)
    registry.set_deal(unit.video_id, record.deal)


_HANDLERS = {"timeline": _write_timeline, "episodes": _write_episodes}


def run_postprocessor(
    cfg: PipelineConfig, pass_cfg: PassConfig, run: Run, registry: Registry,
    unit: WorkUnit, record: BaseModel,
) -> None:
    """Apply the pass's `produces:` handler. Raises ValueError if the record cannot be applied."""
    if unit.k != 0:
        return
    _HANDLERS[pass_cfg.produces](cfg, pass_cfg, run, registry, unit, record)
