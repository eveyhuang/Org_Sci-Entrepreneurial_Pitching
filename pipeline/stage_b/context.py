"""Per-unit context: upstream results rendered into a downstream pass's prompt.

The prompt hash excludes this text (see pipeline.prompts.render_pass_prompt), so context
never fragments unit identity. Later passes never re-identify speakers — they are handed
the roster and told to use its labels, which is what keeps `speaker-k` stable per video.
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline.config import PassConfig, PipelineConfig
from pipeline.errors import ConfigError, SchemaInvalidError
from pipeline.registry import Registry, WorkUnit
from pipeline.schemas.stage_b_timeline import normalised_turns
from pipeline.schemas.stage_b_timeline import roles as timeline_roles
from pipeline.schemas.stage_b_timeline import seconds_to_clock


def load_upstream_record(registry: Registry, video_id: str, pass_name: str) -> dict:
    unit = registry.latest_done_unit(video_id, pass_name)
    if unit is None or not unit.parsed_path or not Path(unit.parsed_path).exists():
        raise SchemaInvalidError(
            f"{video_id}: no finished '{pass_name}' result to build on — run it first "
            f"(scripts/annotate.py --pass {pass_name})"
        )
    return json.loads(Path(unit.parsed_path).read_text())["record"]


def format_turn(turn: dict, roles: dict[str, str], offset: float = 0.0) -> str:
    """Render a NORMALISED turn (`turn_idx, speaker, start_s, end_s, tag`)."""
    start = max(0.0, turn["start_s"] - offset)
    end = max(0.0, turn["end_s"] - offset)
    role = roles.get(turn["speaker"], "other")
    return (
        f"[t{turn['turn_idx']} {seconds_to_clock(start)}–{seconds_to_clock(end)}] "
        f"{turn['speaker']} ({role}): {turn['tag']}"
    )


def _producer(cfg: PipelineConfig, kind: str, needed_by: str) -> str:
    producer = cfg.pass_producing(kind)
    if producer is None:
        raise ConfigError(f"context {needed_by!r} needs an earlier pass with `produces: {kind}`")
    return producer.name


def build_context(
    cfg: PipelineConfig, pass_cfg: PassConfig, registry: Registry, unit: WorkUnit
) -> dict[str, str]:
    keys = pass_cfg.context
    context: dict[str, str] = {}

    timeline_record = None
    turns: list[dict] = []
    roles: dict[str, str] = {}
    if {"roster", "timeline", "timeline_slice"} & set(keys):
        timeline_record = load_upstream_record(
            registry, unit.video_id, _producer(cfg, "timeline", "roster/timeline")
        )
        turns = normalised_turns(timeline_record)
        roles = timeline_roles(timeline_record)

    episode = None
    if {"timeline_slice", "pitch_summary"} & set(keys):
        if unit.episode_idx is None:
            raise ConfigError(
                f"pass {pass_cfg.name!r}: episode context needs an EPISODE or PREPARED unit"
            )
        episode = registry.get_episode(unit.video_id, unit.episode_idx)
        if episode is None:
            raise SchemaInvalidError(f"{unit.video_id}: episode {unit.episode_idx} is not in the registry")

    if "roster" in keys:
        context["roster"] = "\n".join(
            f"{r['speaker']} — {r['role']} — {r['seat']} — {r['descriptor']}"
            for r in timeline_record["roster"]
        )
    if "timeline" in keys:
        context["timeline"] = "\n".join(format_turn(t, roles) for t in turns)
    if "timeline_slice" in keys:
        if episode.start_turn is None or episode.end_turn is None:
            raise SchemaInvalidError(
                f"{unit.video_id}: episode {unit.episode_idx} has no turn range "
                f"(start_turn/end_turn is null) — cannot build its timeline slice"
            )
        inside = [t for t in turns if episode.start_turn <= t["turn_idx"] <= episode.end_turn]
        context["timeline_slice"] = "\n".join(
            format_turn(t, roles, offset=episode.start_s) for t in inside
        )
    if "pitch_summary" in keys:
        segmentation = load_upstream_record(
            registry, unit.video_id, _producer(cfg, "episodes", "pitch_summary")
        )
        topic = f"\nThis episode's topic: {episode.topic}" if episode.topic else ""
        cut_line = ""
        if episode.cut is not None:
            cut_line = f"\nEditing cut inside this episode (EP-CUT): {'yes' if episode.cut else 'no'}"
        context["pitch_summary"] = segmentation["summary"] + topic + cut_line
    if "episode_table" in keys:
        lines = []
        for row in registry.episodes_for(unit.video_id, frozen_only=True, kind=None):
            if row.kind == "prepared":
                lines.append(f"prepared pitch: turns {row.start_turn}–{row.end_turn}")
            else:
                lines.append(
                    f"episode {row.episode_idx}: turns {row.start_turn}–{row.end_turn}, "
                    f"probed by {row.probing_speaker}, topic: {row.topic}"
                )
        context["episode_table"] = "\n".join(lines)
    return context
