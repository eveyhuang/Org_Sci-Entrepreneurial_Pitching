"""Parsed records → tidy tables. This is the instrument/analysis boundary.

Nothing downstream of these files imports pipeline code, and nothing here computes
study quantities: no PCA, no models, no pitch-level roll-ups beyond the
per-construct mean/sd across replicates that describes the measurement itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.config import PipelineConfig
from pipeline.registry import Registry
from pipeline.runs import Run, tables_dir
from pipeline.schemas.stage_b_timeline import clock_to_seconds, normalised_turns, roles


def load_parsed_records(
    registry: Registry, run: Run, pass_name: str | None = None
) -> list[dict]:
    records: list[dict] = []
    for unit in registry.done_units(run.run_id, pass_name=pass_name):
        if not unit.parsed_path:
            continue
        path = Path(unit.parsed_path)
        if not path.exists():
            continue
        records.append(json.loads(path.read_text()))
    return records


# Stage A ratings also carry `confidence`; Stage B's generated ratings carry `turn_refs`.
# The shared core is what makes a payload a rating.
RATING_KEYS = frozenset({"value", "uncodable", "evidence"})


def _numeric(value: object) -> float | None:
    """Ratings as numbers: booleans become 0/1; categorical answers go to value_label."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_rating(payload: object) -> bool:
    """A rating is a dict carrying the full Rating shape.

    Checking the whole shape rather than one key keeps free-text or structured
    non-rating fields out of the long table as the record schema grows.
    """
    return isinstance(payload, dict) and RATING_KEYS.issubset(payload.keys())


def _rating_rows(record: dict) -> list[dict]:
    """Flatten one parsed record into long rows, one per rating-shaped field."""
    span = record.get("span") or [None, None]
    rows: list[dict] = []
    for code_id, payload in record["record"].items():
        if not _is_rating(payload):
            continue  # free-text fields such as overall_notes
        rows.append(
            {
                "run_id": record["run_id"],
                "video_id": record["video_id"],
                "episode_idx": record.get("episode_idx"),
                "span_start_s": span[0],
                "span_end_s": span[1],
                "code_id": code_id,
                "replicate_k": record["k"],
                "value": _numeric(payload.get("value")),
                "value_label": payload["value"] if isinstance(payload.get("value"), str) else None,
                "turn_refs": json.dumps(payload["turn_refs"]) if "turn_refs" in payload else None,
                "uncodable": bool(payload.get("uncodable", False)),
                "confidence": payload.get("confidence"),
                "evidence": payload.get("evidence"),
                "pass": record["pass"],
                "media": record.get("media"),
                "model_requested": record.get("model_requested"),
                "model_version": record.get("model_version"),
                "prompt_hash": record.get("prompt_hash"),
                "unit_id": record.get("unit_id"),
            }
        )
    return rows


def _refs_valid(row: dict, bounds: dict[tuple[str, int], tuple[int, int]]) -> bool | None:
    if row["turn_refs"] is None or row["episode_idx"] is None:
        return None
    span = bounds.get((row["video_id"], int(row["episode_idx"])))
    if span is None:
        return None
    return all(span[0] <= ref <= span[1] for ref in json.loads(row["turn_refs"]))


def build_tables(cfg: PipelineConfig, registry: Registry, run: Run) -> dict[str, Path]:
    """Write tidy tables for this run. Safe to re-run; files are overwritten."""
    out_dir = tables_dir(run)
    records = load_parsed_records(registry, run)
    videos = registry.list_videos(corpus=cfg.corpus.name)

    episodes = [e for v in videos for e in registry.episodes_for(v.video_id, frozen_only=False, kind=None)]
    bounds = {(e.video_id, e.episode_idx): (e.start_turn, e.end_turn)
              for e in episodes if e.start_turn is not None and e.end_turn is not None}

    rows: list[dict] = []
    for record in records:
        rows.extend(_rating_rows(record))
    for row in rows:
        row["turn_refs_valid"] = _refs_valid(row, bounds)
    annotations = pd.DataFrame(rows)
    tables = {"annotations": out_dir / "annotations.parquet"}
    annotations.to_parquet(tables["annotations"], index=False)

    def first_replicate(kind: str) -> list[dict]:
        producer = cfg.pass_producing(kind)
        return [r for r in records if producer and r["pass"] == producer.name and r["k"] == 0]

    turn_rows = []
    for record in first_replicate("timeline"):
        turns = normalised_turns(record["record"])
        role_map = roles(record["record"])
        turn_rows += [
            {"video_id": record["video_id"], "turn_idx": t["turn_idx"], "speaker": t["speaker"],
             "role": role_map.get(t["speaker"]), "start_s": t["start_s"], "end_s": t["end_s"],
             "tag": t["tag"]}
            for t in turns
        ]
    episode_rows = [
        {"video_id": e.video_id, "episode_idx": e.episode_idx, "kind": e.kind,
         "start_s": e.start_s, "end_s": e.end_s, "start_turn": e.start_turn,
         "end_turn": e.end_turn, "probing_speaker": e.probing_speaker, "topic": e.topic,
         "resolution": e.resolution, "cut": e.cut, "frozen": e.frozen}
        for e in episodes
    ]
    # Revision R2: investor decisions (`out`/`offer`/`joins_offer`) and the on-air `deal`
    # are the study's OUTCOMES, one row per shark per video — never predictor episodes.
    decision_rows = [
        {"video_id": d.video_id, "speaker": d.speaker, "decision": d.decision,
         "turn": d.turn, "terms": d.terms}
        for v in videos for d in registry.decisions_for(v.video_id)
    ]
    for name, table_rows in (
        ("turns", turn_rows), ("episodes", episode_rows), ("decisions", decision_rows),
    ):
        if table_rows:
            tables[name] = out_dir / f"{name}.parquet"
            pd.DataFrame(table_rows).to_parquet(tables[name], index=False)

    video_table = _build_videos_table(cfg, registry, annotations)
    video_table["skip_reason"] = video_table["video_id"].map({v.video_id: v.skip_reason for v in videos})
    video_table["deal"] = video_table["video_id"].map({v.video_id: v.deal for v in videos})
    n_offers = pd.Series(
        [d["video_id"] for d in decision_rows if d["decision"] in ("offer", "joins_offer")]
    ).value_counts()
    n_out = pd.Series([d["video_id"] for d in decision_rows if d["decision"] == "out"]).value_counts()
    video_table["n_offers"] = video_table["video_id"].map(n_offers).fillna(0).astype("Int64")
    video_table["n_out"] = video_table["video_id"].map(n_out).fillna(0).astype("Int64")
    if cfg.corpus.dev_set is not None:
        from pipeline.config import read_video_list

        video_table["dev_set"] = video_table["video_id"].isin(read_video_list(cfg.corpus.dev_set))
    segmentation = {r["video_id"]: r["record"] for r in first_replicate("episodes")}
    if segmentation:
        video_table["n_pitches"] = video_table["video_id"].map(
            {k: v["n_pitches"] for k, v in segmentation.items()}).astype("Int64")
        video_table["prepared_end_s"] = video_table["video_id"].map(
            {k: clock_to_seconds(v["prepared_end"]) for k, v in segmentation.items()})
        counts = pd.Series([e.video_id for e in episodes if e.kind == "episode"]).value_counts()
        video_table["n_episodes"] = video_table["video_id"].map(counts).fillna(0).astype("Int64")
    tables["videos"] = out_dir / "videos.parquet"
    video_table.to_parquet(tables["videos"], index=False)
    return tables


def _build_videos_table(
    cfg: PipelineConfig, registry: Registry, annotations: pd.DataFrame
) -> pd.DataFrame:
    meta = pd.DataFrame(
        [
            {
                "video_id": v.video_id, "corpus": v.corpus, "duration_s": v.duration_s,
                "width": v.width, "height": v.height, "has_audio": v.has_audio,
                **{
                    f"meta_{k}": (val if isinstance(val, (str, int, float, bool, type(None)))
                                  else json.dumps(val))
                    for k, val in json.loads(v.meta_json).items()
                },
            }
            for v in registry.list_videos(corpus=cfg.corpus.name)
        ]
    )
    if annotations.empty:
        return meta

    whole_video = annotations[annotations["episode_idx"].isna()]
    if whole_video.empty:
        return meta

    grouped = whole_video.groupby(["video_id", "code_id"])["value"]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    wide = stats.pivot(index="video_id", columns="code_id", values=["mean", "std", "count"])
    wide.columns = [f"{code}_{'sd' if stat == 'std' else 'n' if stat == 'count' else stat}"
                    for stat, code in wide.columns]
    wide = wide.reset_index()
    count_cols = [c for c in wide.columns if c.endswith("_n")]
    wide[count_cols] = wide[count_cols].fillna(0).astype("Int64")
    return meta.merge(wide, on="video_id", how="left")
