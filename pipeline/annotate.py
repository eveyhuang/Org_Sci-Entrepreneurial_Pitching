"""The core annotation operation.

run_pass enumerates work units from config + registry state, executes them with
bounded concurrency, and persists raw text before parsed records. Everything that
could change a result is in the unit_id, so re-running is always safe and editing a
prompt re-enqueues exactly the affected units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from pipeline.config import GENERATED_SCHEMA, PassConfig, PipelineConfig, import_schema
from pipeline.errors import SchemaInvalidError
from pipeline.execute import TokenBucket, UnitOutcome, run_units
from pipeline.media import clip
from pipeline.prompts import load_pass_codebook, render_pass_prompt
from pipeline.providers.base import GenerateRequest, Provider
from pipeline.providers.gemini import gemini_response_schema
from pipeline.registry import Registry, WorkUnit, make_unit_id
from pipeline.runs import Run, parsed_path, prompt_path, raw_path, schema_path, update_manifest
from pipeline.schemas.generate import build_record_model, schema_fingerprint
from pipeline.stage_b.context import build_context
from pipeline.stage_b.postprocess import run_postprocessor

_UNIT_KIND = {"EPISODE": "episode", "PREPARED": "prepared"}


@dataclass
class PassSummary:
    pass_name: str
    prompt_hash: str
    model: str
    enqueued: int = 0
    done: int = 0
    failed: int = 0
    skipped: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    failures: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "prompt_hash": self.prompt_hash, "model": self.model, "enqueued": self.enqueued,
            "done": self.done, "failed": self.failed, "skipped": self.skipped,
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "failures": self.failures,
        }


def _unit_key(pass_name: str, unit: WorkUnit) -> str:
    """Identify a unit to the provider: pass, video, span, and replicate.

    Span and replicate are part of the key so a fixture-driven provider can tell
    replicates and episodes apart — without them, a bug that gave every episode the
    first episode's clip would be invisible in tests.
    """
    span = unit.span
    stem = unit.video_id if span is None else f"{unit.video_id}__{span[0]:.3f}_{span[1]:.3f}"
    return f"{pass_name}__{stem}__k{unit.k}"


def resolve_schema(cfg: PipelineConfig, pass_cfg: PassConfig):
    """The pass's response model: imported, or generated from its codebook."""
    if pass_cfg.schema != GENERATED_SCHEMA:
        return import_schema(pass_cfg.schema)
    return build_record_model(
        load_pass_codebook(cfg, pass_cfg),
        name=f"{pass_cfg.name.title().replace('_', '')}Record", unit=pass_cfg.unit,
    )


def config_fingerprint(cfg: PipelineConfig, pass_cfg: PassConfig) -> str:
    """Settings that change the instruction sent to the model but are not in the prompt.

    Folded into the unit_id so that changing temperature, the output-token budget, or
    the response schema re-enqueues units instead of letting them be skipped as done
    while the manifest asserts the new values.
    """
    model = cfg.model_for(pass_cfg)
    schema_id = pass_cfg.schema
    if pass_cfg.schema == GENERATED_SCHEMA:
        # The prompt hash covers codebook wording; this covers the response contract.
        schema_id = f"codebook:{schema_fingerprint(resolve_schema(cfg, pass_cfg))}"
    return "|".join([
        f"t={model.temperature}",
        f"max_out={model.max_output_tokens}",
        f"schema={schema_id}",
    ])


def enumerate_units(
    cfg: PipelineConfig,
    pass_cfg: PassConfig,
    registry: Registry,
    run: Run,
    prompt_hash: str,
    video_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[WorkUnit]:
    """Build the full set of units this pass requires (regardless of completion)."""
    videos = registry.list_videos(
        corpus=cfg.corpus.name, probe_ok_only=True, include_skipped=False
    )
    if video_ids:
        wanted = set(video_ids)
        videos = [v for v in videos if v.video_id in wanted]
    if limit is not None:
        videos = videos[:limit]

    fingerprint = config_fingerprint(cfg, pass_cfg)
    model_name = cfg.model_for(pass_cfg).name
    units: list[WorkUnit] = []
    for video in videos:
        unit_fingerprint = fingerprint
        if pass_cfg.depends_on:
            # Not ready until the upstream pass has a result for this video. Re-running
            # annotate later picks the video up; nothing is recorded as failed meanwhile.
            upstream = registry.latest_done_unit(video.video_id, pass_cfg.depends_on)
            if upstream is None:
                continue
            # Frozen turn indices refer to one specific timeline: a new upstream result
            # must re-enqueue everything built on the old one.
            unit_fingerprint = f"{fingerprint}|up={upstream.unit_id}"
        if "episode_table" in pass_cfg.context and not registry.has_frozen_episodes(video.video_id):
            continue

        spans: list[tuple[tuple[float, float] | None, int | None]]
        if pass_cfg.unit in _UNIT_KIND:
            spans = [
                ((e.start_s, e.end_s), e.episode_idx)
                for e in registry.episodes_for(
                    video.video_id, frozen_only=True, kind=_UNIT_KIND[pass_cfg.unit]
                )
            ]
        else:
            spans = [(None, None)]
        for span, episode_idx in spans:
            for k in range(pass_cfg.k):
                unit_id = make_unit_id(
                    video_id=video.video_id, pass_name=pass_cfg.name, prompt_hash=prompt_hash,
                    model=model_name, media=pass_cfg.media, k=k, span=span,
                    config_fingerprint=unit_fingerprint,
                )
                units.append(
                    WorkUnit(
                        unit_id=unit_id, run_id=run.run_id, video_id=video.video_id,
                        pass_name=pass_cfg.name, prompt_hash=prompt_hash, model=model_name,
                        media=pass_cfg.media, k=k,
                        span_start_s=span[0] if span else None,
                        span_end_s=span[1] if span else None,
                        episode_idx=episode_idx,
                    )
                )
    return units


def run_pass(
    cfg: PipelineConfig,
    pass_cfg: PassConfig,
    registry: Registry,
    run: Run,
    provider: Provider,
    video_ids: list[str] | None = None,
    limit: int | None = None,
    retry_failed: bool = False,
    progress: bool = True,
) -> PassSummary:
    codebook = load_pass_codebook(cfg, pass_cfg)
    prompt_text, digest = render_pass_prompt(cfg, pass_cfg, codebook)
    schema = resolve_schema(cfg, pass_cfg)
    model = cfg.model_for(pass_cfg)
    # Archive the exact text sent, so the run is self-describing without a checkout.
    prompt_path(run, pass_cfg.name, digest).write_text(prompt_text)
    # Archive the exact wire schema too: the prompt alone does not show the response
    # contract, and the codebook that generated it may since have changed.
    schema_path(run, pass_cfg.name, digest).write_text(
        json.dumps(gemini_response_schema(schema), indent=2, sort_keys=True)
    )
    summary = PassSummary(pass_name=pass_cfg.name, prompt_hash=digest, model=model.name)

    required = enumerate_units(cfg, pass_cfg, registry, run, digest, video_ids, limit)
    todo: list[WorkUnit] = []
    for unit in required:
        registry.enqueue(unit)
        current = registry.get_unit(unit.unit_id)
        if current.status == "done":
            summary.skipped += 1
        elif current.status == "failed" and not retry_failed:
            summary.skipped += 1
        else:
            todo.append(unit)
    summary.enqueued = len(todo)

    if not todo:
        _record(run, summary, registry)
        return summary

    bucket = TokenBucket(cfg.limits.requests_per_minute)
    bar = tqdm(total=len(todo), desc=f"{pass_cfg.name}", disable=not progress)

    def work(unit: WorkUnit):
        video = registry.get_video(unit.video_id)
        if video is None:
            raise SchemaInvalidError(
                f"{unit.video_id}: no such video in the registry — re-run ingest"
            )
        if pass_cfg.produces == "episodes" and registry.has_frozen_episodes(unit.video_id):
            # Refuse before the call, not after it: frozen boundaries are what every
            # downstream code was assigned against, and a refusal should cost nothing.
            raise SchemaInvalidError(
                f"{unit.video_id}: episodes are frozen, so re-segmentation is refused. To redo "
                f"it: scripts/freeze_episodes.py --unfreeze {unit.video_id}"
            )
        media_handle = None
        if pass_cfg.media != "none":
            source = Path(video.path)
            digest_of_source: str | None = video.sha256
            if unit.span is not None:
                source = clip(source, unit.span[0], unit.span[1])
                digest_of_source = None  # the clip is a different file than the source
            media_handle = provider.upload_media(source, file_sha256=digest_of_source)

        unit_prompt = prompt_text
        if pass_cfg.context:
            unit_prompt, _ = render_pass_prompt(
                cfg, pass_cfg, codebook,
                context=build_context(cfg, pass_cfg, registry, unit),
            )

        request = GenerateRequest(
            prompt=unit_prompt, media=media_handle, schema=schema, model=model,
            unit_key=_unit_key(pass_cfg.name, unit),
        )
        response = provider.generate(request)
        if response.parsed is None:
            raise SchemaInvalidError(
                f"{unit.video_id}: provider returned no parsed record",
                raw_text=response.raw_text,
            )

        raw_file = raw_path(
            run, pass_cfg.name, unit.video_id, unit.k, unit.span, prompt_hash=digest
        )
        raw_file.write_text(response.raw_text)
        parsed_file = parsed_path(
            run, pass_cfg.name, unit.video_id, unit.k, unit.span, prompt_hash=digest
        )
        parsed_file.write_text(
            json.dumps(
                {
                    "unit_id": unit.unit_id, "run_id": run.run_id, "video_id": unit.video_id,
                    "pass": pass_cfg.name, "span": list(unit.span) if unit.span else None,
                    "episode_idx": unit.episode_idx,
                    "k": unit.k, "media": pass_cfg.media, "model_requested": model.name,
                    "model_version": response.model_version, "prompt_hash": digest,
                    "record": response.parsed.model_dump(mode="json", by_alias=True),
                },
                indent=2,
            )
        )
        if pass_cfg.produces:
            try:
                run_postprocessor(cfg, pass_cfg, run, registry, unit, response.parsed)
            except ValueError as exc:
                raise SchemaInvalidError(
                    f"{unit.video_id}: {exc}", raw_text=response.raw_text
                ) from exc
        registry.mark_done(
            unit.unit_id, str(raw_file), str(parsed_file),
            response.input_tokens, response.output_tokens, response.thinking_tokens,
        )
        return response

    def handle(outcome: UnitOutcome) -> None:
        bar.update(1)
        unit: WorkUnit = outcome.item
        if outcome.ok:
            summary.done += 1
            summary.input_tokens += outcome.value.input_tokens
            summary.output_tokens += outcome.value.output_tokens
            summary.thinking_tokens += outcome.value.thinking_tokens
        else:
            label = outcome.error_class or "UNKNOWN"
            summary.failed += 1
            summary.failures[label] = summary.failures.get(label, 0) + 1
            # A schema-invalid response still has a body, and it is the most
            # valuable body we get — the failure taxonomy is a published
            # deliverable, so persist it before recording the failure.
            raw_file = None
            raw_text = getattr(outcome.error, "raw_text", None)
            if raw_text:
                path = raw_path(
                    run, pass_cfg.name, unit.video_id, unit.k, unit.span, prompt_hash=digest
                )
                path.write_text(raw_text)
                raw_file = str(path)
            detail = (
                str(outcome.error)
                if outcome.error is not None
                else "not attempted: run stopped after a fatal error"
            )
            registry.mark_failed(unit.unit_id, label, detail, raw_path=raw_file)
            tqdm.write(f"[{label}] {unit.video_id} k{unit.k}: {detail}")

    try:
        run_units(
            todo, work, max_concurrency=cfg.limits.max_concurrency,
            bucket=bucket, limits=cfg.limits, on_result=handle,
        )
    finally:
        bar.close()
        _record(run, summary, registry)
    return summary


def run_all_passes(
    cfg: PipelineConfig,
    registry: Registry,
    run: Run,
    provider: Provider,
    pass_name: str | None = None,
    **kwargs,
) -> list[PassSummary]:
    selected = [cfg.get_pass(pass_name)] if pass_name else list(cfg.passes)
    return [run_pass(cfg, p, registry, run, provider, **kwargs) for p in selected]


def _record(run: Run, summary: PassSummary, registry: Registry) -> None:
    update_manifest(
        run,
        {
            "passes": {summary.pass_name: summary.as_dict()},
            "status_counts": registry.status_counts(run.run_id),
            "failure_counts": registry.failure_counts(run.run_id),
        },
    )
