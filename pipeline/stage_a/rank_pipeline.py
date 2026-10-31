"""Cross-video ranking of prepared presentations, run after a panel run.

Each video's prepared window (from the panel run's primary pass, or the whole file for a corpus
of standalone pitch videos) is cut to a clip; clips are shown to the model in small seeded batches; the model returns one strict order of the batch
per Stage A dimension; orders are aggregated into one relative score per video per dimension.

The model sees prepared-presentation clips only: no questioning, reactions or decisions.
A batch call is retried only when it fails (transport error, unparseable output, or an order
that is not a strict order of the batch's clips); a valid result is final. Everything is
written under `<run_root>/ranking/`; nothing else in the panel run is touched.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Callable, Sequence

from pipeline.errors import ConfigError, RunFatalError, SchemaInvalidError
from pipeline.execute import with_retries
from pipeline.ingest import ffprobe, is_video_file, sha256_file
from pipeline.instrument import InstrumentRelease, check_release
from pipeline.media import clip
from pipeline.prompts import prompt_hash, render_prompt
from pipeline.providers.base import GenerateRequest
from pipeline.schemas.stage_a_rank import RankResponse, check_rank_response
from pipeline.stage_a.rank_aggregate import aggregate
from pipeline.stage_a.rank_design import design_summary, make_design
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.panel_pipeline import PanelConfig, _calls_logged, _dump, _model_dict, _read

RANK_TEMPLATE = "stage_a_rank.j2"
RANKING_DIR = "ranking"
ANNOTATION_CONTEXT = "prepared_clips_only"
SCALE = "relative_within_run"
# what a stored design must share with the current one for a run to be resumed
RESUME_KEYS = ("design", "windows", "prompt_template_sha256", "stage_a_codebook_sha256", "model")


def unit_key(batch_id: str) -> str:
    return f"rank__{batch_id}"


def _window(run_root: Path, video_id: str) -> tuple[dict | None, str | None]:
    """The prepared window of a video, or the reason it has none."""
    vdir = Path(run_root) / video_id
    primary = vdir / "results" / "primary.json"
    if not primary.exists():
        return None, "no primary pass"
    spans = (_read(primary).get("record") or {}).get("prepared_spans") or []
    if not spans:
        return None, "no prepared spans in the primary pass"
    experiment = vdir / "experiment.json"
    source = _read(experiment).get("source") if experiment.exists() else None
    if not source:
        return None, "no source recorded in experiment.json"
    start_s = min(float(s["start_s"]) for s in spans)
    end_s = max(float(s["end_s"]) for s in spans)
    if end_s <= start_s:
        return None, f"prepared window is empty ({start_s}-{end_s})"
    return {"start_s": start_s, "end_s": end_s, "n_spans": len(spans), "source": str(source)}, None


def prepared_window(run_root: Path, video_id: str) -> dict | None:
    """From the first prepared span's start to the last one's end, as the primary pass recorded them.

    With more than one span the window includes whatever interrupted the presentation between
    them; `n_spans` records that instead of hiding it.
    """
    return _window(run_root, video_id)[0]


def prepared_window_reason(run_root: Path, video_id: str) -> tuple[dict | None, str | None]:
    """`prepared_window` with, when there is none, the reason (no primary pass, no prepared spans, no source...)."""
    return _window(run_root, video_id)


def full_video_windows(video_dir: Path, video_ids: Sequence[str],
                       probe: Callable[[Path], object] = ffprobe) -> tuple[dict[str, dict], dict[str, str]]:
    """Whole videos as prepared windows, for a corpus of standalone pitches with no questioning.

    The video `v` is the one complete video file `<video_dir>/v.<ext>`; its window runs from zero
    to the duration ffprobe reports. Returns (windows, skipped-with-reason).
    """
    video_dir = Path(video_dir)
    windows: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for video_id in sorted(set(video_ids)):
        files = sorted(p for p in video_dir.glob(f"{video_id}.*") if is_video_file(p))
        if not files:
            skipped[video_id] = f"no video file {video_id}.* in {video_dir}"
            continue
        if len(files) > 1:
            skipped[video_id] = "more than one video file: " + ", ".join(p.name for p in files)
            continue
        try:
            duration = getattr(probe(files[0]), "duration_s", None)
        except Exception as exc:  # noqa: BLE001 - the reason is recorded, the other videos go on
            skipped[video_id] = f"ffprobe failed: {exc}"
            continue
        if not duration or float(duration) <= 0:
            skipped[video_id] = f"no duration from ffprobe for {files[0].name}"
            continue
        windows[video_id] = {"start_s": 0.0, "end_s": float(duration), "n_spans": 1, "source": str(files[0])}
    return windows, skipped


def whole_source(src: Path, start_s: float, end_s: float, cache_dir: Path | str) -> Path:
    """The cut for a whole-video window: the source file itself, not re-encoded and not copied."""
    return Path(src)


def plan_ranking_from_windows(windows: dict[str, dict], skipped: dict[str, str], batch_size: int,
                              appearances: int, seed: int, windows_from: str) -> dict:
    """The seeded batch design over the videos that have a window; `windows_from` records where
    the windows came from (`panel_primary_pass` or `whole_videos`)."""
    if len(windows) < 2:
        raise ConfigError(f"ranking needs at least 2 videos with a prepared window, got {len(windows)}"
                          + (f" (skipped: {skipped})" if skipped else ""))
    try:
        batches = make_design(list(windows), batch_size=batch_size, appearances=appearances, seed=seed)
    except ValueError as exc:
        raise ConfigError(f"invalid ranking design: {exc}") from exc
    return {
        "design": [{"batch_id": b.batch_id, "video_ids": list(b.video_ids), "labels": list(b.labels)}
                   for b in batches],
        "summary": design_summary(batches),
        "windows": dict(windows),
        "skipped": dict(skipped),
        "settings": {"batch_size": batch_size, "appearances": appearances, "seed": seed},
        "windows_from": windows_from,
    }


def plan_ranking(run_root: Path, video_ids: list[str], batch_size: int, appearances: int, seed: int) -> dict:
    """The design over a panel run's videos, each with the prepared window its primary pass recorded."""
    windows: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    for video_id in sorted(set(video_ids)):
        window, reason = _window(run_root, video_id)
        if window is None:
            skipped[video_id] = reason or "no prepared window"
        else:
            windows[video_id] = window
    return plan_ranking_from_windows(windows, skipped, batch_size, appearances, seed,
                                     windows_from="panel_primary_pass")


def _plain(obj):
    """What `obj` reads back as from JSON, so a stored design compares equal to a fresh one."""
    return json.loads(json.dumps(obj))


def _write_scores(rdir: Path, plan: dict, codes: tuple[str, ...]) -> tuple[int, int]:
    """scores.json and scores.csv from every batch result on disk; returns (ok, failed)."""
    orders: dict[str, list[list[str]]] = {code: [] for code in codes}
    appearances_ok = {video_id: 0 for video_id in sorted(plan["windows"])}
    ok = 0
    for batch in plan["design"]:
        path = rdir / "batches" / batch["batch_id"] / "result.json"
        if not path.exists():
            continue
        ok += 1
        result = _read(path)
        for code in codes:
            orders[code].append(result["rankings"][code])
        for video_id in result["video_ids"]:
            appearances_ok[video_id] += 1
    failed = len(plan["design"]) - ok
    scores = aggregate(orders)
    _dump(rdir / "scores.json", {
        **scores,
        "batches_ok": ok, "batches_failed": failed, "appearances_ok": appearances_ok,
        "annotation_context": ANNOTATION_CONTEXT, "scale": SCALE, "human_validated": False,
    })
    with (rdir / "scores.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video_id", "code", "pl_score", "mean_rank", "n_orders"])
        for code in codes:
            block = scores[code]
            for video_id in sorted(block["pl_score"]):
                writer.writerow([video_id, code, repr(block["pl_score"][video_id]),
                                 repr(block["mean_rank"][video_id]), block["n_orders"]])
    return ok, failed


def run_ranking(provider, release: InstrumentRelease, cfg: PanelConfig, run_root: Path, plan: dict,
                sleep: Callable[[float], None] = time.sleep,
                before_call: Callable[[], None] | None = None, cut=clip) -> dict:
    problems = check_release(release)
    if problems:
        raise ConfigError("instrument release is not clean: " + "; ".join(problems))

    book = load_panel_books(release).stage_a
    codes = book.ids
    rdir = Path(run_root) / RANKING_DIR
    current = {
        **plan,
        "release_id": release.release_id,
        "prompt_template_sha256": sha256_file(release.templates_dir / RANK_TEMPLATE),
        "stage_a_codebook_sha256": sha256_file(release.books["stage_a"].yaml),
        "model": _model_dict(cfg.model),
        "annotation_context": ANNOTATION_CONTEXT,
        "scale": SCALE,
        "human_validated": False,
    }
    design_path = rdir / "design.json"
    if design_path.exists():
        old = _read(design_path)
        for key in RESUME_KEYS:
            if old.get(key) != _plain(current[key]):
                raise ConfigError(
                    f"{design_path}: {key} differs from the current one; this ranking was started "
                    "with a different design or instrument; rank in a new run, or delete the "
                    "ranking folder deliberately"
                )
    else:
        _dump(design_path, current)

    pending = [b for b in plan["design"] if not (rdir / "batches" / b["batch_id"] / "result.json").exists()]

    # one clip per video, however many batches it appears in; a clip that cannot be cut fails
    # the batches it belongs to and no others
    clips: dict[str, tuple[Path, str]] = {}
    clip_errors: dict[str, str] = {}
    for video_id in sorted({v for b in pending for v in b["video_ids"]}):
        window = plan["windows"][video_id]
        try:
            path = Path(cut(Path(window["source"]), window["start_s"], window["end_s"], rdir / "clips"))
            clips[video_id] = (path, sha256_file(path))
        except RunFatalError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded against the batches that need it
            clip_errors[video_id] = f"clip of {video_id} could not be cut: {type(exc).__name__}: {exc}"

    calls_log = rdir / "calls.jsonl"
    retry = {"max_attempts": cfg.limits.max_attempts, "base_s": cfg.limits.backoff_base_s,
             "cap_s": cfg.limits.backoff_cap_s, "sleep": sleep}
    made = 0

    for batch in pending:
        batch_id = batch["batch_id"]
        key = unit_key(batch_id)
        labels = tuple(batch["labels"])
        video_of = dict(zip(batch["labels"], batch["video_ids"]))
        bdir = rdir / "batches" / batch_id
        bdir.mkdir(parents=True, exist_ok=True)

        def fail(message: str) -> None:
            _dump(bdir / "failed.json", {"batch_id": batch_id, "unit_key": key, "message": message})

        uncut = [clip_errors[v] for v in batch["video_ids"] if v in clip_errors]
        if uncut:
            fail("; ".join(uncut))
            continue
        try:
            parts = tuple(
                (label, with_retries(
                    lambda v=video_of[label]: provider.upload_media(clips[v][0], file_sha256=clips[v][1]),
                    **retry))
                for label in labels)
        except RunFatalError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded, the other batches still run
            fail(f"upload failed: {type(exc).__name__}: {exc}")
            continue

        prompt = render_prompt(RANK_TEMPLATE, codebook=book, context={"labels": list(labels)},
                               prompt_dir=release.templates_dir)
        phash = prompt_hash(prompt)
        attempt = 0
        if calls_log.exists():
            attempt = sum(1 for ln in calls_log.read_text().splitlines()
                          if ln.strip() and json.loads(ln).get("unit_key") == key)
        invalid = 0
        failure: str | None = None
        response = None

        def log(ok: bool, error_class: str | None, resp, seconds: float, exc: BaseException | None = None) -> None:
            if resp is not None:
                usage = {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                         "thinking_tokens": resp.thinking_tokens}
                version = resp.model_version
            else:
                # a response that arrived but was rejected still cost tokens; the provider attaches them
                usage = getattr(exc, "usage", None) or {}
                version = getattr(exc, "model_version", None)
            row = {
                "unit_key": key, "pass": "rank", "attempt": attempt, "ok": ok, "error_class": error_class,
                "model_requested": cfg.model.name,
                "model_version": version,
                "prompt_hash": phash,
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "thinking_tokens": int(usage.get("thinking_tokens") or 0),
                "seconds": round(seconds, 3),
            }
            with calls_log.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        def once():
            nonlocal attempt, made
            attempt += 1
            made += 1
            if before_call is not None:
                before_call()
            t0 = time.monotonic()
            try:
                resp = provider.generate(GenerateRequest(
                    prompt=prompt, media=None, schema=RankResponse, model=cfg.model, unit_key=key,
                    media_parts=parts))
            except BaseException as exc:
                log(False, type(exc).__name__, None, time.monotonic() - t0, exc)
                raise
            found = check_rank_response(resp.parsed, codes, labels)
            if found:
                # a tie, an omission or an invented clip is an invalid output like an unparseable body
                invalid_exc = SchemaInvalidError(f"{key}: " + "; ".join(found), raw_text=resp.raw_text)
                invalid_exc.problems = found
                invalid_exc.usage = {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                                     "thinking_tokens": resp.thinking_tokens}
                invalid_exc.model_version = resp.model_version
                log(False, type(invalid_exc).__name__, None, time.monotonic() - t0, invalid_exc)
                raise invalid_exc
            log(True, None, resp, time.monotonic() - t0)
            return resp

        while True:
            try:
                response = with_retries(once, **retry)
                break
            except RunFatalError:
                raise
            except SchemaInvalidError as exc:
                invalid += 1
                (bdir / f"attempt{attempt}_invalid.txt").write_text(exc.raw_text or "")
                _dump(bdir / f"attempt{attempt}_error.json",
                      {"message": str(exc), "problems": getattr(exc, "problems", [])})
                if invalid >= 1 + cfg.invalid_output_retries:
                    failure = str(exc)
                    break
            except Exception as exc:  # noqa: BLE001 - recorded, the other batches still run
                failure = f"{type(exc).__name__}: {exc}"
                body = getattr(exc, "raw_text", None)
                if body is not None:
                    (bdir / f"attempt{attempt}_failed.txt").write_text(body)
                break

        if failure is not None or response is None:
            fail(failure or "no parsed response")
            continue

        by_code = {r.code: r for r in response.parsed.rankings}
        (bdir / "prompt.txt").write_text(prompt)
        (bdir / "raw.json").write_text(response.raw_text)
        _dump(bdir / "result.json", {
            "batch_id": batch_id,
            "video_ids": list(batch["video_ids"]),
            "labels": list(labels),
            "rankings": {code: [video_of[label] for label in by_code[code].order] for code in codes},
            "notes": {code: by_code[code].note for code in codes},
            "batch_notes": response.parsed.notes,
            "model_version": response.model_version,
            "prompt_hash": phash,
        })
        (bdir / "failed.json").unlink(missing_ok=True)

    ok, failed = _write_scores(rdir, plan, codes)
    completion = {
        "status": "complete" if failed == 0 else ("partial" if ok else "failed"),
        "batches_ok": ok,
        "batches_failed": failed,
        "batches_total": len(plan["design"]),
        "generation_calls": made,
        "generation_calls_total": _calls_logged(calls_log),
        "scale": SCALE,
        "human_validated": False,
    }
    _dump(rdir / "completion.json", completion)
    return completion
