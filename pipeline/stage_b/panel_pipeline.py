"""Two identical full-video passes per video (primary, replicate), validated and aligned.

Both passes send the same rendered prompt, schema, media and model settings. A pass is
retried only when its call failed (transport error, unparseable or truncated response, or a
response whose clock times cannot be read); a pass that parsed is final, whatever validation
or the other pass says. A value given together with a missing reason is not a failed call: the
value is dropped and noted. The model returns PanelWire (clock times); results store PanelRecord
(seconds) and, beside it, two things computed locally from it: the prepared window in clock
form and the segment each turn falls in.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import yaml

from pipeline.config import LimitsConfig, ModelConfig
from pipeline.errors import ConfigError, RunFatalError, SchemaInvalidError
from pipeline.execute import with_retries
from pipeline.ingest import sha256_file
from pipeline.instrument import InstrumentRelease, canonical_files, check_release
from pipeline.prompts import prompt_hash
from pipeline.providers.base import GenerateRequest
from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.stage_b_panel import (
    ANNOTATION_CONTEXT, TIME_FORMAT, PanelRecord, PanelWire, normalize_missing_words, seconds_to_clock,
    wire_to_record_with_notes,
)
from pipeline.stage_b.panel_books import PANEL_TEMPLATE, UNCLASSIFIED, PanelBooks, render_panel_prompt, turn_frame
from pipeline.stage_b.panel_validate import validate_record
from pipeline.stage_b.replicate_agreement import AlignmentSettings, align

PASSES = ("primary", "replicate")
SEGMENTS = ("presentation", "qa", "social", "demonstration", "negotiation", "none")
# the segment of a turn that is not a presentation turn, by the kind of its exchange
SEGMENT_OF_KIND = {"substantive": "qa", "social": "social", "demonstration": "demonstration",
                   "negotiation": "negotiation"}


@dataclass(frozen=True)
class PanelConfig:
    run_name: str
    model: ModelConfig
    limits: LimitsConfig
    invalid_output_retries: int
    alignment: AlignmentSettings
    source_path: Path


def load_panel_config(path: Path | str) -> PanelConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"panel config not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    try:
        model = data["model"]
        run_name = data["run_name"]
        if not isinstance(run_name, str) or not run_name:
            raise ConfigError(f"{path}: run_name must be a non-empty string")
        defaults = LimitsConfig()
        lim = data.get("limits") or {}
        limits = LimitsConfig(
            requests_per_minute=float(lim.get("requests_per_minute", defaults.requests_per_minute)),
            tokens_per_minute=int(lim.get("tokens_per_minute", defaults.tokens_per_minute)),
            max_concurrency=int(lim.get("max_concurrency", defaults.max_concurrency)),
            max_attempts=int(lim.get("max_attempts", defaults.max_attempts)),
            backoff_base_s=float(lim.get("backoff_base_s", defaults.backoff_base_s)),
            backoff_cap_s=float(lim.get("backoff_cap_s", defaults.backoff_cap_s)),
        )
        al = data.get("alignment") or {}
        base = AlignmentSettings()
        return PanelConfig(
            run_name=run_name,
            model=ModelConfig(
                name=str(model["name"]),
                temperature=float(model["temperature"]),
                max_output_tokens=int(model["max_output_tokens"]),
            ),
            limits=limits,
            invalid_output_retries=int(data.get("invalid_output_retries", 1)),
            alignment=AlignmentSettings(
                event_tolerance_s=float(al.get("event_tolerance_s", base.event_tolerance_s)),
                decision_tolerance_s=float(al.get("decision_tolerance_s", base.decision_tolerance_s)),
                # older config files also carry turn_iou, which nothing reads any more
            ),
            source_path=path,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: invalid panel config ({exc!r})") from exc


def prepared_window(record: PanelRecord) -> list[str]:
    """The prepared spans as clock ranges, e.g. ["0:15-1:43"]."""
    return [f"{seconds_to_clock(s.start_s)}-{seconds_to_clock(s.end_s)}" for s in record.prepared_spans]


def derive(record: PanelRecord, books: PanelBooks) -> dict:
    """What analysis reads off a record without re-deriving it: no judgment, only bookkeeping.

    A turn's segment is `presentation` when its function is `presents`, otherwise the kind of
    its exchange (substantive is `qa`), otherwise `none`. Labels are read in the codebook's
    spelling. A question or answer turn's frame comes from its concept; an answer's frame match
    compares it with the first question it responds to that has a frame. Every concept or topic
    of `other` is listed with the name the model gave it, for review.
    """
    record = books.canonical_record(record)
    kind_of: dict[int, str] = {}
    for e in record.exchanges:
        kind_of.setdefault(e.exchange_id, e.kind)
    turn_segment: dict[str, str] = {}
    seconds = {name: 0.0 for name in SEGMENTS}
    for t in record.turns:
        if t.function == "presents":
            segment = "presentation"
        else:
            segment = SEGMENT_OF_KIND.get(kind_of.get(t.exchange_id), "none")
        turn_segment[str(t.turn_id)] = segment
        seconds[segment] += max(0.0, t.end_s - t.start_s)
    counts = {kind: 0 for kind in SEGMENT_OF_KIND}
    for e in record.exchanges:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    frames: dict[int, str] = {}
    for t in record.turns:
        frame = turn_frame(t.function, [(c.code, c.value) for c in t.codes])
        if frame is not None:
            frames.setdefault(t.turn_id, frame)
    question_ids = {t.turn_id for t in record.turns if t.function == "question"}
    frame_match: dict[str, str] = {}
    for t in record.turns:
        if t.function != "answer" or t.turn_id not in frames:
            continue
        asked = next((frames[i] for i in t.responds_to if i in question_ids and i in frames), None)
        if asked is not None:
            frame_match[str(t.turn_id)] = _frame_match(asked, frames[t.turn_id])
    return {"turn_segment": turn_segment, "segment_seconds": seconds, "exchange_counts": counts,
            "turn_frame": {str(i): f for i, f in frames.items()}, "frame_match": frame_match,
            "other_concepts": books.other_concepts(record.model_dump(mode="json"))}


def _frame_match(question: str, answer: str) -> str:
    if UNCLASSIFIED in (question, answer):
        return UNCLASSIFIED
    if "NEUTRAL" in (question, answer):
        return "neutral"
    return "match" if question == answer else f"switch_to_{answer}"


def unit_key(video_id: str, pass_name: str) -> str:
    return f"panel__{video_id}__{pass_name}"


def plan_calls(videos: list[dict], run_dir: Path) -> list[dict]:
    rows = []
    for video in videos:
        for name in PASSES:
            cached = (Path(run_dir) / video["video_id"] / "results" / f"{name}.json").exists()
            rows.append({"video_id": video["video_id"], "pass": name, "cached": cached})
    return rows


def _read(path: Path):
    return json.loads(path.read_text())


def _json_text(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(obj))


def _git_commit() -> str | None:
    """HEAD of the repository this code runs from; informational, None when git cannot say."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit if out.returncode == 0 and commit else None


def _calls_logged(calls_log: Path) -> int:
    if not calls_log.exists():
        return 0
    return sum(1 for ln in calls_log.read_text().splitlines() if ln.strip())


def _model_dict(model: ModelConfig) -> dict:
    return {"name": model.name, "temperature": model.temperature, "max_output_tokens": model.max_output_tokens}


_INCLUDE = re.compile(r"""\{%-?\s*include\s+['"]([^'"]+)['"]""")


def _bundle_sources(release: InstrumentRelease) -> dict[str, Path]:
    """The files a run copies and hashes: the manifest and its books, every template the manifest lists (with the
    panel template), and every file those templates include from the templates dir, followed through nested
    includes, so the bundle alone can render the prompt."""
    sources = {name: release.canonical_dir / name for name in canonical_files(release, optional=False)}
    pending = sorted({PANEL_TEMPLATE} | {book.template for book in release.books.values()})
    while pending:
        name = pending.pop(0)
        if name in sources:
            continue
        sources[name] = release.templates_dir / name
        pending.extend(_INCLUDE.findall(sources[name].read_text()))
    return dict(sorted(sources.items()))


def run_video(provider, release: InstrumentRelease, books: PanelBooks, cfg: PanelConfig, video: dict,
              run_root: Path, sleep: Callable[[float], None] = time.sleep,
              before_call: Callable[[], None] | None = None) -> dict:
    problems = check_release(release)
    if problems:
        raise ConfigError("instrument release is not clean: " + "; ".join(problems))

    video_id = video["video_id"]
    duration_s = float(video["duration_s"])
    vdir = Path(run_root) / video_id
    sources = _bundle_sources(release)
    bundle_hashes = {name: sha256_file(path) for name, path in sources.items()}
    prompt = render_panel_prompt(books, release.templates_dir)
    phash = prompt_hash(prompt)
    source = Path(video["path"])
    source_sha = sha256_file(source)
    schema_text = _json_text(gemini_response_schema(PanelWire))
    current = {
        "video_id": video_id,
        "source": str(source),
        "duration_s": duration_s,
        "source_sha256": source_sha,
        "release_id": release.release_id,
        "model": _model_dict(cfg.model),
        "prompt_hash": phash,
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "bundle_sha256": bundle_hashes,
        "annotation_context": ANNOTATION_CONTEXT,
        "passes": list(PASSES),
    }
    exp_path = vdir / "experiment.json"
    if exp_path.exists():
        old = _read(exp_path)
        for key in ("prompt_hash", "schema_sha256", "bundle_sha256", "source_sha256", "model"):
            if old.get(key) != current[key]:
                raise ConfigError(
                    f"{exp_path}: {key} differs from the current one; run was started with a "
                    "different instrument; use a new run name"
                )
    else:
        for sub in ("bundle", "prompts", "raw", "results"):
            (vdir / sub).mkdir(parents=True, exist_ok=True)
        for name, path in sources.items():
            shutil.copy2(path, vdir / "bundle" / name)
        (vdir / "prompts" / "panel_full.txt").write_text(prompt)
        (vdir / "prompts" / "panel_full.schema.json").write_text(schema_text, encoding="utf-8")
        _dump(exp_path, {**current, "git_commit": _git_commit()})
    for sub in ("raw", "results"):
        (vdir / sub).mkdir(parents=True, exist_ok=True)

    pending = [n for n in PASSES if not (vdir / "results" / f"{n}.json").exists()]
    handle = None
    upload_error: str | None = None
    if pending:
        try:
            handle = with_retries(
                lambda: provider.upload_media(source, file_sha256=source_sha),
                max_attempts=cfg.limits.max_attempts, base_s=cfg.limits.backoff_base_s,
                cap_s=cfg.limits.backoff_cap_s, sleep=sleep)
        except RunFatalError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded as failed passes
            upload_error = f"upload failed: {type(exc).__name__}: {exc}"
    calls_log = vdir / "calls.jsonl"
    made = 0
    status: dict[str, str] = {}

    for name in PASSES:
        key = unit_key(video_id, name)
        result_path = vdir / "results" / f"{name}.json"
        if result_path.exists():
            status[name] = "ok"
            continue
        if upload_error is not None:
            _dump(vdir / "results" / f"{name}_failed.json",
                  {"pass": name, "unit_key": key, "message": upload_error})
            status[name] = "failed"
            continue
        prior = 0
        if calls_log.exists():
            prior = sum(1 for ln in calls_log.read_text().splitlines()
                        if ln.strip() and json.loads(ln).get("unit_key") == key)
        attempt = prior
        invalid = 0
        failure: str | None = None
        response = record = None
        notes: list[str] = []

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
                "unit_key": key, "pass": name, "attempt": attempt, "ok": ok, "error_class": error_class,
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
                    prompt=prompt, media=handle, schema=PanelWire, model=cfg.model, unit_key=key))
            except BaseException as exc:
                log(False, type(exc).__name__, None, time.monotonic() - t0, exc)
                raise
            try:
                record, dropped = wire_to_record_with_notes(resp.parsed, books.categorical_code_ids())
            except ValueError as exc:
                # a malformed clock time is an invalid output exactly like an unparseable body
                invalid_exc = SchemaInvalidError(f"{key}: {exc}", raw_text=resp.raw_text)
                invalid_exc.usage = {"input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                                     "thinking_tokens": resp.thinking_tokens}
                invalid_exc.model_version = resp.model_version
                log(False, type(invalid_exc).__name__, None, time.monotonic() - t0, invalid_exc)
                raise invalid_exc from exc
            log(True, None, resp, time.monotonic() - t0)
            return resp, record, dropped

        while True:
            try:
                response, record, notes = with_retries(
                    once, max_attempts=cfg.limits.max_attempts, base_s=cfg.limits.backoff_base_s,
                    cap_s=cfg.limits.backoff_cap_s, sleep=sleep)
                break
            except RunFatalError:
                raise
            except SchemaInvalidError as exc:
                invalid += 1
                (vdir / "raw" / f"{name}__attempt{attempt}_invalid.txt").write_text(exc.raw_text or "")
                _dump(vdir / "raw" / f"{name}__attempt{attempt}_error.json", {"message": str(exc)})
                if invalid >= 1 + cfg.invalid_output_retries:
                    failure = str(exc)
                    break
            except Exception as exc:  # noqa: BLE001 - recorded, the other pass still runs
                failure = f"{type(exc).__name__}: {exc}"
                body = getattr(exc, "raw_text", None)
                if body is not None:
                    (vdir / "raw" / f"{name}__attempt{attempt}_failed.txt").write_text(body)
                break

        if failure is not None or response is None or record is None:
            _dump(vdir / "results" / f"{name}_failed.json",
                  {"pass": name, "unit_key": key, "message": failure or "no parsed response"})
            status[name] = "failed"
            continue

        (vdir / "raw" / f"{name}.json").write_text(response.raw_text)  # wire form, as returned
        report = validate_record(record, books, duration_s, normalizations=notes)
        _dump(vdir / "results" / f"{name}_validation.json", {"errors": report.errors, "flags": report.flags})
        _dump(result_path, {
            "pass": name, "unit_key": key, "annotation_context": ANNOTATION_CONTEXT,
            "prompt_hash": phash, "model_version": response.model_version,
            "time_format": TIME_FORMAT,
            "prepared_window": prepared_window(record),
            "derived": derive(record, books),
            "normalizations": notes,      # values the wire gave together with a missing reason, dropped
            "record": record.model_dump(mode="json"),
        })
        stale = vdir / "results" / f"{name}_failed.json"
        if stale.exists():
            stale.unlink()
        status[name] = "ok"

    if all(status[n] == "ok" for n in PASSES):
        run_agreement(release, books, cfg, video_id, Path(run_root))

    n_ok = sum(1 for n in PASSES if status[n] == "ok")
    validation_errors = {}
    for n in PASSES:
        vp = vdir / "results" / f"{n}_validation.json"
        if status[n] == "ok" and vp.exists():
            validation_errors[n] = len(_read(vp)["errors"])
    completion = {
        "status": "complete" if n_ok == len(PASSES) else ("partial" if n_ok else "failed"),
        "passes": status,
        "validation_errors": validation_errors,
        "generation_calls": made,
        "generation_calls_total": _calls_logged(calls_log),
        "human_validated": False,
    }
    _dump(vdir / "completion.json", completion)
    return completion


def run_agreement(release: InstrumentRelease, books: PanelBooks, cfg: PanelConfig, video_id: str,
                  run_root: Path) -> dict:
    vdir = Path(run_root) / video_id
    records = {}
    for name in PASSES:
        path = vdir / "results" / f"{name}.json"
        if not path.exists():
            raise ConfigError(f"{video_id}: no {name} result; agreement needs both passes")
        records[name] = PanelRecord.model_validate(_read(path)["record"])
    report = align(records["primary"], records["replicate"], books, cfg.alignment)
    report = {
        "video_id": video_id,
        "release_id": release.release_id,
        "annotation_context": ANNOTATION_CONTEXT,
        "settings": asdict(cfg.alignment),
        **report,
    }
    _dump(vdir / "agreement.json", report)
    return report


def revalidate(books: PanelBooks, video_id: str, duration_s: float, run_root: Path) -> list[str]:
    """Rewrite results/<pass>_validation.json from the stored record of each pass that exists.

    A stored record that holds a word for missing as a code value (written before the wire
    conversion repaired those) is repaired first and its result file rewritten with the added
    notes; any other result file is left as it is. raw/<pass>.json always keeps what the model
    returned. Offline. Returns the names of the passes found, in pass order.
    """
    results = Path(run_root) / video_id / "results"
    present = []
    for name in PASSES:
        path = results / f"{name}.json"
        if not path.exists():
            continue
        stored = _read(path)
        repaired = normalize_missing_words(stored["record"], books.categorical_code_ids())
        record = PanelRecord.model_validate(stored["record"])
        # the notes of the wire conversion cannot be recomputed from the record, so they are carried over
        notes = (stored.get("normalizations") or []) + repaired
        if repaired:
            _dump(path, {**stored, "prepared_window": prepared_window(record), "derived": derive(record, books),
                         "normalizations": notes, "record": record.model_dump(mode="json")})
        report = validate_record(record, books, float(duration_s), normalizations=notes)
        _dump(results / f"{name}_validation.json", {"errors": report.errors, "flags": report.flags})
        present.append(name)
    return present


def _validation_errors(results: Path, pass_name: str) -> int | None:
    """Local validation errors of a pass, or None when the pass or its validation report is absent."""
    report = results / f"{pass_name}_validation.json"
    if not (results / f"{pass_name}.json").exists() or not report.exists():
        return None
    return len(_read(report).get("errors") or [])


def load_best_record(video_dir: Path | str) -> tuple[dict, str]:
    """The result analysis should read for a video: (result dict, "reviewed" | "primary" | "replicate").

    The human-reviewed result (results/reviewed.json, written by scripts/review_panel.py) when there
    is one. Otherwise the pass that passes local validation: the primary, unless it has validation
    errors and the replicate has fewer (the two passes are identical requests, so either stands for the
    video; the rule looks only at validation, never at content). All carry "record", "derived" and
    "prepared_window".
    """
    results = Path(video_dir) / "results"
    reviewed = results / "reviewed.json"
    if reviewed.exists():
        return _read(reviewed), "reviewed"
    primary = results / "primary.json"
    if not primary.exists():
        raise ConfigError(f"{Path(video_dir).name}: no primary result in {results}")
    name = model_pass(video_dir)
    return _read(results / f"{name}.json"), name


def model_pass(video_dir: Path | str) -> str:
    """The model's pass that analysis reads when no human-reviewed result stands in for it (load_best_record's rule):
    "primary", unless the primary has validation errors and the replicate has fewer ("replicate"). The human coders'
    sheets, the clip session's cards and the cut are drawn from this pass, so that coders validate the answers the
    analysis reads (final review I1, 2026-10-22)."""
    results = Path(video_dir) / "results"
    primary_errors = _validation_errors(results, "primary")
    replicate_errors = _validation_errors(results, "replicate")
    if primary_errors and replicate_errors is not None and replicate_errors < primary_errors:
        return "replicate"
    return "primary"


STRUCTURAL = "structural"       # a review resolution: the unit itself is wrong, or one pass lacks it
_DECISION_PATHS = ("decision:", "replicate_only:decision@")


def review_open_issues(video_dir: Path | str) -> dict:
    """What a reviewed record still leaves open, so that analysis reading it (load_best_record) can say so.

    reviewed_validation_errors: the errors of results/reviewed_validation.json (0 when the file is
    absent). review_structural_on_decisions: the reviewed paths of review/review_log.jsonl whose last
    adjudicator entry is a structural judgment on a decision (a wrong or missing decision, or one judged absent or
    present against the primary), which no value correction could apply to the record.
    """
    video_dir = Path(video_dir)
    validation = video_dir / "results" / "reviewed_validation.json"
    errors = len(_read(validation).get("errors") or []) if validation.exists() else 0
    log = video_dir / "review" / "review_log.jsonl"
    last: dict[str, dict] = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                # a coder's line measures the model and never corrects the record, so it opens no issue
                if not entry.get("implied") and entry.get("role", "adjudicator") == "adjudicator":
                    last[entry["path"]] = entry
    structural = sum(1 for path, entry in last.items()
                     if entry.get("resolution") == STRUCTURAL and path.startswith(_DECISION_PATHS))
    return {"reviewed_validation_errors": errors, "review_structural_on_decisions": structural}
