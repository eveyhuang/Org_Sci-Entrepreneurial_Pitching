"""Run configuration: typed, validated at load time, fail fast."""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel

from pipeline.errors import ConfigError

Unit = Literal["PITCH", "EPISODE", "PREPARED"]
Media = Literal["full", "none"]

VALID_UNITS = ("PITCH", "EPISODE", "PREPARED")
# "full" is the complete audio-video stream. "none" is a text-only pass whose construct is
# defined over text (cross-episode consistency) — not a degraded-input condition.
VALID_MEDIA = ("full", "none")
VALID_CONTEXT = ("roster", "timeline", "timeline_slice", "pitch_summary", "episode_table")
VALID_PRODUCES = ("timeline", "episodes")
GENERATED_SCHEMA = "codebook"  # literal `schema:` value: build the record model from the codebook

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"
DOCS_CODEBOOKS = Path(__file__).resolve().parents[1] / "docs" / "codebooks"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    temperature: float = 0.0
    max_output_tokens: int = 16384


@dataclass(frozen=True)
class LimitsConfig:
    requests_per_minute: float = 8.0
    # Accepted for forward compatibility but NOT yet enforced: no limiter reads it.
    # Control spend with requests_per_minute and max_concurrency.
    tokens_per_minute: int = 800_000
    max_concurrency: int = 4
    max_attempts: int = 6
    backoff_base_s: float = 5.0
    backoff_cap_s: float = 120.0


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    video_dir: Path
    metadata_csv: Path
    max_duration_s: float | None = None  # longer videos are registered but never enqueued
    dev_set: Path | None = None  # fixed, recorded list of dev-set video ids


@dataclass(frozen=True)
class PassConfig:
    name: str
    unit: Unit
    prompt: str
    schema: str
    k: int = 1
    media: Media = "full"
    depends_on: str | None = None
    model: ModelConfig | None = None      # per-pass override; still exactly one pinned model
    codebook: Path | None = None          # per-pass codebook; None means the run-level one
    context: tuple[str, ...] = ()         # upstream artifacts rendered into the prompt per unit
    produces: str | None = None           # post-processor run after a record validates


@dataclass(frozen=True)
class PipelineConfig:
    run_name: str
    corpus: CorpusConfig
    model: ModelConfig
    limits: LimitsConfig
    passes: tuple[PassConfig, ...]
    codebook: Path | None
    source_path: Path
    prompt_dir: Path
    raw_yaml: str = field(repr=False, default="")
    historical: bool = False

    def get_pass(self, name: str) -> PassConfig:
        for p in self.passes:
            if p.name == name:
                return p
        raise ConfigError(f"No pass named {name!r} in {self.source_path}")

    def model_for(self, pass_cfg: PassConfig) -> ModelConfig:
        return pass_cfg.model or self.model

    def codebook_for(self, pass_cfg: PassConfig) -> Path | None:
        return pass_cfg.codebook or self.codebook

    def pass_producing(self, kind: str) -> PassConfig | None:
        return next((p for p in self.passes if p.produces == kind), None)

    def dev_video_ids(self) -> list[str]:
        if self.corpus.dev_set is None:
            raise ConfigError(f"no dev set configured (corpus.dev_set) in {self.source_path}")
        ids = read_video_list(self.corpus.dev_set)
        if not ids:
            raise ConfigError(f"{self.corpus.dev_set} lists no videos yet")
        return ids


def load_config(path: Path | str, prompt_dir: Path | None = None) -> PipelineConfig:
    """Load and fully validate a run config. Raises ConfigError on any problem."""
    path = Path(path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw_yaml = path.read_text()
    data = yaml.safe_load(raw_yaml) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")

    base = path.parent
    historical = data.get("historical", False)
    if not isinstance(historical, bool):
        raise ConfigError(f"{path}: `historical` must be true or false, got {historical!r}")
    if prompt_dir is not None:
        prompt_dir = Path(prompt_dir)
    elif data.get("prompt_dir") is not None:
        prompt_dir = _resolve(base, _require(data, "prompt_dir", str, path))
        if not prompt_dir.is_dir():
            raise ConfigError(f"prompt_dir is not a directory: {prompt_dir}")
    else:
        prompt_dir = DEFAULT_PROMPT_DIR
    if not historical and prompt_dir.resolve().is_relative_to(DOCS_CODEBOOKS):
        raise ConfigError(
            f"{path}: prompt_dir {prompt_dir} is under docs/codebooks/ (a draft or archive). "
            f"Current runs read pipeline/prompts/ only; set `historical: true` for a "
            f"reproduction of an archived instrument."
        )

    run_name = _require(data, "run_name", str, path)
    corpus = _load_corpus(_require(data, "corpus", dict, path), base, path)
    model = _load_model(_require(data, "model", dict, path))
    limits = _load_limits(data.get("limits") or {})
    codebook: Path | None = None
    if data.get("codebook") is not None:
        codebook = _resolve(base, _require(data, "codebook", str, path))
        if not codebook.exists():
            raise ConfigError(f"codebook not found: {codebook}")

    passes = _load_passes(data.get("passes"), prompt_dir, path, base, codebook)

    if not historical:
        for book_path in [codebook, *[p.codebook for p in passes if p.codebook is not None]]:
            if book_path is not None and book_path.resolve().is_relative_to(DOCS_CODEBOOKS):
                raise ConfigError(
                    f"{path}: codebook {book_path} is under docs/codebooks/ (a draft or archive). "
                    f"Current runs read pipeline/codebooks/ only; set `historical: true` for a "
                    f"reproduction of an archived instrument."
                )

    return PipelineConfig(
        run_name=run_name,
        corpus=corpus,
        model=model,
        limits=limits,
        passes=passes,
        codebook=codebook,
        source_path=path,
        prompt_dir=prompt_dir,
        raw_yaml=raw_yaml,
        historical=historical,
    )


def _load_corpus(data: dict[str, Any], base: Path, path: Path) -> CorpusConfig:
    video_dir = _resolve(base, _require(data, "video_dir", str, path))
    if not video_dir.is_dir():
        raise ConfigError(f"corpus.video_dir is not a directory: {video_dir}")
    metadata_csv = _resolve(base, _require(data, "metadata_csv", str, path))
    if not metadata_csv.exists():
        raise ConfigError(f"corpus.metadata_csv not found: {metadata_csv}")
    max_duration_s = data.get("max_duration_s")
    if max_duration_s is not None:
        max_duration_s = float(max_duration_s)
        if max_duration_s <= 0:
            raise ConfigError("corpus.max_duration_s must be positive")
    dev_set = None
    if data.get("dev_set") is not None:
        dev_set = _resolve(base, _require(data, "dev_set", str, path))
        if not dev_set.exists():
            raise ConfigError(f"corpus.dev_set not found: {dev_set}")
    return CorpusConfig(
        name=_require(data, "name", str, path),
        video_dir=video_dir,
        metadata_csv=metadata_csv,
        max_duration_s=max_duration_s,
        dev_set=dev_set,
    )


def _load_model(data: dict[str, Any]) -> ModelConfig:
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(
            "model.name must be a single model identifier string; "
            "fallback chains are forbidden (a run pins one model)"
        )
    return ModelConfig(
        name=name,
        temperature=float(data.get("temperature", 0.0)),
        max_output_tokens=int(data.get("max_output_tokens", 16384)),
    )


def _load_limits(data: dict[str, Any]) -> LimitsConfig:
    defaults = LimitsConfig()
    return LimitsConfig(
        requests_per_minute=float(data.get("requests_per_minute", defaults.requests_per_minute)),
        tokens_per_minute=int(data.get("tokens_per_minute", defaults.tokens_per_minute)),
        max_concurrency=int(data.get("max_concurrency", defaults.max_concurrency)),
        max_attempts=int(data.get("max_attempts", defaults.max_attempts)),
        backoff_base_s=float(data.get("backoff_base_s", defaults.backoff_base_s)),
        backoff_cap_s=float(data.get("backoff_cap_s", defaults.backoff_cap_s)),
    )


def _load_passes(
    raw: Any, prompt_dir: Path, path: Path, base: Path, run_codebook: Path | None
) -> tuple[PassConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"config must declare a non-empty `passes` list: {path}")
    seen: list[str] = []
    passes: list[PassConfig] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ConfigError(f"each pass must be a mapping: {path}")
        name = _require(entry, "name", str, path)
        unit = _require(entry, "unit", str, path)
        if unit not in VALID_UNITS:
            raise ConfigError(f"pass {name!r}: unit must be one of {VALID_UNITS}, got {unit!r}")
        media = entry.get("media", "full")
        if media not in VALID_MEDIA:
            raise ConfigError(f"pass {name!r}: media must be one of {VALID_MEDIA}, got {media!r}")
        prompt = _require(entry, "prompt", str, path)
        if not (prompt_dir / prompt).exists():
            raise ConfigError(f"pass {name!r}: prompt template not found: {prompt_dir / prompt}")
        if media == "none" and unit != "PITCH":
            raise ConfigError(f"pass {name!r}: media: none is only valid for unit PITCH")
        pass_model = None
        if entry.get("model") is not None:
            if not isinstance(entry["model"], dict):
                raise ConfigError(f"pass {name!r}: model must be a mapping with a `name`")
            pass_model = _load_model(entry["model"])
        pass_codebook = None
        if entry.get("codebook") is not None:
            pass_codebook = _resolve(base, _require(entry, "codebook", str, path))
            if not pass_codebook.exists():
                raise ConfigError(f"pass {name!r}: codebook not found: {pass_codebook}")
        schema = _require(entry, "schema", str, path)
        if schema == GENERATED_SCHEMA:
            if pass_codebook is None and run_codebook is None:
                raise ConfigError(
                    f"pass {name!r}: `schema: codebook` needs a codebook (per-pass or run-level)"
                )
        else:
            import_schema(schema)  # raises ConfigError if unimportable
        depends_on = entry.get("depends_on")
        if depends_on is not None and depends_on not in seen:
            raise ConfigError(
                f"pass {name!r}: depends_on {depends_on!r} must name an earlier pass"
            )
        context = entry.get("context") or []
        if not isinstance(context, list) or any(c not in VALID_CONTEXT for c in context):
            raise ConfigError(f"pass {name!r}: context entries must be among {VALID_CONTEXT}")
        produces = entry.get("produces")
        if produces is not None and produces not in VALID_PRODUCES:
            raise ConfigError(f"pass {name!r}: produces must be one of {VALID_PRODUCES}")
        k = int(entry.get("k", 1))
        if k < 1:
            raise ConfigError(f"pass {name!r}: k must be >= 1")
        passes.append(
            PassConfig(
                name=name, unit=unit, prompt=prompt, schema=schema, k=k, media=media,
                depends_on=depends_on, model=pass_model, codebook=pass_codebook,
                context=tuple(context), produces=produces,
            )
        )
        seen.append(name)
    return tuple(passes)


def import_schema(dotted: str) -> type[BaseModel]:
    """Import a pydantic model from a dotted path like `pkg.mod.Model`."""
    module_name, _, attr = dotted.rpartition(".")
    if not module_name:
        raise ConfigError(f"schema must be a dotted path, got {dotted!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(f"cannot import schema module {module_name!r}: {exc}") from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ConfigError(f"schema {attr!r} not found in {module_name!r}") from exc
    if not (isinstance(obj, type) and issubclass(obj, BaseModel)):
        raise ConfigError(f"schema {dotted!r} is not a pydantic BaseModel subclass")
    return obj


def read_video_list(path: Path) -> list[str]:
    lines = (line.strip() for line in Path(path).read_text().splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def config_hash(cfg: PipelineConfig) -> str:
    """Stable hash of the config text — used to decide run-directory reuse."""
    return hashlib.sha256(cfg.raw_yaml.encode()).hexdigest()[:16]


def _require(data: dict[str, Any], key: str, kind: type, where: Path):
    value = data.get(key)
    if not isinstance(value, kind):
        raise ConfigError(f"missing or invalid {key!r} (expected {kind.__name__}) in {where}")
    return value


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()
