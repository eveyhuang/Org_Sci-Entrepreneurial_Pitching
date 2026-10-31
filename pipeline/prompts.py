"""Prompt rendering and hashing.

The prompt hash covers the fully rendered text — template plus codebook content —
so editing either invalidates exactly the affected work units in the registry.
StrictUndefined turns a typo'd template variable into a loud failure rather than a
silently truncated instruction.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pipeline.codebook import Codebook, load_codebook
from pipeline.config import PassConfig, PipelineConfig

DEFAULT_PROMPT_DIR = Path(__file__).parent / "prompts"


@lru_cache(maxsize=8)
def _environment(prompt_dir: Path) -> Environment:
    return Environment(
        # Templates (and anything they include) resolve from the pass's prompt_dir only.
        loader=FileSystemLoader(str(prompt_dir)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )


def render_prompt(
    template_name: str,
    *,
    codebook: Codebook,
    context: dict | None = None,
    prompt_dir: Path | str | None = None,
) -> str:
    directory = Path(prompt_dir) if prompt_dir else DEFAULT_PROMPT_DIR
    template = _environment(directory).get_template(template_name)
    return template.render(codebook=codebook, entries=codebook.entries, **(context or {}))


def prompt_hash(rendered: str) -> str:
    """First 16 hex chars of the SHA-256 of the rendered prompt."""
    return hashlib.sha256(rendered.encode()).hexdigest()[:16]


def load_pass_codebook(cfg: PipelineConfig, pass_cfg: PassConfig) -> Codebook:
    """The pass's codebook, else the run's, else an empty one (structural passes need none)."""
    path = cfg.codebook_for(pass_cfg)
    return load_codebook(path) if path is not None else Codebook(entries=[])


def render_pass_prompt(
    cfg: PipelineConfig,
    pass_cfg: PassConfig,
    codebook: Codebook | None = None,
    context: dict | None = None,
) -> tuple[str, str]:
    """Render a pass's prompt and return (text, hash).

    The hash covers the instruction text alone — the template rendered with no
    per-item context — so that a per-item context block (a Stage B episode header,
    say) never changes a work unit's identity, while an edit to the template or the
    codebook always does.
    """
    book = codebook if codebook is not None else load_pass_codebook(cfg, pass_cfg)
    instructions = render_prompt(
        pass_cfg.prompt, codebook=book, context=None, prompt_dir=cfg.prompt_dir
    )
    digest = prompt_hash(instructions)
    if not context:
        return instructions, digest
    text = render_prompt(
        pass_cfg.prompt, codebook=book, context=context, prompt_dir=cfg.prompt_dir
    )
    return text, digest
