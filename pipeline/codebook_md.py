"""Markdown codebooks: the Markdown file is the master, the YAML is generated.

People read and edit `<name>.md`. The pipeline loads `<name>.yaml`, which carries a
header recording the hash of the Markdown it was generated from and the hash of its
own body. `ensure_synced` runs every time a codebook is loaded, so a run can never
annotate with a YAML that is older than its Markdown — and a YAML edited by hand is
refused rather than silently overwritten or silently used.

A codebook with no sibling `.md` is loaded as plain YAML, unchanged.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from pathlib import Path
from typing import Any

import yaml

from pipeline.errors import ConfigError

WRAP = 92
META_KEYS = ("name", "applies_to", "dimension", "parent", "side", "tier", "type", "scale", "unit", "observability", "values")
TEXT_LABELS = {
    "definition": "definition",
    "include if": "include_if",
    "exclude if": "exclude_if",
    "provenance": "provenance",
}
_LABEL = re.compile(r"^\*\*([A-Za-z][A-Za-z ]*?)\.?\*\*\.?\s*(?:\*\([^)]*\)\*\.?)?\s*(.*)$")
_ANCHOR_INT = re.compile(r"^(\d+)\.\s+(.*)$")
_ANCHOR_NUM = re.compile(r"^-\s+(-?\d+(?:\.\d+)?):\s+(.*)$")
_EXAMPLE = re.compile(r"^-\s+\[(real|illustrative)(?:\s*\|\s*([^\]]+?))?\]\s+(.*)$")
_HEADER_SRC = re.compile(r"^# source-sha256: ([0-9a-f]{64})$", re.M)
_HEADER_BODY = re.compile(r"^# body-sha256: ([0-9a-f]{64})$", re.M)
_BODY_MARK = "# ---- generated body below ----\n"

ARCHIVE_ROOT = Path(__file__).resolve().parents[1] / "docs" / "codebooks" / "archive"


def is_archived(path: Path | str) -> bool:
    """True when the file lives under the frozen archive; such files are never regenerated."""
    try:
        Path(path).resolve().relative_to(ARCHIVE_ROOT.resolve())
    except ValueError:
        return False
    return True


class CodebookMarkdownError(ConfigError):
    """The Markdown codebook could not be parsed."""


# --------------------------------------------------------------------------
# Markdown -> entries
# --------------------------------------------------------------------------
def md_to_entries(text: str, source: str = "<markdown>") -> list[dict[str, Any]]:
    """Parse a Markdown codebook into the list-of-dicts shape the YAML holds."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    block: str | None = None  # which labelled block we are inside
    buffer: list[str] = []

    def fail(lineno: int, message: str) -> None:
        code = current["id"] if current else "preamble"
        raise CodebookMarkdownError(f"{source}:{lineno}: [{code}] {message}")

    def flush() -> None:
        nonlocal buffer
        if current is None or block is None:
            buffer = []
            return
        if block in TEXT_LABELS.values():
            current[block] = _join_paragraphs(buffer)
        elif block == "anchors" and buffer and current["anchors"]:
            current["anchors"][-1]["description"] = _join_paragraphs(
                [current["anchors"][-1]["description"], *buffer]
            )
        elif block == "examples" and buffer and current["examples"]:
            current["examples"][-1]["text"] = _join_paragraphs(
                [current["examples"][-1]["text"], *buffer]
            )
        buffer = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        stripped = line.strip()

        if line.startswith("## "):
            flush()
            code_id = line[3:].strip().strip("`")
            if not code_id:
                fail(lineno, "empty code id in heading")
            current = {"id": code_id}
            entries.append(current)
            block = "meta"
            continue
        if current is None or line.startswith("# ") or stripped == "---":
            continue  # preamble, section dividers, horizontal rules

        label = _LABEL.match(stripped)
        if label and stripped.startswith("**"):
            name, rest = label.group(1).strip().lower(), label.group(2).strip()
            flush()
            if name in TEXT_LABELS:
                block = TEXT_LABELS[name]
                buffer = [rest] if rest else []
            elif name == "scale":
                block = "anchors"
                current["anchors"] = []
            elif name == "examples":
                block = "examples"
                current["examples"] = []
            else:
                fail(lineno, f"unknown section label **{label.group(1)}** — expected one of "
                             "Definition, Scale, Include if, Exclude if, Provenance, Examples")
            continue

        if block == "meta":
            if not stripped:
                continue
            if not stripped.startswith("- "):
                fail(lineno, "expected a '- key: value' metadata line or a **Definition.** label")
            for part in stripped[2:].split(" · "):
                key, sep, value = part.partition(":")
                key, value = key.strip().lower(), value.strip()
                if not sep or key not in META_KEYS:
                    fail(lineno, f"unknown metadata key {key!r} — expected one of {', '.join(META_KEYS)}")
                current[key] = _parse_meta(key, value, lambda m: fail(lineno, m))
            continue

        if block == "anchors":
            match = _ANCHOR_INT.match(stripped) or _ANCHOR_NUM.match(stripped)
            if match:
                flush()
                number = float(match.group(1))
                current["anchors"].append(
                    {"value": int(number) if number.is_integer() else number,
                     "description": match.group(2).strip()}
                )
                continue
            if stripped and not current["anchors"]:
                fail(lineno, "scale items must look like '1. description'")
        if block == "examples":
            match = _EXAMPLE.match(stripped)
            if match:
                flush()
                example = {"text": match.group(3).strip(), "provenance": match.group(1)}
                if match.group(2):
                    example["source"] = match.group(2).strip()
                current["examples"].append(example)
                continue
            if stripped and not current["examples"]:
                fail(lineno, "examples must look like '- [real | Source] text' or '- [illustrative] text'")

        buffer.append(stripped)

    flush()
    if not entries:
        raise CodebookMarkdownError(f"{source}: no '## CODE-ID' headings found")
    return entries


def illustrative_examples(md_path: Path) -> list[tuple[str, str]]:
    """(code id, text) of every placeholder example. Must be empty before a codebook freeze."""
    entries = md_to_entries(Path(md_path).read_text(), source=str(md_path))
    return [(entry["id"], example["text"]) for entry in entries
            for example in entry.get("examples", []) if example["provenance"] == "illustrative"]


def _parse_meta(key: str, value: str, fail) -> Any:
    if key == "scale":
        parts = re.split(r"\s*(?:–|—|-|to)\s*", value.strip("[] "), maxsplit=1)
        try:
            lo, hi = (float(p) for p in parts)
        except ValueError:
            fail(f"scale must look like '1–5', got {value!r}")
        return [int(lo) if lo.is_integer() else lo, int(hi) if hi.is_integer() else hi]
    if key == "values":
        values = [v.strip().strip("`") for v in value.split(",") if v.strip()]
        if not values:
            fail("values must be a comma-separated list")
        return values
    return value


def _join_paragraphs(lines: list[str]) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


# --------------------------------------------------------------------------
# entries -> Markdown   (used once, to bootstrap a Markdown master from YAML)
# --------------------------------------------------------------------------
def entries_to_md(entries: list[dict[str, Any]], preamble: str) -> str:
    out: list[str] = [preamble.rstrip(), ""]
    last_dimension = None
    for entry in entries:
        if entry.get("dimension") != last_dimension:
            last_dimension = entry.get("dimension")
            out += ["---", "", f"# {last_dimension}", ""]
        out.append(f"## {entry['id']}")
        if entry.get("name"):
            out.append(f"- name: {entry['name']}")
        out.append(f"- dimension: {entry['dimension']}")
        if entry.get("parent"):
            out.append(f"- parent: {entry['parent']}")
        meta = [f"side: {entry.get('side', 'none')}", f"tier: {entry['tier']}", f"type: {entry['type']}"]
        if entry.get("scale"):
            lo, hi = entry["scale"]
            meta.append(f"scale: {_num(lo)}–{_num(hi)}")
        meta += [f"unit: {entry['unit']}", f"observability: {entry['observability']}"]
        if entry.get("applies_to"):
            meta.append(f"applies_to: {entry['applies_to']}")
        out.append("- " + " · ".join(meta))
        if entry.get("values"):
            out.append("- values: " + ", ".join(entry["values"]))
        out += ["", _wrap(f"**Definition.** {entry['definition'].strip()}"), ""]
        if entry.get("anchors"):
            out.append("**Scale**")
            for anchor in entry["anchors"]:
                value = anchor["value"]
                lead = f"{int(value)}. " if float(value).is_integer() else f"- {value}: "
                out.append(_wrap(lead + anchor["description"].strip(), indent="   "))
            out.append("")
        out += [_wrap(f"**Include if.** {entry['include_if'].strip()}"), ""]
        out += [_wrap(f"**Exclude if.** {entry['exclude_if'].strip()}"), ""]
        if entry.get("provenance"):
            out += [_wrap(f"**Provenance** *(never sent to the model)*. {entry['provenance'].strip()}"), ""]
        if entry.get("examples"):
            out.append("**Examples**")
            for example in entry["examples"]:
                tag = example["provenance"] + (f" | {example['source']}" if example.get("source") else "")
                out.append(_wrap(f"- [{tag}] {example['text'].strip()}", indent="  "))
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _wrap(text: str, indent: str = "") -> str:
    paragraphs = text.split("\n\n")
    wrapped = [
        textwrap.fill(p, width=WRAP, subsequent_indent=indent, break_long_words=False,
                      break_on_hyphens=False)
        for p in paragraphs
    ]
    return "\n\n".join(wrapped)


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------
def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def render_yaml(md_path: Path) -> str:
    """Parse and validate the Markdown, return the full generated YAML text."""
    from pipeline.codebook import Codebook  # local import: codebook imports this module
    from pydantic import ValidationError

    md_text = md_path.read_text()
    entries = md_to_entries(md_text, source=str(md_path))
    try:
        Codebook(entries=entries)
    except ValidationError as exc:
        raise CodebookMarkdownError(f"{md_path} parsed, but the codebook is invalid:\n{exc}") from exc
    body = yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=100)
    header = (
        f"# GENERATED from {md_path.name} — DO NOT EDIT THIS FILE.\n"
        f"# Edit the Markdown. It is re-synced automatically whenever the codebook is loaded\n"
        f"# (every annotate run), or on demand: python scripts/codebook_sync.py\n"
        f"# source-sha256: {_sha(md_text)}\n"
        f"# body-sha256: {_sha(body)}\n"
        f"{_BODY_MARK}"
    )
    return header + body


def _read_header(yaml_text: str) -> tuple[str | None, str | None, str]:
    src = _HEADER_SRC.search(yaml_text)
    body_hash = _HEADER_BODY.search(yaml_text)
    body = yaml_text.split(_BODY_MARK, 1)[1] if _BODY_MARK in yaml_text else yaml_text
    return (src.group(1) if src else None, body_hash.group(1) if body_hash else None, body)


def status(yaml_path: Path) -> str:
    """One of: 'no-markdown', 'in-sync', 'stale', 'missing', 'hand-edited', 'unmanaged'."""
    md_path = yaml_path.with_suffix(".md")
    if not md_path.exists():
        return "no-markdown"
    if not yaml_path.exists():
        return "missing"
    src, body_hash, body = _read_header(yaml_path.read_text())
    if src is None or body_hash is None:
        return "unmanaged"
    if _sha(body) != body_hash:
        return "hand-edited"
    return "in-sync" if src == _sha(md_path.read_text()) else "stale"


def ensure_synced(yaml_path: Path | str) -> bool:
    """Bring the YAML up to date with its Markdown master. Returns True if rewritten.

    Raises ConfigError when the YAML was edited by hand, or when the Markdown does not
    parse or validate — a run must never start on a codebook nobody can vouch for.
    """
    yaml_path = Path(yaml_path)
    state = status(yaml_path)
    if state in ("no-markdown", "in-sync"):
        return False
    md_path = yaml_path.with_suffix(".md")
    if is_archived(yaml_path):
        raise ConfigError(
            f"{yaml_path} is a frozen archive record and is {state}; archives are never "
            f"regenerated. Restore the file from git instead of editing it."
        )
    if state == "hand-edited":
        raise ConfigError(
            f"{yaml_path} was edited by hand, but it is generated from {md_path.name}. "
            f"Move your change into the Markdown, then run: "
            f"python scripts/codebook_sync.py {md_path} --force"
        )
    if state == "unmanaged":
        raise ConfigError(
            f"{yaml_path} has a Markdown sibling but no generated-file header. Run: "
            f"python scripts/codebook_sync.py {md_path} --force"
        )
    write_yaml(md_path)
    return True


def write_yaml(md_path: Path) -> Path:
    yaml_path = md_path.with_suffix(".yaml")
    if is_archived(yaml_path):
        raise ConfigError(f"{yaml_path} is under the frozen archive; archives are never regenerated.")
    text = render_yaml(md_path)
    tmp = yaml_path.with_suffix(".yaml.tmp")
    tmp.write_text(text)
    tmp.replace(yaml_path)
    return yaml_path
