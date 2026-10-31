"""The whole instrument laid out for a human coder: every entry of every codebook of the release, in the order the
release lists its books, grouped by each entry's own dimension, with what a coder needs to apply it (definition,
scale anchors, the meaning of each value, include and exclude rules, examples). Notes written for the researcher
(an entry's provenance) are left out. Nothing here names a book, a code or a value: titles come from each book's
Markdown heading and everything else from the entries."""
from __future__ import annotations

import html
from pathlib import Path

from pipeline.codebook import Codebook, CodeEntry
from pipeline.instrument import InstrumentRelease
from pipeline.stage_b.panel_books import PanelBooks
from pipeline.stage_b.review_labels import Labels, definition_html, value_glosses

PROTOCOL_FILE = "stage_a_perceived_delivery_protocol.md"      # in the release's templates dir; heads the Stage A block
PROTOCOL_ID, PROTOCOL_NAME = "protocol", "Shared coding protocol"

# which loaded book a release's book key is (the keys load_panel_books reads)
_BOOK_OF = {"stage_b_panel": "panel", "stage_a": "stage_a", "venture_context": "venture"}


def book_title(markdown: Path | None, fallback: str) -> str:
    """The first `# ` heading of a book's Markdown master, else the fallback."""
    try:
        for line in Path(markdown).read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except (OSError, TypeError):
        pass
    return fallback


def _paragraphs(text: str | None) -> str:
    return "".join(f"<p>{html.escape(p.strip())}</p>" for p in (text or "").split("\n\n") if p.strip())


def entry_model(entry: CodeEntry, labels: Labels, on_sheet: bool, book: Codebook | None = None) -> dict:
    values = tuple(entry.values or [])
    glosses = value_glosses(entry, values, labels, book)
    return {"id": entry.id, "name": entry.name or entry.id.replace("_", " ").capitalize(), "type": entry.type,
            "unit": entry.unit, "on_sheet": on_sheet,
            "definition_html": definition_html(entry),          # escaped; carries the scale anchors when it has them
            "values": [{"value": v, "gloss": glosses.get(v, "")} for v in values],
            "include_html": _paragraphs(entry.include_if), "exclude_html": _paragraphs(entry.exclude_if),
            "examples": [html.escape(e.text) for e in entry.examples],
            # what the page's search box looks in: the name, the id, the definition and the values, lower-cased
            "search": " ".join([entry.name or "", entry.id, entry.definition, *values,
                                *(a.description for a in entry.anchors)]).lower()}


def protocol_paragraphs(release: InstrumentRelease) -> list[str]:
    """The paragraphs of the Stage A shared coding protocol, each on one line, without its heading; [] when the
    release's templates dir has no such file."""
    try:
        text = (Path(release.templates_dir) / PROTOCOL_FILE).read_text(encoding="utf-8")
    except OSError:
        return []
    return [" ".join(p.split()) for p in text.split("\n\n") if p.strip() and not p.lstrip().startswith("# ")]


def protocol_entry(paragraphs: list[str], on_sheet: bool) -> dict:
    """The protocol as a codebook entry: no values, no examples, the paragraphs as its definition."""
    return {"id": PROTOCOL_ID, "name": PROTOCOL_NAME, "type": None, "unit": None, "on_sheet": on_sheet,
            "definition_html": _paragraphs("\n\n".join(paragraphs)), "values": [], "include_html": "",
            "exclude_html": "", "examples": [], "search": " ".join([PROTOCOL_NAME, PROTOCOL_ID, *paragraphs]).lower()}


def codebook_model(release: InstrumentRelease, books: PanelBooks, labels: Labels,
                   on_sheet: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """[{"key", "title", "groups": [{"title", "entries": [entry_model, ...]}]}], one per book of the release.
    `on_sheet` holds the ids of the entries the coder's cards ask about; those entries are marked."""
    out = []
    for key, ref in release.books.items():
        book: Codebook | None = getattr(books, _BOOK_OF.get(key, ""), None)
        if book is None:
            continue
        groups: dict[str, list[dict]] = {}
        for entry in book.entries:
            groups.setdefault(entry.dimension, []).append(entry_model(entry, labels, entry.id in on_sheet, book))
        paragraphs = protocol_paragraphs(release) if key == "stage_a" else []
        if paragraphs:
            if not groups:
                groups[""] = []
            first = next(iter(groups))
            groups[first].insert(0, protocol_entry(paragraphs, any(e.id in on_sheet for e in book.entries)))
        out.append({"key": key, "title": book_title(ref.markdown, key),
                    "groups": [{"title": title, "entries": entries} for title, entries in groups.items()]})
    return out
