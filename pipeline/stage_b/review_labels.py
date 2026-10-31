# pipeline/stage_b/review_labels.py
"""Wording a coder sees, built from the codebooks so that another instrument gets a usable page unchanged.

Every function takes the codebook entry behind a path and the record. No code id or value is named here: the
structural ids live in panel_books.STRUCTURE_CODES, the rest come from the path or the books. Only the path grammar
of review_items and the five unit kinds are literal. A path whose entry the books lack gets no entry (None), and the
question then reads the path and its unit.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

import yaml

from pipeline.codebook import CodeEntry, Codebook
from pipeline.errors import ConfigError
from pipeline.instrument import InstrumentRelease
from pipeline.schemas.stage_b_panel import MISSING_REASONS
from pipeline.stage_b.panel_books import PanelBooks, load_panel_books
from pipeline.stage_b.review_items import (ASK as VENTURE_ASK, MISSING_PREFIX, VENTURE_CODES, choices_of, is_cut,
                                           is_outcome, parse_path)

DEFAULT_STRUCTURAL = {
    "turn": "the video at these times is not what this question assumes (wrong speaker, another kind of turn, or cut mid-way)",
    "exchange": "these turns are not one line of questioning",
    "pitch": "the prepared presentation is not where the page says it is",
    "pair": "one of the two videos is missing, or its window is not the prepared presentation",
    "actor": "the list of people is wrong (not a separate person, or someone is missing)",
}
DEFAULT_EXCHANGE_UNIT = "Line of questioning"      # overridable as labels.sections["exchange_unit"]
PITCH_LEVEL_UNIT = "the whole video"               # the {unit} of the pitch's outcome and first decision
# `first_decision` heads the outcomes section when it holds only the first-decision card (no outcome card since
# 2026-10-21)
DEFAULT_SECTIONS = {"pitch": "Prepared presentation", "conversation": "Lines of questioning", "outcomes": "Outcomes",
                    "first_decision": "The first decision",
                    "clips": "Prepared presentation (clip session)", "pairs": "Which shows more?",
                    # the task page (review_panel.py session-export, 2026-10-21): a video task's two sections and a
                    # comparison task's block
                    "prepared": "The prepared presentation", "qa": "The questions and answers",
                    "compare": "Compare the two prepared presentations"}
DEFAULT_HINTS = {
    "pitch_intro": "Watch this part first, in full, then answer the items below. Stop where the questioning begins.",
    "pitch_card": "Judge only the part shown by the button; hover a number to read what it means.",
    "outcome_card": "Watch to the end; the panel's decisions may come late in the video.",
    "prep_pres_card": "The content of the prepared pitch only; what the investors later ask is not evidence.",
    # the line under the first-decision card's question (the labels word it for the corpus)
    "cut_card": ("The first time anyone on the panel states a decision (an exit or an offer), even if others keep "
                 "asking questions afterwards. Press ‘mark now’ at its first words."),
    "pairs_intro": ("You have rated all ten prepared presentations. For each pair, play each video's prepared "
                    "presentation from its ▶ button (between the times shown; the player pauses where it ends): "
                    "which presentation shows more? Choose even when close, and tick ‘close’."),
    # the prepared section's intro on a full-video sheet that rates nothing of the prepared presentation, only locates it
    "clip_session_intro": ("Delivery and the content of the prepared presentation are coded on the presentation cards "
                           "of your task page, which play this same full video at its prepared window; not on this "
                           "sheet. If this sheet asks where the prepared presentation is, tick the box on that card "
                           "when the ▶ button does not cover it."),
    # the coders' version of the Stage A protocol's first paragraph (pitch_intro, which stays the model's verbatim),
    # shown above each video of the clip session: the page plays the full video at its prepared window
    "protocol_for_coders": ("Play the prepared presentation with sound, from the ▶ button to the end of the times "
                            "shown on it, before assigning ratings; this page plays the full video, and the player "
                            "pauses once where the prepared presentation ends. Rate only the prepared presentation, "
                            "between the times shown, not the investors' questions that follow or anything after it. "
                            "Human coders and Gemini rate the prepared presentation with the same substantive "
                            "instructions, without investment outcomes or the other coders' ratings."),
    # above a comparison task of the task page: the two videos side by side, each from its ▶ window button
    "compare_intro": ("Play each video's prepared presentation from its ▶ button (between the times shown; the player "
                      "pauses where it ends), then answer each row: left or right. Choose even when close, and tick "
                      "‘close’. Tick “can't judge” only when the two cannot be compared at all, and say why in the "
                      "note."),
}
PAIR_VALUES = ("left", "right")      # a pair judgement: which of the two prepared presentations shows more
DEFAULT_PAIR_GLOSSES = {"left": "the left video's prepared presentation shows more",
                        "right": "the right video's prepared presentation shows more"}
_GLOSS = re.compile(r"\b([A-Z][A-Z_]{2,}): (.*?)(?=\.\s+[A-Z][A-Z_]{2,}:|\.?$)", re.S)
_CONCEPT = re.compile(r"(?:(?<=^)|(?<=[.:)] ))([A-Z][A-Z_]{3,}) — ")


@dataclass(frozen=True)
class Labels:
    questions: dict[str, str] = field(default_factory=dict)
    structural: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)
    glosses: dict[str, dict[str, str]] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    missing_reasons: dict[str, str] = field(default_factory=dict)      # missing-reason id -> what a coder reads
    # code id -> {"gloss": the line under the card's question, "values": {value: {"label", "text"}}}: the coders'
    # own wording of a code, shown on its card in place of the codebook's scale text (the codebook view keeps the
    # codebook's); `label` is what a value pill reads (the sheet still takes the value), `text` what the value means
    anchors: dict[str, dict] = field(default_factory=dict)


SECTIONS = ("questions", "structural", "sections", "glosses", "hints", "missing_reasons", "anchors")
ANCHOR_KEYS, ANCHOR_VALUE_KEYS = ("gloss", "values"), ("label", "text")
PLACEHOLDERS = ("who", "turn", "unit", "dimension")


def _filled(template: str, parts: dict[str, str], key: str, source: object = "review labels") -> str:
    """A question template with its placeholders filled; ConfigError for a placeholder it cannot fill."""
    try:
        return str(template).format(**parts)
    except (KeyError, IndexError, ValueError) as exc:
        raise ConfigError(f"{source}: the question for {key!r} has a placeholder that cannot be filled "
                          f"({exc}); the placeholders allowed are "
                          + ", ".join("{" + p + "}" for p in PLACEHOLDERS)) from None


def load_labels(release: InstrumentRelease) -> Labels:
    """The release's review_labels.yaml when the manifest lists one; empty labels otherwise. A file that is not
    a mapping of the known sections, each a mapping, is refused, and so is a question with an unknown placeholder."""
    path = release.review_labels
    if path is None or not path.exists():
        return Labels()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return Labels()
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: the file must be a mapping of {', '.join(SECTIONS)}, not a {type(data).__name__}")
    unknown = [str(k) for k in data if k not in SECTIONS]
    if unknown:
        raise ConfigError(f"{path}: unknown key {', '.join(repr(k) for k in unknown)}; the keys are {', '.join(SECTIONS)}")
    for key, section in data.items():
        if section is not None and not isinstance(section, dict):
            raise ConfigError(f"{path}: `{key}` must be a mapping, not a {type(section).__name__}")
    for key, template in (data.get("questions") or {}).items():
        _filled(template, dict.fromkeys(PLACEHOLDERS, ""), str(key), path)
    anchors = _anchors(data.get("anchors") or {}, release, path)
    return Labels(**{k: dict(data.get(k) or {}) for k in SECTIONS if k != "anchors"}, anchors=anchors)


def _anchors(section: dict, release: InstrumentRelease, source: object) -> dict[str, dict]:
    """The `anchors` section checked against the release's books and put in one shape: {code id: {"gloss": str,
    "values": {value: {"label": str | None, "text": str | None}}}}. A code the books lack, a value the code does not
    take, an unknown key, or wording that is not text is refused. A value's wording is its text, or a mapping of
    `label` (what its pill reads) and `text`."""
    if not section:
        return {}
    books = load_panel_books(release, sync=False)
    out: dict[str, dict] = {}
    for code_id, spec in section.items():
        code_id = str(code_id)
        entry = next((e for b in (books.panel, books.stage_a, books.venture) for e in b.entries if e.id == code_id),
                     None)
        if entry is None:
            raise ConfigError(f"{source}: `anchors` names {code_id!r}, which is not a code of the instrument's books")
        if not isinstance(spec, dict) or any(k not in ANCHOR_KEYS for k in spec):
            raise ConfigError(f"{source}: `anchors` {code_id!r} must be a mapping of "
                              f"{' and '.join(ANCHOR_KEYS)}")
        gloss = spec.get("gloss")
        if gloss is not None and not isinstance(gloss, str):
            raise ConfigError(f"{source}: `anchors` {code_id!r} gloss must be text")
        given = spec.get("values") or {}
        if not isinstance(given, dict):
            raise ConfigError(f"{source}: `anchors` {code_id!r} values must be a mapping of value to wording")
        allowed = choices_of(entry) or ()
        values: dict[str, dict] = {}
        for value, wording in given.items():
            value = str(value).lower() if isinstance(value, bool) else str(value)
            if value not in allowed:
                raise ConfigError(f"{source}: `anchors` {code_id!r} has no value {value!r}; its values are "
                                  + (", ".join(allowed) or "none"))
            if isinstance(wording, str):
                wording = {"text": wording}
            if (not isinstance(wording, dict) or not wording or any(k not in ANCHOR_VALUE_KEYS for k in wording)
                    or not all(isinstance(v, str) for v in wording.values())):
                raise ConfigError(f"{source}: `anchors` {code_id!r} value {value!r} must be text or a mapping of "
                                  f"{' and '.join(ANCHOR_VALUE_KEYS)} (text)")
            values[value] = {"label": wording.get("label"), "text": wording.get("text")}
        out[code_id] = {"gloss": gloss or "", "values": values}
    return out


def _coder_wording(entry: CodeEntry | None, labels: Labels | None) -> dict:
    return (labels.anchors.get(entry.id) or {}) if entry is not None and labels is not None else {}


def coder_anchors(entry: CodeEntry | None, labels: Labels | None) -> dict[str, str]:
    """{value: what it means} in the coders' own wording (the labels' `anchors`); empty for a code it does not word."""
    return {v: w["text"] for v, w in (_coder_wording(entry, labels).get("values") or {}).items() if w.get("text")}


def value_labels(entry: CodeEntry | None, labels: Labels | None) -> dict[str, str]:
    """{value: what its pill reads} where the labels' `anchors` give one (`true` shown as `yes`, say)."""
    return {v: w["label"] for v, w in (_coder_wording(entry, labels).get("values") or {}).items() if w.get("label")}


def question_gloss(entry: CodeEntry | None, labels: Labels | None) -> str:
    """The line under a card's question, from the labels' `anchors`; empty for a code it does not word."""
    return _coder_wording(entry, labels).get("gloss") or ""


def _pitch_level(path: str) -> bool:
    """The outcome or the first decision of the pitch: judged over the whole encounter."""
    return is_outcome(path) or is_cut(path)


def unit_kind(path: str) -> str:
    if _pitch_level(path):
        return "actor"
    ref = parse_path(path)
    if ref.unit in ("turn", "exchange", "pair"):
        return ref.unit
    return "pitch"


def section_of(path: str) -> str:
    kind = unit_kind(path)
    return {"actor": "outcomes", "pitch": "pitch", "pair": "pairs"}.get(kind, "conversation")


def _find(book: Codebook, code_id: object) -> CodeEntry | None:
    return next((e for e in book.entries if e.id == code_id), None)


def entry_for(path: str, books: PanelBooks) -> CodeEntry | None:
    """The codebook entry behind a path; None when the path has none or the instrument's books lack it."""
    if is_outcome(path):
        return None          # a yes/no/unknown question about the pitch: the labels' `outcome` glosses define it
    if is_cut(path):
        return books.structure_entry("decision")
    ref = parse_path(path)
    if ref.unit == "decision" and ref.field != "kind":
        return None          # PathRef(unit, unit_id, field, sub): sub is a turn's code id or a pair's Stage A item
    if ref.unit == "stage_a":
        return _find(books.stage_a, ref.unit_id)
    if ref.unit == "pair":
        return _find(books.stage_a, ref.sub)
    if ref.unit == "prep_pres":
        return _find(books.panel, ref.unit_id)
    if ref.unit == "venture":
        return _find(books.venture, VENTURE_CODES.get(ref.unit_id))
    if ref.unit == "turn" and ref.field == "code":
        return _find(books.panel, ref.sub)
    return books.structure_entry(ref.field_class)


def _who(record: dict, actor_id: str | None) -> str:
    for a in record["actors"]:
        if a["actor_id"] == actor_id:
            return a["label"]
    return actor_id or ""


def _unit_text(path: str, record: dict, labels: Labels | None = None) -> dict[str, str]:
    """{who, turn, unit} for the placeholders of a question. The words for a line of questioning and for the
    prepared part are the page's own (the labels' sections, else the defaults), in lower case."""
    sections = labels.sections if labels else {}
    if _pitch_level(path):
        return {"who": "", "turn": "", "unit": PITCH_LEVEL_UNIT, "dimension": ""}
    ref = parse_path(path)
    if ref.unit == "turn":
        turn = next((t for t in record["turns"] if t["turn_id"] == ref.unit_id), None)
        who = _who(record, turn["speaker"]) if turn else ""
        return {"who": who, "turn": str(ref.unit_id), "dimension": "",
                "unit": f"turn {ref.unit_id} ({who})" if who else f"turn {ref.unit_id}"}
    if ref.unit == "exchange":
        exchange = (sections.get("exchange_unit") or DEFAULT_EXCHANGE_UNIT).lower()
        return {"who": "", "turn": "", "unit": f"{exchange} {ref.unit_id}", "dimension": ""}
    if ref.unit == "pair":
        return {"who": "", "turn": "", "unit": "this pair", "dimension": (ref.sub or "").replace("_", " ")}
    return {"who": "", "turn": "", "dimension": "",
            "unit": f"the {(sections.get('pitch') or DEFAULT_SECTIONS['pitch']).lower()}"}


def question_text(path: str, record: dict, books: PanelBooks, labels: Labels) -> str:
    entry = entry_for(path, books)
    ref = parse_path(path)
    keys = ["outcome"] if is_outcome(path) else [path, entry.id if entry else "", ref.field_class]
    if ref.unit == "pair":          # a pair's own item first (`pair:<item>`), never the item's Stage A question
        keys = [path, f"{ref.unit}:{ref.sub}", ref.field_class]
    key = next((k for k in keys if k in labels.questions), None)
    parts = _unit_text(path, record, labels)
    if entry is not None and parse_path(path).unit == "pair":
        parts["dimension"] = (entry.name or entry.id.replace("_", " ")).lower()
    if key is not None and labels.questions[key]:
        return _filled(labels.questions[key], parts, key)
    name = "Outcome" if is_outcome(path) else (entry.name or entry.id.replace("_", " ").capitalize()) if entry else path
    return f"{name}: {parts['unit']}"


def structural_text(path: str, labels: Labels) -> str:
    kind = unit_kind(path)
    return labels.structural.get(kind) or DEFAULT_STRUCTURAL[kind]


def missing_options(path: str, labels: Labels) -> list[dict[str, str]]:
    """The "cannot be rated" answers a card offers: `missing:<reason>` for each reason the import accepts at the
    path, with the words a coder reads (the reason id with its underscores as blanks, unless the labels give
    others). The import takes a missing reason for a turn's code, a prepared presentation code and the ask, and
    for nothing else: a Stage A item that cannot be rated is an empty value, which a page's card cannot write, so
    it offers none."""
    ref = parse_path(path)
    if not ((ref.unit == "turn" and ref.field == "code") or ref.unit == "prep_pres"
            or (ref.unit == "venture" and ref.unit_id == VENTURE_ASK)):
        return []
    return [{"value": f"{MISSING_PREFIX}{reason}", "label": labels.missing_reasons.get(reason) or reason.replace("_", " ")}
            for reason in MISSING_REASONS if reason != "none"]


def scale_anchors(entry: CodeEntry | None) -> dict[str, str]:
    if entry is None or not entry.anchors:
        return {}
    return {str(int(a.value)) if float(a.value).is_integer() else str(a.value): a.description for a in entry.anchors}


def _concept_runs(definition: str) -> dict[str, str]:
    """The `NAME — gloss` runs of a concept list's definition, by lower-cased name. A section heading that ends a
    run ("Prevention concepts (concern with losses):") is not part of its gloss."""
    parts = _CONCEPT.split(definition)
    out = {}
    for name, rest in zip(parts[1::2], parts[2::2]):
        rest = re.sub(r"\s*[A-Z][^.:]*:\s*$", "", rest).strip().rstrip(".").strip()
        if rest:
            out[name.lower()] = rest
    return out


def _parsed_glosses(entry: CodeEntry) -> dict[str, str]:
    found = {m.group(1).lower(): m.group(2).strip() for m in _GLOSS.finditer(entry.definition)}
    return {**_concept_runs(entry.definition), **found}


def value_glosses(entry: CodeEntry | None, values: tuple[str, ...], labels: Labels,
                  book: Codebook | None = None) -> dict[str, str]:
    """A short meaning per value: the release's glosses, else the `VALUE: gloss` and `VALUE — gloss` runs of the
    definition, else (given the book) those of another entry of the book with the same values, since two codes
    that share a vocabulary describe it once."""
    if entry is None:
        return {}
    given = labels.glosses.get(entry.id) or {}
    found = _parsed_glosses(entry)
    if book is not None and entry.values:
        for other in book.entries:
            if other.id != entry.id and other.values and set(other.values) == set(entry.values):
                for name, gloss in _parsed_glosses(other).items():
                    found.setdefault(name, gloss)
    return {v: given.get(v) or found.get(v.lower(), "") for v in values if given.get(v) or found.get(v.lower())}


def path_glosses(path: str, entry: CodeEntry | None, values: tuple[str, ...], labels: Labels,
                 book: Codebook | None = None) -> dict[str, str]:
    """The meaning of each value a card offers: the labels' `outcome` glosses for the pitch's outcome (it has no
    codebook entry), the labels' `pair` glosses (else the defaults) for left and right of a pair, else the entry's
    value glosses."""
    if is_outcome(path):
        given = labels.glosses.get("outcome") or {}
        return {v: given[v] for v in values if given.get(v)}
    if parse_path(path).unit == "pair":
        given = labels.glosses.get("pair") or {}
        return {v: given.get(v) or DEFAULT_PAIR_GLOSSES[v] for v in values if given.get(v) or v in DEFAULT_PAIR_GLOSSES}
    return value_glosses(entry, values, labels, book)


def book_holding(books: PanelBooks, entry: CodeEntry | None) -> Codebook | None:
    """The book an entry belongs to; None for no entry."""
    if entry is None:
        return None
    return next((b for b in (books.panel, books.stage_a, books.venture) if entry in b.entries), None)


def definition_html(entry: CodeEntry | None) -> str:
    if entry is None:
        return ""
    text = html.escape(entry.definition)
    parts = _CONCEPT.split(text)       # [lead, NAME, rest, NAME, rest, ...]
    if len(parts) < 3:
        out = [f"<p>{text}</p>"]
    else:
        out = [f"<p>{parts[0].strip()}</p>"] if parts[0].strip() else []
        for name, rest in zip(parts[1::2], parts[2::2]):
            out.append(f"<p><b>{name.lower().replace('_', ' ')}</b> — {rest.strip()}</p>")
    out += [f"<p><b>{html.escape(k)}</b> — {html.escape(v)}</p>" for k, v in scale_anchors(entry).items()]
    return "".join(out)
