"""Markdown-master codebooks: parsing, round-trip, and automatic sync on load."""
from pathlib import Path

import pytest
import yaml

from pipeline.codebook import CodeEntry, load_codebook
from pipeline.codebook_md import (
    CodebookMarkdownError,
    ensure_synced,
    entries_to_md,
    md_to_entries,
    status,
    write_yaml,
)
from pipeline.errors import ConfigError

REPO = Path(__file__).resolve().parents[1]

MD = """# A test codebook

Preamble prose that is ignored by the parser.

---

# Section heading, also ignored

## warmth

- dimension: state · side: founder · tier: T2
- type: ordinal · scale: 1–5 · unit: PITCH · observability: interpretive

**Definition.** How warm the founder appears
across the whole pitch.

**Scale**

1. Cold throughout.
2. Mostly cold.
3. Mixed.
4. Mostly warm.
5. Warm throughout.

**Include if.** Smiles, friendly tone.

**Exclude if.** Scripted pleasantries only.

**Provenance** *(never sent to the model)*. SECRET-TOOL v2.

**Examples**

- [illustrative] Founder smiles while greeting the panel.
- [real | Pitch 12] "So glad to be here."

## PR-FRAME

- dimension: probe · side: investor · tier: T2
- type: categorical · unit: EPISODE · observability: observable
- values: PROMOTION, PREVENTION, NEUTRAL

**Definition.** Regulatory focus of the question.

**Include if.** A question is asked.

**Exclude if.** No question.
"""


def _write(tmp_path: Path, text: str = MD) -> Path:
    md = tmp_path / "cb.md"
    md.write_text(text)
    return md


def test_parses_metadata_prose_anchors_examples():
    warmth, frame = md_to_entries(MD)
    assert warmth["id"] == "warmth"
    assert warmth["scale"] == [1, 5]
    assert warmth["tier"] == "T2"
    assert warmth["definition"] == "How warm the founder appears across the whole pitch."
    assert [a["value"] for a in warmth["anchors"]] == [1, 2, 3, 4, 5]
    assert warmth["anchors"][4]["description"] == "Warm throughout."
    assert warmth["provenance"] == "SECRET-TOOL v2."
    assert warmth["examples"][1]["source"] == "Pitch 12"
    assert frame["values"] == ["PROMOTION", "PREVENTION", "NEUTRAL"]
    assert "anchors" not in frame or not frame["anchors"]


def test_round_trip_is_lossless():
    entries = md_to_entries(MD)
    again = md_to_entries(entries_to_md(entries, "# A test codebook\n"))
    assert again == entries


def test_parse_error_names_file_line_and_code():
    bad = MD.replace("- type: ordinal ·", "- typo: ordinal ·")
    with pytest.raises(CodebookMarkdownError) as exc:
        md_to_entries(bad, source="cb.md")
    message = str(exc.value)
    assert "cb.md:" in message and "[warmth]" in message and "typo" in message


def test_markdown_without_codes_is_rejected():
    with pytest.raises(CodebookMarkdownError):
        md_to_entries("# Just a title\n\nNo codes here.\n")


def test_invalid_codebook_is_reported_against_the_markdown(tmp_path):
    md = _write(tmp_path, MD.replace("5. Warm throughout.\n", ""))  # anchor 5 missing
    with pytest.raises(CodebookMarkdownError):
        write_yaml(md)
    assert not md.with_suffix(".yaml").exists()


def test_load_generates_yaml_when_missing(tmp_path):
    md = _write(tmp_path)
    yaml_path = md.with_suffix(".yaml")
    assert status(yaml_path) == "missing"
    codebook = load_codebook(yaml_path)
    assert [e.id for e in codebook.entries] == ["warmth", "PR-FRAME"]
    assert status(yaml_path) == "in-sync"
    assert yaml_path.read_text().startswith("# GENERATED from cb.md")


def test_load_picks_up_markdown_edits_automatically(tmp_path):
    md = _write(tmp_path)
    yaml_path = md.with_suffix(".yaml")
    load_codebook(yaml_path)
    md.write_text(MD.replace("Cold throughout.", "Flat and distant throughout."))
    assert status(yaml_path) == "stale"
    codebook = load_codebook(yaml_path)
    assert codebook.entries[0].anchors[0].description == "Flat and distant throughout."
    assert status(yaml_path) == "in-sync"


def test_in_sync_yaml_is_not_rewritten(tmp_path):
    md = _write(tmp_path)
    yaml_path = write_yaml(md)
    before = yaml_path.stat().st_mtime_ns
    assert ensure_synced(yaml_path) is False
    assert yaml_path.stat().st_mtime_ns == before


def test_hand_edited_yaml_is_refused(tmp_path):
    md = _write(tmp_path)
    yaml_path = write_yaml(md)
    yaml_path.write_text(yaml_path.read_text().replace("Cold throughout.", "Edited in YAML."))
    assert status(yaml_path) == "hand-edited"
    with pytest.raises(ConfigError, match="edited by hand"):
        load_codebook(yaml_path)


def test_yaml_without_header_beside_markdown_is_refused(tmp_path):
    md = _write(tmp_path)
    yaml_path = md.with_suffix(".yaml")
    yaml_path.write_text(yaml.safe_dump(md_to_entries(MD), sort_keys=False))
    assert status(yaml_path) == "unmanaged"
    with pytest.raises(ConfigError, match="--force"):
        load_codebook(yaml_path)


def test_illustrative_examples_are_listed_for_the_freeze_check(tmp_path):
    from pipeline.codebook_md import illustrative_examples

    md = _write(tmp_path)
    assert illustrative_examples(md) == [("warmth", "Founder smiles while greeting the panel.")]
    md.write_text(MD.replace("- [illustrative] Founder smiles while greeting the panel.\n", ""))
    assert illustrative_examples(md) == []


def test_yaml_without_markdown_loads_as_before(tmp_path):
    yaml_path = tmp_path / "plain.yaml"
    yaml_path.write_text(yaml.safe_dump(md_to_entries(MD), sort_keys=False))
    assert status(yaml_path) == "no-markdown"
    assert len(load_codebook(yaml_path).entries) == 2


def _categorical(**overrides):
    base = dict(
        id="X", dimension="probe", side="investor", tier="T2", type="categorical",
        unit="EPISODE", observability="observable", definition="d", include_if="i", exclude_if="e",
        values=["A", "NEUTRAL"],
    )
    base.update(overrides)
    return base


def test_categorical_requires_values_with_a_residual():
    CodeEntry(**_categorical())
    with pytest.raises(ValueError):
        CodeEntry(**_categorical(values=None))
    with pytest.raises(ValueError):
        CodeEntry(**_categorical(values=["A", "B"]))


def test_values_rejected_on_non_categorical():
    with pytest.raises(ValueError):
        CodeEntry(**_categorical(type="binary"))


def _managed_markdown() -> list[Path]:
    found = []
    for root in (REPO / "pipeline" / "codebooks", REPO / "docs" / "codebooks"):
        for md in sorted(root.rglob("*.md")):
            text = md.read_text()
            if "\n## " in text and "**Definition.**" in text:
                found.append(md)
    return found


def test_repo_has_managed_codebooks():
    assert REPO / "pipeline" / "codebooks" / "stage_a_perceived_delivery.md" in _managed_markdown()


@pytest.mark.parametrize("md", _managed_markdown(), ids=lambda p: p.name)
def test_every_shipped_codebook_is_in_sync(md):
    """Fails if a generated YAML was hand-edited or a Markdown edit was committed unsynced."""
    assert status(md.with_suffix(".yaml")) == "in-sync"


def test_archived_codebooks_are_detected_as_frozen():
    from pipeline.codebook_md import is_archived
    archived = [md for md in _managed_markdown() if "archive" in md.parts]
    assert archived and all(is_archived(md) for md in archived)
    assert not is_archived(REPO / "pipeline" / "codebooks" / "stage_a_perceived_delivery.md")


_AS_RUN = sorted((REPO / "docs" / "codebooks" / "archive" / "stage_a").glob("*__*.yaml"))


def test_stage_a_has_an_as_run_record():
    assert _AS_RUN


@pytest.mark.parametrize("record", _AS_RUN, ids=lambda p: p.stem)
def test_as_run_stage_a_codebook_still_renders_its_recorded_prompt(record):
    """The frozen record is evidence of what a run sent; its filename carries that hash."""
    from pipeline.prompts import prompt_hash, render_prompt

    recorded = record.stem.rsplit("__", 1)[1]
    # The ten-item template is archived with the v3-dev8 bundle; the as-run records render through it.
    text = render_prompt("stage_a_holistic.j2", codebook=load_codebook(record),
                         prompt_dir=REPO / "docs" / "codebooks" / "archive" / "stage_b_panel_v3dev8")
    assert prompt_hash(text) == recorded


_PANEL_STYLE = """# test book

## Q-FRAME
- name: Question framing
- dimension: A. Investor question
- side: investor · tier: T1 · type: categorical · unit: EVENT · observability: observable · applies_to: question
- values: promotion, prevention, NEUTRAL

**Definition.** Focus of the question.

**Include if.** Every question.

**Exclude if.** Nothing else.

## D-KIND
- dimension: 0. Types
- side: none · tier: T3 · type: categorical · unit: DECISION · observability: observable
- values: offer, exit, NONE

**Definition.** One decision.

**Include if.** Every decision.

**Exclude if.** Interest without an offer.
"""


def test_name_applies_to_and_event_units_parse_and_validate():
    from pipeline.codebook import Codebook
    from pipeline.codebook_md import md_to_entries

    entries = md_to_entries(_PANEL_STYLE)
    book = Codebook(entries=entries)
    q = book.by_id("Q-FRAME")
    assert q.name == "Question framing" and q.applies_to == "question" and q.unit == "EVENT"
    d = book.by_id("D-KIND")
    assert d.name is None and d.applies_to is None and d.unit == "DECISION"
    assert "name" not in entries[1] and "applies_to" not in entries[1]


def test_name_and_applies_to_round_trip_through_markdown():
    from pipeline.codebook_md import entries_to_md, md_to_entries

    entries = md_to_entries(_PANEL_STYLE)
    assert md_to_entries(entries_to_md(entries, "# test book")) == entries
