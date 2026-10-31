"""Every Stage B template that renders codebook entries must keep provenance out.

Exercised against the frozen v1 bundle; the templates and books live together in its archive."""
from pathlib import Path
import pytest

from pipeline.codebook import load_codebook
from pipeline.prompts import render_prompt

ARCHIVE = Path(__file__).resolve().parents[1] / "docs/codebooks/archive/stage_b_v1_episode"
CASES = [
    ("stage_b_segmentation.j2", str(ARCHIVE / "stage_b_segmentation.yaml")),
    ("stage_b_episode.j2", str(ARCHIVE / "stage_b_episode.yaml")),
]
LEAK_MARKERS = ("Schegloff", "Krippendorff", "Kanze", "ORIGINAL TO THIS PROJECT", "et al", "(19", "(20")


@pytest.mark.parametrize("template, book_path", CASES)
def test_provenance_never_reaches_a_stage_b_prompt(template, book_path):
    book = load_codebook(book_path)
    text = render_prompt(template, codebook=book, prompt_dir=ARCHIVE)
    for entry in book.entries:
        assert entry.id in text and entry.definition[:40] in text
        assert entry.provenance, f"{entry.id} has no provenance to check against"
        assert entry.provenance[:60] not in text
    for marker in LEAK_MARKERS:
        assert marker not in text, f"{marker!r} leaked into {template}"


def test_segmentation_prompt_mentions_decisions_and_deal():
    book = load_codebook(CASES[0][1])
    text = render_prompt("stage_b_segmentation.j2", codebook=book, prompt_dir=ARCHIVE)
    assert "`decisions`" in text and "`deal`" in text
    assert "no_decision_shown" in text and "joins_offer" in text


def test_segmentation_prompt_renders_without_context_and_with_it():
    book = load_codebook(CASES[0][1])
    bare = render_prompt("stage_b_segmentation.j2", codebook=book, prompt_dir=ARCHIVE)
    assert "SPEAKER ROSTER" not in bare
    full = render_prompt("stage_b_segmentation.j2", codebook=book,
                         context={"roster": "speaker-1 — founder — x", "timeline": "[t0 …] hi"}, prompt_dir=ARCHIVE)
    assert full.startswith(bare.rstrip()[:200]) and "SPEAKER ROSTER" in full and "[t0 …] hi" in full


def test_episode_prompt_renders_each_code_by_its_type():
    book = load_codebook(CASES[1][1])
    text = render_prompt("stage_b_episode.j2", codebook=book, prompt_dir=ARCHIVE)
    assert "promotion | prevention | NEUTRAL" in text             # categorical: its values
    assert "Answer true" in text and "whole number" in text        # binary, count
    for anchor in book.by_id("PREP-SUBS").anchors:                 # ordinal: every anchor
        assert f"{int(anchor.value)} — {anchor.description}" in text
    # PREP-SUBS's verbatim exclude_if legitimately says "...fluency, confidence or
    # charisma" (delivery, not a schema field) — check the OUTPUT contract doesn't
    # declare a `confidence` field the way it declares `value`/`uncodable`/etc.
    assert "`confidence`" not in text
    full = render_prompt("stage_b_episode.j2", codebook=book, context={
        "roster": "R", "timeline_slice": "[t5 0:00–0:02] speaker-3 (investor): challenges the valuation",
        "pitch_summary": "A kettle."}, prompt_dir=ARCHIVE)
    assert "[t5 0:00–0:02]" in full and "A kettle." in full


from pipeline.codebook import Codebook

TEMPLATES = ["stage_b_segmentation.j2", "stage_b_episode.j2"]


@pytest.mark.parametrize("template", TEMPLATES)
def test_only_real_examples_reach_a_prompt_and_never_their_source(template):
    unit = "PITCH" if template != "stage_b_episode.j2" else "EPISODE"
    book = Codebook(entries=[dict(
        id="X-CODE", dimension="d", tier="T2", type="binary", unit=unit, observability="observable",
        definition="def", include_if="inc", exclude_if="exc",
        examples=[{"text": "REAL-EXCERPT", "provenance": "real", "source": "st_secret t4"},
                  {"text": "MADE-UP-EXCERPT", "provenance": "illustrative"}])])
    text = render_prompt(template, codebook=book, prompt_dir=ARCHIVE)
    assert "REAL-EXCERPT" in text
    assert "MADE-UP-EXCERPT" not in text and "st_secret" not in text


def test_a_code_without_real_examples_renders_no_examples_heading():
    book = load_codebook(CASES[1][1])
    # The shipped codebook has no real examples yet (Task 18 supplies them later).
    assert not any(e.provenance == "real" for entry in book.entries for e in entry.examples)
    assert "Examples from real pitches" not in render_prompt("stage_b_episode.j2", codebook=book, prompt_dir=ARCHIVE)


def test_a_code_with_a_real_example_renders_the_examples_heading():
    book = Codebook(entries=[dict(
        id="X-CODE", dimension="d", tier="T2", type="binary", unit="EPISODE",
        observability="observable", definition="def", include_if="inc", exclude_if="exc",
        examples=[{"text": "REAL-EXCERPT", "provenance": "real", "source": "st_secret t4"}],
    )])
    assert "Examples from real pitches" in render_prompt("stage_b_episode.j2", codebook=book, prompt_dir=ARCHIVE)
