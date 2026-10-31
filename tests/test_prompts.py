from pathlib import Path

import pytest
from jinja2 import UndefinedError

from pipeline.codebook import load_codebook
from pipeline.prompts import prompt_hash, render_prompt

BOOK = "pipeline/codebooks/stage_a_perceived_delivery.yaml"
TEMPLATE = "stage_a_perceived_delivery.j2"
V3DEV8 = Path(__file__).resolve().parents[1] / "docs" / "codebooks" / "archive" / "stage_b_panel_v3dev8"


def test_renders_every_construct_with_definition_and_criteria():
    book = load_codebook(BOOK)
    text = render_prompt(TEMPLATE, codebook=book)
    for entry in book.entries:
        assert entry.id in text
        assert entry.definition.split("\n")[0][:40] in text
        assert entry.include_if[:30] in text
        assert entry.exclude_if[:30] in text


def test_provenance_never_reaches_the_rendered_prompt():
    book = load_codebook(BOOK)
    text = render_prompt(TEMPLATE, codebook=book)
    for entry in book.entries:
        assert entry.provenance, f"{entry.id} has no provenance to check"
        # No sentence of the provenance note, and no source-tool name, may appear.
        assert entry.provenance.strip()[:60] not in text
    for tool in ("Face++", "pyAudioAnalysis", "Loughran", "Nicolas", "LSTM", "Berlin"):
        assert tool not in text, f"{tool} leaked into the prompt"


def test_anchors_are_rendered_for_every_construct():
    book = load_codebook(BOOK)
    text = render_prompt(TEMPLATE, codebook=book)
    for entry in book.entries:
        assert len(entry.anchors) == 5
        for anchor in entry.anchors:
            assert anchor.description.strip()[:40] in text
            assert str(int(anchor.value)) in text


def test_prompt_states_the_non_negotiable_rules():
    text = render_prompt(TEMPLATE, codebook=load_codebook(BOOK))
    lowered = text.lower()
    assert "evidence" in lowered
    assert "uncodable" in lowered
    assert "do not infer" in lowered or "unexpressed" in lowered


def test_hash_is_stable_and_content_sensitive():
    book = load_codebook(BOOK)
    a = render_prompt(TEMPLATE, codebook=book)
    b = render_prompt(TEMPLATE, codebook=book)
    assert prompt_hash(a) == prompt_hash(b)
    assert len(prompt_hash(a)) == 16
    assert prompt_hash(a) != prompt_hash(a + " ")


def test_codebook_edit_changes_the_hash(tmp_path):
    book = load_codebook(BOOK)
    before = prompt_hash(render_prompt(TEMPLATE, codebook=book))
    book.entries[0].definition = "Something else entirely."
    after = prompt_hash(render_prompt(TEMPLATE, codebook=book))
    assert before != after


def test_undefined_template_variable_raises(tmp_path):
    (tmp_path / "bad.j2").write_text("Hello {{ nonexistent_variable }}")
    with pytest.raises(UndefinedError):
        render_prompt("bad.j2", codebook=load_codebook(BOOK), prompt_dir=tmp_path)


def test_per_item_context_changes_the_text_but_not_the_hash(tmp_path):
    from pipeline.config import load_config
    from pipeline.prompts import render_pass_prompt
    from tests.test_config import BASE, write_config

    prompts = tmp_path / "prompts"
    config_path = write_config(tmp_path, BASE)
    prompts.mkdir(exist_ok=True)
    (prompts / "holistic.j2").write_text(
        "INSTRUCTIONS\n{% if header is defined %}HEADER: {{ header }}\n{% endif %}"
    )
    cfg = load_config(config_path, prompt_dir=prompts)
    pass_cfg = cfg.passes[0]

    plain_text, plain_hash = render_pass_prompt(cfg, pass_cfg, load_codebook(BOOK))
    ctx_text, ctx_hash = render_pass_prompt(
        cfg, pass_cfg, load_codebook(BOOK), context={"header": "episode 3"}
    )

    assert "HEADER: episode 3" in ctx_text
    assert "HEADER" not in plain_text
    assert ctx_hash == plain_hash  # per-item context must not change unit identity


def test_entry_without_a_scale_renders(tmp_path):
    """scale is legal-null for categorical/binary codes; the template must not crash."""
    book = load_codebook(BOOK)
    entry = book.entries[0]
    entry.type = "categorical"
    entry.scale = None
    text = render_prompt(TEMPLATE, codebook=book)
    assert entry.id in text
    assert "scale None" not in text


def test_archived_ten_item_book_still_renders_its_as_run_prompt_hash():
    """The 2026-10-06 restore edited only the preamble; the instrument the model saw is unchanged in the archive."""
    book = load_codebook(V3DEV8 / "stage_a_pitch_factor.yaml", sync=False)
    text = render_prompt("stage_a_holistic.j2", codebook=book, prompt_dir=V3DEV8)
    assert prompt_hash(text) == "6a7750e44d39b335"


def test_template_loader_searches_prompt_dir_only(tmp_path):
    """A template cannot include files from docs/codebooks (archive or draft content)."""
    import jinja2

    from pipeline.codebook import Codebook
    from pipeline.config import DOCS_CODEBOOKS
    from pipeline.prompts import render_prompt

    assert (DOCS_CODEBOOKS / "REFERENCES.md").exists()
    (tmp_path / "t.j2").write_text("{% include 'REFERENCES.md' %}")
    with pytest.raises(jinja2.TemplateNotFound):
        render_prompt("t.j2", codebook=Codebook(entries=[]), prompt_dir=tmp_path)
