import textwrap

import pytest

from pipeline.codebook import Codebook, load_codebook
from pipeline.errors import ConfigError

VALID = """
- id: visual_positive
  dimension: "H. Founder state"
  side: founder
  tier: T1
  type: continuous
  scale: [0.0, 1.0]
  unit: PITCH
  observability: interpretive
  definition: >
    Proportion of the video in which presenters display happiness.
  include_if: "The presenter's face is visible for a substantial share of the video"
  exclude_if: "Spans where no presenter is on screen"
  examples:
    - text: "Whole-video Visual-Positive mean = 0.17 in the accelerator sample"
      provenance: real
      source: "Hu & Ma 2025"
  anchors:
    - value: 0.0
      description: "No positive affect at any point."
    - value: 1.0
      description: "Sustained positive affect throughout."
"""


def test_loads_valid_codebook(tmp_path):
    path = tmp_path / "book.yaml"
    path.write_text(textwrap.dedent(VALID))
    book = load_codebook(path)
    assert book.ids == ("visual_positive",)
    entry = book.by_id("visual_positive")
    assert entry.tier == "T1"
    assert entry.scale == (0.0, 1.0)
    assert entry.examples[0].provenance == "real"


def test_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "book.yaml"
    path.write_text(textwrap.dedent(VALID) + textwrap.dedent(VALID))
    with pytest.raises(ConfigError, match="duplicate"):
        load_codebook(path)


def test_rejects_unknown_tier(tmp_path):
    path = tmp_path / "book.yaml"
    path.write_text(textwrap.dedent(VALID).replace("tier: T1", "tier: T9"))
    with pytest.raises(ConfigError, match="tier"):
        load_codebook(path)


def test_rejects_missing_definition(tmp_path):
    body = textwrap.dedent(VALID).replace(
        "  definition: >\n    Proportion of the video in which presenters display happiness.\n", ""
    )
    path = tmp_path / "book.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match="definition"):
        load_codebook(path)


def test_ordinal_and_continuous_require_scale(tmp_path):
    body = textwrap.dedent(VALID).replace("  scale: [0.0, 1.0]\n", "")
    path = tmp_path / "book.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match="scale"):
        load_codebook(path)


def test_codebook_model_rejects_duplicate_ids_on_direct_construction():
    import yaml as _yaml
    from pydantic import ValidationError

    from pipeline.codebook import Codebook

    entries = _yaml.safe_load(textwrap.dedent(VALID)) * 2
    with pytest.raises(ValidationError, match="duplicate"):
        Codebook(entries=entries)


def test_rejects_malformed_yaml(tmp_path):
    path = tmp_path / "book.yaml"
    path.write_text("- id: visual_positive\n  definition: \"unterminated\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_codebook(path)


def test_stage_a_codebook_matches_schema_fields():
    from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES

    book = load_codebook("pipeline/codebooks/stage_a_perceived_delivery.yaml")
    assert book.ids == STAGE_A_DELIVERY_CODES
    assert all(e.tier == "T2" for e in book.entries)
    assert all(e.unit == "PITCH" for e in book.entries)
    assert all(e.type == "ordinal" for e in book.entries)
    assert all(e.scale == (1.0, 5.0) for e in book.entries)
    assert all(len(e.anchors) == 5 for e in book.entries)


def test_ordinal_entry_without_anchors_is_rejected(tmp_path):
    body = textwrap.dedent(VALID).replace("type: continuous", "type: ordinal").replace(
        "\n  anchors:\n    - value: 0.0\n      description: \"No positive affect at any point.\"\n    - value: 1.0\n      description: \"Sustained positive affect throughout.\"\n",
        "\n",
    )
    path = tmp_path / "book.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match="visual_positive"):
        load_codebook(path)


def test_anchor_value_outside_scale_is_rejected(tmp_path):
    body = textwrap.dedent(VALID).replace(
        "    - value: 1.0\n      description: \"Sustained positive affect throughout.\"",
        "    - value: 2.0\n      description: \"Out of range.\"",
    )
    path = tmp_path / "book.yaml"
    path.write_text(body)
    with pytest.raises(ConfigError, match="visual_positive"):
        load_codebook(path)


def _number_entry(**extra):
    from pipeline.codebook import CodeEntry
    return CodeEntry(id="VC-ASK", dimension="V. Venture context", side="founder", tier="T3", type="number",
                     unit="PITCH", observability="observable", definition="The ask.", include_if="The first ask.",
                     exclude_if="Not counter-offers.", **extra)


def test_a_number_entry_needs_no_scale_anchors_or_values():
    from pydantic import ValidationError
    entry = _number_entry()
    assert entry.type == "number" and entry.scale is None and entry.anchors == [] and entry.values is None
    with pytest.raises(ValidationError, match="only allowed on categorical"):
        _number_entry(values=["a", "NONE"])


def test_the_venture_book_records_the_ask_as_a_number_entry():
    book = load_codebook("pipeline/codebooks/venture_context.yaml")
    assert book.ids[-1] == "VC-ASK"
    entry = book.by_id("VC-ASK")
    assert (entry.type, entry.unit, entry.side, entry.tier, entry.observability) == (
        "number", "PITCH", "founder", "T3", "observable")
    assert entry.name == "The ask" and entry.values is None and entry.scale is None
    assert "ask_amount_usd" in entry.definition and "ask_equity_pct" in entry.definition
