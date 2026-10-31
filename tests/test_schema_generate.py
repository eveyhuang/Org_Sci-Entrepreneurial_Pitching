import pytest
from pydantic import ValidationError

from pipeline.codebook import Codebook
from pipeline.errors import ConfigError
from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.generate import build_record_model, schema_fingerprint


def entry(code_id, kind, **extra):
    base = dict(id=code_id, dimension="d", tier="T2", type=kind, unit="EPISODE",
                observability="observable", definition="def", include_if="inc", exclude_if="exc")
    if kind == "ordinal":
        base.update(scale=[1, 5], anchors=[{"value": v, "description": f"level {v}"} for v in range(1, 6)])
    if kind == "categorical":
        base.update(values=["promotion", "prevention", "NEUTRAL"])
    return {**base, **extra}


BOOK = Codebook(entries=[
    entry("PR-FRAME", "categorical"), entry("RS-EVID", "binary"),
    entry("PREP-SUBS", "ordinal"), entry("RAP-INTERRUPT", "count"),
    entry("PITCH-ONLY", "count", unit="PITCH"),
])


def coded(value, **extra):
    return {"value": value, "uncodable": False, "evidence": "t3", "turn_refs": [3], **extra}


def valid():
    return {"PR-FRAME": coded("prevention"), "RS-EVID": coded(True),
            "PREP-SUBS": coded(3), "RAP-INTERRUPT": coded(0)}


def test_one_field_per_code_of_the_unit_keyed_by_exact_id():
    model = build_record_model(BOOK)
    record = model.model_validate(valid())
    dumped = record.model_dump(mode="json", by_alias=True)
    assert list(dumped) == ["PR-FRAME", "RS-EVID", "PREP-SUBS", "RAP-INTERRUPT"]  # PITCH code excluded
    assert dumped["RS-EVID"]["value"] is True and dumped["RAP-INTERRUPT"]["value"] == 0


@pytest.mark.parametrize("code, bad", [
    ("PR-FRAME", "aggressive"), ("PREP-SUBS", 6), ("PREP-SUBS", 0), ("RAP-INTERRUPT", -1),
])
def test_values_outside_the_codebook_are_rejected(code, bad):
    payload = valid()
    payload[code] = coded(bad)
    with pytest.raises(ValidationError):
        build_record_model(BOOK).model_validate(payload)


@pytest.mark.parametrize("rating", [
    {"value": 3, "uncodable": True, "evidence": "e", "turn_refs": []},     # both
    {"value": None, "uncodable": False, "evidence": "e", "turn_refs": []},  # neither
    {"value": 3, "uncodable": False, "evidence": "  ", "turn_refs": []},    # blank evidence
    {"value": 3, "uncodable": False, "evidence": "e"},                      # turn_refs missing
    {"value": 3, "uncodable": False, "evidence": "e", "turn_refs": [], "confidence": "high"},
])
def test_the_rating_invariant(rating):
    payload = valid()
    payload["PREP-SUBS"] = rating
    with pytest.raises(ValidationError):
        build_record_model(BOOK).model_validate(payload)


def test_uncodable_with_null_value_is_valid_and_missing_codes_are_not():
    payload = valid()
    payload["RS-EVID"] = {"value": None, "uncodable": True, "evidence": "off camera", "turn_refs": []}
    assert build_record_model(BOOK).model_validate(payload)
    del payload["PR-FRAME"]
    with pytest.raises(ValidationError):
        build_record_model(BOOK).model_validate(payload)


def test_unsupported_types_collisions_and_empty_units_fail_loudly():
    with pytest.raises(ConfigError, match="span"):
        build_record_model(Codebook(entries=[entry("EP-BOUND", "span")]))
    with pytest.raises(ConfigError, match="collide"):
        build_record_model(Codebook(entries=[entry("A-B", "binary"), entry("A_B", "binary")]))
    with pytest.raises(ConfigError, match="no EPISODE codes"):
        build_record_model(Codebook(entries=[entry("X", "binary", unit="PITCH")]))


def test_wire_schema_is_gemini_safe_and_requires_every_key():
    schema = gemini_response_schema(build_record_model(BOOK))
    text = str(schema)
    for banned in ("additionalProperties", "$ref", "$defs", "anyOf", "pattern"):
        assert banned not in text
    assert schema["required"] == ["PR-FRAME", "RS-EVID", "PREP-SUBS", "RAP-INTERRUPT"]
    frame = schema["properties"]["PR-FRAME"]
    assert sorted(frame["required"]) == ["evidence", "turn_refs", "uncodable", "value"]
    assert frame["properties"]["value"]["enum"] == ["promotion", "prevention", "NEUTRAL"]
    assert frame["properties"]["value"]["nullable"] is True


def test_fingerprint_tracks_the_contract_not_the_wording():
    base = schema_fingerprint(build_record_model(BOOK))
    reworded = Codebook(entries=[entry("PR-FRAME", "categorical", definition="new words"),
                                 *BOOK.entries[1:]])
    assert schema_fingerprint(build_record_model(reworded)) == base
    revalued = Codebook(entries=[entry("PR-FRAME", "categorical", values=["a", "b", "NONE"]),
                                 *BOOK.entries[1:]])
    assert schema_fingerprint(build_record_model(revalued)) != base


import json
import os
from pathlib import Path

from pipeline.codebook import load_codebook

SNAPSHOT = Path(__file__).parent / "snapshots" / "stage_b_episode_schema.json"
WIRE_SNAPSHOT = Path(__file__).parent / "snapshots" / "stage_b_episode_wire_schema.json"


def test_shipped_episode_schema_matches_its_snapshot():
    """A factory bug must not silently change a response contract. After a DELIBERATE
    codebook change: UPDATE_SNAPSHOTS=1 pytest tests/test_schema_generate.py, then review the diff."""
    model = build_record_model(load_codebook(Path(__file__).resolve().parents[1] / "docs/codebooks/archive/stage_b_v1_episode/stage_b_episode.yaml"))
    actual = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOT.parent.mkdir(exist_ok=True)
        SNAPSHOT.write_text(actual)
    assert actual == SNAPSHOT.read_text()


def test_shipped_episode_wire_schema_matches_its_snapshot():
    """The Gemini-safe wire schema (what actually goes on the wire) archived per run —
    a second snapshot of the SAME contract, converted. Same UPDATE_SNAPSHOTS mechanism."""
    model = build_record_model(load_codebook(Path(__file__).resolve().parents[1] / "docs/codebooks/archive/stage_b_v1_episode/stage_b_episode.yaml"))
    actual = json.dumps(gemini_response_schema(model), indent=2, sort_keys=True) + "\n"
    if os.environ.get("UPDATE_SNAPSHOTS"):
        WIRE_SNAPSHOT.parent.mkdir(exist_ok=True)
        WIRE_SNAPSHOT.write_text(actual)
    assert actual == WIRE_SNAPSHOT.read_text()
