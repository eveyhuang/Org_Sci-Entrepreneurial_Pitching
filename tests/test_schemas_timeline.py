import copy

import pytest
from pydantic import ValidationError

from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.stage_b_timeline import (
    TimelineRecord,
    clock_to_seconds,
    normalised_turns,
    roles,
    seconds_to_clock,
)
from tests.stage_b_helpers import timeline_payload


def broken(mutate):
    payload = copy.deepcopy(timeline_payload())
    mutate(payload)
    return payload


def test_valid_payload_parses():
    record = TimelineRecord.model_validate(timeline_payload())
    assert len(record.turns) == 8
    assert len(record.roster) == 3
    assert record.video_length == "0:12"


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["turns"][3].update(turn_idx=9), "contiguous"),
    (lambda p: p["turns"][3].update(speaker="speaker-9"), "roster"),
    (lambda p: p["turns"][3].update(speaker="Mark Cuban"), "speaker-"),
    (lambda p: p["turns"][3].update(end="0:04"), "end is before start"),
    (lambda p: p["turns"][3].update(start="4.07"), "minutes:seconds"),
    (lambda p: p["turns"][3].update(start="1:75"), "minutes:seconds"),
    (lambda p: p["roster"].append(dict(p["roster"][0])), "twice"),
    (lambda p: p["turns"][0].update(note="x"), "Extra inputs"),
])
def test_timeline_rejects(mutate, message):
    with pytest.raises(ValidationError, match=message):
        TimelineRecord.model_validate(broken(mutate))


def test_normalised_turns_returns_float_seconds_and_tag():
    turns = normalised_turns(timeline_payload())
    assert turns[2] == {
        "turn_idx": 2, "speaker": "speaker-2", "start_s": 4.0, "end_s": 5.0,
        "tag": "asks about margins",
    }
    assert all(isinstance(t["start_s"], float) and isinstance(t["end_s"], float) for t in turns)


def test_roles_maps_speaker_to_role():
    assert roles(timeline_payload()) == {
        "speaker-1": "founder", "speaker-2": "investor", "speaker-3": "investor",
    }


def test_clock_conversions_round_trip():
    assert seconds_to_clock(204) == "3:24"
    assert clock_to_seconds("3:24") == 204.0


def test_wire_schema_uses_only_constructs_gemini_accepted_in_stage_a():
    text = str(gemini_response_schema(TimelineRecord))
    for banned in ("additionalProperties", "$ref", "$defs", "anyOf", "pattern", "exclusiveM"):
        assert banned not in text, banned
