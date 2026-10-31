import copy

import pytest
from pydantic import ValidationError

from pipeline.providers.gemini import gemini_response_schema
from pipeline.schemas.stage_b import SegmentationRecord
from tests.stage_b_helpers import segmentation_payload


def broken(payload_fn, mutate):
    payload = copy.deepcopy(payload_fn())
    mutate(payload)
    return payload


def test_valid_payloads_parse():
    assert len(SegmentationRecord.model_validate(segmentation_payload()).episodes) == 2


def test_zero_pitches_is_a_valid_segmentation():
    payload = broken(segmentation_payload, lambda p: p.update(n_pitches=0, episodes=[]))
    assert SegmentationRecord.model_validate(payload).n_pitches == 0


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["episodes"][1].update(start_turn=4), "overlap"),
    (lambda p: p["episodes"][0].update(start_turn=5, end_turn=2), "start_turn"),
    (lambda p: p["episodes"][0].update(resolution="won"), "resolution"),
    (lambda p: p.update(n_pitches=-1), "n_pitches"),
    (lambda p: p["episodes"][0].update(probing_speaker="Lori"), "speaker-"),
    (lambda p: p["episodes"][0].update(start="four oh five"), "minutes"),
    (lambda p: p["episodes"][0].update(start="0:09", end="0:06"), "before"),
])
def test_segmentation_rejects(mutate, message):
    with pytest.raises(ValidationError, match=message):
        SegmentationRecord.model_validate(broken(segmentation_payload, mutate))


def test_a_video_with_no_episodes_is_a_valid_segmentation():
    payload = broken(segmentation_payload, lambda p: p.update(episodes=[]))
    assert SegmentationRecord.model_validate(payload).episodes == []


def test_wire_schema_uses_only_constructs_gemini_accepted_in_stage_a():
    text = str(gemini_response_schema(SegmentationRecord))
    for banned in ("additionalProperties", "$ref", "$defs", "anyOf", "pattern", "exclusiveM"):
        assert banned not in text, banned


# --- Revision R2: investor decisions are outcomes, not episodes ------------------------


def test_decisions_and_deal_parse():
    record = SegmentationRecord.model_validate(segmentation_payload())
    assert [(d.speaker, d.decision) for d in record.decisions] == [
        ("speaker-2", "offer"), ("speaker-3", "out")]
    assert record.deal is True


@pytest.mark.parametrize("mutate, message", [
    (lambda p: p["decisions"][1].update(turn=None), "turn"),                       # missing on "out"
    (lambda p: p["decisions"][1].update(terms="10% equity"), "terms"),             # terms on "out"
    (lambda p: p["decisions"].append(dict(p["decisions"][0])), "speaker"),         # duplicate speaker
    (lambda p: p.update(deal=True, decisions=[p["decisions"][1]]), "deal"),        # deal=True, no offer
])
def test_decision_rejects(mutate, message):
    with pytest.raises(ValidationError, match=message):
        SegmentationRecord.model_validate(broken(segmentation_payload, mutate))


def test_no_decision_shown_needs_no_turn():
    payload = broken(
        segmentation_payload,
        lambda p: p.update(deal=False, decisions=[
            {"speaker": "speaker-2", "decision": "no_decision_shown"}]),
    )
    record = SegmentationRecord.model_validate(payload)
    assert record.decisions[0].turn is None
