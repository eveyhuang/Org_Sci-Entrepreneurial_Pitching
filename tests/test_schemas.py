import pytest
from pydantic import ValidationError

from pipeline.schemas.common import Rating
from pipeline.schemas.stage_a import STAGE_A_CONSTRUCTS, StageARecord


def rating(**overrides):
    base = dict(value=4, uncodable=False, confidence="high", evidence="00:03 sustained smile")
    base.update(overrides)
    return base


def test_rating_accepts_valid_payload():
    r = Rating(**rating())
    assert r.value == 4 and r.uncodable is False


def test_rating_rejects_out_of_range_value():
    with pytest.raises(ValidationError):
        Rating(**rating(value=6))
    with pytest.raises(ValidationError):
        Rating(**rating(value=0))


def test_rating_rejects_non_integer_value():
    with pytest.raises(ValidationError):
        Rating(**rating(value=3.5))


def test_uncodable_rating_must_have_null_value():
    ok = Rating(**rating(value=None, uncodable=True, evidence="investor off-camera throughout"))
    assert ok.value is None
    with pytest.raises(ValidationError, match="uncodable"):
        Rating(**rating(value=3, uncodable=True))


def test_codable_rating_requires_a_value():
    with pytest.raises(ValidationError, match="value"):
        Rating(**rating(value=None, uncodable=False))


def test_rating_requires_nonempty_evidence():
    with pytest.raises(ValidationError):
        Rating(**rating(evidence="  "))


def test_stage_a_record_requires_all_ten_constructs():
    payload = {name: rating() for name in STAGE_A_CONSTRUCTS}
    record = StageARecord(**payload, overall_notes="steady delivery")
    assert len(STAGE_A_CONSTRUCTS) == 10
    assert record.visual_positive.value == 4

    payload.pop("verbal_ability")
    with pytest.raises(ValidationError, match="verbal_ability"):
        StageARecord(**payload)


def test_stage_a_record_rejects_unknown_fields():
    payload = {name: rating() for name in STAGE_A_CONSTRUCTS}
    with pytest.raises(ValidationError):
        StageARecord(**payload, pitch_factor=0.9)


def test_rating_requires_confidence():
    payload = rating()
    del payload["confidence"]
    with pytest.raises(ValidationError, match="confidence"):
        Rating(**payload)


def test_rating_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Rating(**rating(), notes="extra field")


def test_rating_rejects_invalid_confidence_value():
    with pytest.raises(ValidationError, match="confidence"):
        Rating(**rating(confidence="very high"))
