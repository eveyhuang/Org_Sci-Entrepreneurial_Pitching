import json

import pytest

from pipeline.config import ModelConfig
from pipeline.errors import ProviderError, SchemaInvalidError
from pipeline.providers.base import GenerateRequest
from pipeline.providers.fake import FakeProvider
from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES, StageADeliveryRecord


def valid_payload():
    rating = {"value": 4, "uncodable": False, "evidence": "Steady, warm delivery at 0:03."}
    return {name: dict(rating) for name in STAGE_A_DELIVERY_CODES} | {"overall_notes": "ok"}


@pytest.fixture
def fixtures(tmp_path):
    directory = tmp_path / "fixtures"
    directory.mkdir()
    (directory / "holistic__vidA.json").write_text(json.dumps(valid_payload()))
    return directory


def request_for(video_id: str, schema=StageADeliveryRecord) -> GenerateRequest:
    return GenerateRequest(
        prompt="rate this video",
        media=None,
        schema=schema,
        model=ModelConfig(name="fake-model"),
        unit_key=f"holistic__{video_id}",
    )


def test_replays_fixture_and_parses(fixtures):
    provider = FakeProvider(fixtures)
    response = provider.generate(request_for("vidA"))
    assert isinstance(response.parsed, StageADeliveryRecord)
    assert response.parsed.pitch_positivity.value == 4
    assert response.model_version == "fake-model-v1"
    assert response.input_tokens > 0


def test_missing_fixture_raises_provider_error(fixtures):
    with pytest.raises(ProviderError, match="no fixture"):
        FakeProvider(fixtures).generate(request_for("unknown"))


def test_invalid_fixture_raises_schema_invalid(fixtures):
    (fixtures / "holistic__bad.json").write_text('{"pitch_positivity": {"value": 5.0}}')
    with pytest.raises(SchemaInvalidError):
        FakeProvider(fixtures).generate(request_for("bad"))


def test_error_plan_injects_failures_then_succeeds(fixtures):
    provider = FakeProvider(
        fixtures, error_plan={"holistic__vidA": [ProviderError("503 busy", status=503)]}
    )
    with pytest.raises(ProviderError):
        provider.generate(request_for("vidA"))
    assert provider.generate(request_for("vidA")).parsed is not None


def test_upload_media_returns_handle(tmp_path, fixtures):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not really a video")
    provider = FakeProvider(fixtures)
    handle = provider.upload_media(media)
    assert handle.local_path == media
    assert handle.name.startswith("fake-files/")
    assert provider.uploads == [media]

    second = tmp_path / "clip2.mp4"
    second.write_bytes(b"another")
    provider.upload_media(second)
    assert provider.uploads == [media, second]


def test_records_calls_for_assertions(fixtures):
    provider = FakeProvider(fixtures)
    provider.generate(request_for("vidA"))
    assert provider.calls == ["holistic__vidA"]


def test_full_unit_key_falls_back_to_the_pass_video_fixture(fixtures):
    """A whole-video fixture serves every replicate of that video."""
    provider = FakeProvider(fixtures)
    request = GenerateRequest(
        prompt="rate this video", media=None, schema=StageADeliveryRecord,
        model=ModelConfig(name="fake-model"), unit_key="holistic__vidA__k3",
    )
    assert provider.generate(request).parsed is not None
    assert provider.calls == ["holistic__vidA__k3"]  # the FULL key is recorded


def test_exact_unit_key_fixture_wins_over_the_fallback(fixtures):
    payload = valid_payload()
    payload["pitch_positivity"]["value"] = 2
    (fixtures / "holistic__vidA__k1.json").write_text(json.dumps(payload))
    provider = FakeProvider(fixtures)
    request = GenerateRequest(
        prompt="p", media=None, schema=StageADeliveryRecord,
        model=ModelConfig(name="fake-model"), unit_key="holistic__vidA__k1",
    )
    assert provider.generate(request).parsed.pitch_positivity.value == 2
