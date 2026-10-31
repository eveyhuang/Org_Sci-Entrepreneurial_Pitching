"""Multi-clip generation requests: several labelled media parts in one call."""

import json

import pytest

from pipeline.config import ModelConfig
from pipeline.providers.base import GenerateRequest, MediaHandle
from pipeline.providers.fake import FakeProvider
from pipeline.providers.gemini import GeminiProvider
from pipeline.registry import Registry
from pipeline.schemas.stage_a_perceived_delivery import StageADeliveryRecord
from tests.test_provider_gemini import StubClient, StubFile, StubModels, valid_json

MODEL = ModelConfig(name="gemini-3.1-pro-preview")


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "registry.sqlite")
    yield r
    r.close()


def _request(media=None, media_parts=None):
    kwargs = {} if media_parts is None else {"media_parts": media_parts}
    return GenerateRequest(prompt="rank them", media=media, schema=StageADeliveryRecord, model=MODEL,
                           unit_key="rank__b01", **kwargs)


def _parts(n=3):
    return tuple((label, MediaHandle(name=f"files/{label.lower()}")) for label in "ABCDE"[:n])


def test_media_parts_defaults_to_empty():
    assert _request().media_parts == ()


def test_three_clips_are_sent_in_order_each_after_its_label(reg):
    models = StubModels(text=valid_json())
    GeminiProvider(StubClient(models), reg).generate(_request(media_parts=_parts(3)))
    contents = models.calls[0]["contents"]
    assert len(contents) == 7
    assert contents[0] == "rank them"
    assert contents[1::2] == ["Clip A:", "Clip B:", "Clip C:"]
    assert [part.name for part in contents[2::2]] == ["files/a", "files/b", "files/c"]
    assert all(isinstance(part, StubFile) for part in contents[2::2])


def test_labels_are_used_verbatim(reg):
    models = StubModels(text=valid_json())
    parts = (("Z9", MediaHandle(name="files/1")), ("second one", MediaHandle(name="files/2")))
    GeminiProvider(StubClient(models), reg).generate(_request(media_parts=parts))
    assert models.calls[0]["contents"][1::2] == ["Clip Z9:", "Clip second one:"]


def test_media_together_with_media_parts_raises_before_any_call(reg):
    models = StubModels(text=valid_json())
    client = StubClient(models)
    with pytest.raises(ValueError, match="media_parts"):
        GeminiProvider(client, reg).generate(
            _request(media=MediaHandle(name="files/whole"), media_parts=_parts(2)))
    assert models.calls == [] and client.files.get_calls == 0


def test_single_media_request_is_unchanged(reg):
    models = StubModels(text=valid_json())
    GeminiProvider(StubClient(models), reg).generate(_request(media=MediaHandle(name="files/whole")))
    contents = models.calls[0]["contents"]
    assert len(contents) == 2 and contents[0] == "rank them"
    assert contents[1] == StubFile(name="files/whole", state=contents[1].state)
    assert not any(isinstance(c, str) and c.startswith("Clip") for c in contents)


def test_request_without_media_sends_the_prompt_alone(reg):
    models = StubModels(text=valid_json())
    GeminiProvider(StubClient(models), reg).generate(_request())
    assert models.calls[0]["contents"] == ["rank them"]


def test_fake_provider_accepts_and_records_media_parts(tmp_path):
    (tmp_path / "rank__b01.json").write_text(valid_json())
    provider = FakeProvider(tmp_path)
    request = _request(media_parts=_parts(3))
    response = provider.generate(request)
    assert isinstance(response.parsed, StageADeliveryRecord)
    assert provider.calls == ["rank__b01"]
    assert provider.requests[0].media_parts == request.media_parts
    assert [label for label, _ in provider.requests[0].media_parts] == ["A", "B", "C"]
    assert json.loads(response.raw_text) == json.loads(valid_json())
