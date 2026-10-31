import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from pipeline.config import ModelConfig
from pipeline.errors import ProviderError, RunFatalError, SchemaInvalidError
from pipeline.providers.base import GenerateRequest, MediaHandle
from pipeline.providers.gemini import GeminiProvider
from pipeline.registry import Registry
from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES, StageADeliveryRecord


def valid_json() -> str:
    rating = {"value": 4, "uncodable": False, "evidence": "Steady, warm delivery at 0:03."}
    return json.dumps({name: dict(rating) for name in STAGE_A_DELIVERY_CODES} | {"overall_notes": ""})


@dataclass
class StubFile:
    name: str
    uri: str = "gs://x"
    state: object = None


class State:
    def __init__(self, name):
        self.name = name


class StubFiles:
    def __init__(self, get_raises=False):
        self.uploaded: list[str] = []
        self.get_calls = 0
        self.get_raises = get_raises

    def upload(self, file):
        self.uploaded.append(str(file))
        return StubFile(name=f"files/{len(self.uploaded)}", state=State("ACTIVE"))

    def get(self, name):
        self.get_calls += 1
        if self.get_raises:
            raise ProviderError("gone", status=404)
        return StubFile(name=name, state=State("ACTIVE"))


class StubModels:
    def __init__(self, text=None, exc=None, finish_reason="STOP", thoughts_token_count=None):
        self.text = text
        self.exc = exc
        self.finish_reason = finish_reason
        self.thoughts_token_count = thoughts_token_count
        self.calls: list[dict] = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.exc:
            raise self.exc
        usage_attrs = {"prompt_token_count": 1234, "candidates_token_count": 567}
        if self.thoughts_token_count is not None:
            usage_attrs["thoughts_token_count"] = self.thoughts_token_count
        return type(
            "Response", (), {
                "text": self.text,
                "parsed": None,
                "model_version": "gemini-3.1-pro-preview-20260401",
                "usage_metadata": type("U", (), usage_attrs)(),
                "candidates": [type("C", (), {"finish_reason": self.finish_reason})()],
                "prompt_feedback": None,
            },
        )()


class StubClient:
    def __init__(self, models, files=None):
        self.models = models
        self.files = files or StubFiles()


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "registry.sqlite")
    yield r
    r.close()


def make_request(media=None):
    return GenerateRequest(
        prompt="rate it", media=media, schema=StageADeliveryRecord,
        model=ModelConfig(name="gemini-3.1-pro-preview"), unit_key="holistic__vidA",
    )


def test_generate_parses_and_reports_model_version(reg):
    client = StubClient(StubModels(text=valid_json()))
    response = GeminiProvider(client, reg).generate(make_request())
    assert isinstance(response.parsed, StageADeliveryRecord)
    assert response.model_version == "gemini-3.1-pro-preview-20260401"
    assert (response.input_tokens, response.output_tokens) == (1234, 567)


def test_generate_reports_thinking_tokens(reg):
    client = StubClient(StubModels(text=valid_json(), thoughts_token_count=1234))
    response = GeminiProvider(client, reg).generate(make_request())
    assert response.thinking_tokens == 1234
    assert response.output_tokens == 567  # visible output stays separate from thinking


def test_generate_defaults_thinking_tokens_to_zero_when_absent(reg):
    client = StubClient(StubModels(text=valid_json()))
    response = GeminiProvider(client, reg).generate(make_request())
    assert response.thinking_tokens == 0


def test_generate_sends_pinned_model_and_schema_config(reg):
    models = StubModels(text=valid_json())
    GeminiProvider(StubClient(models), reg).generate(make_request())
    call = models.calls[0]
    assert call["model"] == "gemini-3.1-pro-preview"
    assert call["config"]["response_mime_type"] == "application/json"
    assert isinstance(call["config"]["response_schema"], dict)
    assert "additionalProperties" not in json.dumps(call["config"]["response_schema"])
    assert call["config"]["temperature"] == 0.0


def test_unparseable_text_raises_schema_invalid_not_salvage(reg):
    client = StubClient(StubModels(text="Sure! Here is the JSON: {oops"))
    with pytest.raises(SchemaInvalidError):
        GeminiProvider(client, reg).generate(make_request())


def test_truncated_response_raises_schema_invalid(reg):
    client = StubClient(StubModels(text=valid_json()[:120], finish_reason="MAX_TOKENS"))
    with pytest.raises(SchemaInvalidError, match="MAX_TOKENS"):
        GeminiProvider(client, reg).generate(make_request())


def test_truncated_response_error_carries_usage_and_model_version(reg):
    client = StubClient(StubModels(text=valid_json()[:120], finish_reason="MAX_TOKENS", thoughts_token_count=89))
    with pytest.raises(SchemaInvalidError) as excinfo:
        GeminiProvider(client, reg).generate(make_request())
    assert excinfo.value.usage == {"input_tokens": 1234, "output_tokens": 567, "thinking_tokens": 89}
    assert excinfo.value.model_version == "gemini-3.1-pro-preview-20260401"


def test_unparseable_and_empty_response_errors_carry_usage(reg):
    for body in ("Sure! Here is the JSON: {oops", ""):
        with pytest.raises(SchemaInvalidError) as excinfo:
            GeminiProvider(StubClient(StubModels(text=body)), reg).generate(make_request())
        assert excinfo.value.usage == {"input_tokens": 1234, "output_tokens": 567, "thinking_tokens": 0}
        assert excinfo.value.model_version == "gemini-3.1-pro-preview-20260401"


def test_safety_block_error_carries_usage_and_model_version(reg):
    class BlockedModels(StubModels):
        def generate_content(self, *, model, contents, config):
            resp = super().generate_content(model=model, contents=contents, config=config)
            resp.prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()
            return resp

    with pytest.raises(ProviderError) as excinfo:
        GeminiProvider(StubClient(BlockedModels(text="")), reg).generate(make_request())
    assert excinfo.value.usage == {"input_tokens": 1234, "output_tokens": 567, "thinking_tokens": 0}
    assert excinfo.value.model_version == "gemini-3.1-pro-preview-20260401"


def test_transport_error_carries_no_usage(reg):
    client = StubClient(StubModels(exc=ProviderError("busy", status=503)))
    with pytest.raises(ProviderError) as excinfo:
        GeminiProvider(client, reg).generate(make_request())
    assert not hasattr(excinfo.value, "usage")


def test_auth_failure_is_run_fatal(reg):
    client = StubClient(StubModels(exc=ProviderError("no key", status=403)))
    with pytest.raises(RunFatalError):
        GeminiProvider(client, reg).generate(make_request())


def test_upload_is_cached_by_content_hash(tmp_path, reg):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")
    files = StubFiles()
    provider = GeminiProvider(StubClient(StubModels(text=valid_json()), files), reg)

    first = provider.upload_media(media)
    second = provider.upload_media(media)
    assert first.name == second.name
    assert len(files.uploaded) == 1  # second call served from cache


def test_expired_cache_entry_triggers_reupload(tmp_path, reg):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")
    files = StubFiles()
    provider = GeminiProvider(StubClient(StubModels(text=valid_json()), files), reg)
    provider.upload_media(media)

    stale = datetime(2020, 1, 1, tzinfo=timezone.utc)
    from pipeline.ingest import sha256_file
    reg.put_upload(sha256_file(media), "files/1", now=stale)

    provider.upload_media(media)
    assert len(files.uploaded) == 2


def test_vanished_remote_file_triggers_reupload(tmp_path, reg):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")
    files = StubFiles()
    provider = GeminiProvider(StubClient(StubModels(text=valid_json()), files), reg)
    provider.upload_media(media)
    files.get_raises = True
    provider.upload_media(media)
    assert len(files.uploaded) == 2


def test_schema_invalid_error_carries_the_raw_response(reg):
    body = "Sure! Here is the JSON: {oops"
    client = StubClient(StubModels(text=body))
    with pytest.raises(SchemaInvalidError) as excinfo:
        GeminiProvider(client, reg).generate(make_request())
    assert excinfo.value.raw_text == body


def test_safety_blocked_response_raises_provider_error(reg):
    response_cls = StubModels().generate_content

    class BlockedModels(StubModels):
        def generate_content(self, *, model, contents, config):
            resp = response_cls(model=model, contents=contents, config=config)
            resp.prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()
            return resp

    with pytest.raises(ProviderError, match="blocked"):
        GeminiProvider(StubClient(BlockedModels(text=valid_json())), reg).generate(make_request())


def test_upload_polls_until_active(tmp_path, reg):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")

    class SlowFiles(StubFiles):
        def __init__(self):
            super().__init__()
            self.polls = 0

        def upload(self, file):
            self.uploaded.append(str(file))
            return StubFile(name="files/slow", state=State("PROCESSING"))

        def get(self, name):
            self.polls += 1
            state = "ACTIVE" if self.polls >= 2 else "PROCESSING"
            return StubFile(name=name, state=State(state))

    files = SlowFiles()
    sleeps: list[float] = []
    provider = GeminiProvider(
        StubClient(StubModels(text=valid_json()), files), reg,
        poll_interval_s=0.0, sleep=sleeps.append,
    )
    handle = provider.upload_media(media)
    assert handle.name == "files/slow"
    assert files.polls >= 2 and sleeps  # it actually waited and re-fetched


def test_upload_raises_when_processing_fails(tmp_path, reg):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")

    class FailingFiles(StubFiles):
        def upload(self, file):
            self.uploaded.append(str(file))
            return StubFile(name="files/bad", state=State("FAILED"))

    provider = GeminiProvider(
        StubClient(StubModels(text=valid_json()), FailingFiles()), reg,
        poll_interval_s=0.0, sleep=lambda s: None,
    )
    with pytest.raises(ProviderError, match="processing failed"):
        provider.upload_media(media)


def test_upload_processing_failure_is_retryable_and_clears_the_cache(tmp_path, reg):
    """A FAILED remote file must be RETRYABLE (503), not ITEM_FATAL, and must not leave
    behind a cached upload row pointing at the dead file, so the retry re-uploads."""
    from pipeline.errors import ErrorClass, classify_exception
    from pipeline.ingest import sha256_file

    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")
    digest = sha256_file(media)
    reg.put_upload(digest, "files/stale")  # simulate a cache entry left by an earlier attempt

    class FailingFiles(StubFiles):
        def get(self, name):
            self.get_calls += 1
            return StubFile(name=name, state=State("FAILED"))  # the cached copy is also dead

        def upload(self, file):
            self.uploaded.append(str(file))
            return StubFile(name="files/bad", state=State("FAILED"))

    provider = GeminiProvider(
        StubClient(StubModels(text=valid_json()), FailingFiles()), reg,
        poll_interval_s=0.0, sleep=lambda s: None,
    )
    with pytest.raises(ProviderError) as excinfo:
        provider.upload_media(media)
    assert excinfo.value.status == 503
    assert classify_exception(excinfo.value) is ErrorClass.RETRYABLE
    assert reg.get_upload(digest) is None


def test_supplied_digest_is_used_instead_of_rehashing(tmp_path, reg):
    """ingest already hashed the file; re-reading it per work unit is pure waste."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"video bytes")
    files = StubFiles()
    provider = GeminiProvider(StubClient(StubModels(text=valid_json()), files), reg)

    # Seed the cache under a digest that is NOT the file's real hash. If upload_media
    # honours the supplied value, the cache hits and nothing is uploaded.
    reg.put_upload("not-the-real-hash", "files/preseeded")
    handle = provider.upload_media(media, file_sha256="not-the-real-hash")
    assert handle.name == "files/preseeded"
    assert files.uploaded == []

    # Without the hint it falls back to hashing the file, misses, and uploads.
    provider.upload_media(media)
    assert len(files.uploaded) == 1


def test_safety_block_preserves_the_response_body(reg):
    """The safety-blocked body is exactly what the published failure taxonomy needs."""
    class BlockedModels(StubModels):
        def generate_content(self, *, model, contents, config):
            resp = super().generate_content(model=model, contents=contents, config=config)
            resp.prompt_feedback = type("F", (), {"block_reason": "SAFETY"})()
            return resp

    body = valid_json()
    with pytest.raises(ProviderError) as excinfo:
        GeminiProvider(StubClient(BlockedModels(text=body)), reg).generate(make_request())
    assert excinfo.value.raw_text == body


# --- gemini_response_schema conversion ------------------------------------

from pipeline.providers.gemini import gemini_response_schema  # noqa: E402


def test_response_schema_has_no_additional_properties():
    schema = gemini_response_schema(StageADeliveryRecord)
    dumped = json.dumps(schema)
    assert "additionalProperties" not in dumped
    assert "$ref" not in dumped
    assert "$defs" not in dumped


def test_response_schema_expresses_optional_as_nullable():
    schema = gemini_response_schema(StageADeliveryRecord)
    value_prop = schema["properties"]["pitch_positivity"]["properties"]["value"]
    assert value_prop["nullable"] is True
    assert value_prop["type"] == "integer"
    assert value_prop["minimum"] == 1
    assert value_prop["maximum"] == 5
    dumped = json.dumps(schema)
    assert '"type": "null"' not in dumped
    assert "'type': 'null'" not in str(schema)


def test_response_schema_inlines_nested_models():
    schema = gemini_response_schema(StageADeliveryRecord)
    for name in STAGE_A_DELIVERY_CODES:
        assert name in schema["properties"]
        construct = schema["properties"][name]
        assert construct["type"] == "object"
        for key in ("value", "uncodable", "evidence"):
            assert key in construct["properties"]


def test_generate_sends_the_converted_schema_not_the_model_class(reg):
    models = StubModels(text=valid_json())
    GeminiProvider(StubClient(models), reg).generate(make_request())
    schema = models.calls[0]["config"]["response_schema"]
    assert isinstance(schema, dict)
    assert "additionalProperties" not in json.dumps(schema)


def test_extra_fields_are_still_rejected_at_parse_time(reg):
    payload = json.loads(valid_json())
    payload["unexpected_extra_field"] = "should not be allowed"
    client = StubClient(StubModels(text=json.dumps(payload)))
    with pytest.raises(SchemaInvalidError):
        GeminiProvider(client, reg).generate(make_request())


def test_dict_parsed_from_sdk_is_revalidated_through_the_schema(reg):
    """The SDK returns a plain dict when given a dict schema; that must not skip validation."""
    payload = valid_json()
    models = StubModels(text=payload)
    original = models.generate_content

    def with_dict_parsed(*, model, contents, config):
        response = original(model=model, contents=contents, config=config)
        response.parsed = json.loads(payload)  # a dict, as the real SDK returns
        return response

    models.generate_content = with_dict_parsed
    response = GeminiProvider(StubClient(models), reg).generate(make_request())
    assert isinstance(response.parsed, StageADeliveryRecord)


def test_dict_parsed_with_extra_field_still_raises(reg):
    """Strictness must survive: an invented field fails even when the SDK pre-parsed it."""
    bad = json.loads(valid_json())
    bad["invented_construct"] = {"value": 3, "uncodable": False, "evidence": "0:01 x"}
    body = json.dumps(bad)
    models = StubModels(text=body)
    original = models.generate_content

    def with_dict_parsed(*, model, contents, config):
        response = original(model=model, contents=contents, config=config)
        response.parsed = bad
        return response

    models.generate_content = with_dict_parsed
    with pytest.raises(SchemaInvalidError):
        GeminiProvider(StubClient(models), reg).generate(make_request())


def test_non_ascii_filename_is_uploaded_as_a_handle(tmp_path, reg):
    """The SDK puts a path's basename in an ASCII-only header; curly quotes in titles broke uploads."""
    media = tmp_path / "“The Product Is Crap!” It’s.mp4"
    media.write_bytes(b"video bytes")
    seen = []

    class HandleFiles(StubFiles):
        def upload(self, file, config=None):
            seen.append((isinstance(file, str), config))
            return super().upload(file)

    GeminiProvider(StubClient(StubModels(text=valid_json()), HandleFiles()), reg).upload_media(media)
    assert seen == [(False, {"mime_type": "video/mp4"})]
