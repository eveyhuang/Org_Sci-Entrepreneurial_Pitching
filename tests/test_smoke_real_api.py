"""One real API call. Excluded from default runs; run with `pytest -m real_api`."""

import os

import pytest

from pipeline.codebook import load_codebook
from pipeline.config import ModelConfig
from pipeline.prompts import render_prompt
from pipeline.providers.base import GenerateRequest
from pipeline.providers.gemini import build_gemini_provider
from pipeline.registry import Registry
from pipeline.schemas.stage_a_perceived_delivery import StageADeliveryRecord
from tests.conftest import make_test_video, needs_ffmpeg

MODEL = os.environ.get("SMOKE_MODEL", "gemini-3.1-flash-lite-preview")


@pytest.mark.real_api
@needs_ffmpeg
def test_real_gemini_call_returns_valid_stage_a_record(tmp_path):
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    registry = Registry(tmp_path / "registry.sqlite")
    provider = build_gemini_provider(registry)
    video = make_test_video(tmp_path / "clip.mp4", seconds=3.0)
    handle = provider.upload_media(video)
    prompt = render_prompt(
        "stage_a_perceived_delivery.j2",
        codebook=load_codebook("pipeline/codebooks/stage_a_perceived_delivery.yaml"),
    )
    response = provider.generate(
        GenerateRequest(
            prompt=prompt, media=handle, schema=StageADeliveryRecord,
            model=ModelConfig(name=MODEL), unit_key="smoke__clip",
        )
    )
    assert isinstance(response.parsed, StageADeliveryRecord)
    assert response.model_version
    registry.close()
