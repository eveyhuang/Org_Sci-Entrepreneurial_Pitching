import json

import pandas as pd
import pytest

from pipeline.aggregate import build_tables
from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES
from tests.conftest import needs_ffmpeg
from tests.test_annotate import build, project  # noqa: F401  (fixtures)


def run_two_replicates(tmp_path, fixtures):
    from pipeline.annotate import run_pass
    from pipeline.providers.fake import FakeProvider

    cfg, registry, run = build(tmp_path, k=2)
    run_pass(cfg, cfg.passes[0], registry, run, FakeProvider(fixtures), progress=False)
    return cfg, registry, run


@needs_ffmpeg
def test_annotations_table_is_long_with_one_row_per_code_and_replicate(project):
    tmp_path, fixtures = project
    cfg, registry, run = run_two_replicates(tmp_path, fixtures)
    tables = build_tables(cfg, registry, run)

    annotations = pd.read_parquet(tables["annotations"])
    # 2 videos x 2 replicates x 3 codes
    assert len(annotations) == 12
    assert set(annotations["code_id"]) == set(STAGE_A_DELIVERY_CODES)
    assert set(annotations["replicate_k"]) == {0, 1}
    for column in (
        "video_id", "episode_idx", "code_id", "replicate_k", "value", "uncodable",
        "confidence", "evidence", "pass", "model_version", "prompt_hash", "run_id",
    ):
        assert column in annotations.columns
    registry.close()


@needs_ffmpeg
def test_videos_table_has_one_row_per_video_with_mean_and_sd(project):
    tmp_path, fixtures = project
    cfg, registry, run = run_two_replicates(tmp_path, fixtures)
    tables = build_tables(cfg, registry, run)

    videos = pd.read_parquet(tables["videos"])
    assert len(videos) == 2
    assert set(videos["video_id"]) == {"abc", "def"}
    assert videos.loc[videos.video_id == "abc", "pitch_positivity_mean"].iloc[0] == pytest.approx(4)
    assert videos.loc[videos.video_id == "abc", "pitch_positivity_sd"].iloc[0] == pytest.approx(0.0)
    assert videos.loc[videos.video_id == "abc", "pitch_positivity_n"].iloc[0] == 2
    assert videos["pitch_positivity_n"].dtype == "Int64"
    assert "duration_s" in videos.columns
    registry.close()


@needs_ffmpeg
def test_uncodable_ratings_are_missing_not_zero(project):
    tmp_path, fixtures = project
    payload = json.loads((fixtures / "holistic__abc.json").read_text())
    payload["pitch_positivity"] = {
        "value": None, "uncodable": True, "evidence": "presenter off camera"
    }
    (fixtures / "holistic__abc.json").write_text(json.dumps(payload))

    cfg, registry, run = run_two_replicates(tmp_path, fixtures)
    tables = build_tables(cfg, registry, run)

    annotations = pd.read_parquet(tables["annotations"])
    row = annotations[(annotations.video_id == "abc") & (annotations.code_id == "pitch_positivity")]
    assert row["value"].isna().all()
    assert row["uncodable"].all()

    videos = pd.read_parquet(tables["videos"])
    assert pd.isna(videos.loc[videos.video_id == "abc", "pitch_positivity_mean"].iloc[0])
    assert videos.loc[videos.video_id == "abc", "pitch_positivity_n"].iloc[0] == 0
    registry.close()


@needs_ffmpeg
def test_build_tables_is_rerunnable(project):
    tmp_path, fixtures = project
    cfg, registry, run = run_two_replicates(tmp_path, fixtures)
    first = build_tables(cfg, registry, run)
    second = build_tables(cfg, registry, run)
    assert first == second
    assert pd.read_parquet(second["annotations"]).shape[0] == 12
    registry.close()


@needs_ffmpeg
def test_non_rating_fields_are_not_emitted_as_constructs(project):
    tmp_path, fixtures = project
    cfg, registry, run = run_two_replicates(tmp_path, fixtures)
    tables = build_tables(cfg, registry, run)
    annotations = pd.read_parquet(tables["annotations"])
    assert "overall_notes" not in set(annotations["code_id"])
    assert set(annotations["code_id"]) == set(STAGE_A_DELIVERY_CODES)
    registry.close()


def test_is_rating_rejects_lookalike_payloads():
    from pipeline.aggregate import _is_rating

    assert _is_rating({"value": 4, "uncodable": False, "confidence": "high", "evidence": "x"})
    assert not _is_rating("plain string")
    assert not _is_rating({"evidence": "a structured note that is not a rating"})
    assert not _is_rating({"value": 4, "evidence": "missing the other keys"})
