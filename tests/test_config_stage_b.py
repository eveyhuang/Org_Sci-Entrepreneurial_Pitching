"""Stage B config fields. Additive: every Stage A config must stay valid."""
from pathlib import Path

import pytest
import yaml

from pipeline.config import load_config
from pipeline.errors import ConfigError

P = {"name": "a", "unit": "PITCH", "prompt": "a.j2",
     "schema": "pipeline.schemas.stage_a.StageARecord"}


def make(tmp_path: Path, passes: list[dict], *, codebook: str | None = "book.yaml", corpus: dict | None = None):
    (tmp_path / "videos").mkdir(exist_ok=True)
    (tmp_path / "meta.csv").write_text("video_id\n")
    (tmp_path / "book.yaml").write_text("[]\n")
    (tmp_path / "other.yaml").write_text("[]\n")
    prompts = tmp_path / "prompts"
    prompts.mkdir(exist_ok=True)
    for name in ("a.j2", "b.j2"):
        (prompts / name).write_text("hello")
    data = {
        "run_name": "t",
        "corpus": {"name": "c", "video_dir": "videos", "metadata_csv": "meta.csv", **(corpus or {})},
        "model": {"name": "gemini-pro-x"},
        "passes": passes,
    }
    if codebook is not None:
        data["codebook"] = codebook
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_config(path, prompt_dir=prompts)


def test_pass_model_overrides_run_model(tmp_path):
    cfg = make(tmp_path, [
        {**P, "model": {"name": "gemini-flash-x", "max_output_tokens": 65536}},
        {**P, "name": "b", "prompt": "b.j2"},
    ])
    assert cfg.model_for(cfg.passes[0]).name == "gemini-flash-x"
    assert cfg.model_for(cfg.passes[0]).max_output_tokens == 65536
    assert cfg.model_for(cfg.passes[1]).name == "gemini-pro-x"


def test_pass_model_list_is_still_rejected(tmp_path):
    with pytest.raises(ConfigError, match="fallback"):
        make(tmp_path, [{**P, "model": {"name": ["a", "b"]}}])


def test_pass_codebook_resolves_and_must_exist(tmp_path):
    cfg = make(tmp_path, [{**P, "codebook": "other.yaml"}])
    assert cfg.codebook_for(cfg.passes[0]) == (tmp_path / "other.yaml").resolve()
    with pytest.raises(ConfigError, match="codebook"):
        make(tmp_path, [{**P, "codebook": "nope.yaml"}])


def test_run_codebook_is_optional(tmp_path):
    cfg = make(tmp_path, [P], codebook=None)
    assert cfg.codebook is None and cfg.codebook_for(cfg.passes[0]) is None


def test_generated_schema_needs_some_codebook(tmp_path):
    cfg = make(tmp_path, [{**P, "schema": "codebook"}])
    assert cfg.passes[0].schema == "codebook"
    with pytest.raises(ConfigError, match="schema: codebook"):
        make(tmp_path, [{**P, "schema": "codebook"}], codebook=None)


def test_context_and_produces_use_fixed_vocabularies(tmp_path):
    cfg = make(tmp_path, [{**P, "context": ["roster", "timeline"], "produces": "episodes"}])
    assert cfg.passes[0].context == ("roster", "timeline")
    assert cfg.passes[0].produces == "episodes"
    assert cfg.pass_producing("episodes").name == "a"
    assert cfg.pass_producing("timeline") is None
    with pytest.raises(ConfigError, match="context"):
        make(tmp_path, [{**P, "context": ["subtitles"]}])
    with pytest.raises(ConfigError, match="produces"):
        make(tmp_path, [{**P, "produces": "magic"}])
    # The old (pre-Revision-R1) words are no longer part of the vocabulary.
    with pytest.raises(ConfigError, match="context"):
        make(tmp_path, [{**P, "context": ["transcript"]}])
    with pytest.raises(ConfigError, match="produces"):
        make(tmp_path, [{**P, "produces": "transcript"}])


def test_media_none_only_on_pitch_units(tmp_path):
    assert make(tmp_path, [{**P, "media": "none"}]).passes[0].media == "none"
    with pytest.raises(ConfigError, match="media: none"):
        make(tmp_path, [{**P, "unit": "EPISODE", "media": "none"}])


def test_prepared_unit_is_accepted(tmp_path):
    assert make(tmp_path, [{**P, "unit": "PREPARED"}]).passes[0].unit == "PREPARED"


def test_max_duration(tmp_path):
    assert make(tmp_path, [P]).corpus.max_duration_s is None
    assert make(tmp_path, [P], corpus={"max_duration_s": 1800}).corpus.max_duration_s == 1800.0
    with pytest.raises(ConfigError, match="max_duration_s"):
        make(tmp_path, [P], corpus={"max_duration_s": 0})


def test_dev_set_is_a_recorded_list_of_ids(tmp_path):
    (tmp_path / "dev.txt").write_text("# development set\nst_aaa\n\nst_bbb  \n")
    cfg = make(tmp_path, [P], corpus={"dev_set": "dev.txt"})
    assert cfg.dev_video_ids() == ["st_aaa", "st_bbb"]
    with pytest.raises(ConfigError, match="dev_set"):
        make(tmp_path, [P], corpus={"dev_set": "missing.txt"})
    with pytest.raises(ConfigError, match="no dev set"):
        make(tmp_path, [P]).dev_video_ids()
    (tmp_path / "dev.txt").write_text("# nothing yet\n")
    with pytest.raises(ConfigError, match="lists no videos"):
        make(tmp_path, [P], corpus={"dev_set": "dev.txt"}).dev_video_ids()
