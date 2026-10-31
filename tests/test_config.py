import textwrap

import pytest

from pipeline.config import load_config
from pipeline.errors import ConfigError


def write_config(tmp_path, body: str, *, make_files: bool = True):
    root = tmp_path
    if make_files:
        (root / "videos").mkdir(exist_ok=True)
        (root / "meta.csv").write_text("video_id,platform,url\n")
        (root / "book.yaml").write_text("[]\n")
        prompts = root / "prompts"
        prompts.mkdir(exist_ok=True)
        (prompts / "holistic.j2").write_text("hello")
    path = root / "cfg.yaml"
    path.write_text(textwrap.dedent(body))
    return path


BASE = """
    run_name: stage_a_v1
    corpus:
      name: hu-ma
      video_dir: videos
      metadata_csv: meta.csv
    model:
      name: gemini-3.1-pro-preview
      temperature: 0.0
      max_output_tokens: 16384
    limits:
      requests_per_minute: 8
      max_concurrency: 4
    passes:
      - name: holistic
        unit: PITCH
        prompt: holistic.j2
        schema: pipeline.schemas.stage_a.StageARecord
        k: 5
        media: full
    codebook: book.yaml
"""


def test_loads_valid_config(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE), prompt_dir=tmp_path / "prompts")
    assert cfg.run_name == "stage_a_v1"
    assert cfg.model.name == "gemini-3.1-pro-preview"
    assert cfg.limits.max_concurrency == 4
    assert cfg.limits.max_attempts == 6  # default applied
    assert len(cfg.passes) == 1
    assert cfg.passes[0].k == 5
    assert cfg.passes[0].unit == "PITCH"
    assert cfg.corpus.video_dir == tmp_path / "videos"


def test_paths_resolve_relative_to_config_file(tmp_path):
    cfg = load_config(write_config(tmp_path, BASE), prompt_dir=tmp_path / "prompts")
    assert cfg.codebook.is_absolute()
    assert cfg.codebook.exists()


def test_rejects_model_list(tmp_path):
    body = BASE.replace("name: gemini-3.1-pro-preview", "name: [a, b]")
    with pytest.raises(ConfigError, match="single model"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_rejects_missing_prompt_template(tmp_path):
    body = BASE.replace("prompt: holistic.j2", "prompt: nope.j2")
    with pytest.raises(ConfigError, match="prompt template"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_rejects_unimportable_schema(tmp_path):
    body = BASE.replace(
        "schema: pipeline.schemas.stage_a.StageARecord",
        "schema: pipeline.schemas.stage_a.NoSuchRecord",
    )
    with pytest.raises(ConfigError, match="schema"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_rejects_schema_that_is_not_a_pydantic_model(tmp_path):
    body = BASE.replace(
        "schema: pipeline.schemas.stage_a.StageARecord",
        "schema: pipeline.config.load_config",
    )
    with pytest.raises(ConfigError, match="BaseModel"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_rejects_forward_depends_on(tmp_path):
    # The second pass must land inside the `passes:` block, not after `codebook:`.
    second_pass = """      - name: episode
        unit: EPISODE
        prompt: holistic.j2
        schema: pipeline.schemas.stage_a.StageARecord
        depends_on: nonexistent
"""
    body = BASE.replace("    codebook: book.yaml", second_pass + "    codebook: book.yaml")
    with pytest.raises(ConfigError, match="depends_on"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_rejects_bad_unit_and_media(tmp_path):
    with pytest.raises(ConfigError, match="unit"):
        load_config(
            write_config(tmp_path, BASE.replace("unit: PITCH", "unit: SCENE")),
            prompt_dir=tmp_path / "prompts",
        )
    with pytest.raises(ConfigError, match="media"):
        load_config(
            write_config(tmp_path, BASE.replace("media: full", "media: hologram")),
            prompt_dir=tmp_path / "prompts",
        )


def test_rejects_missing_video_dir(tmp_path):
    body = BASE.replace("video_dir: videos", "video_dir: absent")
    with pytest.raises(ConfigError, match="video_dir"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_chunk_unit_is_rejected(tmp_path):
    """CHUNK was never implemented; accepting it silently ran a whole-video pass."""
    path = write_config(tmp_path, BASE.replace("unit: PITCH", "unit: CHUNK"))
    with pytest.raises(ConfigError, match="unit"):
        load_config(path, prompt_dir=tmp_path / "prompts")


from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVED_BOOK = REPO / "docs" / "codebooks" / "archive" / "stage_a" / "stage_a_pitch_factor__6a7750e44d39b335.yaml"


def test_prompt_dir_key_in_yaml_is_resolved_relative_to_the_config(tmp_path):
    templates = tmp_path / "frozen_templates"
    templates.mkdir()
    (templates / "holistic.j2").write_text("x")
    cfg = load_config(write_config(tmp_path, BASE + "    prompt_dir: frozen_templates\n"))
    assert cfg.prompt_dir == templates.resolve()
    assert cfg.passes[0].prompt == "holistic.j2"


def test_explicit_prompt_dir_argument_overrides_the_yaml_key(tmp_path):
    (tmp_path / "frozen_templates").mkdir()
    cfg = load_config(write_config(tmp_path, BASE + "    prompt_dir: frozen_templates\n"), prompt_dir=tmp_path / "prompts")
    assert cfg.prompt_dir == tmp_path / "prompts"


def test_missing_prompt_dir_key_target_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="prompt_dir"):
        load_config(write_config(tmp_path, BASE + "    prompt_dir: nowhere\n"))


def test_a_current_config_may_not_read_a_codebook_under_docs_codebooks(tmp_path):
    body = BASE.replace("codebook: book.yaml", f"codebook: {ARCHIVED_BOOK}")
    with pytest.raises(ConfigError, match="historical"):
        load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")


def test_a_historical_config_may_read_an_archived_codebook(tmp_path):
    body = BASE.replace("codebook: book.yaml", f"codebook: {ARCHIVED_BOOK}") + "    historical: true\n"
    cfg = load_config(write_config(tmp_path, body), prompt_dir=tmp_path / "prompts")
    assert cfg.historical is True and cfg.codebook == ARCHIVED_BOOK


ARCHIVE_PROMPTS = REPO / "docs" / "codebooks" / "archive" / "stage_b_v1_episode"


def test_a_current_config_may_not_use_a_prompt_dir_under_docs_codebooks_via_the_yaml_key(tmp_path):
    body = BASE + f"    prompt_dir: {ARCHIVE_PROMPTS}\n"
    with pytest.raises(ConfigError, match="historical"):
        load_config(write_config(tmp_path, body))


def test_a_current_config_may_not_use_a_prompt_dir_under_docs_codebooks_via_the_argument(tmp_path):
    with pytest.raises(ConfigError, match="historical"):
        load_config(write_config(tmp_path, BASE), prompt_dir=ARCHIVE_PROMPTS)


def test_a_historical_config_may_use_an_archived_prompt_dir(tmp_path):
    body = BASE.replace("holistic.j2", "stage_b_timeline.j2") + f"    prompt_dir: {ARCHIVE_PROMPTS}\n    historical: true\n"
    cfg = load_config(write_config(tmp_path, body))
    assert cfg.prompt_dir == ARCHIVE_PROMPTS.resolve()


@pytest.mark.parametrize("value", ['"false"', '"true"', "1", "0", "yes_please"])
def test_historical_must_be_a_boolean(tmp_path, value):
    with pytest.raises(ConfigError, match="historical"):
        load_config(write_config(tmp_path, BASE + f"    historical: {value}\n"), prompt_dir=tmp_path / "prompts")
