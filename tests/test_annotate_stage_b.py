import dataclasses
import json

from pipeline.annotate import config_fingerprint, run_pass
from pipeline.codebook import load_codebook
from pipeline.prompts import prompt_hash, render_prompt, render_pass_prompt
from pipeline.providers.fake import FakeProvider
from pipeline.registry import PREPARED_IDX
from pipeline.runs import schema_path
from tests.conftest import needs_ffmpeg
from tests.stage_b_helpers import STAGE_A_BOOK, VIDEO, make_project

HOLISTIC = {"name": "holistic", "unit": "PITCH", "prompt": "stage_a_perceived_delivery.j2",
            "schema": "pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord", "codebook": STAGE_A_BOOK}


@needs_ffmpeg
def test_pass_model_is_the_model_that_is_called_and_recorded(tmp_path):
    cfg, registry, run, fixtures = make_project(
        tmp_path, [{**HOLISTIC, "model": {"name": "gemini-flash-x"}}]
    )
    provider = FakeProvider(fixtures)
    summary = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    assert (summary.done, summary.model) == (1, "gemini-flash-x")
    assert provider.requests[0].model.name == "gemini-flash-x"
    unit = registry.done_units(run.run_id)[0]
    assert unit.model == "gemini-flash-x"
    assert json.loads(open(unit.parsed_path).read())["model_requested"] == "gemini-flash-x"
    registry.close()


@needs_ffmpeg
def test_media_none_sends_no_media(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, [{**HOLISTIC, "media": "none"}])
    provider = FakeProvider(fixtures)
    assert run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False).done == 1
    assert provider.uploads == [] and provider.requests[0].media is None
    registry.close()


@needs_ffmpeg
def test_run_pass_archives_the_wire_schema_next_to_the_prompt(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, [HOLISTIC])
    provider = FakeProvider(fixtures)
    summary = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    path = schema_path(run, "holistic", summary.prompt_hash)
    assert path.exists()
    schema = json.loads(path.read_text())
    assert "properties" in schema and "additionalProperties" not in json.dumps(schema)
    registry.close()


@needs_ffmpeg
def test_pass_codebook_is_rendered_and_absence_renders_an_empty_one(tmp_path):
    bare = {k: v for k, v in HOLISTIC.items() if k != "codebook"}
    cfg, registry, _, _ = make_project(tmp_path, [HOLISTIC, {**bare, "name": "bare"}])
    with_book, _ = render_pass_prompt(cfg, cfg.passes[0])
    without, _ = render_pass_prompt(cfg, cfg.passes[1])
    # The heading is rendered only from codebook entries, so it distinguishes the two.
    assert "--- pitch_positivity ---" in with_book
    assert "--- pitch_positivity ---" not in without
    registry.close()


@needs_ffmpeg
def test_thinking_tokens_are_summed_and_recorded_in_manifest(tmp_path):
    cfg, registry, run, fixtures = make_project(
        tmp_path, [{**HOLISTIC, "k": 3}], k={"holistic": 3}
    )

    class ThinkingProvider(FakeProvider):
        def generate(self, request):
            response = super().generate(request)
            return dataclasses.replace(response, thinking_tokens=5)

    provider = ThinkingProvider(fixtures)
    summary = run_pass(cfg, cfg.passes[0], registry, run, provider, progress=False)
    assert summary.done == 3
    assert summary.thinking_tokens == 5 * summary.done

    from pipeline.runs import read_manifest

    manifest = read_manifest(run)
    assert manifest["passes"]["holistic"]["thinking_tokens"] == 5 * summary.done
    registry.close()


@needs_ffmpeg
def test_prepared_pass_runs_the_unforked_stage_a_instrument_on_the_prepared_clip(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation", "prepared"])
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    prepared = cfg.get_pass("prepared")
    assert run_pass(cfg, prepared, registry, run, provider, progress=False).enqueued == 0  # not frozen
    registry.set_frozen(VIDEO, True)
    summary = run_pass(cfg, prepared, registry, run, provider, progress=False)
    assert (summary.done, summary.failed) == (1, 0)

    stage_a = render_prompt("stage_a_perceived_delivery.j2", codebook=load_codebook(STAGE_A_BOOK))
    assert summary.prompt_hash == prompt_hash(stage_a)
    unit = registry.done_units(run.run_id, pass_name="prepared")[0]
    assert unit.span == (0.0, 4.0)
    assert json.loads(open(unit.parsed_path).read())["episode_idx"] == PREPARED_IDX
    assert "0.000_4.000" in provider.uploads[-1].name  # the clip, not the whole video
    registry.close()


def test_stage_a_fingerprint_is_unchanged(tmp_path):
    # Paid-for ten-item Stage A unit ids hash this exact string; changing it re-enqueues everything.
    # Those runs now reproduce from the archived config.
    from tests.test_stage_a_perceived_delivery import config_with_local_corpus
    cfg = config_with_local_corpus("archived_stage_a_pitch_factor.yaml", tmp_path)
    assert config_fingerprint(cfg, cfg.passes[0]) == (
        "t=0.0|max_out=16384|schema=pipeline.schemas.stage_a.StageARecord"
    )


@needs_ffmpeg
def test_episode_pass_codes_each_frozen_episode_with_its_own_slice(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ["timeline", "segmentation", "episode"])
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    registry.set_frozen(VIDEO, True)
    summary = run_pass(cfg, cfg.get_pass("episode"), registry, run, provider, progress=False)
    assert (summary.done, summary.failed) == (2, 0)

    sent = {r.unit_key: r.prompt for r in provider.requests if r.unit_key.startswith("episode")}
    second = sent[f"episode__{VIDEO}__7.000_12.000__k0"]
    assert "[t5 0:00–0:02] speaker-3 (investor): challenges the valuation" in second
    assert "asks about margins" not in second            # episode 0's turns are not leaked in
    assert "This episode's topic: valuation" in second
    record = json.loads(open(registry.done_units(run.run_id, pass_name="episode")[0].parsed_path).read())
    assert list(record["record"]) == ["PR-FRAME", "RS-EVID", "PREP-SUBS", "RAP-INTERRUPT"]
    assert record["episode_idx"] == 0 and record["record"]["PR-FRAME"]["value"] == "prevention"
    registry.close()
