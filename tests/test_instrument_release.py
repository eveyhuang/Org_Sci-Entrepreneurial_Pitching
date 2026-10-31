"""Canonical instrument location, release manifest, and frozen archives."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

from pipeline import codebook_md
from pipeline.codebook import load_codebook
from pipeline.errors import ConfigError
from pipeline.instrument import canonical_files, check_release, load_release

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "docs" / "codebooks" / "archive"
CANONICAL = REPO / "pipeline" / "codebooks"

# Byte hashes recorded by the runs that used these files (verified against runs/*/codebooks snapshots). The archive is evidence; it is moved, never edited.
RECORDED_SHA256 = {
    "stage_b_v2_structured_2026-09-27/episode.draft.md": "fa8043ad123fe8815303c7831405c3ab86bcaffbc277f16af61573a479174b9e",
    "stage_b_v2_structured_2026-09-27/episode.draft.yaml": "1fc35d928edd63f87a5cc215c6241ea8a4fe0eccd1cd17ec1cadf32f0163970c",
    "stage_b_v2_structured_2026-09-27/segmentation.draft.md": "7da6d2b7d0c5c315e74ccc7be307968132f351f2a76d6018887ded0f9892f705",
    "stage_b_v2_structured_2026-09-27/segmentation.draft.yaml": "c04e5d76494f9d30269a57bda42ea1c86de51477f63a355fd093fcf7bda21447",
    "stage_b_v2_structured_2026-09-27/pitch.draft.md": "ae598cb92c704e779c1b9dc5faced27e2c2781227a3f59de2ad939a14262165f",
    "stage_b_v2_structured_2026-09-27/pitch.draft.yaml": "6f3cfdf0737a1ef59a3eb79d9515591e3b5e47a5b3a807eafd39b9215aeade62",
    "stage_b_panel_v3dev8/current_instrument.yaml": "4a7f9c2dda434268ad5ab1adcb9f083aa5bc454a8aec3ace7db36c8e732b5920",
    "stage_b_panel_v3dev8/panel_full.j2": "381fec88167d973a78fd07af9360eac5efeaed88d487189944526941fb87013b",
    "stage_b_panel_v3dev8/stage_a_pitch_factor.md": "e3aca5b66ad4af433650f3ed51d3f3dfdcc9ab813a5aa03c1fa62a981eba1f88",
    "stage_b_panel_v3dev8/stage_a_pitch_factor.yaml": "41db6623b808e182f5ff251117b67548bf529958b472334f9c257348099715d7",
    "stage_b_panel_v3dev8/stage_b_panel.md": "31394b08700fbd8c62ea02f31e1d9d6afbef7e65e3320f6784341677428a6bb4",
    "stage_b_panel_v3dev8/stage_b_panel.yaml": "4a69aa87427aa307fde0422ced69478c12593e549331b1b677a30929ceb2bebd",
    "stage_b_panel_v3dev8/venture_context.md": "691924c9e2ca4a5e550e3fa581b8b8bf0f3f1ac16a681f7a08fb2f7114f67cae",
    "stage_b_panel_v3dev8/venture_context.yaml": "92f0e78fe00fe2cbc1e7a2201fee24d910c9af3bee314beda64a1d433d68956a",
    # recorded as prompt_template_sha256 by the three ranking runs (design.json), not in bundle_sha256
    "stage_b_panel_v3dev8/stage_a_rank.j2": "ef624f1788aa7c94cc4b5d9259e2ec34c805769f70c1bc61141f11a7e3ac5f42",
}
V3DEV8_RANK_TEMPLATE = "stage_b_panel_v3dev8/stage_a_rank.j2"
V3DEV8_RANKINGS = ("ranking", "ranking_first48", "ranking_pilot5")

# Byte pins taken at archive time, not hashes any run recorded: the 2026-10-06 bundles were pure renames (R100) of
# the working copies, and stage_b_panel_v3dev8/stage_a_holistic.j2 was copied with cp on 2026-10-20 (no run hashes
# that template file). See each archive README for which prompt hashes a run recorded.
ARCHIVE_PIN_SHA256 = {
    "stage_a_perceived_delivery_v2/stage_a_perceived_delivery.j2": "63c3a7ace8606b4d23b2faac0458eaab970e32e263d25d087f99a4532ec5cc6f",
    "stage_a_perceived_delivery_v2/stage_a_perceived_delivery.md": "ec3c302ccda9e608ee6e302aa2f1770a2fccd56e1de48947a0d52038117edc04",
    "stage_a_perceived_delivery_v2/stage_a_perceived_delivery.yaml": "bb176aef314e6e81e26f0201def1c5367f16e77ae2a9acae08ad9f9a5d33b933",
    "stage_a_perceived_delivery_v2/stage_a_perceived_delivery_protocol.md": "7d82e0fa9a91e6f54e7e4175a397c37edff8411ce45667a564d752af24e4a425",
    "stage_b_v1_episode/stage_a_perceived_delivery.j2": "63c3a7ace8606b4d23b2faac0458eaab970e32e263d25d087f99a4532ec5cc6f",
    "stage_b_v1_episode/stage_a_perceived_delivery_protocol.md": "7d82e0fa9a91e6f54e7e4175a397c37edff8411ce45667a564d752af24e4a425",
    "stage_b_v1_episode/stage_b_episode.j2": "935986fd568d7f8d6db341e615ab7e9161b3e45df72aab54adf902121016ec2a",
    "stage_b_v1_episode/stage_b_episode.md": "9862f43cb8792b07499bbf695c1262b7dc7b322ea84c0b7bec54cafb5c9f8362",
    "stage_b_v1_episode/stage_b_episode.yaml": "baf9efa423f31d0a7b9c96809e59dc77dfffbe21ffe2aea8b8f1e9a1caa3e027",
    "stage_b_v1_episode/stage_b_segmentation.j2": "3cc4657fa7be08e3a79067cede93a9486fa737f02c35d949ef4cc32e3bf35efe",
    "stage_b_v1_episode/stage_b_segmentation.md": "ff63e77aea7f18e463b3ca18746af8937fe2efdabe9ea0fe95312c43a4eaae4e",
    "stage_b_v1_episode/stage_b_segmentation.yaml": "db113c7595c13ca45b7ba28d18e7f1787d70601256f1849fb9a91f6482ddd8c8",
    "stage_b_v1_episode/stage_b_timeline.j2": "dca8d97ba90d08cb886083cf4c0489623f80c79d6bc950aa525603c790416c4f",
    "stage_b_panel_v3dev8/stage_a_holistic.j2": "5da9001e9db1b4917b88ac0d18a1b4afbe4e877d1f3723336294e01be875cf1f",
}

EXPECTED_ARCHIVE_FILES = {
    "stage_a_perceived_delivery_v2": {
        "README.md", "stage_a_perceived_delivery.md", "stage_a_perceived_delivery.yaml",
        "stage_a_perceived_delivery_protocol.md", "stage_a_perceived_delivery.j2",
    },
    "stage_b_v1_episode": {
        "README.md", "stage_b_episode.md", "stage_b_episode.yaml", "stage_b_segmentation.md",
        "stage_b_segmentation.yaml", "stage_b_episode.j2", "stage_b_segmentation.j2", "stage_b_timeline.j2",
        "stage_a_perceived_delivery.j2", "stage_a_perceived_delivery_protocol.md",
    },
    "stage_b_v2_structured_2026-09-27": {
        "README.md", "episode.draft.md", "episode.draft.yaml", "segmentation.draft.md",
        "segmentation.draft.yaml", "pitch.draft.md", "pitch.draft.yaml", "stage_b_exchanges.j2",
        "EXCHANGE_REVISION_2026-09-27.md", "REVISION_2026-09-24.md", "STRUCTURED_WORKFLOW_2026-09-25.md",
    },
    "stage_b_panel_v3dev8": {
        "README.md", "current_instrument.yaml", "stage_b_panel.md", "stage_b_panel.yaml",
        "stage_a_pitch_factor.md", "stage_a_pitch_factor.yaml", "venture_context.md", "venture_context.yaml",
        "panel_full.j2", "stage_a_holistic.j2", "stage_a_rank.j2",
    },
}


@pytest.mark.parametrize("relative, digest", sorted(RECORDED_SHA256.items()))
def test_archived_file_matches_the_hash_its_runs_recorded(relative, digest):
    assert hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest() == digest


def test_v3dev8_archive_hashes_are_the_ones_every_run_recorded():
    runs = sorted((REPO / "runs" / "old_version_runs" / "stage_b_panel_v3dev8").glob("st_*/experiment.json"))
    assert len(runs) == 100
    recorded = {tuple(sorted(json.loads(p.read_text())["bundle_sha256"].items())) for p in runs}
    assert len(recorded) == 1
    expected = {
        k.split("/", 1)[1]: v for k, v in RECORDED_SHA256.items()
        if k.startswith("stage_b_panel_v3dev8/") and k != V3DEV8_RANK_TEMPLATE
    }
    assert len(expected) == 8
    assert dict(next(iter(recorded))) == expected


@pytest.mark.parametrize("ranking", V3DEV8_RANKINGS)
def test_v3dev8_rank_template_archive_is_the_one_every_ranking_run_recorded(ranking):
    design = json.loads((REPO / "runs" / "old_version_runs" / "stage_b_panel_v3dev8" / ranking / "design.json").read_text())
    assert design["prompt_template_sha256"] == RECORDED_SHA256[V3DEV8_RANK_TEMPLATE]
    assert hashlib.sha256((ARCHIVE / V3DEV8_RANK_TEMPLATE).read_bytes()).hexdigest() == design["prompt_template_sha256"]


@pytest.mark.parametrize("relative, digest", sorted(ARCHIVE_PIN_SHA256.items()))
def test_archived_file_matches_its_archive_time_pin(relative, digest):
    assert hashlib.sha256((ARCHIVE / relative).read_bytes()).hexdigest() == digest


def test_every_non_readme_file_in_the_two_bundles_is_pinned():
    for release in ("stage_a_perceived_delivery_v2", "stage_b_v1_episode"):
        on_disk = {f"{release}/{p.name}" for p in (ARCHIVE / release).iterdir() if p.name != "README.md"}
        assert on_disk == {k for k in ARCHIVE_PIN_SHA256 if k.startswith(release + "/")}


@pytest.mark.parametrize("name", ["stage_a_perceived_delivery.j2", "stage_a_perceived_delivery_protocol.md"])
def test_v1_episode_copies_are_byte_equal_to_the_stage_a_originals(name):
    original = (ARCHIVE / "stage_a_perceived_delivery_v2" / name).read_bytes()
    assert (ARCHIVE / "stage_b_v1_episode" / name).read_bytes() == original


@pytest.mark.parametrize("release, names", sorted(EXPECTED_ARCHIVE_FILES.items()))
def test_archive_release_contains_exactly_its_bundle(release, names):
    assert {p.name for p in (ARCHIVE / release).iterdir() if p.name != "__pycache__"} == names


def test_nothing_superseded_remains_in_the_canonical_directory():
    canonical = {p.name for p in (REPO / "pipeline" / "codebooks").iterdir()}
    for stale in ("stage_a_pitch_factor.md", "stage_a_pitch_factor.yaml", "stage_b_episode.md", "stage_b_segmentation.md"):
        assert stale not in canonical
    assert not (REPO / "docs" / "codebooks" / "drafts" / "stage_b").exists()
    assert not (REPO / "docs" / "codebooks" / "stage_a_perceived_delivery_protocol.md").exists()
    assert (REPO / "docs" / "codebooks" / "REFERENCES.md").exists()


def test_repo_manifest_loads_and_is_ready():
    release = load_release()
    assert release.release_id == "v3-dev9" and release.status in ("development", "frozen")
    assert "stage_a" in release.books
    assert release.books["stage_a"].yaml == CANONICAL / "stage_a_perceived_delivery.yaml"
    archives = yaml.safe_load((CANONICAL / "current_instrument.yaml").read_text())["archives"]
    assert "../../docs/codebooks/archive/stage_b_panel_v3dev8" in archives
    assert all((CANONICAL / a).resolve().is_dir() for a in archives)
    assert check_release(release) == []


def test_canonical_directory_holds_exactly_the_manifest_files():
    release = load_release()
    assert {p.name for p in CANONICAL.iterdir() if p.name != "__pycache__"} == canonical_files(release)


def _write_release(tmp_path, extra_files=(), stale=False):
    canonical = tmp_path / "codebooks"
    templates = tmp_path / "prompts"
    canonical.mkdir(); templates.mkdir()
    (templates / "stage_a_perceived_delivery.j2").write_text("{% for e in entries %}{{ e.id }}{% endfor %}")
    md = canonical / "stage_a_perceived_delivery.md"
    md.write_text((CANONICAL / "stage_a_perceived_delivery.md").read_text())
    from pipeline.codebook_md import write_yaml
    write_yaml(md)
    if stale:
        md.write_text(md.read_text() + "\n<!-- edited after sync -->\n")
    for name in extra_files:
        (canonical / name).write_text("x")
    manifest = canonical / "current_instrument.yaml"
    manifest.write_text(yaml.safe_dump({
        "release_id": "test-release", "status": "development", "templates_dir": "../prompts",
        "books": {"stage_a": {"markdown": "stage_a_perceived_delivery.md", "yaml": "stage_a_perceived_delivery.yaml",
                              "template": "stage_a_perceived_delivery.j2",
                              "schema": "pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord"}},
    }))
    return manifest


def test_release_check_passes_on_a_complete_bundle(tmp_path):
    assert check_release(load_release(_write_release(tmp_path))) == []


def test_release_check_names_an_unexpected_file_in_the_canonical_directory(tmp_path):
    problems = check_release(load_release(_write_release(tmp_path, extra_files=("stage_b_episode.md",))))
    assert any("unexpected file" in p and "stage_b_episode.md" in p for p in problems)


def test_release_check_names_a_stale_generated_yaml(tmp_path):
    problems = check_release(load_release(_write_release(tmp_path, stale=True)))
    assert any("stale" in p and "stage_a_perceived_delivery.yaml" in p for p in problems)


def test_release_check_names_a_missing_template_and_bad_schema(tmp_path):
    manifest = _write_release(tmp_path)
    data = yaml.safe_load(manifest.read_text())
    data["books"]["stage_a"]["template"] = "missing.j2"
    data["books"]["stage_a"]["schema"] = "pipeline.schemas.nowhere.Nothing"
    manifest.write_text(yaml.safe_dump(data))
    problems = check_release(load_release(manifest))
    assert any("template" in p and "missing.j2" in p for p in problems)
    assert any("schema" in p for p in problems)


def test_manifest_with_unknown_status_is_rejected(tmp_path):
    manifest = _write_release(tmp_path)
    data = yaml.safe_load(manifest.read_text()); data["status"] = "maybe"
    manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError):
        load_release(manifest)


def _load_sync_script():
    spec = importlib.util.spec_from_file_location("codebook_sync", REPO / "scripts" / "codebook_sync.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["codebook_sync"] = module
    spec.loader.exec_module(module)
    return module


def _frozen_pair(tmp_path, monkeypatch):
    archive = tmp_path / "archive" / "old_release"
    archive.mkdir(parents=True)
    md = archive / "book.md"
    md.write_text((CANONICAL / "stage_a_perceived_delivery.md").read_text())
    codebook_md.write_yaml(md)
    monkeypatch.setattr(codebook_md, "ARCHIVE_ROOT", tmp_path / "archive")
    return md


def test_loading_an_in_sync_archived_codebook_works(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    assert len(load_codebook(md.with_suffix(".yaml")).ids) == 3


def test_loading_a_stale_archived_codebook_refuses_to_regenerate(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    yaml_before = md.with_suffix(".yaml").read_text()
    md.write_text(md.read_text() + "\n<!-- edited -->\n")
    with pytest.raises(ConfigError, match="frozen"):
        load_codebook(md.with_suffix(".yaml"))
    assert md.with_suffix(".yaml").read_text() == yaml_before


def test_write_yaml_refuses_a_target_under_the_archive(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    yaml_before = md.with_suffix(".yaml").read_text()
    with pytest.raises(ConfigError, match="frozen"):
        codebook_md.write_yaml(md)
    assert md.with_suffix(".yaml").read_text() == yaml_before


def test_a_missing_yaml_under_the_archive_is_never_generated(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    md.with_suffix(".yaml").unlink()
    with pytest.raises(ConfigError, match="frozen"):
        load_codebook(md.with_suffix(".yaml"))
    assert not md.with_suffix(".yaml").exists()


def test_a_hand_edited_archived_yaml_gets_no_force_advice(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    yaml_path = md.with_suffix(".yaml")
    yaml_path.write_text(yaml_path.read_text() + "# tampered\n")
    with pytest.raises(ConfigError, match="frozen") as err:
        load_codebook(yaml_path)
    assert "--force" not in str(err.value)


def test_sync_script_never_writes_under_the_archive(tmp_path, monkeypatch):
    md = _frozen_pair(tmp_path, monkeypatch)
    sync = _load_sync_script()
    monkeypatch.setattr(sync, "ARCHIVE_DIR", tmp_path / "archive")
    md.write_text(md.read_text() + "\n<!-- edited -->\n")
    yaml_before = md.with_suffix(".yaml").read_text()
    assert sync.process(md, check=False, force=True, no_illustrative=False) == 1
    assert md.with_suffix(".yaml").read_text() == yaml_before
    assert sync.process(md, check=True, force=False, no_illustrative=False) == 1


def test_sync_script_check_passes_on_the_repository():
    sync = _load_sync_script()
    total = sum(sync.process(md, check=True, force=False, no_illustrative=False) for md in sync.managed_markdown())
    assert total == 0
    assert all(not codebook_md.is_archived(md) for md in sync.writable_markdown())


def test_release_directory_without_the_review_files_loads_with_none(tmp_path):
    release = load_release(_write_release(tmp_path))
    assert release.review_labels is None and release.validation_thresholds is None
    assert release.validation_reporting is None
    assert check_release(release) == []


def test_review_files_beside_the_manifest_are_found_by_name(tmp_path):
    manifest = _write_release(tmp_path)
    (manifest.parent / "review_labels.yaml").write_text("questions: {}\n")
    (manifest.parent / "validation_thresholds.yaml").write_text("fields: {}\n")
    (manifest.parent / "validation_reporting.yaml").write_text("kinds: {}\n")
    release = load_release(manifest)
    assert release.review_labels == (manifest.parent / "review_labels.yaml").resolve()
    assert release.validation_thresholds == (manifest.parent / "validation_thresholds.yaml").resolve()
    assert release.validation_reporting == (manifest.parent / "validation_reporting.yaml").resolve()
    review_files = {"review_labels.yaml", "validation_thresholds.yaml", "validation_reporting.yaml"}
    assert review_files <= canonical_files(release)
    # none is part of what a run copies and hashes: no model pass reads them
    assert not review_files & canonical_files(release, optional=False)
    assert check_release(release) == []


# The manifest is hashed into every run's bundle_sha256 and a run refuses to resume when it differs: it must not
# change without starting a new run. The v3-dev8 runs recorded the archived copy's hash (RECORDED_SHA256 above);
# this is the sha256 of release v3-dev9 (three-item books.stage_a, v3-dev8 bundle archived; 2026-10-20).
MANIFEST_SHA256 = "041c3168e92b4604203aeba059a0700acbadc8a6d2a320a50bfd92a04ddd4975"


def test_the_manifest_is_byte_identical_to_its_pinned_hash():
    assert hashlib.sha256((CANONICAL / "current_instrument.yaml").read_bytes()).hexdigest() == MANIFEST_SHA256


def test_the_manifest_differs_from_the_v3dev8_one_only_in_release_id_archives_and_the_stage_a_book():
    live = yaml.safe_load((CANONICAL / "current_instrument.yaml").read_text())
    v3dev8 = yaml.safe_load((ARCHIVE / "stage_b_panel_v3dev8" / "current_instrument.yaml").read_text())
    assert (live.pop("release_id"), v3dev8.pop("release_id")) == ("v3-dev9", "v3-dev8")
    assert live.pop("archives") == v3dev8.pop("archives") + ["../../docs/codebooks/archive/stage_b_panel_v3dev8"]
    assert live["books"].pop("stage_a") == {
        "markdown": "stage_a_perceived_delivery.md", "yaml": "stage_a_perceived_delivery.yaml",
        "template": "stage_a_perceived_delivery.j2",
        "schema": "pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord"}
    assert v3dev8["books"].pop("stage_a")["markdown"] == "stage_a_pitch_factor.md"
    assert live == v3dev8
