import pandas as pd
import pytest

from pipeline.aggregate import build_tables
from pipeline.annotate import run_pass
from pipeline.providers.fake import FakeProvider
from tests.conftest import needs_ffmpeg
from tests.stage_b_helpers import VIDEO, make_project

ALL = ["timeline", "segmentation", "prepared", "episode"]


@pytest.fixture
def tables(tmp_path):
    cfg, registry, run, fixtures = make_project(tmp_path, ALL)
    provider = FakeProvider(fixtures)
    for name in ("timeline", "segmentation"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    registry.set_frozen(VIDEO, True)
    for name in ("prepared", "episode"):
        run_pass(cfg, cfg.get_pass(name), registry, run, provider, progress=False)
    built = {name: pd.read_parquet(path) for name, path in build_tables(cfg, registry, run).items()}
    registry.close()
    return built


@needs_ffmpeg
def test_annotations_hold_both_stages_shapes(tables):
    ann = tables["annotations"]
    assert len(ann) == 3 + 2 * 4                        # prepared: 3 codes; 2 episodes x 4 codes
    frame = ann[(ann.code_id == "PR-FRAME") & (ann.episode_idx == 0)].iloc[0]
    assert pd.isna(frame["value"]) and frame["value_label"] == "prevention"
    evid = ann[(ann.code_id == "RS-EVID") & (ann.episode_idx == 0)].iloc[0]
    assert evid["value"] == 1.0 and pd.isna(evid["value_label"])
    interrupt = ann[ann.code_id == "RAP-INTERRUPT"].iloc[0]
    assert bool(interrupt["uncodable"]) and pd.isna(interrupt["value"])
    prepared = ann[ann["pass"] == "prepared"]
    assert set(prepared.episode_idx) == {-1} and set(prepared.value) == {4.0}
    assert prepared["turn_refs"].isna().all() and prepared["turn_refs_valid"].isna().all()


@needs_ffmpeg
def test_turn_refs_are_checked_against_the_episode(tables):
    ann = tables["annotations"]
    subs = ann[ann.code_id == "PREP-SUBS"].set_index("episode_idx")
    assert subs.loc[0, "turn_refs"] == "[2]" and bool(subs.loc[0, "turn_refs_valid"])   # turns 2-4
    assert not bool(subs.loc[1, "turn_refs_valid"])                                      # turns 5-7
    empty = ann[(ann.code_id == "RAP-INTERRUPT") & (ann.episode_idx == 0)].iloc[0]
    assert empty["turn_refs"] == "[]" and bool(empty["turn_refs_valid"])                 # vacuously valid


@needs_ffmpeg
def test_structural_tables(tables):
    assert set(tables.keys()) == {"annotations", "videos", "turns", "episodes", "decisions"}
    turns = tables["turns"]
    assert len(turns) == 8 and turns.iloc[2].to_dict() == {
        "video_id": VIDEO, "turn_idx": 2, "speaker": "speaker-2", "role": "investor",
        "start_s": 4.0, "end_s": 5.0, "tag": "asks about margins"}
    episodes = tables["episodes"]
    assert list(episodes.kind) == ["prepared", "episode", "episode"]
    assert list(episodes.topic)[1:] == ["gross margins", "valuation"]
    assert bool(episodes.frozen.all())
    video = tables["videos"].iloc[0]
    assert (video["n_pitches"], video["prepared_end_s"], video["n_episodes"]) == (1, 4.0, 2)
    assert video["skip_reason"] is None


@needs_ffmpeg
def test_decisions_table_shape(tables):
    decisions = tables["decisions"]
    assert len(decisions) == 2  # one row per shark per video
    assert list(decisions.columns) == ["video_id", "speaker", "decision", "turn", "terms"]
    assert decisions.set_index("speaker").loc["speaker-2"].to_dict() == {
        "video_id": VIDEO, "decision": "offer", "turn": 4, "terms": "100k for ten percent"}
    out_row = decisions.set_index("speaker").loc["speaker-3"]
    assert (out_row["decision"], out_row["turn"]) == ("out", 7)
    assert pd.isna(out_row["terms"])


@needs_ffmpeg
def test_videos_table_gains_deal_and_offer_counts(tables):
    video = tables["videos"].iloc[0]
    assert video["deal"] == True  # noqa: E712 - pandas boolean, `is True` can mis-compare
    assert (video["n_offers"], video["n_out"]) == (1, 1)


@needs_ffmpeg
def test_dev_set_membership_is_a_column(tmp_path):
    import yaml

    cfg, registry, run, _ = make_project(tmp_path, ["timeline"])
    (tmp_path / "dev.txt").write_text(f"{VIDEO}\n")
    data = yaml.safe_load((tmp_path / "cfg.yaml").read_text())
    data["corpus"]["dev_set"] = "dev.txt"
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(data))
    from pipeline.config import load_config

    videos = pd.read_parquet(build_tables(load_config(tmp_path / "cfg.yaml"), registry, run)["videos"])
    assert list(videos.dev_set) == [True]
    registry.close()
