import csv
import unicodedata

import yaml

from pipeline.config import load_config
from pipeline.ingest import (
    build_metadata_rows, discover_videos, ingest_corpus, read_metadata,
    video_id_for_filename, write_metadata_csv,
)
from pipeline.registry import Registry
from tests.conftest import make_test_video, needs_ffmpeg

HOSTILE = '"It Only Holds Three Cans?" Café | Shark Tank US | Global.mp4'


def test_video_id_is_stable_shell_safe_and_normalisation_proof():
    vid = video_id_for_filename(HOSTILE)
    assert vid == video_id_for_filename(HOSTILE)
    assert vid.startswith("st_") and len(vid) == 13 and vid[3:].isalnum()
    nfd = unicodedata.normalize("NFD", HOSTILE)
    assert nfd != HOSTILE and video_id_for_filename(nfd) == vid


def test_metadata_rows_cover_videos_only(tmp_path):
    for name in (HOSTILE, "plain.mp4", "notes.txt", "half.mp4.part"):
        (tmp_path / name).write_bytes(b"x")
    rows = build_metadata_rows(tmp_path)
    assert sorted(r["filename"] for r in rows) == sorted([HOSTILE, "plain.mp4"])
    row = next(r for r in rows if r["filename"] == HOSTILE)
    assert row["title"] == HOSTILE[:-4] and row["video_id"] == video_id_for_filename(HOSTILE)


def test_csv_roundtrip_and_filename_mode_discovery(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in (HOSTILE, "plain.mp4", "unlisted.mp4"):
        (videos / name).write_bytes(b"x")
    out = tmp_path / "clips.csv"
    assert write_metadata_csv(videos, out) == 3
    with open(out, newline="") as handle:
        assert {r["filename"] for r in csv.DictReader(handle)} == {HOSTILE, "plain.mp4", "unlisted.mp4"}
    metadata = read_metadata(out)
    del metadata[video_id_for_filename("unlisted.mp4")]
    found = dict(discover_videos(videos, metadata))
    assert found == {
        video_id_for_filename(HOSTILE): videos / HOSTILE,
        video_id_for_filename("plain.mp4"): videos / "plain.mp4",
    }


@needs_ffmpeg
def test_too_long_videos_are_registered_but_flagged(tmp_path):
    videos = tmp_path / "videos"
    make_test_video(videos / "short.mp4", seconds=1.0)
    make_test_video(videos / "long.mp4", seconds=4.0)
    write_metadata_csv(videos, tmp_path / "clips.csv")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "a.j2").write_text("hello")

    def config(max_duration_s):
        (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({
            "run_name": "t",
            "corpus": {"name": "st", "video_dir": "videos", "metadata_csv": "clips.csv",
                       "max_duration_s": max_duration_s},
            "model": {"name": "m"},
            "passes": [{"name": "a", "unit": "PITCH", "prompt": "a.j2",
                        "schema": "pipeline.schemas.stage_a.StageARecord"}],
        }))
        return load_config(tmp_path / "cfg.yaml", prompt_dir=prompts)

    registry = Registry(tmp_path / "r.sqlite")
    summary = ingest_corpus(config(2.0), registry)
    assert (summary.registered, summary.too_long) == (2, 1)
    long_id, short_id = video_id_for_filename("long.mp4"), video_id_for_filename("short.mp4")
    assert registry.get_video(long_id).skip_reason == "too_long"
    assert registry.get_video(short_id).skip_reason is None
    assert [v.video_id for v in registry.list_videos("st", include_skipped=False)] == [short_id]

    # Raising the limit later clears the flag without re-probing.
    summary = ingest_corpus(config(10.0), registry)
    assert (summary.skipped, summary.too_long) == (2, 0)
    assert registry.get_video(long_id).skip_reason is None
    registry.close()
