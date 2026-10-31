import json

import pytest

from pipeline.ingest import (
    discover_videos,
    ffprobe,
    ingest_corpus,
    read_metadata,
    sha256_file,
)
from pipeline.registry import Registry
from tests.conftest import needs_ffmpeg


@needs_ffmpeg
def test_ffprobe_reads_duration_and_audio(video_factory):
    path = video_factory("vidA.mp4", seconds=2.0, with_audio=True)
    probe = ffprobe(path)
    assert 1.5 < probe.duration_s < 2.6
    assert (probe.width, probe.height) == (128, 96)
    assert probe.has_audio is True


@needs_ffmpeg
def test_ffprobe_detects_missing_audio(video_factory):
    probe = ffprobe(video_factory("silent.mp4", with_audio=False))
    assert probe.has_audio is False


def test_sha256_is_content_based(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert sha256_file(a) == sha256_file(b)
    b.write_bytes(b"different")
    assert sha256_file(a) != sha256_file(b)


def test_read_metadata_keys_by_video_id(tmp_path):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("video_id,platform,url\nabc,youtube,https://x\n")
    meta = read_metadata(csv_path)
    assert meta["abc"]["url"] == "https://x"


@needs_ffmpeg
def test_discover_matches_ids_and_skips_partials(tmp_path, video_factory):
    video_factory("abc.mp4")
    video_factory("ghi.f137.mp4")
    (tmp_path / "videos" / "def.f137.mp4.part").write_bytes(b"partial")
    (tmp_path / "videos" / "jkl.mkv").write_bytes(b"not in csv")
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "video_id,platform,url\nabc,youtube,https://x\ndef,youtube,https://y\nghi,youtube,https://z\n"
    )
    found = discover_videos(tmp_path / "videos", read_metadata(csv_path))
    assert [vid for vid, _ in found] == ["abc", "ghi"]


@needs_ffmpeg
def test_mp4_wins_when_several_containers_share_an_id(tmp_path, video_factory):
    video_factory("abc.mp4")
    (tmp_path / "videos" / "abc.mkv").write_bytes(b"other container")
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("video_id,platform,url\nabc,youtube,https://x\n")
    found = discover_videos(tmp_path / "videos", read_metadata(csv_path))
    assert [(vid, path.suffix) for vid, path in found] == [("abc", ".mp4")]


def test_read_metadata_rejects_csv_without_video_id_column(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("id,platform,url\nabc,youtube,https://x\n")
    with pytest.raises(ValueError, match="video_id"):
        read_metadata(csv_path)


@needs_ffmpeg
def test_ingest_registers_videos_and_is_idempotent(tmp_path, video_factory, monkeypatch):
    import textwrap

    from pipeline.config import load_config

    video_factory("abc.mp4")
    (tmp_path / "meta.csv").write_text("video_id,platform,url\nabc,youtube,https://x\n")
    (tmp_path / "book.yaml").write_text("[]\n")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "holistic.j2").write_text("hi")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent("""
        run_name: t
        corpus: {name: hu-ma, video_dir: videos, metadata_csv: meta.csv}
        model: {name: gemini-x}
        passes:
          - {name: holistic, unit: PITCH, prompt: holistic.j2, schema: pipeline.schemas.stage_a.StageARecord}
        codebook: book.yaml
    """))
    cfg = load_config(cfg_path, prompt_dir=prompts)
    reg = Registry(tmp_path / "registry.sqlite")

    first = ingest_corpus(cfg, reg)
    assert (first.discovered, first.registered) == (1, 1)
    row = reg.get_video("abc")
    assert row.probe_ok is True and row.corpus == "hu-ma"
    assert json.loads(row.meta_json)["url"] == "https://x"

    second = ingest_corpus(cfg, reg)
    assert (second.registered, second.skipped) == (0, 1)
    reg.close()


def test_unreadable_video_is_recorded_as_data_not_raised(tmp_path):
    import json as _json
    import textwrap

    from pipeline.config import load_config

    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "broken.mp4").write_bytes(b"this is not a video")
    (tmp_path / "meta.csv").write_text("video_id,platform,url\nbroken,youtube,https://b\n")
    (tmp_path / "book.yaml").write_text("[]\n")
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "holistic.j2").write_text("hi")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(textwrap.dedent("""
        run_name: t
        corpus: {name: hu-ma, video_dir: videos, metadata_csv: meta.csv}
        model: {name: gemini-x}
        passes:
          - {name: holistic, unit: PITCH, prompt: holistic.j2, schema: pipeline.schemas.stage_a.StageARecord}
        codebook: book.yaml
    """))
    cfg = load_config(cfg_path, prompt_dir=prompts)
    reg = Registry(tmp_path / "registry.sqlite")

    summary = ingest_corpus(cfg, reg)  # must not raise
    assert (summary.discovered, summary.registered, summary.failed) == (1, 0, 1)
    row = reg.get_video("broken")
    assert row.probe_ok is False
    assert "probe_error" in _json.loads(row.meta_json)
    assert row.duration_s is None
    reg.close()
