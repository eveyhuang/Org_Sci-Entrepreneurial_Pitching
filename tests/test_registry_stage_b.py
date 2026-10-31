import sqlite3

import pytest

from pipeline.registry import (
    PREPARED_IDX,
    DecisionRow,
    EpisodeRow,
    Registry,
    VideoRow,
    WorkUnit,
    make_unit_id,
)

OLD_SCHEMA = """
CREATE TABLE videos (video_id TEXT PRIMARY KEY, corpus TEXT NOT NULL, path TEXT NOT NULL,
    sha256 TEXT, duration_s REAL, width INTEGER, height INTEGER, has_audio INTEGER,
    probe_ok INTEGER, meta_json TEXT);
CREATE TABLE episodes (video_id TEXT NOT NULL, episode_idx INTEGER NOT NULL, start_s REAL NOT NULL,
    end_s REAL NOT NULL, source_unit_id TEXT, frozen INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (video_id, episode_idx));
"""

# Pre-6T work_units schema (no thinking_tokens column), copied from registry.py's SCHEMA
# before this migration was added.
OLD_WORK_UNITS_SCHEMA = """
CREATE TABLE work_units (
    unit_id       TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL,
    video_id      TEXT NOT NULL,
    pass_name     TEXT NOT NULL,
    span_start_s  REAL,
    span_end_s    REAL,
    prompt_hash   TEXT NOT NULL,
    model         TEXT NOT NULL,
    media         TEXT NOT NULL,
    k             INTEGER NOT NULL,
    status        TEXT NOT NULL,
    error_class   TEXT,
    error_detail  TEXT,
    raw_path      TEXT,
    parsed_path   TEXT,
    input_tokens  INTEGER,
    output_tokens INTEGER,
    finished_at   TEXT
);
"""


@pytest.fixture
def reg(tmp_path):
    registry = Registry(tmp_path / "r.sqlite")
    yield registry
    registry.close()


def test_migration_preserves_existing_rows_and_is_idempotent(tmp_path):
    db = tmp_path / "old.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(OLD_SCHEMA)
        conn.execute("INSERT INTO videos VALUES ('v1','hu-ma','/x.mp4','abc',12.5,640,360,1,1,'{}')")
        conn.execute("INSERT INTO episodes VALUES ('v1',0,1.0,2.0,'u1',1)")
    for _ in range(2):  # opening twice must not fail on already-added columns
        registry = Registry(db)
        video = registry.get_video("v1")
        assert (video.duration_s, video.sha256, video.skip_reason) == (12.5, "abc", None)
        assert video.deal is None
        episode = registry.episodes_for("v1")[0]
        assert (episode.start_s, episode.frozen, episode.kind, episode.topic) == (1.0, True, "episode", None)
        registry.close()


def test_work_units_migration_adds_thinking_tokens_and_keeps_rows(tmp_path):
    db = tmp_path / "old_units.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(OLD_WORK_UNITS_SCHEMA)
        conn.execute(
            """INSERT INTO work_units
               (unit_id, run_id, video_id, pass_name, span_start_s, span_end_s, prompt_hash,
                model, media, k, status, input_tokens, output_tokens)
               VALUES ('u1','r1','v1','holistic',NULL,NULL,'ph','m','video',0,'done',10,20)"""
        )
    for _ in range(2):  # opening twice must not fail on an already-added column
        registry = Registry(db)
        unit = registry.get_unit("u1")
        assert (unit.input_tokens, unit.output_tokens, unit.thinking_tokens) == (10, 20, None)
        registry.close()


def test_mark_done_records_and_defaults_thinking_tokens(reg):
    unit = WorkUnit(
        unit_id=make_unit_id(video_id="v", pass_name="holistic", prompt_hash="ph",
                              model="m", media="video", k=0),
        run_id="r", video_id="v", pass_name="holistic", prompt_hash="ph",
        model="m", media="video", k=0,
    )
    reg.enqueue(unit)
    reg.mark_done(unit.unit_id, "raw", "parsed", 1, 2, thinking_tokens=77)
    assert reg.get_unit(unit.unit_id).thinking_tokens == 77

    other = WorkUnit(
        unit_id=make_unit_id(video_id="v2", pass_name="holistic", prompt_hash="ph",
                              model="m", media="video", k=0),
        run_id="r", video_id="v2", pass_name="holistic", prompt_hash="ph",
        model="m", media="video", k=0,
    )
    reg.enqueue(other)
    reg.mark_done(other.unit_id, "raw", "parsed", 1, 2)  # no thinking_tokens supplied
    assert reg.get_unit(other.unit_id).thinking_tokens is None


def test_skip_reason_roundtrip_and_filter(reg):
    reg.upsert_video(VideoRow("a", "st", "/a.mp4", probe_ok=True))
    reg.upsert_video(VideoRow("b", "st", "/b.mp4", probe_ok=True, skip_reason="too_long"))
    assert [v.video_id for v in reg.list_videos("st")] == ["a", "b"]
    assert [v.video_id for v in reg.list_videos("st", include_skipped=False)] == ["a"]
    reg.set_skip_reason("b", None)
    assert reg.get_video("b").skip_reason is None


def rows(video_id="v"):
    return [
        EpisodeRow(video_id, PREPARED_IDX, 0.0, 4.0, kind="prepared", start_turn=0, end_turn=1),
        EpisodeRow(video_id, 0, 4.0, 7.0, probing_speaker="speaker-2", start_turn=2, end_turn=4,
                   cut=False, topic="margins", resolution="resolved"),
        EpisodeRow(video_id, 1, 7.0, 12.0, probing_speaker="speaker-3", start_turn=5, end_turn=7,
                   cut=True, topic="valuation", resolution="conceded"),
    ]


def test_replace_unfrozen_then_kinds_and_metadata(reg):
    reg.replace_unfrozen_episodes("v", rows())
    reg.replace_unfrozen_episodes("v", rows()[:2])  # a re-segmentation replaces, never appends
    assert reg.episodes_for("v") == []  # nothing frozen yet
    unfrozen = reg.episodes_for("v", frozen_only=False)
    assert [(e.episode_idx, e.topic, e.cut) for e in unfrozen] == [(0, "margins", False)]
    prepared = reg.episodes_for("v", frozen_only=False, kind="prepared")
    assert [(e.episode_idx, e.end_s) for e in prepared] == [(PREPARED_IDX, 4.0)]
    assert len(reg.episodes_for("v", frozen_only=False, kind=None)) == 2
    assert reg.get_episode("v", 0).probing_speaker == "speaker-2"
    assert reg.get_episode("v", 9) is None


def test_frozen_rows_are_never_overwritten(reg):
    reg.replace_unfrozen_episodes("v", rows())
    assert reg.has_frozen_episodes("v") is False
    assert reg.set_frozen("v", True) == 3
    assert reg.has_frozen_episodes("v") is True
    with pytest.raises(ValueError, match="frozen"):
        reg.replace_unfrozen_episodes("v", rows()[:1])
    assert len(reg.episodes_for("v", kind=None)) == 3
    reg.set_frozen("v", False)
    reg.replace_unfrozen_episodes("v", rows()[:1])
    assert len(reg.episodes_for("v", frozen_only=False, kind=None)) == 1


def unit(run_id, prompt_hash, k=0):
    uid = make_unit_id(video_id="v", pass_name="transcript", prompt_hash=prompt_hash,
                       model="m", media="full", k=k)
    return WorkUnit(unit_id=uid, run_id=run_id, video_id="v", pass_name="transcript",
                    prompt_hash=prompt_hash, model="m", media="full", k=k)


def test_latest_done_unit_spans_runs_and_prefers_the_newest(reg):
    assert reg.latest_done_unit("v", "transcript") is None
    old, new, replicate = unit("run1", "p1"), unit("run2", "p2"), unit("run2", "p2", k=1)
    for u in (old, new, replicate):
        reg.enqueue(u)
    reg.mark_done(old.unit_id, "raw", "parsed_old", 1, 1)
    assert reg.latest_done_unit("v", "transcript").parsed_path == "parsed_old"
    reg.mark_done(new.unit_id, "raw", "parsed_new", 1, 1)
    reg.mark_done(replicate.unit_id, "raw", "parsed_k1", 1, 1)
    assert reg.latest_done_unit("v", "transcript").parsed_path == "parsed_new"
    assert reg.latest_done_unit("v", "transcript", k=1).parsed_path == "parsed_k1"


# --- Revision R2: investor decisions are outcomes, not episodes ------------------------


def decision_rows(video_id="v"):
    return [
        DecisionRow(video_id, "speaker-2", "offer", turn=4, terms="100k for ten percent",
                    source_unit_id="u1"),
        DecisionRow(video_id, "speaker-3", "out", turn=7, source_unit_id="u1"),
    ]


def test_replace_decisions_then_read_back(reg):
    reg.upsert_video(VideoRow("v", "st", "/v.mp4", probe_ok=True))
    reg.replace_decisions("v", decision_rows())
    rows = reg.decisions_for("v")
    assert [(r.speaker, r.decision, r.turn, r.terms) for r in rows] == [
        ("speaker-2", "offer", 4, "100k for ten percent"),
        ("speaker-3", "out", 7, None),
    ]
    assert reg.decisions_for("v")[0].source_unit_id == "u1"


def test_replace_decisions_replaces_not_appends(reg):
    reg.upsert_video(VideoRow("v", "st", "/v.mp4", probe_ok=True))
    reg.replace_decisions("v", decision_rows())
    reg.replace_decisions("v", decision_rows()[:1])
    assert [r.speaker for r in reg.decisions_for("v")] == ["speaker-2"]


def test_decisions_are_not_subject_to_freeze(reg):
    # Decisions are outcomes, not boundaries: replace_decisions has no freeze guard,
    # unlike replace_unfrozen_episodes.
    reg.upsert_video(VideoRow("v", "st", "/v.mp4", probe_ok=True))
    reg.replace_unfrozen_episodes("v", rows())
    reg.set_frozen("v", True)
    reg.replace_decisions("v", decision_rows())  # must not raise
    assert len(reg.decisions_for("v")) == 2


def test_set_deal_roundtrip(reg):
    reg.upsert_video(VideoRow("v", "st", "/v.mp4", probe_ok=True))
    assert reg.get_video("v").deal is None
    reg.set_deal("v", True)
    assert reg.get_video("v").deal is True
    reg.set_deal("v", False)
    assert reg.get_video("v").deal is False
    reg.set_deal("v", None)
    assert reg.get_video("v").deal is None
