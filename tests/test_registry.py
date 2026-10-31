from datetime import datetime, timedelta, timezone

import pytest

from pipeline.registry import Registry, VideoRow, WorkUnit, make_unit_id


@pytest.fixture
def reg(tmp_path):
    r = Registry(tmp_path / "registry.sqlite")
    yield r
    r.close()


def unit(**overrides) -> WorkUnit:
    base = dict(
        unit_id=make_unit_id(
            video_id="vid1", pass_name="holistic", prompt_hash="abc123",
            model="gemini-x", media="full", k=0,
        ),
        run_id="run_1", video_id="vid1", pass_name="holistic",
        span_start_s=None, span_end_s=None, prompt_hash="abc123",
        model="gemini-x", media="full", k=0, status="pending",
    )
    base.update(overrides)
    return WorkUnit(**base)


def test_unit_id_is_deterministic_and_content_addressed():
    args = dict(video_id="v", pass_name="p", prompt_hash="h", model="m", media="full", k=0)
    assert make_unit_id(**args) == make_unit_id(**args)
    for field_name, other in [
        ("video_id", "v2"), ("pass_name", "p2"), ("prompt_hash", "other"),
        ("model", "m2"), ("media", "full2"), ("k", 1),
    ]:
        assert make_unit_id(**{**args, field_name: other}) != make_unit_id(**args), field_name
    assert make_unit_id(**args, span=(0.0, 10.0)) != make_unit_id(**args)
    assert make_unit_id(**args, span=(0.0, 10.0)) != make_unit_id(**args, span=(0.0, 11.0))


def test_enqueue_is_idempotent(reg):
    u = unit()
    assert reg.enqueue(u) is True
    assert reg.enqueue(u) is False
    assert len(reg.pending_units("run_1")) == 1


def test_done_units_are_not_reenqueued_or_returned_as_pending(reg):
    u = unit()
    reg.enqueue(u)
    reg.mark_done(u.unit_id, raw_path="raw/a.json", parsed_path="parsed/a.json",
                  input_tokens=10, output_tokens=20)
    assert reg.enqueue(u) is False
    assert reg.pending_units("run_1") == []
    assert reg.get_unit(u.unit_id).status == "done"


def test_failed_units_returned_only_with_retry_failed(reg):
    u = unit()
    reg.enqueue(u)
    reg.mark_failed(u.unit_id, error_class="RETRY_EXHAUSTED", error_detail="503 x6")
    assert reg.pending_units("run_1") == []
    retried = reg.pending_units("run_1", include_failed=True)
    assert [x.unit_id for x in retried] == [u.unit_id]
    assert reg.failure_counts("run_1") == {"RETRY_EXHAUSTED": 1}


def test_status_counts(reg):
    a, b = unit(), unit(unit_id="other", k=1)
    reg.enqueue(a)
    reg.enqueue(b)
    reg.mark_done(a.unit_id, "r", "p", 1, 2)
    assert reg.status_counts("run_1") == {"done": 1, "pending": 1}


def test_video_roundtrip(reg):
    row = VideoRow(
        video_id="vid1", corpus="hu-ma", path="/tmp/vid1.mp4", sha256="deadbeef",
        duration_s=68.0, width=640, height=360, has_audio=True, probe_ok=True,
        meta_json='{"url": "https://x"}',
    )
    reg.upsert_video(row)
    assert reg.get_video("vid1").duration_s == 68.0
    reg.upsert_video(VideoRow(**{**row.__dict__, "duration_s": 70.0}))
    assert reg.get_video("vid1").duration_s == 70.0
    assert [v.video_id for v in reg.list_videos(corpus="hu-ma")] == ["vid1"]


def test_upload_cache_expires(reg):
    now = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    reg.put_upload("sha1", "files/abc", now=now)
    assert reg.get_upload("sha1", now=now + timedelta(hours=40)) == "files/abc"
    assert reg.get_upload("sha1", now=now + timedelta(hours=45)) is None
    assert reg.get_upload("missing", now=now) is None


def test_episode_roundtrip(reg):
    reg.upsert_episode("vid1", 0, 12.0, 48.5, source_unit_id="u1", frozen=True)
    reg.upsert_episode("vid1", 1, 60.0, 92.0, source_unit_id="u1", frozen=True)
    eps = reg.episodes_for("vid1", frozen_only=True)
    assert [(e.episode_idx, e.start_s) for e in eps] == [(0, 12.0), (1, 60.0)]

    reg.upsert_episode("vid1", 1, 61.5, 95.0, source_unit_id="u2", frozen=False)
    assert [e.episode_idx for e in reg.episodes_for("vid1", frozen_only=True)] == [0]
    all_eps = reg.episodes_for("vid1", frozen_only=False)
    assert [(e.episode_idx, e.start_s, e.frozen) for e in all_eps] == [
        (0, 12.0, True), (1, 61.5, False),
    ]


def test_concurrent_writers_to_one_row_leave_consistent_state(reg):
    from concurrent.futures import ThreadPoolExecutor

    u = unit()
    reg.enqueue(u)

    def flip(i: int) -> None:
        if i % 2:
            reg.mark_done(u.unit_id, f"raw{i}", f"parsed{i}", i, i)
        else:
            reg.mark_failed(u.unit_id, "RETRY_EXHAUSTED", f"err{i}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(flip, range(40)))

    row = reg.get_unit(u.unit_id)
    assert row.status in {"done", "failed"}
    # Whichever writer landed last, the row must not be a mix of both outcomes.
    if row.status == "done":
        assert row.error_class is None and row.raw_path is not None
    else:
        assert row.error_class == "RETRY_EXHAUSTED"


def test_many_distinct_units_enqueue_and_complete_from_threads(reg):
    from concurrent.futures import ThreadPoolExecutor

    units = [unit(unit_id=f"u{i}", k=i) for i in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reg.enqueue, units))
        list(pool.map(lambda u: reg.mark_done(u.unit_id, "r", "p", 1, 1), units))
    assert reg.status_counts("run_1") == {"done": 20}


def test_config_fingerprint_changes_the_unit_id():
    args = dict(video_id="v", pass_name="p", prompt_hash="h", model="m", media="full", k=0)
    base = make_unit_id(**args, config_fingerprint="t=0.0|max_out=16384|schema=A")
    assert base == make_unit_id(**args, config_fingerprint="t=0.0|max_out=16384|schema=A")
    for other in (
        "t=0.7|max_out=16384|schema=A",   # temperature
        "t=0.0|max_out=4096|schema=A",    # output budget
        "t=0.0|max_out=16384|schema=B",   # response schema
    ):
        assert make_unit_id(**args, config_fingerprint=other) != base, other


def test_enqueue_repoints_an_existing_unit_at_the_current_run(reg):
    """unit_id is global, but every read filters by run_id.

    A second run over already-done units must still see them, or aggregate writes
    empty tables with no error at all.
    """
    u = unit()
    reg.enqueue(u)
    reg.mark_done(u.unit_id, "raw/a.json", "parsed/a.json", 10, 20)
    assert [x.unit_id for x in reg.done_units("run_1")] == [u.unit_id]

    second = unit(run_id="run_2")
    assert reg.enqueue(second) is False          # no new row: work is not redone
    assert [x.unit_id for x in reg.done_units("run_2")] == [u.unit_id]
    assert reg.status_counts("run_2") == {"done": 1}
    assert reg.done_units("run_1") == []         # the row moved, it was not copied
    assert reg.get_unit(u.unit_id).status == "done"
