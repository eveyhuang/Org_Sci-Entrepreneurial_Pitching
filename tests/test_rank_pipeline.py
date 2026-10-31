"""The ranking pipeline: prepared windows, clips, one call per batch, retries, resume, scores."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline.errors import ConfigError, ProviderError, RunFatalError
from pipeline.ingest import ffprobe
from pipeline.instrument import load_release
from pipeline.providers.fake import FakeProvider
from pipeline.schemas.stage_a_rank import RankResponse
from pipeline.stage_a.rank_pipeline import plan_ranking, prepared_window, run_ranking
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.panel_pipeline import load_panel_config
from tests.conftest import make_test_video, needs_ffmpeg

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
CONFIG = REPO / "configs" / "stage_b_panel.yaml"
CLI = REPO / "scripts" / "experiments" / "run_stage_a_ranking.py"
NOSLEEP = lambda s: None  # noqa: E731
CODES = load_panel_books(load_release()).stage_a.ids
LEDGER_KEYS = {"unit_key", "pass", "attempt", "ok", "error_class", "model_requested", "model_version",
               "prompt_hash", "input_tokens", "output_tokens", "thinking_tokens", "seconds"}


def _add_video(run_root: Path, video_id: str, spans=((10.0, 70.0),), source: Path | None = None,
               primary: bool = True) -> None:
    """A hand-built panel-run video folder: just what ranking reads."""
    vdir = run_root / video_id
    (vdir / "results").mkdir(parents=True, exist_ok=True)
    if source is None:
        source = run_root.parent / "sources" / f"{video_id}.mp4"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"placeholder " + video_id.encode())
    (vdir / "experiment.json").write_text(json.dumps({"video_id": video_id, "source": str(source)}))
    if primary:
        record = {"prepared_spans": [{"start_s": a, "end_s": b} for a, b in spans]}
        (vdir / "results" / "primary.json").write_text(json.dumps({"pass": "primary", "record": record}))


def _run_root(tmp_path: Path, n: int = 6) -> tuple[Path, list[str]]:
    root = tmp_path / "panel_run"
    ids = [f"vid{i}" for i in range(1, n + 1)]
    for vid in ids:
        _add_video(root, vid)
    return root, ids


def _payload(batch: dict, drop_last_clip_on: str | None = None) -> dict:
    """A response that ranks a batch by video id (vid1 strongest) on every dimension."""
    by_strength = [label for _, label in sorted(zip(batch["video_ids"], batch["labels"]))]
    rankings = []
    for code in CODES:
        order = list(by_strength)
        if code == drop_last_clip_on:
            order = order[:-1]
        rankings.append({"code": code, "order": order, "note": f"{order[0]} leads on {code}; close below."})
    return {"rankings": rankings, "notes": "nothing to add"}


def _fixtures(tmp_path: Path, plan: dict) -> Path:
    fx = tmp_path / "fx"
    fx.mkdir(exist_ok=True)
    for batch in plan["design"]:
        (fx / f"rank__{batch['batch_id']}.json").write_text(json.dumps(_payload(batch)))
    return fx


class FakeCut:
    """Stands in for ffmpeg: writes a small file per (source, window) and records the calls."""

    def __init__(self):
        self.calls: list[tuple[str, float, float]] = []

    def __call__(self, src, start_s, end_s, cache_dir):
        self.calls.append((Path(src).stem, start_s, end_s))
        out = Path(cache_dir) / f"{Path(src).stem}__{start_s:.3f}_{end_s:.3f}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"clip of {Path(src).stem} {start_s}-{end_s}".encode())
        return out


class SequenceProvider(FakeProvider):
    """A FakeProvider whose first responses for a unit key come from a queue of bodies."""

    def __init__(self, fixture_dir, bodies: dict[str, list[dict]], **kw):
        super().__init__(fixture_dir, **kw)
        self.bodies = {k: list(v) for k, v in bodies.items()}

    def _fixture_for(self, unit_key: str) -> Path:
        queued = self.bodies.get(unit_key)
        if queued:
            path = self.fixture_dir / f"{unit_key}.queued{len(queued)}.json"
            path.write_text(json.dumps(queued.pop(0)))
            return path
        return super()._fixture_for(unit_key)


def _setup(tmp_path, n=6, batch_size=3, appearances=2, seed=0, error_plan=None, bodies=None):
    root, ids = _run_root(tmp_path, n)
    plan = plan_ranking(root, ids, batch_size, appearances, seed)
    fx = _fixtures(tmp_path, plan)
    provider = SequenceProvider(fx, bodies or {}, error_plan=error_plan)
    return provider, load_release(), load_panel_config(CONFIG), root, plan


def _go(tmp_path, **kw):
    provider, release, cfg, root, plan = _setup(tmp_path, **kw)
    cut = FakeCut()
    result = run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, cut=cut)
    return provider, result, root / "ranking", plan, cut


def _json(path: Path):
    return json.loads(path.read_text())


# --- windows and plan -------------------------------------------------------------------------

def test_window_from_one_span(tmp_path):
    root = tmp_path / "run"
    _add_video(root, "vid1", spans=((12.5, 95.0),))
    window = prepared_window(root, "vid1")
    assert window == {"start_s": 12.5, "end_s": 95.0, "n_spans": 1,
                      "source": str(tmp_path / "sources" / "vid1.mp4")}


def test_window_from_two_spans_covers_the_interruption_and_says_so(tmp_path):
    root = tmp_path / "run"
    _add_video(root, "vid1", spans=((10.0, 40.0), (55.0, 90.0)))
    window = prepared_window(root, "vid1")
    assert (window["start_s"], window["end_s"], window["n_spans"]) == (10.0, 90.0, 2)


def test_no_primary_pass_or_no_spans_has_no_window(tmp_path):
    root = tmp_path / "run"
    _add_video(root, "nopass", primary=False)
    _add_video(root, "nospans", spans=())
    assert prepared_window(root, "nopass") is None
    assert prepared_window(root, "nospans") is None
    assert prepared_window(root, "never_annotated") is None


def test_plan_skips_videos_without_a_window_and_gives_the_reason(tmp_path):
    root, ids = _run_root(tmp_path, 4)
    _add_video(root, "nopass", primary=False)
    _add_video(root, "nospans", spans=())
    plan = plan_ranking(root, [*ids, "nopass", "nospans", "ghost"], batch_size=5, appearances=2, seed=0)
    assert set(plan["skipped"]) == {"nopass", "nospans", "ghost"}
    assert "primary" in plan["skipped"]["nopass"] and "primary" in plan["skipped"]["ghost"]
    assert "prepared" in plan["skipped"]["nospans"]
    assert set(plan["windows"]) == set(ids)
    ranked = {v for b in plan["design"] for v in b["video_ids"]}
    assert ranked == set(ids)
    assert plan["summary"]["n_videos"] == 4 and plan["summary"]["n_batches"] == 2
    assert plan["settings"] == {"batch_size": 5, "appearances": 2, "seed": 0}
    assert plan["design"][0].keys() == {"batch_id", "video_ids", "labels"}
    json.dumps(plan)  # the plan is stored, so it must be plain data


def test_plan_is_deterministic_and_ignores_input_order(tmp_path):
    root, ids = _run_root(tmp_path, 7)
    a = plan_ranking(root, ids, 3, 2, 4)
    assert a == plan_ranking(root, list(reversed(ids)), 3, 2, 4)
    assert a["design"] != plan_ranking(root, ids, 3, 2, 5)["design"]


def test_plan_needs_two_rankable_videos(tmp_path):
    root = tmp_path / "run"
    _add_video(root, "vid1")
    _add_video(root, "nopass", primary=False)
    with pytest.raises(ConfigError, match="at least 2"):
        plan_ranking(root, ["vid1", "nopass"], 5, 2, 0)
    _add_video(root, "vid2")
    with pytest.raises(ConfigError):
        plan_ranking(root, ["vid1", "vid2"], 1, 2, 0)


# --- calls ------------------------------------------------------------------------------------

def test_one_call_per_batch_with_labelled_clips_only(tmp_path):
    provider, result, rdir, plan, _ = _go(tmp_path)
    assert len(plan["design"]) == 4
    assert provider.calls == [f"rank__{b['batch_id']}" for b in plan["design"]]
    for request, batch in zip(provider.requests, plan["design"]):
        assert request.media is None
        assert request.schema is RankResponse
        assert [label for label, _ in request.media_parts] == batch["labels"] == ["A", "B", "C"]
        # each label carries the clip of the video the design put in that position
        stems = [handle.local_path.name.split("__")[0] for _, handle in request.media_parts]
        assert stems == batch["video_ids"]
        assert all(handle.local_path.parent == rdir / "clips" for _, handle in request.media_parts)
        assert "The clips are labelled A, B, C." in request.prompt
    assert result == {"status": "complete", "batches_ok": 4, "batches_failed": 0, "batches_total": 4,
                      "generation_calls": 4, "generation_calls_total": 4,
                      "scale": "relative_within_run", "human_validated": False}
    assert _json(rdir / "completion.json") == result


def test_the_prompt_shows_no_questioning_decision_or_score_request(tmp_path):
    provider, *_ = _go(tmp_path)
    lowered = " ".join(provider.requests[0].prompt.lower().split())
    assert "do not give scores" in lowered and "no ties" in lowered
    # pitch_positivity's Exclude-if names funding only as something the judgement must not use
    assert lowered.count("funding") == lowered.count("or inferred funding success. do not derive") == 1
    lowered = lowered.replace("or inferred funding success. do not derive", "")
    for word in ("predict", "funding", "offer", "decision"):
        assert word not in lowered


def test_clips_are_cut_once_per_video_from_its_prepared_window(tmp_path):
    provider, _, rdir, plan, cut = _go(tmp_path)
    assert sorted(cut.calls) == [(f"vid{i}", 10.0, 70.0) for i in range(1, 7)]  # six videos, twelve appearances
    assert len(list((rdir / "clips").iterdir())) == 6
    assert len(provider.uploads) == 12  # each batch uploads its own clips; the provider caches by content


def test_batch_files_and_ledger(tmp_path):
    provider, _, rdir, plan, _ = _go(tmp_path)
    batch = plan["design"][0]
    bdir = rdir / "batches" / batch["batch_id"]
    assert {p.name for p in bdir.iterdir()} == {"prompt.txt", "raw.json", "result.json"}
    assert (bdir / "prompt.txt").read_text() == provider.requests[0].prompt
    assert RankResponse.model_validate_json((bdir / "raw.json").read_text())
    result = _json(bdir / "result.json")
    assert set(result) == {"batch_id", "video_ids", "labels", "rankings", "notes", "batch_notes",
                           "model_version", "prompt_hash"}
    assert result["video_ids"] == batch["video_ids"] and result["labels"] == batch["labels"]
    assert set(result["rankings"]) == set(result["notes"]) == set(CODES)
    for code in CODES:  # labels are translated back to video ids, strongest first
        assert result["rankings"][code] == sorted(batch["video_ids"])
    assert result["model_version"] == "fake-model-v1" and result["batch_notes"] == "nothing to add"
    rows = [json.loads(line) for line in (rdir / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and all(set(r) == LEDGER_KEYS for r in rows)
    assert [r["unit_key"] for r in rows] == provider.calls
    assert all(r["ok"] and r["attempt"] == 1 and r["prompt_hash"] == result["prompt_hash"] for r in rows)


def test_design_file_records_the_plan_and_the_instrument(tmp_path):
    _, _, rdir, plan, _ = _go(tmp_path)
    design = _json(rdir / "design.json")
    for key in ("design", "summary", "windows", "skipped", "settings"):
        assert design[key] == plan[key]
    release = load_release()
    assert design["release_id"] == release.release_id
    assert len(design["prompt_template_sha256"]) == 64 and len(design["stage_a_codebook_sha256"]) == 64
    assert design["model"] == {"name": "gemini-3.1-pro-preview", "temperature": 0.7, "max_output_tokens": 65536}
    assert design["annotation_context"] == "prepared_clips_only"
    assert design["scale"] == "relative_within_run" and design["human_validated"] is False


def test_valid_results_aggregate_into_scores_json_and_csv(tmp_path):
    _, _, rdir, plan, _ = _go(tmp_path)
    ids = [f"vid{i}" for i in range(1, 7)]
    scores = _json(rdir / "scores.json")
    assert set(CODES) <= set(scores) and CODES == ("pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth")
    assert scores["batches_ok"] == 4 and scores["batches_failed"] == 0
    assert scores["appearances_ok"] == {v: 2 for v in ids}
    assert scores["scale"] == "relative_within_run" and scores["human_validated"] is False
    for code in CODES:
        block = scores[code]
        assert set(block["pl_score"]) == set(block["mean_rank"]) == set(ids)
        assert block["n_orders"] == 4
        assert block["order"][0] == "vid1" and block["order"][-1] == "vid6"
        assert abs(sum(block["pl_score"].values())) < 1e-9
    with (rdir / "scores.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == ["video_id", "code", "pl_score", "mean_rank", "n_orders"]
    assert len(rows) == 18 and {(r["video_id"], r["code"]) for r in rows} == {(v, c) for v in ids for c in CODES}
    first = next(r for r in rows if r["video_id"] == "vid1" and r["code"] == CODES[0])
    assert float(first["pl_score"]) == pytest.approx(scores[CODES[0]]["pl_score"]["vid1"])
    assert float(first["mean_rank"]) == 1.0 and first["n_orders"] == "4"


def test_one_batch_of_all_videos_recovers_the_planted_order(tmp_path):
    _, result, rdir, plan, _ = _go(tmp_path, n=5, batch_size=5)
    assert result["batches_total"] == 2
    scores = _json(rdir / "scores.json")
    assert scores[CODES[2]]["order"] == ["vid1", "vid2", "vid3", "vid4", "vid5"]


# --- invalid output and provider errors -------------------------------------------------------

def _first_batch(tmp_path, **kw):
    root, ids = _run_root(tmp_path, 6)
    return plan_ranking(root, ids, 3, 2, 0)["design"][0]


def test_invalid_ranking_is_retried_once_then_succeeds(tmp_path):
    batch = _first_batch(tmp_path)
    bad = _payload(batch, drop_last_clip_on="expressed_enthusiasm")
    provider, result, rdir, plan, _ = _go(tmp_path, bodies={"rank__b01": [bad]})
    assert provider.calls == ["rank__b01", "rank__b01", "rank__b02", "rank__b03", "rank__b04"]
    assert result["status"] == "complete" and result["generation_calls"] == 5
    bdir = rdir / "batches" / "b01"
    assert json.loads((bdir / "attempt1_invalid.txt").read_text()) == bad
    error = _json(bdir / "attempt1_error.json")
    assert len(error["problems"]) == 1 and "expressed_enthusiasm" in error["problems"][0]
    assert (bdir / "result.json").exists() and not (bdir / "failed.json").exists()
    rows = [json.loads(line) for line in (rdir / "calls.jsonl").read_text().splitlines()]
    assert [(r["unit_key"], r["attempt"], r["ok"]) for r in rows[:2]] == [
        ("rank__b01", 1, False), ("rank__b01", 2, True)]
    assert rows[0]["error_class"] == "SchemaInvalidError" and rows[0]["input_tokens"] > 0  # it was billed


def test_unparseable_output_is_retried_like_an_invalid_ranking(tmp_path):
    provider, result, rdir, _, _ = _go(tmp_path, bodies={"rank__b02": [{"rankings": "oops"}]})
    assert provider.calls.count("rank__b02") == 2 and result["status"] == "complete"
    assert (rdir / "batches/b02/attempt1_invalid.txt").read_text() == '{"rankings": "oops"}'
    assert _json(rdir / "batches/b02/attempt1_error.json")["problems"] == []


def test_invalid_twice_fails_the_batch_and_the_others_still_run(tmp_path):
    batch = _first_batch(tmp_path)
    bad = _payload(batch, drop_last_clip_on="pitch_positivity")
    provider, result, rdir, plan, _ = _go(tmp_path, bodies={"rank__b01": [bad, bad]})
    assert provider.calls == ["rank__b01", "rank__b01", "rank__b02", "rank__b03", "rank__b04"]
    assert result["status"] == "partial"
    assert (result["batches_ok"], result["batches_failed"], result["generation_calls"]) == (3, 1, 5)
    bdir = rdir / "batches" / "b01"
    failed = _json(bdir / "failed.json")
    assert failed["batch_id"] == "b01" and "pitch_positivity" in failed["message"]
    assert not (bdir / "result.json").exists()
    assert (bdir / "attempt1_invalid.txt").exists() and (bdir / "attempt2_invalid.txt").exists()
    scores = _json(rdir / "scores.json")
    assert scores["batches_ok"] == 3 and scores["batches_failed"] == 1
    assert all(scores[code]["n_orders"] == 3 for code in CODES)
    assert scores["appearances_ok"] == {v: (1 if v in batch["video_ids"] else 2)
                                        for v in (f"vid{i}" for i in range(1, 7))}
    assert set(scores[CODES[0]]["pl_score"]) == {f"vid{i}" for i in range(1, 7)}


def test_every_batch_invalid_is_a_failed_run_with_empty_scores(tmp_path):
    root, ids = _run_root(tmp_path, 3)
    plan = plan_ranking(root, ids, 5, 1, 0)
    bad = _payload(plan["design"][0], drop_last_clip_on=CODES[0])
    provider = SequenceProvider(_fixtures(tmp_path, plan), {"rank__b01": [bad, bad]})
    result = run_ranking(provider, load_release(), load_panel_config(CONFIG), root, plan,
                         sleep=NOSLEEP, cut=FakeCut())
    assert result["status"] == "failed" and result["batches_ok"] == 0
    scores = _json(root / "ranking" / "scores.json")
    assert scores[CODES[0]] == {"pl_score": {}, "mean_rank": {}, "n_orders": 0, "order": []}
    assert scores["appearances_ok"] == {v: 0 for v in ids}


def test_retryable_provider_error_backs_off_and_succeeds(tmp_path):
    provider, release, cfg, root, plan = _setup(
        tmp_path, error_plan={"rank__b02": [ProviderError("busy", status=503)]})
    sleeps = []
    result = run_ranking(provider, release, cfg, root, plan, sleep=sleeps.append, cut=FakeCut())
    assert provider.calls == ["rank__b01", "rank__b02", "rank__b02", "rank__b03", "rank__b04"]
    assert len(sleeps) == 1 and result["status"] == "complete"


def test_non_retryable_provider_error_fails_the_batch_only(tmp_path):
    plan = {"rank__b03": [ProviderError("blocked", status=400, raw_text="BODY")]}
    provider, result, rdir, _, _ = _go(tmp_path, error_plan=plan)
    assert provider.calls.count("rank__b03") == 1 and result["status"] == "partial"
    assert "blocked" in _json(rdir / "batches/b03/failed.json")["message"]
    assert (rdir / "batches/b03/attempt1_failed.txt").read_text() == "BODY"


def test_fatal_error_propagates(tmp_path):
    provider, release, cfg, root, plan = _setup(tmp_path, error_plan={"rank__b02": [RunFatalError("auth")]})
    with pytest.raises(RunFatalError):
        run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, cut=FakeCut())
    assert (root / "ranking/batches/b01/result.json").exists()  # finished work is kept


def test_upload_error_is_retried_and_a_dead_upload_fails_only_its_batches(tmp_path):
    provider, release, cfg, root, plan = _setup(tmp_path)
    real = provider.upload_media
    state = {"n": 0}

    def flaky(path, file_sha256=None):
        state["n"] += 1
        if state["n"] == 1:
            raise ProviderError("busy", status=503)
        if Path(path).name.startswith("vid1__"):
            raise ProviderError("bad file", status=400)
        assert file_sha256 and len(file_sha256) == 64
        return real(path, file_sha256=file_sha256)

    provider.upload_media = flaky
    result = run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, cut=FakeCut())
    with_vid1 = [b["batch_id"] for b in plan["design"] if "vid1" in b["video_ids"]]
    assert len(with_vid1) == 2 and result["batches_failed"] == 2 and result["status"] == "partial"
    assert provider.calls == [f"rank__{b['batch_id']}" for b in plan["design"] if b["batch_id"] not in with_vid1]
    assert "upload failed" in _json(root / "ranking/batches" / with_vid1[0] / "failed.json")["message"]


def test_a_clip_that_cannot_be_cut_fails_only_its_batches(tmp_path):
    provider, release, cfg, root, plan = _setup(tmp_path)
    good = FakeCut()

    def cut(src, start_s, end_s, cache_dir):
        if Path(src).stem == "vid2":
            raise RuntimeError("ffmpeg failed: no such file")
        return good(src, start_s, end_s, cache_dir)

    result = run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, cut=cut)
    assert result["batches_failed"] == 2 and result["batches_ok"] == 2
    assert len(provider.calls) == 2
    failed = [p for p in (root / "ranking/batches").glob("*/failed.json")]
    assert len(failed) == 2 and all("vid2" in _json(p)["message"] for p in failed)


def test_before_call_runs_once_per_generation_call(tmp_path):
    provider, release, cfg, root, plan = _setup(tmp_path)
    ticks = []
    run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, before_call=lambda: ticks.append(1),
                cut=FakeCut())
    assert len(ticks) == len(provider.calls) == 4


# --- resume -----------------------------------------------------------------------------------

def test_resume_makes_no_calls_and_cuts_nothing(tmp_path):
    _, first, rdir, plan, _ = _go(tmp_path)
    before = {p: p.read_bytes() for p in rdir.rglob("*") if p.is_file()}
    provider2 = FakeProvider(tmp_path / "fx")
    cut2 = FakeCut()
    result = run_ranking(provider2, load_release(), load_panel_config(CONFIG), tmp_path / "panel_run", plan,
                         sleep=NOSLEEP, cut=cut2)
    assert provider2.calls == [] and provider2.uploads == [] and cut2.calls == []
    assert result["status"] == "complete" and result["generation_calls"] == 0
    assert result["generation_calls_total"] == 4
    after = {p: p.read_bytes() for p in rdir.rglob("*") if p.is_file()}
    assert {p: b for p, b in after.items() if p.name != "completion.json"} == \
           {p: b for p, b in before.items() if p.name != "completion.json"}


def test_resume_reruns_only_the_failed_batch(tmp_path):
    batch = _first_batch(tmp_path)
    bad = _payload(batch, drop_last_clip_on="pitch_positivity")
    _, first, rdir, plan, _ = _go(tmp_path, bodies={"rank__b01": [bad, bad]})
    assert first["status"] == "partial"
    provider2 = FakeProvider(tmp_path / "fx")
    cut2 = FakeCut()
    result = run_ranking(provider2, load_release(), load_panel_config(CONFIG), tmp_path / "panel_run", plan,
                         sleep=NOSLEEP, cut=cut2)
    assert provider2.calls == ["rank__b01"] and result["status"] == "complete"
    assert sorted(c[0] for c in cut2.calls) == sorted(batch["video_ids"])
    assert not (rdir / "batches/b01/failed.json").exists()
    rows = [json.loads(line) for line in (rdir / "calls.jsonl").read_text().splitlines()]
    assert rows[-1]["unit_key"] == "rank__b01" and rows[-1]["attempt"] == 3
    assert _json(rdir / "scores.json")["batches_ok"] == 4


def test_changed_seed_refuses_to_resume(tmp_path):
    provider, _, rdir, plan, _ = _go(tmp_path)
    root = tmp_path / "panel_run"
    other = plan_ranking(root, [f"vid{i}" for i in range(1, 7)], 3, 2, seed=1)
    assert other["design"] != plan["design"]
    stored = (rdir / "design.json").read_bytes()
    provider2 = FakeProvider(tmp_path / "fx")
    with pytest.raises(ConfigError, match="design"):
        run_ranking(provider2, load_release(), load_panel_config(CONFIG), root, other, sleep=NOSLEEP,
                    cut=FakeCut())
    assert provider2.calls == [] and (rdir / "design.json").read_bytes() == stored


@pytest.mark.parametrize("key", ["windows", "prompt_template_sha256", "stage_a_codebook_sha256", "model"])
def test_changed_instrument_or_windows_refuses_to_resume(tmp_path, key):
    _, _, rdir, plan, _ = _go(tmp_path)
    stored = _json(rdir / "design.json")
    if key == "windows":
        stored["windows"]["vid1"]["end_s"] = 71.0
    elif key == "model":
        stored["model"]["temperature"] = 0.0
    else:
        stored[key] = "0" * 64
    (rdir / "design.json").write_text(json.dumps(stored))
    provider2 = FakeProvider(tmp_path / "fx")
    with pytest.raises(ConfigError, match=key):
        run_ranking(provider2, load_release(), load_panel_config(CONFIG), tmp_path / "panel_run", plan,
                    sleep=NOSLEEP, cut=FakeCut())
    assert provider2.calls == []


def test_unclean_release_makes_no_call_and_writes_nothing(tmp_path, monkeypatch):
    provider, release, cfg, root, plan = _setup(tmp_path)
    monkeypatch.setattr("pipeline.stage_a.rank_pipeline.check_release", lambda r: ["stale yaml"])
    with pytest.raises(ConfigError, match="stale yaml"):
        run_ranking(provider, release, cfg, root, plan, sleep=NOSLEEP, cut=FakeCut())
    assert provider.calls == [] and not (root / "ranking").exists()


def test_ranking_writes_only_inside_the_ranking_folder(tmp_path):
    root, ids = _run_root(tmp_path, 6)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    plan = plan_ranking(root, ids, 3, 2, 0)
    run_ranking(FakeProvider(_fixtures(tmp_path, plan)), load_release(), load_panel_config(CONFIG), root, plan,
                sleep=NOSLEEP, cut=FakeCut())
    outside = {p: p.read_bytes() for p in root.rglob("*") if p.is_file() and "ranking" not in p.parts}
    assert outside == before


# --- CLI --------------------------------------------------------------------------------------

def _cli(*args):
    return subprocess.run([PY, str(CLI), *args], capture_output=True, text=True)


def test_cli_dry_run_prints_the_plan_and_writes_nothing(tmp_path):
    root, ids = _run_root(tmp_path, 6)
    _add_video(root, "nopass", primary=False)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    out = _cli("--run", str(root), "--batch-size", "3")
    assert out.returncode == 0, out.stderr
    plan = plan_ranking(root, ids, 3, 2, 0)
    for batch in plan["design"]:
        assert f"{batch['batch_id']} billable: " + " ".join(
            f"{label}={vid}" for label, vid in zip(batch["labels"], batch["video_ids"])) in out.stdout
    assert "skipped nopass:" in out.stdout
    assert "total billable calls: 4" in out.stdout
    assert "no calls made" in out.stdout and "RANKING" not in out.stdout
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before
    assert not (root / "ranking").exists()


def test_cli_video_ids_and_errors(tmp_path):
    root, ids = _run_root(tmp_path, 6)
    out = _cli("--run", str(root), "--video-ids", "vid1", "vid2", "vid3", "--seed", "2")
    assert out.returncode == 0 and "total billable calls: 2" in out.stdout
    assert "vid4" not in out.stdout
    one = _cli("--run", str(root), "--video-ids", "vid1")
    assert one.returncode == 1 and "at least 2" in one.stderr
    missing = _cli("--run", str(tmp_path / "nowhere"))
    assert missing.returncode == 1 and "not found" in missing.stderr
    nofx = _cli("--run", str(root), "--execute", "--provider", "fake")
    assert nofx.returncode == 1 and "--fixtures" in nofx.stderr
    assert not (root / "ranking").exists()


@needs_ffmpeg
def test_cli_with_the_fake_provider_end_to_end(tmp_path):
    root = tmp_path / "panel_run"
    ids = ["vid1", "vid2", "vid3"]
    for vid in ids:
        source = make_test_video(tmp_path / "videos" / f"{vid}.mp4", seconds=3.0)
        _add_video(root, vid, spans=((0.5, 2.5),), source=source)
    plan = plan_ranking(root, ids, 5, 2, 0)
    fx = _fixtures(tmp_path, plan)
    # the shipped config paces calls at four a minute; the test must not wait for that
    config = tmp_path / "panel.yaml"
    config.write_text(CONFIG.read_text().replace("requests_per_minute: 4", "requests_per_minute: 6000"))
    assert load_panel_config(config).limits.requests_per_minute == 6000
    args = ["--run", str(root), "--config", str(config), "--provider", "fake", "--fixtures", str(fx)]
    dry = _cli(*args)
    assert dry.returncode == 0 and "total billable calls: 2" in dry.stdout
    assert not (root / "ranking").exists()
    out = _cli(*args, "--execute")
    assert out.returncode == 0, out.stderr
    assert "RANKING complete" in out.stdout
    assert f"{CODES[0]}: vid1 > vid2 > vid3" in out.stdout
    assert out.stdout.count(" > vid2 > ") == len(CODES)
    rdir = root / "ranking"
    assert _json(rdir / "scores.json")["batches_ok"] == 2
    clips = sorted((rdir / "clips").rglob("*.mp4"))
    assert len(clips) == 3
    assert not (rdir / "registry.sqlite").exists()  # the fake provider keeps no upload cache
    again = _cli(*args)
    assert "total billable calls: 0" in again.stdout and again.stdout.count("cached") == 2


@needs_ffmpeg
def test_default_cut_makes_a_real_two_second_clip_of_the_prepared_window(tmp_path):
    source = make_test_video(tmp_path / "videos" / "vid1.mp4", seconds=4.0)
    root = tmp_path / "panel_run"
    _add_video(root, "vid1", spans=((1.0, 3.0),), source=source)
    _add_video(root, "vid2", spans=((0.0, 2.0),), source=source)
    plan = plan_ranking(root, ["vid1", "vid2"], 5, 1, 0)
    (tmp_path / "fx").mkdir()
    (tmp_path / "fx" / "rank__b01.json").write_text(json.dumps(_payload(plan["design"][0])))
    provider = FakeProvider(tmp_path / "fx")
    result = run_ranking(provider, load_release(), load_panel_config(CONFIG), root, plan, sleep=NOSLEEP)
    assert result["status"] == "complete"
    window = prepared_window(root, "vid1")
    assert (window["start_s"], window["end_s"]) == (1.0, 3.0)
    clips = [handle.local_path for _, handle in provider.requests[0].media_parts]
    assert len(clips) == 2 and clips[0] != clips[1]
    for path in clips:
        assert (root / "ranking" / "clips") in path.parents
        assert 1.6 < ffprobe(path).duration_s < 2.4
