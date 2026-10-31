"""The clip session (plan C, Task 6; Task 7c): the balanced pair design, the export of the session's sheet,
manifest and pages (each pitch's full video at its prepared window since 2026-10-21), the import of a coder's session
sheet with Bradley-Terry scores, and pair agreement."""
import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_a.rank_design import design_summary  # noqa: E402
from pipeline.stage_b import review_clips, review_page  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_items import SHEET_COLUMNS  # noqa: E402
from pipeline.stage_b.review_labels import load_labels  # noqa: E402
from tests.test_review_cli import make_run, read_sheet  # noqa: E402

VIDEOS = [f"v{i:02d}" for i in range(10)]


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def scores_for(books, video_ids=VIDEOS):
    """A hand-written ranking: the first item rises with the video's number, the second falls, the third is a mix."""
    codes = books.stage_a_ids()
    mixed = [3, 7, 1, 9, 0, 5, 2, 8, 4, 6]
    score = {codes[0]: {v: 0.3 * i - 1.35 for i, v in enumerate(video_ids)},
             codes[1]: {v: 1.35 - 0.3 * i for i, v in enumerate(video_ids)},
             codes[2]: {v: 0.25 * mixed[i % len(mixed)] - 1.1 for i, v in enumerate(video_ids)}}
    return {code: {"pl_score": s, "mean_rank": {v: 0.0 for v in s}, "n_orders": 30,
                   "order": sorted(s, key=s.get, reverse=True)} for code, s in score.items()} | {
        "batches_ok": 30, "batches_failed": 0, "annotation_context": "prepared_clips_only",
        "scale": "relative_within_run", "human_validated": False}


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, books):
    base = tmp_path_factory.mktemp("clips")
    for v in VIDEOS:
        make_run(base, books, video_id=v)
    run = base / "run"
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books)))
    return run


@pytest.fixture
def run(base_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def clips_export(run, *extra):
    return rp.main(["clips-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", "ana", "ben",
                    "--seed", "4", *extra])


def manifest_of(run):
    return json.loads((run / "review" / "clips" / "export_manifest.json").read_text())


def fill_session(run, coder, answer, close=lambda row: False, rating=lambda row: None, skip=lambda row: False):
    """A coder's filled session sheet, made from the exported template the way the page writes one: `answer(row)`
    gives a pair row's left/right, `rating(row)` a per-video row's value (None: left blank)."""
    template = run / "review" / "clips" / "clips.csv"
    rows = read_sheet(template)
    manifest = {r["row"]: r for r in manifest_of(run)["rows"]}
    for row in rows:
        exported = manifest[int(row["row"])]
        if skip(exported):
            continue
        if exported["field_class"] == "pair":
            row["human_value"] = answer(exported) or ""
            row["note"] = "[close]" if row["human_value"] and close(exported) else ""
        else:
            row["human_value"] = rating(exported) or ""
    sheet = run.parent / f"clips__{coder}.csv"
    with open(sheet, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return sheet


def clips_import(run, sheet, coder):
    return rp.main(["clips-import", "--run", str(run), "--sheet", str(sheet), "--reviewer", coder])


def log_of(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


# ---- the design ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(10))
def test_the_design_is_twenty_five_pairs_five_per_video_without_a_repeat(seed):
    batches = review_clips.pair_design(list(reversed(VIDEOS)), seed=seed)
    summary = design_summary(batches)
    assert len(batches) == 25 and summary["batch_sizes"] == [2] * 25 and summary["repeated_pairs"] == 0
    assert set(summary["appearances"].values()) == {5} and len(summary["appearances"]) == 10
    assert [b.batch_id for b in batches] == [f"b{i:02d}" for i in range(1, 26)]
    assert review_clips.pair_design(VIDEOS, seed=seed) == batches            # the order the ids come in is no input


# ---- export ----------------------------------------------------------------------------------------------------

def test_clips_export_writes_the_manifest_the_sheet_and_one_page_per_coder(run, books, capsys):
    assert clips_export(run) == 0
    out = run / "review" / "clips"
    manifest = manifest_of(run)
    assert {"seed", "videos", "order", "rows", "pairs", "video_files", "windows", "scores_source", "release_id"} <= set(manifest)
    assert manifest["seed"] == 4 and manifest["videos"] == VIDEOS and manifest["scores_source"] == "ranking/scores.json"
    assert manifest["release_id"] == load_release().release_id
    assert manifest["video_files"] == {v: f"{v}.mp4" for v in VIDEOS} and set(manifest["windows"]) == set(VIDEOS)
    assert sorted(manifest["order"]) == ["ana", "ben"] and sorted(manifest["order"]["ana"]) == VIDEOS
    assert manifest["order"]["ana"] != manifest["order"]["ben"]              # each coder's own seeded order
    assert [p["batch_id"] for p in manifest["pairs"]] == [f"b{i:02d}" for i in range(1, 26)]
    assert all(p["codes"] == list(books.stage_a_ids()) for p in manifest["pairs"])
    # seven cards per video (three Stage A items, three prepared codes: not PRES-CTX, the ask), three per pair
    per_video = [r for r in manifest["rows"] if r["field_class"] != "pair"]
    pairs = [r for r in manifest["rows"] if r["field_class"] == "pair"]
    assert len(per_video) == 70 and len(pairs) == 75
    first = [r["path"] for r in per_video if r["video_id"] == "v00"]
    assert first == [*(f"stage_a:{c}" for c in books.stage_a_ids()),
                     *(f"prep_pres:{e.id}" for e in books.prepared_codes() if e.id != "PRES-CTX"), "venture:ask"]
    for row in manifest["rows"]:
        assert {"path", "field_class", "context", "primary", "replicate", "allowed_values"} <= set(row)
    assert next(r for r in per_video if r["path"] == "venture:ask")["primary"] == "200000;10"
    assert pairs[0]["path"] == f"pair:b01/{books.stage_a_ids()[0]}" and pairs[0]["allowed_values"] == ["left", "right"]
    # the sheet: the 15 columns, one row per card, blind
    sheet = read_sheet(out / "clips.csv")
    assert list(sheet[0]) == SHEET_COLUMNS and len(sheet) == 145
    assert [int(r["row"]) for r in sheet] == list(range(1, 146)) == [r["row"] for r in manifest["rows"]]
    assert all(r["primary"] == r["replicate"] == r["human_value"] == "" for r in sheet)
    assert {r["path"] for r in sheet if r["field_class"] == "pair"} == {r["path"] for r in pairs}
    for coder in ("ana", "ben"):
        assert (out / "pages" / f"{coder}.html").exists()
    printed = capsys.readouterr().out
    assert "25 pairs" in printed and "ana.html" in printed


def test_the_model_side_of_a_pair_and_its_gap_come_from_the_rank_scores(run, books):
    assert clips_export(run) == 0
    scores = scores_for(books)
    manifest = manifest_of(run)
    for row in (r for r in manifest["rows"] if r["field_class"] == "pair"):
        pl = scores[row["code"]]["pl_score"]
        assert row["primary"] == ("left" if pl[row["left"]] > pl[row["right"]] else "right")
        assert row["gap"] == pytest.approx(abs(pl[row["left"]] - pl[row["right"]]))
        assert row["replicate"] is None
        pair = next(p for p in manifest["pairs"] if p["batch_id"] == row["batch_id"])
        assert (row["left"], row["right"]) == (pair["left"], pair["right"])
    left_wins = [r for r in manifest["rows"] if r["field_class"] == "pair" and r["code"] == books.stage_a_ids()[0]]
    assert all(r["primary"] == ("left" if r["left"] > r["right"] else "right") for r in left_wins)


def test_clips_export_refuses_without_the_rank_scores_and_names_the_command(run, books, capsys):
    (run / "ranking" / "scores.json").unlink()
    assert clips_export(run) == 1
    err = capsys.readouterr().err
    assert f"scripts/experiments/run_stage_a_ranking.py --run {run} --appearances 3 --execute" in err
    assert not (run / "review" / "clips").exists()
    partial = scores_for(books)
    del partial[books.stage_a_ids()[2]]
    (run / "ranking" / "scores.json").write_text(json.dumps(partial))
    assert clips_export(run) == 1
    err = capsys.readouterr().err
    assert books.stage_a_ids()[2] in err and "run_stage_a_ranking.py" in err
    assert not (run / "review" / "clips").exists()


def test_clips_export_refuses_to_replace_a_session_without_force(run, capsys):
    assert clips_export(run) == 0
    assert clips_export(run) == 1
    assert "--force" in capsys.readouterr().err
    assert clips_export(run, "--force") == 0


# ---- the page --------------------------------------------------------------------------------------------------

def _page(run, coder="ana"):
    html = (run / "review" / "clips" / "pages" / f"{coder}.html").read_text()
    return json.loads(html.split("const PAGE = ", 1)[1].split(";\n", 1)[0])


def test_the_clips_page_has_each_video_with_its_seven_cards_then_the_pairs(run, books):
    assert clips_export(run) == 0
    page = _page(run)
    manifest = manifest_of(run)
    assert page["mode"] == review_page.CLIPS and page["list_title"] == "Videos in your clip session"
    assert [v["video_id"] for v in page["videos"][:-1]] == manifest["order"]["ana"]
    for video in page["videos"][:-1]:
        assert video["file_name"] == f"{video['video_id']}.mp4" and video["window"] == "0:05-1:00"
        assert [s["key"] for s in video["sections"]] == ["clips"]
        cards = video["sections"][0]["cards"]
        assert len(cards) == 7 and all(c["start"] == c["end"] == "" for c in cards)
        assert [c["input"] for c in cards][-2:] == [None, "ask"]           # PRES-EVID in bands, then the ask
    pairs = page["videos"][-1]
    assert pairs["video_id"] == review_clips.PAIRS_ID and [s["key"] for s in pairs["sections"]] == ["pairs"]
    assert pairs["sections"][0]["intro"] == page["hints"]["pairs_intro"]
    assert [p["batch_id"] for p in pairs["pairs"]] == [p["batch_id"] for p in manifest["pairs"]]
    assert pairs["pairs"][0]["left_file"] == f"{pairs['pairs'][0]['left']}.mp4"
    cards = pairs["sections"][0]["cards"]
    assert len(cards) == 75 and {c["input"] for c in cards} == {"choice"}
    assert all(c["values"] == ["left", "right"] for c in cards)
    # the numbers a coder reads are the sheet's rows
    by_row = {r["row"]: r["path"] for r in manifest["rows"]}
    assert all(by_row[c["row"]] == c["path"] for v in page["videos"] for s in v["sections"] for c in s["cards"])
    # no model value reaches the coder's page
    from tests.test_review_page import CARD_KEYS
    assert all(set(c) == CARD_KEYS for v in page["videos"] for s in v["sections"] for c in s["cards"])
    assert '"gap"' not in json.dumps(page) and '"model"' not in json.dumps(page)


def test_the_page_draws_two_players_for_a_pair_with_the_video_file_names(run):
    from tests.test_review_page import _lines_of, _node, _script
    assert clips_export(run) == 0
    pair = _page(run)["videos"][-1]["pairs"][0]
    script = _script()
    html = _node("\n".join([_lines_of(script, "const esc = ", "const pairHtml = "),
                            f"console.log(JSON.stringify(pairHtml({json.dumps(pair)}, 0)));"]))
    assert html.count("<video") == 2 and html.index(">left<") < html.index(">right<")
    assert pair["left_file"] in html and pair["right_file"] in html
    # a clip the folder lacks is named on the page, and the sheet still exports the pair unanswered
    assert "not found in the folder" in script and "sessionSheet" in script and "clips__" in script


def test_the_session_sheet_the_page_writes_is_one_the_clip_import_takes(run, books, capsys):
    from tests.test_review_page import _value_source, _node
    assert clips_export(run) == 0
    page = _page(run)
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    first = page["videos"][0]
    pair_cards = page["videos"][-1]["sections"][0]["cards"]
    state = {first["video_id"]: {c["path"]: {**blank, "value": "4"} for c in first["sections"][0]["cards"][:3]},
             review_clips.PAIRS_ID: {pair_cards[0]["path"]: {**blank, "value": "left", "close": True},
                                     pair_cards[1]["path"]: {**blank, "value": "right"}}}
    text = _node(_value_source(page, state, "console.log(JSON.stringify(sessionSheet()));"))
    sheet = run.parent / "clips__ana.csv"
    sheet.write_text(text, encoding="utf-8")
    rows = read_sheet(sheet)
    assert len(rows) == 145 and rows[0]["path"] and list(rows[0]) == SHEET_COLUMNS
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 0
    out = capsys.readouterr().out
    assert "2 pairs answered" in out and "73 unanswered" in out
    pairs = log_of(run / "review" / "clips" / "pairs_log.jsonl")
    assert [(l["batch_id"], l["value"], l["close"]) for l in pairs] == [("b01", "left", True), ("b01", "right", False)]
    log = log_of(run / first["video_id"] / "review" / "review_log.jsonl")
    assert sorted(l["path"] for l in log if not l["implied"]) == sorted(f"stage_a:{c}" for c in books.stage_a_ids())


# ---- import ----------------------------------------------------------------------------------------------------

def test_a_filled_session_sheet_goes_to_the_video_logs_and_the_pairs_log(run, books, capsys):
    assert clips_export(run) == 0
    sheet = fill_session(run, "ana", answer=lambda r: r["primary"], close=lambda r: r["batch_id"] == "b03",
                         rating=lambda r: {"stage_a": "4", "prep_pres": "3"}.get(r["field_class"])
                         if r["path"] != "prep_pres:PRES-EVID" else "1-2")
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 0
    out = capsys.readouterr().out
    assert "75 pairs answered, 0 unanswered" in out
    for v in VIDEOS:
        log = log_of(run / v / "review" / "review_log.jsonl")
        asked = [l for l in log if not l["implied"]]
        assert len(asked) == 6 and {l["role"] for l in asked} == {"coder"} and {l["sheet"] for l in asked} == {sheet.name}
        assert {l["reviewer"] for l in asked} == {"ana"}
        assert next(l for l in asked if l["path"] == "prep_pres:PRES-EVID")["new"] == "1-2"
        assert (run / v / "review" / "human_agreement.json").exists()
        assert not (run / v / "results" / "reviewed.json").exists()        # a coder's judgment never corrects
    pairs = log_of(run / "review" / "clips" / "pairs_log.jsonl")
    assert len(pairs) == 75 and {l["reviewer"] for l in pairs} == {"ana"}
    for line in pairs:
        assert set(line) >= {"reviewer", "batch_id", "code", "left", "right", "value", "close", "model", "gap",
                             "imported_at"}
        assert line["value"] == line["model"] and isinstance(line["gap"], float)
        assert line["close"] is (line["batch_id"] == "b03")
    scores = json.loads((run / "review" / "clips" / "ana_scores.json").read_text())
    assert list(scores) == list(books.stage_a_ids())
    model = scores_for(books)
    for code, by_video in scores.items():
        assert set(by_video) == set(VIDEOS)
        # a coder who chose as the model did ranks the videos close to the model's scores (25 of the 45 possible
        # pairs are judged, so neighbours never compared may swap)
        assert review_clips.spearman(by_video, model[code]["pl_score"]) > 0.85


def test_an_unanswered_pair_is_counted_and_left_out_of_the_fit(run, books, capsys):
    assert clips_export(run) == 0
    skipped = {"b02", "b07"}
    sheet = fill_session(run, "ana", answer=lambda r: None if r["batch_id"] in skipped else r["primary"])
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 0
    out = capsys.readouterr().out
    assert "69 pairs answered, 6 unanswered (skipped)" in out
    pairs = log_of(run / "review" / "clips" / "pairs_log.jsonl")
    assert len(pairs) == 69 and not {l["batch_id"] for l in pairs} & skipped
    # no per-video row was answered: no video log is written
    assert not any((run / v / "review" / "review_log.jsonl").exists() for v in VIDEOS)


def test_the_clip_import_refuses_a_sheet_that_does_not_match_the_design_and_writes_nothing(run, books, capsys):
    assert clips_export(run) == 0
    sheet = fill_session(run, "ana", answer=lambda r: "middle" if r["batch_id"] == "b04" else r["primary"])
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 1
    err = capsys.readouterr().err
    assert "is not left or right" in err and "nothing was written" in err
    assert not (run / "review" / "clips" / "pairs_log.jsonl").exists()
    rows = read_sheet(sheet)
    rows[-1]["path"] = "pair:b99/pitch_positivity"
    rows[-1]["human_value"] = "left"
    for row in rows:
        if row["human_value"] == "middle":
            row["human_value"] = "left"
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assert clips_import(run, sheet, "ana") == 1
    assert "pair:b99/pitch_positivity" in capsys.readouterr().err
    assert not (run / "review" / "clips" / "pairs_log.jsonl").exists()


def test_bradley_terry_orders_a_hand_built_set_of_choices():
    lines = [{"reviewer": "ana", "batch_id": f"b{i:02d}", "code": "x", "left": left, "right": right, "value": value}
             for i, (left, right, value) in enumerate([("a", "b", "left"), ("b", "c", "left"), ("a", "c", "left"),
                                                       ("c", "d", "left"), ("b", "d", "left"), ("d", "a", "right")])]
    lines.append({**lines[0], "batch_id": "b90", "value": None})        # an unanswered pair: no evidence
    scores = review_clips.bradley_terry(lines, "ana")
    assert sorted(scores["x"], key=scores["x"].get, reverse=True) == ["a", "b", "c", "d"]
    assert sum(scores["x"].values()) == pytest.approx(0.0, abs=1e-9)


# ---- agreement -------------------------------------------------------------------------------------------------

def _pair(coder, batch, left, right, value, model, gap, close=False, code="x"):
    return {"reviewer": coder, "batch_id": batch, "code": code, "left": left, "right": right, "value": value,
            "close": close, "model": model, "gap": gap}


def test_pair_agreement_gives_the_shares_the_terciles_and_an_interval():
    videos = [f"w{i}" for i in range(6)]
    design = [(f"b{i + 1:02d}", videos[i % 6], videos[(i + 1 + i // 6) % 6], gap)
              for i, gap in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])]
    lines = []
    for n, (batch, left, right, gap) in enumerate(design):
        lines.append(_pair("ana", batch, left, right, "left", "left", gap, close=n == 0))
        # ben agrees with the model on the six widest gaps only, and ticks close on the two narrowest
        lines.append(_pair("ben", batch, left, right, "left" if n >= 3 else "right", "left", gap, close=n < 2))
    # ratings: ana rates the left clip of b01 above its right, so her choice of left there is coherent
    lines += [{"reviewer": "ana", "video_id": "w0", "path": "stage_a:x", "new": "5"},
              {"reviewer": "ana", "video_id": "w1", "path": "stage_a:x", "new": "3"}]
    got = review_clips.pair_agreement(lines, videos, n_boot=199, seed=0)["x"]
    assert got["coder_model"] == {"ana": 1.0, "ben": pytest.approx(6 / 9)}
    assert got["coder_coder"] == pytest.approx(6 / 9) and got["n_pairs"] == 9
    terciles = got["by_gap_tercile"]
    assert [t["tercile"] for t in terciles] == [1, 2, 3] and [t["n"] for t in terciles] == [3, 3, 3]
    assert terciles[0]["coder_model"] == {"ana": 1.0, "ben": 0.0} and terciles[2]["coder_model"]["ben"] == 1.0
    assert terciles[0]["coder_coder"] == 0.0 and terciles[2]["coder_coder"] == 1.0
    assert got["close_in_lowest_tercile_share"] == 1.0                 # all three close ticks on the narrowest gaps
    assert got["coherence"]["ana"] == 1.0 and got["coherence"]["ben"] is None
    ci = got["ci95"]
    assert ci["coder_model"]["ana"] == [1.0, 1.0]
    low, high = ci["coder_model"]["ben"]
    assert 0 <= low <= 6 / 9 <= high <= 1 and low < high
    assert ci["coder_coder"][0] <= 6 / 9 <= ci["coder_coder"][1]


def test_a_pair_counts_once_per_draw_of_each_of_its_videos():
    """The resampling weight of a pair is m_a × m_b: a pair whose two videos are drawn twice and once counts twice."""
    weights = review_clips._pair_weights([[2, 1, 0]], [0, 0, 1], [1, 2, 2])
    assert weights.tolist() == [[2, 0, 0]]


def test_the_pooled_pairs_feed_the_paper_tables_pair_rows():
    from pipeline.stage_b.review_apply import pooled_summary
    from pipeline.stage_b.review_report import paper_rows
    videos = [f"w{i}" for i in range(6)]
    lines = [_pair(c, f"b{i:02d}", videos[i % 6], videos[(i + 2) % 6], "left", "left", 0.1 * i)
             for i in range(12) for c in ("ana", "ben")]
    pairs = review_clips.pooled_pairs(review_clips.pair_agreement(lines, videos, n_boot=99))
    assert pairs["x"]["coder_model"] == 1.0 and pairs["x"]["coder_coder"] == 1.0 and pairs["x"]["n"] == 24
    assert pairs["x"]["videos"] == 6 and pairs["x"]["coder_model_by_coder"] == {"ana": 1.0, "ben": 1.0}
    pooled = pooled_summary({}, pairs=pairs)
    assert pooled["pairs"] == pairs
    # a Stage A item's dimension gives its comparison row: k of n against 0.5, no target
    from pipeline.stage_b.review_report import load_reporting
    reporting = load_reporting(load_panel_books(load_release(), sync=False))
    item = load_panel_books(load_release(), sync=False).stage_a_ids()[0]
    pooled = pooled_summary({}, pairs={item: pairs["x"]}, reporting=reporting)
    row = next(r for r in paper_rows(pooled, reporting) if r["field"] == f"pair:{item}")
    assert row["kind"] == "comparison" and row["n"] == 24 and row["counts"]["agree"] == 24
    # twelve pairs of clips, each judged by both coders: 12 units, so the small-n flag (final review I1)
    assert row["units"] == 12 and row["flags"] == ["small_n"]
    assert row["band"] is None and "met" not in row


# ---- fix round 1: the session's design id, ties, the session's coders, withdrawn answers, the summary ---------

def _design(run):
    return manifest_of(run)["design_id"]


def test_a_reexport_with_another_design_refuses_the_old_sheet_and_leaves_old_answers_out(run, books, capsys):
    assert clips_export(run) == 0
    old = _design(run)
    assert len(old) == 12 and all(r["context"].startswith(f"design {old}: ") for r in read_sheet(
        run / "review" / "clips" / "clips.csv"))
    sheet = fill_session(run, "ana", answer=lambda r: r["primary"])
    assert clips_import(run, sheet, "ana") == 0
    assert {l["design_id"] for l in log_of(run / "review" / "clips" / "pairs_log.jsonl")} == {old}
    before = rp.session_pairs(run, books)
    assert before[books.stage_a_ids()[0]]["coder_model"] == 1.0
    # the same seed and videos again: the same design, nothing left out, the same figures
    capsys.readouterr()
    assert clips_export(run, "--force") == 0
    assert _design(run) == old and "left out" not in capsys.readouterr().out
    assert rp.session_pairs(run, books) == before
    # another seed: another design; the logged answers stay in the log but leave the agreement and the fit
    assert clips_export(run, "--force", "--seed", "5") == 0
    new = _design(run)
    out = capsys.readouterr().out
    assert new != old and f"75 pair answers logged for an earlier design ({old}) are left out" in out
    assert rp.session_pairs(run, books) == {}
    assert not (run / "review" / "clips" / "ana_scores.json").exists()
    assert (run / "review" / "clips" / f"earlier_{old}" / "ana_scores.json").exists()
    # the old sheet now pairs other clips: refused, nothing written
    lines = len(log_of(run / "review" / "clips" / "pairs_log.jsonl"))
    assert clips_import(run, sheet, "ana") == 1
    err = capsys.readouterr().err
    assert f"design {old}" in err and f"design {new}" in err and "nothing was written" in err
    assert len(log_of(run / "review" / "clips" / "pairs_log.jsonl")) == lines
    # the new design's sheet imports, and its figures are its own
    assert clips_import(run, fill_session(run, "ana", answer=lambda r: r["primary"]), "ana") == 0
    assert rp.session_pairs(run, books)[books.stage_a_ids()[0]]["n_pairs"] == 25


def test_a_sheet_without_a_design_id_is_refused(run, books, capsys):
    assert clips_export(run) == 0
    sheet = fill_session(run, "ana", answer=lambda r: r["primary"])
    rows = read_sheet(sheet)
    for row in rows:
        row["context"] = ""
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 1
    assert "carries no design id" in capsys.readouterr().err


def test_the_page_keeps_its_answers_per_design():
    from tests.test_review_page import _script
    key = next(l for l in _script().splitlines() if l.startswith("const KEY = "))
    assert "PAGE.design_id" in key and "PAGE.mode && PAGE.mode !== 'validation' ? PAGE.mode : 'coding'" in key


def test_coders_are_compared_by_the_clip_they_chose_and_each_line_by_its_own_model_side():
    videos = [f"w{i}" for i in range(6)]
    lines = []
    for i in range(6):
        a, b = videos[i], videos[(i + 1) % 6]
        lines.append(_pair("ana", f"b{i:02d}", a, b, "left", "left", 0.1 * i))
        # ben's line names the sides the other way round; he chose the same clip, as the model did
        lines.append(_pair("ben", f"b{i:02d}", b, a, "right", "right", 0.1 * i))
    got = review_clips.pair_agreement(lines, videos, n_boot=99)["x"]
    assert got["coder_coder"] == 1.0 and got["coder_model"] == {"ana": 1.0, "ben": 1.0}


def test_a_tie_in_the_rank_scores_is_no_model_answer(run, books, capsys):
    scores = scores_for(books)
    code = books.stage_a_ids()[0]
    scores[code]["pl_score"] = {v: 0.5 for v in VIDEOS}           # every pair a tie on the first item
    (run / "ranking" / "scores.json").write_text(json.dumps(scores))
    assert clips_export(run) == 0
    tied = [r for r in manifest_of(run)["rows"] if r.get("code") == code]
    assert all(r["primary"] is None and r["tie"] is True and r["gap"] == 0.0 for r in tied)
    assert not any(r.get("tie") for r in manifest_of(run)["rows"] if r.get("code") not in (None, code))
    assert clips_import(run, fill_session(run, "ana", answer=lambda r: r["primary"] or "left"), "ana") == 0
    assert clips_import(run, fill_session(run, "ben", answer=lambda r: r["primary"] or "left"), "ben") == 0
    pairs = rp.session_pairs(run, books)[code]
    assert pairs["coder_model"] is None and pairs["coder_coder"] == 1.0 and pairs["n_pairs"] == 25
    assert pairs["n_model_ties"] == 25
    lines = [_pair("ana", "b01", "a", "b", "left", None, 0.0), _pair("ana", "b02", "a", "c", "left", "left", 0.4)]
    got = review_clips.pair_agreement(lines, ["a", "b", "c"], n_boot=19)["x"]
    assert got["coder_model"] == {"ana": 1.0} and got["n_by_coder"] == {"ana": 1} and got["n_model_ties"] == 1


def test_a_reviewer_who_is_not_one_of_the_sessions_coders_is_refused(run, capsys):
    assert clips_export(run) == 0
    sheet = fill_session(run, "zed", answer=lambda r: r["primary"])
    capsys.readouterr()
    assert clips_import(run, sheet, "zed") == 1
    err = capsys.readouterr().err
    assert "'zed' is not a coder of this clip session (ana, ben)" in err
    assert not (run / "review" / "clips" / "pairs_log.jsonl").exists()


def test_a_pair_left_blank_on_a_later_import_is_withdrawn(run, books, capsys):
    assert clips_export(run) == 0
    assert clips_import(run, fill_session(run, "ana", answer=lambda r: r["primary"]), "ana") == 0
    capsys.readouterr()
    again = fill_session(run, "ana", answer=lambda r: None if r["batch_id"] == "b01" else r["primary"])
    assert clips_import(run, again, "ana") == 0
    assert "3 earlier answers withdrawn" in capsys.readouterr().out
    withdrawn = [l for l in log_of(run / "review" / "clips" / "pairs_log.jsonl") if l.get("withdrawn")]
    assert len(withdrawn) == 3 and {l["batch_id"] for l in withdrawn} == {"b01"} and {l["value"] for l in withdrawn} == {None}
    pairs = rp.session_pairs(run, books)
    assert all(f["n_pairs"] == 24 for f in pairs.values())
    # a third import with b01 still blank withdraws nothing more
    assert clips_import(run, again, "ana") == 0
    assert "withdrawn" not in capsys.readouterr().out


def test_the_summary_reads_the_pairs_and_the_coders_own_ratings_from_the_logs(run, books, capsys):
    assert clips_export(run) == 0
    model = scores_for(books)

    def rating(row):
        if row["field_class"] == "stage_a":                     # ratings that rise with the model's scores
            return str(min(5, max(1, round(3 + model[row["path"].split(":")[1]]["pl_score"][row["video_id"]]))))
        return {"prep_pres": "3", "venture": "250000;12"}.get(row["field_class"]) if row["path"] != "prep_pres:PRES-EVID" else "1-2"

    for coder in ("ana", "ben"):
        assert clips_import(run, fill_session(run, coder, answer=lambda r: r["primary"], rating=rating), coder) == 0
    ratings = review_clips.rating_lines(run, VIDEOS)
    assert len(ratings) == 2 * 10 * 3 and {l["reviewer"] for l in ratings} == {"ana", "ben"}
    ask = [l for l in log_of(run / "v00" / "review" / "review_log.jsonl") if l["path"] == "venture:ask"]
    assert [l["new"] for l in ask] == ["250000;12", "250000;12"]
    pairs = rp.session_pairs(run, books)
    first = pairs[books.stage_a_ids()[0]]
    assert first["coder_model"] == 1.0 and first["coder_coder"] == 1.0
    assert first["coherence"] == {"ana": 1.0, "ben": 1.0}               # where two ratings differ, the choice agrees
    capsys.readouterr()
    out = run.parent / "table.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    text = capsys.readouterr().out
    assert "## Pair judgements (the clip session)" in text and f"| {books.stage_a_ids()[0]} | 25 | 50 | 1.00 |" in text
    rows = {r["field"]: r for r in json.loads(out.with_suffix(".json").read_text())["rows"]}
    assert {f"pair:{c}" for c in books.stage_a_ids()} <= set(rows)
    first = rows[f"pair:{books.stage_a_ids()[0]}"]
    assert first["kind"] == "comparison" and first["n"] == 50 and first["counts"]["agree"] == 50
    assert first["figures"]["agreement"]["value"] == 1.0 and "met" not in first


def test_a_clip_the_folder_lacks_is_named_on_its_pair(run):
    """loadPair run in node over a stand-in for the page: no folder chosen and no clip opened, so both files are
    named as missing and the players get no source."""
    from tests.test_review_page import _lines_of, _node, _script
    script = _script()
    block = lambda start: "\n".join(script.splitlines()[script.splitlines().index(start):][
        :next(i for i, l in enumerate(script.splitlines()[script.splitlines().index(start):]) if l == "}") + 1])
    stub = """
const mk = (file) => { const pf = {dataset: {file}, textContent: '', cls: new Set(), classList: null};
  pf.classList = {add: c => pf.cls.add(c), remove: c => pf.cls.delete(c)};
  const pv = {dataset: {vid: file.split('.')[0]}, src: ''};
  return {pv, pf, box: {querySelector: s => s === 'video' ? pv : pf}}; };
const sides = [mk('v01.mp4'), mk('v02.mp4')];
const el = {querySelectorAll: () => sides.map(s => s.box)};
const urls = {}; let dirHandle = null;
"""
    source = "\n".join([stub, block("async function clipUrl(vid, file){"), block("async function loadPair(el){"),
                        "loadPair(el).then(() => console.log(JSON.stringify(sides.map(s => [s.pf.textContent, [...s.pf.cls], s.pv.src]))));"])
    got = _node(source)
    assert [g[1] for g in got] == [["missingfile"], ["missingfile"]] and [g[2] for g in got] == ["", ""]
    assert got[0][0].startswith("v01.mp4 not found in the folder")


def test_the_pairs_intro_names_the_number_of_videos_and_the_pairs_say_when_cards_are_still_open(base_run, tmp_path, books):
    from tests.test_review_page import _script
    run = tmp_path / "run"
    shutil.copytree(base_run, run)
    assert rp.main(["clips-export", "--run", str(run), "--video-ids", *VIDEOS[:8], "--coders", "ana", "--seed", "1"]) == 0
    page = _page(run)
    assert len(page["videos"][-1]["pairs"]) == 20
    intro = page["videos"][-1]["sections"][0]["intro"]
    assert "all 8 prepared presentations" in intro and "ten" not in intro
    assert review_clips.pairs_intro("You have seen all ten clips. Choose.", 8) == "You have seen all 8 clips. Choose."
    assert "still open" in _script()
    assert rp.main(["clips-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", "ana", "--force"]) == 0
    assert "all ten prepared presentations" in _page(run)["videos"][-1]["sections"][0]["intro"]


def test_the_clip_commands_use_the_public_prepared_window():
    import inspect
    from pipeline.stage_a import rank_pipeline
    assert rank_pipeline.prepared_window_reason
    assert "_window(" not in inspect.getsource(rp).replace("prepared_window(", "")


# ---- Task 7c: the full video at its prepared window; no context card; checkable claims in bands -----------------

def _two_spans(run, video_id="v00", spans=((5.0, 30.0), (40.0, 60.0))):
    """Give a video's primary record a prepared presentation in several spans (nine of the 100 v3-dev9 videos)."""
    path = run / video_id / "results" / "primary.json"
    result = json.loads(path.read_text())
    result["record"]["prepared_spans"] = [{"start_s": a, "end_s": b} for a, b in spans]
    path.write_text(json.dumps(result))


def test_clip_session_has_seven_cards_per_video_without_pres_ctx(run, books):
    assert clips_export(run) == 0
    manifest = manifest_of(run)
    per_video = [r for r in manifest["rows"] if r["field_class"] != "pair"]
    assert len(per_video) == 70
    first = [r["path"] for r in per_video if r["video_id"] == "v00"]
    assert first == [*(f"stage_a:{c}" for c in books.stage_a_ids()),
                     *(f"prep_pres:{e.id}" for e in books.prepared_codes() if e.id != "PRES-CTX"), "venture:ask"]
    assert len(first) == 7 and "prep_pres:PRES-CTX" not in {r["path"] for r in manifest["rows"]}
    page = _page(run)
    assert all(len(v["sections"][0]["cards"]) == 7 for v in page["videos"][:-1])
    # the model keeps PRES-CTX: the record is untouched and the code stays in the books
    assert "PRES-CTX" in [e.id for e in books.prepared_codes()]


TITLED = "Is The Real Elf A Real Business? | Shark Tank US.mp4"      # a real source's name, not <video_id>.mp4


def _titled(run, video_id="v02", name=TITLED, tmp=None):
    experiment = json.loads((run / video_id / "experiment.json").read_text())
    source = (tmp / name) if tmp else Path("/videos") / name
    (run / video_id / "experiment.json").write_text(json.dumps({**experiment, "source": str(source)}))
    return source


def test_clip_page_plays_the_full_video_with_its_window(run, books):
    _two_spans(run)
    _titled(run)
    assert clips_export(run) == 0
    manifest = manifest_of(run)
    assert "clip_files" not in manifest
    # the file is the experiment.json source base name, not one made from the video id
    assert manifest["video_files"] == {v: TITLED if v == "v02" else f"{v}.mp4" for v in VIDEOS}
    assert next(v for v in _page(run)["videos"] if v["video_id"] == "v02")["file_name"] == TITLED
    assert all(TITLED in r["context"] for r in manifest["rows"] if r.get("video_id") == "v02")
    assert any(TITLED in r["context"] for r in manifest["rows"] if r["field_class"] == "pair")
    assert manifest["windows"]["v00"] == "0:05-0:30, 0:40-1:00" and manifest["windows"]["v01"] == "0:05-1:00"
    page = _page(run)
    videos = {v["video_id"]: v for v in page["videos"][:-1]}
    assert videos["v00"]["file_name"] == "v00.mp4" and videos["v00"]["window"] == "0:05-0:30, 0:40-1:00"
    assert videos["v01"]["window"] == "0:05-1:00"
    # the sheet's context names the full video and its window, not a clip
    rows = [r for r in manifest["rows"] if r.get("video_id") == "v00"]
    assert all("v00.mp4" in r["context"] and "0:05-0:30, 0:40-1:00" in r["context"] for r in rows)
    assert "__prepared" not in json.dumps(manifest) and "__prepared" not in json.dumps(page)
    # the per-video intro says to watch the window only
    assert "the times shown" in videos["v00"]["sections"][0]["intro"]


def test_pair_entries_carry_both_windows(run, books):
    _two_spans(run)
    assert clips_export(run) == 0
    pairs = _page(run)["videos"][-1]["pairs"]
    windows = manifest_of(run)["windows"]
    for p in pairs:
        assert p["left_file"] == f"{p['left']}.mp4" and p["right_file"] == f"{p['right']}.mp4"
        assert (p["left_window"], p["right_window"]) == (windows[p["left"]], windows[p["right"]])
    rows = [r for r in manifest_of(run)["rows"] if r["field_class"] == "pair"]
    assert all(f"{r['left']}.mp4 ({windows[r['left']]})" in r["context"] for r in rows)


def test_an_old_manifest_with_clip_files_still_loads(run, books):
    assert clips_export(run) == 0
    manifest = manifest_of(run)
    old = {k: v for k, v in manifest.items() if k not in ("video_files", "windows")}
    old["clip_files"] = {v: review_clips.clip_file(v) for v in manifest["videos"]}
    page = review_page.clips_page_model(run, old, "ana", books, load_labels(load_release()))
    assert page["videos"][0]["file_name"].endswith("__prepared.mp4") and page["videos"][0]["window"] == ""
    assert page["videos"][-1]["pairs"][0]["left_file"].endswith("__prepared.mp4")


def test_pres_evid_allows_bands_on_the_coder_sheet(run, books):
    from pipeline.schemas.stage_b_panel import PanelRecord
    from pipeline.stage_b.review_items import EVID_BANDS, allowed_values
    assert EVID_BANDS == ("0", "1-2", "3-5", "6+")
    record = PanelRecord.model_validate(json.loads((run / "v00" / "results" / "primary.json").read_text())["record"])
    assert allowed_values("prep_pres:PRES-EVID", record, books, coder=True) == EVID_BANDS
    assert allowed_values("prep_pres:PRES-EVID", record, books) is None                 # the record keeps a count
    assert allowed_values("prep_pres:PRES-SUBS", record, books, coder=True) == ("1", "2", "3", "4", "5")
    assert clips_export(run) == 0
    evid = next(r for r in manifest_of(run)["rows"] if r["path"] == "prep_pres:PRES-EVID")
    assert evid["allowed_values"] == list(EVID_BANDS) and evid["primary"] == "5"      # the model's count, as recorded
    card = next(c for c in _page(run)["videos"][0]["sections"][0]["cards"] if c["path"] == "prep_pres:PRES-EVID")
    assert card["values"] == list(EVID_BANDS) and card["input"] is None
    assert card["q"] == ("Roughly how many claims in the prepared presentation are backed by a checkable specific "
                         "(a figure with its unit, a named customer or partner, a dated event, a described test)?")


def test_evid_band():
    from pipeline.stage_b.review_items import evid_band
    assert [evid_band(n) for n in (0, 1, 2, 3, 5, 6, 14)] == ["0", "1-2", "1-2", "3-5", "3-5", "6+", "6+"]
    assert evid_band("4") == "3-5"
    with pytest.raises(ValueError):
        evid_band(-1)


def test_pres_evid_agreement_compares_bands(run, books, capsys):
    from pipeline.stage_b.review_apply import _figures, agrees, within_one
    assert agrees("prep_pres:PRES-EVID", "3-5", "4") and agrees("prep_pres:PRES-EVID", "4", "5")
    assert not agrees("prep_pres:PRES-EVID", "1-2", "4") and not agrees("prep_pres:PRES-SUBS", "3", "4")
    assert within_one("1-2", "4", "prep_pres:PRES-EVID") and not within_one("0", "4", "prep_pres:PRES-EVID")
    # alpha over the four ordered bands: counts within a band are one value
    items = [(h, m, None, "prep_pres:PRES-EVID", "ordinal")
             for h, m in [("3-5", "4"), ("0", "0"), ("6+", "9"), ("1-2", "2"), ("3-5", "5")]]
    figures = _figures(items)
    assert figures["human_vs_primary"]["agree"] == 1.0 and figures["alpha"] == 1.0
    # end to end: a coder's band against the model's count (the fixture's PRES-EVID is 5)
    assert clips_export(run) == 0
    sheet = fill_session(run, "ana", answer=lambda r: r["primary"],
                         rating=lambda r: "3-5" if r["path"] == "prep_pres:PRES-EVID" else None)
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 0
    line = next(l for l in log_of(run / "v00" / "review" / "review_log.jsonl") if l["path"] == "prep_pres:PRES-EVID")
    assert line["new"] == "3-5" and line["primary"] == "5" and line["level"] == "ordinal"
    agreement = json.loads((run / "v00" / "review" / "human_agreement.json").read_text())
    prep = agreement["by_field_class"]["prep_pres"]
    assert prep["n"] == 1 and prep["human_vs_primary"] == {"agree": 1.0, "n": 1}       # "3-5" against 5: one band
    assert prep["within_one"]["human_vs_primary"] == {"agree": 1.0, "n": 1}
    assert not (run / "v00" / "results" / "reviewed.json").exists()
    record = json.loads((run / "v00" / "results" / "primary.json").read_text())["record"]
    assert next(p for p in record["prepared"] if p["code"] == "PRES-EVID")["value"] == "5"      # the count stays


def test_an_adjudicator_writes_a_count_not_a_band(books):
    from pipeline.schemas.stage_b_panel import PanelRecord
    from pipeline.stage_b.review_apply import Correction, ReviewError, apply_corrections
    from tests.panel_fixtures import synthetic_record
    record = PanelRecord.model_validate(synthetic_record())
    with pytest.raises(ReviewError, match="count"):
        apply_corrections(record, [Correction("prep_pres:PRES-EVID", "3-5", "correct", "", "adj")], books)
    written, _ = apply_corrections(record, [Correction("prep_pres:PRES-EVID", "4", "correct", "", "adj")], books)
    assert next(p for p in written["prepared"] if p["code"] == "PRES-EVID")["value"] == "4"


def test_copy_videos_clips_is_refused_with_the_new_instruction(run, tmp_path, capsys):
    assert clips_export(run) == 0
    target = tmp_path / "handout"
    assert rp.main(["copy-videos", "--run", str(run), "--clips", "--to", str(target)]) == 1
    err = capsys.readouterr().err
    assert "full videos" in err and f"copy-videos --run {run} --to" in err and "--training" in err
    assert not target.exists()


def test_copy_videos_copies_the_clip_sessions_full_videos(run, tmp_path, capsys):
    """Without a drawn sample the clip session's own videos are the ones to hand out."""
    for v in VIDEOS:
        source = tmp_path / "sources" / (TITLED if v == "v02" else f"{v}.mp4")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b"x")
        experiment = json.loads((run / v / "experiment.json").read_text())
        (run / v / "experiment.json").write_text(json.dumps({**experiment, "source": str(source)}))
    capsys.readouterr()
    assert clips_export(run) == 0
    out = capsys.readouterr().out
    assert f"copy-videos --run {run} --to" in out and "--clips" not in out and "cut" not in out
    target = tmp_path / "handout"
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(target)]) == 0
    assert sorted(p.name for p in target.iterdir()) == sorted([TITLED] + [f"{v}.mp4" for v in VIDEOS if v != "v02"])
    assert "10 of 10 videos" in capsys.readouterr().out
    assert next(v for v in _page(run)["videos"] if v["video_id"] == "v02")["file_name"] == TITLED   # the page finds it


def test_the_clips_page_has_a_window_button_per_video_and_per_pair_player(run):
    from tests.test_review_page import _lines_of, _node, _script
    _two_spans(run)
    assert clips_export(run) == 0
    pair = _page(run)["videos"][-1]["pairs"][0]
    script = _script()
    got = _node("\n".join([_lines_of(script, "const esc = ", "const winBtn = ", "const pairHtml = "),
                           "const html = pairHtml(" + json.dumps(pair) + ", 0);",
                           "console.log(JSON.stringify([html, winBtn({window: '0:05-0:30, 0:40-1:00'})]));"]))
    html, button = got
    assert button.count("▶") == 1 and 'data-c="0:05"' in button and "0:05-0:30, 0:40-1:00" in button
    assert html.count("<video") == 2 and html.count("▶") == 2
    for side in ("left", "right"):
        assert f'data-c="{pair[side + "_window"].split("-")[0]}"' in html and pair[side + "_window"] in html
        assert pair[side + "_file"] in html
    # each video's clip-session cards open with the window button: the intro card above them, rendered in node
    video = _page(run)["videos"][0]
    intro = _node("\n".join([f"const PAGE = {{hints: {{pitch_intro: 'unused'}}}};",
                              _lines_of(script, "const esc = ", "const winBtn = ", "const introHtml = "),
                              f"console.log(JSON.stringify(introHtml({json.dumps(video)}, "
                              f"{json.dumps(video['sections'][0])}, 0)));"]))
    assert intro.count("<button") == 1 and f'data-c="{video["window"].split("-")[0]}"' in intro and video["window"] in intro
    assert "the times shown" in intro and "unused" not in intro


def test_a_player_pauses_once_at_the_windows_end(run):
    """The page's one-time pause, run in node over a stand-in player: playing across the end of a span pauses it the
    first time only; a seek past the end does not."""
    from tests.test_review_page import _lines_of, _node, _script
    script = _script()
    block = lambda start: "\n".join(script.splitlines()[script.splitlines().index(start):][
        :next(i for i, l in enumerate(script.splitlines()[script.splitlines().index(start):]) if l == "}") + 1])
    stub = """
const ev = {}; let pauses = 0;
const pv = {currentTime: 0, paused: false, addEventListener: (k, f) => { (ev[k] = ev[k] || []).push(f); },
            pause: () => { pauses++; pv.paused = true; }};
const fire = k => (ev[k] || []).forEach(f => f());
const at = t => { pv.currentTime = t; fire('timeupdate'); };
const seekTo = t => { fire('seeking'); pv.currentTime = t; fire('timeupdate'); fire('seeked'); };
"""
    source = "\n".join([stub, _lines_of(script, "const cs = ", "const pausedAt = "), block("function pauseOnce(pv, key, span){"),
                        "pauseOnce(pv, 'p1', '0:05-0:30, 0:40-1:00');",
                        "seekTo(5); for(let t=5; t<=31; t+=0.25) at(t); const first = pauses;",       # across 0:30
                        "pv.paused = false; seekTo(20); for(let t=20; t<=31; t+=0.25) at(t); const again = pauses;",
                        "seekTo(70); const skipped = pauses;",                                        # past 1:00
                        "seekTo(50); for(let t=50; t<=61; t+=0.25) at(t);",
                        "console.log(JSON.stringify([first, again, skipped, pauses]));"])
    assert _node(source) == [1, 1, 1, 2]


# ---- Task 7c, fix round 1 -------------------------------------------------------------------------------------------

def test_what_a_coder_reads_in_the_clip_session_names_the_prepared_presentation_not_a_clip(run, books):
    """I1: the intros, the pairs' intro and the pair answers' meanings speak of the full video and its prepared
    presentation between the times shown; the protocol paragraph the model received (pitch_intro) is not shown."""
    assert clips_export(run) == 0
    page = _page(run)
    labels = load_labels(load_release())
    shown = [s["intro"] for v in page["videos"] for s in v["sections"]]
    shown += [d for v in page["videos"] for s in v["sections"] for c in s["cards"] if c["field_class"] == "pair"
              for d in c["defs"].values()]
    assert shown and not [t for t in shown if "clip" in t.lower()]
    assert all(s["intro"] == labels.hints["protocol_for_coders"] for v in page["videos"][:-1] for s in v["sections"])
    assert labels.hints["pitch_intro"] not in json.dumps(page["videos"])
    assert "the times shown" in labels.hints["protocol_for_coders"] and "full video" in labels.hints["protocol_for_coders"]
    # the model-answers page's pairs and intros say the same
    scores = review_clips.model_scores(run, VIDEOS, list(books.stage_a_ids()))
    walk = review_page.clips_walkthrough(run, manifest_of(run), books, labels, scores)
    reasons = [c["model"]["reason"] for v in walk for s in v["sections"] for c in s["cards"] if c["field_class"] == "pair"]
    assert reasons and not [r for r in reasons if "clip" in r.lower()]
    assert all(s["intro"] == labels.hints["protocol_for_coders"] for v in walk[:-1] for s in v["sections"])


def test_a_coders_band_may_be_typed_with_a_dash_or_spaces_and_a_count_is_reported(run, books, capsys):
    """M1: an en dash or spaces fold into the band; a bare count is accepted, banded for agreement and reported; any
    other answer is refused naming the bands."""
    assert clips_export(run) == 0
    typed = {"v00": "3–5", "v01": "6 +", "v02": "4"}
    sheet = fill_session(run, "ana", answer=lambda r: None,
                         rating=lambda r: typed.get(r["video_id"]) if r["path"] == "prep_pres:PRES-EVID" else None)
    capsys.readouterr()
    assert clips_import(run, sheet, "ana") == 0
    out = capsys.readouterr().out
    new = {v: next(l for l in log_of(run / v / "review" / "review_log.jsonl") if l["path"] == "prep_pres:PRES-EVID")["new"]
           for v in typed}
    assert new == {"v00": "3-5", "v01": "6+", "v02": "4"}
    assert "v02: prep_pres:PRES-EVID answered as a count, not a band" in out and "v00: prep_pres:PRES-EVID" not in out
    for bad in ("seven", "4.0", "-1", "3-6"):
        sheet = fill_session(run, "ben", answer=lambda r: None,
                             rating=lambda r: bad if r["path"] == "prep_pres:PRES-EVID" and r["video_id"] == "v03" else None)
        assert clips_import(run, sheet, "ben") == 1, bad
        err = capsys.readouterr().err
        assert "takes a band: 0, 1-2, 3-5, 6+" in err and "nothing was written" in err, bad


def test_a_model_missing_reason_is_never_banded(books):
    """M7: the model's `missing:<reason>` stays a reason: no band, a disagreement in the share, no unit in alpha."""
    from pipeline.stage_b.review_apply import _figures, agrees, within_one
    from pipeline.stage_b.review_items import band_rank, banded
    path, missing = "prep_pres:PRES-EVID", "missing:not_applicable"
    assert banded(path, missing) == missing and band_rank(path, missing) == missing
    assert not agrees(path, "0", missing) and not within_one("0", missing, path)
    items = [("0", missing, None, path, "ordinal"), ("1-2", "2", None, path, "ordinal"), ("6+", "7", None, path, "ordinal")]
    figures = _figures(items)
    assert figures["human_vs_primary"] == {"agree": 2 / 3, "n": 3}
    from pipeline.stage_b.review_apply import _alpha_units
    assert _alpha_units(items, 0, 1) == 2


def test_the_design_id_changes_with_the_cards_and_the_page_drops_an_answer_its_card_no_longer_offers(run, books):
    """M2: a session whose cards changed (PRES-CTX dropped) is another design, so its page and sheets start afresh;
    and a stored value a choice card no longer offers (a count where a band is asked) is dropped on load."""
    from tests.test_review_page import _lines_of, _node, _script
    batches = review_clips.pair_design(VIDEOS, seed=4)
    codes = list(books.stage_a_ids())
    seven = [r["path"] for r in review_clips._video_rows(run, "v00", books)]
    eight = [*seven[:5], "prep_pres:PRES-CTX", *seven[5:]]
    assert review_clips.design_id(4, VIDEOS, batches, codes, seven) != review_clips.design_id(4, VIDEOS, batches, codes, eight)
    assert clips_export(run) == 0
    assert manifest_of(run)["design_id"] == review_clips.design_id(4, VIDEOS, batches, codes, seven)
    script = _script()
    cards = [{"path": "a", "input": None, "values": ["0", "1-2"]}, {"path": "b", "input": "choice", "values": ["left", "right"]},
             {"path": "c", "input": "number", "values": []}, {"path": "d", "input": None, "values": ["1", "2"]}]
    got = _node("\n".join([_lines_of(script, "const offered = "),
                            f"const cards = {json.dumps(cards)};",
                            "console.log(JSON.stringify([offered(cards[0], '5'), offered(cards[0], '1-2'), "
                            "offered(cards[1], 'middle'), offered(cards[2], '7'), offered(cards[3], 'missing:not_applicable'), "
                            "offered(cards[3], '')]));"]))
    assert got == [False, True, False, True, True, True]
