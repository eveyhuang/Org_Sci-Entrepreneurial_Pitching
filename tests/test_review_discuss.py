"""The discussion page (2026-10-29, the researcher's request): the researcher and one coder go through the cases where
the coder and Gemini disagree, on the task page's interface. The fixture is a session whose coder MX has imported
answers: structural flags on question-answer turns (one taken back by a later import, one given by another coder),
question concepts in another frame than Gemini's (and one in the same frame) and pair choices against the model's. Nothing the page writes enters the validation."""
import csv
import io
import json
import re
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_b import review_discuss  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_items import SHEET_COLUMNS  # noqa: E402
from tests.test_review_cli import fill, make_run, read_sheet  # noqa: E402
from tests.test_review_clips import scores_for  # noqa: E402
from tests.test_review_page import _lines_of, _node, _script  # noqa: E402
from tests.test_review_tasks import _block, page_of  # noqa: E402

VIDEOS = ["v00", "v01", "v02", "v03"]
CODES = ["pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth"]


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def _answer_all(sheet, edits):
    """Every turn card answered with its first value, a question's concept with Gemini's (operating_efficiency in the
    synthetic record), the first decision too; then `edits` as given."""
    rows = read_sheet(sheet)
    out = {}
    for row in rows:
        if row["path"].endswith("/code:PR-CONCEPT"):
            out[row["path"]] = {"human_value": "operating_efficiency", "resolution": "correct"}
        elif row["path"].startswith("turn:"):
            out[row["path"]] = {"human_value": row["allowed_values"].split(" | ")[0], "resolution": "correct"}
        elif row["path"] == "pitch:first_decision":
            out[row["path"]] = {"human_value": "1:51", "resolution": "correct"}
    out.update(edits)
    fill(sheet, out)


def _import(run, video_id, edits, reviewer="MX"):
    review = run / video_id / "review"
    sheet = review / f"{video_id}__{reviewer}.csv"
    shutil.copy(review / f"{video_id}__review.csv", sheet)
    _answer_all(sheet, edits)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", reviewer,
                    "--video-id", video_id]) == 0


STRUCT = {"resolution": "structural", "human_value": "", "note": "no question here"}
CONCEPT = lambda value: {"human_value": value, "resolution": "correct", "note": ""}     # noqa: E731
# MX's pair choices: the model's on every row but these, where MX chose the other side
AGAINST = {("b01", "pitch_positivity"), ("b02", "pitch_positivity"), ("b03", "expressed_enthusiasm"),
           ("b04", "expressed_enthusiasm"), ("b04", "interpersonal_warmth"), ("b05", "interpersonal_warmth"),
           ("b06", "interpersonal_warmth")}


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, books):
    base = tmp_path_factory.mktemp("discuss")
    for v in VIDEOS:
        make_run(base, books, video_id=v)
    run = base / "run"
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books, VIDEOS)))
    assert rp.main(["session-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", "MX", "AV", "--seed",
                    "4", "--qa-per-video", "2"]) == 0
    # v00: the question of pair (2, 3) flagged, with the not-sure tick on another card; v01: the answer of (4, 5)
    # flagged, and its question's concept in the promotion frame where Gemini's is in the prevention frame
    _import(run, "v00", {"turn:2/code:PR-SEV": STRUCT,
                         "turn:3/code:RS-RESP": {"human_value": "partial", "resolution": "correct",
                                                 "note": "[unsure] half an answer [summary shown]"}})
    _import(run, "v01", {"turn:5/code:RS-EVID": STRUCT, "turn:4/code:PR-CONCEPT": CONCEPT("forecast")})
    # v02: flagged, then taken back by a later import (the latest line decides), with a concept of Gemini's frame;
    # v03: only another coder flagged, and MX's concept of turn 2 is in another frame
    _import(run, "v02", {"turn:4/code:PR-SEV": STRUCT})
    _import(run, "v02", {"turn:2/code:PR-CONCEPT": CONCEPT("execution")})
    _import(run, "v03", {"turn:2/code:PR-SEV": STRUCT}, reviewer="AV")
    _import(run, "v03", {"turn:2/code:PR-CONCEPT": CONCEPT("usage")})
    session = json.loads((run / "review" / "clips" / "export_manifest.json").read_text())
    sheet = run.parent / "clips__MX.csv"
    shutil.copy(run / "review" / "clips" / "clips.csv", sheet)
    other = {"left": "right", "right": "left"}
    fill(sheet, {r["path"]: {"human_value": other[r["primary"]] if (r["batch_id"], r["code"]) in AGAINST
                             else r["primary"], "note": "[close] too close" if r["batch_id"] == "b04" else ""}
                 for r in session["rows"] if r["field_class"] == "pair"})
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "MX"]) == 0
    return run


@pytest.fixture
def run(base_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def discuss(run, *extra):
    return rp.main(["discuss", "--run", str(run), "--coder", "MX", "--pairs", "4", "--seed", "1", *extra])


def _page(run):
    return page_of(run / "review" / "pages" / "MX__discussion.html")


def _manifest(run):
    return json.loads((run / "review" / "discussion" / "MX" / "manifest.json").read_text())


def _tree(run):
    return {p.relative_to(run): p.read_bytes() for p in run.rglob("*") if p.is_file()}


# ---- what the page holds ------------------------------------------------------------------------------------------

def test_the_page_holds_the_flagged_videos_flagged_pairs_and_n_comparisons(run, capsys):
    assert discuss(run) == 0
    out = capsys.readouterr().out
    page = _page(run)
    assert page["mode"] == "discussion" and page["coder"] == "MX"
    videos = [t for t in page["videos"] if t["task"] == "video"]
    compares = [t for t in page["videos"] if t["task"] == "compare"]
    assert page["videos"] == videos + compares                      # the videos first, then the comparisons
    # v02's flag was taken back and its concept is in Gemini's frame; v03's flag is AV's, but its concept differs
    assert sorted(t["video_id"] for t in videos) == ["v00", "v01", "v03"]
    cards = {t["video_id"]: [c["path"] for s in t["sections"] for c in s["cards"]] for t in videos}
    six = lambda q, a: {f"turn:{q}/code:PR-CONCEPT", f"turn:{q}/code:PR-SEV", f"turn:{a}/code:RS-RESP",  # noqa: E731
                        f"turn:{a}/code:RS-EVID", f"turn:{a}/code:PREP-SUBS", f"turn:{a}/code:PREP-COH"}
    assert set(cards["v00"]) == six(2, 3) and len(cards["v00"]) == 6
    assert set(cards["v01"]) == six(4, 5) and len(cards["v01"]) == 6
    assert set(cards["v03"]) == six(2, 3) and len(cards["v03"]) == 6
    flagged = {t["video_id"]: [c["path"] for s in t["sections"] for c in s["cards"] if c["discuss"]["flagged"]]
               for t in videos}
    assert flagged == {"v00": ["turn:2/code:PR-SEV"], "v01": ["turn:5/code:RS-EVID"], "v03": []}
    # N comparison tasks, one row each, every one a row where MX's choice is not the model's
    assert len(compares) == 4
    for t in compares:
        (card,) = t["sections"][0]["cards"]
        ref = card["path"].split(":", 1)[1].split("/")
        assert (ref[0], ref[1]) in AGAINST and t["pair"]["batch_id"] == ref[0]
        assert t["video_id"] == f"compare-{ref[0]}-{ref[1]}"
    # spread over the three dimensions
    assert {c["path"].rsplit("/", 1)[1] for t in compares for c in t["sections"][0]["cards"]} == set(CODES)
    manifest = _manifest(run)
    assert [v["video_id"] for v in manifest["videos"]] == [t["video_id"] for t in videos]
    held = {v["video_id"]: v for v in manifest["videos"]}
    assert held["v00"]["pairs"] == [{"pair": [2, 3], "selected_for": ["structural flag"]}]
    assert held["v00"]["flagged_paths"] == ["turn:2/code:PR-SEV"] and held["v00"]["frame_paths"] == []
    assert held["v01"]["pairs"] == [{"pair": [4, 5], "selected_for": ["frame disagreement", "structural flag"]}]
    assert held["v03"]["pairs"] == [{"pair": [2, 3], "selected_for": ["frame disagreement"]}]
    assert {c["path"] for c in held["v01"]["cards"]} == set(cards["v01"])
    # the videos come in the coder's own order of the session
    order = json.loads((run / "review" / "clips" / "export_manifest.json").read_text())["order"]["MX"]
    assert [t["video_id"] for t in videos] == [v for v in order if v in ("v00", "v01", "v03")]
    assert manifest["comparisons"]["candidates"] == len(AGAINST) and len(manifest["comparisons"]["selected"]) == 4
    assert "random.Random(1)" in manifest["comparisons"]["rule"] and "gap" in manifest["comparisons"]["rule"]
    assert manifest["passes"] == {"v00": "primary", "v01": "primary", "v03": "primary"}
    assert manifest["sheet"] == "MX__discussion.csv"
    assert ("3 video tasks (3 question-answer pairs: 2 with a structural flag, 2 with a frame disagreement (2 concept "
            "cards)), 4 comparison tasks") in out


def test_each_card_carries_the_coders_answer_and_geminis(run):
    assert discuss(run) == 0
    page = _page(run)
    v00 = next(t for t in page["videos"] if t["video_id"] == "v00")
    by_path = {c["path"]: c for s in v00["sections"] for c in s["cards"]}
    rows = {r["path"]: r for r in json.loads((run / "v00" / "review" / "export_manifest.json").read_text())
            ["videos"]["v00"]["rows"]}
    for path, c in by_path.items():
        assert c["discuss"]["gemini_value"] == rows[path]["primary"]
        assert c["discuss"]["gemini_other"] == rows[path]["replicate"]
        assert "model" not in c and c["discuss"]["context"] == "v00"
    sev = by_path["turn:2/code:PR-SEV"]["discuss"]["coder"]
    assert (sev["value"], sev["resolution"], sev["note"], sev["cell"]) == ("", "structural", "no question here",
                                                                           "[structural]")
    resp = by_path["turn:3/code:RS-RESP"]["discuss"]["coder"]
    assert (resp["value"], resp["unsure"], resp["summary_shown"], resp["note"], resp["cell"]) == (
        "partial", True, True, "half an answer", "partial [unsure]")
    first = by_path["turn:2/code:PR-CONCEPT"]
    assert first["discuss"]["coder"]["value"] == "operating_efficiency"
    assert first["discuss"]["frames"] == {"coder": "prevention", "gemini": "prevention", "gemini_other": "prevention"}
    session = json.loads((run / "review" / "clips" / "export_manifest.json").read_text())
    pair_rows = {r["path"]: r for r in session["rows"] if r["field_class"] == "pair"}
    for t in (t for t in page["videos"] if t["task"] == "compare"):
        (card,) = t["sections"][0]["cards"]
        d, row = card["discuss"], pair_rows[card["path"]]
        assert d["gemini_value"] == row["primary"] and d["coder"]["value"] == {"left": "right", "right": "left"}[row["primary"]]
        assert f"gap {row['gap']:.2f}" in d["gemini_detail"] and "rank scores: left" in d["gemini_detail"]
        assert card["row"] == row["row"] and d["context"] == f"pair {row['batch_id']}: left {row['left']}, right {row['right']}"
        if row["batch_id"] == "b04":
            assert d["coder"]["close"] and d["coder"]["note"] == "too close" and d["coder"]["cell"].endswith("[close]")


def test_the_draw_spreads_over_the_dimensions_prefers_larger_gaps_and_is_seeded():
    lines = [{"reviewer": "MX", "batch_id": f"b{i:02d}", "code": code, "left": "a", "right": "b", "value": "left",
              "model": "right", "gap": gap, "design_id": "d"}
             for i, (code, gap) in enumerate([(CODES[0], 0.01), (CODES[0], 5.0), (CODES[1], 1.0), (CODES[1], 1.0),
                                              (CODES[2], 2.0)])]
    lines += [{**lines[0], "reviewer": "AV"}, {**lines[0], "batch_id": "b90", "model": None},
              {**lines[0], "batch_id": "b91", "value": "right"}, {**lines[0], "batch_id": "b92", "design_id": "old"}]
    drawn, how = review_discuss.draw_comparisons(lines, "MX", "d", 3, 7, CODES)
    assert sorted(l["code"] for l in drawn) == sorted(CODES) and how["candidates"] == 5
    assert drawn == review_discuss.draw_comparisons(lines, "MX", "d", 3, 7, CODES)[0]
    # the larger gap is drawn far more often than the one 500 times smaller
    picks = [next(l for l in review_discuss.draw_comparisons(lines, "MX", "d", 3, s, CODES)[0] if l["code"] == CODES[0])
             ["gap"] for s in range(200)]
    assert picks.count(5.0) > 190
    everything, _ = review_discuss.draw_comparisons(lines, "MX", "d", 50, 0, CODES)
    assert len(everything) == 5
    # a later answer that agrees with the model takes the row out
    agreed = [*lines, {**lines[1], "value": "right"}]
    assert all(l["batch_id"] != "b01" for l in review_discuss.draw_comparisons(agreed, "MX", "d", 50, 0, CODES)[0])


# ---- the sheet and storage ----------------------------------------------------------------------------------------

STATE_BLANK = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}


def _sheets_source(page, state, tail):
    script = _script()
    start, end = script.index("const cols = PAGE.columns;"), script.index("const AGAIN = ")
    return "\n".join([f"const PAGE = {json.dumps(page)};", f"let state = {json.dumps(state)};",
                      _lines_of(script, "const esc = ", "const st = ", "const answered = "), script[start:end],
                      _block(script, "function sheets(){ const any = vd => Object.values(state[vd.video_id]||{}).some(answered);"),
                      tail])


def test_the_page_writes_one_discussion_sheet_that_no_import_reads(run, capsys):
    assert discuss(run) == 0
    page = _page(run)
    assert _node(_sheets_source(page, {}, "console.log(JSON.stringify(sheets()));")) == []
    v00 = next(t for t in page["videos"] if t["video_id"] == "v00")
    compare = next(t for t in page["videos"] if t["task"] == "compare")
    sev = "turn:2/code:PR-SEV"
    ppath = compare["sections"][0]["cards"][0]["path"]
    state = {"v00": {sev: {**STATE_BLANK, "note": "agreed: not a question"},
                     "turn:3/code:RS-RESP": {**STATE_BLANK, "value": "addresses", "unsure": True, "note": "it does"}},
             compare["video_id"]: {ppath: {**STATE_BLANK, "value": "left", "close": True, "note": "Gemini is right"}}}
    written = _node(_sheets_source(page, state, "console.log(JSON.stringify(sheets()));"))
    assert [name for name, _ in written] == ["MX__discussion.csv"]
    text = written[0][1]
    rows = list(csv.DictReader(io.StringIO(text.lstrip("﻿"))))
    assert list(rows[0]) == [*SHEET_COLUMNS, "coder_value", "gemini_value", "gemini_other_pass"]
    assert len(rows) == sum(len(s["cards"]) for t in page["videos"] for s in t["sections"])
    by = {(r["context"], r["path"]): r for r in rows}
    row = by[("v00", sev)]
    # a reason without an answer is kept: the discussion may keep the coder's answer
    assert (row["human_value"], row["resolution"], row["note"], row["coder_value"]) == ("", "", "agreed: not a question",
                                                                                        "[structural]")
    resp = by[("v00", "turn:3/code:RS-RESP")]
    assert (resp["human_value"], resp["note"], resp["coder_value"]) == ("addresses", "[unsure] it does", "partial [unsure]")
    assert resp["gemini_value"] == "addresses"
    card = compare["sections"][0]["cards"][0]
    prow = by[(card["discuss"]["context"], ppath)]
    assert (prow["human_value"], prow["note"], prow["gemini_value"]) == ("left", "Gemini is right [close]",
                                                                          card["discuss"]["gemini_value"])
    assert prow["coder_value"] == card["discuss"]["coder"]["cell"] and prow["row"] == str(card["row"])
    # no import reads it, and nothing it sees changes the logs
    drop = run.parent / "drop"
    drop.mkdir()
    (drop / "MX__discussion.csv").write_text(text, encoding="utf-8")
    before = _tree(run)
    capsys.readouterr()
    # import-folder skips it (a coder's folder may hold it beside their sheets) and imports nothing from it
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(drop)]) == 0
    assert "skipped: MX__discussion.csv is a discussion sheet" in capsys.readouterr().err
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(drop / "MX__discussion.csv"), "--reviewer", "MX"]) == 1
    assert "discussion sheet" in capsys.readouterr().err
    assert _tree(run) == before


def test_the_page_keeps_its_answers_under_a_key_of_its_own(run):
    assert discuss(run) == 0
    page = _page(run)
    script = _script()
    got = _node("\n".join([f"const PAGE = {json.dumps(page)};", _lines_of(script, "const KEY = "),
                           "console.log(JSON.stringify(KEY));"]))
    assert got == f"discussion:{run.name}:MX"


def test_the_card_shows_the_coder_and_gemini_before_the_inputs(run):
    assert discuss(run) == 0
    page = _page(run)
    v00 = next(t for t in page["videos"] if t["video_id"] == "v00")
    cards = {c["path"]: c for s in v00["sections"] for c in s["cards"]}
    compare = next(t for t in page["videos"] if t["task"] == "compare")
    script = _script()
    got = _node("\n".join([f"const PAGE = {json.dumps(page)};",
                           _lines_of(script, "const esc = ", "const vname = ", "const shown = ", "const discussHtml = "),
                           f"console.log(JSON.stringify([discussHtml({json.dumps(cards['turn:2/code:PR-SEV'])}), "
                           f"discussHtml({json.dumps(cards['turn:3/code:RS-RESP'])}), "
                           f"discussHtml({json.dumps(compare['sections'][0]['cards'][0])})]));"]))
    sev, resp, pair = got
    for html in got:
        assert "MX:" in html and "Gemini:" in html and "After discussion" in html
        assert html.index("MX:") < html.index("Gemini:") < html.index("After discussion")
    assert "structurally wrong" in sev and "flagged" in sev and "no question here" in sev
    assert "not sure" in resp and "half an answer" in resp and "partial" in resp
    assert "gap" in pair and "close" in script.split("const discussHtml = ", 1)[1].split("\n", 1)[0]
    # the template wires it into both kinds of card, and the comparisons' note is the reason
    assert "${c.discuss ? discussHtml(c) : ''}" in script and "REASON_NOTE" in script


# ---- refusals and what is left alone ------------------------------------------------------------------------------

def test_discuss_writes_only_its_page_and_manifest_and_needs_force_again(run, capsys):
    before = _tree(run)
    assert discuss(run) == 0
    after = _tree(run)
    added = {str(p) for p in set(after) - set(before)}
    assert added == {"review/pages/MX__discussion.html", "review/discussion/MX/manifest.json"}
    assert all(after[p] == before[p] for p in before)            # the task pages, the logs and the sheets unchanged
    capsys.readouterr()
    assert discuss(run) == 1 and "--force" in capsys.readouterr().err
    assert discuss(run, "--force") == 0


def test_discuss_refuses_a_coder_with_nothing_imported(run, capsys):
    assert rp.main(["discuss", "--run", str(run), "--coder", "ZZ"]) == 1
    assert "nothing" in capsys.readouterr().err
    assert not (run / "review" / "pages" / "ZZ__discussion.html").exists()


# ---- frame disagreements (the researcher's addition of 2026-10-29) ------------------------------------------------

def test_a_question_concept_in_another_frame_brings_its_pair_with_both_concepts_and_frames(run):
    assert review_discuss.frame("usage") == review_discuss.frame("forecast") == "promotion"
    assert review_discuss.frame("execution") == review_discuss.frame("operating_efficiency") == "prevention"
    assert {review_discuss.frame(c) for c in ("deal_terms", "other", "NONE")} == {"neither"}
    assert review_discuss.frame("") is None and review_discuss.frame(None) is None
    assert discuss(run) == 0
    page = _page(run)
    tasks = {t["video_id"]: t for t in page["videos"] if t["task"] == "video"}
    cards = {(v, c["path"]): c for v, t in tasks.items() for s in t["sections"] for c in s["cards"]}
    v03 = cards[("v03", "turn:2/code:PR-CONCEPT")]["discuss"]
    assert v03["reasons"] == ["frame disagreement"] and not v03["flagged"]
    assert (v03["coder"]["value"], v03["gemini_value"]) == ("usage", "operating_efficiency")
    assert v03["frames"] == {"coder": "promotion", "gemini": "prevention", "gemini_other": "prevention"}
    assert cards[("v03", "turn:2/code:PR-SEV")]["discuss"]["reasons"] == []           # the pair's context
    assert cards[("v01", "turn:4/code:PR-CONCEPT")]["discuss"]["reasons"] == ["frame disagreement"]
    assert cards[("v01", "turn:5/code:RS-EVID")]["discuss"]["reasons"] == ["structural flag"]
    # the same frame (v02: execution against operating_efficiency) is no disagreement
    assert "v02" not in tasks
    held = {v["video_id"]: v for v in _manifest(run)["videos"]}
    assert held["v03"]["frame_paths"] == ["turn:2/code:PR-CONCEPT"]
    assert {c["path"]: c["selected_for"] for c in held["v03"]["cards"]}["turn:2/code:PR-CONCEPT"] == ["frame disagreement"]
    assert {c["path"]: c["selected_for"] for c in held["v03"]["cards"]}["turn:3/code:RS-RESP"] == ["context of its pair"]
    script = _script()
    html = _node("\n".join([f"const PAGE = {json.dumps(page)};",
                            _lines_of(script, "const esc = ", "const vname = ", "const shown = ", "const discussHtml = "),
                            f"console.log(JSON.stringify(discussHtml({json.dumps(cards[('v03', 'turn:2/code:PR-CONCEPT')])})));"]))
    text = " ".join(re.sub(r"<[^>]+>", " ", html).split())
    assert "MX: usage (promotion)" in text and "Gemini: operating efficiency (prevention)" in text
    assert "frames differ" in text
