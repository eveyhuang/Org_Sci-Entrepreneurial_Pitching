"""One coding page per coder, as a task list (plan C, Task 7e; the researcher's "order A", 2026-10-21): each video's
prepared presentation and Q&A as one task, each comparison as its own task as soon as both its videos are done, a
counter at the top. `session-export` writes the clip session and the blind full-video sheets in one step; the page
writes the same sheets as the two separate pages, so `clips-import` and `import-folder` read them unchanged."""
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
from pipeline.stage_b import review_clips, review_page  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_labels import load_labels  # noqa: E402
from tests.test_review_cli import make_run, read_sheet  # noqa: E402
from tests.test_review_clips import TITLED, scores_for  # noqa: E402
from tests.test_review_page import CARD_KEYS, _lines_of, _node, _script  # noqa: E402

VIDEOS = ["v00", "v01", "v02", "v03"]


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, books):
    base = tmp_path_factory.mktemp("tasks")
    for v in VIDEOS:
        make_run(base, books, video_id=v)
    run = base / "run"
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books, VIDEOS)))
    # one video with a real source's title, to give a short title on the comparisons
    experiment = json.loads((run / "v01" / "experiment.json").read_text())
    (run / "v01" / "experiment.json").write_text(json.dumps({**experiment, "source": f"/videos/{TITLED}"}))
    return run


@pytest.fixture
def run(base_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def session_export(run, *extra, coders=("ana", "ben")):
    return rp.main(["session-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", *coders,
                    "--seed", "4", *extra])


def page_of(path):
    return json.loads(path.read_text().split("const PAGE = ", 1)[1].split(";\n", 1)[0])


def manifest_of(run):
    return json.loads((run / "review" / "clips" / "export_manifest.json").read_text())


def entry_of(run, video_id):
    return json.loads((run / video_id / "review" / "export_manifest.json").read_text())["videos"][video_id]


# ---- the order of the tasks ---------------------------------------------------------------------------------------

def test_each_comparison_comes_as_soon_as_both_its_videos_are_done():
    pairs = [{"batch_id": "b01", "left": "A", "right": "B"}, {"batch_id": "b02", "left": "C", "right": "A"},
             {"batch_id": "b03", "left": "B", "right": "C"}]
    assert review_page.task_order(["A", "B", "C"], pairs) == [("video", "A"), ("video", "B"), ("pair", "b01"),
                                                              ("video", "C"), ("pair", "b02"), ("pair", "b03")]
    assert review_page.task_order(["C", "A", "B"], pairs) == [("video", "C"), ("video", "A"), ("pair", "b02"),
                                                              ("video", "B"), ("pair", "b01"), ("pair", "b03")]
    assert review_page.short_title("Is The Real Elf A Real Business? | Shark Tank US") == "Is The Real Elf A Real Business?"
    assert review_page.short_title("v00") == "v00"


# ---- session-export -------------------------------------------------------------------------------------------------

def test_session_export_writes_the_session_the_blind_sheets_and_one_page_per_coder(run, books, capsys):
    assert session_export(run) == 0
    out = capsys.readouterr().out
    manifest = manifest_of(run)
    # four videos: each compared with the three others, so six pairs
    assert manifest["videos"] == VIDEOS and manifest["per_video"] == 3 and len(manifest["pairs"]) == 6
    assert len(read_sheet(run / "review" / "clips" / "clips.csv")) == 4 * 7 + 6 * 3
    assert not (run / "review" / "clips" / "pages").exists()          # no clip-session page beside the task pages
    for video_id in VIDEOS:
        entry = entry_of(run, video_id)
        assert entry["blind"] and (run / video_id / "review" / f"{video_id}__review.csv").exists()
        paths = [r["path"] for r in entry["rows"]]
        assert "pitch:first_decision" in paths and not [p for p in paths if p.startswith(("stage_a:", "venture:"))]
    for coder in ("ana", "ben"):
        page = page_of(run / "review" / "pages" / f"{coder}__tasks.html")
        assert page["mode"] == review_page.TASKS and page["design_id"] == manifest["design_id"] and page["codebook"]
        assert page["list_title"] == "Your tasks" and page["banner"] == ""
        order = manifest["order"][coder]
        expected = [x if kind == "video" else f"compare-{x}" for kind, x in review_page.task_order(order, manifest["pairs"])]
        assert [t["video_id"] for t in page["videos"]] == expected
        assert f"{coder}__tasks.html" in out
    assert "6 comparisons" in out and "copy-videos" in out


def test_session_export_refuses_without_the_rank_scores_and_writes_nothing(run, capsys):
    (run / "ranking" / "scores.json").unlink()
    assert session_export(run) == 1
    assert "run_stage_a_ranking.py" in capsys.readouterr().err
    assert not (run / "review").exists() and not (run / "v00" / "review").exists()


def test_session_export_over_an_earlier_one_needs_force_and_keeps_a_blind_export(run, capsys):
    assert session_export(run) == 0
    sheet = (run / "v00" / "review" / "v00__review.csv").read_bytes()
    assert session_export(run) == 1 and "--force" in capsys.readouterr().err
    # a blind export the sample's draw wrote stays as it is; one that is not blind is refused
    shutil.rmtree(run / "review")
    assert session_export(run) == 0 and "its blind full-video sheet is kept" in capsys.readouterr().out
    assert (run / "v00" / "review" / "v00__review.csv").read_bytes() == sheet
    shutil.rmtree(run / "review")
    assert rp.main(["export", "--run", str(run), "--video-ids", "v00", "--force"]) == 0       # an adjudication sheet
    capsys.readouterr()
    assert session_export(run) == 1 and "not blind" in capsys.readouterr().err
    assert not (run / "review" / "clips").exists()
    assert session_export(run, "--force") == 0 and entry_of(run, "v00")["blind"]


def test_two_videos_make_one_comparison_and_an_odd_number_is_refused(run, capsys):
    assert rp.main(["session-export", "--run", str(run), "--video-ids", "v00", "v01", "--coders", "evey"]) == 0
    page = page_of(run / "review" / "pages" / "evey__tasks.html")
    assert [t["task"] for t in page["videos"]] == ["video", "video", "compare"]
    capsys.readouterr()
    assert rp.main(["session-export", "--run", str(run), "--video-ids", "v00", "v01", "v02", "--coders", "evey",
                    "--force"]) == 1
    assert "even number of videos" in capsys.readouterr().err


# ---- the page model -------------------------------------------------------------------------------------------------

def test_a_video_task_carries_its_seven_presentation_cards_then_its_full_video_cards(run, books):
    assert session_export(run) == 0
    manifest = manifest_of(run)
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    labels = load_labels(load_release())
    for task in [t for t in page["videos"] if t["task"] == "video"]:
        video_id = task["video_id"]
        assert task["title"].endswith(" — presentation and Q&A") and task["file_name"] == manifest["video_files"][video_id]
        assert task["window"] == manifest["windows"][video_id]
        prepared, qa = task["sections"]
        assert (prepared["key"], prepared["title"], prepared["sheet"]) == ("prepared", "The prepared presentation", "session")
        assert (qa["key"], qa["title"], qa["sheet"]) == ("qa", "The questions and answers", "video")
        assert prepared["intro"] == labels.hints["protocol_for_coders"]
        # the seven cards are the session's rows of this video, numbered as its sheet
        rows = [r for r in manifest["rows"] if r.get("video_id") == video_id]
        assert [(c["row"], c["path"]) for c in prepared["cards"]] == [(r["row"], r["path"]) for r in rows]
        assert len(prepared["cards"]) == 7
        # the full-video cards: the blind sheet's rows, in clock order with the first decision last
        entry = entry_of(run, video_id)
        assert sorted(c["row"] for c in qa["cards"]) == list(range(1, len(entry["rows"]) + 1))
        assert {c["path"] for c in qa["cards"]} == {r["path"] for r in entry["rows"]}
        assert qa["cards"][-1]["path"] == "pitch:first_decision"
        starts = [c["start"] for c in qa["cards"][:-1]]
        assert starts == sorted(starts, key=lambda c: tuple(map(int, c.split(":"))))
        assert all(set(c) == CARD_KEYS for s in task["sections"] for c in s["cards"])
    assert '"gap"' not in json.dumps(page) and '"model"' not in json.dumps(page)


def test_a_comparison_task_carries_one_pair_with_both_windows(run, books):
    assert session_export(run) == 0
    manifest = manifest_of(run)
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    compares = [t for t in page["videos"] if t["task"] == "compare"]
    assert len(compares) == 6
    for task in compares:
        batch = task["video_id"].removeprefix("compare-")
        p = next(p for p in manifest["pairs"] if p["batch_id"] == batch)
        pair = task["pair"]
        assert (pair["left"], pair["right"]) == (p["left"], p["right"]) and task["videos"] == [p["left"], p["right"]]
        assert pair["left_file"] == manifest["video_files"][p["left"]]
        assert pair["left_window"] == manifest["windows"][p["left"]] and pair["right_window"] == manifest["windows"][p["right"]]
        assert task["title"] == f"Compare: {pair['left_title']} vs {pair['right_title']}"
        [section] = task["sections"]
        assert section["sheet"] == "session" and section["key"] == "compare"
        cards = section["cards"]
        assert [c["path"] for c in cards] == [f"pair:{batch}/{code}" for code in books.stage_a_ids()]
        assert [c["q"] for c in cards] == ["Which pitch comes across as more positive?",
                                           "Which pitch comes across as more enthusiastic?",
                                           "Which pitch comes across as warmer?"]
        assert all(c["values"] == ["left", "right"] and c["input"] == "choice" and c["gloss"] for c in cards)
        # the gloss is the item's definition from the Stage A codebook, its first sentence
        assert cards[0]["gloss"] == books.stage_a.by_id("pitch_positivity").definition.split(". ")[0] + "."
    titled = next(t for t in compares if "v01" in t["videos"])
    side = "left" if titled["pair"]["left"] == "v01" else "right"
    assert titled["pair"][f"{side}_title"] == "Is The Real Elf A Real Business?"
    # the training page: one pair three times, told apart by a count
    labels = load_labels(load_release())
    two = review_clips.build_manifest(run, ["v00", "v01"], ["ana"], 3, books, "x", {}, training=True)
    model = review_page.tasks_page_model(run, two, "ana", books, labels, training=True)
    titles = [t["title"] for t in model["videos"] if t["task"] == "compare"]
    assert len(titles) == 3 and titles[1].endswith(" (2)") and titles[2].endswith(" (3)")
    assert model["mode"] == review_page.TRAINING_TASKS and model["banner"] == review_page.TRAINING_BANNER


def test_nothing_a_coder_reads_on_the_task_page_says_clip_or_which_shows_more(run):
    assert session_export(run) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    shown = [page["heading"], page["list_title"], *page["hints"].values()]
    for task in page["videos"]:
        shown += [task["title"], *(s["title"] for s in task["sections"]), *(s["intro"] or "" for s in task["sections"])]
        for s in task["sections"]:
            for c in s["cards"]:
                shown += [c["q"], c["gloss"], c["structural"], *c["defs"].values(), *c["value_labels"].values()]
    text = " ".join(shown).lower()
    assert "clip" not in text and "which shows more" not in text


# ---- the template -------------------------------------------------------------------------------------------------

def _source(page, state, cur, tail):
    script = _script()
    start, end = script.index("const cols = PAGE.columns;"), script.index("const AGAIN = ")
    block = lambda first: _block(script, first)        # noqa: E731
    return "\n".join([f"const PAGE = {review_page.embed_json(page)};", f"let state = {json.dumps(state)};",
                      f"let cur = {json.dumps(cur)};",
                      _lines_of(script, "const esc = ", "const st = ", "const answered = ", "const needsName = ",
                                "const complete = ", "const tickOnly = ", "function counts(vd)", "function counterText(",
                                "function taskListHtml("),
                      script[start:end], block("function sheets(){ const any = vd => Object.values(state[vd.video_id]||{}).some(answered);"),
                      tail])


def _block(script, first):
    """A multi-line definition of the template's script, from its first line to the line that is only "}"."""
    lines = script.splitlines()[script.splitlines().index(first):]
    return "\n".join(lines[:lines.index("}") + 1])


BLANK = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}


def test_the_template_renders_the_task_list_and_the_counter(run):
    assert session_export(run) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    first, second = page["videos"][0], page["videos"][1]
    total = sum(len(s["cards"]) for t in page["videos"] for s in t["sections"])
    # the first task finished, one card of the second answered; the second task is open
    state = {first["video_id"]: {c["path"]: {**BLANK, "value": (c["values"] or ["1:00"])[0]}
                                 for s in first["sections"] for c in s["cards"]},
             second["video_id"]: {second["sections"][0]["cards"][0]["path"]: {**BLANK, "unclear": True}}}
    done = len(state[first["video_id"]]) + 1
    got = _node(_source(page, state, second["video_id"], "console.log(JSON.stringify([counterText(), taskListHtml()]));"))
    counter, html = got
    assert counter == f"Task 2 of {len(page['videos'])} · {done} of {total} cards"
    rows = re.findall(r'<div class="vitem ([^"]*)" data-v="([^"]+)">(.*?)</div>', html)
    assert [r[1] for r in rows] == [t["video_id"] for t in page["videos"]]
    assert "✓" in rows[0][2] and "✓" not in rows[1][2]
    assert rows[1][0].split() == ["cur", "next"]                  # open, and the first unfinished task
    assert "next" not in rows[0][0] and "cur" not in rows[0][0]
    assert "1." in rows[0][2] and "presentation and Q&amp;A" in rows[0][2]
    visible = re.sub(r"<[^>]+>", " ", html).lower()
    assert "clip" not in visible and "which shows more" not in visible


def test_the_comparison_block_renders_three_rows_one_cant_judge_tick_and_one_note(run):
    assert session_export(run) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    task = next(t for t in page["videos"] if t["task"] == "compare")
    script = _script()
    section = task["sections"][0]
    xs = [{**BLANK, "value": "left", "close": True}, {**BLANK}, {**BLANK}]
    got = _node("\n".join([f"const PAGE = {review_page.embed_json(page)};",
                           _lines_of(script, "const esc = ", "const vname = ", "const chosenHtml = ", "const pickerHtml = "),
                           _block(script, "function controlHtml(c, x){"),
                           _lines_of(script, "const duoHtml = ", "const PAIR_NOTE = "),
                           "\n".join(l for l in script.splitlines()[script.splitlines().index(next(
                               l for l in script.splitlines() if l.startswith("const compareHtml = "))):][:2]),
                           f"console.log(JSON.stringify([duoHtml({json.dumps(task['pair'])}), "
                           f"compareHtml({json.dumps(section)}, {json.dumps(xs)})]));"]))
    duo, block = got
    assert duo.count("<video") == 2 and duo.count("▶") == 2
    for side in ("left", "right"):
        assert task["pair"][f"{side}_window"] in duo and task["pair"][f"{side}_file"] in duo
        assert esc_(task["pair"][f"{side}_title"]) in duo
    assert block.count('class="crow"') == 3 and block.count('class="close"') == 3
    assert block.count('class="cantjudge"') == 1 and block.count("<textarea") == 1
    assert "Which pitch comes across as warmer?" in block and section["cards"][0]["gloss"] in block
    assert 'class="opt on" data-v="left"' in block and "checked" in block.split('class="cantjudge"')[0]
    visible = re.sub(r"<[^>]+>", " ", duo + block).lower()
    assert "clip" not in visible and "which shows more" not in visible and "can't judge" in visible


def esc_(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def test_a_folder_without_the_file_says_which_folder_to_pick_and_offers_the_file():
    script = _script()
    line = _lines_of(script, "const NOT_IN_FOLDER = ")
    text = _node(line + "\nconsole.log(JSON.stringify(NOT_IN_FOLDER('a | b.mp4')));")
    assert "a | b.mp4" in text and "the folder that directly holds the .mp4 files" in text
    assert "Choose this video's file…" in text
    assert "alert(NOT_IN_FOLDER(video().file_name))" in script


# ---- the sheets the page writes -----------------------------------------------------------------------------------

def _answers(task):
    """A coder's answers on one task: the first value of every card with values, a clock, an ask; a pair's first row
    left with close, its second right, its third left blank, and its can't-judge tick off."""
    out = {}
    for s in task["sections"]:
        for i, c in enumerate(s["cards"]):
            if c["input"] == "choice":
                if i < 2:
                    out[c["path"]] = {**BLANK, "value": ["left", "right"][i], "close": i == 0, "note": "hard one"}
            elif c["input"] == "clock":
                out[c["path"]] = {**BLANK, "value": "1:50"}
            elif c["input"] == "ask":
                out[c["path"]] = {**BLANK, "usd": "200000", "pct": "10", "value": "200000;10"}
            elif c["values"]:
                out[c["path"]] = {**BLANK, "value": c["values"][0]}
    return out


def test_the_sheets_the_task_page_writes_are_the_separate_pages_sheets_and_import(run, books, capsys):
    assert session_export(run, coders=("ana",)) == 0
    manifest = manifest_of(run)
    labels = load_labels(load_release())
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    state = {}
    for task in page["videos"]:
        state[task["video_id"]] = _answers(task)
    tail = "console.log(JSON.stringify(sheets()));"
    written = dict(_node(_source(page, state, page["videos"][0]["video_id"], tail)))
    assert set(written) == {"clips__ana.csv", *(f"{v}__ana.csv" for v in VIDEOS)}
    # the separate pages (the clip session's and the full-video page) given the same answers write the same bytes
    clips_page = review_page.clips_page_model(run, manifest, "ana", books, labels)
    old_state = {}
    for vd in clips_page["videos"]:
        if vd["video_id"] == review_clips.PAIRS_ID:
            old_state[vd["video_id"]] = {path: x for t in page["videos"] if t["task"] == "compare"
                                         for path, x in state[t["video_id"]].items()}
        else:
            old_state[vd["video_id"]] = {c["path"]: state[vd["video_id"]][c["path"]] for c in vd["sections"][0]["cards"]
                                         if c["path"] in state[vd["video_id"]]}
    old = dict(_node(_source(clips_page, old_state, clips_page["videos"][0]["video_id"], tail)))
    assert old == {"clips__ana.csv": written["clips__ana.csv"]}
    full_page = review_page.page_model(run, list(manifest["order"]["ana"]), "ana", books, labels)
    full = dict(_node(_source(full_page, state, full_page["videos"][0]["video_id"], tail)))
    assert full == {name: text for name, text in written.items() if name != "clips__ana.csv"}
    # and the imports take them unchanged
    drop = run.parent / "drop"
    drop.mkdir()
    for name, text in written.items():
        (drop / name).write_text(text, encoding="utf-8")
    capsys.readouterr()
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(drop / "clips__ana.csv"), "--reviewer", "ana"]) == 0
    out = capsys.readouterr().out
    assert "12 pairs answered" in out and "6 unanswered" in out
    pairs = [json.loads(l) for l in (run / "review" / "clips" / "pairs_log.jsonl").read_text().splitlines()]
    assert {(l["value"], l["close"], l["note"]) for l in pairs} == {("left", True, "hard one"), ("right", False, "hard one")}
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(drop)]) == 0
    assert "4 of 4 sheets imported" in capsys.readouterr().out
    for video_id in VIDEOS:
        log = [json.loads(l) for l in (run / video_id / "review" / "review_log.jsonl").read_text().splitlines()]
        coded = {l["path"] for l in log if not l.get("implied")}
        assert "pitch:first_decision" in coded and "stage_a:pitch_positivity" in coded and "venture:ask" in coded


def test_a_cant_judge_pair_is_written_as_unclear_on_its_three_rows_and_imported_as_unanswered(run, capsys):
    assert session_export(run, coders=("ana",)) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    task = next(t for t in page["videos"] if t["task"] == "compare")
    paths = [c["path"] for c in task["sections"][0]["cards"]]
    state = {task["video_id"]: {path: {**BLANK, "unclear": True, "note": "one video will not play"} for path in paths}}
    # one rated card of a video, as a sheet with only unanswered pairs is no review for the import
    first = next(t for t in page["videos"] if t["task"] == "video")
    state[first["video_id"]] = {first["sections"][0]["cards"][0]["path"]: {**BLANK, "value": "3"}}
    written = dict(_node(_source(page, state, task["video_id"], "console.log(JSON.stringify(sheets()));")))
    assert set(written) == {"clips__ana.csv"}
    sheet = run.parent / "clips__ana.csv"
    sheet.write_text(written["clips__ana.csv"], encoding="utf-8")
    rows = {r["path"]: r for r in read_sheet(sheet)}
    assert all((rows[p]["resolution"], rows[p]["human_value"], rows[p]["note"]) ==
               ("unclear", "", "one video will not play") for p in paths)
    capsys.readouterr()
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    out = capsys.readouterr().out
    assert "0 pairs answered" in out and "18 unanswered" in out



def test_the_coders_zip_of_sheets_imports_as_the_folder_it_holds(run, books, capsys, tmp_path):
    """The page's one download (<coder>__sheets.zip, written by the template's zipOf) goes to clips-import and
    import-folder as it is; so does the zip of the folder Safari unpacked it into, compressed again in the Finder."""
    import zipfile
    from tests.test_review_page import _zip_bytes
    assert session_export(run, coders=("ana",)) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    state = {task["video_id"]: _answers(task) for task in page["videos"]}
    written = _node(_source(page, state, page["videos"][0]["video_id"], "console.log(JSON.stringify(sheets()));"))
    archive = _zip_bytes(written, tmp_path).rename(tmp_path / "ana__sheets.zip")
    capsys.readouterr()
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(archive), "--reviewer", "ana"]) == 0
    assert "12 pairs answered" in capsys.readouterr().out
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(archive)]) == 0
    out = capsys.readouterr()
    assert f"4 of 4 sheets imported from {archive}" in out.out and "clips__ana.csv is a clip-session sheet" in out.err
    for video_id in VIDEOS:
        assert (run / video_id / "review" / "review_log.jsonl").exists()
    # Safari's folder, compressed again: the sheets one folder down, the Finder's __MACOSX beside it
    again = tmp_path / "ana__sheets_folder.zip"
    with zipfile.ZipFile(again, "w") as zf:
        for name, text in written:
            zf.writestr(f"ana__sheets/{name}", text.encode("utf-8"))
            zf.writestr(f"__MACOSX/ana__sheets/._{name}", b"\x00")
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(again), "--reviewer", "ana"]) == 0
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(again)]) == 0
    assert "4 of 4 sheets imported" in capsys.readouterr().out


def test_a_zip_with_a_path_outside_its_folder_is_refused_whole(run, capsys, tmp_path):
    import zipfile
    assert session_export(run, coders=("ana",)) == 0
    for bad in ("../clips__ana.csv", "/tmp/v00__ana.csv", "sheets/../../v00__ana.csv"):
        archive = tmp_path / "ana__sheets.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("v01__ana.csv", "row\n")
            zf.writestr(bad, "row\n")
        capsys.readouterr()
        assert rp.main(["import-folder", "--run", str(run), "--dir", str(archive)]) == 1
        assert "a path outside its folder" in capsys.readouterr().err
        assert rp.main(["clips-import", "--run", str(run), "--sheet", str(archive), "--reviewer", "ana"]) == 1
        assert "a path outside its folder" in capsys.readouterr().err
    assert not (tmp_path.parent / "clips__ana.csv").exists() and not (tmp_path.parent / "v00__ana.csv").exists()
    assert not any((run / v / "review" / "review_log.jsonl").exists() for v in VIDEOS)
    assert not (run / "review" / "clips" / "pairs_log.jsonl").exists()
    # a zip without the coder's session sheet says which file it looked for
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("v01__ana.csv", "row\n")
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(archive), "--reviewer", "ana"]) == 1
    assert f"no clips__ana.csv in {archive}" in capsys.readouterr().err


# ---- training -------------------------------------------------------------------------------------------------------

def test_training_with_clips_writes_the_task_page(tmp_path, books, capsys):
    from tests.test_review_training import clip_training, clip_training_run
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    tdir = run / "review" / "training"
    manifest = json.loads((tdir / "clips" / "export_manifest.json").read_text())
    assert not (tdir / "clips" / "pages").exists()
    for coder in ("ana", "ben"):
        page = page_of(tdir / "pages" / f"{coder}.html")
        assert page["mode"] == review_page.TRAINING_TASKS and page["banner"] == review_page.TRAINING_BANNER
        assert page["design_id"] == manifest["design_id"] and page["codebook"]
        assert [t["task"] for t in page["videos"]] == ["video", "video", "compare", "compare", "compare"]
        assert [t["video_id"] for t in page["videos"][:2]] == manifest["order"][coder]
    assert "training tasks" in capsys.readouterr().out



# ---- fix round 1: re-running a live session --------------------------------------------------------------------------

def _rows(run, video_id):
    return [r["path"] for r in entry_of(run, video_id)["rows"]]


def test_force_draws_each_full_video_sheet_again_with_its_own_seed_not_the_sessions(run, books, capsys):
    """I1, the reviewer's reproduction: the sample's sheets exported with the sample's seed (step 3), the session
    built on them, then the session exported again with --force to add a coder: every sheet keeps its rows."""
    assert rp.main(["export", "--run", str(run), "--sample-videos", "4", "--seed", "20261013", "--blind"]) == 0
    before = {v: _rows(run, v) for v in VIDEOS}
    sheets = {v: (run / v / "review" / f"{v}__review.csv").read_bytes() for v in VIDEOS}
    assert rp.main(["session-export", "--run", str(run), "--coders", "ana", "ben"]) == 0
    assert "is kept (seed 20261013)" in capsys.readouterr().out
    assert {v: _rows(run, v) for v in VIDEOS} == before
    assert rp.main(["session-export", "--run", str(run), "--coders", "ana", "ben", "cy", "--force"]) == 0
    out = capsys.readouterr().out
    assert out.count("drawn again with its own seed 20261013") == 4
    assert {v: _rows(run, v) for v in VIDEOS} == before
    assert {v: (run / v / "review" / f"{v}__review.csv").read_bytes() for v in VIDEOS} == sheets
    assert all(entry_of(run, v)["seed"] == 20261013 for v in VIDEOS)
    # a sample video without a sheet is drawn with its sample draw's seed, as step 3 would have, and the plan recorded
    shutil.rmtree(run / "v02" / "review")
    assert rp.main(["session-export", "--run", str(run), "--coders", "ana", "--force"]) == 0
    assert _rows(run, "v02") == before["v02"] and entry_of(run, "v02")["seed"] == 20261013
    planned = entry_of(run, "v02")["sample"]["planned"]
    assert planned["rows_per_field"] == 60 and planned["n_videos"] == 4 and planned["qa"] == 15


def test_a_session_drawn_with_qa_per_video_records_no_plan_and_a_redraw_drops_a_stale_one(run):
    """Final review I2: `planned` is recorded only when --qa-per-video was not given; a sheet drawn again with --force
    keeps its recorded plan only when that plan gave its number of pairs."""
    assert session_export(run, "--qa-per-video", "1") == 0
    assert all("planned" not in entry_of(run, v)["sample"] and entry_of(run, v)["sample"]["requested"]["qa"] == 1
               for v in VIDEOS)
    # an older export that recorded a plan beside another requested number: the redraw does not carry it
    path = run / "v00" / "review" / "export_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["videos"]["v00"]["sample"]["planned"] = {"rows_per_field": 60, "n_videos": 4, "exchanges": 0, "qa": 15}
    path.write_text(json.dumps(manifest))
    assert session_export(run, "--force") == 0
    assert "planned" not in entry_of(run, "v00")["sample"] and entry_of(run, "v00")["sample"]["requested"]["qa"] == 1


def test_a_new_sheet_outside_the_sample_takes_the_sheet_seed_never_the_pair_seed(run):
    assert session_export(run) == 0                     # --seed 4 seeds the pairs and the orders only
    assert all(entry_of(run, v)["seed"] == 0 for v in VIDEOS)
    shutil.rmtree(run / "review")
    for v in VIDEOS:
        shutil.rmtree(run / v / "review")
    assert session_export(run, "--sheet-seed", "7") == 0
    assert all(entry_of(run, v)["seed"] == 7 for v in VIDEOS)


def test_a_kept_sheet_with_fewer_codes_is_named_and_one_with_a_session_card_is_refused(run, capsys):
    five = ["question:PR-CONCEPT", "question:PR-SEV", "answer:RS-RESP", "answer:RS-EVID", "answer:PREP-SUBS"]
    assert rp.main(["export", "--run", str(run), "--video-ids", *VIDEOS, "--blind", "--qa-codes", *five]) == 0
    capsys.readouterr()
    assert session_export(run) == 0
    out = capsys.readouterr().out
    assert out.count("warning: it was drawn with 5 question-answer codes, not today's 6 (PREP-COH missing)") == 4
    # a sheet carrying a card of the session is refused before anything is written
    shutil.rmtree(run / "review")
    assert rp.main(["export", "--run", str(run), "--video-ids", "v01", "--blind", "--force",
                    "--stage-a-per-video", "all"]) == 0
    capsys.readouterr()
    assert session_export(run) == 1
    assert "stage_a:" in capsys.readouterr().err and not (run / "review").exists()
    assert session_export(run, "--force") == 0 and not [p for p in _rows(run, "v01") if p.startswith("stage_a:")]


def test_page_never_overwrites_the_task_page(run, capsys):
    """I2: `page --coder ana` writes the full-video page as ana.html; the task page is ana__tasks.html."""
    assert session_export(run) == 0
    tasks = (run / "review" / "pages" / "ana__tasks.html").read_bytes()
    assert rp.main(["page", "--run", str(run), "--coder", "ana", "--video-ids", *VIDEOS]) == 0
    assert (run / "review" / "pages" / "ana__tasks.html").read_bytes() == tasks
    assert page_of(run / "review" / "pages" / "ana.html")["mode"] == "validation"
    assert page_of(run / "review" / "pages" / "ana__tasks.html")["mode"] == review_page.TASKS


STORE = """
const kept = {};
const store = {get: k => k in kept ? kept[k] : null, set: (k, v) => { kept[k] = v; return true; }, keys: () => Object.keys(kept)};
"""


def _storage(page, script, tail):
    return "\n".join([STORE, f"const PAGE = {review_page.embed_json(page)};",
                      _lines_of(script, "const KEY = ", "const slotOf = "), _block(script, "function loadState(){"),
                      _block(script, "function saveState(){"), tail])


def test_a_new_pair_design_keeps_the_videos_answers_and_reads_the_first_layout(run):
    """I3: a video's answers are kept under the run, the coder and the video; the comparisons under the design. A
    page of another design opens with every video's answers and none of the old comparisons'; answers kept under the
    first layout (the whole page under the design) are read."""
    assert session_export(run, coders=("ana",)) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    script = _script()
    video = next(t for t in page["videos"] if t["task"] == "video")
    compare = next(t for t in page["videos"] if t["task"] == "compare")
    vpath, ppath = video["sections"][1]["cards"][0]["path"], compare["sections"][0]["cards"][0]["path"]
    state = {video["video_id"]: {vpath: {**BLANK, "value": "2"}}, compare["video_id"]: {ppath: {**BLANK, "value": "left"}}}
    other = {**page, "design_id": "0123456789ab"}
    got = _node(_storage(page, script, f"""
let state = {json.dumps(state)};
saveState();
const same = loadState();
const PAGE2 = {review_page.embed_json(other)};
const again = (() => {{ const PAGE = PAGE2; {_lines_of(script, "const KEY = ", "const slotOf = ")}
  {_block(script, "function loadState(){")} return loadState(); }})();
console.log(JSON.stringify([Object.keys(kept).sort(), same, again]));"""))
    keys, same, again = got
    assert keys == sorted([f"tasks:{page['run']}:ana:video:{video['video_id']}", f"tasks:{page['run']}:ana:design:{page['design_id']}"])
    assert same == state
    assert again == {video["video_id"]: state[video["video_id"]]}           # the comparison belonged to the old design
    # the first layout: the whole page under tasks:<run>:<coder>:<design>
    legacy = {f"tasks:{page['run']}:ana:{page['design_id']}": json.dumps(state)}
    got = _node(_storage(page, script, f"""Object.assign(kept, {json.dumps(legacy)});
const PAGE2 = {review_page.embed_json(other)};
const there = loadState();
const elsewhere = (() => {{ const PAGE = PAGE2; {_lines_of(script, "const KEY = ", "const slotOf = ")}
  {_block(script, "function loadState(){")} return loadState(); }})();
console.log(JSON.stringify([there, elsewhere]));"""))
    assert got == [state, {video["video_id"]: state[video["video_id"]]}]


def test_an_empty_task_page_writes_no_sheet_and_the_pair_tick_and_note_reach_every_row(run):
    assert session_export(run, coders=("ana",)) == 0
    page = page_of(run / "review" / "pages" / "ana__tasks.html")
    assert _node(_source(page, {}, page["videos"][0]["video_id"], "console.log(JSON.stringify(sheets()));")) == []
    script = _script()
    got = _node("\n".join([_lines_of(script, "const pairTick = ", "const pairNote = "),
                           "const xs = [{unclear: false, note: ''}, {unclear: false, note: ''}, {unclear: false, note: ''}];",
                           "pairTick(xs, true); pairNote(xs, 'will not play');",
                           "console.log(JSON.stringify(xs));"]))
    assert got == [{"unclear": True, "note": "will not play"}] * 3
    assert "pairTick(xs, e.target.checked)" in script and "pairNote(xs, e.target.value)" in script
    # the counter is written on every progress(), and a comparison hides the big player
    assert "document.getElementById('taskprog').textContent = counterText()" in script
    assert "document.getElementById('vbox').style.display = document.querySelector('.left .vc').style.display = vd.pair ? 'none' : ''" in script


def test_demo_namespace_does_not_migrate_another_designs_answers():
    script = _script()
    p = {'mode': 'tasks', 'run': 'reference', 'coder': 'Evey', 'design_id': 'a'*12,
         'storage_namespace': 'demo', 'videos': [{'video_id': 'one'}]}
    tail = '''
let state = {one: {answer: 'A'}};
saveState();
const current = loadState();
const PAGE2 = {...PAGE, design_id: 'bbbbbbbbbbbb'};
const changed = (()=>{const PAGE=PAGE2;
''' + _lines_of(script, 'const KEY = ', 'const slotOf = ') + '\n' + _block(script, 'function loadState(){') + '''
return loadState();})();
console.log(JSON.stringify({current,changed}));
'''
    got = _node(_storage(p, script, tail))
    assert got['current'] == {'one': {'answer': 'A'}}
    assert got['changed'] == {}


def test_completed_page_initializes_embedded_answers_and_never_saves():
    script = _script()
    initialization = _lines_of(script, 'let state = ')
    got = _node('''
const PAGE={read_only:true, completed_answers:{one:{field:{value:'3',note:'actual'}}}};
const store={set:()=>{throw Error('wrote draft');}};
function loadState(){throw Error('read draft');}
''' + initialization + '\n' + _block(script, 'function saveState(){') + '''
console.log(JSON.stringify({state,saved:saveState()}));
''')
    assert got['state']['one']['field']['value'] == '3' and got['saved']
