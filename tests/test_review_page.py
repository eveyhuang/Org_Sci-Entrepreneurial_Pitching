import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_b import review_page  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_items import SHEET_COLUMNS  # noqa: E402
from pipeline.stage_b.review_labels import Labels, load_labels  # noqa: E402
from tests.test_review_cli import OLD_SHEET, make_run, read_sheet  # noqa: E402

CARD_KEYS = {"row", "number", "input", "path", "field_class", "kind", "q", "start", "end", "unit_label", "turns",
             "values", "defs", "html_def", "missing", "structural", "entry", "gloss", "value_labels", "name_values"}


HINT_KEYS = {"pitch_intro", "pitch_card", "outcome_card", "prep_pres_card", "cut_card", "pairs_intro", "clip_session_intro",
             "protocol_for_coders", "compare_intro"}


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def blind_export(run, *extra):
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all", "--venture-per-video", "0",
                    *extra]) == 0


def entry_of(vdir):
    return json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"][vdir.name]


def test_the_full_video_page_shows_every_span_of_the_window_as_the_task_page_does(tmp_path, books):
    # final review M8: the window is every span (review_clips.window_text), not the first span only
    from tests.test_review_clips import _two_spans
    run, vdir = make_run(tmp_path, books)
    _two_spans(run, "vid1")
    blind_export(run)
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    assert model["window"] == "0:05-0:30, 0:40-1:00"


def test_the_default_sheet_has_no_exchange_topic_card(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run)
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    kinds = [c["kind"] for c in model["sections"][1]["cards"]]
    assert kinds and "exchange_topic" not in kinds


def test_video_model(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--exchanges-per-video", "4")
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    assert model["video_id"] == "vid1" and model["file_name"] == "vid1.mp4" and model["title"] == "vid1"
    assert model["actors"]["S2"] == "Investor B" and model["actors_first"]["S2"] == "1:20"
    assert model["window"] == "0:05-1:00"
    assert [s["key"] for s in model["sections"]] == ["pitch", "conversation", "outcomes"]
    pitch = model["sections"][0]["cards"]
    assert [c["path"].partition(":")[2] or c["path"] for c in pitch] == [*books.stage_a_ids()]    # no window card
    assert pitch[0]["values"] == ["1", "2", "3", "4", "5"] and set(pitch[0]["defs"]) == {"1", "2", "3", "4", "5"}
    conv = model["sections"][1]["cards"]
    kinds = [c["kind"] for c in conv]
    assert "exchange_topic" in kinds and "closing" not in kinds          # no closing state since v3-dev9
    sev = next(c for c in conv if c["kind"] == "PR-SEV")
    assert sev["q"] == f"How hard does the question in turn {sev['turns'][0]['turn']} push on what the founder has said?"
    assert sev["values"] == ["1", "2", "3"] and sev["turns"][0]["who"]
    assert sev["structural"].startswith("the video at these times")      # not "clip" since Task 7e: the coders see the full video
    topic = next(c for c in conv if c["kind"] == "exchange_topic")
    assert "not_an_exchange" in topic["values"] and "<b>" in topic["html_def"]
    outcomes = model["sections"][2]["cards"]
    assert [c["path"] for c in outcomes] == ["pitch:first_decision"]          # the cut; no outcome card since Task 7d
    assert model["sections"][2]["title"] == "The first decision"            # so headed, not "Outcomes"
    for section in model["sections"]:
        for card in section["cards"]:
            assert set(card) == CARD_KEYS, card["path"]    # no model-side key can reach a blind page
            assert card["number"] == card["row"]            # the number a coder reads is the sheet's row
    assert "customer acquisition cost" not in json.dumps(model)     # the model's own exchange label


def test_card_rows_are_the_rows_of_the_exported_sheet(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run)
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    sheet = {r["path"]: int(r["row"]) for r in read_sheet(vdir / "review" / "vid1__review.csv")}
    cards = [c for s in model["sections"] for c in s["cards"]]
    assert len(cards) == len(sheet) and {c["path"]: c["row"] for c in cards} == sheet


def test_conversation_cards_are_in_clock_order_with_turn_codes_after_their_exchange(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    conv = model["sections"][1]["cards"]
    starts = [c["start"] for c in conv]
    secs = [int(s.split(":")[0]) * 60 + int(s.split(":")[1]) for s in starts]
    assert secs == sorted(secs)
    codes = [c["kind"] for c in conv if c["kind"] in ("PR-CONCEPT", "PR-SEV", "RS-RESP", "RS-EVID", "PREP-SUBS")]
    assert codes[:5] == ["PR-CONCEPT", "PR-SEV", "RS-RESP", "RS-EVID", "PREP-SUBS"]


def test_a_question_answer_card_lists_both_turns_each_with_its_own_summary(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    conv = model["sections"][1]["cards"]
    resp = next(c for c in conv if c["kind"] == "RS-RESP")
    assert len(resp["turns"]) == 2
    question, answer = resp["turns"]
    assert question["turn"] != answer["turn"] and question["summary"] and answer["summary"]
    assert "answer turn" not in question["summary"] and "turn " not in question["summary"]
    assert question["summary"] != answer["summary"]


def test_turns_in_ends_a_summary_at_the_answer_arrow():
    context = ("question turn 3 (1:00-1:10) S2 (Ann): asks about price -> answer turn 4 (1:10-1:20) S4 (Kevin): "
               "replies | turn 5 (1:20-1:30) S2 (Ann): follows up")
    turns = review_page._turns_in(context)
    assert [t["turn"] for t in turns] == ["3", "4", "5"]
    assert turns[0]["summary"] == "asks about price" and turns[1]["summary"] == "replies"


def test_page_model_lists_videos_in_the_order_given(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    make_run(tmp_path, books, "vid2")
    blind_export(run)
    model = review_page.page_model(run, ["vid2", "vid1"], "ana", books, load_labels(load_release()))
    assert model["coder"] == "ana" and [v["video_id"] for v in model["videos"]] == ["vid2", "vid1"]
    assert model["columns"] == rp.COLUMNS == SHEET_COLUMNS and model["release_id"] == load_release().release_id


def test_page_model_refuses_a_video_without_a_blind_export(tmp_path, books):
    run, _ = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run)]) == 0           # not blind
    with pytest.raises(review_page.PageError, match="blind"):
        review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))


def test_render_page_embeds_the_model_and_no_external_resource(tmp_path, books):
    run, _ = make_run(tmp_path, books)
    blind_export(run)
    model = review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))
    html = review_page.render_page(model)
    assert html.startswith("<!doctype html>") and "__PAGE__" not in html
    assert review_page.embed_json(model) in html and "<title>Coding sheets · ana</title>" in html
    assert "http://" not in html and "https://" not in html and "<script src" not in html and "<link " not in html
    for needle in ("showDirectoryPicker", "localStorage", "indexedDB"):
        assert needle in html
    # every storage call is guarded: the template only reaches localStorage through `store` and indexedDB through idb()
    body = html.split("<script>", 1)[1]
    assert body.count("localStorage.") == 2 and body.count("indexedDB.open") == 1


def test_embedded_model_cannot_close_the_script_element(tmp_path, books):
    run, _ = make_run(tmp_path, books)
    blind_export(run)
    model = review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))
    model["videos"][0]["title"] = "a </script><b> caf\u00e9"
    html = review_page.render_page(model)
    after_open = html.split("<script>", 1)[1]
    assert after_open.count("</script>") == 1
    text = review_page.embed_json(model)
    assert "</" not in text and text.isascii() and text in html
    assert json.loads(text) == model


def test_template_chrome_names_no_role_or_corpus_word():
    template = review_page.TEMPLATE.read_text(encoding="utf-8")
    assert "__PAGE__" in template
    for token in ("'pitch'", ".pitch_intro", ".pitch_card"):     # model keys read by the script, not wording
        template = template.replace(token, "")
    for word in ("shark", "founder", "investor", "pitch"):
        assert word not in template.lower(), word


def test_page_model_carries_the_hints_from_the_labels(tmp_path, books):
    run, _ = make_run(tmp_path, books)
    blind_export(run)
    bare = review_page.page_model(run, ["vid1"], "ana", books, Labels({}, {}, {}, {}))
    assert set(bare["hints"]) == HINT_KEYS and "shark" not in json.dumps(bare["hints"])
    given = review_page.page_model(run, ["vid1"], "ana", books, Labels({}, {}, {}, {}, {"pitch_card": "Only this part."}))
    assert given["hints"]["pitch_card"] == "Only this part." and given["hints"]["pitch_intro"] == bare["hints"]["pitch_intro"]
    assert load_labels(load_release()).hints.keys() == HINT_KEYS


def test_page_command_writes_one_file_per_coder_for_the_sample(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books, "vid1")
    make_run(tmp_path, books, "vid2")
    make_run(tmp_path, books, "vid3")
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "2", "--seed", "1"]) == 0
    assert rp.main(["page", "--run", str(run), "--coder", "ana", "--sample"]) == 0
    page = run / "review" / "pages" / "ana.html"
    assert page.exists()
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    for v in chosen:
        assert f'"video_id": "{v}"' in page.read_text()
    assert "ana.html" in capsys.readouterr().out
    unexported = next(v for v in ("vid1", "vid2", "vid3") if v not in chosen)
    assert rp.main(["page", "--run", str(run), "--coder", "ben", "--video-ids", unexported]) == 1


# ---- the "cannot be rated" answer, the sheet the page writes, and the order of the videos -----------

def test_a_turn_code_card_offers_the_missing_reasons_and_a_stage_a_card_does_not(tmp_path, books):
    from pipeline.schemas.stage_b_panel import MISSING_REASONS
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    labels = Labels(missing_reasons={"too_short": "over too quickly"})
    model = review_page.video_model(vdir, entry_of(vdir), books, labels)
    cards = [c for s in model["sections"] for c in s["cards"]]
    subs = next(c for c in cards if c["kind"] == "PREP-SUBS")
    assert [m["value"] for m in subs["missing"]] == [f"missing:{r}" for r in MISSING_REASONS if r != "none"]
    labels_shown = {m["value"]: m["label"] for m in subs["missing"]}
    assert labels_shown["missing:too_short"] == "over too quickly" and labels_shown["missing:not_observable"] == "not observable"
    assert subs["values"] == ["1", "2", "3", "4", "5"]            # the scale is as it was: the reasons come after it
    for card in cards:
        is_code = card["path"].startswith("turn:") and "/code:" in card["path"]
        assert bool(card["missing"]) is is_code, card["path"]
    assert {c["kind"] for c in cards if c["missing"]} == {"PR-CONCEPT", "PR-SEV", "RS-RESP", "RS-EVID", "PREP-SUBS",
                                                          "PREP-COH"}


def _script():
    return review_page.TEMPLATE.read_text(encoding="utf-8").split("<script>", 1)[1].rsplit("</script>", 1)[0]


def test_the_template_writes_a_missing_reason_as_the_value_and_a_note_only_for_an_answered_card():
    script = _script()
    # the note cell is guarded by `answered`: a card left open keeps its note in the browser, not on the sheet
    # a best guess ticked "not sure" is marked first, before the coder's own note (plan C Task 7f)
    assert ("const note = answered(x) ? [unsureOf(c, x) ? '[unsure]' : '', x.note, x.summary_shown?'[summary shown]':'']"
            ".filter(Boolean).join(' ') : '';") in script
    assert script.count("[summary shown]") == 1
    # the control lists the model's reasons and puts the chosen one where a value goes
    assert 'select class="missing"' in script and "c.missing.map(" in script
    assert "missSel.onchange = ()=>{ x.value = missSel.value;" in script


def test_the_template_keeps_its_folder_handles_per_run_and_coder_and_says_how_to_resume_saving():
    script = _script()
    assert "getH(KEY+':dir')" in script and "getH(KEY+':save')" in script
    assert "putH(KEY+':dir', dirHandle)" in script and "putH(KEY+':save', saveHandle)" in script
    assert "'dir'" not in script.replace("KEY+':dir'", "") and "putH('save'" not in script and "getH('save'" not in script
    assert "Click 'Save sheets to a folder' again to keep saving to it" in script
    assert "Answers are kept in this browser." in script and "queryPermission({mode:'readwrite'})" in script


def _node(source):
    import shutil
    import subprocess
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed: the template's script is checked as text only")
    done = subprocess.run([node, "-e", source], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _lines_of(script, *starts):
    """The template's own lines that begin so (its definitions are one per line)."""
    found = [next(l for l in script.splitlines() if l.startswith(start)) for start in starts]
    return "\n".join(found)


def test_the_templates_clock_parser_reads_clocks_of_an_hour_or_more():
    script = _script()
    assert "split(':').reduce(" in _lines_of(script, "const cs = ")
    got = _node(_lines_of(script, "const cs = ") + "\nconsole.log(JSON.stringify(['1:02:03','0:05','12:44','59:59','1:00:00','',null].map(cs)));")
    assert got == [3723, 5, 764, 3599, 3600, 0, 0]


def test_the_sheet_the_template_writes_is_one_its_own_import_takes(tmp_path, books, capsys):
    """The page's sheet writer run in node over a real page model: an opened summary and a note on a card that was
    never answered leave no note cell; a "cannot be rated" answer is written as missing:<reason>; the import takes
    the sheet whole."""
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    model = review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))
    cards = [c for s in model["videos"][0]["sections"] for c in s["cards"]]
    subs = next(c for c in cards if c["kind"] == "PREP-SUBS")
    sev = next(c for c in cards if c["kind"] == "PR-SEV")
    concept = next(c for c in cards if c["kind"] == "PR-CONCEPT")
    cut = next(c for c in cards if c["kind"] == "cut")
    state = {"vid1": {
        subs["path"]: {"value": "missing:too_short", "unclear": False, "structural": False, "note": "two words", "summary_shown": True},
        sev["path"]: {"value": "", "unclear": False, "structural": False, "note": "come back to this", "summary_shown": True},
        concept["path"]: {"value": "", "unclear": True, "structural": False, "note": "inaudible", "summary_shown": False},
        cut["path"]: {"value": "1:51", "unclear": False, "structural": False, "note": "", "summary_shown": False}}}
    script = _script()
    start, end = script.index("const cols = PAGE.columns;"), script.index("const AGAIN = ")
    source = "\n".join([f"const PAGE = {review_page.embed_json(model)};", f"let state = {json.dumps(state)};",
                        _lines_of(script, "const st = ", "const answered = "), script[start:end],
                        "console.log(JSON.stringify(sheetFor(PAGE.videos[0])));"])
    text = _node(source)
    assert text.startswith("﻿") and text.splitlines()[0].lstrip("﻿") == ",".join(rp.COLUMNS)
    sheet = vdir / "review" / "vid1__ana.csv"
    sheet.write_text(text, encoding="utf-8")
    rows = {r["path"]: r for r in read_sheet(sheet)}
    assert rows[subs["path"]]["human_value"] == "missing:too_short" and rows[subs["path"]]["note"] == "two words [summary shown]"
    assert (rows[sev["path"]]["human_value"], rows[sev["path"]]["resolution"], rows[sev["path"]]["note"]) == ("", "", "")
    assert (rows[concept["path"]]["resolution"], rows[concept["path"]]["note"]) == ("unclear", "inaudible")
    assert sum(bool(r["note"]) for r in rows.values()) == 2
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", "--video-id", "vid1"]) == 0
    assert "3 rows reviewed" in capsys.readouterr().out
    log = [json.loads(l) for l in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert next(l for l in log if l["path"] == subs["path"])["new"] == "missing:too_short"


def test_each_coders_page_lists_the_videos_in_an_order_of_its_own_seeded_and_recorded(tmp_path, books, capsys):
    import random
    import re
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4", "vid5", "vid6"):
        make_run(tmp_path, books, v)
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "6", "--seed", "1"]) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]

    def order(coder, *extra):
        capsys.readouterr()
        assert rp.main(["page", "--run", str(run), "--coder", coder, "--sample", *extra]) == 0
        printed = capsys.readouterr().out
        page = (run / "review" / "pages" / f"{coder}.html").read_text()
        model = json.loads(re.search(r"const PAGE = (.*);\n", page).group(1).replace("<\\/", "</"))
        got = [v["video_id"] for v in model["videos"]]
        assert f"video order (seed {model['order_seed']}): {', '.join(got)}" in printed
        return got, model["order_seed"]

    def expected(coder, seed):
        ids = list(chosen)
        random.Random(f"{seed}:{coder}").shuffle(ids)
        return ids

    ana, seed = order("ana")
    ben, _ = order("ben")
    assert seed == 0 and ana == expected("ana", 0) and ben == expected("ben", 0)
    assert sorted(ana) == sorted(ben) == sorted(chosen) and ana != ben
    assert order("ana") == (ana, 0)                                     # the same page again
    again, seed = order("ana", "--seed", "7")
    assert seed == 7 and again == expected("ana", 7) and again != ana
    assert json.loads((run / "review" / "video_sample.json").read_text())["chosen"] == chosen     # the sample is not touched


def test_a_page_is_refused_for_a_name_that_is_not_a_coder_name(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books)
    blind_export(run)
    for name in ("coder1 (1)", "ana ben", "ana.b", "", "ana/../x"):
        assert rp.main(["page", "--run", str(run), "--coder", name, "--video-ids", "vid1"]) == 1
        assert "is not a coder name (letters, digits, - and _ only)" in capsys.readouterr().err
    assert not (run / "review" / "pages").exists()
    assert rp.main(["page", "--run", str(run), "--coder", "coder_1-b", "--video-ids", "vid1"]) == 0


def test_the_outcome_card_shows_the_outcome_glosses_as_its_definition(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, *OLD_SHEET)                   # an old sheet: no outcome card is drawn since Task 7d
    model = review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release()))
    outcome = next(c for s in model["sections"] for c in s["cards"] if c["path"] == "pitch:any_offer")
    assert outcome["entry"] is None and outcome["html_def"] == ""
    assert set(outcome["defs"]) == {"yes", "no", "unknown"} and "never seen deciding" in outcome["defs"]["no"]


# ---- card numbers, inputs, the picker and the kept video (plan C, Task 5) ------------------------------------

def _block(script, start):
    """A multi-line definition of the template: from the line that begins so to the first line that is `}`."""
    lines = script.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith(start))
    j = next(n for n in range(i, len(lines)) if lines[n] == "}")
    return "\n".join(lines[i:j + 1])


def _esc(text):
    """The template's `esc`: &, <, > and " as entities."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _cards(model):
    return [c for v in model["videos"] for s in v["sections"] for c in s["cards"]] if "videos" in model \
        else [c for s in model["sections"] for c in s["cards"]]


def full_export(run):
    assert rp.main(["export", "--run", str(run), "--blind", "--preset", "full", "--rows-per-field", "20",
                    "--field-classes", "prep_pres", "venture", "prepared_window"]) == 0


def test_each_card_names_the_input_it_needs(tmp_path, books):
    run, vdir = make_run(tmp_path / "a", books)
    full_export(run)
    cards = {c["path"]: c for c in _cards(review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release())))}
    # a coding page is a coder's sheet: the count of checkable claims is chosen as a band (Task 7c)
    assert cards["prep_pres:PRES-EVID"]["input"] is None and cards["prep_pres:PRES-EVID"]["values"] == ["0", "1-2", "3-5", "6+"]
    assert cards["prep_pres:PRES-SUBS"]["input"] is None and cards["prep_pres:PRES-SUBS"]["values"] == ["1", "2", "3", "4", "5"]
    assert cards["venture:ask"]["input"] == "ask" and cards["venture:product_form"]["input"] is None
    assert cards["prepared_window"]["input"] is None and cards["prepared_window"]["values"] == []
    # the import takes missing:<reason> for the ask and a prepared code, so their cards offer the reasons
    assert cards["venture:ask"]["missing"] and cards["prep_pres:PRES-EVID"]["missing"]
    assert not cards["venture:product_form"]["missing"] and not cards["prepared_window"]["missing"]
    run2, vdir2 = make_run(tmp_path / "b", books)
    blind_export(run2, *OLD_SHEET)
    cards = {c["path"]: c for c in _cards(review_page.video_model(vdir2, entry_of(vdir2), books, load_labels(load_release())))}
    assert cards["pitch:first_decision"]["input"] == "clock" and cards["pitch:any_offer"]["input"] is None
    assert review_page._input("decision:1/time", books) == "clock"
    assert review_page._input("pair:b01/pitch_positivity", books) == "choice"
    assert review_page._input("turn:4/code:PR-CONCEPT", books) is None


def test_a_pair_card_offers_left_and_right_with_their_meanings_not_the_items_scale(books):
    from tests.panel_fixtures import synthetic_record
    row = {"path": "pair:b01/pitch_positivity", "field_class": "pair", "context": "", "clock_start": "", "clock_end": ""}
    card = review_page._card(row, 7, None, synthetic_record(), books, load_labels(load_release()))
    assert card["input"] == "choice" and card["values"] == ["left", "right"] and card["number"] == card["row"] == 7
    # the release's wording, under "Which pitch comes across as more …?" (plan C Task 7e); the defaults are role-free
    assert card["defs"] == {"left": "the pitch on the left", "right": "the pitch on the right"}
    worded = review_page._card(row, 7, None, synthetic_record(), books, Labels(glosses={"pair": {"left": "links"}}))
    assert worded["defs"] == {"left": "links", "right": "the right video's prepared presentation shows more"}


def test_a_pair_card_has_a_section_on_a_page():
    from pipeline.stage_b.review_labels import section_of
    assert review_page.SECTION_ORDER[:3] == ("pitch", "conversation", "outcomes")
    assert {"clips", "pairs"} <= set(review_page.SECTION_ORDER)
    assert section_of("pair:b01/pitch_positivity") in review_page.SECTION_ORDER


def test_the_pitch_intro_says_where_delivery_is_coded_when_the_sheet_has_no_rating_of_it(tmp_path, books):
    labels = load_labels(load_release())
    run, vdir = make_run(tmp_path / "a", books)
    blind_export(run, "--stage-a-per-video", "0", *OLD_SHEET)        # an old sheet: its window card, nothing rated
    pitch = next(s for s in review_page.video_model(vdir, entry_of(vdir), books, labels)["sections"] if s["key"] == "pitch")
    assert [c["path"] for c in pitch["cards"]] == ["prepared_window"]
    assert pitch["intro"] == labels.hints["clip_session_intro"] and "presentation cards" in pitch["intro"]
    # final review M3: the window card is no longer drawn by default, so the hint asks only "if" the sheet has one
    assert "clip session" not in pitch["intro"] and "If this sheet asks where" in pitch["intro"]
    bare = next(s for s in review_page.video_model(vdir, entry_of(vdir), books, Labels())["sections"] if s["key"] == "pitch")
    assert bare["intro"] == review_page.DEFAULT_HINTS["clip_session_intro"]          # the default without the labels
    run2, vdir2 = make_run(tmp_path / "b", books)
    blind_export(run2)                                                      # every Stage A item on the sheet
    pitch = next(s for s in review_page.video_model(vdir2, entry_of(vdir2), books, labels)["sections"] if s["key"] == "pitch")
    assert pitch["intro"] is None                                           # the page shows hints.pitch_intro
    assert "s.intro || PAGE.hints.pitch_intro" in _script()


def test_the_model_side_of_a_prepared_rating_and_of_the_window():
    from tests.panel_fixtures import synthetic_record
    d = synthetic_record()
    side = review_page.model_side({"path": "prep_pres:PRES-EVID", "primary": "5"}, d)
    assert side["answer"] == "5" and side["reason"]
    window = review_page.model_side({"path": "prepared_window", "primary": "0:05-1:00"}, d)
    assert "0:05-1:00" in window["reason"]
    ask = review_page.model_side({"path": "venture:ask", "primary": "200000;10"}, d)
    assert ask["reason"] == "Stated in the opening."


def test_the_template_numbers_its_cards_and_keeps_a_chosen_video_file():
    script = _script()
    assert "#${c.number}" in script                         # every card's header, the sheet's row
    assert "urls[" in script and "urls[vid]" in script      # a file-picked video is kept in page memory
    assert "document.getElementById('cbback').onclick = ()=> open(cur)" not in script
    assert "mark now" in script and "clk(v.currentTime)" in script


def _value_source(model, state, tail):
    script = _script()
    start, end = script.index("const cols = PAGE.columns;"), script.index("const AGAIN = ")
    return "\n".join([f"const PAGE = {review_page.embed_json(model)};", f"let state = {json.dumps(state)};",
                      _lines_of(script, "const st = ", "const answered = "), script[start:end], tail])


def test_value_of_serialises_every_kind_of_card():
    cards = {"ask": {"input": "ask", "values": []}, "clock": {"input": "clock", "values": []},
             "number": {"input": "number", "values": []}, "choice": {"input": "choice", "values": ["left", "right"]},
             "pills": {"input": None, "values": ["1", "2", "3"]}}
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    cases = [("ask", {**blank, "usd": "200000", "pct": "10", "value": "200000;10"}),
             ("ask", {**blank, "usd": " 150000 ", "pct": "7.5", "value": "x"}),
             ("ask", {**blank, "usd": "200000", "pct": "", "value": ""}),                  # half an ask is no answer
             ("ask", {**blank, "usd": "200000", "pct": "10", "value": "missing:not_observable"}),
             ("clock", {**blank, "value": " 1:51 "}), ("number", {**blank, "value": "4"}),
             ("number", {**blank, "value": "missing:too_short"}), ("choice", {**blank, "value": "left", "close": True}),
             ("pills", {**blank, "value": "2"}), ("pills", {**blank, "value": "2", "unclear": True}),
             ("clock", {**blank, "value": "1:51", "structural": True})]
    model = {"columns": list(SHEET_COLUMNS)}
    got = _node(_value_source(model, {}, f"const cases = {json.dumps([[cards[k], x] for k, x in cases])};\n"
                                         "console.log(JSON.stringify(cases.map(([c, x]) => valueOf(c, x))));"))
    assert got == ["200000;10", "150000;7.5", "", "missing:not_observable", "1:51", "4", "missing:too_short", "left",
                   "2", "", ""]


def test_the_sheet_the_template_writes_for_the_new_inputs_is_one_the_import_takes(tmp_path, books, capsys):
    """The ask, a count, a clock and a picked concept written by the page's own writer, then imported."""
    run, vdir = make_run(tmp_path / "a", books)
    full_export(run)
    model = review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    state = {"vid1": {"venture:ask": {**blank, "usd": "250000", "pct": "12.5", "value": "250000;12.5"},
                      "prep_pres:PRES-EVID": {**blank, "value": "3"},
                      "prep_pres:PRES-SUBS": {**blank, "value": "4"},
                      "prepared_window": {**blank, "structural": True, "note": "starts later"}}}
    text = _node(_value_source(model, state, "console.log(JSON.stringify(sheetFor(PAGE.videos[0])));"))
    sheet = vdir / "review" / "vid1__ana.csv"
    sheet.write_text(text, encoding="utf-8")
    rows = {r["path"]: r for r in read_sheet(sheet)}
    assert rows["venture:ask"]["human_value"] == "250000;12.5" and rows["prep_pres:PRES-EVID"]["human_value"] == "3"
    assert rows["prepared_window"]["resolution"] == "structural" and rows["prepared_window"]["human_value"] == ""
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", "--video-id", "vid1"]) == 0
    log = {l["path"]: l for l in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    assert log["venture:ask"]["new"] == "250000;12.5" and log["prep_pres:PRES-EVID"]["new"] == "3"
    # the clock of the first decision, from the full-video sheet
    run2, vdir2 = make_run(tmp_path / "b", books)
    blind_export(run2, "--qa-per-video", "10")
    model = review_page.page_model(run2, ["vid1"], "ana", books, load_labels(load_release()))
    concept = next(c for c in _cards(model) if c["kind"] == "PR-CONCEPT")
    state = {"vid1": {"pitch:first_decision": {**blank, "value": "1:51"},
                      concept["path"]: {**blank, "value": concept["values"][-1]}}}
    text = _node(_value_source(model, state, "console.log(JSON.stringify(sheetFor(PAGE.videos[0])));"))
    sheet = vdir2 / "review" / "vid1__ana.csv"
    sheet.write_text(text, encoding="utf-8")
    rows = {r["path"]: r for r in read_sheet(sheet)}
    assert rows["pitch:first_decision"]["human_value"] == "1:51" and rows[concept["path"]]["human_value"] == concept["values"][-1]
    assert rp.main(["import", "--run", str(run2), "--sheet", str(sheet), "--reviewer", "ana", "--video-id", "vid1"]) == 0


def test_a_close_choice_is_written_in_the_note_column():
    model = {"columns": list(SHEET_COLUMNS), "videos": [{"video_id": "v", "sections": [{"key": "pairs", "cards": [
        {"row": 1, "number": 1, "input": "choice", "path": "pair:b01/pitch_positivity", "field_class": "pair",
         "start": "", "end": "", "values": ["left", "right"]},
        {"row": 2, "number": 2, "input": "choice", "path": "pair:b02/pitch_positivity", "field_class": "pair",
         "start": "", "end": "", "values": ["left", "right"]}]}]}]}
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    state = {"v": {"pair:b01/pitch_positivity": {**blank, "value": "right", "close": True, "note": "hard"},
                   "pair:b02/pitch_positivity": {**blank, "close": True}}}       # no choice: nothing written
    text = _node(_value_source(model, state, "console.log(JSON.stringify(sheetFor(PAGE.videos[0])));"))
    lines = text.lstrip("﻿").splitlines()
    assert lines[1].endswith(",right,,hard [close]") and lines[2].endswith(",,,")


def test_a_long_value_list_gets_a_searchable_picker_with_each_values_meaning(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    cards = _cards(review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release())))
    concept = next(c for c in cards if c["kind"] == "PR-CONCEPT")
    sev = next(c for c in cards if c["kind"] == "PR-SEV")
    assert len(concept["values"]) > 6 and all(concept["defs"].get(v) for v in concept["values"])
    script = _script()
    assert 'class="picker"' in script
    source = "\n".join([_lines_of(script, "const esc = ", "const vname = ", "const chosenHtml = ", "const pickerHtml = "), _block(script, "function controlHtml("),
                        f"const cards = {json.dumps([concept, sev])};",
                        "const x = {value: cards[0].values[1], unclear: false, structural: false, note: '', summary_shown: false};",
                        "const y = {value: '', unclear: false, structural: false, note: '', summary_shown: false};",
                        "console.log(JSON.stringify([controlHtml(cards[0], x), controlHtml(cards[1], y)]));"])
    picker, pills = _node(source)
    assert 'class="picker"' in picker and picker.count('class="prow') == len(concept["values"])
    chosen = concept["values"][1]
    assert _esc(concept["defs"][chosen]) in picker.split('class="plist"')[0]           # the chosen value shown with its meaning
    for value in concept["values"]:
        if concept["defs"].get(value):
            assert _esc(concept["defs"][value]) in picker
    assert 'class="picker"' not in pills and pills.count('class="opt ') == 3       # three values: pills, as before


def test_the_number_clock_and_ask_inputs_render(tmp_path, books):
    script = _script()
    source = "\n".join([_lines_of(script, "const esc = ", "const vname = ", "const chosenHtml = ", "const pickerHtml = "), _block(script, "function controlHtml("),
                        "const blank = {value: '', unclear: false, structural: false, note: '', summary_shown: false};",
                        "const kinds = ['number', 'clock', 'ask'].map(input => controlHtml({input, values: [], defs: {}}, blank));",
                        "kinds.push(controlHtml({input: null, values: [], defs: {}}, blank));",
                        "kinds.push(controlHtml({input: 'choice', values: ['left', 'right'], defs: {}}, {...blank, close: true}));",
                        "console.log(JSON.stringify(kinds));"])
    number, clock, ask, none, choice = _node(source)
    assert '<input type="number" class="numin" min="0" step="1"' in number
    assert 'class="clockin"' in clock and "M:SS" in clock and "mark now" in clock
    assert ask.count('type="number"') == 2 and 'class="usd"' in ask and 'class="pct"' in ask
    assert none == ""                                                             # the window: the tick answers it
    assert choice.count('class="opt ') == 2 and 'class="close"' in choice and "checked" in choice


# ---- the coders' cards after the researcher's pilot (plan C, Task 7d) ------------------------------------------

# PREP-SUBS and PRES-SUBS share a wording; answer coherence (PREP-COH) since plan C Task 7e
WORDED = {"RS-EVID", "PR-SEV", "PREP-SUBS", "PRES-SUBS", "PRES-COH", "PREP-COH"}


def test_the_coders_wording_replaces_the_scale_text_for_its_codes_and_nowhere_else(tmp_path, books):
    import dataclasses
    from pipeline.stage_b.review_labels import scale_anchors
    labels = load_labels(load_release())
    assert set(labels.anchors) == WORDED
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")                   # every Stage A item and every question-answer code
    worded = _cards(review_page.video_model(vdir, entry_of(vdir), books, labels))
    plain = _cards(review_page.video_model(vdir, entry_of(vdir), books, dataclasses.replace(labels, anchors={})))
    assert {c["entry"] for c in worded} >= {"RS-EVID", "PR-SEV", "PREP-SUBS", "PR-CONCEPT", "RS-RESP", "PREP-COH"}
    coh_qa = next(c for c in worded if c["entry"] == "PREP-COH")
    assert coh_qa["q"].endswith("hang together?") and coh_qa["defs"]["4"] == "Clear throughout: each point follows."
    for c, bare in zip(worded, plain):
        assert c["path"] == bare["path"] and c["html_def"] == bare["html_def"]     # the codebook view keeps the model's
        if c["entry"] in WORDED:
            assert c["gloss"] == labels.anchors[c["entry"]]["gloss"] and c["gloss"]
            texts = {v: w["text"] for v, w in labels.anchors[c["entry"]]["values"].items() if w["text"]}
            assert all(c["defs"][v] == text for v, text in texts.items())
        else:                                                   # nowhere else: as without the coders' wording
            assert (c["defs"], c["gloss"], c["value_labels"]) == (bare["defs"], "", {}), c["path"]
    sev = next(c for c in worded if c["entry"] == "PR-SEV")
    assert sev["defs"]["2"].startswith("Asks the founder to back up something they said")
    assert "Verification-seeking" in sev["html_def"] and sev["defs"] != scale_anchors(books.panel.by_id("PR-SEV"))
    evid = next(c for c in worded if c["entry"] == "RS-EVID")
    assert evid["values"] == ["true", "false"] and evid["value_labels"] == {"true": "yes", "false": "no"}
    assert evid["gloss"].startswith("A checkable specific is a detail someone could look up or verify")
    subs = next(c for c in worded if c["entry"] == "PREP-SUBS")
    assert subs["defs"]["1"] == "Slogan only: nothing about how it works."
    # the clip session's cards: the presentation's substance and coherence
    row = lambda code: {"path": f"prep_pres:{code}", "field_class": "prep_pres", "context": "",   # noqa: E731
                        "clock_start": "", "clock_end": ""}
    scale = ("1", "2", "3", "4", "5")
    pres = review_page._card(row("PRES-SUBS"), 1, scale, synthetic(), books, labels)
    assert pres["defs"] == subs["defs"] and pres["gloss"] == subs["gloss"]
    assert pres["q"] == "How much does the presentation explain about how the business works?"
    coh = review_page._card(row("PRES-COH"), 2, scale, synthetic(), books, labels)
    assert coh["q"] == "Does the presentation’s story hang together?"
    assert coh["defs"]["4"].startswith("Clear throughout: problem → product → why it sells")
    assert coh["gloss"].endswith("Do not count nerves, accents or stumbles.")
    ctx = review_page._card(row("PRES-CTX"), 3, scale, synthetic(), books, labels)       # not worded: the codebook's
    assert ctx["defs"] == scale_anchors(books.panel.by_id("PRES-CTX")) and ctx["gloss"] == ""


def synthetic():
    from tests.panel_fixtures import synthetic_record
    return synthetic_record()


def test_the_first_decision_card_reads_the_coders_question_and_its_line(tmp_path, books):
    labels = load_labels(load_release())
    run, vdir = make_run(tmp_path, books)
    blind_export(run)
    cut = next(c for c in _cards(review_page.video_model(vdir, entry_of(vdir), books, labels)) if c["kind"] == "cut")
    assert cut["q"] == "When does the first shark say they are out or make an offer?"
    hint = labels.hints["cut_card"]
    assert hint.startswith("The first time any shark states a decision") and hint.endswith("at its first words.")
    assert "PAGE.hints.cut_card" in _script()                    # shown under the question, on the card's own line
    assert "shark" not in review_page.DEFAULT_HINTS["cut_card"]  # the default names no role
    # an old sheet with the outcome card keeps the "Outcomes" heading over both
    run2, vdir2 = make_run(tmp_path / "b", books)
    blind_export(run2, *OLD_SHEET)
    old = review_page.video_model(vdir2, entry_of(vdir2), books, labels)["sections"]
    assert next(s for s in old if s["key"] == "outcomes")["title"] == "Outcomes"
    assert review_page.video_model(vdir, entry_of(vdir), books, Labels())["sections"][-1]["title"] == \
        review_page.DEFAULT_SECTIONS["first_decision"]


def test_hovering_a_value_shows_its_meaning_at_once_and_leaving_restores_the_chosen_one():
    script = _script()
    assert "hoverDefs(el, c, x);" in script                      # every card wires it
    source = "\n".join([_block(script, "function hoverDefs("),
                        "const line = {textContent: 'one'};",
                        "const pills = ['1', '2'].map(v => ({dataset: {v}}));",
                        "const el = {querySelector: q => q === '.anchor' ? line : null, querySelectorAll: q => q === '.opt' ? pills : []};",
                        "const c = {defs: {'1': 'one', '2': 'two'}}; const x = {value: '1'};",
                        "hoverDefs(el, c, x); const out = [];",
                        "pills[1].onmouseenter(); out.push(line.textContent); pills[1].onmouseleave(); out.push(line.textContent);",
                        "x.value = ''; pills[0].onmouseenter(); out.push(line.textContent); pills[0].onmouseleave(); out.push(line.textContent);",
                        "hoverDefs({querySelector: () => null, querySelectorAll: () => { throw new Error('no line, no wiring'); }}, c, x);",
                        "console.log(JSON.stringify(out));"])
    assert _node(source) == ["two", "one", "one", ""]


def test_a_value_shows_its_coder_label_and_the_sheet_keeps_the_value():
    script = _script()
    card = {"input": None, "values": ["true", "false"], "defs": {}, "value_labels": {"true": "yes", "false": "no"}}
    source = "\n".join([_lines_of(script, "const esc = ", "const vname = ", "const chosenHtml = ", "const pickerHtml = "),
                        _block(script, "function controlHtml("),
                        f"const c = {json.dumps(card)};",
                        "const x = {value: 'true', unclear: false, structural: false, note: '', summary_shown: false};",
                        "console.log(JSON.stringify([controlHtml(c, x), chosenHtml(c, 'false'), vname({values: []}, 'not_an_exchange')]));"])
    pills, chosen, plain = _node(source)
    assert 'data-v="true"' in pills and '>yes</span>' in pills and '>no</span>' in pills and ">true<" not in pills
    assert "<b>no</b>" in chosen and plain == "not an exchange"
    got = _node(_value_source({"columns": list(SHEET_COLUMNS)}, {},
                              f"console.log(JSON.stringify(valueOf({json.dumps(card)}, "
                              "{value: 'true', unclear: false, structural: false, note: ''})));"))
    assert got == "true"


def test_choosing_other_needs_a_name_in_the_note_before_the_card_counts_as_done(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    blind_export(run, "--qa-per-video", "10")
    model = review_page.page_model(run, ["vid1"], "ana", books, load_labels(load_release()))
    concept = next(c for c in _cards(model) if c["kind"] == "PR-CONCEPT")
    sev = next(c for c in _cards(model) if c["kind"] == "PR-SEV")
    assert concept["name_values"] == ["other"] and sev["name_values"] == []
    script = _script()
    assert "note (required when you tick a box, or when you choose other: name the matter in a few words)" in script
    assert '<textarea class="note" placeholder="${notePlaceholder(c)}">' in script
    shown = _node("\n".join([_lines_of(script, "const PLAIN_NOTE = ", "const NAME_NOTE = ", "const notePlaceholder = "),
                              f"console.log(JSON.stringify([{json.dumps(concept)}, {json.dumps(sev)}].map(notePlaceholder)));"]))
    assert shown == ["note (required when you tick a box, or when you choose other: name the matter in a few words)",
                     "note (required when you tick a box)"]
    assert "if(complete(c, x)) d++" in script and "classList.toggle('done', complete(c, x))" in script
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    cases = [{**blank, "value": "other"}, {**blank, "value": "other", "note": "  "},
             {**blank, "value": "other", "note": "franchise fees"}, {**blank, "value": "other", "unclear": True},
             {**blank, "value": "sales"}, blank]
    source = "\n".join([_lines_of(script, "const answered = ", "const needsName = ", "const complete = "),
                        f"const c = {json.dumps(concept)}; const cases = {json.dumps(cases)};",
                        "console.log(JSON.stringify(cases.map(x => [needsName(c, x), complete(c, x)])));"])
    assert _node(source) == [[True, False], [True, False], [False, True], [False, True], [False, True], [False, False]]
    # the progress count: an unnamed other is open, a named one is done
    video = model["videos"][0]
    for note, done in (("", 0), ("franchise fees", 1)):
        state = {"vid1": {concept["path"]: {**blank, "value": "other", "note": note}}}
        source = "\n".join([f"const PAGE = {review_page.embed_json(model)};", f"let state = {json.dumps(state)};",
                            _lines_of(script, "const st = ", "const answered = ", "const needsName = ", "const complete = ",
                                      "const tickOnly = ", "function counts("),
                            "console.log(JSON.stringify(counts(PAGE.videos[0])));"])
        assert _node(source) == [done, sum(len(s["cards"]) for s in video["sections"])]


def test_a_pair_split_by_other_turns_names_its_own_question_and_answer_turn(tmp_path, books):
    """Turn 5 answers question 2 across turns 3 and 4: each card names the turn it is about, so a coder playing from
    the question does not code the answer in between."""
    def split(d):
        for t in d["turns"]:
            if t["turn_id"] == 5:
                t["responds_to"] = [2]
    run, vdir = make_run(tmp_path, books, mutate_primary=split, mutate=split)
    blind_export(run, "--qa-per-video", "10")
    cards = _cards(review_page.video_model(vdir, entry_of(vdir), books, load_labels(load_release())))
    by_path = {c["path"]: c for c in cards}
    assert {"turn:2/code:PR-SEV", "turn:5/code:RS-EVID", "turn:5/code:PREP-SUBS"} <= set(by_path)
    assert [t["turn"] for t in by_path["turn:5/code:RS-EVID"]["turns"]] == ["2", "5"]      # the pair, not turns 3-4
    assert by_path["turn:2/code:PR-SEV"]["q"] == "How hard does the question in turn 2 push on what the founder has said?"
    assert by_path["turn:5/code:RS-EVID"]["q"] == "Does the answer in turn 5 give at least one checkable specific?"
    assert by_path["turn:5/code:PREP-SUBS"]["q"] == "How much does the answer in turn 5 explain about how the business works?"


def test_a_walkthrough_card_shows_a_values_meaning_on_hover_too():
    script = _script()
    model_card = _block(script, "function modelCard(")
    assert "hoverDefs(el, c, {value: ''});" in model_card and '<div class="anchor"></div>' in model_card


def test_the_page_finds_a_video_whose_punctuation_was_changed_on_the_way():
    """A copy that went through a browser download, a zip or a cloud drive has ' and | replaced by _; the page
    still finds it (exact name first, then a name equal once punctuation and spacing are ignored)."""
    script = _script()
    source = "\n".join([
        _lines_of(script, "async function fileIn(", "const nameKey = ", "async function looseFileIn("),
        "const files = {\"Everybody Loves Mama O_s Premium Kimchi _ Shark Tank US _ Shark Tank Global.mp4\": 'kimchi',"
        " \"other.mp4\": 'other'};",
        "const dir = { getFileHandle: async n => { if(!(n in files)) throw new Error('no'); return {getFile: async () => files[n]}; },",
        "  entries: async function*(){ for(const n of Object.keys(files)) yield [n, {kind: 'file', getFile: async () => files[n]}]; } };",
        "(async () => { const out = [];",
        "  out.push(await fileIn(dir, [\"Everybody Loves Mama O's Premium Kimchi | Shark Tank US | Shark Tank Global.mp4\"]));",
        "  out.push(await fileIn(dir, ['other.mp4']));",
        "  try { await fileIn(dir, ['missing.mp4']); out.push('found'); } catch(e){ out.push('missing'); }",
        "  console.log(JSON.stringify(out)); })();"])
    assert _node(source) == ["kimchi", "other", "missing"]


def test_the_clock_reads_the_player_that_sent_the_event_so_switching_tasks_cannot_crash_it():
    """Switching tasks removes the player and sets `v` to null; the removed player's last timeupdate must not read
    `v` (it crashed the page with "Cannot read properties of null (reading 'currentTime')")."""
    script = _script()
    line = next(l for l in script.splitlines() if "addEventListener('timeupdate'" in l and "'clk'" in l)
    assert "v.currentTime" not in line.replace("player.currentTime", "")
    got = _node("\n".join([
        "const clk = s => String(Math.round(s));",
        "const handlers = []; let shown = null;",
        "const document = {getElementById: () => ({set textContent(x){ shown = x; }})};",
        "let v = {currentTime: 12, addEventListener: (e, f) => handlers.push(f)};",
        line,
        "handlers[0](); const first = shown;",
        "v = null; handlers[0]();",
        "console.log(JSON.stringify([first, shown]));"]))
    assert got == ["12", "12"]


# ---- "Download all sheets": one zip (Safari keeps only the first of several downloads one click starts) --------------

ZIP_LINES = ("const CRC_TABLE = ", "const crc32 = ", "const zipRecord = ")


def _zip_bytes(files, tmp_path):
    """The template's zipOf over [[name, text]] run in node, its bytes written to a file (through base64 on stdout)."""
    import base64
    script = _script()
    source = "\n".join([_lines_of(script, *ZIP_LINES), _block(script, "function zipOf("),
                        f"const files = {json.dumps(files)};",
                        "console.log(JSON.stringify(Buffer.from(zipOf(files, new Date(2026, 9, 23, 14, 5, 30))).toString('base64')));"])
    out = tmp_path / "sheets.zip"
    out.write_bytes(base64.b64decode(_node(source)))
    return out


def test_the_templates_zip_holds_every_sheet_byte_for_byte_with_its_crc_and_utf8_name(tmp_path):
    import zipfile
    files = [["clips__AV.csv", "﻿row,path,human_value,note\n1,stage_a:x,3,\"l'équipe a dit « oui » | non\"\n"],
             ["vid_ü__AV.csv", "﻿row,values\n2,low | medium | high,crème brûlée — 日本\n"]]
    with zipfile.ZipFile(_zip_bytes(files, tmp_path)) as zf:
        assert zf.testzip() is None                                 # every member's CRC-32 checks
        infos = zf.infolist()
        assert [i.filename for i in infos] == [name for name, _ in files]
        assert all(i.compress_type == zipfile.ZIP_STORED and i.flag_bits & 0x800 for i in infos)
        assert all(i.date_time == (2026, 10, 23, 14, 5, 30) for i in infos)
        for name, text in files:
            assert zf.read(name) == text.encode("utf-8")            # the bytes the single download wrote, BOM and all
    # one sheet is zipped all the same
    with zipfile.ZipFile(_zip_bytes(files[:1], tmp_path)) as zf:
        assert zf.testzip() is None and zf.namelist() == ["clips__AV.csv"]


def _download_all(mode, sheets):
    """The page's "Download all sheets" handler run in node with a stub button and a counting download."""
    script = _script()
    handler = next(l for l in script.splitlines() if l.startswith("document.getElementById('dl').onclick = "))
    return _node("\n".join([
        f"const PAGE = {json.dumps({'coder': 'AV', 'mode': mode})};", "const calls = [], alerts = [];",
        "const el = {}; const document = { getElementById: () => el }; const alert = m => alerts.push(m);",
        f"const sheets = () => {json.dumps(sheets)};", "const zipOf = files => files.map(([name]) => name);",
        "function download(data, name, type){ calls.push([data, name, type]); }",
        _lines_of(script, "const zipName = "), handler, "el.onclick();",
        "console.log(JSON.stringify([calls, alerts.length]));"]))


def test_download_all_sheets_downloads_one_zip_named_for_the_coder():
    sheets = [["clips__AV.csv", "a"], ["v1__AV.csv", "b"]]
    assert _download_all("tasks", sheets) == [[[["clips__AV.csv", "v1__AV.csv"], "AV__sheets.zip", "application/zip"]], 0]
    assert _download_all("validation", sheets[1:]) == [[[["v1__AV.csv"], "AV__sheets.zip", "application/zip"]], 0]
    assert _download_all("training_tasks", sheets)[0][0][1] == "AV__training_sheets.zip"
    assert _download_all("training", sheets)[0][0][1] == "AV__training_sheets.zip"
    assert _download_all("tasks", []) == [[], 1]                    # nothing answered: said so, nothing downloaded
    script = _script()
    assert "for(const [name, text] of sheets()) download(" not in script
