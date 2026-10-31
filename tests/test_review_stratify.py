"""Plan C Task 7f (the researcher's decisions of 2026-10-21): a validation sample stratified by the pitch's outcome,
question-answer pairs drawn only before the pitch's first decision, and a "not sure" tick beside a coder's best guess."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from analysis.panel_dataset import build_data, pitch_cut  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.reliability.agreement import krippendorff_alpha, stratified_alpha  # noqa: E402
from pipeline.schemas.stage_b_panel import PanelRecord  # noqa: E402
from pipeline.stage_b import review_page  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_apply import UNSURE, pooled_summary, unnamed_others  # noqa: E402
from pipeline.stage_b.review_items import SHEET_COLUMNS, cut_position  # noqa: E402
from pipeline.stage_b.review_labels import load_labels  # noqa: E402
from pipeline.stage_b.review_report import (  # noqa: E402
    load_reporting, note_of, paper_rows, paper_table_markdown, stratum_rows)
from pipeline.stage_b.review_select import select_exchange_sheet  # noqa: E402
from tests.panel_fixtures import synthetic_record  # noqa: E402
from tests.test_review_cli import make_run, read_sheet  # noqa: E402
from tests.test_review_page import _block, _lines_of, _node, _script, _value_source  # noqa: E402

CLEAN = {"errors": [], "flags": []}


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def no_offer(d):
    """Every investor exits and none offers: the pitch has no offer (any_offer "no")."""
    d["decisions"] = [
        {"decision_id": 1, "actor": "S3", "kind": "exit", "time_s": 111.0, "turn_id": 10, "offer_id": None,
         "terms": "", "note": "Out."},
        {"decision_id": 2, "actor": "S1", "kind": "exit", "time_s": 159.0, "turn_id": 17, "offer_id": None,
         "terms": "", "note": "Out."},
        {"decision_id": 3, "actor": "S2", "kind": "exit", "time_s": 164.0, "turn_id": 18, "offer_id": None,
         "terms": "", "note": "Out."}]


def no_decision(d):
    d["decisions"] = []


def stratified_run(tmp_path, books, offers=6, without=6, unknown=1):
    for i in range(offers):
        run, _ = make_run(tmp_path, books, video_id=f"yes{i}")
    for i in range(without):
        run, _ = make_run(tmp_path, books, video_id=f"no{i}", mutate_primary=no_offer)
    for i in range(unknown):
        run, _ = make_run(tmp_path, books, video_id=f"unk{i}", mutate_primary=no_decision)
    make_run(tmp_path, books, video_id="solo", replicate=False)              # one pass only: never eligible
    return run


def export(run, *extra):
    return rp.main(["export", "--run", str(run), "--blind", *extra])


# ---- 1. the stratified draw --------------------------------------------------------------------------------------

def test_a_stratified_draw_takes_half_of_the_videos_from_each_outcome_and_records_the_strata(tmp_path, books, capsys):
    run = stratified_run(tmp_path, books)
    assert export(run, "--sample-videos", "10", "--stratify", "any_offer", "--seed", "3") == 0
    out = capsys.readouterr().out
    record = json.loads((run / "review" / "video_sample.json").read_text())
    draw = record["draws"][0]
    assert draw["stratify"] == "any_offer"
    yes, no = draw["strata"]["yes"], draw["strata"]["no"]
    assert yes["eligible"] == [f"yes{i}" for i in range(6)] and no["eligible"] == [f"no{i}" for i in range(6)]
    assert (yes["n_eligible"], no["n_eligible"]) == (6, 6)
    assert len(yes["chosen"]) == len(no["chosen"]) == 5
    assert set(yes["chosen"]) < set(yes["eligible"]) and set(no["chosen"]) < set(no["eligible"])
    assert draw["unknown"] == ["unk0"]                     # neither stratum: the record says neither yes nor no
    assert draw["chosen"] == record["chosen"] == sorted(yes["chosen"] + no["chosen"])
    assert (draw["seed"], draw["n"]) == (3, 10)
    exported = sorted(p.parent.parent.name for p in run.glob("*/review/export_manifest.json"))
    assert exported == record["chosen"]
    assert "5 with an offer, 5 without" in out
    # the same call again is the same draw; the unstratified draw of the same seed and N is another draw
    assert export(run, "--sample-videos", "10", "--stratify", "any_offer", "--seed", "3", "--force") == 0
    assert json.loads((run / "review" / "video_sample.json").read_text()) == record
    # the spreadsheet beside it carries no outcome (final review C1): the strata stay in video_sample.json
    rows = read_sheet(run / "review" / "video_sample.csv")
    assert sorted(r["video_id"] for r in rows) == record["chosen"]
    assert set(rows[0]) == set(rp.SAMPLE_COLUMNS)
    assert not {"stratum", "any_offer", "outcome"} & set(rows[0])


def test_an_odd_number_or_a_thin_stratum_is_refused_and_nothing_is_written(tmp_path, books, capsys):
    run = stratified_run(tmp_path, books, offers=5, without=3, unknown=0)
    assert export(run, "--sample-videos", "7", "--stratify", "any_offer") == 1
    assert "an even number" in capsys.readouterr().err
    assert export(run, "--sample-videos", "8", "--stratify", "any_offer") == 1
    err = capsys.readouterr().err
    assert "3 eligible pitches without an offer" in err and "4" in err
    assert not (run / "review" / "video_sample.json").exists()
    assert export(run, "--sample-videos", "6", "--stratify", "any_offer") == 0
    assert export(run, "--stratify", "any_offer", "--video-ids", "yes0", "--force") == 1
    assert "--stratify belongs to --sample-videos" in capsys.readouterr().err


def test_excluded_videos_stay_out_of_both_strata(tmp_path, books):
    run = stratified_run(tmp_path, books, offers=4, without=4, unknown=0)
    assert export(run, "--sample-videos", "4", "--stratify", "any_offer", "--exclude-video-ids", "yes0", "no1") == 0
    draw = json.loads((run / "review" / "video_sample.json").read_text())["draws"][0]
    assert draw["excluded"] == ["no1", "yes0"]
    assert "yes0" not in draw["strata"]["yes"]["eligible"] and draw["strata"]["yes"]["n_eligible"] == 3
    assert "no1" not in draw["strata"]["no"]["eligible"] and draw["strata"]["no"]["n_eligible"] == 3
    assert not {"yes0", "no1"} & set(draw["chosen"])


def test_a_different_stratification_is_named_when_the_sample_is_refused(tmp_path, books, capsys):
    run = stratified_run(tmp_path, books, offers=3, without=3, unknown=0)
    assert export(run, "--sample-videos", "2", "--seed", "1") == 0
    capsys.readouterr()
    assert export(run, "--sample-videos", "2", "--seed", "1", "--stratify", "any_offer") == 1
    err = capsys.readouterr().err
    assert "(not stratified)" in err and "--stratify" in err
    export(run, "--sample-videos", "2", "--seed", "1")          # the recorded draw again (its sheets already exist)
    assert "already records seed 1, n 2, these exclusions and no stratification" in capsys.readouterr().out


def test_the_unstratified_draw_is_unchanged(tmp_path, books):
    run = stratified_run(tmp_path, books, offers=3, without=3, unknown=1)
    assert export(run, "--sample-videos", "4", "--seed", "5") == 0
    record = json.loads((run / "review" / "video_sample.json").read_text())
    draw = record["draws"][0]
    assert set(draw) == {"seed", "n", "excluded", "eligible", "chosen"}            # no strata recorded
    eligible = sorted(["no0", "no1", "no2", "unk0", "yes0", "yes1", "yes2"])
    assert draw["eligible"] == eligible
    assert draw["chosen"] == sorted(random.Random("5:videos").sample(eligible, 4))
    assert set(read_sheet(run / "review" / "video_sample.csv")[0]) == set(rp.SAMPLE_COLUMNS)
    assert "stratum" not in rp.SAMPLE_COLUMNS
    assert rp.sample_strata(run) is None


def test_sample_strata_gives_each_videos_stratum_and_the_runs_sizes(tmp_path, books):
    run = stratified_run(tmp_path, books, offers=4, without=2, unknown=0)
    assert export(run, "--sample-videos", "4", "--stratify", "any_offer", "--seed", "1") == 0
    strata = rp.sample_strata(run)
    assert strata["stratify"] == "any_offer" and strata["run"] == {"yes": 4, "no": 2}
    assert sorted(strata["of"].values()) == ["no", "no", "yes", "yes"]


# ---- 2. question-answer pairs before the first decision ----------------------------------------------------------

def _sheet(books, mutate=None, qa=10):
    d = synthetic_record()
    if mutate:
        mutate(d)
    p = PanelRecord.model_validate(d)
    return select_exchange_sheet(p, PanelRecord.model_validate(d), None, CLEAN, books, seed=0, key="v", qa=qa)


def test_only_pairs_before_the_first_decision_are_drawn(books):
    # the fixture's first decision is turn 10 (an exit, position 9): pairs 2-3 and 4-5 come before it, 11-12 and
    # 14-15 after it
    sheet = _sheet(books)
    sample = sheet["sample"]
    assert sample["drawn"]["qa"] == [[2, 3], [4, 5]]
    assert sample["qa_before_cut"] is True
    assert sample["available"]["qa"] == 2
    assert sample["population"]["turn_code"] == 2 * 6
    turns = {int(r["path"].split("/")[0].split(":")[1]) for r in sheet["rows"] if r["path"].startswith("turn:")}
    assert turns == {2, 3, 4, 5}


def test_a_pitch_without_a_decision_gives_pairs_from_the_whole_video(books):
    sheet = _sheet(books, no_decision)
    assert sheet["sample"]["drawn"]["qa"] == [[2, 3], [4, 5], [11, 12], [14, 15]]
    assert sheet["sample"]["population"]["turn_code"] == 4 * 6 and sheet["sample"]["qa_before_cut"] is True


def test_a_question_before_the_cut_whose_answer_comes_after_it_is_not_drawn(books):
    def late_exit(d):             # the first decision moves to turn 12's position: the answer of 11 is at the cut
        no_offer(d)
        d["decisions"][0].update(turn_id=12, time_s=120.0)
    assert _sheet(books, late_exit)["sample"]["drawn"]["qa"] == [[2, 3], [4, 5]]


def test_a_question_no_investor_asked_is_never_drawn(books):
    # final review M1: analysis.panel_dataset.answer_followups reads an investor's questions only
    def founder_asks(d):
        d["turns"][3]["speaker"] = "F1"            # turn 4's question, asked by the founder
    assert _sheet(books, founder_asks)["sample"]["drawn"]["qa"] == [[2, 3]]


def _variants():
    def unplaced(d):              # a decision no turn speaks: placed by its time (the last turn started by then)
        d["decisions"][0].update(turn_id=None, time_s=118.0)

    def founder_only(d):          # a founder's decision is no investor's: it does not cut
        d["decisions"] = [x for x in d["decisions"] if x["actor"] == "F1"]
    return {"fixture": None, "no_decision": no_decision, "no_offer": no_offer, "unplaced": unplaced,
            "founder_only": founder_only}


@pytest.mark.parametrize("name", sorted(_variants()))
def test_the_review_cut_is_the_analysis_cut(books, name):
    d = synthetic_record()
    if _variants()[name]:
        _variants()[name](d)
    data = build_data([{"video_id": "v", "record": d, "source": "primary"}], books=books)
    assert cut_position(d) == int(pitch_cut(data).cut_pos.iloc[0])


# ---- 3. the "not sure" tick ----------------------------------------------------------------------------------------

def test_the_page_offers_a_not_sure_tick_on_value_cards_only():
    script = _script()
    source = "\n".join([_lines_of(script, "const esc = ", "const valueCard = ", "const unsureHtml = "),
                        "const blank = {value: '', unclear: false, structural: false, note: '', summary_shown: false};",
                        "const cards = [{input: null, values: ['1', '2'], structural: 'wrong'},"
                        " {input: 'clock', values: [], structural: 'wrong'}, {input: 'number', values: [], structural: 'w'},"
                        " {input: 'ask', values: [], structural: 'w'}, {input: 'choice', values: ['left', 'right'], structural: 'w'},"
                        " {input: null, values: [], structural: 'the window is wrong'}];",
                        "console.log(JSON.stringify([...cards.map(c => unsureHtml(c, blank)), unsureHtml(cards[0], {...blank, unsure: true})]));"])
    got = _node(source)
    ticks = [('class="unsure"' in html) for html in got]
    assert ticks == [True, True, True, True, False, False, True]           # a pair row has its "close"; the window none
    assert "not sure" in got[0] and 'class="unsure" checked' in got[-1]
    assert "el.querySelector('.unsure')" in script                          # the tick is wired to the card's state
    assert "${control}${unsureHtml(c, x)}" in _block(script, "function card(")    # beside the answer, on every card
    # the compare rows of a comparison task have their close tick and no not-sure tick
    assert "unsure" not in _block(script, "function compareCard(") and "unsure" not in _lines_of(script, "const compareHtml = ")


def test_the_not_sure_tick_is_written_into_the_note_before_the_coders_own_note():
    assert UNSURE == "[unsure]"
    model = {"columns": list(SHEET_COLUMNS), "videos": [{"video_id": "v", "sections": [{"key": "qa", "cards": [
        {"row": n, "number": n, "input": None, "path": f"turn:{n}/code:PR-SEV", "field_class": "turn_code",
         "start": "", "end": "", "values": ["1", "2", "3"]} for n in (1, 2, 3, 4)] + [
        {"row": 5, "number": 5, "input": "choice", "path": "pair:b01/pitch_positivity", "field_class": "pair",
         "start": "", "end": "", "values": ["left", "right"]}]}]}]}
    blank = {"value": "", "unclear": False, "structural": False, "note": "", "summary_shown": False}
    state = {"v": {"turn:1/code:PR-SEV": {**blank, "value": "2", "unsure": True},
                   "turn:2/code:PR-SEV": {**blank, "value": "3", "unsure": True, "note": "half heard",
                                          "summary_shown": True},
                   "turn:3/code:PR-SEV": {**blank, "unsure": True},                          # no answer: no row
                   "turn:4/code:PR-SEV": {**blank, "unclear": True, "unsure": True, "note": "inaudible"},
                   "pair:b01/pitch_positivity": {**blank, "value": "left", "unsure": True}}}
    text = _node(_value_source(model, state, "console.log(JSON.stringify(sheetFor(PAGE.videos[0])));"))
    lines = text.lstrip("﻿").splitlines()
    assert lines[1].endswith(",2,,[unsure]")
    assert lines[2].endswith(",3,,[unsure] half heard [summary shown]")
    assert lines[3].endswith(",,,")
    assert lines[4].endswith(",,unclear,inaudible")                       # can't tell: the row is missing, not unsure
    assert lines[5].endswith(",left,,")                                   # a pair row never carries it


def test_an_unsure_row_is_imported_with_its_marker_and_counts_as_a_review(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run, "--qa-per-video", "1") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    code = next(r for r in rows if r["path"].endswith("/code:PR-SEV"))
    for r in rows:
        r.update({"human_value": "", "resolution": "", "note": ""})
    code.update({"human_value": "2", "note": "[unsure] hard to hear"})
    from tests.test_review_cli import write_sheet
    write_sheet(sheet, rows)
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "1 rows reviewed" in capsys.readouterr().out
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert [(l["path"], l["note"]) for l in log] == [(code["path"], "[unsure] hard to hear")]
    # an `other` answered unsure still needs the matter named
    assert unnamed_others([{"path": "turn:2/code:PR-CONCEPT", "human_value": "other", "note": "[unsure]"}])


def _line(path, primary, new, note="", cls="turn_code"):
    return {"path": path, "field_class": cls, "primary": primary, "replicate": primary, "new": new,
            "resolution": "confirm_primary" if new == primary else "correct", "selected_as": "sampled", "note": note,
            "level": "nominal"}


def test_agreement_is_reported_with_and_without_the_unsure_rows(books):
    videos = {"a": [_line("turn:3/code:RS-RESP", "addresses", "addresses"),
                    _line("turn:5/code:RS-RESP", "partial", "partial"),
                    _line("turn:12/code:RS-RESP", "addresses", "partial", "[unsure] mumbled"),
                    _line("turn:15/code:RS-RESP", "partial", "addresses", "[unsure]")]}
    reporting = load_reporting(books)
    out = pooled_summary(videos, n_boot=99, reporting=reporting)
    assert out["sampled"]["turn_code"]["human_vs_primary"] == {"agree": 0.5, "n": 4}
    without = out["without_unsure"]
    assert without["n_unsure"] == 2
    assert without["sampled"]["turn_code"]["human_vs_primary"] == {"agree": 1.0, "n": 2}
    assert without["by_variable"]["RS-RESP"]["n"] == 2
    row = next(r for r in paper_rows(out, reporting) if r["field"] == "RS-RESP")
    assert (row["without_unsure"]["n"], row["without_unsure"]["exact"]) == (2, 1.0)
    text = paper_table_markdown(out, reporting, note_of(out))
    detail = text.split("## Coder against model")[1].splitlines()
    assert "Without unsure: point (n)" in detail[2]                        # the detail table, after the summary
    assert "| RS-RESP |" in text and "| 1.00 (2) |" in text.split("## Coder against coder")[0]
    assert "[unsure]" in text and "2 of 4" in text                          # the note says how many were unsure
    # nothing unsure: the same figures, and the table says so
    plain = pooled_summary({"a": videos["a"][:2]}, n_boot=99)
    assert plain["without_unsure"]["n_unsure"] == 0
    assert plain["without_unsure"]["sampled"] == plain["sampled"]


# ---- 4. figures per stratum and reweighted to the run's mix --------------------------------------------------------

def test_the_reweighted_agreement_is_the_hand_computed_one():
    # stratum yes (video a): (x, x) and (y, y) agree; stratum no (video b): (x, y) and (y, x) do not. The run has 3
    # pitches with an offer and 1 without, the sample 1 and 1: weights (3/4)/(1/2) = 1.5 and (1/4)/(1/2) = 0.5.
    # Share: (1.5 + 1.5) / (1.5 * 2 + 0.5 * 2) = 0.75 (unweighted 0.5).
    # Nominal alpha on weighted coincidences: o = [[3, 1], [1, 3]], n = 8, D_o = 2, D_e = 4 * 4 * 2 / 7, so
    # alpha = 1 - 2 * 7 / 32 = 0.5625 (unweighted 1 - 4 * 7 / 32 = 0.125).
    videos = {"a": [_line("turn:3/code:RS-RESP", "x", "x"), _line("turn:5/code:RS-RESP", "y", "y")],
              "b": [_line("turn:3/code:RS-RESP", "x", "y"), _line("turn:5/code:RS-RESP", "y", "x")]}
    strata = {"stratify": "any_offer", "of": {"a": "yes", "b": "no"}, "run": {"yes": 3, "no": 1}}
    out = pooled_summary(videos, n_boot=99, strata=strata)
    s = out["strata"]
    assert s["stratify"] == "any_offer"
    assert s["strata"]["yes"] == {"run": 3, "planned": 1, "sample": 1, "weight": 1.5, "videos": ["a"]}
    assert s["strata"]["no"] == {"run": 1, "planned": 1, "sample": 1, "weight": 0.5, "videos": ["b"]}
    overall = s["reweighted"]["sampled"]["turn_code"]
    assert overall["agree"] == pytest.approx(0.75) and overall["agree_unweighted"] == pytest.approx(0.5)
    assert overall["alpha"] == pytest.approx(0.5625) and overall["alpha_unweighted"] == pytest.approx(0.125)
    assert overall["n"] == 4 and overall["videos"] == 2
    assert s["reweighted"]["by_variable"]["RS-RESP"]["alpha"] == pytest.approx(0.5625)
    assert s["by_stratum"]["yes"]["sampled"]["turn_code"]["human_vs_primary"] == {"agree": 1.0, "n": 2}
    assert s["by_stratum"]["no"]["sampled"]["turn_code"]["human_vs_primary"] == {"agree": 0.0, "n": 2}
    assert out["sampled"]["turn_code"]["human_vs_primary"] == {"agree": 0.5, "n": 4}        # the main figure stays
    # the weighted alpha of agreement.py on its own
    units = {"": [["x", "x"], ["y", "y"], ["x", "y"], ["y", "x"]]}
    assert stratified_alpha(units, "nominal", {"": [1.5, 1.5, 0.5, 0.5]}) == pytest.approx(0.5625)
    assert stratified_alpha(units, "nominal", {"": [1, 1, 1, 1]}) == krippendorff_alpha(units[""], "nominal")
    # the paper table: a stratum table of the reported codes (the same figures on RS-RESP's own values), the main
    # figures unweighted
    reporting = load_reporting(load_panel_books(load_release(), sync=False))
    word = {"x": "addresses", "y": "partial"}
    real = {v: [{**l, "primary": word[l["primary"]], "replicate": word[l["replicate"]], "new": word[l["new"]]}
                for l in lines] for v, lines in videos.items()}
    out = pooled_summary(real, n_boot=99, strata=strata, reporting=reporting)
    row = next(r for r in stratum_rows(out, reporting) if r["field"] == "RS-RESP")
    assert row["reweighted_alpha"] == pytest.approx(0.5625) and row["unweighted_alpha"] == pytest.approx(0.125)
    assert row["reweighted_share"] == pytest.approx(0.75) and row["unweighted_share"] == pytest.approx(0.5)
    assert row["by_stratum"]["yes"]["share"] == 1.0 and row["by_stratum"]["no"]["share"] == 0.0
    text = paper_table_markdown(out, reporting)
    assert "## By outcome stratum" in text and "reweighted" in text
    assert "The main table's figures are unweighted." in text and "Met " not in text
    unstratified = paper_table_markdown(pooled_summary(real, n_boot=99, reporting=reporting), reporting)
    assert "By outcome stratum" not in unstratified and "Reweighted" not in unstratified


def test_summary_reports_the_strata_of_a_stratified_sample(tmp_path, books, capsys):
    run = stratified_run(tmp_path, books, offers=2, without=2, unknown=0)
    assert export(run, "--sample-videos", "2", "--stratify", "any_offer", "--seed", "0", "--qa-per-video", "1") == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    from tests.test_review_cli import fill
    for video_id in chosen:
        sheet = run / video_id / "review" / f"{video_id}__review.csv"
        path = next(r["path"] for r in read_sheet(sheet) if r["path"].endswith("/code:RS-RESP"))
        fill(sheet, {path: {"human_value": "addresses", "note": "[unsure]" if video_id.startswith("no") else ""}})
        assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    table = tmp_path / "table.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(table)]) == 0
    printed = capsys.readouterr().out
    assert "## By outcome stratum" in printed and "1 with an offer, 1 without" in printed
    saved = json.loads(table.with_suffix(".json").read_text())
    assert saved["strata"]["stratify"] == "any_offer" and saved["stratum_rows"]
    assert "## By outcome stratum" in table.read_text()


def test_the_weights_count_the_videos_imported_not_the_planned_sample():
    # the sample drew 2 + 2 videos, but only one without an offer is imported: the weights and the statement count the
    # videos pooled (3 with, 1 without of a run of 3 and 1), not the plan (2 and 2)
    videos = {"a": [_line("turn:3/code:RS-RESP", "x", "x"), _line("turn:5/code:RS-RESP", "y", "y")],
              "c": [_line("turn:3/code:RS-RESP", "x", "x"), _line("turn:5/code:RS-RESP", "y", "y")],
              "b": [_line("turn:3/code:RS-RESP", "x", "y"), _line("turn:5/code:RS-RESP", "y", "x")]}
    strata = {"stratify": "any_offer", "of": {"a": "yes", "c": "yes", "b": "no", "d": "no"}, "run": {"yes": 3, "no": 1}}
    s = pooled_summary(videos, n_boot=99, strata=strata)["strata"]
    # present: 2 with, 1 without, so weights (3/4)/(2/3) = 1.125 and (1/4)/(1/3) = 0.75
    assert s["strata"]["yes"] == {"run": 3, "planned": 2, "sample": 2, "weight": pytest.approx(1.125),
                                  "videos": ["a", "c"]}
    assert s["strata"]["no"] == {"run": 1, "planned": 2, "sample": 1, "weight": pytest.approx(0.75), "videos": ["b"]}
    # share: 4 agreeing rows of weight 1.125 against 2 disagreeing of 0.75: 4.5 / 6 = 0.75 (the plan's weights,
    # 1.5 and 0.5, would give 6 / 7)
    assert s["reweighted"]["sampled"]["turn_code"]["agree"] == pytest.approx(0.75)
    from pipeline.stage_b.review_report import strata_statement
    text = strata_statement(s)
    assert "2 with an offer, 1 without imported (of 2 and 2 drawn)" in text and "weights 1.12 and 0.75" in text
    # the per-stratum pools run no bootstrap: points only
    assert s["by_stratum"]["yes"]["sampled"]["turn_code"]["alpha_ci95"] is None
    # a pooled set with no video of the stratified sample (the training videos) carries no strata
    assert "strata" not in pooled_summary({"z": videos["a"]}, n_boot=99, strata=strata)
