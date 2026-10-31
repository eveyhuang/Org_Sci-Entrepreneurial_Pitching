"""analysis.panel_explorer: readable views of a panel run. Synthetic runs only; nothing empirical."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.panel_explorer as px
from pipeline.errors import ConfigError
from pipeline.instrument import load_release
from pipeline.schemas.stage_b_panel import PanelRecord
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.replicate_agreement import align
from tests.panel_analysis_fixtures import write_run
from tests.panel_fixtures import STAGE_A_IDS, synthetic_record


def _snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


@pytest.fixture(scope="module")
def run(tmp_path_factory, books):
    """Three synthetic videos. v000 is the story of tests/panel_fixtures with a replicate that differs in
    known places, a real agreement report, a bundle, a prompt, a call log and validation reports."""
    root = write_run(tmp_path_factory.mktemp("explorer"), 3, seed=4)
    vdir = root / "v000"
    primary = synthetic_record()
    replicate = copy.deepcopy(primary)
    replicate["turns"][1]["codes"][0]["value"] = "competition"          # the first question's concept
    replicate["exchanges"][2]["topic"] = "market_opportunity"           # the repeat-purchase exchange's topic
    replicate["stage_a"][0]["value"] = primary["stage_a"][0]["value"] + (1 if primary["stage_a"][0]["value"] < 5 else -1)
    replicate["stage_a"][1]["value"] = primary["stage_a"][1]["value"] + (2 if primary["stage_a"][1]["value"] < 4 else -2)
    for name, record in (("primary", primary), ("replicate", replicate)):
        _dump(vdir / "results" / f"{name}.json", {"pass": name, "record": record})
    report = align(PanelRecord.model_validate(primary), PanelRecord.model_validate(replicate), books)
    _dump(vdir / "agreement.json", {"video_id": "v000", **report})
    _dump(vdir / "results" / "primary_validation.json",
          {"errors": ["[exchange.topic] exchange 2: topic is 'sales' but the exchange is not substantive"],
           "flags": ["[flag.unanswered_question] turn 4: no answer responds to it", "a message without a rule"]})
    release = load_release()
    bundle = vdir / "bundle"
    bundle.mkdir()
    for path in [release.manifest_path, *(b.yaml for b in release.books.values()),
                 *(b.markdown for b in release.books.values())]:
        shutil.copy(path, bundle / path.name)
    prompt = "PROMPT START\n" + "x" * 5000 + "\nPROMPT END"
    _dump(vdir / "prompts" / "panel_full.txt", None)
    (vdir / "prompts" / "panel_full.txt").write_text(prompt)
    experiment = json.loads((vdir / "experiment.json").read_text())
    experiment["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    _dump(vdir / "experiment.json", experiment)
    (vdir / "calls.jsonl").write_text("\n".join(json.dumps(c) for c in (
        {"pass": "primary", "ok": True, "input_tokens": 1_000_000, "output_tokens": 100_000, "thinking_tokens": 400_000},
        {"pass": "replicate", "ok": False, "input_tokens": None, "output_tokens": None, "thinking_tokens": None},
        {"pass": "replicate", "ok": True, "input_tokens": 1_000_000, "output_tokens": 0, "thinking_tokens": 0})) + "\n")
    return root


@pytest.fixture(scope="module")
def result(run):
    return px.load_pass(run / "v000", "primary")[0]


# ---- the run ---------------------------------------------------------------------------------

def test_resolve_run_takes_a_name_under_runs_or_a_full_path(run, tmp_path):
    assert px.resolve_run(str(run)) == run and px.resolve_run(run) == run
    (tmp_path / "runs" / "mine").mkdir(parents=True)
    assert px.resolve_run("mine", repo=tmp_path) == tmp_path / "runs" / "mine"
    with pytest.raises(ConfigError, match="run 'missing' not found .* Runs available: mine"):
        px.resolve_run("missing", repo=tmp_path)


def test_pick_video_and_load_pass_fall_back_with_a_plain_note(run):
    assert px.pick_video(run).name == "v000" and px.pick_video(run, "v002").name == "v002"
    with pytest.raises(ConfigError, match="'nope' is not in this run. Videos: v000, v001, v002"):
        px.pick_video(run, "nope")
    assert px.passes_present(run / "v000") == ["primary", "replicate"]
    _, used, note = px.load_pass(run / "v000", "replicate")
    assert used == "replicate" and note == "Showing the replicate pass of v000."
    _, used, note = px.load_pass(run / "v000", "reviewed")
    assert used == "primary" and "has no reviewed pass (it has: primary, replicate); showing the primary pass" in note
    with pytest.raises(ConfigError, match="PASS must be one of primary, replicate, reviewed"):
        px.load_pass(run / "v000", "first")


def test_overview_counts_passes_checks_calls_tokens_and_cost(run):
    before = _snapshot(run)
    table = px.run_overview(run, input_usd_per_m=2.0, output_usd_per_m=10.0).set_index("Video")
    assert _snapshot(run) == before
    assert list(table.index) == ["v000", "v001", "v002"]
    row = table.loc["v000"]
    assert row["Passes"] == "primary, replicate" and row["Human-reviewed"] == "no" and row["Release"] == "v3-dev9"
    assert row["primary: errors / flags"] == "1 / 2" and row["replicate: errors / flags"] == "not checked"
    assert row["Model calls"] == 3 and row["Failed calls"] == 1
    assert (row["Input tokens"], row["Output tokens"], row["Thinking tokens"]) == (2_000_000, 100_000, 400_000)
    assert row["Estimated cost (USD)"] == pytest.approx(2 * 2.0 + 0.5 * 10.0)
    assert table.loc["v001", "Model calls"] == 0 and table.loc["v001", "Estimated cost (USD)"] == 0
    assert "reviewed: errors / flags" not in table.columns                      # no video has that pass
    summary = px.overview_summary(table.reset_index())
    assert summary.startswith("3 videos; instrument release v3-dev9; 0 human-reviewed; 3 model calls")
    assert "$9.00" in summary
    # which pass the analysis reads: the primary, also with an error when the replicate was not checked
    assert set(table["Analysis reads"]) == {"primary"} and px.analysis_pass(run / "v000") == "primary"
    assert "The analysis reads: the primary pass of 3 (the replicate only where" in summary


def test_overview_names_the_replicate_where_the_primary_fails_validation_and_the_replicate_fails_less(run, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(run, root)
    _dump(root / "v000" / "results" / "replicate_validation.json", {"errors": [], "flags": []})
    table = px.run_overview(root).set_index("Video")
    assert table.loc["v000", "Analysis reads"] == "replicate" and table.loc["v001", "Analysis reads"] == "primary"
    assert "the primary pass of 2, the replicate pass of 1" in px.overview_summary(table.reset_index())
    (root / "v001" / "results" / "primary.json").unlink()
    assert px.analysis_pass(root / "v001") == px.DASH                            # nothing the analysis could read


def test_video_ids_can_name_a_selection_file_of_the_run(run, tmp_path):
    root = tmp_path / "copy"
    shutil.copytree(run, root)
    assert px.resolve_video_ids(root, None) is None
    assert px.resolve_video_ids(root, ("v002", "v000")) == ["v002", "v000"]
    _dump(root / "selection_a.json", {"selection": "two of them", "videos": [{"video_id": "v002", "path": "x.mp4"},
                                                                             {"video_id": "v000"}]})
    _dump(root / "selection_b.json", {"video_ids": ["v001"]})
    _dump(root / "selection_c.json", ["v000", "v001"])
    assert px.resolve_video_ids(root, "selection_a.json") == ["v002", "v000"]    # the file's order
    assert px.resolve_video_ids(root, "selection_b.json") == ["v001"]
    assert px.resolve_video_ids(root, str(root / "selection_c.json")) == ["v000", "v001"]
    assert list(px.run_overview(root, video_ids=px.resolve_video_ids(root, "selection_a.json"))["Video"]) == ["v000", "v002"]
    with pytest.raises(ConfigError, match="not in .*Selection files there: selection_a.json, selection_b.json, selection_c.json"):
        px.resolve_video_ids(root, "selection_x.json")
    _dump(root / "selection_bad.json", {"videos": [{"id": "v000"}]})
    with pytest.raises(ConfigError, match="is not a selection file"):
        px.resolve_video_ids(root, "selection_bad.json")


# ---- the instrument --------------------------------------------------------------------------

def test_instrument_is_read_from_the_bundle_and_compared_with_the_live_release(run):
    inst = px.instrument(run / "v000")
    assert inst.source == "bundle" and inst.release_id == "v3-dev9"
    assert set(inst.differences["Same content"]) == {"yes"} and len(inst.differences) == 7
    assert "identical to the live release" in px.instrument_note(inst)
    assert inst.prompt_hash == inst.recorded_hash and "matches the hash recorded" in px.prompt_note(inst)
    assert "5,024 characters" in px.prompt_note(inst)
    excerpt = px.prompt_excerpt(inst, head=20, tail=10)
    assert excerpt.startswith("PROMPT START") and excerpt.endswith("PROMPT END") and "characters not shown" in excerpt
    assert px.prompt_excerpt(inst, full=True) == inst.prompt_text

    # a bundle that differs from the live release is shown as it is and the difference is named
    yaml_path = run / "v000" / "bundle" / "venture_context.yaml"
    original = yaml_path.read_text()
    try:
        yaml_path.write_text(original.replace("Primary industry", "Main industry"))
        changed = px.instrument(run / "v000")
        assert "Main industry" in set(px.codebook_table(changed.books.venture)["Name"])
        assert "has since changed in 1 of 7 files: venture_context.yaml" in px.instrument_note(changed)
    finally:
        yaml_path.write_text(original)

    live = px.instrument(run / "v001")                                           # no bundle, no prompt
    assert live.source == "live" and live.differences.empty and live.prompt_text is None
    assert "LIVE release" in px.instrument_note(live) and px.prompt_note(live) == "This video kept no prompt file."
    assert px.prompt_excerpt(live) == ""


def test_codebook_table_gives_values_anchors_and_rules(books):
    table = px.codebook_table(books.panel).set_index("Code")
    assert list(table.columns) == ["Name", "Section", "Applies to", "Type", "Allowed values or scale", "Definition",
                                   "Use when", "Do not use when"]
    assert table.loc["PR-SEV", "Name"] == "Question severity" and table.loc["PR-SEV", "Applies to"] == "turn (question)"
    assert table.loc["PR-SEV", "Allowed values or scale"].startswith("1 = ") and " · 3 = " in table.loc["PR-SEV", "Allowed values or scale"]
    assert table.loc["RS-RESP", "Allowed values or scale"] == "one of: addresses, partial, deflects, NONE"
    assert table.loc["RS-EVID", "Allowed values or scale"] == "true or false"
    stage_a = px.codebook_table(books.stage_a)
    assert list(stage_a["Code"]) == list(STAGE_A_IDS) and stage_a["Name"].iloc[0] == "pitch positivity"
    venture = px.codebook_table(books.venture)
    assert len(venture) == 5 and venture.iloc[-1]["Code"] == "VC-ASK" and venture.iloc[-1]["Allowed values or scale"] == "number"


# ---- one record ------------------------------------------------------------------------------

def test_record_tables_show_people_by_name_and_times_as_clock(result, books, run):
    record = result["record"]
    actors = px.actors_table(result)
    assert list(actors.columns) == ["Id", "Role", "Name"] and len(actors) == len(record["actors"])
    name = {a["actor_id"]: f"{a['label']} ({a['actor_id']})" for a in record["actors"]}

    turns = px.turn_table(result, books, group_by_exchange=False)
    assert list(turns["Turn"]) == [t["turn_id"] for t in record["turns"]]
    first_question = turns[turns.Function == "question"].iloc[0]
    source = next(t for t in record["turns"] if t["function"] == "question")
    assert first_question["Speaker"] == name[source["speaker"]]
    assert first_question["Clock"] == f"{px.clock(source['start_s'])}–{px.clock(source['end_s'])}"
    assert first_question["Concept"] == "operating_efficiency" and first_question["Frame (derived)"] == "prevention"
    assert first_question["Ratings"].startswith("Question severity: ")
    answer = turns[turns.Function == "answer"].iloc[0]
    assert answer["Responds to"] == f"turn {source['turn_id']}" and "Answer directness: " in answer["Ratings"]
    assert "missing (not_applicable)" in answer["Ratings"]                       # a missing value is said, not hidden
    assert turns[turns.Function == "presents"]["Exchange"].iloc[0] == px.NO_EXCHANGE

    grouped = px.turn_table(result, books)
    assert sorted(grouped["Turn"]) == sorted(turns["Turn"])
    runs = [h for i, h in enumerate(grouped["Exchange"]) if i == 0 or h != grouped["Exchange"].iloc[i - 1]]
    exchange_runs = [h for h in runs if h != px.NO_EXCHANGE]
    assert len(exchange_runs) == len(set(exchange_runs)) == len(record["exchanges"])      # each exchange is one block
    assert exchange_runs[0].startswith("Exchange 1 · Question and answer · topic: ")

    exchanges = px.exchange_table(result)
    assert list(exchanges["Exchange"]) == [e["exchange_id"] for e in record["exchanges"]]
    assert list(exchanges.columns) == ["Exchange", "Kind", "Topic", "Label", "Clock", "Turns"]   # no closing since v3-dev9
    assert int(exchanges["Turns"].sum()) == sum(t["exchange_id"] is not None for t in record["turns"])

    decisions = px.decisions_table(result)
    assert list(decisions["Kind"]) == [d["kind"] for d in record["decisions"]]
    assert decisions["Who"].iloc[0] == name[record["decisions"][0]["actor"]]
    assert decisions["Clock"].iloc[0] == px.clock(record["decisions"][0]["time_s"])

    window = px.window_table(result).set_index("Item")["Value"]
    assert window["Prepared presentation"] == "; ".join(
        f"{px.clock(s['start_s'])}–{px.clock(s['end_s'])}" for s in record["prepared_spans"])
    assert window["Footage complete"] in ("yes", "no")

    stage_a = px.stage_a_table(run / "v000", books, run).set_index("Code")
    assert list(stage_a.index) == list(STAGE_A_IDS) and {"primary", "replicate", "Note"} <= set(stage_a.columns)
    assert stage_a.loc[STAGE_A_IDS[0], "replicate"] != stage_a.loc[STAGE_A_IDS[0], "primary"]
    assert "Rank score" not in stage_a.columns                                    # this run has no ranking

    venture = px.venture_table(run / "v000", books).set_index("Field")
    assert venture.loc["Presenting team gender composition", "primary"] == record["venture"]["founder_gender"]
    assert {"primary", "replicate"} <= set(venture.columns)


def test_prepared_and_ask_tables_show_every_pass(run, books):
    """Plan B task 9: the explorer shows the prepared presentation's content ratings and the ask, per pass."""
    record = json.loads((run / "v000" / "results" / "primary.json").read_text())["record"]
    prepared = px.prepared_table(run / "v000", books).set_index("Code")
    assert list(prepared.index) == [e.id for e in books.prepared_codes()] and {"primary", "replicate"} <= set(prepared.columns)
    for item in record["prepared"]:
        shown = prepared.loc[item["code"], "primary"]
        assert shown == item["value"] if item["value"] not in (None, "", "NONE") else shown.startswith("missing (")
    ask = px.ask_table(run / "v000").set_index("Field")
    assert list(ask.index) == ["Amount asked (USD)", "Equity offered (%)", "What is missing from the ask"]
    amount = record["venture"]["ask_amount_usd"]
    assert ask.loc["Amount asked (USD)", "primary"] == (px.DASH if amount is None else f"{amount:,.0f}")
    assert ask.loc["What is missing from the ask", "replicate"] == record["venture"]["ask_missing"]


def test_other_concepts_and_validation_tables(run, result):
    assert px.other_concepts_table(result).empty
    named = copy.deepcopy(result)
    question = next(t for t in named["record"]["turns"] if t["function"] == "question")
    question["codes"][0]["value"] = "other"
    question["other_concept"] = "shelf   life"
    table = px.other_concepts_table(named)
    assert list(table["Name given"]) == ["shelf life"] and table["Where"].iloc[0] == f"turn {question['turn_id']}"
    assert table["Context"].iloc[0] == question["summary"]
    concept = px.turn_table(named, load_panel_books(load_release())).set_index("Turn").loc[question["turn_id"], "Concept"]
    assert concept == "other: shelf   life"

    checks = px.validation_table(run / "v000", "primary")
    assert list(checks["Level"]) == ["error", "flag", "flag"]
    assert list(checks["Rule"]) == ["exchange.topic", "flag.unanswered_question", ""]
    assert checks["Message"].iloc[2] == "a message without a rule"
    assert px.validation_table(run / "v001", "primary").empty


def test_timeline_figure_has_a_row_per_person_and_a_marker_per_decision(result):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    record = result["record"]
    fig = px.timeline_figure(result, "a title")
    ax = fig.axes[0]
    assert ax.get_title() == "a title"
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        f"{a['label']} ({a['actor_id']})" for a in reversed(record["actors"])]
    assert len(ax.lines) == len(record["decisions"]) and len(ax.collections) == len(record["turns"])
    assert {t.get_text() for t in ax.get_legend().get_texts()} >= {"Question and answer", "presentation", "decision"}
    plt.close(fig)


# ---- agreement -------------------------------------------------------------------------------

def test_agreement_table_counts_what_the_report_holds_and_pools_by_summing(run, books):
    reports = px.agreement_reports(run)
    assert set(reports) == {"v000", "v001", "v002"}                              # the stubs have a turns block too
    one = px.agreement_table(px.agreement_reports(run, ["v000"]), books).set_index("Field")
    n_questions_answers = one.loc["Question concept (PR-CONCEPT)", "Compared"]
    assert one.loc["Question concept (PR-CONCEPT)", "Agree"] == n_questions_answers - 1      # the one planted change
    assert one.loc["Question frame (derived from the concept)", "Compared"] == n_questions_answers
    assert one.loc["Question severity (PR-SEV)", "Within one point"] == 1.0
    assert pd.isna(one.loc["Question concept (PR-CONCEPT)", "Within one point"])              # a category has no distance
    topic = one.loc["Exchange topic, among paired substantive exchanges"]
    assert topic["Agree"] == topic["Compared"] - 1 and 0 < topic["Share agreeing"] < 1
    assert "How an investor left an exchange (closing state)" not in one.index
    stage_a = one.loc["Stage A ratings of the prepared presentation"]
    assert (stage_a["Compared"], stage_a["Agree"], stage_a["Within one point"]) == (3, 1, 0.67)
    assert one.loc["Whether each investor ever offered", "Share agreeing"] == 1.0
    assert one.loc["Turn function, among turns paired across passes", "Share agreeing"] == 1.0
    assert one.loc["Venture context (industry, form, customer, team gender)", "Compared"] == 4

    doubled = px.agreement_table({"a": reports["v000"], "b": reports["v000"]}, books).set_index("Field")
    assert (doubled["Compared"] == 2 * one["Compared"]).all() and (doubled["Agree"] == 2 * one["Agree"]).all()
    assert doubled["Share agreeing"].equals(one["Share agreeing"])
    assert px.agreement_table({}, books).empty
    assert px.agreement_table({"v001": reports["v001"]}, books).empty            # a stub holds no counts: no rows, no error
    extras = px.agreement_extras(reports)
    assert list(extras["Video"]) == ["v000", "v001", "v002"] and extras["Prepared window overlap"].iloc[0] == 1.0


def test_agreement_table_covers_outcomes_venture_fields_the_prepared_window_and_every_stage_a_item(run, books):
    report = px.agreement_reports(run, ["v000"])["v000"]
    table = px.agreement_table({"v000": report}, books)
    assert list(table.columns) == px.AGREEMENT_COLUMNS
    assert list(dict.fromkeys(table["Part"])) == list(px.AGREEMENT_PARTS)        # grouped, in the order of the parts
    one = table.set_index("Field")
    assert (one.loc["Whether the pitch got any offer", ["Compared", "Agree"]] == 1).all()
    n_decisions = len(report["decisions"]["matched"])
    for field in ("Decisions found by both passes", "Decisions of the replicate pass also found in the primary"):
        assert (one.loc[field, "Compared"], one.loc[field, "Agree"]) == (n_decisions, n_decisions)
    for field in ("Primary industry (primary_industry)", "Product form (product_form)", "Customer type (customer_type)",
                  "Presenting team gender composition (founder_gender)"):
        assert one.loc[field, "Part"] == "Venture context" and (one.loc[field, ["Compared", "Agree"]] == 1).all()
    window = one.loc["Prepared presentation window (seconds in both passes' windows)"]
    assert window["Compared"] == window["Agree"] == report["prepared"]["primary_s"] and window["Share agreeing"] == 1.0
    # Stage A item by item: the fixture moved the first item by one point and the second by two
    items = one[one.index.str.startswith("Stage A: ")]
    assert len(items) == len(STAGE_A_IDS) and (items["Compared"] == 1).all()
    assert list(items["Agree"]) == [0, 0, 1] and list(items["Within one (count)"]) == [1, 0, 1]
    assert items.index[0].endswith(f"({STAGE_A_IDS[0]})")
    assert one.loc["Question severity (PR-SEV)", "Within one (count)"] == one.loc["Question severity (PR-SEV)", "Compared"]
    assert pd.isna(one.loc["Question concept (PR-CONCEPT)", "Within one (count)"])
    # a window that differs: 100 s and 80 s sharing 80 s
    moved = {**report, "prepared": {"primary_s": 100.0, "replicate_s": 80.0, "iou": 0.8}}
    window = px.agreement_table({"a": moved, "b": moved}, books).set_index("Field").loc[window.name]
    assert (window["Compared"], window["Agree"], window["Share agreeing"]) == (200, 160, 0.8)
    # a pitch whose two passes disagree on any offer, and a decision only the replicate has
    decisions = {**report["decisions"], "any_offer": {"primary": True, "replicate": False}, "unmatched_replicate": [9, 10]}
    changed = px.agreement_table({"a": {**report, "decisions": decisions}}, None).set_index("Field")
    assert changed.loc["Whether the pitch got any offer", "Agree"] == 0
    assert changed.loc["Decisions of the replicate pass also found in the primary", "Compared"] == n_decisions + 2
    assert changed.loc["Decisions found by both passes", "Compared"] == n_decisions
    assert f"Stage A: {STAGE_A_IDS[0]}" in changed.index and "founder_gender" in changed.index   # without books: the ids


def test_function_agreement_table_pools_counts_per_function():
    def report(**functions):
        return {"turns": {"by_function": {name: dict(zip(("primary", "replicate", "matched"), counts))
                                          for name, counts in functions.items()}}}
    table = px.function_agreement_table({"a": report(answer=(10, 8, 7), accepts=(2, 0, 0)),
                                         "b": report(answer=(10, 12, 9), challenges=(1, 3, 1))})
    assert list(table["Function"]) == ["answer", "challenges", "accepts", "all functions"]     # most frequent first
    answer = table.set_index("Function").loc["answer"]
    assert (answer["Primary turns"], answer["Replicate turns"], answer["Matched with the same function"]) == (20, 20, 16)
    assert answer["Agreement"] == 0.8                                              # 2 x 16 / (20 + 20)
    assert table.set_index("Function").loc["challenges", "Agreement"] == 0.5
    assert table.set_index("Function").loc["accepts", "Agreement"] == 0.0
    total = table.iloc[-1]
    assert (total["Primary turns"], total["Replicate turns"], total["Matched with the same function"]) == (23, 23, 17)
    assert px.function_agreement_table({}).empty and px.function_agreement_table({"a": {"turns": {}}}).empty


def test_confusion_tables_and_the_list_of_disagreements(run):
    reports = px.agreement_reports(run, ["v000"])
    concept = px.confusion_table(reports, "PR-CONCEPT")
    assert concept.loc["operating_efficiency", "competition"] == 1
    assert int(concept.to_numpy().sum()) == reports["v000"]["codes"]["PR-CONCEPT"]["both_valued"]
    topic = px.confusion_table(reports, "topic")
    assert int(topic.to_numpy().sum()) == reports["v000"]["exchanges"]["topic_n"]
    assert int(np.trace(topic.to_numpy())) == reports["v000"]["exchanges"]["topic_agree"]
    twice = px.confusion_table({"a": reports["v000"], "b": reports["v000"]}, "topic")
    assert (twice.to_numpy() == 2 * topic.to_numpy()).all()
    assert px.confusion_table(reports, "no-such-code").empty and px.confusion_table({}, "topic").empty
    assert px.confusion_table(reports, "closing").empty                     # no closing state since v3-dev9

    table = px.video_disagreements(run / "v000")
    assert list(table.columns) == ["What", "Primary clock", "Replicate clock", "About", "Detail", "Kind"]
    kinds = set(table["Kind"])
    assert {"code_disagreement", "exchange_topic_disagreement", "stage_a_disagreement"} <= kinds
    code = table[table.Kind == "code_disagreement"].iloc[0]
    primary = json.loads((run / "v000" / "results" / "primary.json").read_text())["record"]
    question = next(t for t in primary["turns"] if t["function"] == "question")
    assert code["What"] == "rating differs" and code["Primary clock"] == code["Replicate clock"] == px.clock(question["start_s"])
    stage = table[table.Kind == "stage_a_disagreement"].iloc[0]
    assert stage["Primary clock"] == px.DASH and stage["About"] == STAGE_A_IDS[1]            # not at a moment
    clocks = [c for c in table["Primary clock"] if c != px.DASH]
    assert clocks == sorted(clocks, key=lambda c: [int(x) for x in c.split(":")])           # in time order
    only = px.video_disagreements(run / "v000", kinds=["exchange_topic_disagreement"])
    assert set(only["Kind"]) == {"exchange_topic_disagreement"}
    assert px.video_disagreements(run / "v001").empty                            # a stub report flags nothing
    (run / "v002" / "agreement.json").rename(run / "v002" / "agreement.off")
    try:
        assert px.video_disagreements(run / "v002").empty and set(px.agreement_reports(run)) == {"v000", "v001"}
    finally:
        (run / "v002" / "agreement.off").rename(run / "v002" / "agreement.json")


# ---- review, ranking, comparison, export -----------------------------------------------------

def test_review_status_summary_and_commands(run):
    status = px.review_status(run).set_index("Video")
    assert set(status["Review sheet"]) == {"not exported"} and set(status["Reviewed record"]) == {"no"}
    assert set(status["Human against model"]) == {"not yet imported"} and status["Rows reviewed"].sum() == 0
    assert px.review_summary(run) is None
    _dump(run / "v001" / "review" / "export_manifest.json", {"videos": {"v001": {"rows": [{}, {}, {}]}}})
    try:
        assert px.review_status(run).set_index("Video").loc["v001", "Review sheet"] == "exported (3 rows)"
    finally:
        shutil.rmtree(run / "v001" / "review")
    text = px.review_commands(run, "v000")
    assert f"review_panel.py export --run {run} --video-ids v000" in text
    assert f"review_panel.py import --run {run} --sheet {run / 'v000' / 'review' / 'v000__review.csv'} --reviewer" in text
    assert "review_panel.py status" in text and "review_panel.py summary" in text


def test_pairwise_order_agreement_counts_pairs_seen_more_than_once():
    assert px.pairwise_order_agreement([["a", "b", "c"], ["a", "b", "c"]]) == {
        "pairs_seen_twice": 3, "consistent": 3, "share": 1.0}
    flipped = px.pairwise_order_agreement([["a", "b", "c"], ["a", "c", "b"]])
    assert flipped == {"pairs_seen_twice": 3, "consistent": 2, "share": pytest.approx(2 / 3)}
    partial = px.pairwise_order_agreement([["a", "b", "c"], ["b", "a", "d"]])      # only a-b is seen twice
    assert partial == {"pairs_seen_twice": 1, "consistent": 0, "share": 0.0}
    assert px.pairwise_order_agreement([["a", "b"]]) == {"pairs_seen_twice": 0, "consistent": 0, "share": None}


def test_ranking_tables_read_batches_and_scores_and_are_empty_without_a_ranking(run, books, tmp_path):
    assert px.ranking_orders(run).empty and px.ranking_scores(run).empty and px.ranking_agreement(run).empty
    ranked = tmp_path / "ranked"
    shutil.copytree(run, ranked)
    videos = ["v000", "v001", "v002"]
    for batch, flip in (("b01", False), ("b02", True)):
        rankings = {code: list(videos) for code in STAGE_A_IDS}
        if flip:
            rankings[STAGE_A_IDS[0]] = ["v001", "v000", "v002"]
        _dump(ranked / "ranking" / "batches" / batch / "result.json",
              {"batch_id": batch, "video_ids": videos, "rankings": rankings})
    _dump(ranked / "ranking" / "scores.json",
          {**{code: {"pl_score": {"v000": 1.234, "v001": 0.0, "v002": -1.2}, "n_orders": 2} for code in STAGE_A_IDS},
           "batches_ok": 2})
    orders = px.ranking_orders(ranked)
    assert len(orders) == 2 * len(STAGE_A_IDS) and list(orders.columns) == ["Dimension", "Batch", "1 (strongest)", "2", "3"]
    first = orders[(orders.Dimension == STAGE_A_IDS[0]) & (orders.Batch == "b02")].iloc[0]
    assert first["1 (strongest)"] == "v001"
    scores = px.ranking_scores(ranked)
    assert list(scores.index) == videos and scores.loc["v000", STAGE_A_IDS[0]] == 1.23 and scores.shape == (3, 3)
    agreement = px.ranking_agreement(ranked).set_index("Dimension")
    assert agreement.loc[STAGE_A_IDS[0], "Share consistent"] == 0.67 and agreement.loc[STAGE_A_IDS[1], "Share consistent"] == 1.0
    assert (agreement["Batches"] == 2).all() and (agreement["Pairs ranked more than once"] == 3).all()
    with_rank = px.stage_a_table(ranked / "v000", books, ranked).set_index("Code")
    assert with_rank.loc[STAGE_A_IDS[0], "Rank score"] == 1.23


def test_spearman_is_the_correlation_of_ranks_with_ties_sharing_a_rank():
    assert px.spearman([1, 2, 3, 4], [10, 20, 30, 400]) == pytest.approx(1.0)       # monotone, not linear
    assert px.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # ranks 1, 2.5, 2.5, 4 against 1, 2, 3, 4
    assert px.spearman([1, 2, 2, 5], [1, 2, 3, 4]) == pytest.approx(np.corrcoef([1, 2.5, 2.5, 4], [1, 2, 3, 4])[0, 1])
    assert np.isnan(px.spearman([1, 1, 1], [1, 2, 3])) and np.isnan(px.spearman([1, 2], [1, 2]))


def test_ranking_stability_sets_each_videos_first_batch_against_its_second(tmp_path, books):
    """Six videos, each ranked in two batches of three: what counts is a video's place in each of its batches."""
    root = write_run(tmp_path, 6, seed=2)
    assert px.ranking_stability(root).empty and px.batch_standings(root).empty
    assert "cannot be computed" in px.ranking_stability_reading(px.ranking_stability(root))
    same, flipped, mixed = STAGE_A_IDS[:3]
    first = {"b01": ["v000", "v001", "v002"], "b02": ["v003", "v004", "v005"]}
    second = {same: {"b03": ["v000", "v004", "v002"], "b04": ["v003", "v001", "v005"]},       # each keeps its place
              flipped: {"b03": ["v002", "v004", "v000"], "b04": ["v005", "v001", "v003"]},    # first and last swap
              mixed: {"b03": ["v000", "v004", "v003"], "b04": ["v001", "v005", "v002"]}}
    for batch in ("b01", "b02", "b03", "b04"):
        orders = {code: (first[batch] if batch in first else second[code][batch]) for code in (same, flipped, mixed)}
        _dump(root / "ranking" / "batches" / batch / "result.json", {"batch_id": batch, "rankings": orders})
    scores = {code: {"pl_score": {f"v{i:03d}": float(i) for i in range(6)}} for code in (same, flipped, mixed)}
    _dump(root / "ranking" / "scores.json", {**scores, "batches_ok": 4})
    standings = px.batch_standings(root)
    row = standings[(standings.code == same) & (standings.video_id == "v001")]
    assert list(row.batch_id) == ["b01", "b04"] and list(row.standing) == [0.5, 0.5] and list(row.appearance) == [1, 2]
    assert set(standings.standing) == {0.0, 0.5, 1.0}                              # 1 = the strongest of its batch
    before = _snapshot(root)
    table = px.ranking_stability(root, n_boot=200, seed=1)
    assert _snapshot(root) == before and list(table.columns) == px.STABILITY_COLUMNS
    by = table.set_index("Dimension")
    column = "Batch-to-batch correlation of standing"
    assert (by["Videos ranked in two batches"] == 6).all()
    assert by.loc[same, column] == 1.0 and by.loc[flipped, column] == -1.0
    first_s = {"v000": 1, "v001": .5, "v002": 0, "v003": 1, "v004": .5, "v005": 0}
    second_s = {"v000": 1, "v004": .5, "v003": 0, "v001": 1, "v005": .5, "v002": 0}
    expected = np.corrcoef([first_s[v] for v in sorted(first_s)], [second_s[v] for v in sorted(first_s)])[0, 1]
    assert by.loc[mixed, column] == round(expected, 2)
    assert by.loc[mixed, "95% interval, low"] <= by.loc[mixed, column] <= by.loc[mixed, "95% interval, high"]
    assert px.ranking_stability(root, n_boot=200, seed=1).equals(table)             # seeded
    # the ratings: both passes of a synthetic video hold the same record, so they agree wherever they vary
    ratings = {v: {i["code"]: i["value"] for i in json.loads((root / v / "results" / "primary.json").read_text())["record"]["stage_a"]}
               for v in first_s}
    values = [ratings[v][mixed] for v in sorted(ratings)]
    assert by.loc[mixed, "Videos rated in both passes"] == 6 == by.loc[mixed, "Videos with a rating and a rank score"]
    if len(set(values)) > 1:
        assert by.loc[mixed, "Primary vs replicate rating (Spearman)"] == 1.0
        assert by.loc[mixed, "Primary rating vs rank score (Spearman)"] == round(px.spearman(values, range(6)), 2)
    reading = px.ranking_stability_reading(table)
    assert reading.startswith("Over 6 videos, each ranked in two different batches") and f"-1.00 ({flipped})" in reading
    assert "Around 0.5 is moderate" in reading and "steadier than either one" in reading and "not validity" in reading


def test_comparison_tables_only_when_the_file_exists(run, tmp_path):
    assert px.comparison_tables(run) is None
    compared = tmp_path / "compared"
    compared.mkdir()
    _dump(compared / "comparison_vs_structured.json", {"old_run": "runs/old", "videos": {"v000": {
        "coverage": {"old_cutoff_s": 125.0, "new_last_turn_end_s": 301.0, "new_turns": 40, "new_turns_after_cutoff": 12,
                     "new_questions": 9, "new_questions_after_cutoff": 2, "new_answers": 9, "new_answers_after_cutoff": 2,
                     "old_episodes": 5, "new_substantive_exchanges_after_cutoff": 1},
        "cost": {"old_calls": 20, "new_calls": 2, "old_input_tokens": 500, "new_input_tokens": 100,
                 "old_output_tokens": 50, "new_output_tokens": 40, "old_thinking_tokens": 70, "new_thinking_tokens": 30}}}})
    tables = px.comparison_tables(compared)
    coverage = tables["coverage"].iloc[0]
    assert coverage["Old pipeline stopped at"] == "2:05" and coverage["This run's last turn ends"] == "5:01"
    assert coverage["Turns after the old stop"] == 12
    cost = tables["cost"].iloc[0]
    assert cost["Calls (structured pilot)"] == 20 and cost["Calls (this run)"] == 2
    assert cost["Thinking tokens (this run)"] == 30
    (compared / "comparison_vs_structured.json").write_text("not json")
    assert px.comparison_tables(compared) is None


def test_export_writes_csv_files_under_the_explorer_folder_only(run, tmp_path):
    copy_root = tmp_path / "copy"
    shutil.copytree(run, copy_root)
    before = _snapshot(copy_root)
    out = px.export_tables(copy_root, {"run_overview": px.run_overview(copy_root),
                                      "confusion": px.confusion_table(px.agreement_reports(copy_root, ["v000"]), "topic"),
                                      "not_a_table": None})
    assert out == copy_root / "analysis" / "explorer"
    assert sorted(p.name for p in out.iterdir()) == ["confusion.csv", "run_overview.csv"]
    assert list(pd.read_csv(out / "run_overview.csv")["Video"]) == ["v000", "v001", "v002"]
    assert pd.read_csv(out / "confusion.csv", index_col=0).shape[0] >= 2          # a labelled index is kept
    after = _snapshot(copy_root)
    assert {k: v for k, v in after.items() if not k.startswith("analysis/explorer/")} == before
    # the explorer folder is not mistaken for a video
    assert [d.name for d in px.video_dirs(copy_root)] == ["v000", "v001", "v002"]


def test_show_prints_the_explanation_when_there_is_nothing_to_show(capsys):
    px.show(pd.DataFrame(), "Nothing here because the run has no ranking.")
    px.show(None, "Nor here.")
    assert capsys.readouterr().out == "Nothing here because the run has no ranking.\nNor here.\n"
    html = px.wide(pd.DataFrame({"Text": ["a < b " + "long " * 40], "N": [None]})).data
    assert "a &lt; b" in html and "long long" in html and "..." not in html and "white-space:normal" in html
    grouped = px.grouped(pd.DataFrame({"G": ["one", "one", "two"], "V": [1, 2, 3]}), by="G").data
    assert grouped.count("colspan") == 2 and grouped.index("one") < grouped.index("two") and "<th>G</th>" not in grouped


# ---- a run still being annotated, and what follows each answer -------------------------------------

def test_video_ids_keep_every_view_to_the_named_videos_whatever_else_is_in_the_run(run, tmp_path):
    root = tmp_path / "live"
    shutil.copytree(run, root)
    (root / "v900" / "results").mkdir(parents=True)                       # started, no pass yet
    (root / "v901" / "results").mkdir(parents=True)
    (root / "v901" / "results" / "primary.json").write_text("{ half written")
    (root / "v901" / "agreement.json").write_text("{ half written")
    before = _snapshot(root)
    keep = ["v000", "v002"]
    assert [d.name for d in px.video_dirs(root, keep)] == keep
    assert [d.name for d in px.video_dirs(root)] == ["v000", "v001", "v002", "v901"]   # v900 has no pass at all
    assert px.pick_video(root, None, keep).name == "v000" and px.pick_video(root, "v002", keep).name == "v002"
    with pytest.raises(ConfigError, match="video 'v001' is not in this run. Videos: v000, v002"):
        px.pick_video(root, "v001", keep)
    assert list(px.run_overview(root, video_ids=keep)["Video"]) == keep
    assert set(px.agreement_reports(root, keep)) <= set(keep)
    assert list(px.review_status(root, keep)["Video"]) == keep and px.review_summary(root, keep) is None
    (root / "ranking").mkdir(exist_ok=True)
    (root / "ranking" / "scores.json").write_text(json.dumps(
        {"SA-X": {"pl_score": {"v000": 0.5, "v001": 0.1, "v002": -0.6}}, "batches_ok": 1}))
    assert list(px.ranking_scores(root, keep).index) == keep and len(px.ranking_scores(root)) == 3
    with pytest.raises(ConfigError, match="VIDEO_IDS names v900, which has no annotation pass"):
        px.video_dirs(root, ["v000", "v900"])
    assert {k: v for k, v in _snapshot(root).items() if k != "ranking/scores.json"} == {
        k: v for k, v in before.items() if k != "ranking/scores.json"}


def test_followups_table_lists_each_answer_with_its_question_and_the_next_move(run, result, books):
    table = px.followups_table(run / "v000", result, books)
    assert list(table.columns) == [c for c in px.FOLLOWUP_HEADERS if c != "Video"] and len(table) >= 1
    record = result["record"]
    answers = {t["turn_id"]: t for t in record["turns"] if t["function"] == "answer"}
    questions = {t["turn_id"] for t in record["turns"] if t["function"] == "question"}
    linked = sum(1 for t in answers.values() for q in dict.fromkeys(t["responds_to"]) if q in questions)
    assert len(table) == linked
    first = table.iloc[0]
    turn_id = int(first["Answer"].split()[1])
    assert first["Answer summary"] == answers[turn_id]["summary"] and first["Answer"].startswith(f"turn {turn_id} at ")
    assert first["Asked by"].endswith(")") and first["Question summary"]
    assert first["Asker's next move"] in px.ASKER_MOVE_WORDS.values()
    assert first["Panel's next move"] in px.PANEL_MOVE_WORDS.values()
    assert set(table["Substance"]) <= {"1", "2", "3", "4", "5", "not rated"}
    assert set(table["Checkable support"]) <= {"yes", "no", "not rated"}
    no_answers = copy.deepcopy(result)
    no_answers["record"]["turns"] = [t for t in record["turns"] if t["function"] == "presents"]
    no_answers["record"]["exchanges"], no_answers["record"]["decisions"] = [], []
    assert px.followups_table(run / "v000", no_answers, books).empty


def test_ranking_stability_with_three_batches_per_video_averages_the_pairs_of_appearances(tmp_path):
    """Six videos in three batches each: the figure is the mean correlation over the three pairs of appearances."""
    root = tmp_path / "run"
    videos = [f"v{i}" for i in range(6)]
    # appearances 1 and 2 agree exactly (r = +1); appearance 3 reverses both (r = -1 twice): mean = -1/3
    batches = {"b01": videos[:3], "b02": videos[3:], "b03": videos[:3], "b04": videos[3:],
               "b05": videos[:3][::-1], "b06": videos[3:][::-1]}
    for batch, order in batches.items():
        _dump(root / "ranking" / "batches" / batch / "result.json", {"batch_id": batch, "rankings": {"interpersonal_warmth": order}})
    _dump(root / "ranking" / "scores.json", {"interpersonal_warmth": {"pl_score": {v: 0.0 for v in videos}}, "batches_ok": 6})
    table = px.ranking_stability(root, n_boot=50, seed=1)
    row = table.iloc[0]
    assert row["Batches per video"] == 3 and row["Videos ranked in two batches"] == 6
    assert abs(row["Batch-to-batch correlation of standing"] - (-1 / 3)) < 0.01
    reading = px.ranking_stability_reading(table)
    assert "each ranked in 3 different batches" in reading and "Spearman-Brown" in reading


def _write_pass(root: Path, video_id: str, record: dict, name: str = "primary") -> None:
    _dump(root / video_id / "results" / f"{name}.json", {"record": record})


def test_code_value_counts_lists_every_allowed_value_with_zero_for_unused_and_missing_reasons_apart(tmp_path, books):
    """Two videos; every allowed value appears, unused ones with 0; a missing reason written as the value
    counts as missing; topics are counted over substantive exchanges only."""
    turns = [
        {"turn_id": 1, "function": "question", "codes": [{"code": "PR-SEV", "value": "1", "missing": "none"},
                                                         {"code": "PR-CONCEPT", "value": "SALES", "missing": "none"}]},
        {"turn_id": 2, "function": "answer", "codes": [
            {"code": "RS-EVID", "value": "true", "missing": "none"},
            {"code": "PREP-SUBS", "value": None, "missing": "too_short"},
            {"code": "PREP-COH", "value": "not_applicable", "missing": "none"}]},
    ]
    exchanges = [{"exchange_id": 1, "kind": "substantive", "topic": "sales"},
                 {"exchange_id": 2, "kind": "social", "topic": "NONE"}]
    record = {"turns": turns, "exchanges": exchanges, "decisions": [{"kind": "exit"}],
              "stage_a": [{"code": "pitch_positivity", "value": 4, "uncodable": False},
                          {"code": "interpersonal_warmth", "value": None, "uncodable": True}],
              "venture": {"founder_gender": "all_men", "product_form": "physical_good"}}
    _write_pass(tmp_path, "v1", record)
    _write_pass(tmp_path, "v2", {**record, "venture": {"founder_gender": "mixed"}})
    counts = px.code_value_counts(tmp_path, books, pass_name="primary")
    assert list(counts.columns) == px.VALUE_COUNT_COLUMNS

    def row(code, value):
        hit = counts[(counts.code == code) & (counts.value == value)]
        assert len(hit) == 1, (code, value)
        return hit.iloc[0]

    assert row("PR-CONCEPT", "sales")["count"] == 2                      # case folded to the codebook spelling
    assert row("PR-CONCEPT", "vision")["count"] == 0                     # unused values are listed
    assert row("PR-SEV", "3")["count"] == 0 and row("PR-SEV", "1").share == 1.0
    assert row("PREP-SUBS", "too_short").kind == "missing"
    coh = row("PREP-COH", "not_applicable")
    assert coh.kind == "missing" and coh["count"] == 2                   # the word written as a value is missing
    assert "EX-CLOSE" not in set(counts.code)                             # no closing state since v3-dev9
    assert row("EX-TOPIC", "sales").n == 2 and row("EX-KIND", "social")["count"] == 2
    assert row("interpersonal_warmth", "uncodable").kind == "missing"
    assert row("VC-GENDER", "mixed")["count"] == 1 and row("VC-FORM", "physical_good").n == 1
    summary = px.value_usage_summary(counts).set_index("Code")
    assert summary.loc["PR-SEV", "Dominant (80% or more)"] and summary.loc["PR-SEV", "Values never used"] == "2, 3"
    assert "NONE" not in summary.loc["DEC-KIND", "Values never used"]   # NONE is never reported as a dead value
    assert summary.loc["PREP-COH", "Left missing (share)"] == 1.0


def test_value_frequency_figure_draws_one_panel_per_code_of_a_group(tmp_path, books):
    import matplotlib
    matplotlib.use("Agg")
    record = {"turns": [{"turn_id": 1, "function": "answer", "codes": [
        {"code": "RS-EVID", "value": "true", "missing": "none"}, {"code": "RS-UNCERT", "value": "false", "missing": "none"}]}],
        "exchanges": [], "decisions": [], "stage_a": [], "venture": {}}
    _write_pass(tmp_path, "v1", record)
    counts = px.code_value_counts(tmp_path, books, pass_name="primary")
    fig = px.value_frequency_figure(counts, "yes_no", "Yes or no")
    assert len([ax for ax in fig.axes if ax.get_visible()]) == 2
    assert px.value_frequency_figure(counts, "stage_a") is None        # nothing counted in that group


def test_frequency_counts_include_pitch_ratings_count_codes_and_missing(tmp_path, books):
    _write_pass(tmp_path, "v1", {
        "prepared": [{"code": "PRES-SUBS", "value": "4", "missing": "none"},
                     {"code": "PRES-EVID", "value": "2", "missing": "none"}],
        "claim_conflicts": [{"reconciled": True}, {"reconciled": False}],
    })
    _write_pass(tmp_path, "v2", {
        "prepared": [{"code": "PRES-SUBS", "value": None, "missing": "too_short"},
                     {"code": "PRES-EVID", "value": "0", "missing": "none"}],
        "claim_conflicts": [],
    })
    counts = px.code_value_counts(tmp_path, books, pass_name="primary")
    substance = counts[counts.code == "PRES-SUBS"]
    assert substance.n.eq(2).all()
    assert substance.loc[substance.value == "4", "share"].item() == .5
    assert substance.loc[substance.value == "too_short", "kind"].item() == "missing"
    for code in ("PRES-EVID", "PREP-CONSIST"):
        part = counts[counts.code == code]
        assert list(part.value) == ["0", "1", "2"]
        assert list(part["count"]) == [1, 0, 1]
        assert part.kind.eq("value").all()
