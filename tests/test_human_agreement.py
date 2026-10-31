"""The human-agreement notebook's helpers, on synthetic training sheets."""
import json
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from analysis import human_agreement as ha  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from tests.test_review_cli import fill, make_run, read_sheet  # noqa: E402
from tests.test_review_training import page_model_of, training_run, tree_digest  # noqa: E402,F401

rp = ha.review_panel()


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def _sheets(tdir: Path, drop: Path, coder: str, value_of, cut: dict | None = None) -> None:
    """Each training sheet filled by `value_of` on the rows with listed values and, given `cut`, the first decision."""
    for sheet in sorted(tdir.glob("*__review.csv")):
        rows = read_sheet(sheet)
        target = drop / f"{sheet.name.split('__')[0]}__{coder}.csv"
        target.write_bytes(sheet.read_bytes())
        fill(target, {**{r["path"]: value_of(r) for r in rows if r["allowed_values"]},
                      **({"pitch:first_decision": cut} if cut else {})})


@pytest.fixture
def trained(tmp_path, books):
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2"]) == 0
    tdir, drop = run / "review" / "training", tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                          # noqa: E731
    _sheets(tdir, drop, "ana", lambda r: {"human_value": first(r), "note": "ana's note"},
            cut={"human_value": "1:52", "note": "ana's note"})
    # ben agrees except on the first decision (no outcome card since Task 7d), and cannot tell on one Stage A item
    _sheets(tdir, drop, "ben", lambda r: ({"resolution": "unclear", "note": "too dark"} if r["path"] == "stage_a:expressed_enthusiasm"
                                           else {"human_value": first(r)}), cut={"human_value": "2:40"})
    return run, drop


def test_load_pools_the_sheets_without_touching_the_run(trained):
    run, drop = trained
    before = tree_digest(run)
    data = ha.load(run, drop)
    assert data.coders == ["ana", "ben"] and data.videos == ["vid1", "vid2"] and data.bar == 0.61     # Landis & Koch "substantial"
    assert not data.import_failed and "imported" in data.import_log
    assert tree_digest(run) == before


def test_cards_hold_both_answers_the_model_and_the_notes(trained):
    run, drop = trained
    data = ha.load(run, drop)
    cards = ha.cards(data)
    assert set(cards["video"]) == {"vid1", "vid2"} and not cards.duplicated(["video", "path"]).any()
    cut = cards[cards["class"] == "cut"]
    assert len(cut) and (cut["coders agree"] == False).all()                                  # noqa: E712
    assert (cut["ben note"] == "").all()          # the import's own note on a first-decision override is not ben's
    unclear = cards[cards["path"] == "stage_a:expressed_enthusiasm"]
    assert (unclear["ben"].isna() & (unclear["ben flag"] == "unclear") & unclear["coders agree"].isna()).all()
    rest = cards[(cards["class"] != "cut") & (cards["path"] != "stage_a:expressed_enthusiasm")]
    assert (rest["coders agree"] == True).all() and (rest["ana note"] == "ana's note").all()     # noqa: E712
    assert cards["model"].notna().any() and set(cards["class"]) >= {"stage_a", "cut", "turn_code"}
    assert "outcome" not in set(cards["class"])
    assert cards[cards["class"] == "stage_a"]["level"].eq("ordinal").all()


def test_a_note_the_import_wrote_itself_is_not_shown_as_the_coders():
    assert ha.coder_note({"note": ha.OVERRIDE_NOTE}) == "" and ha.coder_note({"note": " too dark "}) == "too dark"
    assert ha.coder_note(None) == "" and ha.coder_note({"note": ha.NO_EXCHANGE_NOTE}) == ""


def test_value_of_reads_a_line_as_the_pipeline_does():
    assert ha.value_of({"resolution": "confirm_primary", "primary": "3", "typed": ""}) == "3"
    assert ha.value_of({"resolution": "confirm_primary", "primary": "1:51", "typed": "1:53"}) == "1:53"
    assert ha.value_of({"resolution": "correct", "new": "4"}) == "4"
    assert ha.value_of({"resolution": "outcome_override", "new": "no"}) == "no"
    assert ha.value_of({"resolution": "unclear", "new": None}) is None
    assert ha.value_of({"resolution": "structural", "new": "x"}) is None


def test_tables_follow_the_pooled_figures(trained):
    run, drop = trained
    data = ha.load(run, drop)
    by_class = ha.agreement_by_class(data).set_index("key")
    hh = data.pooled["human_human"]
    for key in by_class.index:
        assert by_class.loc[key, "coder vs coder: agreement"] == pytest.approx((hh.get(key) or {}).get("agree"), nan_ok=True)
    assert by_class.loc["cut", "coder vs coder: agreement"] == 0.0
    assert by_class.loc["cut", "reached the bar"] in (False, None)
    by_code = ha.agreement_by_code(data)
    # every coded variable, and the question concept's frame (plan D: the frame is the concept's reported row)
    assert set(by_code["key"]) == set(ha.cards(data)["variable"]) | {"PR-FRAME"}
    assert "PREP-COH" in set(by_code["key"])       # answer coherence, a question-answer code since plan C Task 7e
    assert {"ana vs model: α", "ben vs model: α", "model pass vs pass: α"} <= set(by_code.columns)
    talk = ha.disagreements(data)
    cards = ha.cards(data)
    expected = ((cards["coders agree"] == False) | (cards["ana flag"] != "") | (cards["ben flag"] != "")).sum()  # noqa: E712
    assert len(talk) == expected and {"ana note", "ben note", "model"} <= set(talk.columns)
    table = ha.confusion(data, "cut")
    same = sum(int(table.loc[v, v]) for v in table.index if v in table.columns)       # answers that met
    assert same == 0 and int(table.values.sum()) == len(cards[cards["class"] == "cut"])


def test_the_code_table_shows_the_reported_figures_of_each_kind(trained):
    # plan D: where pooled["reporting"] exists, each code carries its kind and, per comparison, the reported raw
    # agreement, alpha, AC1/AC2, alpha's band and the flags, as the paper table has them
    run, drop = trained
    data = ha.load(run, drop)
    reported = data.pooled["reporting"]
    by_code = ha.agreement_by_code(data).set_index("key")
    resp, hh = by_code.loc["RS-RESP"], reported["human_human"]["RS-RESP"]
    assert resp["kind"] == "categorical"
    assert resp["coder vs coder: agreement"] == pytest.approx(hh["figures"]["exact"]["value"])
    assert resp["coder vs coder: AC1/AC2"] == pytest.approx(hh["figures"]["ac1"]["value"])
    assert resp["coders vs model: agreement"] == pytest.approx(reported["RS-RESP"]["figures"]["exact"]["value"])
    assert resp["coders vs model: AC1/AC2"] == pytest.approx(reported["RS-RESP"]["figures"]["ac1"]["value"])
    assert "prevalence" in resp["coder vs coder: flags"] and "small n" in resp["coder vs coder: flags"]
    sev = by_code.loc["PR-SEV"]
    model = reported["PR-SEV"]["figures"]
    assert sev["kind"] == "ordinal" and sev["coders vs model: within one"] == pytest.approx(model["within_one"]["value"])
    assert sev["coders vs model: α"] == pytest.approx(model["alpha"]["value"])
    assert sev["coders vs model: AC1/AC2"] == pytest.approx(model["ac2"]["value"])
    assert sev["coders vs model: band of α"] == data.reporting.band_of(model["alpha"]["value"])
    # the clock: same-turn agreement, no alpha for any comparison or coder
    cut = by_code.loc["cut"]
    assert cut["kind"] == "clock" and cut["coder vs coder: agreement"] == 0.0
    assert cut["coders vs model: agreement"] == pytest.approx(reported["cut"]["figures"]["same_turn"]["value"])
    assert all(pd.isna(cut[c]) for c in ("coder vs coder: α", "coders vs model: α", "ana vs model: α", "ben vs model: α"))
    assert cut["coder vs coder: flags"] == "small n" and cut["reached the bar"] is None
    # nor any of the older columns' alpha, alpha units or level: seconds apart instead (Task 5 review I1)
    assert all(pd.isna(cut[c]) for c in ("coders vs model: α without unsure rows", "α units", "level"))
    assert cut["coder vs coder: median seconds apart"] == pytest.approx(48.0)
    assert cut["coders vs model: median seconds apart"] == pytest.approx(
        reported["cut"]["figures"]["median_abs"]["value"])
    assert cut["coders vs model: agreement without unsure rows"] == pytest.approx(
        reported["cut"]["without_unsure"]["share_within_5"])
    # the exact concept is not reported; its frame follows it as a row of its own
    keys = list(by_code.index)
    assert by_code.loc["PR-CONCEPT", "kind"] == ha.NOT_REPORTED and keys.index("PR-FRAME") == keys.index("PR-CONCEPT") + 1
    frame = by_code.loc["PR-FRAME"]
    assert frame["kind"] == "categorical" and frame["class"] == "turn_code"
    assert frame["coders vs model: agreement"] == pytest.approx(reported["PR-FRAME"]["figures"]["exact"]["value"])
    fig = ha.figure_agreement(data, by_code.reset_index())               # the figure still draws every row
    assert len(fig.axes[0].get_yticklabels()) == len(by_code)


def test_the_code_table_without_the_reporting_keeps_the_pooled_figures(trained):
    # an older pooled summary (no "reporting"): the per-code rows are the pooled alpha and agreement, no frame row
    from dataclasses import replace
    run, drop = trained
    data = ha.load(run, drop)
    older = replace(data, pooled={k: v for k, v in data.pooled.items() if k != "reporting"})
    by_code = ha.agreement_by_code(older).set_index("key")
    assert set(by_code.index) == set(ha.cards(data)["variable"]) and "kind" not in by_code.columns
    hh = data.pooled["human_human"]["by_variable"]
    assert by_code.loc["RS-RESP", "coder vs coder: agreement"] == pytest.approx(hh["RS-RESP"]["agree"])
    assert by_code.loc["cut", "coder vs coder: α"] == pytest.approx(hh["cut"]["alpha"], nan_ok=True)
    assert ha.figure_agreement(older, by_code.reset_index()) is not None


def test_figures_name_every_row(trained):
    run, drop = trained
    data = ha.load(run, drop)
    by_code = ha.agreement_by_code(data)
    fig = ha.figure_agreement(data, by_code)
    left = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert len(left) == len(by_code) and all(left) and left[0].startswith(by_code["code"].iloc[0])
    stage = ha.figure_stage_a(data)
    items = [t.get_text() for t in stage.axes[0].get_yticklabels()]
    assert len(items) == len(books_stage_ids()) and all(items)


def books_stage_ids():
    return load_panel_books(load_release()).stage_a_ids()


def test_load_carries_on_without_a_reporting_file_the_release_refuses(trained, monkeypatch, capsys):
    """Final review M7: a refused validation_reporting.yaml leaves the per-code figures to the pooled summary and the
    bar at Landis & Koch's "substantial" floor, as `summary` carries on; the notebook still opens."""
    from pipeline.errors import ConfigError

    def refused(books):
        raise ConfigError("validation_reporting.yaml: a code the instrument lacks")
    monkeypatch.setattr(rp, "load_reporting", refused)
    run, drop = trained
    data = ha.load(run, drop)
    assert data.reporting is None and data.bar == 0.61 and not data.pooled.get("reporting")
    assert "a code the instrument lacks" in capsys.readouterr().err
    by_code = ha.agreement_by_code(data)
    assert "kind" not in by_code.columns and len(by_code)


def test_load_without_any_sheet_is_refused(tmp_path, books):
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "--video-ids", "vid1"]) == 0
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception):
        ha.load(run, empty)


# ---- the clip session's classes and the pair judgements (plan C, Task 6) ----------------------------------------

def test_the_class_order_names_the_pitch_level_classes_and_no_closing_state():
    assert ha.CLASS_ORDER == ("stage_a", "prep_pres", "venture", "turn_code", "cut", ha.OUTCOME, "pair")
    assert "closing" not in ha.CLASS_ORDER and "closing" not in ha.CLASS_WORDS
    assert all(ha.CLASS_WORDS[cls] for cls in ha.CLASS_ORDER)


def test_human_agreement_class_order_has_no_exchange_topic():
    assert "exchange_topic" not in ha.CLASS_ORDER and "exchange_topic" not in ha.CLASS_WORDS
    # a class outside the order is still named, and sorts after the ordered ones
    assert ha.class_word("exchange_topic") == ha.OTHER_CLASS_WORDS["exchange_topic"]
    assert ha.class_word("outcome") == ha.CLASS_WORDS["outcome"] and ha.class_word("mood") == "mood"
    assert ha.class_rank("pair") < ha.class_rank("exchange_topic") == ha.class_rank("mood") == len(ha.CLASS_ORDER)


def test_a_sheet_with_stray_exchange_rows_loads_and_reports_them_after_the_ordered_classes(tmp_path, books):
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2",
                    "--exchanges-per-video", "2"]) == 0
    tdir, drop = run / "review" / "training", tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                          # noqa: E731
    for coder in ("ana", "ben"):
        _sheets(tdir, drop, coder, lambda r: {"human_value": first(r)})
    data = ha.load(run, drop)
    assert not data.import_failed
    cards = ha.cards(data)
    assert "exchange_topic" in set(cards["class"])                      # the rows are kept, not dropped
    order = list(dict.fromkeys(cards["class"]))
    assert order.index("exchange_topic") > max(order.index(c) for c in order if c in ha.CLASS_ORDER)
    by_class = ha.agreement_by_class(data).set_index("key")
    assert by_class.loc["exchange_topic", "code"] == ha.OTHER_CLASS_WORDS["exchange_topic"]
    assert list(by_class.index).index("exchange_topic") > list(by_class.index).index("turn_code")


def _pairs():
    from pipeline.stage_b import review_clips
    videos = [f"w{i}" for i in range(6)]
    lines = [{"reviewer": c, "batch_id": f"b{i:02d}", "code": code, "left": videos[i % 6],
              "right": videos[(i + 2) % 6], "value": "left" if (c == "ana" or i % 3) else "right",
              "close": i == 0, "model": "left", "gap": 0.1 * i}
             for i in range(12) for c in ("ana", "ben") for code in ("pitch_positivity", "expressed_enthusiasm")]
    return review_clips.pair_agreement(lines, videos, n_boot=99)


def test_the_pair_table_has_one_row_per_dimension_with_each_coders_agreement():
    pairs = _pairs()
    table = ha.pair_agreement_table(pairs)
    assert list(table["dimension"]) == ["pitch_positivity", "expressed_enthusiasm"]
    row = table.iloc[0]
    assert row["pairs"] == 12 and row["ana vs model"] == 1.0 and row["ben vs model"] == pytest.approx(8 / 12)
    assert row["coder vs coder"] == pytest.approx(8 / 12) and row["close ticks in the lowest third"] == 1.0
    assert ha.pair_agreement_table({}).empty


def test_the_pair_figure_draws_agreement_by_score_gap_per_dimension():
    fig = ha.figure_pairs(_pairs())
    assert len(fig.axes) == 2 and fig.axes[0].get_title(loc="left").startswith("A")
    assert ha.figure_pairs({}) is None


# ---- the clip session in training (plan C, Task 7) -------------------------------------------------------------

@pytest.fixture
def clip_trained(tmp_path, books):
    from tests.test_review_training import clip_drop, clip_training, clip_training_run
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    return run, clip_drop(tmp_path, run, books)


def test_load_reads_the_clip_sheet_into_the_cards_and_the_pairs(clip_trained, books):
    run, drop = clip_trained
    before = tree_digest(run)
    data = ha.load(run, drop)
    assert not data.import_failed and data.coders == ["ana", "ben"] and data.videos == ["vid1", "vid2"]
    cards = ha.cards(data)
    assert {"stage_a", "prep_pres", "venture", "cut"} <= set(cards["class"]) and "outcome" not in set(cards["class"])
    # the clip session's classes come first, in the page's order
    order = list(dict.fromkeys(cards["class"]))
    assert order.index("stage_a") < order.index("prep_pres") < order.index("venture") < order.index("cut")
    assert sorted(data.pairs) == sorted(books.stage_a_ids())
    assert set(data.pooled["pairs"]) == set(books.stage_a_ids())
    assert tree_digest(run) == before


def test_the_card_table_shows_the_models_claim_count_as_its_band(clip_trained):
    # final review M4: the model's PRES-EVID count is shown as the band the coders chose from (the fixture's is 5)
    run, drop = clip_trained
    cards = ha.cards(ha.load(run, drop))
    evid = cards[cards["path"] == "prep_pres:PRES-EVID"]
    assert len(evid) and (evid["model"] == "3-5").all()
    others = cards[cards["path"] == "prep_pres:PRES-SUBS"]
    assert others["model"].notna().all() and not others["model"].str.contains("-").any()


def test_the_pair_class_is_named_as_the_coders_page_asks_it():
    # final review I3: the task page asks which pitch comes across as more ..., never which clip shows more
    assert ha.CLASS_WORDS["pair"] == "Which of two pitches comes across as more … (comparisons)"
    assert "clip" not in " ".join(ha.CLASS_WORDS.values()).lower()


def test_the_pairs_table_has_one_row_per_dimension_read_against_one_half(clip_trained, books):
    run, drop = clip_trained
    data = ha.load(run, drop)
    table = ha.pairs_table(data)
    codes = list(books.stage_a_ids())
    assert list(table["dimension"]) == codes
    row = table.set_index("dimension").loc[codes[-1]]
    assert row["pairs"] == 3 and row["coders vs model (pooled)"] == 0.5
    assert row["ana vs model"] == 1.0 and row["ben vs model"] == 0.0 and row["coder vs coder"] == 0.0
    from pipeline.reliability.agreement import binomial_interval, binomial_p_vs_half
    # no target (plan D): k of n with its exact interval and p against 0.5
    assert row["95% interval (Clopper–Pearson)"] == list(binomial_interval(3, 6)) and row["p against 0.5"] == 1.0
    first = table.set_index("dimension").loc[codes[0]]
    assert first["coders vs model (pooled)"] == 1.0 and first["p against 0.5"] == binomial_p_vs_half(6, 6)
    assert not {"target", "reached the target"} & set(table.columns)
    # the figures are the reporting's comparison rows (pooled["reporting"]["pair:<item>"]), k of n beside them
    reported = data.pooled["reporting"][f"pair:{codes[-1]}"]
    assert row["k/n"] == f"{reported['counts']['agree']}/{reported['counts']['n']}" == "3/6"
    assert row["close ticks"] == reported["counts"]["close"]
    assert list(table.columns).index("coders vs model (pooled)") == 3
    # an older pooled summary without the reporting: calibrate's own figures, the same values
    from dataclasses import replace
    older = ha.pairs_table(replace(data, pooled={k: v for k, v in data.pooled.items() if k != "reporting"}))
    assert "k/n" not in older.columns and list(older.columns).index("coders vs model (pooled)") == 3
    assert list(older["p against 0.5"]) == list(table["p against 0.5"])
    assert list(older["coders vs model (pooled)"]) == list(table["coders vs model (pooled)"])
    fig = ha.figure_pairs(data)
    assert len(fig.axes) == len(codes)


def test_without_a_clip_sheet_there_are_no_pairs(trained):
    run, drop = trained
    data = ha.load(run, drop)
    assert data.pairs == {} and ha.pairs_table(data).empty and ha.figure_pairs(data) is None


# ---- the "not sure" tick and a stratified sample (plan C Task 7f) ------------------------------------------------

def test_an_unsure_card_is_shown_as_such_and_the_tables_carry_the_figures_without_it(trained):
    run, drop = trained
    sheet = next(drop.glob("*__ana.csv"))
    rows = read_sheet(sheet)
    code = next(r for r in rows if r["path"].startswith("turn:") and r["human_value"])
    from tests.test_review_cli import write_sheet
    code["note"] = "[unsure] ana's note"
    write_sheet(sheet, rows)
    data = ha.load(run, drop)
    table = ha.cards(data).set_index(["video", "path"])
    row = table.loc[(sheet.name.split("__")[0], code["path"])]
    assert bool(row["ana unsure"]) and row["ana note"] == "ana's note"         # the marker is a flag, not her words
    assert int(table["ana unsure"].sum()) == 1 and not table["ben unsure"].any()
    summary = ha.sheets_summary(data).set_index(["coder", "video"])
    assert summary["not sure"].sum() == 1
    assert data.pooled["without_unsure"]["n_unsure"] == 1
    by_class = ha.agreement_by_class(data)
    assert "coders vs model: α without unsure rows" in by_class.columns
    assert "coders vs model: agreement without unsure rows" in by_class.columns
    assert ha.agreement_by_stratum(data).empty                                 # the training videos are no strata


def test_the_stratum_table_lays_out_each_stratum_and_the_reweighted_figures():
    from types import SimpleNamespace

    from pipeline.stage_b.review_apply import pooled_summary

    def line(path, primary, new):
        return {"path": path, "field_class": "turn_code", "primary": primary, "replicate": primary, "new": new,
                "resolution": "confirm_primary" if new == primary else "correct", "selected_as": "sampled",
                "note": "", "level": "nominal"}
    x, y = "addresses", "partial"
    videos = {"a": [line("turn:3/code:RS-RESP", x, x), line("turn:5/code:RS-RESP", y, y)],
              "b": [line("turn:3/code:RS-RESP", x, y), line("turn:5/code:RS-RESP", y, x)]}
    reporting = rp.load_reporting(load_panel_books(load_release()))
    pooled = pooled_summary(videos, n_boot=99, strata={"stratify": "any_offer", "of": {"a": "yes", "b": "no"},
                                                       "run": {"yes": 3, "no": 1}}, reporting=reporting)
    table = ha.agreement_by_stratum(SimpleNamespace(pooled=pooled, reporting=reporting)).set_index("key")
    row = table.loc["RS-RESP"]
    assert row["with an offer: agreement"] == 1.0 and row["without an offer: agreement"] == 0.0
    assert row["reweighted α"] == pytest.approx(0.5625) and row["unweighted α"] == pytest.approx(0.125)
    assert row["reweighted agreement"] == pytest.approx(0.75)


def test_load_reads_the_runs_strata_when_its_videos_are_in_the_stratified_sample(trained, monkeypatch):
    run, drop = trained
    assert ha.agreement_by_stratum(ha.load(run, drop)).empty                   # no stratified sample in the run
    monkeypatch.setattr(rp, "sample_strata", lambda r: {"stratify": "any_offer", "of": {"vid1": "yes", "vid2": "no"},
                                                        "run": {"yes": 3, "no": 1}})
    table = ha.agreement_by_stratum(ha.load(run, drop))
    assert not table.empty and {"with an offer: α", "without an offer: α", "reweighted α"} <= set(table.columns)
