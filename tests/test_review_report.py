import re

import pytest

from pipeline.stage_b.review_apply import pooled_summary
from pipeline.stage_b.review_report import (
    clock_differences, methods_note, note_of, paper_appendix_markdown, paper_rows, paper_table_markdown,
    strata_markdown, strata_statement, stratum_rows, summary_rows)

# ---- plan D Task 4: the table, the appendix and the methods note, from pooled["reporting"]; no gates ---------------


@pytest.fixture(scope="module")
def reporting():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import load_panel_books
    from pipeline.stage_b.review_report import load_reporting
    release = load_release()
    return load_reporting(load_panel_books(release, sync=False), release)


def _line(path, cls, primary, new, resolution="correct", replicate=None, note="", reviewer="ana"):
    return {"path": path, "field_class": cls, "primary": primary, "replicate": replicate, "new": new,
            "resolution": resolution, "selected_as": "sampled", "note": note, "reviewer": reviewer, "role": "coder"}


def _dominant(note=lambda v, i: ""):
    """RS-RESP over ten videos of ten rows: 90% identical and 90% of all answers `addresses` (Review Focus 1)."""
    videos = {}
    for v in range(10):
        pairs = ([("addresses", "addresses")] * 8 + [("addresses", "partial"), ("partial", "addresses")] if v < 5
                 else [("addresses", "addresses")] * 9 + [("partial", "partial")])
        videos[f"v{v}"] = [_line(f"turn:{i}/code:RS-RESP", "turn_code", model, coder, note=note(v, i))
                           for i, (coder, model) in enumerate(pairs, 1)]
    return videos


def _comparisons():
    from pipeline.stage_b import review_clips
    videos = [f"w{i}" for i in range(6)]
    lines = [{"reviewer": coder, "batch_id": f"b{i:02d}", "code": "pitch_positivity", "left": videos[i % 6],
              "right": videos[(i + 2) % 6], "value": "left" if (i < 8 or coder == "ben") else "right",
              "close": coder == "ana" and i == 0, "model": "left", "gap": 0.1 * i}
             for i in range(12) for coder in ("ana", "ben")]
    return review_clips.pooled_pairs(review_clips.pair_agreement(lines, videos, n_boot=99))


def _every_kind():
    """Six videos carrying a code of every kind: categorical (RS-RESP and the frame), binary (RS-EVID), ordinal
    (PR-SEV), the clock (cut), the ask; with the comparisons of a clip session."""
    videos = {}
    clocks = [("1:00", "1:02"), ("3:00", "2:56"), ("5:00", "5:08"), ("7:30", "7:42"), ("9:10", "8:50"), ("12:00", "12:03")]
    for v in range(6):
        videos[f"v{v}"] = [
            _line("turn:1/code:RS-RESP", "turn_code", "addresses", "addresses" if v else "partial"),
            _line("turn:2/code:RS-RESP", "turn_code", "partial", "partial", replicate="partial"),
            _line("turn:1/code:RS-EVID", "turn_code", "true", "true" if v % 2 else "false", replicate="true"),
            _line("turn:2/code:RS-EVID", "turn_code", "false", "false", replicate="true"),
            _line("turn:1/code:PR-SEV", "turn_code", "2", str(1 + v % 3)),
            _line("turn:1/code:PR-CONCEPT", "turn_code", "execution", "execution" if v else "assets"),
            _line("pitch:first_decision", "cut", clocks[v][0], clocks[v][1]),
            _line("venture:ask", "venture", "200000;10", "201000;10" if v else "200000;12")]
    return videos


def _words(text):
    """The words the table must not use: no row is passed or failed (word boundaries: "Methods" is not "Met")."""
    return [w for w in ("Met", "threshold", "second draw", "on_failure", "Threshold")
            if re.search(rf"\b{w}\b", text, flags=re.IGNORECASE if w != "Met" else 0)]


def _cells(text, field):
    line = next(l for l in text.splitlines() if l.startswith(f"| {field} |"))
    return [c.strip() for c in line.strip().strip("|").split(" | ")]


def test_a_dominant_category_says_alpha_is_low_and_points_to_ac1(reporting):
    """Review Focus 1: 90% `addresses`, 90% identical: alpha near 0 in parentheses, its band named, the prevalence
    flag in words with the AC1 to read instead; AC1 itself in the appendix."""
    pooled = pooled_summary(_dominant(), n_boot=199, reporting=reporting)
    [row] = [r for r in paper_rows(pooled, reporting) if r["field"] == "RS-RESP"]
    f = pooled["reporting"]["RS-RESP"]["figures"]
    assert row["kind"] == "categorical" and row["n"] == 100 and row["videos"] == 10
    from pipeline.reliability.agreement import landis_koch_band
    assert row["alpha"]["value"] == f["alpha"]["value"] < 0.5 < 0.85 < f["ac1"]["value"]
    assert row["band"] == landis_koch_band(f["alpha"]["value"]) == "moderate"
    assert row["flags"] == ["prevalence"]
    text = paper_table_markdown(pooled, reporting)
    cells = _cells(text, "RS-RESP")
    assert cells[1] == "categorical" and cells[2] == "100 (10)"
    a = f["alpha"]
    assert cells[3] == (f"0.90 [{f['exact']['low']:.2f}, {f['exact']['high']:.2f}] "
                        f"(α {a['value']:.2f} [{a['low']:.2f}, {a['high']:.2f}])")
    assert cells[4] == row["band"]
    # the flagged statistic's own band sits in the flag; the Band column stays alpha's
    ac1 = f["ac1"]["value"]
    assert row["read"] == {"statistic": "ac1", "value": ac1, "band": landis_koch_band(ac1)} and ac1 > 0.85
    assert cells[5] == f"prevalence: read AC1 ({ac1:.2f}, {landis_koch_band(ac1)}) rather than α"
    appendix = paper_appendix_markdown(pooled, reporting)
    assert f"Gwet's AC1: {f['ac1']['value']:.2f} [{f['ac1']['low']:.2f}, {f['ac1']['high']:.2f}]" in appendix
    # the agreement on each value: addresses 2 x 85 / (90 + 90), partial 2 x 5 / (10 + 10)
    assert "agreement on each value: addresses 0.94, partial 0.50, deflects —, NONE —" in appendix


def test_a_code_with_ten_units_is_flagged_small_and_still_computed(reporting):
    """Review Focus 4: ten units: the small-n flag, the counts shown, every statistic still computed."""
    videos = {f"v{v}": [_line("turn:1/code:PR-SEV", "turn_code", "2", "2" if v < 7 else "3")] for v in range(10)}
    pooled = pooled_summary(videos, n_boot=199, reporting=reporting)
    row, pair = paper_rows(pooled, reporting)
    assert pair["field"] == "pair" and pair["missing"]
    assert row["field"] == "PR-SEV" and row["n"] == 10 and "small_n" in row["flags"]
    assert row["figures"]["exact"]["value"] == pytest.approx(0.7) and row["figures"]["ac2"]["value"] is not None
    cells = _cells(paper_table_markdown(pooled, reporting), "PR-SEV")
    assert cells[3].startswith("identical 0.70 (7/10) [") and "; within one 1.00 (10/10) [" in cells[3] and "(α " in cells[3]
    # the cell says "small n" and the units; the sentence is the methods note's and the definitions'
    assert cells[5].endswith("small n (10 units)") and "chance-corrected" not in cells[5]
    assert "a chance-corrected statistic on this few units is unstable" in methods_note(pooled, reporting)


def test_a_run_without_comparisons_second_coder_or_strata_still_renders_every_table(reporting):
    """Review Focus 5: one coder, no clip session, no strata: every table renders, its empty cells "—"."""
    pooled = pooled_summary(_dominant(), n_boot=99, reporting=reporting)
    text = paper_table_markdown(pooled, reporting)
    for title in ("## Coder against model", "## Coder against coder", "## Model pass against pass"):
        assert title in text
    coder_coder = text.split("## Coder against coder")[1].split("## Model pass against pass")[0]
    assert _cells(coder_coder, "RS-RESP")[2:] == ["—"] * 5
    assert _cells(text, "pair")[3] == "— (no comparisons imported)"
    assert "By outcome stratum" not in text and stratum_rows(pooled, reporting) == []
    assert strata_markdown(pooled, reporting) == []
    appendix = paper_appendix_markdown(pooled, reporting)
    assert "## Statistics and their definitions" in appendix
    # nothing at all: the tables still render
    empty = pooled_summary({}, reporting=reporting)
    assert "## Coder against model" in paper_table_markdown(empty, reporting)
    assert paper_appendix_markdown(empty, reporting).startswith("# Validation appendix")
    assert paper_rows({}, reporting)[0]["field"] == "pair"            # the comparison row says none were imported


def test_every_kind_is_in_the_table_and_the_methods_note_names_each_kind_present(reporting):
    pooled = pooled_summary(_every_kind(), n_boot=199, pairs=_comparisons(), reporting=reporting)
    rows = paper_rows(pooled, reporting)
    fields = [r["field"] for r in rows]
    # the YAML's order: the frame (derived) first, then the codes; PR-CONCEPT has no row
    assert fields == ["PR-FRAME", "RS-RESP", "RS-EVID", "PR-SEV", "cut", "venture:ask", "pair:pitch_positivity"]
    kinds = {r["field"]: r["kind"] for r in rows}
    assert kinds == {"PR-FRAME": "categorical", "RS-RESP": "categorical", "RS-EVID": "binary", "PR-SEV": "ordinal",
                     "cut": "clock", "venture:ask": "exact", "pair:pitch_positivity": "comparison"}
    text = paper_table_markdown(pooled, reporting, note_of(pooled))
    assert _cells(text, "cut")[3] == "same turn —; median 6.0 s; 6 undetermined"  # no reference turns in this fixture
    assert _cells(text, "cut")[4] == "—" and "α" not in _cells(text, "cut")[3]         # a clock has no alpha
    assert _cells(text, "venture:ask")[3].startswith("0.83 (5/6) [") and "α" not in _cells(text, "venture:ask")[3]
    pair = _cells(text, "pair:pitch_positivity")
    assert pair[3].startswith("20/24 = 0.83 [") and pair[3].endswith("; p = 0.002 against 0.5") and pair[4] == "—"
    # binary: the identical share and (alpha), nominal alpha with its video-resampled interval, banded
    from pipeline.reliability.agreement import krippendorff_alpha, landis_koch_band
    evid = next(r for r in rows if r["field"] == "RS-EVID")
    c = evid["counts"]
    units = [["t", "t"]] * c["a"] + [["t", "f"]] * c["b"] + [["f", "t"]] * c["c"] + [["f", "f"]] * c["d"]
    a = evid["alpha"]
    assert a["value"] == pytest.approx(krippendorff_alpha(units, "nominal")) and a["low"] <= a["value"] <= a["high"]
    assert f"(α {a['value']:.2f} [{a['low']:.2f}, {a['high']:.2f}])" in _cells(text, "RS-EVID")[3]
    assert evid["band"] == landis_koch_band(a["value"]) == _cells(text, "RS-EVID")[4]
    # clock and exact rows say "small n" with no word of a chance-corrected statistic they do not have
    assert _cells(text, "cut")[5] == "small n (6 units)" and _cells(text, "venture:ask")[5] == "small n (6 units)"
    note = methods_note(pooled, reporting, note_of(pooled))
    assert note in text
    for kind in ("categorical", "binary", "ordinal", "clock", "exact", "comparison"):
        assert reporting.reason_of(kind) in note, kind
    assert reporting.bands_reason() in note
    assert reporting.reason_not_reported("PR-CONCEPT") in note
    assert "Raw agreement leads each row" in note and "falls when one value dominates" in note
    assert "prevalence flag marks those rows" in note and "AC1 and AC2" in note and "in the appendix" in note
    assert "resampling whole videos" in note and "against 0.5 (a coin flip is 0.5)" in note
    assert note.count("against 0.5") == 1                                    # said once, in the comparison's reason
    assert "pairs share clips" in note and "optimistic" in note
    assert "taken together" in note and "descriptive names, not calibrated benchmarks" in note
    assert "applied to α, the statistic in parentheses" in note
    # the note describes the table as it is: the chance values are in the appendix, alpha is the statistic in
    # parentheses, AC1 the one a prevalence flag points to
    assert "the appendix gives each beside the value two raters" in note and "printed beside" not in note
    assert "is the chance-corrected figure" not in note and "is the statistic to read" not in note
    assert len(note.split("\n\n")) >= 8                                     # paragraphs, not one block
    assert "primary and replicate" in note and "[unsure]" not in note           # no unsure row in this run
    # a kind that is not present is not described
    only = pooled_summary(_dominant(), n_boot=99, reporting=reporting)
    alone = methods_note(only, reporting, note_of(only))
    assert reporting.reason_of("categorical") in alone and reporting.reason_of("clock") not in alone
    assert reporting.reason_of("comparison") not in alone


def test_the_table_the_appendix_and_the_note_name_no_gate(reporting):
    pooled = pooled_summary(_every_kind(), n_boot=99, pairs=_comparisons(), reporting=reporting)
    for text in (paper_table_markdown(pooled, reporting, note_of(pooled)), paper_appendix_markdown(pooled, reporting),
                 methods_note(pooled, reporting, note_of(pooled))):
        assert _words(text) == []
    assert "| Field | Kind | Units (videos) | Agreement (α) | Band of α | Flags | Without unsure: point (n) |" in (
        paper_table_markdown(pooled, reporting))
    note = methods_note(pooled, reporting, note_of(pooled))
    # the note says which figures carry an interval and which are points, without negated verdict words (I2)
    assert "The Agreement and α cells of the tables carry a 95% interval" in note
    assert "the column without unsure units, the figures per stratum and reweighted" in note
    assert "no target" not in note and "passed, failed" not in note


def test_the_appendix_holds_every_other_figure_and_the_definitions(reporting):
    videos = _every_kind()
    pooled = pooled_summary(videos, n_boot=199, pairs=_comparisons(), reporting=reporting)
    appendix = paper_appendix_markdown(pooled, reporting, clock_differences(videos))
    # the frame's confusion matrix (coder in rows, model in columns), RS-EVID's 2x2, PR-SEV's chance values
    frame = appendix.split("### PR-FRAME")[1].split("###")[0]
    assert "| coder \\ model | promotion | prevention | unclassified | NEUTRAL |" in frame
    evid = appendix.split("### RS-EVID")[1].split("###")[0]
    assert "| coder \\ model | true | false |" in evid and "positive agreement" in evid and "bias index" in evid
    sev = appendix.split("### PR-SEV")[1].split("###")[0]
    for words in ("chance identical share", "chance within-one share", "Gwet's AC2", "mean difference",
                  "Spearman's ρ", "coder's values", "model's values"):
        assert words in sev, words
    cut = appendix.split("### cut")[1].split("###")[0]
    assert "limits of agreement" in cut and "| v0 | ana | 1:02 | 1:00 | 2 |" in cut and "| v4 | ana | 8:50 | 9:10 | -20 |" in cut
    pair = appendix.split("### pair:pitch_positivity")[1].split("###")[0]
    assert "| 1 | 8 | 8 |" in pair and "| 3 | 4 | 8 |" in pair and "close ticks: 1" in pair
    # binary alpha is printed once, as the other kinds' alpha is
    assert evid.count("Krippendorff's α") == 1 and "2×2 table)" not in evid
    definitions = appendix.split("## Statistics and their definitions")[1]
    assert "at least 75% of the two sides' answers taken together" in definitions
    assert "fewer than 20 units" in definitions and "more than 10% of the resamples" in definitions
    for name in ("Identical share", "Within-one share", "Krippendorff's α", "Gwet's AC1", "Gwet's AC2",
                 "Agreement on each value", "Prevalence index", "Bias index", "Limits of agreement",
                 "Clopper–Pearson", "Band", "Prevalence flag", "Small-n flag"):
        assert name in definitions, name
    # pass to pass has its own detail
    assert "## Model pass against pass" in appendix


def test_two_coders_fill_the_coder_against_coder_table(reporting):
    ana = {f"v{v}": [_line(f"turn:{i}/code:RS-RESP", "turn_code", "addresses", "addresses") for i in range(1, 4)]
           for v in range(6)}
    ben = {v: [{**l, "reviewer": "ben", "new": "partial" if l["path"].startswith("turn:1/") else l["new"]}
               for l in lines] for v, lines in ana.items()}
    videos = {v: ana[v] + ben[v] for v in ana}
    pooled = pooled_summary(videos, n_boot=99, by_reviewer={"ana": ana, "ben": ben}, reporting=reporting)
    text = paper_table_markdown(pooled, reporting)
    coder_coder = text.split("## Coder against coder")[1].split("## Model pass against pass")[0]
    assert "ana and ben" in coder_coder and _cells(coder_coder, "RS-RESP")[2] == "18 (6)"
    assert _cells(coder_coder, "RS-RESP")[3].startswith("0.67 (12/18) [")
    assert paper_rows(pooled, reporting, "human_human")[0]["raters"] == ["ana", "ben"]


def test_two_coders_count_each_unit_once_in_the_small_n_flag_and_show_the_judgements(reporting):
    """Final review I1: two coders judge the same 18 question-answer pairs against the model; the row has 18 units
    and 36 judgements, so it is flagged small n (18 < 20) and the units column says both."""
    ana = {f"v{v}": [_line(f"turn:{i}/code:RS-RESP", "turn_code", "addresses", "addresses") for i in range(1, 4)]
           for v in range(6)}
    ben = {v: [{**l, "reviewer": "ben", "new": "partial" if l["path"].startswith("turn:1/") else l["new"]}
               for l in lines] for v, lines in ana.items()}
    videos = {v: ana[v] + ben[v] for v in ana}
    pooled = pooled_summary(videos, n_boot=99, by_reviewer={"ana": ana, "ben": ben}, reporting=reporting)
    row = pooled["reporting"]["RS-RESP"]
    assert (row["n"], row["units"], row["flags"]) == (36, 18, ["prevalence", "small_n"])
    text = paper_table_markdown(pooled, reporting)
    coder_model = text.split("## Coder against model")[1].split("## Coder against coder")[0]
    cells = _cells(coder_model, "RS-RESP")
    assert cells[2] == "18 (6); 36 judgements" and "small n (18 units)" in cells[5]
    # coder against coder judges each unit once: no judgements beside the units
    assert _cells(text.split("## Coder against coder")[1], "RS-RESP")[2] == "18 (6)"
    appendix = paper_appendix_markdown(pooled, reporting).split("## Coder against coder")[0]
    assert "- units: 18 in 6 videos; 36 judgements" in appendix.split("### RS-RESP")[1]
    note = methods_note(pooled, reporting, note_of(pooled))
    assert "counting each pitch, question–answer pair or pair of clips once however many coders judged it" in note


def test_a_comparison_leads_with_the_interval_resampling_videos_and_the_appendix_gives_both(reporting):
    """Final review I3: pairs share clips, so the table's interval is the one resampling videos (summary.md's too);
    the exact Clopper-Pearson one sits beside it in the appendix with the reason they differ."""
    from pipeline.reliability.agreement import binomial_interval
    pairs = _comparisons()
    pooled = pooled_summary(_every_kind(), n_boot=99, pairs=pairs, reporting=reporting)
    low, high = pairs["pitch_positivity"]["ci95"]["coder_model"]
    cell = _cells(paper_table_markdown(pooled, reporting), "pair:pitch_positivity")[3]
    assert cell.startswith(f"20/24 = 0.83 [{low:.2f}, {high:.2f}]; p = ")
    exact = binomial_interval(20, 24)
    detail = paper_appendix_markdown(pooled, reporting).split("### pair:pitch_positivity")[1].split("###")[0]
    assert (f"95% interval resampling videos (the table's): [{low:.2f}, {high:.2f}]; exact (Clopper–Pearson): "
            f"[{exact[0]:.2f}, {exact[1]:.2f}]") in detail
    assert "treat the judgements as independent" in detail
    assert "- pairs of clips: 12 in 6 videos; 24 judgements" in detail


def test_the_comparisons_follow_the_stage_a_items_order(reporting):
    """Final review M9: pair rows in the order of the Stage A rating rows, not by name."""
    one = _comparisons()["pitch_positivity"]
    pairs = {item: one for item in ("expressed_enthusiasm", "interpersonal_warmth", "pitch_positivity")}
    pooled = pooled_summary(_every_kind(), n_boot=99, pairs=pairs, reporting=reporting)
    names = [r["field"] for r in paper_rows(pooled, reporting)]
    items = [c for c in reporting.codes if c in pairs]
    assert [n for n in names if n.startswith("pair:")] == [f"pair:{i}" for i in items]
    assert items[0] == "pitch_positivity"


def test_an_ordinal_row_gives_its_modal_share_and_each_sides(reporting):
    """Final review M5: a prevalence-flagged ordinal or binary row shows the modal share it was flagged on."""
    pooled = pooled_summary(_every_kind(), n_boot=99, reporting=reporting)
    appendix = paper_appendix_markdown(pooled, reporting)
    for code in ("PR-SEV", "RS-EVID"):
        section = appendix.split(f"### {code}")[1].split("###")[0]
        assert "- modal value's share: " in section and "; coder " in section and ", model " in section, code


def test_the_reporting_records_the_sha256_of_its_file(reporting):
    import hashlib
    from pipeline.instrument import load_release
    assert reporting.sha256 == hashlib.sha256(load_release().validation_reporting.read_bytes()).hexdigest()


def test_the_strata_section_reports_each_stratum_and_the_reweighted_figures(reporting):
    unsure = lambda v, i: "[unsure]" if v < 5 and i == 9 else ""      # noqa: E731
    strata = {"stratify": "any_offer", "of": {f"v{v}": "yes" if v < 5 else "no" for v in range(10)},
              "run": {"yes": 10, "no": 30}}
    pooled = pooled_summary(_dominant(unsure), n_boot=99, strata=strata, reporting=reporting)
    [row] = [r for r in stratum_rows(pooled, reporting) if r["field"] == "RS-RESP"]
    assert row["by_stratum"]["yes"]["share"] == pytest.approx(0.8) and row["by_stratum"]["yes"]["n"] == 50
    assert row["by_stratum"]["no"]["share"] == pytest.approx(1.0)
    assert row["reweighted_share"] == pytest.approx(0.95) and row["reweighted_alpha"] is not None
    assert row["unweighted_share"] == pytest.approx(0.9)
    lines = strata_markdown(pooled, reporting)
    text = "\n".join(lines)
    assert "## By outcome stratum" in text and "The main table's figures are unweighted." in text
    # raw agreement leads with α in parentheses, as in the main tables, every figure a point
    assert "with Krippendorff's α in parentheses" in text and "Every figure in this section is a point" in text
    assert "The comparisons are left out" in text and "agreement (α), point" in text
    assert _words(text) == []
    cell = _cells(text, "RS-RESP")[1]
    assert cell.startswith("0.80 (α ") and cell.endswith("), n 50") and "[" not in cell
    table = paper_table_markdown(pooled, reporting, note_of(pooled))
    assert "## By outcome stratum" in table
    # the column without unsure: the leading figure without the five unsure units
    assert _cells(table, "RS-RESP")[6] == f"{90 / 95:.2f} (95)"
    assert "5 of 100 reviewed rows" in table


def test_the_strata_statement_counts_the_videos_imported():
    s = {"stratify": "any_offer", "strata": {"yes": {"run": 3, "planned": 2, "sample": 2, "weight": 1.125},
                                             "no": {"run": 1, "planned": 2, "sample": 1, "weight": 0.75}}}
    text = strata_statement(s)
    assert "2 with an offer, 1 without imported (of 2 and 2 drawn)" in text and "weights 1.12 and 0.75" in text
    assert _words(text) == []


def test_the_note_states_the_corpus_facts(reporting):
    videos = _every_kind()
    pooled = pooled_summary(videos, n_boot=99, reporting=reporting)
    note = note_of(pooled, adjudicator_only=2)
    text = paper_table_markdown(pooled, reporting, note)
    assert "segmentation" in text and "blind" in text
    assert "2 further videos were reviewed by an adjudicator only and are not in this table." in text
    assert "Pooled over 6 videos" in text


def test_a_field_name_with_a_bar_is_escaped(reporting):
    from pipeline.stage_b.review_report import _cell
    assert _cell("a|b") == "a\\|b"


def test_a_yaml_complex_key_is_refused_as_a_config_error(tmp_path):
    import dataclasses
    from pipeline.errors import ConfigError
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import load_panel_books
    from pipeline.stage_b.review_report import load_reporting
    release = load_release()
    path = tmp_path / "validation_reporting.yaml"
    path.write_text("codes:\n  ? [PR-SEV, RS-RESP]\n  : ordinal\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_reporting(load_panel_books(release, sync=False), dataclasses.replace(release, validation_reporting=path))


def test_the_band_floor_and_the_band_of_a_value_come_from_the_reporting(reporting):
    assert reporting.band_floor("substantial") == 0.61 and reporting.band_floor("almost perfect") == 0.81
    assert reporting.band_of(0.61) == "substantial" and reporting.band_of(0.6) == "moderate"
    assert reporting.band_of(None) is None
    with pytest.raises(KeyError):
        reporting.band_floor("good")




# ---- the reporting configuration (plan D): kind per code, statistics and reason per kind --------------------------

def _live():
    from pipeline.instrument import load_release
    from pipeline.stage_b.panel_books import load_panel_books
    release = load_release()
    return release, load_panel_books(release, sync=False)


def _reporting_with(tmp_path, edit):
    """load_reporting of the live instrument with its reporting file edited by `edit` (a function of the dict)."""
    import dataclasses
    import yaml
    from pipeline.stage_b.review_report import load_reporting
    release, books = _live()
    data = yaml.safe_load(release.validation_reporting.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "validation_reporting.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_reporting(books, dataclasses.replace(release, validation_reporting=path))


def test_the_reporting_file_loads_for_the_live_instrument():
    from pipeline.stage_b.review_report import REPORTING_KINDS, load_reporting
    release, books = _live()
    reporting = load_reporting(books, release)
    assert reporting.source == "validation_reporting.yaml" and reporting.release_id == release.release_id
    assert set(reporting.kinds) == set(REPORTING_KINDS)
    assert reporting.kind_of("PR-SEV") == "ordinal" and reporting.values_of("PR-SEV") == ["1", "2", "3"]
    assert reporting.not_reported == ["PR-CONCEPT"] and reporting.kind_of("PR-CONCEPT") is None
    assert "frame" in reporting.reason_not_reported("PR-CONCEPT")
    # the keys the pooled summary uses: a Stage A item and a prepared code with their class, a pair per dimension
    for item in books.stage_a_ids():
        assert reporting.kind_of(f"stage_a:{item}") == "ordinal" and reporting.kind_of(f"pair:{item}") == "comparison"
    assert reporting.values_of("prep_pres:PRES-EVID") == ["0", "1-2", "3-5", "6+"]
    assert reporting.kind_of("prep_pres:PRES-SUBS") == "ordinal" and reporting.kind_of("RS-RESP") == "categorical"
    assert reporting.values_of("RS-RESP") == ["addresses", "partial", "deflects", "NONE"]      # the instrument's
    assert reporting.kind_of("RS-EVID") == "binary" and reporting.positive_of("RS-EVID") == "true"
    assert (reporting.kind_of("cut"), reporting.kind_of("venture:ask"), reporting.kind_of("pair")) == (
        "clock", "exact", "comparison")
    assert reporting.kind_of("exchange_topic") is None and reporting.kind_of("stage_a:nope") is None
    # the frame: frame_of's own values, computed from PR-CONCEPT on both sides
    assert reporting.kind_of("PR-FRAME") == "categorical"
    assert reporting.values_of("PR-FRAME") == ["promotion", "prevention", "unclassified", "NEUTRAL"]
    assert reporting.derived["PR-FRAME"]["from"] == "PR-CONCEPT" and reporting.derived["PR-FRAME"]["via"] == "frame_of"
    assert reporting.flags == {"prevalence": 0.75, "small_n": 20}
    assert reporting.bands == "landis_koch" and reporting.comparisons == {"against": 0.5}
    for kind in REPORTING_KINDS:
        assert reporting.reason_of(kind).strip() and reporting.statistics_of(kind)
    assert "AC1 (Gwet 2008)" in reporting.reason_of("categorical") and "Bland & Altman" in reporting.reason_of("clock")
    assert "Spearman" in reporting.reason_of("ordinal") and "(Gwet 2014)" in reporting.reason_of("ordinal")
    # the comparison's level comes from comparisons.against, never from the reason's own text
    assert "against 0.5 (a coin flip is 0.5)" in reporting.reason_of("comparison")
    assert "{against}" in reporting.kinds["comparison"]["reason"] and "{" not in reporting.reason_of("comparison")
    assert reporting.bands_reason().startswith("Bands follow Landis & Koch (1977)") and "0.61–0.80 substantial" in (
        reporting.bands_reason())
    # an old sheet's outcome rows still report
    assert reporting.kind_of("outcome") == "categorical" and reporting.values_of("outcome") == ["yes", "no", "unknown"]
    assert not any(word in reporting.reason_of(kind).lower() for kind in REPORTING_KINDS
                   for word in ("threshold", "met ", "second draw"))


@pytest.mark.parametrize("edit, message", [
    (lambda d: d["codes"].update({"PR-SEV": {"kind": "interval", "values": ["1", "2", "3"]}}), "kind 'interval'"),
    (lambda d: d["kinds"].update({"interval": {"statistics": ["alpha"], "reason": "x"}}), "kind 'interval'"),
    (lambda d: d["codes"].update({"PR-NOPE": "categorical"}), "PR-NOPE"),
    (lambda d: d["codes"].update({"stage_a:pitch_positivity": "ordinal"}), "stage_a:pitch_positivity"),
    (lambda d: d["codes"].update({"venture:nope": "exact"}), "venture:nope"),
    (lambda d: d["codes"].update({"PR-SEV": "ordinal"}), "PR-SEV"),
    (lambda d: d["codes"]["PR-SEV"].pop("values"), "values"),
    (lambda d: d["codes"].update({"RS-EVID": "binary"}), "positive"),
    (lambda d: d["codes"]["RS-EVID"].update(positive="yes"), "positive"),
    (lambda d: d["codes"]["PR-SEV"].update(values=["1", "2", "3", "4", "5"]), "PR-SEV"),
    (lambda d: d["codes"]["PRES-EVID"].update(values=["0", "1", "2+"]), "PRES-EVID"),
    (lambda d: d["codes"].update({"RS-RESP": "binary"}), "RS-RESP"),
    (lambda d: d["kinds"].pop("clock"), "clock"),
    (lambda d: d["kinds"]["clock"].update(reason=" "), "reason"),
    (lambda d: d["kinds"]["ordinal"].update(statistics=["alpha", "kappa"]), "kappa"),
    (lambda d: d["derived"]["PR-FRAME"].update(values=["promotion", "prevention", "neither", "NEUTRAL"]), "neither"),
    (lambda d: d["derived"]["PR-FRAME"].update(via="topic_of"), "topic_of"),
    (lambda d: d["derived"]["PR-FRAME"].update({"from": "PR-NOPE"}), "PR-NOPE"),
    (lambda d: d["derived"].update({"PR-SEV": d["derived"]["PR-FRAME"]}), "PR-SEV"),
    (lambda d: d["not_reported"].update({"RS-RESP": "x"}), "RS-RESP"),
    (lambda d: d["not_reported"].update({"PR-NOPE": "x"}), "PR-NOPE"),
    (lambda d: d["flags"].update(prevalence=1.5), "prevalence"),
    (lambda d: d["flags"].update(small_n=0), "small_n"),
    (lambda d: d["flags"].update(other=1), "other"),
    (lambda d: d.update(bands="cicchetti"), "bands"),
    (lambda d: d["comparisons"].update(against=float("nan")), "against"),
    (lambda d: d.update(gates={}), "gates"),
    # not a string where a name is asked for: refused, not a TypeError
    (lambda d: d["codes"].update({"PR-SEV": {"kind": ["ordinal"], "values": ["1", "2", "3"]}}), "kind ['ordinal']"),
    (lambda d: d["derived"]["PR-FRAME"].update(via=["frame_of"]), "via ['frame_of']"),
    (lambda d: d.update(bands=["landis_koch"]), "bands"),
    (lambda d: d["codes"]["PR-SEV"].update(values=["1", "2", "2", "3"]), "repeat"),
    (lambda d: d["codes"].update({"RS-RESP": {"kind": "categorical", "values": ["addresses", "partial", "deflects",
                                                                                "NONE", "NONE"]}}), "repeat"),
    (lambda d: d["kinds"]["comparison"].update(reason="Agreement is reported against 0.5."), "{against}"),
])
def test_a_reporting_file_the_table_cannot_apply_is_refused(tmp_path, edit, message):
    from pipeline.errors import ConfigError
    with pytest.raises(ConfigError) as exc:
        _reporting_with(tmp_path, edit)
    assert message in str(exc.value)


def test_against_is_read_into_the_comparison_reason_and_only_a_coin_flip_is_accepted(tmp_path, reporting):
    assert "against 0.5 (a coin flip is 0.5)" in reporting.reason_of("comparison")
    # the comparison's p is p_vs_half: another level is refused when the file is read, not by a traceback later (M6)
    from pipeline.errors import ConfigError
    with pytest.raises(ConfigError, match="0.5 only"):
        _reporting_with(tmp_path, lambda d: d["comparisons"].update(against=0.6))


def test_every_code_a_coder_sheet_draws_is_classified():
    from pipeline.stage_b.review_clips import MODEL_ONLY_CODES
    from pipeline.stage_b.review_report import load_reporting
    release, books = _live()
    reporting = load_reporting(books, release)
    drawn = ["PR-SEV", "RS-RESP", "RS-EVID", "PREP-SUBS", "PREP-COH", "cut", "venture:ask", "pair",
             *(f"prep_pres:{e.id}" for e in books.prepared_codes() if e.id not in MODEL_ONLY_CODES),
             *(f"stage_a:{i}" for i in books.stage_a_ids()), *(f"pair:{i}" for i in books.stage_a_ids())]
    assert [v for v in drawn if reporting.kind_of(v) is None] == []
    assert reporting.kind_of("PR-FRAME") and reporting.not_reported == ["PR-CONCEPT"]


@pytest.mark.parametrize("text, message", [
    ("kinds: [unclosed\n", "not valid YAML"),
    ("codes:\n  PR-SEV: ordinal\n  PR-SEV: categorical\n", "written twice"),
])
def test_a_reporting_file_that_is_not_one_yaml_mapping_is_refused(tmp_path, text, message):
    import dataclasses
    from pipeline.errors import ConfigError
    from pipeline.stage_b.review_report import load_reporting
    release, books = _live()
    path = tmp_path / "validation_reporting.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_reporting(books, dataclasses.replace(release, validation_reporting=path))


def test_a_release_without_a_reporting_file_is_refused():
    import dataclasses
    from pipeline.errors import ConfigError
    from pipeline.stage_b.review_report import load_reporting
    release, books = _live()
    with pytest.raises(ConfigError, match="validation_reporting.yaml"):
        load_reporting(books, dataclasses.replace(release, validation_reporting=None))


# ---- the paper's summary table (2026-10-31): AC1 or AC2 with alpha in parentheses, Gemini vs human and human vs human

SUMMARY_HEAD = "| Code | Scale | Gemini vs. human: AC1/AC2 (α) | Human 1 vs. human 2: AC1/AC2 (α) |"


def _summary(text):
    return text.split("## Agreement by code")[1].split("## Coder against model")[0]


def test_the_summary_table_leads_with_ac1_or_ac2_and_alpha_in_parentheses(reporting):
    pooled = pooled_summary(_every_kind(), n_boot=199, pairs=_comparisons(), reporting=reporting)
    text = paper_table_markdown(pooled, reporting, note_of(pooled))
    assert text.index("## Agreement by code") < text.index("## Coder against model")
    summary = _summary(text)
    assert SUMMARY_HEAD in summary
    figures = {r["field"]: r for r in paper_rows(pooled, reporting)}
    rows = {r["field"]: r for r in summary_rows(pooled, reporting)}
    assert list(rows) == list(figures)                       # the same rows, in the same order
    resp = figures["RS-RESP"]
    ac1, alpha = resp["figures"]["ac1"]["value"], resp["alpha"]["value"]
    assert rows["RS-RESP"]["gemini_human"] == f"AC1 {ac1:.2f} (α {alpha:.2f}) †"   # 12 units: small n
    sev = figures["PR-SEV"]
    assert rows["PR-SEV"]["gemini_human"] == f"AC2 {sev['figures']['ac2']['value']:.2f} (α {sev['alpha']['value']:.2f}) †"
    assert rows["RS-EVID"]["gemini_human"].startswith("AC1 ")
    assert rows["PR-SEV"]["scale"] == "1–3" and rows["RS-EVID"]["scale"] == "yes / no"
    assert rows["PR-FRAME"]["code"] == "Question frame (PR-FRAME)"
    # the kinds without a chance-corrected statistic give their own figure, with no α
    assert rows["cut"]["gemini_human"] == "same turn —; no α †"
    assert rows["venture:ask"]["gemini_human"] == "identical 0.83 (5/6); no α †"
    assert rows["pair:pitch_positivity"]["gemini_human"] == "0.83 (20/24) against 0.5, p = 0.002 †"  # 12 pairs
    # one coder on the sheets (the clip session has two): only the comparison fills the human-human column
    assert {r["human_human"] for f, r in rows.items() if not f.startswith("pair:")} == {"—"}
    assert rows["pair:pitch_positivity"]["human_human"].startswith("0.67 (8/12) against 0.5")
    line = next(l for l in summary.splitlines() if l.startswith("| Answer addresses the question (RS-RESP) |"))
    assert line == f"| Answer addresses the question (RS-RESP) | addresses / partial / deflects | AC1 {ac1:.2f} (α {alpha:.2f}) † | — |"
    assert "Gwet's AC1" in summary and "AC2 with quadratic weights" in summary and "† fewer than 20 units" in summary
    assert not _words(summary)


def test_the_summary_table_fills_human_1_vs_human_2_from_the_two_coders(reporting):
    ana = {f"v{v}": [_line(f"turn:{i}/code:RS-RESP", "turn_code", "addresses", "addresses") for i in range(1, 4)]
           for v in range(6)}
    ben = {v: [{**l, "reviewer": "ben", "new": "partial" if l["path"].startswith("turn:1/") else l["new"]}
               for l in lines] for v, lines in ana.items()}
    videos = {v: ana[v] + ben[v] for v in ana}
    pooled = pooled_summary(videos, n_boot=99, by_reviewer={"ana": ana, "ben": ben}, reporting=reporting)
    hh = next(r for r in paper_rows(pooled, reporting, "human_human") if r["field"] == "RS-RESP")
    row = next(r for r in summary_rows(pooled, reporting) if r["field"] == "RS-RESP")
    assert row["human_human"] == f"AC1 {hh['figures']['ac1']['value']:.2f} (α {hh['alpha']['value']:.2f}) †"
    assert "ana and ben" in _summary(paper_table_markdown(pooled, reporting))
