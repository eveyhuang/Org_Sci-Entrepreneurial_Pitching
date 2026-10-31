import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import compare_panel_vs_structured as cmp  # noqa: E402
from tests.panel_fixtures import DURATION_S, synthetic_record, turn  # noqa: E402

VID = "v1"


def wj(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def build_new(root, vid=VID, replicate=True, calls=True):
    d = root / vid
    wj(d / "experiment.json", {"video_id": vid, "duration_s": DURATION_S})
    for name in (("primary", "replicate") if replicate else ("primary",)):
        wj(d / "results" / f"{name}.json", {"pass": name, "record": synthetic_record()})
    wj(d / "agreement.json", {
        "turns": {"speaker_time_agreement": 0.85, "sequence_agreement": 0.75, "recall": 0.9, "precision": 0.8,
                  "recall_any_function": 0.95, "precision_any_function": 0.9},
        "exchanges": {"n_primary": 4, "n_replicate": 5, "matched_at_half": 3, "mean_jaccard": 0.7,
                      "co_membership_agreement": 0.88, "adjusted_co_membership": 0.61, "topic_agree": 1, "topic_n": 2,
                      "closing": {"n": 3, "agree": 2, "confusion": {}}},
        "time": {"matched_turn_pairs": 7, "median_start_diff_s": -1.5, "median_abs_start_diff_s": 2.5,
                 "max_abs_start_diff_s": 41.0, "turn_pairs_beyond_tolerance": 1, "decision_pairs_beyond_tolerance": 0},
        "codes": {"RS-RESP": {"both_valued": 3, "exact": 2}},
        "decisions": {"ever_offer_agreement": 1.0, "offer_membership_agreement": 0.5},
        "stage_a": {"_summary": {"both_rated": 10, "exact": 9, "within_one": 10, "mean_abs_diff": 0.1}},
    })
    if calls:
        lines = [{"ok": True, "input_tokens": 1000, "output_tokens": 200, "thinking_tokens": 50},
                 {"ok": True, "input_tokens": 1000, "output_tokens": 300, "thinking_tokens": 70}]
        (d / "calls.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return d


def old_code(value, uncodable=False):
    return {"value": value, "uncodable": uncodable, "evidence": "x", "turn_refs": [1]}


def build_old(root, vid=VID, prepared=True):
    d = root / vid
    wj(d / "boundaries.json", {
        "prepared_start_s": 5.0, "prepared_end_s": 60.0, "cutoff_s": 111.0,
        "outcomes_model_reviewed_not_gold": {"sharks": [
            {"speaker": "speaker-1", "ever_offer": True,
             "events": [{"kind": "offer", "source_s": 155.0, "evidence": "e"}]},
            {"speaker": "speaker-2", "ever_offer": False,
             "events": [{"kind": "out", "source_s": 105.0, "evidence": "e"},
                        {"kind": "withdraws_offer", "source_s": 300.0, "evidence": "e"}]},
        ]}})
    codes = [
        {"PR-FRAME": old_code("promotion"), "PR-SEV": old_code(1), "RS-RESP": old_code("addresses"),
         "RS-FRAME": old_code("NEUTRAL"), "RS-EVID": old_code(True), "PREP-SUBS": old_code(None, True)},
        {"PR-FRAME": old_code("prevention"), "PR-SEV": old_code(3), "RS-RESP": old_code("addresses"),
         "RS-EVID": old_code(False)},
        {"PR-FRAME": old_code("promotion"), "RS-RESP": old_code("evades")},
    ]
    for i, rec in enumerate(codes, 1):
        wj(d / "results" / f"accepted_ep_{i}.json", {"record": rec})
    wj(d / "results" / "accepted_pitch.json", {"value": 0})
    wj(d / "results" / "episode_ep_1_try1.json", {"record": {}})
    if prepared:
        wj(d / "results" / "accepted_prepared.json", {
            "pitch_positivity": {"value": 4, "uncodable": False, "evidence": "e"},
            "expressed_enthusiasm": {"value": 5, "uncodable": False, "evidence": "e"},
            "interpersonal_warmth": {"value": None, "uncodable": True, "evidence": "e"},
            "overall_notes": "text"})
    lines = [{"key": "a", "input_tokens": 100, "output_tokens": 10, "thinking_tokens": 5},
             {"key": "b", "input_tokens": 200, "output_tokens": 20, "thinking_tokens": 7},
             {"key": "c", "input_tokens": 300, "output_tokens": 30, "thinking_tokens": None}]
    (d / "calls.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return d


def _edit_primary(new, fn):
    path = new / "results" / "primary.json"
    data = json.loads(path.read_text())
    fn(data["record"])
    path.write_text(json.dumps(data))


@pytest.fixture
def pair(tmp_path):
    return build_new(tmp_path / "new"), build_old(tmp_path / "old"), tmp_path


def test_coverage_after_cutoff(pair):
    new, old, _ = pair
    c = cmp.compare_video(new, old)["coverage"]
    assert c["old_cutoff_s"] == 111.0
    assert c["new_last_turn_end_s"] == 172.0
    assert c["new_turns"] == 19 and c["new_turns_after_cutoff"] == 10       # the exit turn starts at the cutoff
    assert c["new_questions"] == 4 and c["new_questions_after_cutoff"] == 2
    assert c["new_answers"] == 4 and c["new_answers_after_cutoff"] == 2
    assert c["old_episodes"] == 3
    for gone in ("new_events", "new_events_after_cutoff", "new_cross_investor_links", "new_delayed_links",
                 "new_kind_counts"):
        assert gone not in c


def test_exchange_coverage(pair):
    new, old, _ = pair
    c = cmp.compare_video(new, old)["coverage"]
    assert c["new_exchanges"] == {"negotiation": 1, "social": 1, "substantive": 2}
    assert c["new_substantive_exchanges_after_cutoff"] == 1     # repeat purchases; acquisition cost began before
    assert c["new_multi_investor_exchanges"] == 1               # acquisition cost: two investors
    assert c["new_reopened_exchanges"] == 1                     # acquisition cost, left and returned to
    assert c["new_social_exchanges"] == 1
    assert c["new_demonstration_seconds"] == 0.0
    assert c["new_closing_counts"] == {}                         # no closing state since v3-dev9
    assert c["new_topic_counts"] == {"customer_retention": 1, "operating_efficiency": 1}
    assert c["new_function_counts"] == {
        "accepts": 2, "answer": 4, "challenges": 1, "decides": 4, "presents": 1, "question": 4, "repair": 1,
        "social": 2}
    assert c["new_unlinked_responses"] == 0


def test_substantive_exchange_starting_exactly_at_the_cutoff_counts(tmp_path):
    new, old = build_new(tmp_path / "n"), build_old(tmp_path / "o")
    path = old / "boundaries.json"
    data = json.loads(path.read_text())
    for cutoff, expected in ((115.0, 1), (115.5, 0), (62.0, 2)):
        data["cutoff_s"] = cutoff
        path.write_text(json.dumps(data))
        assert cmp.compare_video(new, old)["coverage"]["new_substantive_exchanges_after_cutoff"] == expected, cutoff


def test_reopened_needs_a_turn_of_another_exchange_in_between(tmp_path):
    new = build_new(tmp_path / "n")

    def contiguous(rec):
        # the return to acquisition cost becomes its own exchange: nothing is reopened any more
        for t in rec["turns"]:
            if t["turn_id"] in (14, 15, 16):
                t["exchange_id"] = 5
        rec["exchanges"].append({"exchange_id": 5, "kind": "substantive", "topic": "operating_efficiency", "label": ""})
    _edit_primary(new, contiguous)
    c = cmp.compare_video(new, build_old(tmp_path / "o"))["coverage"]
    assert c["new_reopened_exchanges"] == 0 and c["new_exchanges"]["substantive"] == 3
    assert c["new_multi_investor_exchanges"] == 1               # S1 and S2 still both speak in exchange 1
    assert c["new_closing_counts"] == {}
    assert c["new_topic_counts"] == {"customer_retention": 1, "operating_efficiency": 2}

    def only_an_exit_between(rec):
        # exchange 3 now straddles a turn that belongs to no exchange: that is not a reopening
        exit_turn = next(t for t in rec["turns"] if t["turn_id"] == 10)
        rec["turns"].remove(exit_turn)
        rec["turns"].insert(next(i for i, t in enumerate(rec["turns"]) if t["turn_id"] == 12), exit_turn)
    _edit_primary(new, only_an_exit_between)
    assert cmp.compare_video(new, build_old(tmp_path / "o"))["coverage"]["new_reopened_exchanges"] == 0


def test_demonstration_seconds_and_a_reopened_social_exchange(tmp_path):
    new = build_new(tmp_path / "n")

    def mut(rec):
        rec["exchanges"][1]["kind"] = "demonstration"            # turns 8 (3 s) and 9 (2 s)
        rec["turns"].insert(13, turn(20, "S3", 133.0, 133.5, "demonstrates", "Passes the lid along.", ["S2"], 2))
    _edit_primary(new, mut)
    c = cmp.compare_video(new, build_old(tmp_path / "o"))["coverage"]
    assert c["new_demonstration_seconds"] == 5.5
    assert c["new_social_exchanges"] == 0 and c["new_exchanges"] == {"demonstration": 1, "negotiation": 1, "substantive": 2}
    assert c["new_reopened_exchanges"] == 1                      # only substantive exchanges are counted


def test_unparseable_file_warns_and_is_treated_as_missing(tmp_path, capsys):
    new = build_new(tmp_path / "n")
    (new / "agreement.json").write_text("{not json")
    (new / "calls.jsonl").write_text('{"input_tokens": 5}\nnot json\n')
    r = cmp.compare_video(new, build_old(tmp_path / "o"))
    assert r["self_consistency"] is None
    assert r["cost"]["new_calls"] == 1 and r["cost"]["new_input_tokens"] == 5
    err = capsys.readouterr().err
    assert "agreement.json" in err and "calls.jsonl" in err


def test_out_dir_inside_old_run_refused(tmp_path):
    build_new(tmp_path / "new")
    build_old(tmp_path / "old")
    for out in (tmp_path / "old", tmp_path / "old" / "sub"):
        with pytest.raises(SystemExit) as e:
            cmp.main(["--new-run", str(tmp_path / "new"), "--old-run", str(tmp_path / "old"), "--out-dir", str(out)])
        assert e.value.code != 0
    assert not (tmp_path / "old" / "comparison_vs_structured.md").exists()
    with pytest.raises(SystemExit):
        cmp.main(["--new-run", str(tmp_path / "old"), "--old-run", str(tmp_path / "old")])


def test_markdown_opener_and_labels(pair):
    new, old, tmp = pair
    r = cmp.compare_video(new, old)
    md = cmp.render_markdown({"videos": {VID: r}})
    assert "two full-video passes per video plus any retries" in md
    assert "two dozen" not in md
    assert "new_turns" not in md and "New turns" in md
    assert "the new units are turns and the exchanges that group them" in md


def test_decisions(pair):
    new, old, _ = pair
    d = cmp.compare_video(new, old)["decisions"]
    assert d["old_n_investors"] == 2 and d["old_n_ever_offer"] == 1
    assert d["new_n_investors"] == 3 and d["new_n_ever_offer"] == 2
    assert d["any_offer_old"] is True and d["any_offer_new"] is True and d["any_offer_agree"] is True
    # old: offer@155 (new offer@159), out@105 (new exit@111 is 6 s away), withdraws@300 (none)
    assert d["matched_events"] == {"old": 3, "matched": 2, "new": 3}


def test_prepared_iou(pair):
    new, old, _ = pair
    p = cmp.compare_video(new, old)["prepared"]
    assert p["old"] == [5.0, 60.0]
    assert p["new"] == [[5.0, 60.0]]
    assert p["iou"] == pytest.approx(1.0)


def test_code_distributions_split_at_cutoff(pair):
    new, old, _ = pair
    codes = cmp.compare_video(new, old)["codes"]
    assert codes["RS-RESP"]["old"] == {"addresses": 2, "evades": 1}
    assert codes["RS-RESP"]["new_before_cutoff"] == {"addresses": 2}
    assert codes["RS-RESP"]["new_after_cutoff"] == {"addresses": 2}
    assert codes["PR-SEV"]["old"] == {"1": 1, "3": 1}
    assert codes["PR-SEV"]["new_before_cutoff"] == {"1": 1, "3": 1}
    assert codes["PR-SEV"]["new_after_cutoff"] == {"2": 2}
    assert codes["RS-EVID"]["old"] == {"true": 1, "false": 1}
    assert codes["RS-EVID"]["new_before_cutoff"] == {"true": 2}
    assert codes["RS-EVID"]["new_after_cutoff"] == {"true": 1, "false": 1}
    assert codes["RS-UNCERT"]["new_after_cutoff"] == {"false": 2}
    assert codes["PREP-SUBS"]["old"] == {"∅": 1}
    assert codes["PREP-CTX"]["new_before_cutoff"] == {"∅": 2}
    assert codes["RS-FRAME"]["old"] == {"NEUTRAL": 1}
    assert codes["PR-FRAME"]["old"] == {"promotion": 2, "prevention": 1}
    # the new run records concepts; its frames are derived from them
    assert codes["PR-FRAME"]["new_before_cutoff"] == {"prevention": 2}
    assert codes["PR-FRAME"]["new_after_cutoff"] == {"prevention": 2}
    assert codes["RS-FRAME"]["new_before_cutoff"] == {"prevention": 2}
    assert codes["RS-FRAME"]["new_after_cutoff"] == {"prevention": 1, "promotion": 1}
    assert "PR-CONCEPT" not in codes and "RS-CONCEPT" not in codes


def test_stage_a(pair):
    new, old, _ = pair
    s = cmp.compare_video(new, old)["stage_a"]
    assert s["old"]["pitch_positivity"] == {"value": 4, "uncodable": False}
    assert s["old"]["interpersonal_warmth"]["uncodable"] is True
    assert "overall_notes" not in s["old"]
    assert len(s["new"]) == 3
    assert s["new"]["expressed_enthusiasm"] == {"primary": 3, "replicate": 3}


def test_nested_prepared_record_accepted(pair):
    new, old, _ = pair
    path = old / "results" / "accepted_prepared.json"
    data = json.loads(path.read_text())
    path.write_text(json.dumps({"record": data}))
    assert cmp.compare_video(new, old)["stage_a"]["old"]["pitch_positivity"]["value"] == 4


def test_missing_prepared_gives_none(tmp_path):
    new, old = build_new(tmp_path / "n"), build_old(tmp_path / "o", prepared=False)
    assert cmp.compare_video(new, old)["stage_a"]["old"] is None


def test_cost_sums(pair):
    new, old, _ = pair
    c = cmp.compare_video(new, old)["cost"]
    assert (c["old_calls"], c["old_input_tokens"], c["old_output_tokens"], c["old_thinking_tokens"]) == (3, 600, 60, 12)
    assert (c["new_calls"], c["new_input_tokens"], c["new_output_tokens"], c["new_thinking_tokens"]) == (2, 2000, 500, 120)


def test_self_consistency(pair):
    new, old, _ = pair
    s = cmp.compare_video(new, old)["self_consistency"]
    assert s["turn_recall"] == 0.9 and s["turn_precision"] == 0.8
    assert "event_recall" not in s
    assert s["ever_offer_agreement"] == 1.0
    assert s["stage_a_summary"]["exact"] == 9
    assert s["codes"]["RS-RESP"] == {"exact": 2, "both_valued": 3}


def test_missing_files_report_null(tmp_path):
    new = build_new(tmp_path / "n", replicate=False, calls=False)
    (new / "agreement.json").unlink()
    old = build_old(tmp_path / "o")
    (old / "calls.jsonl").unlink()
    (old / "boundaries.json").unlink()
    r = cmp.compare_video(new, old)
    assert r["cost"]["new_calls"] is None and r["cost"]["old_calls"] is None
    assert r["self_consistency"] is None
    assert r["coverage"]["old_cutoff_s"] is None
    assert r["coverage"]["new_turns_after_cutoff"] is None
    assert r["coverage"]["new_substantive_exchanges_after_cutoff"] is None
    assert r["coverage"]["new_exchanges"] == {"negotiation": 1, "social": 1, "substantive": 2}
    assert r["decisions"]["old_n_investors"] is None
    assert r["decisions"]["any_offer_agree"] is None
    assert r["stage_a"]["new"]["expressed_enthusiasm"]["replicate"] is None
    cmp.render_markdown({"runs": {}, "videos": {VID: r}})


def test_missing_primary_gives_null_new(tmp_path):
    new = build_new(tmp_path / "n")
    (new / "results" / "primary.json").unlink()
    wj(new / "results" / "primary_failed.json", {"error": "x"})
    r = cmp.compare_video(new, build_old(tmp_path / "o"))
    assert r["coverage"]["new_turns"] is None and r["coverage"]["new_exchanges"] is None
    assert r["coverage"]["new_closing_counts"] is None
    assert "None" not in cmp.render_markdown({"videos": {VID: r}})
    assert r["decisions"]["new_n_investors"] is None
    assert r["stage_a"]["new"] is None
    assert r["coverage"]["old_episodes"] == 3


def test_main_writes_reports(tmp_path):
    build_new(tmp_path / "new", "a")
    build_new(tmp_path / "new", "b")
    build_old(tmp_path / "old", "a")
    build_old(tmp_path / "old", "c")
    out = tmp_path / "out"
    cmp.main(["--new-run", str(tmp_path / "new"), "--old-run", str(tmp_path / "old"), "--out-dir", str(out)])
    data = json.loads((out / "comparison_vs_structured.json").read_text())
    assert list(data["videos"]) == ["a"]
    md = (out / "comparison_vs_structured.md").read_text()
    assert "not an accuracy test" in md
    assert "not human-validated" in md
    assert "| a |" in md and "## a" in md
    assert "None" not in md


def test_main_explicit_video_ids_and_default_out_dir(tmp_path):
    build_new(tmp_path / "new", "a")
    build_old(tmp_path / "old", "a")
    cmp.main(["--new-run", str(tmp_path / "new"), "--old-run", str(tmp_path / "old"), "--video-ids", "a"])
    assert (tmp_path / "new" / "comparison_vs_structured.md").exists()


def test_coverage_counts_functions_and_unlinked_responses(tmp_path):
    new = build_new(tmp_path / "n")

    def mut(rec):
        for t in rec["turns"]:
            if t["turn_id"] in (6, 7):
                t["responds_to"] = []
        rec["turns"][2]["responds_to"] = []          # an unlinked answer is an error elsewhere, not counted here
    _edit_primary(new, mut)
    r = cmp.compare_video(new, build_old(tmp_path / "o"))
    assert r["coverage"]["new_unlinked_responses"] == 2
    md = cmp.render_markdown({"videos": {VID: r}})
    assert "| New unlinked responses | 2 |" in md
    assert ("| New turns by function | accepts: 2, answer: 4, challenges: 1, decides: 4, presents: 1, question: 4, "
            "repair: 1, social: 2 |") in md
    assert "| New exchanges by kind | negotiation: 1, social: 1, substantive: 2 |" in md
    assert "| New substantive exchanges by topic | customer_retention: 1, operating_efficiency: 1 |" in md
    assert "| New reopened substantive exchanges | 1 |" in md
    assert "| New substantive exchanges with two or more investors | 1 |" in md
    assert "| New substantive exchanges starting at or after the cutoff | 1 |" in md
    assert "| New demonstration seconds | 0 |" in md
    for raw in ("new_function_counts", "new_unlinked_responses", "new_exchanges", "new_closing_counts"):
        assert raw not in md, raw


def test_code_values_differing_only_in_case_are_counted_under_the_codebook_spelling(tmp_path):
    new = build_new(tmp_path / "n")

    def mut(rec):
        for t in rec["turns"]:
            for cv in t["codes"]:
                if cv["code"] == "RS-CONCEPT" and t["turn_id"] in (3, 12):
                    cv["value"] = cv["value"].upper()
                if cv["code"] == "RS-CONCEPT" and t["turn_id"] == 5:
                    cv.update(value=None, missing="ambiguous")
                if cv["code"] == "PR-CONCEPT" and t["turn_id"] == 14:
                    cv["value"] = "none"
                if cv["code"] == "RS-EVID" and t["turn_id"] == 3:
                    cv["value"] = "True"
                if cv["code"] == "RS-RESP" and t["turn_id"] == 3:
                    cv["value"] = "sideways"                         # not a codebook value: left as written
    _edit_primary(new, mut)
    codes = cmp.compare_video(new, build_old(tmp_path / "o"))["codes"]
    assert codes["RS-FRAME"]["new_before_cutoff"] == {"prevention": 1, "∅": 1}
    assert codes["RS-FRAME"]["new_after_cutoff"] == {"prevention": 1, "promotion": 1}
    assert codes["PR-FRAME"]["new_after_cutoff"] == {"prevention": 1, "NEUTRAL": 1}
    r = cmp.compare_video(new, build_old(tmp_path / "o"))
    assert r["coverage"]["new_concept_counts"] == {
        "question": {"NONE": 1, "customer_retention": 1, "operating_efficiency": 2},
        "answer": {"customer_retention": 1, "forecast": 1, "operating_efficiency": 1, "∅": 1}}
    assert codes["RS-EVID"]["new_before_cutoff"] == {"true": 2}
    assert codes["RS-RESP"]["new_before_cutoff"] == {"sideways": 1, "addresses": 1}


def test_self_consistency_reports_turn_and_exchange_measures(pair):
    new, old, _ = pair
    r = cmp.compare_video(new, old)
    s = r["self_consistency"]
    assert s["turn_recall_any_function"] == 0.95 and s["turn_precision_any_function"] == 0.9
    assert s["offer_membership_agreement"] == 0.5
    assert s["speaker_time_agreement"] == 0.85
    assert (s["exchanges_primary"], s["exchanges_replicate"], s["exchanges_matched_at_half"]) == (4, 5, 3)
    assert s["exchange_mean_jaccard"] == 0.7 and s["co_membership_agreement"] == 0.88
    assert s["adjusted_co_membership"] == 0.61
    assert "| co-membership agreement, adjusted for chance | 0.61 |" in cmp.render_markdown({"videos": {VID: r}})
    assert (s["exchange_topic_agree"], s["exchange_topic_n"]) == (1, 2)
    assert (s["closing_agree"], s["closing_n"]) == (2, 3)
    md = cmp.render_markdown({"videos": {VID: r}})
    for row in ("| turn recall | 0.9 |", "| turn precision | 0.8 |", "| turn recall, any function | 0.95 |",
                "| turn precision, any function | 0.9 |", "| offer membership agreement | 0.5 |",
                "| speaker-time agreement | 0.85 |", "| exchanges, primary and replicate | 4 and 5 |",
                "| exchanges matched on half their turns or more | 3 |",
                "| mean overlap of matched exchanges (Jaccard) | 0.7 |", "| co-membership agreement | 0.88 |",
                "| topic agreement, matched substantive exchanges | 1 of 2 |",
                "| closing-state agreement, investors in both passes | 2 of 3 |"):
        assert row in md, row
    assert "event recall" not in md


def test_self_consistency_from_an_older_agreement_report_gives_nulls(tmp_path):
    new = build_new(tmp_path / "n")
    wj(new / "agreement.json", {"events": {"recall": 0.9, "precision": 0.8}})
    r = cmp.compare_video(new, build_old(tmp_path / "o"))
    s = r["self_consistency"]
    assert s["turn_recall"] is None and s["turn_recall_any_function"] is None
    assert s["offer_membership_agreement"] is None and s["speaker_time_agreement"] is None
    assert s["exchanges_matched_at_half"] is None and s["co_membership_agreement"] is None
    assert s["exchange_topic_agree"] is None and s["closing_n"] is None
    assert s["time"] is None and s["turn_sequence_agreement"] is None
    md = cmp.render_markdown({"videos": {VID: r}})
    assert "None" not in md
    assert "| topic agreement, matched substantive exchanges | n/a |" in md


def test_self_consistency_reports_time_diagnostics_and_turn_sequence(pair):
    new, old, _ = pair
    r = cmp.compare_video(new, old)
    s = r["self_consistency"]
    assert s["time"] == {"matched_turn_pairs": 7, "median_start_diff_s": -1.5, "median_abs_start_diff_s": 2.5,
                         "max_abs_start_diff_s": 41.0, "turn_pairs_beyond_tolerance": 1,
                         "decision_pairs_beyond_tolerance": 0}
    assert s["turn_sequence_agreement"] == 0.75
    md = cmp.render_markdown({"videos": {VID: r}})
    summary = md[md.index("## Summary across videos"):md.index(f"## {VID}")]
    header, _, row = [ln for ln in summary.splitlines() if ln.startswith("|")][:3]
    assert header.rstrip(" |").endswith("median abs start diff s")
    assert row.rstrip(" |").endswith("2.5")
    for name in ("turn recall", "turn precision", "exchanges matched at half", "co-membership"):
        assert f"| {name} |" in header, name
    for line in ("| turn sequence agreement | 0.75 |", "| median start difference (s), replicate minus primary | -1.5 |",
                 "| median absolute start difference (s) | 2.5 |", "| largest absolute start difference (s) | 41 |",
                 "| matched turn pairs with times | 7 |",
                 "| turn pairs beyond the time threshold | 1 |", "| decision pairs beyond the time threshold | 0 |"):
        assert line in md, line


def test_summary_shows_no_time_difference_without_an_agreement_report(tmp_path):
    new = build_new(tmp_path / "n")
    (new / "agreement.json").unlink()
    md = cmp.render_markdown({"videos": {VID: cmp.compare_video(new, build_old(tmp_path / "o"))}})
    assert "median abs start diff s" in md and "None" not in md


def test_structure_labels_are_counted_in_their_canonical_form(tmp_path):
    new, old = build_new(tmp_path / "n"), build_old(tmp_path / "o")
    clean = cmp.compare_video(new, old)

    def mut(rec):
        for t in rec["turns"]:
            t["function"] = t["function"].upper()
        for e in rec["exchanges"]:
            e["kind"], e["topic"] = e["kind"].title(), e["topic"].lower()
        for d in rec["decisions"]:
            d["kind"] = d["kind"].upper()
    _edit_primary(new, mut)
    r = cmp.compare_video(new, old)
    assert r["coverage"] == clean["coverage"] and r["decisions"] == clean["decisions"]
    assert r["codes"] == clean["codes"]


def test_concept_counts_in_coverage_and_markdown(pair):
    new, old, _ = pair
    r = cmp.compare_video(new, old)
    assert r["coverage"]["new_concept_counts"] == {
        "question": {"customer_retention": 1, "operating_efficiency": 3},
        "answer": {"customer_retention": 1, "forecast": 1, "operating_efficiency": 2}}
    md = cmp.render_markdown({"videos": {VID: r}})
    assert ("| New concepts | questions: customer_retention: 1, operating_efficiency: 3; "
            "answers: customer_retention: 1, forecast: 1, operating_efficiency: 2 |") in md
    assert "new_concept_counts" not in md
    assert "| PR-FRAME (new: derived from the concept) |" in md and "| RS-FRAME (new: derived from the concept) |" in md


def test_a_re_entry_with_an_offer_id_counts_as_a_new_investor_who_offered(pair):
    new, old, _ = pair

    def reenter(rec):
        rec["decisions"].append({"decision_id": 5, "actor": "S3", "kind": "re_entry", "time_s": 175.0,
                                 "turn_id": None, "offer_id": "O2", "terms": "$200,000 for 25%", "note": "Back."})
    _edit_primary(new, reenter)
    d = cmp.compare_video(new, old)["decisions"]
    assert d["new_n_ever_offer"] == 3
