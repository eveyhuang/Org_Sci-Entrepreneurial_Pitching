import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.schemas.stage_b_panel import ANNOTATION_CONTEXT, PanelRecord  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.panel_pipeline import derive, load_best_record, prepared_window  # noqa: E402
from pipeline.stage_b.panel_validate import validate_record  # noqa: E402
from pipeline.stage_b.replicate_agreement import align  # noqa: E402
from pipeline.stage_b.review_items import SHEET_CLASSES, any_offer, flatten, set_value  # noqa: E402
from tests.panel_fixtures import DURATION_S, synthetic_record  # noqa: E402

# the full-video sheet as it was drawn before 2026-10-21 (plan C Task 7d), with the outcome (pitch:any_offer) and
# prepared-window rows: such old sheets must still import and count, so the tests that fill those rows export it
OLD_SHEET = ("--field-classes", *SHEET_CLASSES)

COLUMNS = ["priority", "row", "path", "field_class", "clock_start", "clock_end", "context", "primary", "replicate", "reasons",
           "selected_as", "allowed_values", "human_value", "resolution", "note"]


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def _mutated(d):
    d["turns"][3]["codes"][1]["value"] = "1"        # turn 4 PR-SEV: 3 in the primary
    d["stage_a"][2]["value"] = 5                    # interpersonal_warmth: 3 in the primary


def make_run(tmp_path, books, video_id="vid1", replicate=True, mutate=None, mutate_primary=None):
    """A two-pass run directory of the synthetic record with a replicate that differs in two places."""
    vdir = tmp_path / "run" / video_id
    (vdir / "results").mkdir(parents=True)
    release = load_release()
    (vdir / "experiment.json").write_text(json.dumps({
        "video_id": video_id, "source": f"/videos/{video_id}.mp4", "duration_s": DURATION_S,
        "release_id": release.release_id}))
    records = {"primary": synthetic_record()}
    if mutate_primary:
        mutate_primary(records["primary"])
    if replicate:
        records["replicate"] = synthetic_record()
        (mutate or _mutated)(records["replicate"])
    parsed = {}
    for name, d in records.items():
        record = parsed[name] = PanelRecord.model_validate(d)
        (vdir / "results" / f"{name}.json").write_text(json.dumps({
            "pass": name, "unit_key": f"panel__{video_id}__{name}", "annotation_context": ANNOTATION_CONTEXT,
            "prepared_window": prepared_window(record), "derived": derive(record, books), "normalizations": [],
            "record": d}, indent=2))
        report = validate_record(record, books, DURATION_S)
        (vdir / "results" / f"{name}_validation.json").write_text(
            json.dumps({"errors": report.errors, "flags": report.flags}))
    if replicate:
        (vdir / "agreement.json").write_text(json.dumps(
            {"video_id": video_id, **align(parsed["primary"], parsed["replicate"], books)}))
    return tmp_path / "run", vdir


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_sheet(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def fill(path, edits):
    """Fill rows of a sheet: {path: {column: value}}; every other row is left (or made) blank."""
    rows = read_sheet(path)
    for row in rows:
        row.update({"human_value": "", "resolution": "", "note": ""})
        row.update(edits.get(row["path"], {}))
    missing = set(edits) - {row["path"] for row in rows}
    assert not missing, missing
    write_sheet(path, rows)


def write_sheet(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def export(run, *extra):
    return rp.main(["export", "--run", str(run), "--preset", "flagged", "--sample-unflagged", "10000", *extra])


def values(record_dict, books):
    return {i.path: i.value for i in flatten(PanelRecord.model_validate(record_dict), books)}


# ---- export -------------------------------------------------------------------------------

def test_export_writes_the_sheet_the_instructions_and_the_manifest(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    hashes = {n: sha(vdir / "results" / f"{n}.json") for n in ("primary", "replicate")}
    assert rp.main(["export", "--run", str(run), "--preset", "flagged", "--sample-unflagged", "12", "--seed", "3"]) == 0
    review = vdir / "review"
    assert sorted(p.name for p in review.iterdir()) == ["export_manifest.json", "vid1__review.csv", "vid1__review.md"]
    raw = (review / "vid1__review.csv").read_bytes().decode("utf-8-sig")
    assert raw.splitlines()[0] == ",".join(COLUMNS)
    rows = read_sheet(review / "vid1__review.csv")
    assert len(raw.splitlines()) == len(rows) + 1                       # one line per item: no cell holds a newline
    assert [r["row"] for r in rows] == [str(i) for i in range(1, len(rows) + 1)]
    by_path = {r["path"]: r for r in rows}
    sev = by_path["turn:4/code:PR-SEV"]
    assert (sev["primary"], sev["replicate"], sev["reasons"], sev["selected_as"]) == ("3", "1", "passes_differ",
                                                                                      "flagged")
    assert sev["allowed_values"] == "1 | 2 | 3" and sev["field_class"] == "turn_code"
    assert sev["priority"] == "4" and {r["priority"] for r in rows if r["selected_as"] == "sampled"} == {"1"}
    assert (sev["clock_start"], sev["clock_end"]) == ("1:20", "1:26")
    assert sev["context"].startswith("turn 4, S2 (Investor B): Disputes")
    assert by_path["stage_a:interpersonal_warmth"]["reasons"] == "stage_a_gap"
    assert all(r["human_value"] == r["resolution"] == r["note"] == "" for r in rows)
    assert [r["selected_as"] for r in rows].count("sampled") == 12
    assert [r["selected_as"] for r in rows].count("flagged") == 2

    entry = json.loads((review / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["seed"] == 3 and entry["sample_unflagged"] == 12 and entry["blind"] is False
    assert entry["release_id"] == load_release().release_id
    assert entry["sheet"] == "vid1__review.csv"
    assert entry["primary_sha256"] == hashes["primary"] and entry["replicate_sha256"] == hashes["replicate"]
    assert entry["counts"]["by_selection"] == {"flagged": 2, "sampled": 12}
    assert entry["counts"]["by_priority"] == {"1": 12, "2": 0, "3": 0, "4": 2}
    assert entry["counts"]["by_reason"] == {"passes_differ": 1, "stage_a_gap": 1}
    assert (entry["max_flagged"], entry["preset"], entry["field_classes"]) == (60, "flagged", None)
    assert entry["sample"] is None
    assert entry["flagged"]["found"]["n"] == entry["flagged"]["on_sheet"]["n"] == 2
    assert entry["flagged"]["left_out"] == {"n": 0, "by_field_class": {}, "by_reason": {}}
    assert sum(entry["counts"]["by_field_class"].values()) == 14
    assert entry["counts"]["by_field_class"]["turn_code"] >= 1
    assert [r["path"] for r in entry["rows"]] == [r["path"] for r in rows]
    listed = {r["path"]: r for r in entry["rows"]}
    assert listed["turn:4/code:PR-SEV"]["reasons"] == ["passes_differ"]
    assert listed["turn:4/code:PR-SEV"]["selected_as"] == "flagged"

    md = (review / "vid1__review.md").read_text()
    for needle in ("/videos/vid1.mp4", "0:05-1:00", "| S2 | investor | Investor B |", "human_value", "resolution",
                   "confirm_replicate", "structural", "missing:", "not_applicable", "PR-SEV", "adjudication",
                   "`priority`", "1 = a sampled row", "unbiased estimate"):
        assert needle in md, needle
    for gone in ("satisfied", "closing", "ever_offer", "exchange_investor"):          # no closing state since v3-dev9
        assert gone not in md, gone
    # the allowed values cover the kinds of row on the sheet: the complete sheet lists every kind
    assert "cut" not in entry["counts"]["by_field_class"] and "pitch:first_decision" not in md     # no such row here
    assert rp.main(["export", "--run", str(run), "--preset", "full", "--sample-unflagged", "10000", "--force"]) == 0
    md = (review / "vid1__review.md").read_text()
    for needle in ("TURN-FUNC", "accepts", "all_women", "pitch:any_offer", "200000;10",
                   "prep_pres:<CODE>", "`turn_link`", "DEC-KIND"):
        assert needle in md, needle
    assert {n: sha(vdir / "results" / f"{n}.json") for n in hashes} == hashes
    assert "vid1" in capsys.readouterr().out


def test_export_is_reproducible_for_a_seed_and_differs_between_seeds(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    sheets = []
    for seed in ("1", "1", "2"):
        assert rp.main(["export", "--run", str(run), "--preset", "flagged", "--seed", seed, "--force"]) == 0
        sheets.append((vdir / "review" / "vid1__review.csv").read_bytes())
    assert sheets[0] == sheets[1] != sheets[2]
    assert len(read_sheet(vdir / "review" / "vid1__review.csv")) == 22      # the default sample of 20 and 2 flagged


def test_blind_export_hides_every_model_value_and_reason(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert export(run, "--blind") == 0
    rows = read_sheet(vdir / "review" / "vid1__review.csv")
    assert rows and all(r["primary"] == r["replicate"] == r["reasons"] == r["selected_as"] == r["priority"] == ""
                        for r in rows)
    by_path = {r["path"]: r for r in rows}
    assert by_path["turn:4/code:PR-SEV"]["context"] == "turn 4, S2 (Investor B)"        # no model summary
    assert by_path["exchange:1/topic"]["context"] == "exchange 1, turns 2-7, 14-16"
    assert by_path["decision:2/kind"]["context"] == "decision 2, turn 17"
    assert by_path["turn:4/code:PR-SEV"]["allowed_values"] == "1 | 2 | 3"
    # a clock that is itself the judged value is not shown: a decision is located by its turn, the window not at all
    assert (by_path["decision:2/time"]["clock_start"], by_path["decision:2/time"]["clock_end"]) == ("2:38", "2:43")
    assert (by_path["decision:2/kind"]["clock_start"], by_path["decision:2/kind"]["clock_end"]) == ("2:38", "2:43")
    assert (by_path["prepared_window"]["clock_start"], by_path["prepared_window"]["clock_end"]) == ("", "")
    assert (by_path["turn:4/code:PR-SEV"]["clock_start"], by_path["turn:4/code:PR-SEV"]["clock_end"]) == ("1:20", "1:26")
    text = (vdir / "review" / "vid1__review.csv").read_text(encoding="utf-8-sig")
    assert "passes_differ" not in text and "Disputes" not in text and "customer acquisition cost" not in text
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["blind"] is True
    listed = {r["path"]: r for r in entry["rows"]}
    assert (listed["turn:4/code:PR-SEV"]["primary"], listed["turn:4/code:PR-SEV"]["replicate"]) == ("3", "1")
    assert listed["turn:4/code:PR-SEV"]["reasons"] == ["passes_differ"]
    md = (vdir / "review" / "vid1__review.md").read_text()
    assert "blind" in md and "passes_differ" not in md and "Flagged rows" not in md and "`priority`" not in md
    assert "replicate_only" not in md


def test_export_does_not_overwrite_a_sheet_unless_forced(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:2/function": {"human_value": "question"}})
    kept = sheet.read_bytes()
    assert export(run) == 1
    assert "--force" in capsys.readouterr().err and sheet.read_bytes() == kept
    assert export(run, "--force") == 0 and sheet.read_bytes() != kept


def test_export_to_another_directory_and_only_the_named_videos(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books)
    make_run(tmp_path, books, video_id="vid2")
    out = tmp_path / "sheets"
    assert export(run, "--out", str(out), "--video-ids", "vid2") == 0
    assert sorted(p.name for p in out.iterdir()) == ["export_manifest.json", "vid2__review.csv", "vid2__review.md"]
    assert export(run, "--out", str(out), "--video-ids", "vid1") == 0
    assert set(json.loads((out / "export_manifest.json").read_text())["videos"]) == {"vid1", "vid2"}
    assert not (run / "vid1" / "review").exists() and not (run / "vid2" / "review").exists()
    assert export(run, "--video-ids", "nope") == 1
    assert "nope" in capsys.readouterr().err


def test_export_without_a_replicate_still_gives_a_sheet(tmp_path, books):
    run, vdir = make_run(tmp_path, books, replicate=False)
    assert rp.main(["export", "--run", str(run), "--preset", "flagged", "--sample-unflagged", "5"]) == 0
    rows = read_sheet(vdir / "review" / "vid1__review.csv")
    assert len(rows) == 5 and all(r["replicate"] == "" for r in rows)
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["replicate_sha256"] is None


def _many_differences(d):
    for i in d["stage_a"]:
        i["value"] = 5                                          # three Stage A gaps
    for t in d["turns"]:
        for c in t["codes"]:
            if c["code"] == "PR-SEV":
                c["value"] = "2" if c["value"] != "2" else "1"  # four severities
    d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]  # one turn the replicate lacks


def test_the_cap_the_presets_and_the_field_classes(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books, mutate=_many_differences)
    manifest = vdir / "review" / "export_manifest.json"

    def exported(*options):
        assert rp.main(["export", "--run", str(run), "--force", *options]) == 0
        return json.loads(manifest.read_text())["videos"]["vid1"], read_sheet(vdir / "review" / "vid1__review.csv")

    entry, rows = exported("--preset", "flagged")
    assert (entry["max_flagged"], entry["sample_unflagged"]) == (60, 20)
    assert entry["counts"]["by_selection"] == {"flagged": 8, "sampled": 20} and len(rows) == 28

    capsys.readouterr()
    entry, rows = exported("--preset", "flagged", "--max-flagged", "6")
    assert "2 left out by --max-flagged 6" in capsys.readouterr().out
    assert entry["counts"]["by_selection"] == {"flagged": 6, "sampled": 20}
    assert "turn:13/present" in {r["path"] for r in rows}
    flagged = entry["flagged"]
    assert (flagged["found"]["n"], flagged["on_sheet"]["n"], flagged["left_out"]["n"]) == (8, 6, 2)
    assert flagged["found"]["by_field_class"] == {"unit_presence": 1, "turn_code": 4, "stage_a": 3}
    assert sum(flagged["left_out"]["by_field_class"].values()) == 2
    assert sum(flagged["left_out"]["by_reason"].values()) == 2
    assert set(flagged["left_out"]["by_reason"]) == {"passes_differ", "stage_a_gap"}
    md = (vdir / "review" / "vid1__review.md").read_text()
    assert "6 of 8" in md and "left out 2" in md
    again, same_rows = exported("--preset", "flagged", "--max-flagged", "6")
    assert same_rows == rows and again == entry                             # reproducible from the seed
    assert [r["path"] for r in exported("--preset", "flagged", "--max-flagged", "6", "--seed", "5")[1]] != [r["path"] for r in rows]

    # the left-out counts travel into the agreement report
    sheet = vdir / "review" / "vid1__review.csv"
    exported("--preset", "flagged", "--max-flagged", "6")
    fill(sheet, {"turn:13/present": {"human_value": "present"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["flagged_left_out"] == flagged["left_out"] and "reviewed subset" in agreement["flagged_note"]

    for options, cap, sample in ((["--preset", "quick"], 25, 15), (["--preset", "full"], None, 30),
                                 (["--preset", "quick", "--sample-unflagged", "3"], 25, 3),
                                 (["--preset", "full", "--max-flagged", "2"], 2, 30),
                                 (["--preset", "flagged", "--max-flagged", "-1"], None, 20)):
        entry, rows = exported(*options)
        assert (entry["max_flagged"], entry["sample_unflagged"]) == (cap, sample), options
        assert entry["counts"]["by_selection"]["sampled"] == sample
        assert entry["preset"] == options[1]

    # the complete sheet takes every field class whole: actors, free text, trivial links and outcomes come back
    entry, rows = exported("--preset", "full", "--sample-unflagged", "10000")
    paths = {r["path"] for r in rows}
    assert {"actor:S1/role", "exchange:1/label", "turn:2/other_concept", "turn:3/responds_to",
            "pitch:any_offer", "turn:13/present"} <= paths
    # every class but the first decision and the pair judgements, which no flag-driven sheet draws
    assert {r["field_class"] for r in rows} == set(rp.SHEET_CLASSES) - {"cut", "pair"}
    assert "pitch:any_offer" not in {r["path"] for r in exported("--preset", "flagged")[1]}

    entry, rows = exported("--preset", "flagged", "--field-classes", "stage_a", "unit_presence",
                           "--sample-unflagged", "0")
    assert {r["field_class"] for r in rows} == {"stage_a", "unit_presence"} and len(rows) == 4
    assert entry["field_classes"] == ["stage_a", "unit_presence"]
    entry, rows = exported("--preset", "flagged", "--field-classes", "actor", "--sample-unflagged", "100")
    assert {r["path"] for r in rows} >= {"actor:S1/label", "actor:S1/role"} and len(rows) == 8
    capsys.readouterr()
    assert rp.main(["export", "--run", str(run), "--force", "--field-classes", "mood"]) == 1
    assert "mood" in capsys.readouterr().err
    # an option of the other kind of preset is refused, not ignored
    assert rp.main(["export", "--run", str(run), "--force", "--max-flagged", "5"]) == 1
    assert "--max-flagged" in capsys.readouterr().err
    assert rp.main(["export", "--run", str(run), "--force", "--preset", "full", "--qa-per-video", "1"]) == 1
    assert "--qa-per-video" in capsys.readouterr().err


def _two_sided(tmp_path, books):
    """The replicate lacks turn 13; the primary lacks decision 1, which the replicate has."""
    def primary(d):
        d["decisions"] = [x for x in d["decisions"] if x["decision_id"] != 1]

    def replicate(d):
        d["turns"] = [t for t in d["turns"] if t["turn_id"] != 13]
    return make_run(tmp_path, books, mutate=replicate, mutate_primary=primary)


def test_a_unit_one_pass_lacks_is_one_row_to_judge_present_or_absent(tmp_path, books):
    run, vdir = _two_sided(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    by_path = {r["path"]: r for r in read_sheet(sheet)}
    here = by_path["turn:13/present"]
    assert (here["field_class"], here["primary"], here["replicate"], here["priority"]) == (
        "unit_presence", "present", "absent", "2")
    assert (here["reasons"], here["allowed_values"]) == ("no_counterpart", "present | absent")
    assert here["context"] == "turn 13, S2 (Investor B): Calls the reorder rate strong."
    there = by_path["replicate_only:decision@1:51"]
    assert (there["primary"], there["replicate"], there["allowed_values"]) == ("absent", "present",
                                                                               "present | absent")
    assert there["context"] == "only in the replicate, decision by S3 (Investor C): exit"
    assert "turn:13/function" not in by_path         # the missing turn is judged as a unit: its fields are not sampled
    md = (vdir / "review" / "vid1__review.md").read_text()
    assert "- `unit_presence`: `present` or `absent`" in md and "replicate_only:" in md

    hashes = {n: sha(vdir / "results" / f"{n}.json") for n in ("primary", "replicate")}
    fill(sheet, {"turn:13/present": {"human_value": "Absent", "note": "nobody speaks there"},
                 "replicate_only:decision@1:51": {"human_value": "present"},
                 "turn:4/code:PR-SEV": {"human_value": "2"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert {n: sha(vdir / "results" / f"{n}.json") for n in hashes} == hashes
    primary = json.loads((vdir / "results" / "primary.json").read_text())
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["corrections"] == 1                                     # the severity; a judgment changes nothing
    set_value(reviewed["record"], "turn:4/code:PR-SEV", "3", books)
    assert reviewed["record"] == primary["record"]                          # turn 13 is still there, no decision added
    log = {e["path"]: e for e in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    gone, missed = log["turn:13/present"], log["replicate_only:decision@1:51"]
    assert (gone["resolution"], gone["old"], gone["new"], gone["note"]) == ("structural", "present", "absent",
                                                                            "nobody speaks there")
    assert (gone["field_class"], gone["selected_as"]) == ("unit_presence", "flagged")
    assert (missed["resolution"], missed["old"], missed["new"]) == ("structural", "absent", "present")
    assert "the primary missed" in missed["note"]
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["by_field_class"]["unit_presence"] == {
        "n": 2, "primary_only": {"judged_present": 0.0, "n": 1}, "replicate_only": {"judged_present": 1.0, "n": 1}}
    assert agreement["n_reviewed"] == 3 and agreement["all"]["n"] == 1
    assert agreement["flagged_left_out"] is None and agreement["flagged_note"] is None

    # the judgments that agree with the primary confirm it
    assert export(run, "--force") == 0
    fill(sheet, {"turn:13/present": {"human_value": "present"},
                 "replicate_only:decision@1:51": {"resolution": "confirm_primary"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()][-2:]
    assert [(e["path"], e["resolution"], e["old"], e["new"]) for e in log] == [
        ("replicate_only:decision@1:51", "confirm_primary", "absent", "absent"),
        ("turn:13/present", "confirm_primary", "present", "present")]
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["by_field_class"]["unit_presence"]["primary_only"] == {"judged_present": 1.0, "n": 1}
    assert agreement["by_field_class"]["unit_presence"]["replicate_only"] == {"judged_present": 0.0, "n": 1}
    assert json.loads((vdir / "results" / "reviewed.json").read_text())["corrections"] == 1


def test_a_unit_presence_row_takes_only_present_or_absent(tmp_path, books, capsys):
    run, vdir = _two_sided(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:13/present": {"human_value": "maybe"}, "turn:4/code:PR-SEV": {"human_value": "2"}})
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    err = capsys.readouterr().err
    assert "turn:13/present takes present or absent, not 'maybe'" in err and "nothing was written" in err
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before


def test_a_blind_sheet_has_no_unit_presence_row_at_all(tmp_path, books):
    """Such a row exists only because the other pass lacks the unit, which is what a blind coder must not learn."""
    run, vdir = _two_sided(tmp_path, books)
    assert export(run, "--blind") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    assert rows and not [r["path"] for r in rows if r["path"].endswith("/present")
                         or r["path"].startswith("replicate_only:") or r["field_class"] == "unit_presence"]
    text = sheet.read_text(encoding="utf-8-sig")
    assert "only in the replicate" not in text and "no_counterpart" not in text and "unit_presence" not in text
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert not [r for r in entry["rows"] if r["field_class"] == "unit_presence"]
    assert "unit_presence" not in entry["counts"]["by_field_class"]
    assert "`unit_presence`: `present` or `absent`" not in (vdir / "review" / "vid1__review.md").read_text()
    # the adjudication sheet of the same run has them
    assert export(run, "--force") == 0
    assert {"turn:13/present"} <= {r["path"] for r in read_sheet(sheet)}


def test_a_blind_page_hides_the_prepared_window_and_says_what_it_still_reveals(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    open_page = (vdir / "review" / "vid1__review.md").read_text()
    assert "- Prepared presentation: 0:05-1:00" in open_page and "still reveals" not in open_page
    open_rows = {r["path"]: r for r in read_sheet(vdir / "review" / "vid1__review.csv")}
    assert (open_rows["stage_a:interpersonal_warmth"]["clock_start"], open_rows["stage_a:interpersonal_warmth"]["clock_end"]) == (
        "0:05", "1:00")
    assert export(run, "--blind", "--force") == 0
    page = (vdir / "review" / "vid1__review.md").read_text()
    assert "0:05-1:00" not in page and "Prepared presentation: judge it as you see it" in page
    blind = {r["path"]: r for r in read_sheet(vdir / "review" / "vid1__review.csv")}
    stage_a = [r for path, r in blind.items() if path.startswith("stage_a:")]
    assert len(stage_a) == 3 and all((r["clock_start"], r["clock_end"]) == ("", "") for r in stage_a)
    assert "judge the prepared presentation as you see it" in page
    flat = " ".join(page.split())
    assert "## What blind mode still reveals" in page
    for phrase in ("the model's turn segmentation", "who speaks each turn",
                   "which turns the model took for questions or answers", "`turn:<id>/code:PR-"):
        assert phrase in flat, phrase


# ---- import -------------------------------------------------------------------------------

def test_a_filled_sheet_becomes_a_reviewed_record_beside_the_untouched_passes(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    hashes = {n: sha(vdir / "results" / f"{n}.json") for n in ("primary", "replicate")}
    fill(sheet, {"turn:4/code:PR-SEV": {"resolution": "confirm_replicate"},
                 "stage_a:interpersonal_warmth": {"human_value": "4", "note": "between the two"},
                 "turn:2/function": {"human_value": "Question"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "3 rows reviewed" in capsys.readouterr().out
    assert {n: sha(vdir / "results" / f"{n}.json") for n in hashes} == hashes

    primary = json.loads((vdir / "results" / "primary.json").read_text())
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert set(reviewed) == {"based_on", "based_on_sha256", "reviewer", "record", "derived", "prepared_window",
                             "corrections", "annotation_context", "human_reviewed", "pitch_override"}
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": None}
    assert (reviewed["based_on"], reviewed["based_on_sha256"]) == ("primary", hashes["primary"])
    assert (reviewed["reviewer"], reviewed["corrections"], reviewed["human_reviewed"]) == ("ana", 2, "partial")
    assert reviewed["annotation_context"] == ANNOTATION_CONTEXT
    record = PanelRecord.model_validate(reviewed["record"])
    assert reviewed["derived"] == derive(record, books) and reviewed["prepared_window"] == ["0:05-1:00"]

    before, after = values(primary["record"], books), values(reviewed["record"], books)
    assert {p for p in before if before[p] != after[p]} == {"turn:4/code:PR-SEV", "stage_a:interpersonal_warmth"}
    assert (after["turn:4/code:PR-SEV"], after["stage_a:interpersonal_warmth"]) == ("1", "4")
    restored = reviewed["record"]
    set_value(restored, "turn:4/code:PR-SEV", "3", books)
    set_value(restored, "stage_a:interpersonal_warmth", "3", books)
    assert restored["stage_a"][2]["note"] == "Steady delivery. [value changed in review]"
    restored["stage_a"][2]["note"] = "Steady delivery."
    assert restored == primary["record"]                    # nothing but the corrected paths (and that note) changed

    validation = json.loads((vdir / "results" / "reviewed_validation.json").read_text())
    assert validation == {"errors": [], "flags": json.loads(
        (vdir / "results" / "primary_validation.json").read_text())["flags"]}
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert sorted((e["path"], e["old"], e["new"], e["resolution"], e["selected_as"]) for e in log) == [
        ("stage_a:interpersonal_warmth", "3", "4", "correct", "flagged"),
        ("turn:2/function", "question", "question", "confirm_primary", "sampled"),
        ("turn:4/code:PR-SEV", "3", "1", "confirm_replicate", "flagged")]
    assert all(e["reviewer"] == "ana" and e["sheet"] == "vid1__review.csv" for e in log)
    assert next(e for e in log if e["path"] == "stage_a:interpersonal_warmth")["note"] == "between the two"
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["n_reviewed"] == 3 and agreement["video_id"] == "vid1"
    # both flagged rows are ordinal (PR-SEV 1 against 3 and 1; Stage A 4 against 3 and 5), so within-one is given too
    assert agreement["by_selection"]["flagged"]["all"] == {
        "n": 2, "human_vs_primary": {"agree": 0.0, "n": 2}, "human_vs_replicate": {"agree": 0.5, "n": 2},
        "primary_vs_replicate": {"agree": 0.0, "n": 2},
        # Stage A and PR-SEV are two variables with one unit each; neither can estimate its expected disagreement
        "alpha": None, "alpha_human_vs_replicate": None, "alpha_primary_vs_replicate": None,
        "within_one": {"n": 2, "human_vs_primary": {"agree": 0.5, "n": 2},
                       "human_vs_replicate": {"agree": 1.0, "n": 2}, "primary_vs_replicate": {"agree": 0.0, "n": 2}}}
    assert agreement["by_selection"]["sampled"]["all"]["human_vs_primary"] == {"agree": 1.0, "n": 1}
    assert agreement["by_field_class"]["stage_a"]["n"] == 1


def test_an_invalid_sheet_writes_nothing_and_reports_every_problem(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:4/code:PR-SEV": {"human_value": "9"},
                 "exchange:1/topic": {"human_value": "pricing"},
                 "turn:13/function": {"human_value": "question"},
                 "turn:2/function": {"resolution": "approve"},
                 "stage_a:interpersonal_warmth": {"human_value": "4"}})
    numbers = {r["path"]: r["row"] for r in read_sheet(sheet)}
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    err = capsys.readouterr().err
    assert f"row {numbers['turn:4/code:PR-SEV']}: turn:4/code:PR-SEV" in err and "outside 1-3" in err
    assert f"row {numbers['exchange:1/topic']}: exchange:1/topic" in err and "not one of" in err
    assert f"row {numbers['turn:13/function']}: turn:13/function" in err and "PR-CONCEPT" in err
    assert f"row {numbers['turn:2/function']}: unknown resolution 'approve'" in err
    assert "nothing was written" in err
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before
    assert not (vdir / "results" / "reviewed.json").exists()

    fill(sheet, {})                                             # nothing reviewed is not a review either
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    assert "no row" in capsys.readouterr().err and not (vdir / "results" / "reviewed.json").exists()


def test_import_refuses_when_the_primary_changed_since_the_export(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    path = vdir / "results" / "primary.json"
    path.write_text(path.read_text() + "\n")
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    assert "sha256" in capsys.readouterr().err
    assert not (vdir / "results" / "reviewed.json").exists() and not (vdir / "review" / "review_log.jsonl").exists()


def test_a_second_import_stacks_on_the_reviewed_record(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    fill(sheet, {"turn:9/exchange": {"resolution": "correct", "note": "outside the joke"},
                 "turn:4/code:PR-SEV": {"human_value": "2"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    after = values(reviewed["record"], books)
    assert (after["stage_a:interpersonal_warmth"], after["turn:9/exchange"], after["turn:4/code:PR-SEV"]) == ("4", None, "2")
    assert (reviewed["corrections"], reviewed["reviewer"]) == (3, "ana, ben")
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert [(e["path"], e["reviewer"], e["import"]) for e in log] == [
        ("stage_a:interpersonal_warmth", "ana", 1), ("turn:4/code:PR-SEV", "ben", 2), ("turn:9/exchange", "ben", 2)]
    assert json.loads((vdir / "review" / "human_agreement.json").read_text())["n_reviewed"] == 3

    # confirming the primary on a path an earlier import corrected is ambiguous, so it is refused
    kept = (vdir / "results" / "reviewed.json").read_bytes()
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "3"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 1
    assert "earlier import" in capsys.readouterr().err
    assert (vdir / "results" / "reviewed.json").read_bytes() == kept
    assert len((vdir / "review" / "review_log.jsonl").read_text().splitlines()) == 3


def test_a_blind_sheet_imports_against_the_values_kept_in_the_manifest(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert export(run, "--blind") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:4/code:PR-SEV": {"human_value": "3"},           # the coder, unaware, agrees with the primary
                 "stage_a:interpersonal_warmth": {"human_value": "5"},        # ... and here with the replicate
                 "turn:2/function": {"human_value": "challenges"}})    # ... and here with neither
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", "--role", "adjudicator"]) == 0
    log = {e["path"]: e for e in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    assert log["turn:4/code:PR-SEV"]["resolution"] == "confirm_primary"
    assert (log["stage_a:interpersonal_warmth"]["resolution"], log["stage_a:interpersonal_warmth"]["replicate"]) == ("correct", "5")
    # the question codes went with the function; they are a consequence, not rows the coder judged
    assert log["turn:2/code:PR-SEV"]["new"] is None and log["turn:2/code:PR-SEV"]["implied"] is True
    assert log["turn:2/function"]["implied"] is False
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["by_field_class"]["stage_a"]["human_vs_replicate"] == {"agree": 1.0, "n": 1}
    assert agreement["by_field_class"]["turn_code"]["human_vs_primary"] == {"agree": 1.0, "n": 1}
    assert agreement["n_reviewed"] == 3
    assert json.loads((vdir / "results" / "reviewed.json").read_text())["corrections"] == 4


def test_a_row_the_coder_added_is_imported_and_a_sheet_elsewhere_names_its_video(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    out = tmp_path / "sheets"
    assert export(run, "--out", str(out)) == 0
    sheet = out / "vid1__review.csv"
    rows = read_sheet(sheet)
    for row in rows:
        if row["path"] == "turn:13/function":
            row["human_value"] = "question"
    blank = dict.fromkeys(COLUMNS, "")
    rows.append({**blank, "path": "turn:13/code:PR-CONCEPT", "human_value": "customer_retention"})
    rows.append({**blank, "path": "turn:13/code:PR-SEV", "human_value": "1"})
    write_sheet(sheet, rows)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["record"]["turns"][12]["function"] == "question"
    assert [c["code"] for c in reviewed["record"]["turns"][12]["codes"]] == ["PR-CONCEPT", "PR-SEV"]
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert [e["selected_as"] for e in log if "/code:" in e["path"]] == ["added", "added"]
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", "--video-id", "vid9"]) == 1
    assert "vid9" in capsys.readouterr().err


def test_import_needs_the_manifest_beside_the_sheet(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    lone = tmp_path / "elsewhere" / "vid1__review.csv"
    lone.parent.mkdir()
    lone.write_bytes((vdir / "review" / "vid1__review.csv").read_bytes())
    assert rp.main(["import", "--run", str(run), "--sheet", str(lone), "--reviewer", "ana"]) == 1
    assert "export_manifest.json" in capsys.readouterr().err


# ---- status and the best record -----------------------------------------------------------

def test_status_prints_the_counts_per_video(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    make_run(tmp_path, books, video_id="vid2")
    assert export(run, "--video-ids", "vid1") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    n_rows = len(read_sheet(sheet))
    fill(sheet, {"turn:4/code:PR-SEV": {"resolution": "confirm_replicate"},
                 "stage_a:interpersonal_warmth": {"human_value": "4"},
                 "turn:2/function": {"human_value": "question"},
                 "turn:6/function": {"resolution": "unclear"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    assert rp.main(["status", "--run", str(run)]) == 0
    lines = capsys.readouterr().out.splitlines()
    one = next(line for line in lines if line.startswith("vid1"))
    two = next(line for line in lines if line.startswith("vid2"))
    assert f"sheet exported ({n_rows} rows)" in one
    assert "4 rows reviewed" in one and "2 corrections applied" in one
    assert "human = primary on 1/1 sampled rows (1.00)" in one and "0/2 flagged rows (0.00)" in one
    assert "no sheet exported" in two and "0 rows reviewed" in two and "0 corrections applied" in two


def test_load_best_record_prefers_the_reviewed_record(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    result, source = load_best_record(vdir)
    assert source == "primary" and result["pass"] == "primary"
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    result, source = load_best_record(vdir)
    assert source == "reviewed" and result["human_reviewed"] == "partial"
    assert next(i for i in result["record"]["stage_a"] if i["code"] == "interpersonal_warmth")["value"] == 4


# ---- the exchange-level sheet (the default) ------------------------------------------------------

# answer coherence joined the question-answer codes on 2026-10-21 (plan C Task 7e)
SIX_CODES = [["question", "PR-CONCEPT"], ["question", "PR-SEV"], ["answer", "RS-RESP"], ["answer", "RS-EVID"],
             ["answer", "PREP-SUBS"], ["answer", "PREP-COH"]]


def default_export(run, *extra):
    return rp.main(["export", "--run", str(run), *extra])


def test_default_full_video_sheet_draws_no_exchange_rows(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run) == 0
    rows = read_sheet(vdir / "review" / "vid1__review.csv")
    assert not [r for r in rows if r["field_class"] == "exchange_topic" or r["path"].startswith("exchange:")]
    # the first decision, 3 pairs asked x 6 codes (no outcome or window card since Task 7d), of which the fixture has
    # 2 before its first decision (plan C Task 7f)
    assert len(rows) == 13
    classes = [r["field_class"] for r in rows]
    assert classes.count("cut") == 1 and classes.count("turn_code") == 12
    assert not [r for r in rows if r["path"] in ("pitch:any_offer", "prepared_window")]
    sample = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]["sample"]
    assert sample["requested"]["exchanges"] == 0 and "exchange_topic" not in sample["on_sheet"]
    assert sample["population"]["exchange_topic"] == 2          # still counted, for a sheet that asks for them
    # asking for them still works, and the topic card keeps its extra answer
    assert default_export(run, "--force", "--exchanges-per-video", "2") == 0
    rows = {r["path"]: r for r in read_sheet(vdir / "review" / "vid1__review.csv")}
    assert {"exchange:1/topic", "exchange:3/topic"} <= set(rows)
    assert rows["exchange:1/topic"]["allowed_values"].endswith("| not_an_exchange")


def test_paper_table_has_no_exchange_row_by_default(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.load_reporting(books).kind_of("exchange_topic") is None
    assert rp.main(["export", "--run", str(run), "--blind", "--video-ids", "vid1"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:first_decision": {"human_value": "1:51"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    fields = [r["field"] for r in json.loads(out.with_suffix(".json").read_text())["rows"]]
    assert fields and "exchange_topic" not in fields
    assert not [l for l in out.read_text().splitlines() if l.startswith("| exchange_topic")]


def test_a_sheet_with_exchange_rows_still_summarises_and_the_unreported_topic_has_no_row(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--video-ids", "vid1", "--exchanges-per-video", "2"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"exchange:1/topic": {"human_value": "sales"}, "exchange:3/topic": {"human_value": "sales"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    rows = {r["field"]: r for r in json.loads(out.with_suffix(".json").read_text())["rows"]}
    # the topic is not a reported code (validation_reporting.yaml): measured in the summary, no row in the table
    assert "exchange_topic" not in rows and "exchange_topic:topic" not in rows
    assert "exchange_topic" in capsys.readouterr().out


def test_instructions_mention_exchanges_only_when_the_sheet_has_them(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    md = vdir / "review" / "vid1__review.md"
    assert default_export(run) == 0
    text = md.read_text()
    for gone in ("turn_exchange", "exchange_kind", "exchange_topic", "EX-TOPIC", "not_an_exchange"):
        assert gone not in text, gone
    assert "`turn_code`" in text and "`turn_link`" not in text          # the allowed values of the rows on the sheet
    assert default_export(run, "--force", "--exchanges-per-video", "2") == 0
    text = md.read_text()
    assert "`exchange_topic` (EX-TOPIC)" in text and "not_an_exchange" in text
    assert "`turn_exchange`" not in text and "`exchange_kind`" not in text          # no such rows on this sheet
    assert export(vdir.parent, "--force", "--sample-unflagged", "10000") == 0
    text = md.read_text()
    assert "`turn_exchange`" in text and "`exchange_kind` (EX-KIND)" in text


def test_the_how_to_fill_text_describes_the_rows_the_sheet_carries(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    md = vdir / "review" / "vid1__review.md"
    assert default_export(run, "--blind") == 0
    flat = " ".join(md.read_text().split())
    for gone in ("exchange or decision it belongs to", "which turns form an exchange", "clock ranges",
                 "Stage A rows carry no clocks"):
        assert gone not in flat, gone
    assert "`context` which turn it belongs to, or for a pitch row the question it asks" in flat
    assert "six codes" in flat and "coherence" in flat and "The `pitch:first_decision` row belongs to the whole video" in flat
    assert "pitch:any_offer" not in flat and "`prepared_window` row" not in flat     # neither is drawn since Task 7d
    assert "- `prepared_window`" not in flat and "no row of it checks the window" in flat
    # choosing `other` for a concept asks for its name in the note
    assert "Choosing `other` for a concept needs a note: name the matter in a few words" in flat
    assert "A clock within five seconds of the model's is the same moment" in flat
    # an old sheet with the window check: a blind coder is told what to type, and the allowed-values line says the same
    assert default_export(run, "--blind", "--force", *OLD_SHEET) == 0
    old = " ".join(md.read_text().split())
    assert "`pitch:any_offer` and `pitch:first_decision` rows" in old and "leave `human_value` empty" in old
    assert ("- `prepared_window`: no value on a blind sheet: `structural` when the prepared presentation is not "
            "where the coding page shows it, with where it starts and ends in `note`") in old
    # a blind coder sees no model value, so the resolutions that confirm one are not offered
    assert "confirm_primary" not in flat and "confirm_replicate" not in flat and "- `correct`:" in flat
    assert default_export(run, "--blind", "--force", "--exchanges-per-video", "2") == 0
    flat = " ".join(md.read_text().split())
    assert "which turn or exchange it belongs to" in flat and "which turns form an exchange" in flat
    assert default_export(run, "--blind", "--force", "--field-classes", "turn_code", "--qa-codes",
                          "question:PR-SEV") == 0
    flat = " ".join(md.read_text().split())
    how = flat.split("## How to fill the sheet", 1)[1].split("## ", 1)[0]       # the allowed values stay a reference
    assert "pitch:any_offer" not in how and "`prepared_window` row" not in how
    assert "Each question-answer pair carries one code: the question's `PR-SEV`" in flat
    # an adjudication sheet: the window as clock ranges, and the 30 s flag kept apart from the five-second match
    assert default_export(run, "--force", *OLD_SHEET) == 0
    flat = " ".join(md.read_text().split())
    assert "- `prepared_window`: the corrected window as clock ranges separated by `;`" in flat
    assert ("flagged `passes_differ` only when the two passes' first decisions are more than 30 seconds apart"
            in flat)
    assert "A clock within five seconds of the record confirms it" in flat
    assert "- `confirm_primary`:" in flat and "- `confirm_replicate`:" in flat


def test_the_default_export_is_the_exchange_level_sheet(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    hashes = {n: sha(vdir / "results" / f"{n}.json") for n in ("primary", "replicate")}
    assert default_export(run, "--exchanges-per-video", "4") == 0
    assert "preset exchange" in capsys.readouterr().out
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    assert len(rows) == 15          # the first decision, 2 topics, the 2 pairs before the first decision x 6 codes
    assert all(r["selected_as"] == "sampled" and r["priority"] == "1" for r in rows)
    by_path = {r["path"]: r for r in rows}
    # no outcome card (the passes agree on it, and it is a public fact) and no window check since Task 7d
    assert "pitch:any_offer" not in by_path and "prepared_window" not in by_path
    assert not [path for path in by_path if path.startswith("investor:")]
    cut = by_path["pitch:first_decision"]
    assert (cut["field_class"], cut["primary"], cut["replicate"], cut["allowed_values"]) == ("cut", "1:51", "1:51", "")
    assert not [path for path in by_path if path.startswith(("venture:", "stage_a:"))]      # the clip session codes these
    assert by_path["exchange:1/topic"]["allowed_values"].endswith("| not_an_exchange")
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert (entry["preset"], entry["seed"], entry["flagged"]) == ("exchange", 0, None)
    assert (entry["max_flagged"], entry["sample_unflagged"]) == (None, None)
    sample = entry["sample"]
    assert sample["requested"] == {"stage_a": 0, "exchanges": 4, "qa": 3, "venture": 0,
                                   "qa_codes": SIX_CODES}
    assert sample["key"] == "vid1" and sample["drawn"]["pitch"] == ["first_decision"]
    assert sample["on_sheet"] == entry["counts"]["by_field_class"] == {"cut": 1, "turn_code": 12, "exchange_topic": 2}
    assert sample["population"] == {"cut": 1, "turn_code": 12, "exchange_topic": 2, "stage_a": 3, "venture": 4}
    assert sample["qa_before_cut"] is True
    assert entry["counts"]["by_selection"] == {"flagged": 0, "sampled": 15}
    md = (vdir / "review" / "vid1__review.md").read_text()
    for needle in ("preset `exchange`", "## Presets", "- `exchange` (this sheet): the full-video sheet, a random sample",
                   "- `full`: the complete sheet", "- `flagged`:", "- `quick`:", "pitch:first_decision", "six codes",
                   "not_an_exchange"):
        assert needle in md, needle
    assert "yes, no, unknown" not in md and "`prepared_window` row asks only whether" not in md
    for gone in ("closing", "investor's closing state"):
        assert gone not in md, gone
    assert {n: sha(vdir / "results" / f"{n}.json") for n in hashes} == hashes

    kept = sheet.read_bytes()
    assert default_export(run, "--force", "--exchanges-per-video", "4") == 0 and sheet.read_bytes() == kept   # reproducible
    # another seed draws other rows (one pair of the two before the first decision, so some seed differs)
    assert default_export(run, "--force", "--exchanges-per-video", "4", "--qa-per-video", "1") == 0
    one = sheet.read_bytes()
    assert any(default_export(run, "--force", "--exchanges-per-video", "4", "--qa-per-video", "1", "--seed", str(seed))
               == 0 and sheet.read_bytes() != one for seed in range(1, 9))
    assert default_export(run, "--force", "--stage-a-per-video", "3", "--exchanges-per-video", "1",
                          "--qa-per-video", "0") == 0
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["sample"]["requested"] == {"stage_a": 3, "exchanges": 1, "qa": 0, "venture": 0,
                                            "qa_codes": SIX_CODES}
    assert entry["counts"]["by_field_class"]["stage_a"] == 3 and "turn_code" not in entry["counts"]["by_field_class"]
    assert default_export(run, "--force", "--field-classes", "outcome") == 0
    assert {r["field_class"] for r in read_sheet(sheet)} == {"outcome"}
    # the old draw: the outcome and the window check when --field-classes names them
    assert default_export(run, "--force", "--exchanges-per-video", "4", *OLD_SHEET) == 0
    by_path = {r["path"]: r for r in read_sheet(sheet)}
    outcome = by_path["pitch:any_offer"]
    assert (outcome["field_class"], outcome["primary"], outcome["replicate"], outcome["allowed_values"]) == (
        "outcome", "yes", "yes", "yes | no | unknown")
    assert outcome["context"] == "the pitch: S3 exit@1:51, S1 offer@2:39, S2 joins_offer@2:44"
    window = by_path["prepared_window"]
    assert (window["field_class"], window["allowed_values"]) == ("prepared_window", "")
    sample = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]["sample"]
    assert len(by_path) == 17 and sample["drawn"]["pitch"] == ["any_offer", "first_decision"]
    md = (vdir / "review" / "vid1__review.md").read_text()
    for needle in ("pitch:any_offer", "`prepared_window` row asks only whether", "yes, no, unknown"):
        assert needle in md, needle
    capsys.readouterr()
    assert default_export(run, "--force", "--stage-a-per-video", "-1") == 1
    assert "--stage-a-per-video" in capsys.readouterr().err


def test_export_takes_all_stage_a_no_venture_and_chosen_qa_codes(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    default_export(run, "--stage-a-per-video", "all", "--venture-per-video", "0",
                   "--qa-codes", "question:PR-SEV", "answer:PREP-SUBS")
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    classes = [row["field_class"] for row in entry["rows"]]
    assert classes.count("stage_a") == 3 and "venture" not in classes
    assert entry["sample"]["requested"]["qa_codes"] == [["question", "PR-SEV"], ["answer", "PREP-SUBS"]]
    codes = {row["path"].rsplit(":", 1)[1] for row in entry["rows"] if row["field_class"] == "turn_code"}
    assert codes == {"PR-SEV", "PREP-SUBS"}


def test_a_blind_exchange_sheet_names_the_moment_but_not_the_value(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run, "--blind", "--qa-per-video", "10", "--stage-a-per-video", "2",
                          "--venture-per-video", "2", "--exchanges-per-video", "4", *OLD_SHEET) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    assert all(r["primary"] == r["replicate"] == r["reasons"] == r["selected_as"] == r["priority"] == "" for r in rows)
    by_path = {r["path"]: r for r in rows}
    assert by_path["pitch:any_offer"]["context"] == "the pitch: Did any investor make an offer? Watch to the end."
    assert (by_path["pitch:any_offer"]["clock_start"], by_path["pitch:any_offer"]["clock_end"]) == ("", "")
    assert by_path["pitch:first_decision"]["context"] == "the pitch: When is the first decision spoken?"
    assert (by_path["pitch:first_decision"]["clock_start"], by_path["pitch:first_decision"]["clock_end"]) == ("", "")
    window = by_path["prepared_window"]
    assert window["context"] == "Is the prepared presentation where the page says it is?"
    assert (window["clock_start"], window["clock_end"], window["allowed_values"]) == ("", "", "")
    topic = by_path["exchange:3/topic"]
    assert topic["context"] == ("exchange 3, 1:55-2:13; investors: S2 (Investor B); questions: turn 11 (1:55-1:59) "
                                "S2 (Investor B): Asks for the repeat purchase rate.")
    assert (topic["clock_start"], topic["clock_end"]) == ("1:55", "2:13")
    assert not [path for path in by_path if "/investor:" in path]           # no closing state since v3-dev9
    assert by_path["turn:3/code:RS-RESP"]["context"].startswith("question turn 2 (1:02-1:06) S1 (Investor A): Asks")
    assert by_path["stage_a:" + next(r["path"] for r in rows if r["field_class"] == "stage_a").split(":")[1]][
        "context"] == "prepared presentation"
    assert by_path["venture:founder_gender"]["context"] == "the venture"
    text = sheet.read_text(encoding="utf-8-sig")
    for leak in ("offer@", "exit@", "customer acquisition cost", "repeat purchase rate\"", "Steady delivery",
                 "Reusable food containers"):
        assert leak not in text, leak
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    listed = {r["path"]: r for r in entry["rows"]}
    assert (listed["pitch:any_offer"]["primary"], listed["exchange:3/topic"]["primary"]) == (
        "yes", "customer_retention")
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}, "exchange:3/topic": {"human_value": "sales"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    log = {e["path"]: e for e in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    assert log["pitch:any_offer"]["resolution"] == "confirm_primary"
    assert (log["exchange:3/topic"]["resolution"], log["exchange:3/topic"]["new"]) == ("correct", "sales")


def test_outcome_judgments_become_overrides_beside_an_unedited_record(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run, "--exchanges-per-video", "4", *OLD_SHEET) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    hashes = {n: sha(vdir / "results" / f"{n}.json") for n in ("primary", "replicate")}
    fill(sheet, {"pitch:any_offer": {"human_value": "Unknown", "note": "the edit cuts away"},
                 "exchange:1/topic": {"human_value": "not_an_exchange"},
                 "exchange:3/topic": {"human_value": "sales"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "1 outcome overrides, 0 first-decision overrides" in capsys.readouterr().out
    assert {n: sha(vdir / "results" / f"{n}.json") for n in hashes} == hashes
    primary = json.loads((vdir / "results" / "primary.json").read_text())
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": "unknown", "first_decision": None}
    assert reviewed["corrections"] == 1                                     # the topic of exchange 3
    assert reviewed["record"]["decisions"] == primary["record"]["decisions"]
    set_value(reviewed["record"], "exchange:3/topic", "customer_retention", books)
    assert reviewed["record"] == primary["record"]
    log = {e["path"]: e for e in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    assert [log["pitch:any_offer"][k] for k in ("resolution", "old", "new")] == ["outcome_override", "yes", "unknown"]
    assert log["pitch:any_offer"]["note"] == "the edit cuts away"
    assert log["pitch:any_offer"]["field_class"] == "outcome"
    gone = log["exchange:1/topic"]
    assert (gone["resolution"], gone["old"], gone["new"]) == ("structural", "operating_efficiency",
                                                              "operating_efficiency")
    assert "not an exchange" in gone["note"]

    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    est = agreement["sampled_estimates"]
    assert list(est) == ["outcome", "exchange_topic"]
    assert est["outcome"]["human_vs_primary"] == {"agree": 0.0, "n": 1} and est["outcome"]["population"] == 1
    low, high = est["outcome"]["human_vs_primary_ci95"]
    assert low == 0.0 < high < 1.0
    assert est["exchange_topic"]["n"] == 1 and est["exchange_topic"]["population"] == 2
    assert (agreement["n_structural"], agreement["n_outcome_overrides"]) == (1, 1)

    # analysis reads the override, not the decision list
    from analysis.panel_dataset import load_run, pitch_outcome
    o1 = pitch_outcome(load_run(run))
    assert o1.any_offer_source.iloc[0] == "human_override" and o1.any_offer.isna().iloc[0]

    # a later import that agrees with the record takes the override away again
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    assert json.loads((vdir / "results" / "reviewed.json").read_text())["pitch_override"] == {
        "any_offer": None, "first_decision": None}
    assert pitch_outcome(load_run(run)).any_offer_source.iloc[0] == "record"


def test_an_invalid_outcome_writes_nothing(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run, "--exchanges-per-video", "4", *OLD_SHEET) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "maybe"}, "exchange:3/topic": {"human_value": "sales"}})
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    err = capsys.readouterr().err
    assert "pitch:any_offer takes yes, no, unknown, not 'maybe'" in err and "nothing was written" in err
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before


# ---- sampling videos ---------------------------------------------------------------------------

def _run_of(tmp_path, books, n=6):
    for i in range(n):
        run, _ = make_run(tmp_path, books, video_id=f"vid{i}")
    make_run(tmp_path, books, video_id="solo", replicate=False)                 # one pass only: not eligible
    return run


def test_sample_videos_draws_a_recorded_seeded_sample_of_the_videos_with_both_passes(tmp_path, books, capsys):
    run = _run_of(tmp_path, books)
    assert default_export(run, "--sample-videos", "3", "--seed", "2") == 0
    out = capsys.readouterr().out
    record = json.loads((run / "review" / "video_sample.json").read_text())
    assert set(record) == {"draws", "chosen"} and len(record["draws"]) == 1
    draw = record["draws"][0]
    assert set(draw) == {"seed", "n", "excluded", "eligible", "chosen"}
    assert (draw["seed"], draw["n"]) == (2, 3) and draw["eligible"] == [f"vid{i}" for i in range(6)]
    chosen = record["chosen"]
    assert chosen == draw["chosen"] and len(chosen) == 3 and chosen == sorted(chosen) and set(chosen) < set(draw["eligible"])
    exported = sorted(p.parent.parent.name for p in run.glob("*/review/export_manifest.json"))
    assert exported == chosen and "video sample: 3 videos (seed 2)" in out
    for video_id in chosen:
        assert json.loads((run / video_id / "review" / "export_manifest.json").read_text())["videos"][video_id][
            "preset"] == "exchange"

    # the same draw again is the same sample; another one is refused unless forced
    assert default_export(run, "--sample-videos", "3", "--seed", "2", "--force") == 0
    assert json.loads((run / "review" / "video_sample.json").read_text()) == record
    capsys.readouterr()
    assert default_export(run, "--sample-videos", "4", "--seed", "2") == 1
    assert "video_sample.json" in capsys.readouterr().err
    assert default_export(run, "--sample-videos", "4", "--seed", "2", "--force") == 0
    assert len(json.loads((run / "review" / "video_sample.json").read_text())["chosen"]) == 4
    capsys.readouterr()
    assert default_export(run, "--sample-videos", "7", "--force") == 1
    assert "6 videos are eligible" in capsys.readouterr().err
    assert default_export(run, "--sample-videos", "2", "--video-ids", "vid1", "--force") == 1
    assert "--video-ids" in capsys.readouterr().err


def test_a_different_sample_is_refused_without_force(tmp_path, books, capsys):
    run = _run_of(tmp_path, books)
    assert default_export(run, "--sample-videos", "2", "--seed", "1") == 0
    record = (run / "review" / "video_sample.json").read_bytes()
    capsys.readouterr()
    assert default_export(run, "--sample-videos", "2", "--seed", "9") == 1
    assert "--force" in capsys.readouterr().err and (run / "review" / "video_sample.json").read_bytes() == record


def test_status_shows_the_sample_and_summary_pools_the_imported_videos(tmp_path, books, capsys):
    run = _run_of(tmp_path, books, n=4)
    assert default_export(run, "--sample-videos", "3", "--seed", "0", "--venture-per-video", "1", "--exchanges-per-video", "4",
                          *OLD_SHEET) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    for video_id, offer in zip(chosen[:2], ("yes", "no")):
        sheet = run / video_id / "review" / f"{video_id}__review.csv"
        fill(sheet, {"pitch:any_offer": {"human_value": offer},
                     "exchange:1/topic": {"human_value": "not_an_exchange"},
                     "exchange:3/topic": {"human_value": "customer_retention"},
                     "venture:founder_gender": {"human_value": "mixed"}})
        assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    assert rp.main(["status", "--run", str(run)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == f"video sample (1 draw, seed 0): 3 videos, 2 reviewed: {', '.join(chosen)}"
    for video_id in chosen:
        assert any(line.startswith(f"{video_id} [in sample]: sheet exported (18 rows)") for line in lines)
    assert any(line.startswith("solo: no sheet exported") for line in lines)
    assert sum("4 rows reviewed" in line for line in lines) == 2

    out_file = tmp_path / "reports" / "summary.md"
    assert rp.main(["summary", "--run", str(run), "--out", str(out_file)]) == 0
    printed = capsys.readouterr().out
    assert out_file.read_text() == printed
    assert "2 videos with an imported sheet, 8 reviewed rows; 2 structural notes, 1 outcome overrides" in printed
    table = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
             for line in printed.split("## Agreement per code")[0].splitlines()
             if line.startswith("| ") and "field class" not in line}
    assert table["outcome"][:3] == ["2", "2", "0.50 (1/2)"] and table["outcome"][5] == "1.00 (2/2)"
    assert table["exchange_topic"][:3] == ["2", "2", "1.00 (2/2)"]
    assert table["venture"][:3] == ["2", "2", "0.00 (0/2)"]
    low, high = (float(x) for x in table["outcome"][3].split("-"))
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert "Flagged and added rows" not in printed

    empty = tmp_path / "other"
    make_run(empty, books)
    assert rp.main(["summary", "--run", str(empty / "run")]) == 0
    assert "No sampled row has been reviewed." in capsys.readouterr().out


# ---- corrections that reach the analysis; tolerant agreement; hardened input -----------------------

def _import(run, sheet, *extra):
    return rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", *extra])


def _log(vdir):
    return [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]


def test_a_sheet_has_no_closing_rows_and_an_added_closing_row_is_refused(tmp_path, books, capsys):
    """Closing states were recorded up to v3-dev8 only: a v3-dev9 record gives no closing row, and a row a coder adds
    for one names a path the record does not have."""
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    assert not [r for r in rows if r["field_class"] in ("closing", "exchange_investor")]
    rows.append({**dict.fromkeys(COLUMNS, ""), "path": "exchange:3/investor:S2/closing", "human_value": "satisfied"})
    write_sheet(sheet, rows)
    capsys.readouterr()
    assert _import(run, sheet) == 1
    assert "no such path" in capsys.readouterr().err
    assert not (vdir / "results" / "reviewed.json").exists()


def test_import_checks_clocks_against_the_videos_duration_and_reads_one_path_however_it_is_spelled(
        tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"decision:1/time": {"human_value": "0:02:55"},        # a spreadsheet's rendering of 2:55
                 "prepared_window": {"human_value": "0:05-9:00"}})
    assert _import(run, sheet) == 1
    err = capsys.readouterr().err
    assert "H:MM:SS" in err and "beyond the video's duration (3:00)" in err and "nothing was written" in err
    assert not (vdir / "results" / "reviewed.json").exists()
    # the same field under two spellings is one field: reviewed twice is refused, and one spelling finds its row
    rows = read_sheet(sheet)
    for row in rows:
        row.update({"human_value": "", "resolution": "", "note": ""})
        if row["path"] == "turn:4/code:PR-SEV":
            row.update(path="turn:0004/code:PR-SEV", human_value="2")
    write_sheet(sheet, rows + [{**dict.fromkeys(COLUMNS, ""), "path": "turn:4/code:PR-SEV", "human_value": "1"}])
    assert _import(run, sheet) == 1
    assert "turn:4/code:PR-SEV is reviewed more than once" in capsys.readouterr().err
    write_sheet(sheet, rows)
    assert _import(run, sheet) == 0
    entry = _log(vdir)[0]
    assert (entry["path"], entry["primary"], entry["replicate"], entry["selected_as"]) == (
        "turn:4/code:PR-SEV", "3", "1", "flagged")           # matched to the exported row, not taken for an added one


def test_a_clock_within_tolerance_is_agreement_and_no_correction(tmp_path, books, capsys):
    def later(d):
        d["decisions"][0]["time_s"] = 140.0          # the replicate hears S3 leave 29 s later, at 2:20
    run, vdir = make_run(tmp_path, books, mutate=later)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"decision:1/time": {"human_value": "1:45"},                    # 6 s from the primary's 1:51
                 "decision:2/time": {"human_value": "2:41", "resolution": "correct"},   # asked for: written
                 "stage_a:pitch_positivity": {"human_value": "4"}})              # one point off: a correction
    assert _import(run, sheet) == 0
    assert "3 rows reviewed, 1 values changed" in capsys.readouterr().out
    result = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert result["corrections"] == 1 and result["record"]["decisions"][0]["time_s"] == 111.0
    assert result["record"]["decisions"][1]["time_s"] == 161.0        # written, though within tolerance
    by = {e["path"]: e for e in _log(vdir)}
    assert by["decision:1/time"]["resolution"] == "confirm_primary"
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["tolerances"] == {"decision_s": 30.0, "window_edge_s": 2.0, "cut_s": 5.0}
    decision = agreement["by_field_class"]["decision"]
    assert decision["human_vs_primary"] == {"agree": 1.0, "n": 2}
    # a value within tolerance is the primary's value (1:51), which is 29 s from the replicate's 2:20
    assert decision["human_vs_replicate"] == {"agree": 1.0, "n": 2}
    assert decision["primary_vs_replicate"] == {"agree": 1.0, "n": 2}
    stage_a = agreement["by_field_class"]["stage_a"]
    assert stage_a["human_vs_primary"] == {"agree": 0.0, "n": 1}
    assert stage_a["within_one"]["human_vs_primary"] == {"agree": 1.0, "n": 1}
    assert rp.main(["summary", "--run", str(run)]) == 0
    printed = capsys.readouterr().out
    table = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
             for line in printed.splitlines() if line.startswith("| ") and "field class" not in line}
    assert table["decision"][2] == "1.00 (2/2)" and table["stage_a"][2] == "0.00 (0/1)"
    assert table["stage_a"][6] == "1.00 (1/1)" and table["decision"][6] == "-"
    assert "within one" in printed and "within tolerance" in " ".join(printed.split())


def test_the_flag_driven_manifest_records_the_unflagged_pool_and_the_estimates_carry_it(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--preset", "flagged", "--sample-unflagged", "12"]) == 0
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    unflagged = entry["unflagged"]
    assert sum(unflagged["on_sheet"].values()) == 12 == entry["counts"]["by_selection"]["sampled"]
    assert sum(unflagged["population"].values()) > 60 and "proportional" in unflagged["note"]
    assert unflagged["excluded_no_counterpart"] == {"n": 0, "by_field_class": {}}
    page = " ".join((vdir / "review" / "vid1__review.md").read_text().split())
    assert "drawn in proportion to each field class" in page and "both passes recorded" in page
    sheet = vdir / "review" / "vid1__review.csv"
    sampled = next(r for r in read_sheet(sheet) if r["selected_as"] == "sampled")
    fill(sheet, {sampled["path"]: {"resolution": "confirm_primary"}})
    assert _import(run, sheet) == 0
    estimates = json.loads((vdir / "review" / "human_agreement.json").read_text())["sampled_estimates"]
    assert estimates[sampled["field_class"]]["population"] == unflagged["population"][sampled["field_class"]]
    # the exchange preset keeps its own record of the draw and has no unflagged pool
    assert default_export(run, "--force") == 0
    entry = json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["unflagged"] is None and entry["sample"]["population"]["cut"] == 1


def test_summary_intervals_resample_videos_from_five_and_say_so_below(tmp_path, books, capsys):
    run = _run_of(tmp_path, books, n=6)
    assert default_export(run, "--venture-per-video", "1", *OLD_SHEET) == 0

    def review(video_id, gender):
        sheet = run / video_id / "review" / f"{video_id}__review.csv"
        fill(sheet, {"pitch:any_offer": {"human_value": "yes"},
                     "venture:founder_gender": {"human_value": gender}})
        assert _import(run, sheet) == 0

    for video_id, gender in (("vid0", "all_men"), ("vid1", "mixed"), ("vid2", "all_men"), ("vid3", "all_men")):
        review(video_id, gender)
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0
    printed = capsys.readouterr().out
    assert "| Wilson; treats rows as independent; optimistic |" in printed
    assert "cluster bootstrap over videos (1,999 draws" not in printed
    for video_id, gender in (("vid4", "mixed"), ("vid5", "all_men")):
        review(video_id, gender)
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0
    printed = capsys.readouterr().out
    table = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:-1]]
             for line in printed.splitlines() if line.startswith("| ") and "field class" not in line}
    assert table["venture"][:3] == ["6", "6", "0.67 (4/6)"]
    assert table["venture"][7] == "cluster bootstrap over videos (1,999 draws, seed 0)"
    low, high = (float(x) for x in table["venture"][3].split("-"))
    assert 0.0 <= low < 0.67 < high <= 1.0
    assert rp.main(["summary", "--run", str(run)]) == 0 and capsys.readouterr().out == printed     # seeded


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_a_semicolon_or_tab_delimited_sheet_is_read(tmp_path, books, delimiter):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:4/code:PR-SEV": {"human_value": "2", "note": "a note, with a comma; and a semicolon"}})
    rows = read_sheet(sheet)
    with open(sheet, "w", newline="", encoding="utf-8") as fh:       # as a spreadsheet in another locale saves it
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)
    assert _import(run, sheet) == 0
    entry = _log(vdir)[0]
    assert (entry["path"], entry["new"], entry["note"]) == ("turn:4/code:PR-SEV", "2",
                                                            "a note, with a comma; and a semicolon")


def test_a_sheet_that_is_not_utf8_or_has_no_readable_columns_gets_one_clear_line(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:4/code:PR-SEV": {"human_value": "2", "note": "très clair"}})
    good = sheet.read_text(encoding="utf-8")
    sheet.write_bytes(good.encode("cp1252"))
    assert _import(run, sheet) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "is not UTF-8" in err and "CSV UTF-8" in err and sheet.name in err
    sheet.write_bytes(good.encode("utf-16"))
    assert _import(run, sheet) == 1 and "is not UTF-8" in capsys.readouterr().err
    sheet.write_text(good.replace(",", "|"), encoding="utf-8")
    assert _import(run, sheet) == 1
    err = capsys.readouterr().err.strip()
    assert len(err.splitlines()) == 1 and "no `path` column" in err and "comma, semicolon or tab" in err
    assert not (vdir / "results" / "reviewed.json").exists()
    sheet.write_bytes(("\ufeff" + good).encode("utf-8"))               # the mark a spreadsheet adds is fine
    assert _import(run, sheet) == 0


def test_import_writes_its_four_files_together_or_not_at_all(tmp_path, books, monkeypatch, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:4/code:PR-SEV": {"human_value": "2"}})

    def snapshot():
        return {str(p.relative_to(vdir)): p.read_bytes() for p in sorted(vdir.rglob("*")) if p.is_file()}

    def failing_after(n):
        real, calls = rp.os.replace, []

        def replace(src, dst):
            calls.append(dst)
            if len([c for c in calls if not str(c).endswith(rp.BACKUP_SUFFIX)]) > n:
                raise OSError("disk full")
            return real(src, dst)
        return replace

    before = snapshot()
    for n in (0, 1, 2, 3):               # the write fails before, or part way through, putting the files in place
        monkeypatch.setattr(rp.os, "replace", failing_after(n))
        with pytest.raises(OSError, match="disk full"):
            _import(run, sheet)
        monkeypatch.undo()
        assert snapshot() == before, n                 # nothing written, nothing half-written, no temporary file
    assert _import(run, sheet) == 0
    first = snapshot()
    assert {"results/reviewed.json", "results/reviewed_validation.json", "review/review_log.jsonl",
            "review/human_agreement.json"} == set(first) - set(before)
    # a second import that fails leaves the first one's files exactly as they were
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"}})
    filled = snapshot()
    monkeypatch.setattr(rp.os, "replace", failing_after(2))
    with pytest.raises(OSError, match="disk full"):
        _import(run, sheet)
    monkeypatch.undo()
    assert snapshot() == filled
    assert _import(run, sheet) == 0
    assert len(_log(vdir)) == 2 and not [p for p in vdir.rglob("*") if p.name.endswith((".tmp", rp.BACKUP_SUFFIX))]


def test_a_stage_a_value_changed_in_review_marks_its_note(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"},
                 "stage_a:pitch_positivity": {"human_value": "3"}})          # confirmed: its note stands
    assert _import(run, sheet) == 0
    stage_a = {i["code"]: i for i in json.loads((vdir / "results" / "reviewed.json").read_text())["record"]["stage_a"]}
    assert stage_a["interpersonal_warmth"]["note"] == "Steady delivery. [value changed in review]"
    assert stage_a["pitch_positivity"]["note"] == "Steady delivery."


def test_a_reviewed_record_with_open_issues_is_still_the_best_record_and_analysis_marks_it(tmp_path, books, capsys):
    """load_best_record keeps preferring the reviewed record; what it leaves open is counted from the same
    files, and the analysis tables carry the mark."""
    from analysis.panel_dataset import load_run
    from pipeline.stage_b.panel_pipeline import review_open_issues

    run, vdir = make_run(tmp_path, books)
    assert load_run(run).videos.source.tolist() == ["primary"]
    assert review_open_issues(vdir) == {"reviewed_validation_errors": 0, "review_structural_on_decisions": 0}
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"turn:2/function": {"resolution": "confirm_primary"}})
    assert _import(run, sheet) == 0
    assert load_run(run).videos.source.tolist() == ["reviewed"]            # a clean review is just reviewed
    # a correction the record takes but validation does not, and a judgment no correction can apply
    fill(sheet, {"decision:2/offer_id": {"human_value": "O9"},
                 "decision:3/kind": {"resolution": "structural", "note": "two decisions in one"}})
    capsys.readouterr()
    assert _import(run, sheet) == 0
    captured = capsys.readouterr()
    assert "validation: 2 errors" in captured.out
    assert "2 validation errors" in captured.err and "reviewed_with_open_issues" in captured.err
    errors = json.loads((vdir / "results" / "reviewed_validation.json").read_text())["errors"]
    assert len(errors) == 2 and all("O1" in e for e in errors)
    result, source = load_best_record(vdir)
    assert source == "reviewed" and result["record"]["decisions"][1]["offer_id"] == "O9"
    assert review_open_issues(vdir) == {"reviewed_validation_errors": 2, "review_structural_on_decisions": 1}
    data = load_run(run)
    video = data.videos.iloc[0]
    assert video.source == "reviewed_with_open_issues"
    assert (video.reviewed_validation_errors, video.review_structural_on_decisions) == (2, 1)
    assert data.decisions.set_index("decision_id").loc[2, "offer_id"] == "O9"       # the tables are the reviewed record
    # settled later: the structural judgment is replaced, the offer id restored, and the mark goes
    fill(sheet, {"decision:2/offer_id": {"human_value": "O1", "resolution": "correct"},
                 "decision:3/kind": {"resolution": "confirm_primary"}})
    assert _import(run, sheet) == 0
    assert review_open_issues(vdir) == {"reviewed_validation_errors": 0, "review_structural_on_decisions": 0}
    assert load_run(run).videos.source.tolist() == ["reviewed"]


def test_load_best_record_reads_the_replicate_when_the_primary_fails_validation_and_it_fails_less(tmp_path):
    results = tmp_path / "st_x" / "results"
    results.mkdir(parents=True)

    def write(name, errors):
        (results / f"{name}.json").write_text(json.dumps({"record": {"pass": name}}))
        (results / f"{name}_validation.json").write_text(json.dumps({"errors": ["e"] * errors, "flags": []}))

    write("primary", 2)
    assert load_best_record(tmp_path / "st_x")[1] == "primary"          # no replicate to turn to
    write("replicate", 2)
    assert load_best_record(tmp_path / "st_x")[1] == "primary"          # a tie keeps the primary
    write("replicate", 1)
    result, source = load_best_record(tmp_path / "st_x")
    assert source == "replicate" and result["record"] == {"pass": "replicate"}
    write("primary", 0)
    assert load_best_record(tmp_path / "st_x")[1] == "primary"          # a clean primary is always read
    (results / "reviewed.json").write_text(json.dumps({"record": {"pass": "reviewed"}}))
    assert load_best_record(tmp_path / "st_x")[1] == "reviewed"


def test_video_sample_excludes_named_videos_and_records_the_draw(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4"):
        make_run(tmp_path, books, v)
    (run / "selection.json").write_text(json.dumps({"pilot_development_videos": ["vid1"]}))
    assert rp.main(["export", "--run", str(run), "--sample-videos", "2", "--seed", "3", "--blind",
                    "--exclude-from", "selection.json"]) == 0
    sample = json.loads((run / "review" / "video_sample.json").read_text())
    assert "vid1" not in sample["chosen"] and len(sample["chosen"]) == 2
    assert sample["draws"][0]["excluded"] == ["vid1"] and sample["draws"][0]["seed"] == 3
    assert not (run / "vid1" / "review" / "export_manifest.json").exists()


def test_the_per_video_draw_follows_the_rows_per_field_target(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3"):
        make_run(tmp_path, books, v)
    # the plan covers the question-answer pairs; it never turns the exchange cards on (they are not drawn by default)
    assert rp.plan_draw(60, 30) == {"exchanges": 0, "qa": 2} and rp.plan_draw(60, 10) == {"exchanges": 0, "qa": 6}
    assert rp.plan_draw(60, 2) == {"exchanges": 0, "qa": 30} and rp.plan_draw(1, 100) == {"exchanges": 0, "qa": 1}
    assert rp.main(["export", "--run", str(run), "--blind", "--video-ids", "vid1", "vid2", "--rows-per-field", "6"]) == 0
    entry = json.loads((run / "vid1" / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["sample"]["planned"] == {"rows_per_field": 6, "n_videos": 2, "exchanges": 0, "qa": 3}
    assert entry["sample"]["requested"]["qa"] == 3 and entry["sample"]["requested"]["exchanges"] == 0
    # the fixture has two pairs before its first decision: the draw takes what exists and says so
    assert entry["sample"]["drawn"]["exchanges"] == [] and len(entry["sample"]["drawn"]["qa"]) == 2
    assert entry["sample"]["available"]["qa"] == 2
    assert rp.main(["export", "--run", str(run), "--blind", "--video-ids", "vid3", "--rows-per-field", "6",
                    "--qa-per-video", "1", "--exchanges-per-video", "5"]) == 0
    entry = json.loads((run / "vid3" / "review" / "export_manifest.json").read_text())["videos"]["vid3"]
    assert entry["sample"]["requested"]["qa"] == 1 and entry["sample"]["requested"]["exchanges"] == 5
    assert entry["sample"]["drawn"]["exchanges"] == [1, 3]          # asked for, so drawn (the fixture has two)
    # --qa-per-video overrides the plan, so no plan is recorded (final review I2)
    assert "planned" not in entry["sample"]
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "2", "--seed", "1", "--force",
                    "--qa-per-video", "4"]) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    entry = json.loads((run / chosen[0] / "review" / "export_manifest.json").read_text())["videos"][chosen[0]]
    assert entry["sample"]["requested"]["qa"] == 4 and "planned" not in entry["sample"]


def test_the_planned_draw_applies_only_when_asked_for_or_when_the_videos_are_drawn(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3"):
        make_run(tmp_path, books, v)
    assert rp.main(["export", "--run", str(run), "--blind", "--video-ids", "vid1"]) == 0
    entry = json.loads((run / "vid1" / "review" / "export_manifest.json").read_text())["videos"]["vid1"]
    assert entry["sample"]["requested"]["exchanges"] == 0 and entry["sample"]["requested"]["qa"] == 3
    assert "planned" not in entry["sample"]
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "2", "--seed", "1", "--force"]) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    entry = json.loads((run / chosen[0] / "review" / "export_manifest.json").read_text())["videos"][chosen[0]]
    assert entry["sample"]["planned"] == {"rows_per_field": 60, "n_videos": 2, "exchanges": 0, "qa": 30}


def test_the_draw_writes_a_spreadsheet_of_the_sampled_videos_and_copy_videos_hands_them_over(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3"):
        make_run(tmp_path, books, v)
    sources = tmp_path / "videos"
    sources.mkdir()
    for v in ("vid1", "vid2", "vid3"):          # make_run names /videos/<id>.mp4; point the experiments at real files
        (sources / f"{v}.mp4").write_bytes(b"fake video " + v.encode())
        exp = run / v / "experiment.json"
        exp.write_text(json.dumps({**json.loads(exp.read_text()), "source": str(sources / f"{v}.mp4")}))
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "2", "--seed", "5"]) == 0
    rows = read_sheet(run / "review" / "video_sample.csv")
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    assert [r["video_id"] for r in rows] == chosen
    assert rows[0]["file_name"] == f"{chosen[0]}.mp4" and rows[0]["title"] == chosen[0]
    assert rows[0]["duration_clock"] == "3:00" and rows[0]["seed"] == "5" and rows[0]["draw"] == "1"
    handover = tmp_path / "handover"
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(handover)]) == 0
    assert sorted(p.name for p in handover.iterdir()) == sorted(f"{v}.mp4" for v in chosen)
    (sources / f"{chosen[0]}.mp4").unlink()
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(tmp_path / "again")]) == 1
    assert (tmp_path / "again" / f"{chosen[1]}.mp4").exists()


def test_a_second_draw_appends_new_videos_without_touching_the_first(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4"):
        make_run(tmp_path, books, v)
    assert rp.main(["export", "--run", str(run), "--sample-videos", "2", "--seed", "1", "--blind"]) == 0
    first = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    assert rp.main(["export", "--run", str(run), "--sample-videos", "1", "--seed", "2", "--blind", "--append"]) == 0
    sample = json.loads((run / "review" / "video_sample.json").read_text())
    assert sample["chosen"][:2] == first and len(sample["chosen"]) == 3 and len(sample["draws"]) == 2
    assert sample["draws"][1]["chosen"][0] not in first
    assert rp.main(["export", "--run", str(run), "--sample-videos", "1", "--seed", "9", "--blind"]) == 1  # no --append, no --force


# ---- roles: a coder's sheet is logged and measured, an adjudicator's corrects the record ----

RESULTS = ("primary.json", "replicate.json", "reviewed.json", "reviewed_validation.json")


def snapshot(vdir):
    """The bytes of every result file, None for one that does not exist."""
    return {n: (vdir / "results" / n).read_bytes() if (vdir / "results" / n).exists() else None for n in RESULTS}


def log_of(vdir):
    return [json.loads(l) for l in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]


def test_a_blind_sheet_is_logged_as_a_coder_and_never_corrects_the_record(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--stage-a-per-video", "all"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"}, "pitch:any_offer": {"human_value": "no"}})
    before = snapshot(vdir)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert snapshot(vdir) == before and before["reviewed.json"] is None
    assert not (vdir / "results" / "reviewed.json").exists()
    lines = log_of(vdir)
    assert {l["role"] for l in lines} == {"coder"} and {l["reviewer"] for l in lines} == {"ana"}
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert "ana" in agreement["by_reviewer"]
    assert agreement["by_reviewer"]["ana"]["n_reviewed"] == agreement["n_reviewed"] == 2


def test_a_coder_import_is_validated_like_an_adjudicators_and_writes_nothing_when_a_value_is_refused(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "9"}})
    before = snapshot(vdir)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    assert snapshot(vdir) == before
    assert not (vdir / "review" / "review_log.jsonl").exists()
    assert not (vdir / "review" / "human_agreement.json").exists()


def test_a_coder_does_not_touch_an_existing_reviewed_record(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "evey",
                    "--role", "adjudicator"]) == 0
    before = snapshot(vdir)
    assert before["reviewed.json"] is not None
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "1"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert snapshot(vdir) == before
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["reviewer"] == "evey" and reviewed["corrections"] == 1


def test_an_adjudicator_writes_the_record_and_coders_stay_apart(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "2"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "evey", "--role", "adjudicator"]) == 0
    assert "exported blind" in capsys.readouterr().err
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    value = next(i["value"] for i in reviewed["record"]["stage_a"] if i["code"] == "interpersonal_warmth")
    assert value == 4 and reviewed["reviewer"] == "evey"
    lines = log_of(vdir)
    by = rp._lines_by_reviewer(lines)
    assert by["ana"]["stage_a:interpersonal_warmth"]["new"] == "5" and by["ben"]["stage_a:interpersonal_warmth"]["new"] == "2"
    assert rp._reviewed_lines(lines, role="adjudicator")["stage_a:interpersonal_warmth"]["new"] == "4"
    assert rp._reviewed_lines(lines, role="coder")["stage_a:interpersonal_warmth"]["new"] == "2"
    assert rp._reviewed_lines(lines)["stage_a:interpersonal_warmth"]["new"] == "4"
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert set(agreement["by_reviewer"]) == {"ana", "ben"}      # the adjudicator is no coder


def test_a_log_line_without_a_role_counts_as_an_adjudicator():
    lines = [{"path": "a", "new": "1", "reviewer": "x"}, {"path": "a", "new": "2", "reviewer": "y", "role": "coder"}]
    assert rp._reviewed_lines(lines, role="adjudicator")["a"]["new"] == "1"
    assert rp._reviewed_lines(lines, role="coder")["a"]["new"] == "2"


def test_a_non_blind_sheet_stays_an_adjudicators_by_default_and_a_role_can_be_given(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "4"}})
    before = snapshot(vdir)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana", "--role", "coder"]) == 0
    assert snapshot(vdir) == before
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    assert "exported blind" not in capsys.readouterr().err
    assert (vdir / "results" / "reviewed.json").exists()
    assert [l["role"] for l in log_of(vdir)][-1] == "adjudicator"


def test_a_row_the_manifest_does_not_list_is_logged_as_added_with_the_record_as_the_model_side(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--qa-codes", "question:PR-CONCEPT"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    rows.append({**{c: "" for c in COLUMNS}, "path": "turn:4/code:PR-SEV", "human_value": "2"})
    write_sheet(sheet, rows)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    added = next(l for l in log_of(vdir) if l["path"] == "turn:4/code:PR-SEV")
    assert added["selected_as"] == "added" and added["primary"] == "3" and added["new"] == "2"


def test_import_folder_takes_the_reviewer_from_the_file_name(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    folder = tmp_path / "drive"
    folder.mkdir()
    (folder / "vid1__ana.csv").write_bytes(sheet.read_bytes())
    (folder / "vid1__ben.csv").write_bytes(sheet.read_bytes())
    (folder / "notes.txt").write_text("ignored")
    before = snapshot(vdir)
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder)]) == 0
    assert {l["reviewer"] for l in log_of(vdir)} == {"ana", "ben"}
    assert snapshot(vdir) == before
    assert (vdir / "review" / "vid1__ana.csv").exists()


def test_import_folder_reports_a_video_without_a_manifest_entry_and_imports_the_rest(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    folder = tmp_path / "drive"
    folder.mkdir()
    (folder / "ghost__ana.csv").write_bytes(sheet.read_bytes())
    (folder / "vid1__ben.csv").write_bytes(sheet.read_bytes())
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder)]) == 1
    captured = capsys.readouterr()
    assert "ghost" in captured.err and "1 of 2 sheets imported" in captured.out
    assert {l["reviewer"] for l in log_of(vdir)} == {"ben"}
    assert not (run / "ghost").exists()


def test_status_counts_rows_per_reviewer(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    assert rp.main(["status", "--run", str(run)]) == 0
    assert "ana: 1" in capsys.readouterr().out


# ---- what measures validity: the coders, never the adjudicator ----

def _blind_run(tmp_path, books, *extra):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", "--stage-a-per-video", "all", *extra]) == 0
    return run, vdir, vdir / "review" / "vid1__review.csv"


def _import_as(run, sheet, reviewer, value, *extra, edits=None):
    fill(sheet, edits or {"stage_a:interpersonal_warmth": {"human_value": value}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", reviewer, *extra]) == 0


def _measures(run, vdir, capsys):
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0
    return {k: v for k, v in agreement.items() if k != "by_reviewer"}, capsys.readouterr().out


def test_an_adjudicator_changes_neither_the_validity_figures_nor_the_summary(tmp_path, books, capsys):
    run, vdir, sheet = _blind_run(tmp_path, books)
    _import_as(run, sheet, "ana", "5")
    _import_as(run, sheet, "ben", "2")
    two_coders = _measures(run, vdir, capsys)
    _import_as(run, sheet, "evey", "4", "--role", "adjudicator")
    assert _measures(run, vdir, capsys) == two_coders
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert set(agreement["by_reviewer"]) == {"ana", "ben"}
    assert agreement["n_reviewed"] == 2          # one item per coder for the one path


def test_the_validity_figures_do_not_depend_on_which_coder_imported_last(tmp_path, books, capsys):
    results = []
    for order in (("ana", "ben"), ("ben", "ana")):
        run, vdir, sheet = _blind_run(tmp_path / "-".join(order), books)
        for name in order:
            _import_as(run, sheet, name, {"ana": "5", "ben": "2"}[name])
        results.append(_measures(run, vdir, capsys))
    assert results[0] == results[1]


def test_a_person_who_codes_and_then_adjudicates_under_one_name_keeps_the_blind_figure(tmp_path, books):
    run, vdir, sheet = _blind_run(tmp_path, books)
    _import_as(run, sheet, "ana", "5")
    blind = json.loads((vdir / "review" / "human_agreement.json").read_text())["by_reviewer"]["ana"]
    _import_as(run, sheet, "ana", "3", "--role", "adjudicator")
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["by_reviewer"]["ana"] == blind


def test_a_coders_outcome_override_never_reaches_the_reviewed_record(tmp_path, books):
    run, vdir, sheet = _blind_run(tmp_path, books, *OLD_SHEET)
    record = json.loads((vdir / "results" / "primary.json").read_text())["record"]
    flipped = "no" if any_offer(record) != "no" else "yes"
    _import_as(run, sheet, "ana", None, edits={"pitch:any_offer": {"human_value": flipped}})
    assert json.loads(next(l for l in (vdir / "review" / "review_log.jsonl").read_text().splitlines()))["resolution"] \
        == "outcome_override"
    _import_as(run, sheet, "evey", "4", "--role", "adjudicator")
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": None}


def test_a_coders_line_neither_opens_nor_clears_an_issue_of_the_reviewed_record(tmp_path, books):
    from pipeline.stage_b.panel_pipeline import review_open_issues
    run, vdir = make_run(tmp_path, books)
    assert export(run, "--blind") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    # adjudicator confirms the decision; a coder then judges it structural: still no open issue
    _import_as(run, sheet, "evey", None, "--role", "adjudicator",
               edits={"decision:3/kind": {"resolution": "confirm_primary"}})
    reviewed = (vdir / "results" / "reviewed.json").read_bytes()
    _import_as(run, sheet, "ana", None, edits={"decision:3/kind": {"resolution": "structural", "note": "two"}})
    assert review_open_issues(vdir)["review_structural_on_decisions"] == 0
    assert (vdir / "results" / "reviewed.json").read_bytes() == reviewed
    # the other way: an adjudicator's open issue is not cleared by a later coder's confirmation
    _import_as(run, sheet, "evey", None, "--role", "adjudicator",
               edits={"decision:3/kind": {"resolution": "structural", "note": "two"}})
    assert review_open_issues(vdir)["review_structural_on_decisions"] == 1
    _import_as(run, sheet, "ben", None, edits={"decision:3/kind": {"resolution": "confirm_primary"}})
    assert review_open_issues(vdir)["review_structural_on_decisions"] == 1


def test_import_folder_skips_the_exported_sheet_of_the_review_directory(tmp_path, books, capsys):
    run, vdir, sheet = _blind_run(tmp_path, books)
    fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": "5"}})
    (vdir / "review" / "vid1__ana.csv").write_bytes(sheet.read_bytes())
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(vdir / "review")]) == 0
    captured = capsys.readouterr()
    assert "skipped: vid1__review.csv" in captured.err and "1 of 1 sheets imported" in captured.out
    assert {l["reviewer"] for l in log_of(vdir)} == {"ana"}


def test_summary_reports_each_coder_the_pair_and_the_structural_rate(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--exchanges-per-video", "4"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}, "exchange:1/topic": {"resolution": "structural", "note": "x"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    fill(sheet, {"pitch:any_offer": {"human_value": "no"}, "exchange:1/topic": {"human_value": "sales"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text(encoding="utf-8"))
    assert agreement["human_human"]["reviewers"] == ["ana", "ben"]
    assert agreement["human_human"]["n_both"] == 1 and agreement["human_human"]["n_dropped_structural"] == 1
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0
    out = capsys.readouterr().out
    assert "## Coders" in out and "ana" in out and "ben" in out
    assert "## Coder against coder" in out and "## Structural" in out
    assert "| outcome | 1 | 1 | 0.00 (0/1)" in out


def test_three_coders_on_one_video_are_reported_and_everything_else_is_still_written(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    for name, value in (("ana", "yes"), ("ben", "no"), ("cai", "yes")):
        fill(sheet, {"pitch:any_offer": {"human_value": value}})
        capsys.readouterr()
        assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", name]) == 0
    assert "more than two coders" not in capsys.readouterr().err        # reported by summary, not on every import
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text(encoding="utf-8"))
    assert set(agreement["by_reviewer"]) == {"ana", "ben", "cai"} and "human_human" not in agreement
    assert "more than two coders on vid1" in agreement["human_human_note"]
    assert rp.main(["summary", "--run", str(run)]) == 0
    captured = capsys.readouterr()
    assert "more than two coders on vid1: coder-against-coder is not computed" in captured.err
    assert "## Coder against coder" in captured.out and "No video qualifies" in captured.out


def test_an_imported_coder_line_records_what_was_typed(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--exchanges-per-video", "4"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": " yes "},
                 "exchange:1/topic": {"resolution": "unclear", "note": "cannot tell"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    log = [json.loads(x) for x in (vdir / "review" / "review_log.jsonl").read_text(encoding="utf-8").splitlines()]
    asked = {x["path"]: x for x in log if not x.get("implied")}
    assert asked["pitch:any_offer"]["typed"] == "yes"
    assert asked["exchange:1/topic"]["typed"] is None
    assert all("typed" not in x for x in log if x.get("implied"))       # implied lines carry no `typed` key


def test_the_thresholds_file_is_kept_as_the_record_and_read_by_nothing(books):
    """Plan D: validation_thresholds.yaml stays in git as the October 8 record; no script or pipeline module reads
    it, and the table names no gate."""
    release = load_release()
    assert release.validation_thresholds is not None and release.validation_thresholds.exists()
    assert not hasattr(rp, "load_thresholds") and not hasattr(rp, "check_thresholds")
    readers = [p for folder in ("scripts", "pipeline", "analysis") for p in (ROOT / folder).rglob("*.py")
               if "validation_thresholds" in p.read_text(encoding="utf-8") and p.name != "instrument.py"]
    assert readers == []
    assert "validation-spec" not in rp.parse_args.__code__.co_consts       # the freeze command is gone


def test_summary_paper_table_reads_the_reporting_file_and_refuses_one_it_cannot_apply(tmp_path, books, monkeypatch,
                                                                                    capsys):
    import dataclasses
    import yaml
    reporting = rp.load_reporting(books)
    assert reporting.source == "validation_reporting.yaml" and reporting.kind_of("PR-SEV") == "ordinal"
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    release = load_release()
    data = yaml.safe_load(release.validation_reporting.read_text())
    data["codes"]["PR-SEV"].pop("values")
    path = tmp_path / "reporting" / "validation_reporting.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    monkeypatch.setattr(rp, "load_release", lambda: dataclasses.replace(release, validation_reporting=path))
    capsys.readouterr()
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 1
    assert "PR-SEV" in capsys.readouterr().err and not out.exists()
    assert rp.main(["summary", "--run", str(run)]) == 0           # without the table the file is not read
    monkeypatch.setattr(rp, "load_release", lambda: dataclasses.replace(release, validation_reporting=None))
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 1
    assert "validation_reporting.yaml" in capsys.readouterr().err


def test_summary_paper_table_writes_three_files_with_no_gate(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(tmp_path / "t.md")]) == 0
    assert (tmp_path / "t.md").exists() and (tmp_path / "t.json").exists() and (tmp_path / "t_appendix.md").exists()
    data = json.loads((tmp_path / "t.json").read_text())
    assert set(data) == {"summary_rows", "rows", "human_human_rows", "pass_to_pass_rows", "clock_by_video", "reporting", "note",
                         "methods"}
    assert data["reporting"]["source"] == "validation_reporting.yaml" and data["reporting"]["release_id"]
    assert data["reporting"]["bands"] == "landis_koch" and "PR-CONCEPT" in data["reporting"]["not_reported"]
    text = (tmp_path / "t.md").read_text()
    assert "## Note on the statistics" in text and data["methods"] in text
    for name in ("t.md", "t_appendix.md"):
        written = (tmp_path / name).read_text()
        assert not re.search(r"\bMet\b", written) and "threshold" not in written.lower()
        assert "second draw" not in written.lower()
    printed = capsys.readouterr().out
    assert "paper table ->" in printed and "failed its" not in printed and "second draw" not in printed
    # the summary itself prints the per-kind table
    assert "## Agreement per code, coder against model (the validation table)" in printed
    assert ("| Field | Kind | Units (videos) | Agreement (α) | Band of α | Flags | Without unsure: point (n) |"
            in printed)
    # the table records which reporting file it was built from (final review M2)
    import hashlib
    from pipeline.instrument import load_release
    source = load_release().validation_reporting
    assert data["reporting"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def _coder_video(run, vdir, edits, reviewer="ana", *export_extra):
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--video-ids", vdir.name, *export_extra]) == 0
    sheet = vdir / "review" / f"{vdir.name}__review.csv"
    fill(sheet, edits)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", reviewer]) == 0


def test_paper_table_flags_are_printed_and_the_markdown_matches_the_json(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    # every question-answer code answered with its last allowed value, and the outcome against the record's
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--video-ids", "vid1", "--qa-per-video", "10"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    edits = {r["path"]: {"human_value": r["allowed_values"].split(" | ")[-1]}
             for r in read_sheet(sheet) if r["field_class"] == "turn_code"}
    fill(sheet, {"pitch:any_offer": {"human_value": "no"}, **edits})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    capsys.readouterr()
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    printed = capsys.readouterr().out
    data = json.loads(out.with_suffix(".json").read_text())
    flagged = [r for r in data["rows"] if r["flags"]]
    assert flagged and all(f"flagged: {r['field']} (" in printed for r in flagged)
    assert all("small_n" in r["flags"] for r in flagged)                  # one video: every row has few units
    assert "PR-CONCEPT" not in {r["field"] for r in data["rows"]} and "PR-FRAME" in {r["field"] for r in data["rows"]}
    row = next(r for r in data["rows"] if r["alpha"] and r["alpha"]["value"] is not None)
    cell = [l for l in out.read_text().splitlines() if l.startswith(f"| {row['field']} |")][0]
    assert f"(α {row['alpha']['value']:.2f}" in cell and f"{row['figures']['exact']['value']:.2f}" in cell
    assert row["band"] in cell


def test_paper_table_counts_outcome_and_first_decision_overrides_apart(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    _coder_video(run, vdir, {"pitch:any_offer": {"human_value": "no"}, "pitch:first_decision": {"human_value": "2:30"}})
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    note = json.loads(out.with_suffix(".json").read_text())["note"]
    assert (note["outcome_overrides"], note["cut_overrides"]) == (1, 1)
    assert ("1 coder answer to the outcome differs from the record's, and 1 first-decision clock falls more than 5 s "
            "from the record's; none of these changes the record.") in out.read_text()
    fields = [r["field"] for r in json.loads(out.with_suffix(".json").read_text())["rows"] if not r["missing"]]
    assert fields == ["cut", "outcome"] and not any("closing" in f for f in fields)
    rows = json.loads(out.with_suffix(".json").read_text())["rows"]
    assert rows[-1]["field"] == "pair" and rows[-1]["missing"]           # no clip session: the row says so


def test_paper_table_leaves_out_a_video_only_an_adjudicator_reviewed(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    _, vdir2 = make_run(tmp_path, books, video_id="vid2")
    _coder_video(run, vdir, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["export", "--run", str(run), *OLD_SHEET, "--video-ids", "vid2"]) == 0
    sheet = vdir2 / "review" / "vid2__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "lead"]) == 0
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    assert "1 further video was reviewed by an adjudicator only and is not in this table." in out.read_text()
    assert json.loads(out.with_suffix(".json").read_text())["note"]["videos"] == 1
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0           # the plain summary still counts both
    printed = capsys.readouterr().out
    assert "2 videos with an imported sheet" in printed
    # but its per-kind table is the validation table's, from the coder-coded video only (final review M3)
    per_kind = printed.split("## Agreement per code, coder against model (the validation table)")[1].split("\n## ")[0]
    assert "The 1 video only an adjudicator reviewed is left out, as in the paper table." in per_kind
    outcome = next(l for l in per_kind.splitlines() if l.startswith("| outcome |"))
    assert "| 1 (1) |" in outcome


def test_paper_table_without_a_coder_video_exits_one_and_writes_nothing(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "lead"]) == 0
    out = tmp_path / "t.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 1
    assert "coder" in capsys.readouterr().err and not out.exists() and not out.with_suffix(".json").exists()


def test_end_to_end_two_coders_to_paper_table(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4", "vid5", "vid6"):
        make_run(tmp_path, books, v)
    before = {p: sha(p) for p in run.glob("*/results/*.json")}
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--sample-videos", "5", "--seed", "4",
                    "--stage-a-per-video", "all", "--venture-per-video", "0"]) == 0
    for coder in ("ana", "ben"):
        assert rp.main(["page", "--run", str(run), "--coder", coder, "--sample"]) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    assert len(chosen) == 5
    drive = tmp_path / "drive"
    drive.mkdir()
    for v in chosen:
        sheet = run / v / "review" / f"{v}__review.csv"
        for coder, value in (("ana", "4"), ("ben", "5")):
            fill(sheet, {"stage_a:interpersonal_warmth": {"human_value": value}, "pitch:any_offer": {"human_value": "yes"}})
            (drive / f"{v}__{coder}.csv").write_bytes(sheet.read_bytes())
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(drive)]) == 0
    # coder imports are measured, never written into the record
    assert not list(run.glob("*/results/reviewed.json"))
    out = run / "review" / "paper_table.md"
    assert rp.main(["summary", "--run", str(run), "--paper-table", str(out)]) == 0
    saved = json.loads((run / "review" / "paper_table.json").read_text())
    table = saved["rows"]
    fields = {r["field"]: r for r in table}
    assert fields["outcome"]["videos"] == 5
    # one Stage A item, one unit per video over 5 videos: the coders' disagreement (4 against 5 in every video) is
    # their row of coder against coder, raw agreement 0 and within one 1, ana one point under ben
    hh = {r["field"]: r for r in saved["human_human_rows"]}["interpersonal_warmth"]
    assert hh["raters"] == ["ana", "ben"] and hh["n"] == 5
    assert hh["figures"]["exact"]["value"] == 0 and hh["figures"]["within_one"]["value"] == 1
    assert hh["figures"]["mean_difference"]["value"] == -1
    from analysis.panel_explorer import human_validation
    frame = human_validation(run)
    assert list(frame["field"]) == [r["field"] for r in table]
    assert human_validation(tmp_path / "nowhere") is None
    # the model's passes are untouched by the whole flow
    assert {p: sha(p) for p in run.glob("*/results/*.json")} == before


# ---- the final fix wave: what a coder can answer, who a sheet belongs to, what a draw and a freeze record ----

def whole_video(d):
    """No decision: the full-video sheet draws its question-answer pairs from the whole video, all four of the fixture's
    (with its decisions only the two before the first one, plan C Task 7f)."""
    d["decisions"] = []


def _coder_sheet(tmp_path, books, *extra, mutate_primary=None):
    run, vdir = make_run(tmp_path, books, mutate_primary=mutate_primary)
    assert rp.main(["export", "--run", str(run), "--blind", "--qa-per-video", "10", "--stage-a-per-video", "2",
                    *extra]) == 0
    return run, vdir, vdir / "review" / "vid1__review.csv"


def test_a_code_judged_not_rateable_imports_as_missing_with_its_reason(tmp_path, books, capsys):
    run, vdir, sheet = _coder_sheet(tmp_path, books, mutate_primary=whole_video)
    subs = [r["path"] for r in read_sheet(sheet) if r["path"].endswith("/code:PREP-SUBS")]
    fill(sheet, {subs[0]: {"human_value": "missing:too_short"}, subs[1]: {"human_value": "MISSING:Ambiguous"},
                 subs[2]: {"human_value": "2"}})
    before = snapshot(vdir)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert snapshot(vdir) == before
    lines = {l["path"]: l for l in log_of(vdir)}
    assert lines[subs[0]]["new"] == "missing:too_short" and lines[subs[1]]["new"] == "missing:ambiguous"
    assert all(lines[p]["level"] == "ordinal" for p in subs[:3])
    # a number against "not rateable" is a disagreement in the share and no unit of the alpha
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["by_field_class"]["turn_code"]["human_vs_primary"]["n"] == 3
    fill(sheet, {subs[0]: {"human_value": "missing:none"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 1
    assert "is not a missing reason" in capsys.readouterr().err
    stage_a = next(r["path"] for r in read_sheet(sheet) if r["path"].startswith("stage_a:"))
    fill(sheet, {stage_a: {"human_value": "missing:too_short"}})          # a Stage A item takes no missing reason
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 1
    assert "is not an integer" in capsys.readouterr().err


def test_a_coder_import_skips_a_note_without_an_answer_and_an_adjudicators_refuses_it(tmp_path, books, capsys):
    run, vdir, sheet = _coder_sheet(tmp_path, books, *OLD_SHEET)
    paths = [r["path"] for r in read_sheet(sheet) if r["path"].startswith("turn:")]
    edits = {"pitch:any_offer": {"human_value": "yes"},
             paths[0]: {"note": "[summary shown]"}, paths[1]: {"note": "come back to this"}}
    fill(sheet, edits)
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    out = capsys.readouterr().out
    assert "2 rows with a note but no answer were skipped" in out and "1 rows reviewed" in out
    assert [l["path"] for l in log_of(vdir)] == ["pitch:any_offer"]
    # a sheet with nothing skipped says nothing of it
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    assert "skipped" not in capsys.readouterr().out
    # a sheet of notes only still imports nothing
    fill(sheet, {paths[0]: {"note": "[summary shown]"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "cy"]) == 1
    assert "no row of the sheet is reviewed" in capsys.readouterr().err
    # an adjudicator's sheet is refused whole, as before
    fill(sheet, edits)
    lines = len(log_of(vdir))
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "lead", "--role", "adjudicator"]) == 1
    err = capsys.readouterr().err
    assert err.count("a note without human_value or resolution is not a review") == 2 and "nothing was written" in err
    assert len(log_of(vdir)) == lines and not (vdir / "results" / "reviewed.json").exists()


def test_import_folder_imports_as_a_coder_whatever_the_manifest_says(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), *OLD_SHEET, "--preset", "exchange"]) == 0             # not blind
    sheet = vdir / "review" / "vid1__review.csv"
    assert json.loads((vdir / "review" / "export_manifest.json").read_text())["videos"]["vid1"]["blind"] is False
    fill(sheet, {"pitch:any_offer": {"human_value": "no"}})
    folder = tmp_path / "drive"
    folder.mkdir()
    (folder / "vid1__ana.csv").write_bytes(sheet.read_bytes())
    before = snapshot(vdir)
    capsys.readouterr()
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder)]) == 0
    out = capsys.readouterr().out
    assert snapshot(vdir) == before and not (vdir / "results" / "reviewed.json").exists()
    assert [l["role"] for l in log_of(vdir)] == ["coder"]
    assert "ana as coder" in out and "1 of 1 sheets imported" in out and out.rstrip().endswith("as coder")
    # only the explicit role makes a folder of sheets an adjudicator's
    (folder / "vid1__ana.csv").unlink()
    (folder / "vid1__lead.csv").write_bytes(sheet.read_bytes())
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder), "--role", "adjudicator"]) == 0
    assert (vdir / "results" / "reviewed.json").exists() and log_of(vdir)[-1]["role"] == "adjudicator"


def test_a_duplicated_download_is_not_a_third_coder(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    folder = tmp_path / "drive"
    folder.mkdir()
    for name in ("vid1__coder1.csv", "vid1__coder1 (1).csv", "vid1__coder-2_b.csv"):
        (folder / name).write_bytes(sheet.read_bytes())
    capsys.readouterr()
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder)]) == 1
    captured = capsys.readouterr()
    assert ("problem: vid1__coder1 (1).csv: 'coder1 (1)' is not a coder name (letters, digits, - and _ only); a browser's "
            "duplicate download adds ' (1)': delete the duplicate") in captured.err
    assert "2 of 3 sheets imported" in captured.out
    assert {l["reviewer"] for l in log_of(vdir)} == {"coder1", "coder-2_b"}
    assert not (vdir / "review" / "vid1__coder1 (1).csv").exists()         # not even staged beside the manifest
    # the same rule for a single import
    for name in ("coder1 (1)", "ana ben", "ana,ben", ""):
        assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", name]) == 1
        assert "is not a coder name (letters, digits, - and _ only)" in capsys.readouterr().err
    assert {l["reviewer"] for l in log_of(vdir)} == {"coder1", "coder-2_b"}
    assert rp.reviewer_name("Coder_1-b") == "Coder_1-b"
    with pytest.raises(rp.ConfigError):
        rp.reviewer_name("coder1\n")


def _sample_file(run):
    return run / "review" / "video_sample.json"


def test_an_append_run_again_returns_its_draw_and_draws_nothing_new(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4", "vid5", "vid6"):
        make_run(tmp_path, books, v)
    first = ["export", "--run", str(run), "--sample-videos", "2", "--seed", "1", "--blind"]
    second = ["export", "--run", str(run), "--sample-videos", "2", "--seed", "2", "--blind", "--append"]
    assert rp.main(first) == 0 and rp.main(second) == 0
    recorded = _sample_file(run).read_bytes()
    sample = json.loads(recorded)
    assert len(sample["chosen"]) == 4 and len(sample["draws"]) == 2
    sheet_csv = (run / "review" / "video_sample.csv").read_bytes()
    capsys.readouterr()
    # the append again (with --force, which a re-export of existing sheets needs): the same two videos, no third draw
    assert rp.main([*second, "--force"]) == 0
    out = capsys.readouterr().out
    assert _sample_file(run).read_bytes() == recorded and (run / "review" / "video_sample.csv").read_bytes() == sheet_csv
    assert "draw 2 of" in out and "nothing new is drawn" in out
    assert sorted(l.split(":")[0] for l in out.splitlines() if "preset exchange" in l) == sorted(sample["draws"][1]["chosen"])
    # without --force the recorded draw is still found; it is the existing sheets that stop the export
    assert rp.main(second) == 1
    assert "pass --force to replace it" in capsys.readouterr().err and _sample_file(run).read_bytes() == recorded
    # the first draw's own arguments are that draw, not a different sample to refuse
    assert rp.sample_videos(run, 2, 1, False) == sample["draws"][0]["chosen"]
    assert rp.sample_videos(run, 2, 1, False, append=True) == sample["draws"][0]["chosen"]
    assert _sample_file(run).read_bytes() == recorded
    # another seed, another n or another exclusion is a new draw: appended, or refused without --append
    with pytest.raises(rp.ConfigError, match="--append adds a draw"):
        rp.sample_videos(run, 2, 9, False)
    assert len(rp.sample_videos(run, 1, 2, False, append=True)) == 1
    assert len(json.loads(_sample_file(run).read_text())["draws"]) == 3
    assert len(rp.sample_videos(run, 1, 2, False, exclude={"vid1"}, append=True)) == 1
    assert len(json.loads(_sample_file(run).read_text())["draws"]) == 4


def test_the_first_draws_command_is_accepted_after_an_append(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4"):
        make_run(tmp_path, books, v)
    first = ["export", "--run", str(run), "--sample-videos", "2", "--seed", "1", "--blind"]
    assert rp.main(first) == 0
    assert rp.main(["export", "--run", str(run), "--sample-videos", "1", "--seed", "2", "--blind", "--append"]) == 0
    recorded = _sample_file(run).read_bytes()
    chosen = json.loads(recorded)["draws"][0]["chosen"]
    for v in chosen:                                          # so that the export has no sheet to protect
        (run / v / "review" / f"{v}__review.csv").unlink()
    capsys.readouterr()
    assert rp.main(first) == 0
    out = capsys.readouterr().out
    assert "draw 1 of" in out and _sample_file(run).read_bytes() == recorded
    assert sorted(l.split(":")[0] for l in out.splitlines() if "preset exchange" in l) == sorted(chosen)


def test_exclude_from_refuses_a_file_without_the_key_and_takes_another_key(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4"):
        make_run(tmp_path, books, v)
    (run / "selection.json").write_text(json.dumps({"development": ["vid1"], "seed": 3}))
    draw = ["export", "--run", str(run), "--sample-videos", "2", "--seed", "3", "--blind", "--exclude-from", "selection.json"]
    assert rp.main(draw) == 1
    err = capsys.readouterr().err
    assert "'pilot_development_videos'" in err and "its keys: development, seed" in err
    assert not _sample_file(run).exists() and not list(run.glob("*/review"))
    assert rp.main([*draw, "--exclude-key", "development"]) == 0
    sample = json.loads(_sample_file(run).read_text())
    assert sample["draws"][0]["excluded"] == ["vid1"] and "vid1" not in sample["chosen"]


def test_every_log_line_says_when_it_was_imported_and_at_what_level_alpha_reads_it(tmp_path, books):
    import datetime
    run, vdir = make_run(tmp_path, books)
    assert export(run, "--field-classes", "turn_code", "turn_function", "decision", "exchange_topic", "stage_a", "outcome") == 0
    sheet = vdir / "review" / "vid1__review.csv"
    edits = {"turn:4/code:PR-SEV": {"human_value": "2"}, "turn:4/code:PR-CONCEPT": {"resolution": "confirm_primary"},
             "turn:5/code:RS-EVID": {"resolution": "confirm_primary"}, "decision:1/time": {"resolution": "confirm_primary"},
             "decision:1/kind": {"resolution": "confirm_primary"}, "exchange:1/label": {"human_value": "the cost of a customer"},
             "exchange:1/topic": {"resolution": "unclear", "note": "two topics"},
             "stage_a:interpersonal_warmth": {"human_value": "4"}, "pitch:any_offer": {"human_value": "yes"},
             "turn:14/function": {"human_value": "challenges"}}
    fill(sheet, edits)
    start = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "lead"]) == 0
    lines = log_of(vdir)
    stamps = {l["imported_at"] for l in lines}
    assert len(stamps) == 1                                                  # one import, one time, on every line
    when = datetime.datetime.fromisoformat(stamps.pop())
    assert when.tzinfo is not None and when.utcoffset() == datetime.timedelta(0)
    assert start <= when <= datetime.datetime.now(datetime.timezone.utc)
    asked = {l["path"]: l for l in lines if not l["implied"]}
    assert {p: l["level"] for p, l in asked.items()} == {
        "turn:4/code:PR-SEV": "ordinal", "turn:4/code:PR-CONCEPT": "nominal", "turn:5/code:RS-EVID": "nominal",
        "decision:1/time": "interval", "decision:1/kind": "nominal", "exchange:1/label": None,
        "exchange:1/topic": "nominal", "stage_a:interpersonal_warmth": "ordinal", "pitch:any_offer": "nominal",
        "turn:14/function": "nominal"}
    implied = [l for l in lines if l["implied"]]
    assert implied and all("level" not in l and l["imported_at"] for l in implied)   # the codes a function change removed


def test_an_ordinal_code_of_any_id_is_logged_and_measured_as_ordinal(tmp_path, books, monkeypatch, capsys):
    """An instrument whose severity code is called QX-PRESS: the level comes from its entry, not from its id."""
    from dataclasses import replace
    from pipeline.codebook import Codebook

    def rename(d):
        for t in d["turns"]:
            for c in t["codes"]:
                if c["code"] == "PR-SEV":
                    c["code"] = "QX-PRESS"

    def replicate(d):
        rename(d)
        d["turns"][3]["codes"][1]["value"] = "1"

    def primary(d):
        rename(d)
        whole_video(d)
    other = replace(books, panel=Codebook(entries=[
        e.model_copy(update={"id": "QX-PRESS"}) if e.id == "PR-SEV" else e for e in books.panel.entries]))
    monkeypatch.setattr(rp, "load_panel_books", lambda release: other)
    videos = [f"vid{i}" for i in range(1, 7)]
    for v in videos:
        run, _ = make_run(tmp_path, other, v, mutate=replicate, mutate_primary=primary)
    assert rp.main(["export", "--run", str(run), "--blind", "--qa-per-video", "10", "--qa-codes", "question:QX-PRESS"]) == 0
    for i, v in enumerate(videos):
        sheet = run / v / "review" / f"{v}__review.csv"
        paths = [r["path"] for r in read_sheet(sheet) if r["path"].endswith("/code:QX-PRESS")]
        assert len(paths) >= 3
        fill(sheet, {p: {"human_value": str(1 + (i + k) % 3)} for k, p in enumerate(paths)})
        assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
        assert {l["level"] for l in log_of(run / v) if l["path"] in paths} == {"ordinal"}
    with pytest.raises(rp.ConfigError, match="PR-SEV"):                # the reporting file names PR-SEV
        rp.load_reporting(other)
    capsys.readouterr()
    assert rp.main(["summary", "--run", str(run)]) == 0                 # the plain summary, without the per-kind table
    assert "left out of the summary" in capsys.readouterr().err
    lines = {v: rp.validity_lines(rp._log_lines(run / v / "review" / "review_log.jsonl")) for v in videos}
    figure = rp.pooled_summary(lines)["by_variable"]["QX-PRESS"]
    assert figure["alpha_level"] == "ordinal" and figure["alpha"] is not None and "within_one" in figure
    stripped = {v: [{k: x for k, x in l.items() if k != "level"} for l in ls] for v, ls in lines.items()}
    by_path = rp.pooled_summary(stripped)["by_variable"]["QX-PRESS"]            # an older log: the path rule
    assert by_path["alpha_level"] == "nominal" and by_path["alpha"] != figure["alpha"]


def test_copy_videos_reports_a_collision_and_a_file_it_cannot_copy_and_copies_the_rest(tmp_path, books, monkeypatch, capsys):
    import shutil
    run, _ = make_run(tmp_path, books, "vid1")
    for v in ("vid2", "vid3", "vid4"):
        make_run(tmp_path, books, v)
    sources = tmp_path / "videos"
    sources.mkdir()
    for v in ("vid1", "vid2", "vid3", "vid4"):
        (sources / f"{v}.mp4").write_bytes(b"video of " + v.encode())
        exp = run / v / "experiment.json"
        exp.write_text(json.dumps({**json.loads(exp.read_text()), "source": str(sources / f"{v}.mp4")}))
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "4", "--seed", "5"]) == 0
    target = tmp_path / "handover"
    target.mkdir()
    (target / "vid1.mp4").write_bytes(b"another file of this name")              # another size: a collision
    (target / "vid2.mp4").write_bytes(b"video of vid2")                          # the same size: already there
    real = shutil.copy2

    def copy(source, dest):
        if Path(source).name == "vid3.mp4":
            raise OSError(28, "No space left on device")
        return real(source, dest)
    monkeypatch.setattr(rp.shutil, "copy2", copy)
    capsys.readouterr()
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(target)]) == 1
    captured = capsys.readouterr()
    assert "2 of 4 videos in" in captured.out                                    # vid2 (there) and vid4 (copied)
    assert "failed: vid1: collision:" in captured.err and "exists with another size" in captured.err
    assert "failed: vid3: vid3.mp4:" in captured.err and "No space left on device" in captured.err
    assert (target / "vid1.mp4").read_bytes() == b"another file of this name"    # left as it was
    assert (target / "vid4.mp4").read_bytes() == b"video of vid4" and not (target / "vid3.mp4").exists()


def test_the_sample_spreadsheet_holds_no_path(tmp_path, books):
    run, _ = make_run(tmp_path, books, "vid1")
    make_run(tmp_path, books, "vid2")
    assert rp.main(["export", "--run", str(run), "--blind", "--sample-videos", "2", "--seed", "5"]) == 0
    raw = (run / "review" / "video_sample.csv").read_bytes().decode("utf-8-sig")
    assert raw.splitlines()[0] == "video_id,title,file_name,duration_s,duration_clock,draw,seed"
    assert rp.SAMPLE_COLUMNS == ["video_id", "title", "file_name", "duration_s", "duration_clock", "draw", "seed"]
    assert "/videos/" not in raw and "/" not in raw                       # make_run's sources are /videos/<id>.mp4
    rows = read_sheet(run / "review" / "video_sample.csv")
    assert rows[0]["file_name"] == "vid1.mp4" and rows[0]["title"] == "vid1" and rows[0]["duration_clock"] == "3:00"
    # written again from the recorded sample, without a draw
    sample = json.loads(_sample_file(run).read_text())
    assert rp.write_sample_sheet(run, sample).read_bytes().decode("utf-8-sig") == raw


def test_an_import_says_how_many_rows_were_not_on_the_exported_sheet(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind", *OLD_SHEET, "--qa-codes", "question:PR-CONCEPT"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "not on the exported sheet" not in capsys.readouterr().out
    rows = read_sheet(sheet)
    rows.append({**{c: "" for c in COLUMNS}, "path": "turn:4/code:PR-SEV", "human_value": "2"})
    rows.append({**{c: "" for c in COLUMNS}, "path": "turn:5/code:PREP-SUBS", "human_value": "missing:too_short"})
    write_sheet(sheet, rows)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    assert "vid1: 2 rows are not on the exported sheet and were logged as added" in capsys.readouterr().out


def _as_v3_dev8(vdir, release_id):
    """Rewrite the video's passes into the v3-dev8 shape (exchanges list investors with closing states, no prepared
    presentation codes) and record release_id in experiment.json (removed when None)."""
    for name in ("primary", "replicate"):
        path = vdir / "results" / f"{name}.json"
        result = json.loads(path.read_text())
        for e in result["record"]["exchanges"]:
            e["investors"] = [{"actor": "S1", "closing": "NONE", "closing_turn": None}]
        del result["record"]["prepared"]
        path.write_text(json.dumps(result))
    path = vdir / "experiment.json"
    experiment = json.loads(path.read_text())
    experiment.pop("release_id")
    path.write_text(json.dumps({**experiment, **({"release_id": release_id} if release_id else {})}))


@pytest.mark.parametrize("release_id, named", [("v3-dev8", "release v3-dev8"), (None, "an earlier release")])
def test_a_v3_dev8_record_is_refused_with_its_release_and_nothing_is_written(tmp_path, books, capsys, release_id,
                                                                            named):
    run, vdir = make_run(tmp_path, books)
    _as_v3_dev8(vdir, release_id)
    capsys.readouterr()
    assert export(run) == 1
    err = capsys.readouterr().err
    assert f"was annotated with {named} and is not a v3-dev9 panel record (ValidationError)" in err
    assert "the review tooling reads v3-dev9 records only" in err and "until plan C" not in err
    assert "validation error" not in err.lower()            # one line, not pydantic's error listing
    assert not (vdir / "review").exists()


def test_a_malformed_record_of_the_live_release_is_refused_without_naming_an_earlier_release(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    path = vdir / "results" / "primary.json"
    result = json.loads(path.read_text())
    result["record"]["decisions"][0]["time_s"] = "soon"                 # not a number: fails the schema
    path.write_text(json.dumps(result))
    capsys.readouterr()
    assert export(run) == 1
    err = capsys.readouterr().err
    assert f"{path} fails the {load_release().release_id} panel record schema (ValidationError" in err
    assert "decisions.0.time_s" in err                      # where the first problem is
    assert "annotated with" not in err and "earlier release" not in err
    assert len(err.strip().splitlines()) == 1 and not (vdir / "review").exists()


def test_an_old_review_log_line_on_an_exchange_topic_is_still_applied_after_the_cards_were_dropped(tmp_path, books):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run, "--exchanges-per-video", "2") == 0         # an adjudication sheet with topic rows
    sheet = vdir / "review" / "vid1__review.csv"
    row = {r["path"]: r for r in read_sheet(sheet)}["exchange:1/topic"]
    topic = next(v.strip() for v in row["allowed_values"].split("|") if v.strip() not in (row["primary"], "not_an_exchange"))
    fill(sheet, {"exchange:1/topic": {"human_value": topic}})
    assert _import(run, sheet) == 0
    assert default_export(run, "--force") == 0                            # today's sheet: no exchange rows
    sheet_rows = read_sheet(sheet)
    assert not [r for r in sheet_rows if r["field_class"] == "exchange_topic"]
    fill(sheet, {"pitch:first_decision": {"human_value": "1:51"}})
    assert _import(run, sheet) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert values(reviewed["record"], books)["exchange:1/topic"] == topic   # the earlier correction still stands
    assert [e["path"] for e in _log(vdir)].count("exchange:1/topic") == 1
    # what the second import recomputes from the whole log still reads the old line: the record's correction count,
    # and the agreement file's exchange_topic class (the second sheet holds no exchange row at all)
    assert reviewed["corrections"] == 1
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert agreement["n_reviewed"] == 2 and agreement["by_field_class"]["exchange_topic"]["n"] == 1


def test_the_readme_documents_every_review_subcommand_and_flag(capsys):
    """Every subcommand of `--help` is written as a command of its own (`import` is not satisfied by `import-folder`),
    and every flag any subcommand's `--help` lists is named in the section."""
    with pytest.raises(SystemExit):
        rp.parse_args(["--help"])
    names = re.search(r"\{([a-z,-]+)\}", capsys.readouterr().out).group(1).split(",")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("#### Human review", 1)[1].split("\n### ", 1)[0]
    flags = set()
    for name in names:
        assert re.search(rf"review_panel\.py {re.escape(name)}(?![\w-])", section), name
        with pytest.raises(SystemExit):
            rp.parse_args([name, "--help"])
        flags |= set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", capsys.readouterr().out)) - {"--help"}
    missing = sorted(f for f in flags if not re.search(rf"{re.escape(f)}(?![\w-])", section))
    assert not missing, missing
    for usage in ("--training", "--clips", "--paper-table", "reads `v3-dev9` records only", "A validation round, in order"):
        assert usage in section, usage
    assert "closing" not in section and "about 25-30 rows" not in section


# ---- the pitch-level fields -------------------------------------------------------------------------

def _with_rows(sheet, *added, answers=None):
    """The sheet with every answer blanked but `answers` ({path: {column: value}}) and rows a coder added
    (path, human_value) at its end; the number of its last row."""
    rows = read_sheet(sheet)
    for row in rows:
        row.update({"human_value": "", "resolution": "", "note": ""})
        row.update((answers or {}).get(row["path"], {}))
    blank = dict.fromkeys(COLUMNS, "")
    rows += [{**blank, "row": str(len(rows) + i), "path": path, "human_value": value}
             for i, (path, value) in enumerate(added, 1)]
    write_sheet(sheet, rows)
    return len(rows)


def test_ask_value_round_trips_and_refuses_junk(tmp_path, books, capsys):
    """Review Focus 1: digits and separators are normalised, anything else is a row-numbered problem."""
    run, vdir = make_run(tmp_path, books)
    assert default_export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    last = _with_rows(sheet, ("venture:ask", "$250,000;12.5%"))
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    venture = json.loads((vdir / "results" / "reviewed.json").read_text())["record"]["venture"]
    assert (venture["ask_amount_usd"], venture["ask_equity_pct"], venture["ask_missing"]) == (250000, 12.5, "none")
    log = [json.loads(line) for line in (vdir / "review" / "review_log.jsonl").read_text().splitlines()]
    assert (log[-1]["old"], log[-1]["new"], log[-1]["level"]) == ("200000;10", "250000;12.5", None)

    fill(sheet, {"venture:ask": {"human_value": "two hundred"}})        # the row the coder added, retyped
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    texts = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    err = capsys.readouterr().err
    assert f"row {last}: venture:ask: 'two hundred' is not amount;equity" in err and "nothing was written" in err
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before
    assert {p: p.read_bytes() for p in run.rglob("*") if p.is_file()} == texts


def test_old_investor_outcome_rows_are_refused_as_no_such_path(tmp_path, books, capsys):
    """Review Focus 5: a sheet of the old instrument (one outcome row per investor) on a v3-dev9 record."""
    run, vdir = make_run(tmp_path, books)
    assert default_export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    last = _with_rows(sheet, ("investor:S1/ever_offer", "yes"), ("investor:S3/ever_offer", "no"))
    before = sorted(p.relative_to(run).as_posix() for p in run.rglob("*"))
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 1
    err = capsys.readouterr().err
    assert f"row {last - 1}: investor:S1/ever_offer: no such path" in err
    assert f"row {last}: investor:S3/ever_offer: no such path" in err and "nothing was written" in err
    assert sorted(p.relative_to(run).as_posix() for p in run.rglob("*")) == before


def test_reviewed_json_carries_the_pitch_override(tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run, *OLD_SHEET) == 0             # an old sheet, with the outcome row
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    assert [r["path"] for r in rows if r["field_class"] == "outcome"] == ["pitch:any_offer"]
    assert [r["path"] for r in rows if r["field_class"] == "cut"] == ["pitch:first_decision"]
    _with_rows(sheet, answers={"pitch:any_offer": {"human_value": "No", "note": "the offer is a joke"},
                               "pitch:first_decision": {"human_value": "2:30"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "1 outcome overrides, 1 first-decision overrides" in capsys.readouterr().out
    primary = json.loads((vdir / "results" / "primary.json").read_text())
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert "outcome_overrides" not in reviewed
    assert reviewed["pitch_override"] == {"any_offer": "no", "first_decision": "2:30"}
    assert reviewed["record"] == primary["record"]                      # judged, never written
    log = {e["path"]: e for e in map(json.loads, (vdir / "review" / "review_log.jsonl").read_text().splitlines())}
    assert [log["pitch:any_offer"][k] for k in ("old", "new", "resolution", "field_class")] == [
        "yes", "no", "outcome_override", "outcome"]
    assert [log["pitch:first_decision"][k] for k in ("old", "new", "field_class", "level")] == [
        "1:51", "2:30", "cut", "interval"]

    # analysis reads the pitch verdict; the cut stays the record's (the clock is for the paper table only)
    from analysis.panel_dataset import load_run, pitch_cut, pitch_outcome
    data = load_run(run)
    o1 = pitch_outcome(data)
    assert (o1.any_offer.iloc[0], o1.any_offer_source.iloc[0]) == (0.0, "human_override")
    assert pitch_cut(data).cut_pos.iloc[0] == 9                         # turn 10, S3's exit, at position 9

    # a later import that agrees with the record takes the override away again
    fill(sheet, {"pitch:any_offer": {"human_value": "yes"}, "pitch:first_decision": {"human_value": "1:53"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": None}


def test_a_clock_only_override_is_counted_apart_from_the_outcome_and_no_decision_is_stored_as_none(
        tmp_path, books, capsys):
    run, vdir = make_run(tmp_path, books)
    assert default_export(run) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    _with_rows(sheet, answers={"pitch:any_offer": {"human_value": "yes"}, "pitch:first_decision": {"human_value": "2:30"}})
    capsys.readouterr()
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    assert "0 outcome overrides, 1 first-decision overrides" in capsys.readouterr().out
    agreement = json.loads((vdir / "review" / "human_agreement.json").read_text())
    assert (agreement["n_outcome_overrides"], agreement["n_cut_overrides"]) == (0, 1)
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": "2:30"}
    assert rp.main(["summary", "--run", str(run)]) == 0
    assert "0 outcome overrides, 1 first-decision overrides" in capsys.readouterr().out

    # "the footage shows no decision" is an override of its own, apart from no override at all
    fill(sheet, {"pitch:first_decision": {"resolution": "correct"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": "none"}
    fill(sheet, {"pitch:first_decision": {"human_value": "1:52"}})
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    reviewed = json.loads((vdir / "results" / "reviewed.json").read_text())
    assert reviewed["pitch_override"] == {"any_offer": None, "first_decision": None}


def test_an_other_concept_without_a_name_imports_with_a_warning(tmp_path, books, capsys):
    """The coding page asks for the matter's name in the note when a concept is `other`; a sheet without one still
    imports (old sheets must), with a warning naming the row, and the note is kept as typed."""
    run, vdir = make_run(tmp_path, books, mutate_primary=whole_video)
    assert rp.main(["export", "--run", str(run), "--blind", "--qa-per-video", "10"]) == 0
    sheet = vdir / "review" / "vid1__review.csv"
    rows = read_sheet(sheet)
    concepts = [r for r in rows if r["path"].endswith("/code:PR-CONCEPT")]
    assert "Choosing `other` for a concept needs a note" in (vdir / "review" / "vid1__review.md").read_text()
    fill(sheet, {concepts[0]["path"]: {"human_value": "other"},
                 concepts[1]["path"]: {"human_value": "other", "note": "franchise fees"},
                 concepts[2]["path"]: {"human_value": "other", "note": "[summary shown]"}})
    capsys.readouterr()
    assert _import(run, sheet) == 0
    out = capsys.readouterr().out
    warned = [line for line in out.splitlines() if "warning" in line]
    assert len(warned) == 2 and all("is `other` but the note does not name the matter" in w for w in warned)
    assert f"row {concepts[0]['row']}: {concepts[0]['path']}" in out and concepts[1]["path"] not in out
    log = {e["path"]: e for e in _log(vdir)}
    assert log[concepts[1]["path"]]["new"] == "other" and log[concepts[1]["path"]]["note"] == "franchise fees"
    assert log[concepts[0]["path"]]["new"] == "other"
