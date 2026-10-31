"""The final review's I1 (2026-10-22): the coders' sheets, the clip session's cards and the cut are drawn from the pass
the analysis reads (`panel_pipeline.load_best_record`'s rule: the replicate when the primary has validation errors
and the replicate has fewer), and the manifests record which pass."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.schemas.stage_b_panel import PanelRecord  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.panel_pipeline import load_best_record, model_pass  # noqa: E402
from pipeline.stage_b.review_items import canonical_full, read_value  # noqa: E402
from tests.test_review_cli import make_run, read_sheet, sha, write_sheet  # noqa: E402
from tests.test_review_clips import scores_for  # noqa: E402

VIDEOS = ["v00", "v01"]


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def _replicate_differs(d):
    d["turns"][3]["codes"][1]["value"] = "1"        # turn 4 PR-SEV: 3 in the primary (before the cut)
    d["stage_a"][2]["value"] = 5                    # interpersonal_warmth: 3 in the primary
    d["actors"][2]["label"] = "Investor Bee"        # S2: "Investor B" in the primary
    d["decisions"][0]["time_s"] = 112.0             # the first decision (the cut): 1:51 in the primary


def _primary_fails(vdir):
    (vdir / "results" / "primary_validation.json").write_text(json.dumps({"errors": ["broken"], "flags": []}))


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, books):
    base = tmp_path_factory.mktemp("analysed_pass")
    for v in VIDEOS:
        _, vdir = make_run(base, books, video_id=v, mutate=_replicate_differs)
        _primary_fails(vdir)
    run = base / "run"
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books, VIDEOS)))
    return run


@pytest.fixture
def run(base_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def _canonical(run, video_id, name, books):
    d = json.loads((run / video_id / "results" / f"{name}.json").read_text())["record"]
    return canonical_full(PanelRecord.model_validate(d), books).model_dump(mode="json")


def test_model_pass_is_the_rule_of_load_best_record(run):
    assert model_pass(run / "v00") == "replicate" == load_best_record(run / "v00")[1]
    (run / "v00" / "results" / "primary_validation.json").write_text(json.dumps({"errors": [], "flags": []}))
    assert model_pass(run / "v00") == "primary" == load_best_record(run / "v00")[1]


def test_the_blind_sheet_and_its_cut_are_drawn_from_the_analysed_pass(run, books, capsys):
    assert rp.main(["export", "--run", str(run), "--video-ids", "v00", "--blind", "--qa-per-video", "9",
                    "--seed", "1"]) == 0
    entry = json.loads((run / "v00" / "review" / "export_manifest.json").read_text())["videos"]["v00"]
    assert entry["pass"] == "replicate"
    replicate, primary = _canonical(run, "v00", "replicate", books), _canonical(run, "v00", "primary", books)
    by_path = {row["path"]: row for row in entry["rows"]}
    for row in entry["rows"]:
        # `primary` holds the analysed pass's value, `replicate` the other pass's
        assert row["primary"] == read_value(replicate, row["path"]), row["path"]
    assert by_path["pitch:first_decision"]["primary"] == "1:52"            # the replicate's cut, not the primary's
    assert by_path["turn:4/code:PR-SEV"]["primary"] == "1" and by_path["turn:4/code:PR-SEV"]["replicate"] == "3"
    assert read_value(primary, "turn:4/code:PR-SEV") == "3"


def test_a_coder_sheet_of_the_analysed_pass_imports_and_is_refused_when_that_pass_changed(run, books, capsys):
    assert rp.main(["export", "--run", str(run), "--video-ids", "v00", "--blind", "--seed", "1"]) == 0
    sheet = run / "v00" / "review" / "v00__review.csv"
    rows = read_sheet(sheet)
    for row in rows:
        if row["path"] == "pitch:first_decision":
            row["human_value"] = "1:52"
    write_sheet(sheet, rows)
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ana"]) == 0
    line = [json.loads(x) for x in (run / "v00" / "review" / "review_log.jsonl").read_text().splitlines()][-1]
    assert line["primary"] == "1:52"
    # the analysed pass changed since the export: refused, naming it
    replicate = run / "v00" / "results" / "replicate.json"
    replicate.write_text(replicate.read_text() + " ")
    assert rp.main(["import", "--run", str(run), "--sheet", str(sheet), "--reviewer", "ben"]) == 1
    assert "replicate" in capsys.readouterr().err


def test_the_session_cards_and_the_task_page_read_the_analysed_pass(run, books):
    assert rp.main(["session-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", "ana",
                    "--seed", "4"]) == 0
    manifest = json.loads((run / "review" / "clips" / "export_manifest.json").read_text())
    assert manifest["passes"] == {v: "replicate" for v in VIDEOS}
    assert manifest["pass_sha256"] == {v: sha(run / v / "results" / "replicate.json") for v in VIDEOS}
    warmth = next(r for r in manifest["rows"] if r["video_id"] == "v00" and r["path"] == "stage_a:interpersonal_warmth")
    assert (warmth["primary"], warmth["replicate"]) == ("5", "3")
    page = json.loads((run / "review" / "pages" / "ana__tasks.html").read_text()
                      .split("const PAGE = ", 1)[1].split(";\n", 1)[0])
    task = next(t for t in page["videos"] if t.get("task") == "video")
    assert task["actors"]["S2"] == "Investor Bee"                          # the replicate's people
    # a clip-session sheet of the analysed pass imports; a changed pass is refused
    sheet = run / "review" / "clips" / "clips.csv"
    rows = read_sheet(sheet)
    for row in rows:
        if row["path"] == "stage_a:interpersonal_warmth":
            row["human_value"] = "4"
    out = run / "clips__ana.csv"
    write_sheet(out, rows)
    assert rp.main(["clips-import", "--run", str(run), "--sheet", str(out), "--reviewer", "ana"]) == 0
