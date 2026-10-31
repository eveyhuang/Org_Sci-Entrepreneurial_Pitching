"""Training pages with the codebook on them, and the calibration of the coders on their training sheets."""
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
from pipeline.stage_b import review_page  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from pipeline.stage_b.review_codebook import book_title, codebook_model  # noqa: E402
from pipeline.stage_b.review_labels import load_labels  # noqa: E402
from tests.test_review_cli import fill, make_run, read_sheet  # noqa: E402


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


def page_model_of(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(re.search(r"const PAGE = (.*);\n", text).group(1).replace("<\\/", "</"))


def tree_digest(run: Path) -> dict[str, str]:
    """Every file of the run outside review/training, with its hash: what training and calibration must not touch."""
    training = run / "review" / "training"
    return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(run.rglob("*")) if p.is_file() and training not in p.parents}


def training_run(tmp_path, books, n=4):
    run, _ = make_run(tmp_path, books, "vid1")
    for i in range(2, n + 1):
        make_run(tmp_path, books, f"vid{i}")
    (run / "selection.json").write_text(json.dumps({"pilot_development_videos": ["vid1", "vid2", "vid3"]}))
    return run


# ---- the codebook on the page --------------------------------------------------------------------------

def test_the_codebook_lays_out_every_entry_of_every_book_in_the_release_order(books):
    release = load_release()
    model = codebook_model(release, books, load_labels(release), on_sheet={"PR-SEV", "expressed_enthusiasm"})
    assert [b["key"] for b in model] == list(release.books)
    assert [b["title"] for b in model] == [book_title(ref.markdown, key) for key, ref in release.books.items()]
    assert all(not b["title"].startswith("#") for b in model)
    ids = [e["id"] for b in model for g in b["groups"] for e in g["entries"]]
    assert sorted(ids) == sorted(["protocol", *books.panel.ids, *books.stage_a.ids, *books.venture.ids])
    entries = {e["id"]: e for b in model for g in b["groups"] for e in g["entries"]}
    sev = entries["PR-SEV"]
    assert sev["on_sheet"] and not entries["PR-CONCEPT"]["on_sheet"]
    assert "Understanding-seeking" in sev["definition_html"] and sev["include_html"] and sev["exclude_html"]
    assert "EX-CLOSE" not in entries                                      # no closing state since v3-dev9
    function = entries["TURN-FUNC"]
    assert [v["value"] for v in function["values"]] == list(books.turn_functions())
    assert "accepts" in function["search"] and function["search"] == function["search"].lower()
    assert any(v["gloss"] for v in function["values"])
    # the groups are each entry's own dimension, in the book's order
    panel = next(b for b in model if b["key"] == "stage_b_panel")
    assert [g["title"] for g in panel["groups"]] == list(dict.fromkeys(e.dimension for e in books.panel.entries))
    # notes written for the researcher stay off the page
    text = json.dumps(model)
    assert "provenance" not in text
    some = next(e for e in books.panel.entries if e.provenance)
    assert some.provenance[:60] not in text


def test_the_stage_a_block_opens_with_the_shared_coding_protocol(books):
    release = load_release()
    stage_a = next(b for b in codebook_model(release, books, load_labels(release)) if b["key"] == "stage_a")
    first = stage_a["groups"][0]["entries"][0]
    assert first["id"] == "protocol" and first["name"] == "Shared coding protocol" and not first["on_sheet"]
    assert first["values"] == [] and first["examples"] == [] and first["include_html"] == first["exclude_html"] == ""
    text = (release.templates_dir / "stage_a_perceived_delivery_protocol.md").read_text()
    paragraphs = [" ".join(p.split()) for p in text.split("\n\n") if p.strip() and not p.startswith("# ")]
    assert first["definition_html"].count("<p>") == len(paragraphs) >= 5
    assert "<p>Watch the entire supplied clip" in first["definition_html"] and "# Shared" not in first["definition_html"]
    assert "protocol" in first["search"] and "clip" in first["search"]
    assert [e["id"] for e in stage_a["groups"][0]["entries"][1:]] == [e.id for e in books.stage_a.entries
                                                                       if e.dimension == books.stage_a.entries[0].dimension]
    # on the sheet only when a Stage A card is
    marked = codebook_model(release, books, load_labels(release), on_sheet={"pitch_positivity"})
    assert next(b for b in marked if b["key"] == "stage_a")["groups"][0]["entries"][0]["on_sheet"]
    other = codebook_model(release, books, load_labels(release), on_sheet={"PR-SEV"})
    assert not next(b for b in other if b["key"] == "stage_a")["groups"][0]["entries"][0]["on_sheet"]
    # the other books get no protocol entry
    assert all(e["id"] != "protocol" for b in marked if b["key"] != "stage_a" for g in b["groups"] for e in g["entries"])


def test_a_release_without_the_protocol_file_gets_no_protocol_entry(books, tmp_path):
    import dataclasses
    release = dataclasses.replace(load_release(), templates_dir=tmp_path)
    ids = [e["id"] for b in codebook_model(release, books, load_labels(release)) for g in b["groups"] for e in g["entries"]]
    assert "protocol" not in ids


def test_book_title_falls_back_when_the_markdown_is_missing(tmp_path):
    assert book_title(tmp_path / "nowhere.md", "stage_a") == "stage_a"
    (tmp_path / "b.md").write_text("intro\n# The Book — v2\n## A\n")
    assert book_title(tmp_path / "b.md", "x") == "The Book — v2"


# ---- the training command -------------------------------------------------------------------------------

def test_training_writes_pages_with_the_codebook_and_a_model_answers_page_outside_the_validation(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    before = tree_digest(run)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--from", "selection.json",
                    "--videos", "2", "--seed", "1"]) == 0
    tdir = run / "review" / "training"
    ana, ben = page_model_of(tdir / "pages" / "ana.html"), page_model_of(tdir / "pages" / "ben.html")
    chosen = sorted(v["video_id"] for v in ana["videos"])
    assert len(chosen) == 2 and set(chosen) <= {"vid1", "vid2", "vid3"}
    assert sorted(v["video_id"] for v in ben["videos"]) == chosen
    assert ana["mode"] == "training" and ana["heading"] == "Training sheets" and ana["banner"]
    assert "<title>Training sheets · ana</title>" in (tdir / "pages" / "ana.html").read_text()
    # the codebook is on the page, and the entries the cards ask about are marked
    cards = [c for v in ana["videos"] for s in v["sections"] for c in s["cards"]]
    asked = {c["entry"] for c in cards if c["entry"]}
    marked = {e["id"] for b in ana["codebook"] for g in b["groups"] for e in g["entries"] if e["on_sheet"]}
    assert asked and marked - {"protocol"} == asked
    assert "protocol" in marked                                           # the page has Stage A cards
    # the model's answers for the walkthrough: the same cards, each with the model's side, read-only
    answers = page_model_of(tdir / "model_answers.html")
    assert answers["mode"] == "walkthrough" and answers["heading"] == "Model answers" and answers["codebook"]
    assert "Do not give this page to a coder" in answers["banner"]
    assert sorted(v["video_id"] for v in answers["videos"]) == chosen
    manifest = json.loads((tdir / "export_manifest.json").read_text())["videos"]
    for vd in answers["videos"]:
        primary = {r["path"]: r["primary"] for r in manifest[vd["video_id"]]["rows"]}
        model_cards = [c for s in vd["sections"] for c in s["cards"]]
        assert len(model_cards) == len(primary)
        assert all(c["model"]["answer"] == primary[c["path"]] for c in model_cards)
        assert all(c["model"]["reason"] for c in model_cards if c["kind"] in ("stage_a", "outcome"))
    assert not (tdir / "model_values.csv").exists() and not (tdir / "pages" / "model_answers.html").exists()
    # the training videos as a spreadsheet, the columns of the validation sample's
    with open(tdir / "training_videos.csv", newline="", encoding="utf-8-sig") as fh:
        listed = list(csv.DictReader(fh))
    assert list(listed[0]) == rp.SAMPLE_COLUMNS and [r["video_id"] for r in listed] == chosen
    assert all(r["file_name"] == f"{r['video_id']}.mp4" and r["duration_clock"] == "3:00" for r in listed)
    assert {r["draw"] for r in listed} == {"training"} and {r["seed"] for r in listed} == {"1"}
    # nothing outside review/training changed: the run's videos have no review folder, the results are untouched
    assert tree_digest(run) == before
    out = capsys.readouterr().out
    assert "do not give it to the coders" in out and "model_answers.html" in out and "training videos:" in out


def test_training_refuses_a_video_of_the_validation_sample_and_a_selection_without_the_key(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    (run / "review").mkdir(exist_ok=True)
    (run / "review" / "video_sample.json").write_text(json.dumps({"draws": [], "chosen": ["vid2"]}))
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "--video-ids", "vid1", "vid2"]) == 1
    assert "vid2 is in the validation sample" in capsys.readouterr().err
    (run / "other.json").write_text(json.dumps({"videos": ["vid1"]}))
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "--from", "other.json"]) == 1
    assert "no key 'pilot_development_videos'" in capsys.readouterr().err
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ana", "--video-ids", "vid1"]) == 1
    assert rp.main(["training", "--run", str(run), "--coders", "ana (1)", "--video-ids", "vid1"]) == 1
    assert not (run / "review" / "training" / "pages").exists()


def test_a_validation_page_carries_the_codebook_unless_asked_not_to(tmp_path, books):
    run, _ = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind"]) == 0
    assert rp.main(["page", "--run", str(run), "--coder", "ana", "--video-ids", "vid1"]) == 0
    page = page_model_of(run / "review" / "pages" / "ana.html")
    assert page["codebook"] and page["mode"] == "validation" and page["heading"] == "Coding sheets"
    assert page["banner"] == "" and all("model" not in c for v in page["videos"] for s in v["sections"] for c in s["cards"])
    assert rp.main(["page", "--run", str(run), "--coder", "ana", "--video-ids", "vid1", "--no-codebook"]) == 0
    assert page_model_of(run / "review" / "pages" / "ana.html")["codebook"] == []


def test_the_template_keeps_training_answers_apart_and_names_no_role_in_its_codebook_view():
    template = review_page.TEMPLATE.read_text(encoding="utf-8")
    assert "PAGE.mode && PAGE.mode !== 'validation' ? PAGE.mode : 'coding'" in template   # a separate storage key
    assert "function modelCard" in template and "PAGE.mode === 'walkthrough' ? modelCard(vd, c) : card(vd, c)" in template
    for needle in ("showCodebook", "cbsearch", "open in the codebook", "Back to the cards", "PAGE.banner"):
        assert needle in template
    chrome = template.split("<script>", 1)[1].lower()
    for word in ("shark", "founder", "investor"):
        assert word not in chrome
    with pytest.raises(review_page.PageError, match="mode"):
        review_page.page_model(Path("."), [], "ana", None, None, mode="exam")


# ---- calibration ---------------------------------------------------------------------------------------

def _fill_training(tdir: Path, drop: Path, coder: str, value_of, clock: str | None = None) -> None:
    """Each training sheet filled by `value_of` on the rows with listed values (and, with `clock`, the first-decision
    clock typed as that)."""
    for sheet in sorted(tdir.glob("*__review.csv")):
        video_id = sheet.name.split("__")[0]
        rows = read_sheet(sheet)
        edits = {r["path"]: {"human_value": value_of(r)} for r in rows if r["allowed_values"]}
        if clock:
            edits["pitch:first_decision"] = {"human_value": clock}
        target = drop / f"{video_id}__{coder}.csv"
        target.write_bytes(sheet.read_bytes())
        fill(target, edits)


def test_calibrate_reports_the_coders_agreement_without_touching_the_run(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2"]) == 0
    tdir = run / "review" / "training"
    drop = tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    # the coders disagree on the first decision only (the sheets carry no outcome card since Task 7d)
    _fill_training(tdir, drop, "ana", first, clock="1:50")
    _fill_training(tdir, drop, "ben", first, clock="2:40")
    before = tree_digest(run)
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop), "--bar", "0.67"]) == 0
    out = capsys.readouterr().out
    data = json.loads((tdir / "calibration.json").read_text())
    assert data["coders"] == ["ana", "ben"] and data["videos"] == ["vid1", "vid2"] and data["bar"] == 0.67
    rows = {(r["scope"], r["field"]): r for r in data["rows"]}
    # plan D: the per-code figures of the paper table's coder-against-coder table, the first decision in seconds
    assert {scope for scope, _ in rows} == {"code"}
    assert rows[("code", "cut")]["agree"] == 0.0                          # every first decision far apart (50 s)
    assert rows[("code", "cut")]["bar"] is None and rows[("code", "cut")]["reached"] is None      # no alpha
    assert ("code", "outcome") not in rows and rows[("code", "PR-FRAME")]["kind"] == "categorical"
    assert all(r["agree"] == 1.0 for (scope, field), r in rows.items() if field != "cut")
    assert rows[("code", "RS-EVID")]["bar"] == 0.67 and rows[("code", "RS-EVID")]["units"] == 4
    text = (tdir / "calibration.md").read_text()
    assert "# Calibration of ana and ben" in text and "## Coder against coder" in out
    assert ("| Field | Kind | Units (videos) | Agreement (α) | Band of α | Flags | Without unsure: point (n) "
            "| α ≥ 0.67 |") in text
    assert ("| cut | clock | 2 (2) | within 5 s 0.00 (0/2) [—]; median 50.0 s | — | small n (2 units) | 0.00 (2) "
            "| — |") in text
    assert "| not estimable |" in text and "Field classes" not in text
    assert tree_digest(run) == before                                    # the run's logs and results are untouched
    assert not any((run / v / "review").exists() for v in ("vid1", "vid2"))


def test_calibrate_without_a_reporting_file_gives_the_pooled_alpha_per_class(tmp_path, books, capsys, monkeypatch):
    """A refused validation_reporting.yaml: calibrate carries on with the pooled summary's alpha per field class and
    field, against the "substantial" floor, and says why."""
    from pipeline.errors import ConfigError
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2"]) == 0
    tdir, drop = run / "review" / "training", tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    _fill_training(tdir, drop, "ana", first, clock="1:50")
    _fill_training(tdir, drop, "ben", first, clock="2:40")

    def refused(books):
        raise ConfigError("validation_reporting.yaml: a code the instrument lacks")
    monkeypatch.setattr(rp, "load_reporting", refused)
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 0
    assert "a code the instrument lacks" in capsys.readouterr().err
    data = json.loads((tdir / "calibration.json").read_text())
    assert data["bar"] == 0.61 and {r["scope"] for r in data["rows"]} == {"class", "variable"}
    assert "Field classes" in (tdir / "calibration.md").read_text()


def test_calibrate_needs_a_training_export_and_some_sheet(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(empty)]) == 1
    assert "run `review_panel.py training` first" in capsys.readouterr().err
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "--video-ids", "vid1"]) == 0
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(empty)]) == 1


def test_calibration_rows_read_the_reported_alpha_against_the_bar(books):
    """Plan D: with the reporting, one row per code as the paper table's coder-against-coder table gives it, its alpha
    read against the bar at two decimals; a kind without alpha has no bar."""
    reporting = rp.load_reporting(books)
    hh = {"RS-RESP": {"kind": "categorical", "n": 30, "units": 30, "videos": 6, "flags": [],
                      "figures": {"exact": {"value": 0.9, "low": 0.8, "high": 1.0},
                                  "alpha": {"value": 0.607, "low": 0.4, "high": 0.8}}},
          "PR-SEV": {"kind": "ordinal", "n": 30, "units": 30, "videos": 6, "flags": [],
                     "figures": {"exact": {"value": 0.5, "low": None, "high": None},
                                 "within_one": {"value": 0.9, "low": None, "high": None},
                                 "alpha": {"value": 0.5, "low": None, "high": None}}},
          "cut": {"kind": "clock", "n": 6, "units": 6, "videos": 6, "flags": ["small_n"],
                  "figures": {"share_within_5": {"value": 0.5, "low": None, "high": None},
                              "median_abs": {"value": 4.0, "low": None, "high": None}},
                  "counts": {"n": 6, "within_5": 3}}}
    pooled = {"reporting": {"human_human": hh}}
    rows = {r["field"]: r for r in rp.calibration_rows(pooled, 0.61, reporting)}
    assert rows["RS-RESP"]["reached"] is True and rows["RS-RESP"]["band"] == "substantial"   # 0.607 prints 0.61
    assert rows["PR-SEV"]["reached"] is False and rows["PR-SEV"]["within_one"] == 0.9
    assert rows["cut"]["bar"] is None and rows["cut"]["reached"] is None and rows["cut"]["agree"] == 0.5
    text = rp.calibration_markdown(list(rows.values()), ["ana", "ben"], 6, 0.61,
                                   rp.calibration_table(pooled, reporting))
    lines = {line.split(" | ")[0]: line for line in text.splitlines() if line.startswith("| ")}
    assert lines["| RS-RESP"].endswith("| yes |") and lines["| PR-SEV"].endswith("| not yet |")
    assert lines["| cut"].endswith("| — |") and "α ≥ 0.61 |" in lines["| Field"]


def test_calibration_rows_judge_alpha_against_the_bar():
    pooled = {"human_human": {"closing": {"videos": 2, "n": 8, "agree": 0.75, "alpha": 0.66999999, "alpha_units": 8},
                              "outcome": {"videos": 2, "n": 6, "agree": 1.0, "alpha": None, "alpha_units": 0},
                              "by_variable": {"PR-SEV": {"videos": 2, "n": 4, "agree": 0.5, "alpha": 0.2}}}}
    rows = {r["field"]: r for r in rp.calibration_rows(pooled, 0.67)}
    assert rows["closing"]["reached"] is True                             # 0.66999999 rounds to the bar
    # the bar is read at the two decimals the band reads: 0.607 prints 0.61, "substantial", and reaches 0.61
    near = rp.calibration_rows({"human_human": {"x": {"alpha": 0.607}}}, 0.61)
    assert near[0]["reached"] is True
    assert rows["outcome"]["reached"] is None and rows["PR-SEV"]["reached"] is False
    assert rows["PR-SEV"]["scope"] == "variable" and "by_variable" not in rows
    text = rp.calibration_markdown(list(rows.values()), ["ana", "ben"], 2, 0.67)
    assert "| outcome | 6 | 0 | 1.00 | – | not estimable |" in text and "| PR-SEV | 4 | – | 0.50 | 0.20 | not yet |" in text


# ---- the model's side of a card --------------------------------------------------------------------------

def test_model_side_shows_the_answer_and_what_the_model_recorded_for_each_kind_of_card():
    from tests.panel_fixtures import synthetic_record
    d = synthetic_record()
    side = lambda path, primary="x", replicate=None: review_page.model_side(                          # noqa: E731
        {"path": path, "primary": primary, "replicate": replicate}, d)
    sa = side("stage_a:expressed_enthusiasm", "3", "4")
    assert (sa["answer"], sa["second"]) == ("3", "4") and sa["reason"].startswith("Steady delivery.")
    out = side("pitch:any_offer", "yes")
    assert out["reason"].startswith("Investor C: exit at 1:51: Out. · Investor A: offer at 2:39 (200k for 20%): "
                                    "States terms.")
    assert [t["turn"] for t in out["turns"]] == ["10", "17", "18"]
    cut = side("pitch:first_decision", "1:51")
    assert cut["reason"] == "Investor C: exit at 1:51: Out." and [t["turn"] for t in cut["turns"]] == ["10"]
    silent = review_page.model_side({"path": "pitch:any_offer", "primary": "unknown"}, {**d, "decisions": []})
    assert silent["reason"] == "The model records no decision of the panel." and silent["turns"] == []
    assert side("exchange:1/topic")["reason"] == "The model's own name for this line of questioning: customer acquisition cost."
    code = side("turn:4/code:PR-SEV", "3")
    assert code["reason"] == review_page.NO_REASON and code["turns"][0]["summary"].startswith("Disputes the paid-channel")


def test_copy_videos_can_copy_the_training_videos(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    sources = tmp_path / "videos"
    sources.mkdir()
    for v in ("vid1", "vid2", "vid3", "vid4"):
        (sources / f"{v}.mp4").write_bytes(b"fake " + v.encode())
        exp = run / v / "experiment.json"
        exp.write_text(json.dumps({**json.loads(exp.read_text()), "source": str(sources / f"{v}.mp4")}))
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(tmp_path / "out"), "--training"]) == 1
    assert "run `review_panel.py training` first" in capsys.readouterr().err
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "--video-ids", "vid3", "vid1"]) == 0
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(tmp_path / "out"), "--training"]) == 0
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["vid1.mp4", "vid3.mp4"]


# ---- the clip session in training (plan C, Task 7) ---------------------------------------------------------------

def clip_training_run(tmp_path, books):
    """The training run with the model's rank scores for every video, which the clip session's pairs need."""
    from tests.test_review_clips import scores_for
    run = training_run(tmp_path, books)
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books, ["vid1", "vid2", "vid3", "vid4"])))
    return run


def clip_training(run, *extra):
    return rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2",
                    "--clips", "--seed", "3", *extra])


def test_training_with_clips_writes_a_clip_session_for_two_coders_and_the_model_answers(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    before = tree_digest(run)
    assert clip_training(run) == 0
    tdir = run / "review" / "training"
    cdir = tdir / "clips"
    manifest = json.loads((cdir / "export_manifest.json").read_text())
    codes = list(books.stage_a_ids())
    assert manifest["videos"] == ["vid1", "vid2"] and manifest["coders"] == ["ana", "ben"]
    assert re.fullmatch(r"[0-9a-f]{12}", manifest["design_id"]) and manifest["training"] is True
    # two videos: the one pair three times (each video in three comparisons), sides seeded
    assert manifest["per_video"] == 3 and [p["batch_id"] for p in manifest["pairs"]] == ["b01", "b02", "b03"]
    assert all({p["left"], p["right"]} == {"vid1", "vid2"} for p in manifest["pairs"])
    per_video = [r for r in manifest["rows"] if r["field_class"] != "pair"]
    pairs = [r for r in manifest["rows"] if r["field_class"] == "pair"]
    assert len(per_video) == 14 and len(pairs) == 9            # seven cards per video (no PRES-CTX since Task 7c)
    assert all(r["context"].startswith(f"design {manifest['design_id']}: ") for r in manifest["rows"])
    assert len(read_sheet(cdir / "clips.csv")) == 23
    # each coder's page is the task page (plan C Task 7e): each video's tasks, then the comparisons; no clip-session page
    assert not (cdir / "pages").exists()
    for coder in ("ana", "ben"):
        page = page_model_of(tdir / "pages" / f"{coder}.html")
        assert page["mode"] == review_page.TRAINING_TASKS and page["design_id"] == manifest["design_id"] and page["codebook"]
        assert page["banner"] == review_page.TRAINING_BANNER and "Training" in page["heading"]
        assert [v["video_id"] for v in page["videos"]] == [*manifest["order"][coder],
                                                            *(f"compare-{p['batch_id']}" for p in manifest["pairs"])]
    # the full-video training sheets leave the clip session's cards to it, as the validation sheets do
    full = [r["path"] for sheet in tdir.glob("*__review.csv") for r in read_sheet(sheet)]
    assert full and not [p for p in full if p.split(":")[0] in ("stage_a", "prep_pres") or p == "venture:ask"]
    assert "pitch:first_decision" in full
    # the model answers: each video's clip-session cards and the pairs, with the model's side, beside the
    # full-video cards; the clip-session entries play the same full video at its window under their own name
    answers = page_model_of(tdir / "model_answers.html")
    ids = [v["video_id"] for v in answers["videos"]]
    assert ids == ["vid1__prepared", "vid2__prepared", "pairs", "vid1", "vid2"]
    exported = {(r["video_id"], r["path"]): r["primary"] for r in per_video}
    for vd in answers["videos"][:2]:
        video_id = vd["video_id"].removesuffix("__prepared")
        assert vd["file_name"] == f"{video_id}.mp4" and vd["window"] == "0:05-1:00"
        cards = [c for s in vd["sections"] for c in s["cards"]]
        # the model's count of checkable claims is shown as the band a coder chooses, the count in the reason
        assert len(cards) == 7 and all(c["model"]["answer"] == (exported[(video_id, c["path"])] if c["path"] !=
                                                                 "prep_pres:PRES-EVID" else "3-5") for c in cards)
        evid = next(c for c in cards if c["path"] == "prep_pres:PRES-EVID")
        assert evid["values"] == ["0", "1-2", "3-5", "6+"] and "count: 5" in evid["model"]["reason"]
        stage = [c for c in cards if c["field_class"] == "stage_a"]
        assert all("rank score" in c["model"]["reason"] for c in stage)
    held = answers["videos"][2]
    assert [(p["left"], p["right"]) for p in held["pairs"]] == [(f"{p['left']}__prepared", f"{p['right']}__prepared")
                                                               for p in manifest["pairs"]]
    assert {p["left_file"] for p in held["pairs"]} | {p["right_file"] for p in held["pairs"]} == {"vid1.mp4", "vid2.mp4"}
    assert {p["left_window"] for p in held["pairs"]} == {"0:05-1:00"}
    pair_cards = [c for s in held["sections"] for c in s["cards"]]
    assert [c["model"]["answer"] for c in pair_cards] == [r["primary"] for r in pairs]
    assert all("rank score" in c["model"]["reason"] for c in pair_cards)
    assert tree_digest(run) == before                    # nothing outside review/training changed
    out = capsys.readouterr().out
    assert "clip session" in out and f"copy-videos --run {run} --training --to" in out and "--training --clips" not in out


def test_training_with_clips_refuses_without_the_rank_scores_and_writes_nothing(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    assert clip_training(run) == 1
    err = capsys.readouterr().err
    assert "ranking/scores.json" in err and "run_stage_a_ranking.py" in err
    assert not (run / "review" / "training").exists()


def test_training_with_clips_refuses_to_replace_a_session_without_force(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    capsys.readouterr()
    assert clip_training(run) == 1
    assert "--force" in capsys.readouterr().err
    assert clip_training(run, "--force") == 0


def test_the_walkthrough_answers_a_pair_card_with_the_models_side():
    template = review_page.TEMPLATE.read_text(encoding="utf-8")
    pairs_branch = template.split("if(s.key==='pairs' && vd.pairs){", 1)[1].split("continue; }", 1)[0]
    assert "PAGE.mode === 'walkthrough' ? modelCard(vd, c) : card(vd, c)" in pairs_branch


def _fill_clips(cdir: Path, drop: Path, coder: str, answer) -> None:
    """A coder's clip-session sheet made from the exported one, as the page writes it: every clip card answered (its
    first allowed value, the ask as dollars;per cent, a count as 2), each pair as `answer(row)` says."""
    from pipeline.stage_b.review_items import SHEET_COLUMNS
    rows = read_sheet(cdir / "clips.csv")
    exported = {r["row"]: r for r in json.loads((cdir / "export_manifest.json").read_text())["rows"]}
    for row in rows:
        known = exported[int(row["row"])]
        if known["field_class"] == "pair":
            row["human_value"] = answer(known)
        elif known["path"] == "venture:ask":
            row["human_value"] = "200000;10"
        else:
            row["human_value"] = (known["allowed_values"] or ["2"])[0]
    with open(drop / f"clips__{coder}.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def other_side(row):
    return "left" if row["primary"] == "right" else "right"


def clip_drop(tmp_path, run, books):
    """Both coders' full-video and clip-session training sheets: they agree on every card; ana chooses as the model on
    every pair, ben as the model on the first Stage A item and against it on the last."""
    codes = list(books.stage_a_ids())
    tdir, drop = run / "review" / "training", tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    _fill_training(tdir, drop, "ana", first, clock="1:50")
    _fill_training(tdir, drop, "ben", first, clock="1:52")              # within the cut's five seconds
    _fill_clips(tdir / "clips", drop, "ana", lambda r: r["primary"])
    _fill_clips(tdir / "clips", drop, "ben", lambda r: other_side(r) if r["code"] == codes[-1] else r["primary"])
    return drop


def test_calibrate_pools_the_clip_session_with_the_full_video_sheets_and_reports_the_pairs(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)
    before = tree_digest(run)
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop), "--bar", "0.67"]) == 0
    out = capsys.readouterr().out
    tdir = run / "review" / "training"
    data = json.loads((tdir / "calibration.json").read_text())
    rows = {(r["scope"], r["field"]): r for r in data["rows"]}
    # the clip sheet's cards pooled with the full-video sheets' by path, per code; alpha against the bar where the
    # kind has one (the first decision and the ask have none)
    for key in (("code", "pitch_positivity"), ("code", "PRES-SUBS"), ("code", "PRES-EVID")):
        assert key in rows and rows[key]["bar"] == 0.67, key
    for key in (("code", "cut"), ("code", "venture:ask")):
        assert key in rows and rows[key]["bar"] is None and rows[key]["reached"] is None, key
    assert rows[("code", "PRES-EVID")]["agree"] == 1.0 and rows[("code", "cut")]["agree"] == 1.0
    assert rows[("code", "venture:ask")]["agree"] == 1.0 and rows[("code", "venture:ask")]["n"] == 2
    # one pair row per Stage A item: the pooled coder-model share read against 0.5 (plan D: no target), each
    # coder's beside it
    from pipeline.reliability.agreement import binomial_interval, binomial_p_vs_half
    codes = list(books.stage_a_ids())
    good, bad = rows[("pair", f"pair:{codes[0]}")], rows[("pair", f"pair:{codes[-1]}")]
    assert good["bar"] is bad["bar"] is None and good["reached"] is bad["reached"] is None
    assert good["agree"] == 1.0 and good["coder_coder"] == 1.0 and good["by_coder"] == {"ana": 1.0, "ben": 1.0}
    assert good["n"] == 6 and good["pairs"] == 3 and good["k"] == 6
    assert good["interval"] == list(binomial_interval(6, 6)) and good["p_vs_half"] == binomial_p_vs_half(6, 6)
    assert bad["agree"] == 0.5 and bad["by_coder"] == {"ana": 1.0, "ben": 0.0} and bad["coder_coder"] == 0.0
    low, high = binomial_interval(3, 6)
    text = (tdir / "calibration.md").read_text()
    assert "## Pair judgements" in text and "read against 0.5" in text and "Reached the target" not in text
    assert f"| {codes[-1]} | 3 | 6 | 0.50 | {low:.2f}–{high:.2f}; p = 1.000 | ana 1.00, ben 0.00 | 0.00 |" in text
    assert "Pair judgements" in out and "pair:" not in out.split("below the bar:")[-1]
    assert tree_digest(run) == before
    assert not any((run / v / "review").exists() for v in ("vid1", "vid2"))
    assert not (tdir / "clips" / "pairs_log.jsonl").exists()


def test_copy_videos_hands_out_the_training_videos_in_full_and_refuses_clips(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    for v in ("vid1", "vid2"):
        source = tmp_path / "sources" / f"{v}.mp4"
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(b"x")
        experiment = json.loads((run / v / "experiment.json").read_text())
        (run / v / "experiment.json").write_text(json.dumps({**experiment, "source": str(source)}))
    target = tmp_path / "handout"
    capsys.readouterr()
    assert rp.main(["copy-videos", "--run", str(run), "--training", "--clips", "--to", str(target)]) == 1
    assert "full videos" in capsys.readouterr().err and not target.exists()
    assert rp.main(["copy-videos", "--run", str(run), "--training", "--to", str(target)]) == 0
    assert sorted(p.name for p in target.iterdir()) == ["vid1.mp4", "vid2.mp4"]
    assert "2 of 2 videos" in capsys.readouterr().out


# ---- fix round 1 (plan C, Task 7 review) -------------------------------------------------------------------------

@pytest.mark.parametrize("video_ids", [["vid1"], ["vid1", "vid2", "vid3"], ["vid1", "vid2", "vid3", "vid4"]])
def test_training_with_clips_takes_exactly_two_videos_and_writes_nothing_else(tmp_path, books, capsys, video_ids):
    run = clip_training_run(tmp_path, books)
    before = tree_digest(run)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", *video_ids,
                    "--clips", "--seed", "3"]) == 1
    err = capsys.readouterr().err
    assert "exactly two" in err and "development videos" in err and f"got {len(video_ids)}" in err
    assert "Traceback" not in err and "choose another seed" not in err
    assert not (run / "review" / "training").exists() and tree_digest(run) == before


def test_training_without_clips_over_a_clip_session_needs_force_and_sets_the_session_aside(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    tdir = run / "review" / "training"
    design = json.loads((tdir / "clips" / "export_manifest.json").read_text())["design_id"]
    capsys.readouterr()
    plain = ["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2", "--seed", "3"]
    assert rp.main(plain) == 1
    assert "--force" in capsys.readouterr().err and (tdir / "clips" / "export_manifest.json").exists()
    assert rp.main([*plain, "--force"]) == 0
    assert not (tdir / "clips").exists()
    aside = tdir / f"earlier_clips_{design}"
    assert (aside / "export_manifest.json").exists() and (aside / "clips.csv").exists()     # set aside, not deleted
    assert "set aside" in capsys.readouterr().out


def test_a_second_set_aside_of_the_same_design_never_deletes_the_first(tmp_path, books, capsys):
    """clips -> plain -> clips with the same videos and seed (the same design) -> plain: both earlier sessions are kept,
    the second under a free name, and the message says where."""
    run = clip_training_run(tmp_path, books)
    tdir = run / "review" / "training"
    plain = ["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2", "--seed", "3",
             "--force"]
    assert clip_training(run) == 0
    design = json.loads((tdir / "clips" / "export_manifest.json").read_text())["design_id"]
    (tdir / "clips" / "lead_notes.txt").write_text("first session")         # something only the first one holds
    assert rp.main(plain) == 0
    assert clip_training(run, "--force") == 0
    assert json.loads((tdir / "clips" / "export_manifest.json").read_text())["design_id"] == design
    capsys.readouterr()
    assert rp.main(plain) == 0
    first, second = tdir / f"earlier_clips_{design}", tdir / f"earlier_clips_{design}_2"
    for aside in (first, second):
        assert (aside / "export_manifest.json").exists() and (aside / "clips.csv").exists(), aside
    assert (first / "lead_notes.txt").read_text() == "first session" and not (second / "lead_notes.txt").exists()
    assert str(second) in capsys.readouterr().out and not (tdir / "clips").exists()


def test_stale_clip_sheets_do_not_override_a_retraining_without_clips(tmp_path, books, capsys):
    """The reviewer's probe: after a retraining without --clips, the coders' old clip sheets are still in the folder;
    two coders who disagree on every Stage A card must not come out agreeing."""
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)                   # the old session's sheets, agreeing on Stage A
    for old in drop.glob("vid*__*.csv"):
        old.unlink()
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2",
                    "--seed", "3", "--force"]) == 0
    tdir = run / "review" / "training"
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    last = lambda r: r["allowed_values"].split(" | ")[-1]                                   # noqa: E731
    _fill_training(tdir, drop, "ana", first)
    _fill_training(tdir, drop, "ben", lambda r: last(r) if r["field_class"] == "stage_a" else first(r))
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop), "--bar", "0.67"]) == 1
    assert "no training clip session" in capsys.readouterr().err
    rows = {(r["scope"], r["field"]): r for r in json.loads((tdir / "calibration.json").read_text())["rows"]}
    stage_a = [rows[("code", c)] for c in books.stage_a_ids()]
    assert all(r["agree"] == 0.0 for r in stage_a) and not any(scope == "pair" for scope, _ in rows)


def test_a_clip_sheet_of_an_earlier_training_design_is_refused(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)
    assert clip_training(run, "--force", "--seed", "4") == 0           # another design, the old sheets stay
    for old in drop.glob("vid*__*.csv"):
        old.unlink()
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    tdir = run / "review" / "training"
    _fill_training(tdir, drop, "ana", first, clock="1:50")
    _fill_training(tdir, drop, "ben", first, clock="1:50")
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 1
    assert "filled for design" in capsys.readouterr().err
    rows = {(r["scope"], r["field"]) for r in json.loads((tdir / "calibration.json").read_text())["rows"]}
    assert not any(scope == "pair" for scope, _ in rows)
    assert not any(("code", c) in rows for c in books.stage_a_ids())


def test_calibrate_takes_a_folder_of_clip_sheets_alone(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)
    for sheet in drop.glob("vid*__*.csv"):
        sheet.unlink()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 0
    rows = {(r["scope"], r["field"]) for r in json.loads((run / "review" / "training" / "calibration.json")
                                                         .read_text())["rows"]}
    assert {("code", c) for c in books.stage_a_ids()} | {("code", "PRES-EVID"), ("code", "venture:ask")} <= rows
    assert ("code", "cut") not in rows and sum(scope == "pair" for scope, _ in rows) == 3


def test_calibrate_with_clip_sheets_but_no_training_clip_session_says_so(tmp_path, books, capsys):
    run = training_run(tmp_path, books)
    assert rp.main(["training", "--run", str(run), "--coders", "ana", "ben", "--video-ids", "vid1", "vid2"]) == 0
    drop = tmp_path / "drop"
    drop.mkdir()
    first = lambda r: r["allowed_values"].split(" | ")[0]                                   # noqa: E731
    _fill_training(run / "review" / "training", drop, "ana", first)
    (drop / "clips__ana.csv").write_text("row,path\n")
    capsys.readouterr()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 1
    assert "clips__ana.csv: no training clip session" in capsys.readouterr().err


def test_a_pair_dimension_with_one_coders_clip_sheet_is_not_estimable(tmp_path, books, capsys):
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)
    (drop / "clips__ben.csv").unlink()
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 0
    tdir = run / "review" / "training"
    pairs = [r for r in json.loads((tdir / "calibration.json").read_text())["rows"] if r["scope"] == "pair"]
    assert len(pairs) == 3 and all(r["agree"] == 1.0 and r["reached"] is None for r in pairs)
    assert all("coder against coder" in r["note"] for r in pairs)
    assert "not estimable" in (tdir / "calibration.md").read_text()


def test_the_calibration_reads_each_pair_row_against_one_half():
    from pipeline.reliability.agreement import binomial_interval, binomial_p_vs_half
    pairs = {"a": {"coder_model": 0.8, "coder_coder": 0.9, "coder_model_by_coder": {"x": 0.8}, "n": 10, "n_pairs": 5,
                   "videos": 2},
             "b": {"coder_model": 0.5, "coder_coder": None, "coder_model_by_coder": {"x": 0.5}, "n": 10,
                   "n_pairs": 5, "videos": 2}}
    rows = {r["field"]: r for r in rp.calibration_rows({"pairs": pairs}, 0.67)}
    assert rows["pair:a"]["k"] == 8 and rows["pair:a"]["p_vs_half"] == binomial_p_vs_half(8, 10)
    assert rows["pair:a"]["reached"] is None and rows["pair:a"]["bar"] is None
    assert "coder against coder unmeasured" in rows["pair:b"]["note"]
    text = rp.calibration_markdown(list(rows.values()), ["x", "y"], 2, 0.67)
    low, high = binomial_interval(8, 10)
    assert f"| a | 5 | 10 | 0.80 | {low:.2f}–{high:.2f}; p = {binomial_p_vs_half(8, 10):.3f} | x 0.80 | 0.90 |" in text
    assert "| b | 5 | 10 | 0.50 |" in text and "p = 1.000 | x 0.50 | – |" in text
    assert "≥" not in text.split("## Pair judgements")[1] and "Reached the target" not in text


def test_calibrate_defaults_its_bar_to_the_lower_bound_of_substantial(tmp_path, books, capsys):
    """Plan D: the bar is Landis & Koch's "substantial" lower bound (0.61), from validation_reporting.yaml."""
    run = clip_training_run(tmp_path, books)
    assert clip_training(run) == 0
    drop = clip_drop(tmp_path, run, books)
    assert rp.main(["calibrate", "--run", str(run), "--dir", str(drop)]) == 0
    data = json.loads((run / "review" / "training" / "calibration.json").read_text())
    assert data["bar"] == 0.61 == rp.load_reporting(books).band_floor("substantial")
    assert all(r["bar"] == 0.61 for r in data["rows"] if r["scope"] != "pair" and r["kind"] not in ("clock", "exact"))
    assert "α ≥ 0.61" in (run / "review" / "training" / "calibration.md").read_text()
    capsys.readouterr()
    with pytest.raises(SystemExit):
        rp.parse_args(["calibrate", "--help"])
    assert "substantial" in capsys.readouterr().out


def test_import_folder_leaves_a_clip_session_sheet_to_clips_import(tmp_path, books, capsys):
    run, _ = make_run(tmp_path, books)
    assert rp.main(["export", "--run", str(run), "--blind"]) == 0
    folder = tmp_path / "in"
    folder.mkdir()
    sheet = run / "vid1" / "review" / "vid1__review.csv"
    (folder / "vid1__ana.csv").write_bytes(sheet.read_bytes())
    fill(folder / "vid1__ana.csv", {r["path"]: {"human_value": r["allowed_values"].split(" | ")[0]}
                                    for r in read_sheet(sheet) if r["allowed_values"]})
    (folder / "clips__ana.csv").write_text("row,path\n")
    capsys.readouterr()
    assert rp.main(["import-folder", "--run", str(run), "--dir", str(folder)]) == 0
    out = capsys.readouterr()
    assert "skipped: clips__ana.csv is a clip-session sheet" in out.err and "1 of 1 sheets imported" in out.out
