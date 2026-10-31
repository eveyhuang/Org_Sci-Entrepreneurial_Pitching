import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

import list_other_concepts as loc  # noqa: E402
from tests.panel_fixtures import set_other, synthetic_record  # noqa: E402


def _write(run, video, name, record):
    path = run / video / "results" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pass": name, "record": record}))


def _run(tmp_path):
    run = tmp_path / "run"
    a = synthetic_record()
    set_other(a["turns"][10], "founder's own investment")
    a["exchanges"][2].update(topic="Other", label="Money the founder put in")
    b = synthetic_record()
    set_other(b["turns"][10], "Founder's  own investment ")
    set_other(b["turns"][14], "a patent")
    c = synthetic_record()
    set_other(c["turns"][1], "a patent")
    _write(run, "vid_b", "primary", a)
    _write(run, "vid_b", "replicate", b)
    _write(run, "vid_a", "primary", c)
    _write(run, "vid_a", "replicate", synthetic_record())
    (run / "ranking").mkdir()                                   # a folder that is not a video
    (run / "ranking" / "notes.json").write_text("{}")
    (run / "vid_c" / "results").mkdir(parents=True)             # a video whose passes failed
    (run / "vid_c" / "results" / "primary_failed.json").write_text("{}")
    (run / "registry.sqlite").write_text("")
    return run


def test_rows_cover_every_video_and_both_passes(tmp_path):
    rows = loc.collect(_run(tmp_path))
    assert [(r["video_id"], r["pass"], r["clock"], r["where"], r["code"], r["name"]) for r in rows] == [
        ("vid_a", "primary", "1:02", "turn", "PR-CONCEPT", "a patent"),
        ("vid_b", "primary", "1:55", "turn", "PR-CONCEPT", "founder's own investment"),
        ("vid_b", "primary", "1:55", "exchange", "EX-TOPIC", "Money the founder put in"),
        ("vid_b", "replicate", "1:55", "turn", "PR-CONCEPT", "Founder's own investment"),
        ("vid_b", "replicate", "2:18", "turn", "RS-CONCEPT", "a patent"),
    ]
    assert rows[0]["context"] == "Asks what it costs to acquire one customer."        # the turn summary
    assert rows[2]["context"] == "Money the founder put in"                           # the exchange label


def test_markdown_table_and_case_insensitive_tally(tmp_path):
    run = _run(tmp_path)
    out = tmp_path / "other.md"
    assert loc.main(["--run", str(run), "--out", str(out)]) == 0
    md = out.read_text()
    assert "| video | pass | time | where | code | name | turn summary or exchange label |" in md
    assert "| vid_b | replicate | 2:18 | turn | RS-CONCEPT | a patent | Quotes the paid-channel cost and expects it to fall. |" in md
    tally = md[md.index("## Names"):]
    assert [ln for ln in tally.splitlines() if ln.startswith("| ") and "---" not in ln][1:] == [
        "| a patent | 2 |", "| founder's own investment | 2 |", "| Money the founder put in | 1 |"]
    assert "5 items in 2 videos" in md and "model-coded" in md
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    loc.main(["--run", str(run), "--out", str(out)])
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}


def test_prints_when_no_out_is_given_and_handles_a_run_with_nothing(tmp_path, capsys):
    run = tmp_path / "run"
    _write(run, "vid_a", "primary", synthetic_record())
    assert loc.main(["--run", str(run)]) == 0
    text = capsys.readouterr().out
    assert "No other concepts" in text and "| video |" not in text
    with pytest.raises(SystemExit):
        loc.main(["--run", str(tmp_path / "nope")])


def test_an_unnamed_other_and_an_exchange_without_turns_are_still_listed(tmp_path):
    run = tmp_path / "run"
    record = synthetic_record()
    set_other(record["turns"][10], None)
    record["exchanges"].append({"exchange_id": 9, "kind": "substantive", "topic": "other", "label": "", "investors": []})
    _write(run, "vid_a", "primary", record)
    rows = loc.collect(run)
    assert [(r["where"], r["id"], r["name"], r["clock"]) for r in rows] == [("turn", 11, "", "1:55"), ("exchange", 9, "", None)]
    md = loc.render(run, rows)
    assert "| vid_a | primary | 1:55 | turn | PR-CONCEPT | (unnamed) | Asks for the repeat purchase rate. |" in md
    assert "| vid_a | primary | — | exchange | EX-TOPIC | (unnamed) |  |" in md
    assert "| (unnamed) | 2 |" in md and "2 items in 1 video." in md
