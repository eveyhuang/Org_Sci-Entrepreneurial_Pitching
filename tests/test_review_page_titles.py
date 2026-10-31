"""The researcher's decision of 2026-10-22 (reversing the final review's I4): coders see each pitch's episode title,
which they use to find the right file. A video task reads "<short title> — presentation and Q&A", a comparison
"Compare: <short A> vs <short B>"; `copy-videos` writes each video under its source name, and the page looks that
name up first and `<video_id>.mp4` as a fallback. Some titles hint at the outcome: a stated limit."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import review_panel as rp  # noqa: E402
from pipeline.instrument import load_release  # noqa: E402
from pipeline.stage_b.panel_books import load_panel_books  # noqa: E402
from tests.test_review_cli import make_run  # noqa: E402
from tests.test_review_clips import scores_for  # noqa: E402
from tests.test_review_page import _lines_of, _node, _script  # noqa: E402
from tests.test_review_training import page_model_of  # noqa: E402

VIDEOS = ["v00", "v01", "v02", "v03", "v04", "v05"]
TITLE = "Could Frescos Naturales Be Mr Wonderful's First Beverage Deal?"
SOURCE = f"{TITLE} | Shark Tank US.mp4"


@pytest.fixture(scope="module")
def books():
    return load_panel_books(load_release())


@pytest.fixture(scope="module")
def base_run(tmp_path_factory, books):
    base = tmp_path_factory.mktemp("titles")
    for v in VIDEOS:
        make_run(base, books, video_id=v)
    run = base / "run"
    (run / "ranking").mkdir()
    (run / "ranking" / "scores.json").write_text(json.dumps(scores_for(books, VIDEOS)))
    sources = base / "sources"
    sources.mkdir()
    for v in VIDEOS:            # every source carries a title, as the real ones do
        source = sources / (SOURCE if v == "v01" else f"{v} title | Shark Tank US.mp4")
        source.write_bytes(b"video " + v.encode())
        experiment = json.loads((run / v / "experiment.json").read_text())
        (run / v / "experiment.json").write_text(json.dumps({**experiment, "source": str(source)}))
    (run / "selection.json").write_text(json.dumps({"pilot_development_videos": ["v00", "v01"]}))
    return run


@pytest.fixture
def run(base_run, tmp_path):
    copy = tmp_path / "run"
    shutil.copytree(base_run, copy)
    return copy


def _short(v):
    return TITLE if v == "v01" else f"{v} title"


def test_the_task_page_shows_each_pitchs_title_and_looks_up_its_source_file_first(run):
    assert rp.main(["session-export", "--run", str(run), "--video-ids", *VIDEOS, "--coders", "ana",
                    "--seed", "4"]) == 0
    page = page_model_of(run / "review" / "pages" / "ana__tasks.html")
    videos = [t for t in page["videos"] if t["task"] == "video"]
    assert {t["video_id"]: t["title"] for t in videos} == {v: f"{_short(v)} — presentation and Q&A" for v in VIDEOS}
    v01 = next(t for t in videos if t["video_id"] == "v01")
    assert (v01["file_name"], v01["file_alts"]) == (SOURCE, ["v01.mp4"])
    for task in (t for t in page["videos"] if t["task"] == "compare"):
        pair = task["pair"]
        assert task["title"] == f"Compare: {_short(pair['left'])} vs {_short(pair['right'])}"
        assert pair["left_alts"] == [f"{pair['left']}.mp4"] and pair["left_file"].endswith("| Shark Tank US.mp4")


def test_copy_videos_writes_each_video_under_its_source_name(run, tmp_path):
    assert rp.main(["export", "--run", str(run), "--sample-videos", "2", "--seed", "1", "--blind",
                    "--exclude-from", "selection.json"]) == 0
    target = tmp_path / "handout"
    assert rp.main(["copy-videos", "--run", str(run), "--to", str(target)]) == 0
    chosen = json.loads((run / "review" / "video_sample.json").read_text())["chosen"]
    assert sorted(p.name for p in target.iterdir()) == sorted(f"{v} title | Shark Tank US.mp4" if v != "v01"
                                                              else SOURCE for v in chosen)


def test_the_page_looks_up_the_source_name_first_and_the_video_id_next():
    script = _script()
    lines = script.splitlines()
    start = lines.index(next(l for l in lines if l.startswith("const FILES = {};")))
    got = _node("\n".join([
        "const PAGE = {videos: [{video_id: 'v01', file_name: 'Titled | Show.mp4', file_alts: ['v01.mp4']},"
        " {video_id: 'compare-b01', pair: {left: 'v01', right: 'v02', left_file: 'Titled | Show.mp4',"
        " right_file: 'Other | Show.mp4', left_alts: ['v01.mp4'], right_alts: ['v02.mp4']}}]};",
        "\n".join(lines[start:start + 4]),
        _lines_of(script, "async function fileIn("),
        "const dir = names => ({getFileHandle: async n => { if(!names.includes(n)) throw new Error('none');"
        " return {getFile: async () => ({name: n})}; }});",
        "(async () => { const out = [];",
        " out.push((await fileIn(dir(['v01.mp4', 'Titled | Show.mp4']), FILES.v01)).name);",
        " out.push((await fileIn(dir(['v01.mp4']), FILES.v01)).name);",
        " out.push((await fileIn(dir(['v02.mp4']), FILES.v02)).name);",
        " console.log(JSON.stringify(out)); })();"]))
    assert got == ["Titled | Show.mp4", "v01.mp4", "v02.mp4"]
