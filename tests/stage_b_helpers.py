"""A synthetic Stage B project: one 12-second clip and a fixture for every pass.

The timeline has 8 turns (words-free: `tag` names what each turn DOES, never what was
said). Turns 0-1 are the prepared pitch (0-4 s); episode 0 is turns 2-4 (4-7 s, speaker-2
probes); episode 1 is turns 5-7 (7-12 s, speaker-3 probes).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from pipeline.config import load_config
from pipeline.ingest import ingest_corpus
from pipeline.registry import Registry
from pipeline.runs import create_or_resume_run
from pipeline.schemas.stage_a_perceived_delivery import STAGE_A_DELIVERY_CODES
from tests.conftest import make_test_video

REPO = Path(__file__).resolve().parents[1]
ARCHIVE_B1 = REPO / "docs" / "codebooks" / "archive" / "stage_b_v1_episode"
VIDEO = "clipA"
STAGE_A_BOOK = str(REPO / "pipeline" / "codebooks" / "stage_a_perceived_delivery.yaml")

_TURNS = [
    ("speaker-1", "0:00", "0:02", "greets the panel, names the product"),
    ("speaker-1", "0:02", "0:04", "asks 100k for ten percent"),
    ("speaker-2", "0:04", "0:05", "asks about margins"),
    ("speaker-1", "0:05", "0:06", "states sixty percent gross margin"),
    ("speaker-2", "0:06", "0:07", "accepts the answer"),
    ("speaker-3", "0:07", "0:09", "challenges the valuation"),
    ("speaker-1", "0:09", "0:10", "offers fifteen percent instead"),
    ("speaker-3", "0:10", "0:12", "declines to invest"),
]


def timeline_payload() -> dict:
    return {
        "video_length": "0:12",
        "roster": [
            {"speaker": "speaker-1", "role": "founder", "seat": "standing centre",
             "descriptor": "woman, green jacket, standing centre", "first_heard": "0:00"},
            {"speaker": "speaker-2", "role": "investor",
             "seat": "panel seat 1 of 5 from the left",
             "descriptor": "man, grey suit, seated far left", "first_heard": "0:04"},
            {"speaker": "speaker-3", "role": "investor",
             "seat": "panel seat 5 of 5 from the left",
             "descriptor": "woman, red dress, seated far right", "first_heard": "0:07"},
        ],
        "turns": [
            {"turn_idx": i, "speaker": s, "start": a, "end": b, "tag": t}
            for i, (s, a, b, t) in enumerate(_TURNS)
        ],
    }


def segmentation_payload() -> dict:
    # The model's own clock strings deliberately disagree with the turn times by a
    # second: the post-processor must derive clip bounds from the turns, not from these.
    return {
        "n_pitches": 1,
        "prepared_end": "0:04",
        "summary": "A founder pitches a folding kettle, asking 100k for ten percent.",
        "episodes": [
            {"start": "0:05", "end": "0:06", "start_turn": 2, "end_turn": 4,
             "probing_speaker": "speaker-2", "topic": "gross margins", "resolution": "resolved",
             "cut": False, "evidence": "t2 asks for margins; t4 accepts the answer"},
            {"start": "0:08", "end": "0:11", "start_turn": 5, "end_turn": 7,
             "probing_speaker": "speaker-3", "topic": "valuation", "resolution": "conceded",
             "cut": False, "evidence": "t5 challenges valuation; t6 founder gives ground"},
        ],
        "decisions": [
            {"speaker": "speaker-2", "decision": "offer", "turn": 4, "terms": "100k for ten percent"},
            {"speaker": "speaker-3", "decision": "out", "turn": 7},
        ],
        "deal": True,
    }


def stage_a_payload() -> dict:
    rating = {"value": 4, "uncodable": False, "evidence": "Steady, warm delivery at 0:12."}
    return {name: dict(rating) for name in STAGE_A_DELIVERY_CODES} | {"overall_notes": ""}


def episode_payload() -> dict:
    # turn_refs are GLOBAL turn indices. [2] lies inside episode 0 (turns 2-4) and outside
    # episode 1 (turns 5-7); the shared fixture therefore yields one invalid ref per code.
    def coded(value):
        return {"value": value, "uncodable": False, "evidence": "t2 question", "turn_refs": [2]}

    return {
        "PR-FRAME": coded("prevention"),
        "RS-EVID": coded(True),
        "PREP-SUBS": coded(3),
        "RAP-INTERRUPT": {"value": None, "uncodable": True,
                          "evidence": "overlap not audible", "turn_refs": []},
    }


_FLASH = {"name": "gemini-flash-x", "temperature": 0.0, "max_output_tokens": 65536}
PASSES: dict[str, dict] = {
    "timeline": {
        "name": "timeline", "unit": "PITCH", "prompt": "stage_b_timeline.j2",
        "schema": "pipeline.schemas.stage_b_timeline.TimelineRecord",
        "model": {"name": "gemini-pro-big", "temperature": 0.0, "max_output_tokens": 65536},
        "produces": "timeline",
    },
    "segmentation": {
        "name": "segmentation", "unit": "PITCH", "prompt": "stage_b_segmentation.j2",
        "schema": "pipeline.schemas.stage_b.SegmentationRecord", "depends_on": "timeline",
        "context": ["roster", "timeline"], "produces": "episodes",
        "codebook": str(ARCHIVE_B1 / "stage_b_segmentation.yaml"),
    },
    "prepared": {
        "name": "prepared", "unit": "PREPARED", "prompt": "stage_a_perceived_delivery.j2",
        "schema": "pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord", "codebook": STAGE_A_BOOK,
        "depends_on": "segmentation",
    },
    "episode": {
        "name": "episode", "unit": "EPISODE", "prompt": "stage_b_episode.j2",
        "schema": "codebook",
        "codebook": str(ARCHIVE_B1 / "stage_b_episode.yaml"),
        "depends_on": "segmentation", "context": ["roster", "timeline_slice", "pitch_summary"],
    },
}

FIXTURES = {
    "timeline": timeline_payload, "segmentation": segmentation_payload,
    "prepared": stage_a_payload, "holistic": stage_a_payload,
    "episode": episode_payload,
}


def write_fixtures(directory: Path, video_id: str = VIDEO) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for pass_name, payload in FIXTURES.items():
        # `<pass>__<video>.json` is FakeProvider's fallback for every unit of that video.
        (directory / f"{pass_name}__{video_id}.json").write_text(json.dumps(payload()))
    return directory


def make_prompt_dir(destination: Path) -> Path:
    """Templates resolve only from a config's prompt_dir: gather the archived v1 Stage B templates
    plus the live three-item Stage A template and its protocol in a scratch directory so the archive stays
    frozen."""
    destination.mkdir(parents=True, exist_ok=True)
    for source in (*ARCHIVE_B1.glob("*.j2"), *ARCHIVE_B1.glob("*_protocol.md"),
                   REPO / "pipeline" / "prompts" / "stage_a_perceived_delivery.j2",
                   REPO / "pipeline" / "prompts" / "stage_a_perceived_delivery_protocol.md"):
        shutil.copy2(source, destination / source.name)
    return destination


def make_project(tmp_path: Path, passes: list, k: dict[str, int] | None = None):
    """Build config + registry + run for the named passes. Requires ffmpeg."""
    make_test_video(tmp_path / "videos" / f"{VIDEO}.mp4", seconds=12.0)
    (tmp_path / "meta.csv").write_text(f"video_id\n{VIDEO}\n")
    prompts = make_prompt_dir(tmp_path / "prompts")
    entries = []
    for item in passes:
        entry = dict(PASSES[item]) if isinstance(item, str) else dict(item)
        entry["k"] = (k or {}).get(entry["name"], 1)
        entries.append(entry)
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({
        "run_name": "stb",
        "prompt_dir": str(prompts), "historical": True,
        "corpus": {"name": "shark-tank", "video_dir": "videos", "metadata_csv": "meta.csv"},
        "model": {"name": "gemini-pro-x"},
        "limits": {"max_concurrency": 2, "requests_per_minute": 6000, "max_attempts": 2,
                   "backoff_base_s": 0.001, "backoff_cap_s": 0.002},
        "passes": entries,
    }))
    cfg = load_config(tmp_path / "cfg.yaml")
    registry = Registry(tmp_path / "registry.sqlite")
    ingest_corpus(cfg, registry)
    run = create_or_resume_run(cfg, runs_root=tmp_path / "runs")
    return cfg, registry, run, write_fixtures(tmp_path / "fixtures")
