"""Roster + words-free speaking timeline: the first Stage B pass.

Asking Gemini for the dialogue itself trips its recitation filter on broadcast pitches
(Checkpoint A, 2026-09-21): verbatim and paraphrased transcription attempts both came back
empty (`RECITATION`). Recording WHO speaks WHEN, plus a short non-quoting tag of what each
turn does, passed the filter on every clip tested. No pass in this pipeline asks for dialogue.

Times are asked for as player-style `M:SS` strings, the format the model sees natively; code
converts them to seconds via `clock_to_seconds` / `seconds_to_clock`.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SPEAKER = re.compile(r"^speaker-\d+$")
_CLOCK = re.compile(r"^(\d{1,3}):([0-5]\d)$")


def clock_to_seconds(value: str) -> float:
    match = _CLOCK.match(value.strip())
    if not match:
        raise ValueError(f"times look like '4:07' (minutes:seconds); got {value!r}")
    return float(int(match.group(1)) * 60 + int(match.group(2)))


def seconds_to_clock(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    return f"{whole // 60}:{whole % 60:02d}"


def normalised_turns(record: dict) -> list[dict]:
    """The one internal turn shape: seconds as floats, whatever the wire format was."""
    return [
        {"turn_idx": t["turn_idx"], "speaker": t["speaker"],
         "start_s": float(clock_to_seconds(t["start"])), "end_s": float(clock_to_seconds(t["end"])),
         "tag": t["tag"]}
        for t in record["turns"]
    ]


def roles(record: dict) -> dict[str, str]:
    return {p["speaker"]: p["role"] for p in record["roster"]}


def _speaker_label(value: str) -> str:
    if not _SPEAKER.match(value):
        raise ValueError(f"speaker labels look like 'speaker-3', never a name; got {value!r}")
    return value


class TimelinePerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    speaker: str
    role: Literal["founder", "investor", "host", "other"]
    seat: str = Field(min_length=1, max_length=120)
    descriptor: str = Field(min_length=1, max_length=300)
    first_heard: str

    _label = field_validator("speaker")(_speaker_label)
    _clock = field_validator("first_heard")(lambda v: (clock_to_seconds(v), v)[1])


class TimelineTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_idx: int = Field(ge=0)
    speaker: str
    start: str
    end: str
    tag: str = Field(min_length=1, max_length=80)

    _label = field_validator("speaker")(_speaker_label)

    @model_validator(mode="after")
    def _ordered(self) -> "TimelineTurn":
        if clock_to_seconds(self.end) < clock_to_seconds(self.start):
            raise ValueError(f"turn {self.turn_idx}: end is before start")
        return self


class TimelineRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_length: str
    roster: list[TimelinePerson]
    turns: list[TimelineTurn]

    @model_validator(mode="after")
    def _consistent(self) -> "TimelineRecord":
        clock_to_seconds(self.video_length)
        labels = [p.speaker for p in self.roster]
        if len(labels) != len(set(labels)):
            raise ValueError("a speaker appears twice in the roster")
        for position, turn in enumerate(self.turns):
            if turn.turn_idx != position:
                raise ValueError(f"turn_idx must be contiguous from 0 (position {position})")
            if turn.speaker not in labels:
                raise ValueError(f"turn {position}: {turn.speaker} is not in the roster")
        return self
