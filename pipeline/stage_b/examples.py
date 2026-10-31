"""Find candidate codebook examples in dev-set recordings.

A `real` example is a verbatim excerpt from a recording, typed by a person who has
watched the video. The pipeline already holds what is needed to propose WHERE to
look: each episode rating cites the timeline turns it rests on, and the timeline
records when each turn starts. This module lists, per code and answer, episodes
where the replicates agree, with the cited turns' timeline tags shown as a pointer —
NOT as a transcript, since no pass in this pipeline asks for dialogue and the tags
are the model's notes of what a turn DOES, never what was said. A PERSON then opens
the video at the given timestamp, types the verbatim excerpt they hear, and pastes
the resulting line into the codebook Markdown. Nothing here edits a codebook.
"""

from __future__ import annotations

import json
import re

import pandas as pd

from pipeline.schemas.stage_b_timeline import seconds_to_clock

_STAMP = re.compile(r"^\[t(\d+) (\d+:\d{2})\]")


def _answer(row) -> str:
    if isinstance(row["value_label"], str):
        return row["value_label"]
    value = float(row["value"])
    return str(int(value)) if value.is_integer() else str(value)


def find_candidates(annotations: pd.DataFrame, turns: pd.DataFrame,
                    dev_ids: list[str] | None = None, per_answer: int = 5) -> pd.DataFrame:
    coded = annotations[(annotations["pass"] == "episode") & ~annotations["uncodable"].astype(bool)]
    if dev_ids is not None:
        coded = coded[coded["video_id"].isin(dev_ids)]
    lookup = {(t.video_id, t.turn_idx): t for t in turns.itertuples()}
    rows = []
    for (code_id, video_id, episode_idx), group in coded.groupby(["code_id", "video_id", "episode_idx"]):
        first = group.sort_values("replicate_k").iloc[0]
        refs = json.loads(first["turn_refs"] or "[]")
        valid = first["turn_refs_valid"]  # a numpy bool or None once read from parquet: never use `is`
        flagged = valid is not None and not pd.isna(valid) and not bool(valid)
        if not refs or flagged or any((video_id, r) not in lookup for r in refs):
            continue  # nothing checkable to point at
        answers = {_answer(row) for _, row in group.iterrows()}
        quoted = []
        for ref in refs:
            turn = lookup[(video_id, ref)]
            stamp = seconds_to_clock(turn.start_s)  # where to look in the video
            quoted.append(f"[t{ref} {stamp}] {turn.speaker} ({turn.role}): {turn.tag}")
        rows.append({"code_id": code_id, "answer": _answer(first), "video_id": video_id,
                     "episode_idx": int(episode_idx), "replicates": len(group),
                     "agreed": len(answers) == 1, "evidence": first["evidence"],
                     "turn_refs": refs, "excerpt": "\n".join(quoted)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["code_id", "answer", "agreed", "replicates", "video_id"],
                              ascending=[True, True, False, False, True])
    return frame.groupby(["code_id", "answer"], sort=False).head(per_answer).reset_index(drop=True)


def candidates_markdown(frame: pd.DataFrame) -> str:
    out = ["# Example candidates — CHECK AGAINST THE VIDEO before using any of these", "",
           "The timeline tags below are the model's NOTES of what each cited turn does — never the",
           "words spoken (no pass in this pipeline asks for dialogue). To use a row: open the video",
           "at the timestamp shown, TYPE THE VERBATIM EXCERPT you hear, then paste the resulting",
           "`- [real | ...]` line under **Examples** in the codebook `.md`. Prefer rows where",
           "replicates agreed.", ""]
    for code_id, by_code in frame.groupby("code_id", sort=False):
        out += [f"## {code_id}", ""]
        for answer, by_answer in by_code.groupby("answer", sort=False):
            out += [f"### answer: {answer}", ""]
            for row in by_answer.itertuples():
                turns = ", ".join(f"t{r}" for r in row.turn_refs)
                match = _STAMP.match(row.excerpt.splitlines()[0])
                stamp = match.group(2) if match else ""
                out += [f"{row.video_id} episode {row.episode_idx} — "
                        f"{'agreed' if row.agreed else 'DISAGREED'} over {row.replicates} replicate(s); "
                        f"model's evidence: {row.evidence}", "", "```", row.excerpt, "```", "",
                        f"- [real | {row.video_id} {turns} {stamp}] TYPE THE VERBATIM EXCERPT HERE", ""]
    return "\n".join(out)
