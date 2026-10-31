"""Stage A: the ten Hu & Ma (2025) Pitch Factor constructs (the Stage A book of release v3-dev8).

Definitions are inherited verbatim from the source; see
docs/codebooks/archive/stage_b_panel_v3dev8/stage_a_pitch_factor.yaml for the full construct text that
the prompt renders. This module fixes only the response shape. Archived 2026-10-20: the live Stage A
schema is pipeline.schemas.stage_a_perceived_delivery; this one serves
configs/archived_stage_a_pitch_factor.yaml.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pipeline.schemas.common import Rating

STAGE_A_CONSTRUCTS: tuple[str, ...] = (
    "visual_positive",
    "visual_negative",
    "vocal_arousal",
    "vocal_valence",
    "vocal_positive",
    "vocal_negative",
    "verbal_positive",
    "verbal_negative",
    "verbal_warmth",
    "verbal_ability",
)


class StageARecord(BaseModel):
    """One holistic annotation of a whole pitch video."""

    model_config = ConfigDict(extra="forbid")

    visual_positive: Rating
    visual_negative: Rating
    vocal_arousal: Rating
    vocal_valence: Rating
    vocal_positive: Rating
    vocal_negative: Rating
    verbal_positive: Rating
    verbal_negative: Rating
    verbal_warmth: Rating
    verbal_ability: Rating
    overall_notes: str = Field(default="", max_length=1500)
