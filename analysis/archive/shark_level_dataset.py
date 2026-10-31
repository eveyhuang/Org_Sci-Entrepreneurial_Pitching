"""The per-investor half of the panel dataset (the retired O2 design): each investor's outcome, cut and history.

Moved verbatim from `analysis.panel_dataset`, which keeps the pitch-level tables, `pitch_outcome`,
`pitch_cut` and `eligible_turns`. An investor's cut is that investor's own first offer, joined offer or
exit; the history it may read is `eligible_turns` before that cut minus the investor's own turns of
FOCAL_EXCLUDED_FUNCTIONS (none since v3-dev9, when expresses_interest merged into accepts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.panel_dataset import (
    CUT_KINDS, PanelData, _ever_offer, _implies_offer, _investors, _ordered_decisions, eligible_turns,
    exclusion_reasons,
)

FOCAL_EXCLUDED_FUNCTIONS = ()   # expresses_interest merged into accepts in v3-dev9; the per-shark cut is retired in plan B
REASON_OWN_INTEREST = "this investor's own expression of interest (too close to their own decision)"


def investor_outcomes(data: PanelData) -> pd.DataFrame:
    """o2, one row per investor, read from the decision list only.

    ever_offer is 1 when the investor made or joined an offer at any time (also after an exit and
    re-entry; also when the record holds only the revision or withdrawal of an offer, which must have
    existed), 0 when the investor exited and never offered, missing when neither was observed.

    A human verdict in data.outcome_overrides replaces an investor's ever_offer (yes 1, no 0, unknown
    missing) and marks the row ever_offer_source = "human_override" (otherwise "record"). The decision
    columns and the cuts still describe the record.
    """
    decisions = _ordered_decisions(data)
    o2_rows = []
    for a in _investors(data).itertuples():
        in_video = decisions[decisions.video_id == a.video_id]
        own = in_video[in_video.actor == a.actor_id]
        ever = _ever_offer(data, in_video, a.video_id, a.actor_id)      # the pitch path's rule, override included
        override = data.outcome_overrides.get(a.video_id, {}).get(a.actor_id)
        o2_rows.append({"video_id": a.video_id, "investor": a.actor_id, "label": a.label, "ever_offer": ever,
                        "ever_offer_source": "record" if override is None else "human_override",
                        "first_decision_kind": own.kind.iloc[0] if len(own) else None,
                        "first_decision_pos": float(own.pos.iloc[0]) if len(own) else np.nan})
    return pd.DataFrame({
        "video_id": pd.Series([r["video_id"] for r in o2_rows], dtype=object),
        "investor": pd.Series([r["investor"] for r in o2_rows], dtype=object),
        "label": pd.Series([r["label"] for r in o2_rows], dtype=object),
        "ever_offer": pd.Series([r["ever_offer"] for r in o2_rows], dtype=float),
        "first_decision_kind": pd.Series([r["first_decision_kind"] for r in o2_rows], dtype=object),
        "first_decision_pos": pd.Series([r["first_decision_pos"] for r in o2_rows], dtype=float),
        "ever_offer_source": pd.Series([r["ever_offer_source"] for r in o2_rows], dtype=object),
    })


def investor_cuts(data: PanelData) -> pd.DataFrame:
    """o2_cut: the turn position at which each investor row's history stops.

    An investor's cut is the position of that investor's first offer, joined offer or exit (or of
    the first revision or withdrawal, when that is all the record holds of the offer). Without such a
    decision the cut is the number of turns, so the whole encounter is history.
    """
    n_turns = dict(zip(data.videos.video_id, data.videos.n_turns))
    decisions = data.decisions[data.decisions.role == "investor"]
    o2_rows = []
    for a in _investors(data).itertuples():
        own = decisions[(decisions.video_id == a.video_id) & (decisions.actor == a.actor_id)
                        & (decisions.kind.isin(CUT_KINDS) | _implies_offer(decisions))]
        o2_rows.append((a.video_id, a.actor_id, int(own.pos.min()) if len(own) else int(n_turns[a.video_id])))
    return pd.DataFrame({"video_id": pd.Series([r[0] for r in o2_rows], dtype=object),
                         "investor": pd.Series([r[1] for r in o2_rows], dtype=object),
                         "cut_pos": pd.Series([r[2] for r in o2_rows], dtype=int)})


def eligible_turns_for_investor(data: PanelData, video_id: str, cut_pos: int, focal: str) -> pd.DataFrame:
    """The turns the `focal` investor's row may read: `eligible_turns` minus that investor's own turns of
    FOCAL_EXCLUDED_FUNCTIONS."""
    t = eligible_turns(data, video_id, cut_pos)
    return t[~((t.speaker == focal) & t.function.isin(FOCAL_EXCLUDED_FUNCTIONS))]


def exclusion_reasons_for_investor(data: PanelData, video_id: str, cut_pos: int, focal: str) -> pd.Series:
    """`exclusion_reasons` for the `focal` investor's row: the turns with an empty reason are exactly those of
    `eligible_turns_for_investor`. The focal investor's own expressed interest is named last, after the cut, a
    decides or negotiates turn and a negotiation exchange."""
    reason = exclusion_reasons(data, video_id, cut_pos)
    t = data.turns.loc[reason.index]
    own_interest = t.function.isin(FOCAL_EXCLUDED_FUNCTIONS) & (t.speaker == focal) & (reason == "")
    reason[own_interest] = REASON_OWN_INTEREST
    return reason
