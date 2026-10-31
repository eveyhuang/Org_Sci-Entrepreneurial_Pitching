"""The figure style, under its older name: `analysis.human_agreement` and `scripts/experiments/reaction_trajectory_test.py`
draw with these helpers. The figures of the analysis are in `analysis.pitch_figures`, which defines everything here.

The legacy figures of the investor-row (O2) notebook were removed in plan B task 9, when the notebook moved to the
pitch-level analysis; only the shared style and drawing helpers remain, re-exported unchanged.
"""
from __future__ import annotations

from analysis.pitch_figures import (  # noqa: F401  (re-exported: analysis.human_agreement and the experiments read them here)
    BLACK, BLUE, DEFAULT_WIDTH, FONT_SIZE, FONTS, GREEN, GREY, GRID, INK, LINE_COLOURS, LINE_DASHES, LINE_INCHES,
    LINE_MARKERS, MOVE_WORDS, NEGATIVE, OFFER_STYLE, ORANGE, POSITIVE, PURPLE, ROW_PAD, SAVE_DPI, SCREEN_DPI,
    SIGNIFICANCE_WORDS, SKY, STACK_BELOW, SUPPORT_STYLE, UNADJUSTED_WORDS, UNCLEAR, VERMILLION, WIDTHS, _context,
    _grid, _handle, _interval_point, _label_rows, _layout, _letter, _point, _reference_ticks, _significance_handles,
    _stacked, _wrap, figure_7_after_an_answer, font_name, plain, save_figure, significance_look, style, width_inches,
)
