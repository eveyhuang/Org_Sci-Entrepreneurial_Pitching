"""Figures for the paper at the pitch level: what predicts whether any shark offers.

Nine figures and one supplementary figure, drawn from the pitch-level analysis of
`scripts/analysis/analyze_panel.py` (`compute` returns its `Analysis`). Nothing is estimated here: every figure
reads a table that analysis computed (the ladder's metrics and held-out predictions, the sensitivities' metrics,
the single-feature table, the answer-quality-by-offer table, the H4 shares, the coefficients of C5, the H9
frames, and with `--with-shark-level` the archived shark-level metrics), so a figure and its table cannot disagree.

    1  model_ladder               skill over the base rate of C0-C5 with C2p and C3q; the five sensitivities as
                                  hollow markers on the same axis
    2  auc_per_model              AUC of each model of the ladder
    3  roc_curves                 ROC curves of the base-rate-adjusted held-out score
    4  concept_forest             each pitch feature on its own (correlation, FDR marks)
    5  answer_quality_by_outcome  substance and checkable support of the answers by any offer
    6  calibration                calibration of the held-out predictions
    7  after_an_answer            the panel's first reaction after an answer, by checkable support (H4)
    8  model_coefficients         the coefficients of C5 refitted on all pitches
    9  h9_forest                  H9 coefficients with intervals, all questions (A) and hard questions (B)
    S1 shark_level_ladder         the archived shark-level ladder; drawn only when the shark-level path ran

Style, the same in every figure (`analysis.panel_figures` re-exports it under its older name for
`analysis.human_agreement`): white background, one sans-serif font, no text below 8 pt at the final size, the colour-blind-safe
palette of Okabe and Ito in which a series or a direction is never told by colour alone, plain-language labels,
and a panel letter per panel when a figure has more than one. `WIDTHS` holds the width presets in inches.

`write_figures` saves each figure as PNG (300 dpi) and vector PDF with a caption file (`figure_N_caption.md`)
that states the sample, that the data are model-coded and exploratory, what the intervals are and how to read
the figure. A figure that cannot be drawn from an analysis is left out and, when the caller asks, named with
the reason (the manifest's `figures_skipped`).
"""
from __future__ import annotations

from contextlib import contextmanager

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import pitch_report as pr
from analysis.answer_level import BINARY_OUTCOMES, CONTINUOUS_OUTCOMES, COVARIATES as H9_COVARIATES, MODELS as H9_MODELS
from analysis.panel_bottleneck import FDR_LEVEL, H4_SHARE_MOVES, NOTE_COUNT, STAGE_A_FALLBACK_NOTE, roc_points
from analysis.panel_dataset import ALL_QUESTIONS, HARD_QUESTIONS, MODEL_SOURCES, REVIEWED_SOURCES
from analysis.panel_features import FAMILIES

WIDTHS = {"single": 3.5, "double": 7.0}      # inches: one journal column, two journal columns
DEFAULT_WIDTH = "double"
STACK_BELOW = 5.0                             # narrower than this (inches): panels are stacked, not side by side
SAVE_DPI = 300                                # the PNG files
SCREEN_DPI = 150                              # the notebook's inline picture
FONT_SIZE = 8                                 # the smallest text of any figure, in points at the final size
FONTS = ("Arial", "Helvetica", "DejaVu Sans")  # the first one installed is used for every figure
MAX_HEIGHT = 11.0                             # inches: a figure fits on a page

# Okabe and Ito's colour-blind-safe palette; grey marks what should not be read as a result
BLUE, VERMILLION, GREEN, PURPLE, BLACK, ORANGE, SKY = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#000000",
                                                       "#E69F00", "#56B4E9")
GREY, INK, GRID = "#6B6B6B", "#222222", "#D9D9D9"
LINE_COLOURS = (BLUE, VERMILLION, GREEN, PURPLE, BLACK, ORANGE, SKY, GREY)
LINE_DASHES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (7, 2, 1, 2)))
LINE_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
POSITIVE, NEGATIVE, UNCLEAR = (BLUE, ">"), (VERMILLION, "<"), (GREY, "o")       # direction: colour and marker
SIGNIFICANCE_WORDS = ((POSITIVE, True, "positive, survives FDR 5% (filled)"),
                      (NEGATIVE, True, "negative, survives FDR 5% (filled)"),
                      (POSITIVE, False, "p < 0.05 before adjustment only (hollow)"),
                      (UNCLEAR, True, "p of 0.05 or more (grey circle)"))
UNADJUSTED_WORDS = ((POSITIVE, False, "positive, p < 0.05"), (NEGATIVE, False, "negative, p < 0.05"),
                    (UNCLEAR, True, "p of 0.05 or more (grey circle)"))
MOVE_WORDS = {"receipt": "Receipt\n(accepts)", "question": "Another\nquestion", "challenge": "Challenge"}
SUPPORT_STYLE = {"yes": {"color": BLUE, "marker": "o", "filled": True, "label": "answer gives checkable support"},
                 "no": {"color": VERMILLION, "marker": "s", "filled": False, "label": "answer gives none"}}
OFFER_STYLE = {"offered": {"color": BLUE, "marker": "o", "filled": True, "label": "offer"},
               "not_offered": {"color": VERMILLION, "marker": "s", "filled": False, "label": "no offer"}}

FIGURES = {1: ("model_ladder", "Model ladder: skill over the base rate of C0 to C5 with C2p and C3q, and the "
                               "sensitivities"),
           2: ("auc_per_model", "AUC of each model of the ladder"),
           3: ("roc_curves", "ROC curves of the base-rate-adjusted held-out score"),
           4: ("concept_forest", "Each pitch feature on its own"),
           5: ("answer_quality_by_outcome", "Substance and checkable support of the answers, by any offer"),
           6: ("calibration", "Calibration of the held-out predictions"),
           7: ("after_an_answer", "What the panel does after an answer, by checkable support (H4)"),
           8: ("model_coefficients", "Coefficients of the full model C5"),
           9: ("h9_forest", "H9: what follows an answer, coefficients with intervals for all and hard questions"),
           10: ("gains_over_baseline", "Ranking: the baseline C2p and each family's gain over it, alone and cumulative"),
           "S1": ("shark_level_ladder", "Shark-level ladder (archived design): skill over the base rate"),
           "S2": ("gains_over_baseline_skill", "Skill: the baseline C2p and each family's gain over it, alone and cumulative")}
SUPPLEMENTARY = ("S1", "S2")
NO_SHARK_LEVEL = "shark-level path not run"
H9_SKIPPED = "H9 skipped (--skip-h9)"

# the ladder's rungs: what each model adds to the one before it (analysis.panel_bottleneck.LADDER)
LADDER_WORDS = {"C0": "offer rate only", "C1": "+ venture, ask and exposure", "C2": "+ Stage A rank score",
                "C2p": "+ prepared presentation", "C3q": "+ the sharks' questions", "C3": "+ the answers",
                "C4": "+ reactions to the answers", "C5": "+ rapport and repair",
                # the one-family models: the baseline C2p plus one family alone (spec amendment 6.9)
                "C3a": "C2p + the answers only", "C4r": "C2p + reactions only", "C5r": "C2p + rapport and repair only"}
# figure 10: the families compared with the baseline, in ladder order, with their plain names
GAIN_FAMILY_WORDS = {"questions": "The sharks' questions", "answers": "The answers",
                     "reactions": "Reactions to the answers", "rapport_repair": "Rapport and repair"}
BASELINE_WORDS = "Baseline: venture, ask, exposure, delivery and the prepared pitch"
# figure 10, panel B: the skill axis is held to this range so that the models' points (all within it) can be told
# apart; an interval that runs past an edge is cut there and marked with an arrowhead (the full interval is in
# metrics.csv)
SKILL_AXIS_LIMITS = (-10.0, 12.0)
GAIN_STYLE = {"alone": (BLUE, "o", "alone: the baseline plus this family only"),
              "cumulative": (VERMILLION, "s", "cumulative: on top of the families before it")}
FALLBACK_C2_WORDS = "+ Stage A rating (no rank)"
# the same wording for the sensitivity that adds every Stage A item, when the run has no ranking
FALLBACK_ALL_STAGE_A_WORDS = "every Stage A rating in C2 (no rank)"
# the sensitivities of analyze_panel, in its order: words for the legend and the caption, and their look (hollow)
SENSITIVITY_WORDS = {"both_passes": "both passes averaged", "rates": "exposure as rates per minute",
                     "non_development": "without the development videos", "fixed_penalty_10": "penalty fixed at 10",
                     "all_stage_a": "every Stage A rank score in C2"}
SENSITIVITY_STYLE = {"both_passes": (VERMILLION, "s"), "rates": (GREEN, "^"), "non_development": (PURPLE, "D"),
                     "fixed_penalty_10": (ORANGE, "v"), "all_stage_a": (SKY, "P")}
FAMILY_WORDS = {"control": "Venture, team and ask", "exposure": "How much was said",
                "stage_a": "Prepared presentation (Stage A)", "prepared": "Prepared presentation (PRES ratings)",
                "questions": "The sharks' questions", "answers": "The answers", "reactions": "Reactions to the answers",
                "rapport_repair": "Rapport and repair"}
FOREST_TOP_N = {"wide": 20, "narrow": 12}        # features in figure 4 (largest correlation), by width
COEFFICIENT_TOP_N = {"wide": 20, "narrow": 12}   # coefficients in figure 8 (largest), by width
QUESTION_WORDS = {"all questions": "All questions", "hard questions only (severity 2-3)": "Hard questions\n(severity 2 to 3)"}
# H9: the outcomes in the order of answer_level.OUTCOMES, the predictors, and one marker per nested model (the
# exchange-based sensitivity outcomes and the frame covariate q_promotion are never drawn)
H9_OUTCOME_WORDS = {"next_accept": "Next move accepts", "next_challenge": "Next move challenges",
                    "next_further_question": "Next move asks a further question",
                    "next_question_same_frame": "Next question keeps the frame",
                    "next_question_sev": "Severity of the next question",
                    "further_questions_same_frame": "Further questions in the same frame"}
H9_PREDICTOR_WORDS = {"evid": "checkable support (0/1)", "uncert": "admits uncertainty (0/1)",
                      "subs": "substance (per SD)", "coh": "coherence (per SD)", "ctx": "context (per SD; Gemini only)",
                      "repair_before_next": "repair before the next turn (0/1)"}
# at a narrow width (one journal column) the same rows in fewer words; the caption spells them out
H9_NARROW_OUTCOME_WORDS = {"next_accept": "Accepts", "next_challenge": "Challenges",
                           "next_further_question": "Asks a further question",
                           "next_question_same_frame": "Same frame next",
                           "next_question_sev": "Next question's severity",
                           "further_questions_same_frame": "Same-frame run (count)"}
H9_NARROW_PREDICTOR_WORDS = {"evid": "support (0/1)", "uncert": "uncertainty (0/1)", "subs": "substance (SD)",
                             "coh": "coherence (SD)", "ctx": "context (SD; Gemini)", "repair_before_next": "repair (0/1)"}
H9_MARKERS = {"support": "o", "substance": "s", "full": "D", "responsiveness": "^"}
# one axis row per unit: the binary next moves share log-odds; each continuous outcome has its own scale
H9_KINDS = (("binary", BINARY_OUTCOMES, "Log-odds (conditional logit)", "Log-odds"),
            ("next_question_sev", ("next_question_sev",), "Change in the next question's severity (pitch fixed effects)",
             "Change in severity"),
            ("further_questions_same_frame", ("further_questions_same_frame",),
             "Change in the number of further same-frame questions (pitch fixed effects)", "Change in count"))
assert {o for _, outcomes, _, _ in H9_KINDS for o in outcomes} == set(BINARY_OUTCOMES) | set(CONTINUOUS_OUTCOMES)
# why a coefficient has no interval, in three kinds (the CSVs keep the detail)
H9_TOO_FEW, H9_SEPARATED, H9_NOT_ESTIMABLE = "too few estimable draws", "separated", "not estimable"
H9_PANELS = ((ALL_QUESTIONS, "All questions"), (HARD_QUESTIONS, "Hard questions"))
H9_NARROW_PANELS = {ALL_QUESTIONS: "All", HARD_QUESTIONS: "Hard"}


# ---- style (re-exported by analysis.panel_figures) -----------------------------------------------

def font_name() -> str:
    """The sans-serif font every figure uses: the first of FONTS that is installed."""
    from matplotlib import font_manager

    for name in FONTS:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except ValueError:
            continue
    return "sans-serif"


def style() -> dict:
    """The matplotlib settings of every figure (applied while it is built and while it is saved)."""
    return {"font.family": [font_name()], "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE + 1, "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": "white",
            "figure.facecolor": "white", "savefig.facecolor": "white", "axes.edgecolor": INK, "text.color": INK,
            "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "lines.linewidth": 1.2,
            "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": False}


@contextmanager
def _context():
    """The style of every figure while it is built (matplotlib.rc_context), keeping `interactive` as the figure left
    it: a notebook's inline backend turns it on when it draws its first figure, and restoring it on leaving the
    context would show no later figure."""
    import matplotlib

    with matplotlib.rc_context(style()):
        try:
            yield
        finally:
            interactive = matplotlib.is_interactive()
    matplotlib.interactive(interactive)


def width_inches(width=DEFAULT_WIDTH) -> float:
    """A width preset ("single", "double") or a number of inches, as inches."""
    if isinstance(width, str) and width in WIDTHS:
        return float(WIDTHS[width])
    if isinstance(width, (int, float)) and not isinstance(width, bool) and 2.5 <= width <= 12:
        return float(width)
    raise ValueError(f"FIGURE_WIDTH must be one of {', '.join(repr(w) for w in WIDTHS)} or a number of inches "
                     f"between 2.5 and 12, not {width!r}")


def _stacked(width) -> bool:
    return width_inches(width) < STACK_BELOW


def _grid(n: int, width, panel_height: float, extra: float = 0.0, stack: bool | None = None, **kw):
    """A figure with `n` panels: side by side at a wide preset, stacked at a narrow one."""
    import matplotlib.pyplot as plt

    stack = _stacked(width) if stack is None else stack
    rows, cols = (n, 1) if stack else (1, n)
    fig, axes = plt.subplots(rows, cols, figsize=(width_inches(width), panel_height * rows + extra), dpi=SCREEN_DPI,
                             layout="constrained", squeeze=False, **kw)
    return fig, list(axes.ravel()), stack


def _letter(ax, index: int, title: str) -> None:
    ax.set_title(f"{'ABCDEFGH'[index]}  {title}", loc="left", fontweight="bold")


def _point(ax, x, y, colour: str, marker: str, filled: bool = True, size: float = 5.0, **kw) -> None:
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=size, color=colour, markeredgewidth=1.1,
            markerfacecolor=colour if filled else "white", zorder=3, **kw)


def _handle(colour: str, marker: str, filled: bool, label: str, dash="-", line: bool = True):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=colour, marker=marker, markersize=5, markeredgewidth=1.1, linewidth=1.2 if line else 0,
                  linestyle=dash if line else "none", markerfacecolor=colour if filled else "white", label=label)


def _reference_ticks(ax, at: float, word: str) -> None:
    """Name the reference line in its own tick label, where nothing can collide with it."""
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}\n{word}" if abs(v - at) < 1e-9 else f"{v:g}"))


def _wrap(text: str, chars: int) -> str:
    return "\n".join(textwrap.wrap(text, chars)) or text


def plain(label: str) -> str:
    """A feature label of the dictionary as figure text: no underscores, first letter capital."""
    text = str(label).replace("_", " ")
    return text[:1].upper() + text[1:]


LINE_INCHES, ROW_PAD = 0.135, 0.04


def _layout(entries: list[tuple[str, str, object]]) -> tuple[list[float], float]:
    """Centres (inches from the top) of rows whose labels have one or more lines, and the total height."""
    centres, top = [], 0.0
    for kind, text, _ in entries:
        height = LINE_INCHES * (text.count("\n") + 1) + ROW_PAD + (0.05 if kind == "header" else 0.0)
        centres.append(top + height / 2 + (0.05 if kind == "header" else 0.0))
        top += height
    return centres, top


def _label_rows(ax, entries, centres, total: float, labels: bool = True) -> None:
    ax.set_ylim(total, 0)
    ax.set_yticks(centres)
    ax.set_yticklabels([text for _, text, _ in entries])
    ax.tick_params(axis="y", length=0, labelleft=labels)
    for tick, (kind, _, _) in zip(ax.get_yticklabels(), entries):
        if kind == "header":
            tick.set_fontweight("bold")
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def significance_look(value: float, p, survives=None) -> tuple[str, str, bool]:
    """(colour, marker, filled) of an estimate. An unadjusted p of 0.05 or more (or none): grey circle.
    Below 0.05: a triangle whose colour and direction say the sign (blue pointing right = positive,
    vermillion pointing left = negative), filled when the estimate also survives the false-discovery
    adjustment (`survives`), hollow when it does not or when no adjustment applies (`survives=None`)."""
    if p is None or p != p or p >= 0.05:
        return (*UNCLEAR, True)
    return (*(POSITIVE if value > 0 else NEGATIVE), bool(survives))


def _interval_point(ax, value, low, high, y, p, survives=None) -> None:
    colour, marker, filled = significance_look(value, p, survives)
    ax.plot([low, high], [y, y], color=colour, linewidth=1.2, solid_capstyle="butt", zorder=2)
    _point(ax, value, y, colour, marker, filled, size=5.5 if marker != "o" else 4.5)


def _significance_handles(words=SIGNIFICANCE_WORDS):
    return [_handle(look[0], look[1], filled, text) for look, filled, text in words]


def save_figure(fig, folder: Path, names: dict, text: str) -> list[str]:
    """Save `fig` as PNG (SAVE_DPI) and vector PDF under `names` in `folder`, close it, and write its caption file;
    return the three file names."""
    import matplotlib.pyplot as plt

    with _context():
        fig.savefig(folder / names["png"], dpi=SAVE_DPI)
        fig.savefig(folder / names["pdf"])
    plt.close(fig)
    (folder / names["caption"]).write_text(text + "\n")
    return [names["png"], names["pdf"], names["caption"]]


# ---- figure 7: what the panel does after an answer (H4; shared with the legacy figures) ------------

def figure_7_after_an_answer(a, width=DEFAULT_WIDTH):
    """The share of answers followed by a receipt, another question or a challenge as the panel's first
    reaction, for answers with checkable support (filled blue circle) and without (hollow vermillion
    square), with 95% pitch-bootstrap intervals. Panel A: answers to all questions; panel B: to hard
    questions. Above each pair: the difference with minus without, in percentage points, with its interval."""
    table = a.h4_shares
    if table is None or table.empty:
        return None
    with _context():
        fig, axes, stack = _grid(2, width, 2.9, extra=0.4, sharey=True)
        top = 100 * float(table[["yes_high", "no_high", "share_yes", "share_no"]].max().max())
        for index, (ax, (questions, title)) in enumerate(zip(axes, ((ALL_QUESTIONS, "After answers to all questions"),
                                                                    (HARD_QUESTIONS, "After answers to hard questions")))):
            _letter(ax, index, title)
            part = table[table.questions == questions].set_index("move")
            moves = [m for m in H4_SHARE_MOVES if m in part.index]
            for at, move in enumerate(moves):
                r = part.loc[move]
                for offset, key in ((-0.14, "yes"), (0.14, "no")):
                    look, share = SUPPORT_STYLE[key], r[f"share_{key}"]
                    if share != share:
                        continue
                    ax.plot([at + offset] * 2, [100 * r[f"{key}_low"], 100 * r[f"{key}_high"]], color=look["color"],
                            linewidth=1.2, linestyle="-" if key == "yes" else (0, (3, 2)), zorder=2)
                    _point(ax, at + offset, 100 * share, look["color"], look["marker"], look["filled"])
                if r.difference == r.difference:
                    ax.text(at, top * 1.34, f"{100 * r.difference:+.0f}{pr.stars(r.p_value)}\n"
                            f"[{100 * r.diff_low:+.0f}, {100 * r.diff_high:+.0f}]",
                            ha="center", va="top", linespacing=1.15)
            ax.set_xticks(range(len(moves)))
            ax.set_xticklabels([MOVE_WORDS.get(m, plain(m)) for m in moves])
            ax.set_xlim(-0.55, len(moves) - 0.45)
            ax.set_ylim(0, top * 1.38)
            n = part.iloc[0] if len(part) else None
            ax.set_xlabel("" if n is None else f"The panel's first reaction\n(answers: {int(n.n_yes)} with support, "
                          f"{int(n.n_no)} without)")
            ax.tick_params(axis="y", labelleft=True)
            if stack or index == 0:
                ax.set_ylabel("Share of answers (%)")
            ax.grid(axis="y", color=GRID, linewidth=0.5)
            ax.set_axisbelow(True)
        handles = [_handle(s["color"], s["marker"], s["filled"], s["label"], "-" if key == "yes" else (0, (3, 2)))
                   for key, s in SUPPORT_STYLE.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=1 if stack else 2)
    return fig


# ---- the sample behind the figures ----------------------------------------------------------------

def _estimated(a) -> bool:
    return a.result.get("status") == "estimated"


def _degenerate(a) -> str:
    """Why the outcome cannot be predicted at all (every known pitch with an offer, or none); empty when it varies."""
    n, events = int(a.result.get("n_groups", 0)), int(a.result.get("n_events", 0))
    return "" if 0 < events < n else f"the outcome does not vary ({events} of {n} pitches with an offer)"


def sample_sentence(a) -> str:
    """The sample, the coding and the label, as every caption states them."""
    sources = a.data.videos.source
    model_coded, reviewed = int(sources.isin(MODEL_SOURCES).sum()), int(sources.isin(REVIEWED_SOURCES).sum())
    coded = ("All annotations are model-coded (no human review)" if not reviewed else
             f"{model_coded} pitches are model-coded and {reviewed} human-reviewed")
    demo = (" DEMONSTRATION ONLY: with fewer than 20 pitches the figure shows that the method runs, not a result."
            if str(a.result.get("label", "")).startswith("demonstration") else "")
    return (f"Sample: {int(a.result.get('n_groups', 0))} pitches with a known outcome, "
            f"{int(a.result.get('n_events', 0))} with at least one offer, of {len(a.data.videos)} analysed. {coded}; "
            f"the results are exploratory, not confirmatory.{demo}")


def interval_sentence(a, what: str = "Bars", fixed: bool = False, draws: int | None = None) -> str:
    tail = (" of the fixed held-out predictions (models are not refitted in a draw, so the intervals understate "
            "the uncertainty)" if fixed else "")
    draws = int(a.n_boot if draws is None else draws)
    return f"{what} are 95% percentile intervals from resampling whole pitches ({draws} draws){tail}."


def test_sentence(a) -> str:
    """Which test the stars beside a model come from."""
    if pr.p_kind(a.result) == "permutation":
        return (f"Stars come from the pitch-level permutation test with {a.n_perm} permutations (the pitch outcomes "
                "are shuffled across pitches and the whole leave-one-pitch-out fit is repeated); "
                f"p = (1 + permutations at least as good) / (1 + {a.n_perm}). {pr.STARS_NOTE}.")
    return (f"Stars are two-sided bootstrap p-values from the same {a.n_boot} resamples of pitches as the intervals "
            f"(no permutation test was run). {pr.STARS_NOTE}.")


AUC_SCORE_SENTENCE = ("AUCs rank the base-rate-adjusted held-out score logit(p) - logit(b), b the training base "
                      "rate of the pitch's fold, which reduces the leave-one-pitch-out artefact of a model near the "
                      "base rate (a residual bias against weak models remains: a weak model can still score below "
                      "0.5); they are not comparable with the AUCs of earlier (v3-dev8) analyses")
EXPLORATORY_SENTENCE = ("The p-values describe this sample of model-coded videos; they are not confirmatory tests "
                        "(the pre-registered verdicts are in report.md).")


# ---- figures 1 and 2: the ladder ------------------------------------------------------------------

LADDER_COLUMNS = ["model", "label", "skill_pct", "skill_low", "skill_high", "auc", "auc_low", "auc_high",
                  "auc_meaningful", "skill_p", "auc_p"]


def rating_fallback(a) -> bool:
    """Whether C2 read the Stage A rating because the run has no ranking (`a.stage_a_ranking`, M1 of plan B's final
    review): the figures then never say "rank score"."""
    return getattr(a, "stage_a_ranking", None) == pr.analyze_panel().RANKING_ABSENT


def sensitivity_words(a) -> dict[str, str]:
    """SENSITIVITY_WORDS, with the all-Stage-A sensitivity worded for the rating when the run has no ranking."""
    return {**SENSITIVITY_WORDS, **({"all_stage_a": FALLBACK_ALL_STAGE_A_WORDS} if rating_fallback(a) else {})}


def not_run_sensitivities(a) -> dict[str, str]:
    """{setting: why} for each sensitivity of the analysis that was not run (I2 of plan B's final review: the one
    without the development videos when it would equal the main analysis); no marker is drawn for these."""
    return {name: str(ev.get("status")) for name, ev in (getattr(a, "sensitivities", None) or {}).items()
            if str(ev.get("status", "")).startswith("not run")}


def ladder_label(model: str, note: str = "", fallback: bool = False) -> str:
    """The rung's label; C2 says "rating" when `fallback` (the run has no ranking) or the note says so."""
    rating = fallback or STAGE_A_FALLBACK_NOTE in str(note)
    words = FALLBACK_C2_WORDS if model == "C2" and rating else LADDER_WORDS.get(model, "")
    return f"{model}  {words}".strip()


def ladder_table(a) -> pd.DataFrame:
    """What figures 1 and 2 draw: the ladder's metrics, unchanged, with the label of each rung and the p-value shown
    beside its skill and its AUC (`pitch_report.model_p`: the permutation p when the test ran, else the bootstrap p)."""
    if not _estimated(a):
        return pd.DataFrame(columns=LADDER_COLUMNS)
    m = a.result["metrics"]
    records = m.to_dict("records")
    fallback = rating_fallback(a)
    return m.assign(label=[ladder_label(r["model"], r.get("note", ""), fallback) for r in records],
                    skill_p=[pr.model_p(r, "log_loss")[0] for r in records],
                    auc_p=[pr.model_p(r, "auc")[0] for r in records])[LADDER_COLUMNS].reset_index(drop=True)


def sensitivity_table(a) -> pd.DataFrame:
    """The sensitivities' metrics of the same models, one row per setting and model (estimated settings only)."""
    parts = [ev["metrics"][["model", "skill_pct", "auc", "auc_meaningful"]].assign(setting=name)
             for name, ev in (a.sensitivities or {}).items() if ev.get("status") == "estimated"]
    return (pd.concat(parts, ignore_index=True) if parts else
            pd.DataFrame(columns=["model", "skill_pct", "auc", "auc_meaningful", "setting"]))


def _ladder_axis(ax, labels: list[str], reference: float, word: str, xlabel: str) -> None:
    ax.axvline(reference, color=INK, linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    _reference_ticks(ax, reference, word)
    ax.set_xlabel(xlabel)


def figure_1_model_ladder(a, width=DEFAULT_WIDTH):
    """Skill over the base rate (%) of each model of the ladder, C0 to C5 with C2p and C3q: the main estimate as a
    filled blue circle with its 95% interval and the stars of its p-value, and each sensitivity as a hollow marker
    of its own shape and colour just below it on the same axis (no interval). A line at 0: the offer rate."""
    import matplotlib.pyplot as plt

    table = ladder_table(a)
    if table.empty:
        return None
    sens = sensitivity_table(a)
    settings = [s for s in SENSITIVITY_STYLE if s in set(sens.setting)]
    slots = 1 + len(settings)
    offsets = (np.arange(slots) - (slots - 1) / 2) * (0.76 / max(slots - 1, 1))
    stack = _stacked(width)
    with _context():
        fig, ax = plt.subplots(figsize=(width_inches(width), 0.5 * len(table) + (2.4 if stack else 1.7)),
                               dpi=SCREEN_DPI, layout="constrained")
        for i, row in enumerate(table.itertuples()):
            y = i + offsets[0]
            ax.plot([row.skill_low, row.skill_high], [y, y], color=BLUE, linewidth=1.2, solid_capstyle="butt", zorder=2)
            _point(ax, row.skill_pct, y, BLUE, "o", True)
            mark = pr.stars(row.skill_p) if a.models.get(row.model) else ""      # the base rate has no test
            if mark:
                ax.annotate(mark, (max(row.skill_high, row.skill_pct), y), xytext=(3, -1), textcoords="offset points",
                            va="center", ha="left")
            for k, setting in enumerate(settings):
                part = sens[(sens.setting == setting) & (sens.model == row.model)]
                if len(part) and part.skill_pct.iloc[0] == part.skill_pct.iloc[0]:
                    colour, marker = SENSITIVITY_STYLE[setting]
                    _point(ax, float(part.skill_pct.iloc[0]), i + offsets[k + 1], colour, marker, False, size=4.2)
        _ladder_axis(ax, [_wrap(t, 28 if stack else 60) for t in table.label], 0.0, "offer rate",
                     "Skill over the base rate, %\n(higher is better)" if stack else
                     "Skill over the base rate, % (higher is better)")
        ax.margins(x=0.1)
        words = sensitivity_words(a)
        handles = [_handle(BLUE, "o", True, "main analysis, with its 95% interval")]
        handles += [_handle(*SENSITIVITY_STYLE[s], False, f"sensitivity: {words[s]}", line=False)
                    for s in settings]
        from matplotlib.lines import Line2D
        handles += [Line2D([], [], linestyle="none", label=f"sensitivity: {words.get(s, s)}: {why}")
                    for s, why in not_run_sensitivities(a).items()]          # named, no marker
        handles.append(Line2D([], [], linestyle="none", label=f"stars: {pr.p_kind(a.result)} p, {pr.STARS_NOTE}"))
        fig.legend(handles=handles, loc="outside lower center", ncol=1 if stack else 2, handlelength=1.6)
    return fig


NEAR_CONSTANT_WORDS = "near-constant predictions (diagnostic only)"


def figure_2_auc(a, width=DEFAULT_WIDTH):
    """AUC with its 95% interval per model of the ladder; a line at 0.5 (chance); stars for the p-value of the AUC.
    Every finite AUC is drawn (ruling I3 of plan B's final review: `auc_meaningful` is a diagnostic, not a filter):
    a model with `auc_meaningful` False (predictions within about 3 points of the offer rate) is drawn hollow and
    grey, its AUC still counting. The base-rate model ranks nothing; "base rate: no ranking" is written in the right
    margin, outside the AUC axis, as is "no AUC" for a model whose AUC is missing."""
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory

    table = ladder_table(a)
    if table.empty:
        return None
    stack = _stacked(width)
    with _context():
        fig, ax = plt.subplots(figsize=(width_inches(width), 0.32 * len(table) + (1.9 if stack else 1.4)),
                               dpi=SCREEN_DPI, layout="constrained")
        margin = blended_transform_factory(ax.transAxes, ax.transData)
        hollow = False
        for i, row in enumerate(table.itertuples()):
            base = not a.models.get(row.model)
            if base or row.auc != row.auc:                     # off the data: in the right margin
                words = ("base rate:\nno ranking" if stack else "base rate: no ranking") if base else "no AUC"
                ax.text(1.02, i, words, color=GREY, style="italic", va="center", ha="left", linespacing=1.0,
                        clip_on=False, transform=margin)
                continue
            real = bool(row.auc_meaningful)
            hollow |= not real
            colour = BLUE if real else GREY
            ax.plot([row.auc_low, row.auc_high], [i, i], color=colour, linewidth=1.2, solid_capstyle="butt", zorder=2)
            _point(ax, row.auc, i, colour, "o", real)
            mark = pr.stars(row.auc_p)
            if mark:
                ax.annotate(mark, (row.auc_high, i), xytext=(3, -1), textcoords="offset points", va="center", ha="left")
        _ladder_axis(ax, [_wrap(t, 28 if stack else 60) for t in table.label], 0.5, "chance", "AUC (higher is better)")
        ax.set_xlim(-0.03, 1.03)                 # a point at 0 or 1 is drawn whole, not cut by the frame
        ax.set_xticks([0.0, 0.5, 1.0])
        from matplotlib.lines import Line2D
        handles = [_handle(BLUE, "o", True, "estimate with its 95% interval")]
        if hollow:
            handles.append(_handle(GREY, "o", False, NEAR_CONSTANT_WORDS))
        handles.append(Line2D([], [], linestyle="none", label=f"stars: {pr.p_kind(a.result)} p, {pr.STARS_NOTE}"))
        fig.legend(handles=handles, loc="outside lower center", ncol=1 if stack else 2, handlelength=1.6)
    return fig


# ---- figure 10: each family's gain over the baseline C2p, alone and cumulative ----------------------

def baseline_gain_table(a) -> pd.DataFrame:
    """What figure 10 draws: `panel_bottleneck.baseline_gains` of the main evaluation (one row per family and
    metric, the gain over C2p alone and cumulative, with intervals and bootstrap p)."""
    from analysis.panel_bottleneck import BASELINE_GAIN_COLUMNS, baseline_gains

    if not _estimated(a):
        return pd.DataFrame(columns=BASELINE_GAIN_COLUMNS)
    return baseline_gains(a.result["deltas"], a.result["metrics"])


def _clipped_interval(ax, low, high, y, colour, limits) -> None:
    """A horizontal 95% interval from `low` to `high` at height `y`; with `limits` (x0, x1) the part beyond an edge
    is cut and an arrowhead at that edge says the interval continues."""
    x0, x1 = limits if limits else (-np.inf, np.inf)
    left, right = max(low, x0), min(high, x1)
    ax.plot([left, right], [y, y], color=colour, linewidth=1.2, solid_capstyle="butt", zorder=2)
    for edge, marker, cut in ((x0, "<", low < x0), (x1, ">", high > x1)):
        if cut:
            ax.plot([edge], [y], marker=marker, color=colour, markersize=5, linestyle="none", clip_on=False, zorder=3)


GAIN_PANELS = {
    # metric: (metrics column, low, high, x label, reference value and word, scale, digits, unit, axis limits)
    "auc": ("auc", "auc_low", "auc_high", "Pairs ranked correctly, per 100\n(one funded and one unfunded pitch; AUC x 100)",
            50.0, "chance", 100.0, 0, "%", None),
    "skill": ("skill_pct", "skill_low", "skill_high", "Skill over the base rate, %\n(0 = guessing the offer rate for every pitch)",
              0.0, "offer rate", 1.0, 1, "%", SKILL_AXIS_LIMITS)}


def _gains_figure(a, metric: str, width=DEFAULT_WIDTH):
    """One panel: the baseline model C2p in the first row (black diamond) and, per family, the family entering alone
    (filled blue circle) and in the cumulative ladder (hollow vermillion square), each at its own level of `metric`
    with its 95% interval. A solid line marks the baseline's value, a dotted line chance (AUC) or the offer rate
    (skill); beside each family's point, its gain over the baseline (as a percentage) and the bootstrap p of that
    gain. For the questions the two models are the same (C3q), drawn once."""
    from analysis.panel_bottleneck import BASELINE

    table = baseline_gain_table(a)
    if table.empty:
        return None
    m = a.result["metrics"].set_index("model")
    if BASELINE not in m.index:
        return None
    column, low, high, xlabel, chance, chance_word, scale, digits, unit, limits = GAIN_PANELS[metric]
    families = [f for f in GAIN_FAMILY_WORDS if f in set(table.family)]
    labels = [BASELINE_WORDS] + [GAIN_FAMILY_WORDS[f] for f in families]
    part = table[table.metric == metric].set_index("family")
    base = scale * float(m.loc[BASELINE, column])
    with _context():
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.ticker import MultipleLocator

        stack = _stacked(width)
        fig, ax = plt.subplots(figsize=(width_inches(width), 0.7 * len(labels) + (2.6 if stack else 2.0)),
                               dpi=SCREEN_DPI, layout="constrained")
        ax.axvline(chance, color=GREY, linewidth=0.8, linestyle=(0, (1, 2)), zorder=1)
        _clipped_interval(ax, scale * m.loc[BASELINE, low], scale * m.loc[BASELINE, high], 0, INK, limits)
        _point(ax, base, 0, INK, "D", True, size=5.5)
        ax.annotate(f"{base:.{digits}f}{unit}", (1.05, 0), xycoords=("axes fraction", "data"), va="center",
                    ha="left", fontsize=8, color=INK, clip_on=False)
        for i, family in enumerate(families, start=1):
            r = part.loc[family]
            same = r.alone_model == r.cumulative_model
            kinds = (("alone", 0.0),) if same else (("alone", -0.18), ("cumulative", 0.18))
            for kind, offset in kinds:
                model, delta, p = getattr(r, f"{kind}_model"), getattr(r, f"{kind}_delta"), getattr(r, f"{kind}_p")
                if model not in m.index or m.loc[model, column] != m.loc[model, column]:
                    continue
                colour, marker, _ = GAIN_STYLE[kind]
                y = i + offset
                _clipped_interval(ax, scale * m.loc[model, low], scale * m.loc[model, high], y, colour, limits)
                _point(ax, scale * float(m.loc[model, column]), y, colour, marker, kind == "alone")
                gain = f"{scale * delta:+.{digits}f}{unit}" if delta == delta else "no gain computed"
                ax.annotate(f"{kind}: {gain}, p = {p:.3f}" if p == p else f"{kind}: {gain}", (1.05, y),
                            xycoords=("axes fraction", "data"), va="center", ha="left", fontsize=8, color=colour,
                            clip_on=False)
        ax.axvline(base, color=INK, linewidth=0.8, zorder=1)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([_wrap(t, 24 if stack else 34) for t in labels])
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        if limits:
            ax.set_xlim(*limits)
        else:
            ax.margins(x=0.12)
        handles = [_handle(INK, "D", True, "the baseline, with its 95% interval")]
        handles += [_handle(colour, marker, kind == "alone", words) for kind, (colour, marker, words) in GAIN_STYLE.items()]
        handles += [Line2D([], [], color=INK, linewidth=0.8, label="solid line: the baseline's value"),
                    Line2D([], [], color=GREY, linewidth=0.8, linestyle=(0, (1, 2)), label=f"dotted line: {chance_word}")]
        if limits:
            handles.append(Line2D([], [], color=GREY, marker=">", linestyle="none", markersize=5,
                                  label="arrowhead: the interval continues past the edge"))
        fig.legend(handles=handles, loc="outside lower center", ncol=1, handlelength=1.6)
    return fig


def figure_10_gains(a, width=DEFAULT_WIDTH):
    """Figure 10: the ranking panel (AUC as pairs ranked correctly per 100) of `_gains_figure`."""
    return _gains_figure(a, "auc", width)


def figure_s2_gains_skill(a, width=DEFAULT_WIDTH):
    """Supplementary figure S2: the skill panel of `_gains_figure`, held to SKILL_AXIS_LIMITS."""
    return _gains_figure(a, "skill", width)


# ---- figures 3 and 6: ROC curves and calibration ----------------------------------------------------

def _curve_models(a) -> list[str]:
    return pr.roc_models(a.result) if _estimated(a) else []


def _auc_words(metrics: pd.DataFrame, name: str) -> str:
    """The legend words of an ROC curve: its AUC and that AUC's p-value; "near-constant" when `auc_meaningful` is
    False (a diagnostic mark, ruling I3); "no AUC" when it is missing."""
    if metrics.auc[name] != metrics.auc[name]:
        return "no AUC"
    p = pr.model_p(metrics.loc[name], "auc")[0]
    near = "" if bool(metrics.auc_meaningful[name]) else ", near-constant"
    return f"AUC {metrics.auc[name]:.2f}" + (f", {pr.p_phrase(p)}" if p == p else "") + near


def _curve_figure(width, entries: int):
    """One square panel with a strip below it for its legend: (figure, panel, strip)."""
    import matplotlib.pyplot as plt

    narrow = _stacked(width)
    side = min(width_inches(width) - 1.1, 3.4)          # room on the left for the two-line y label
    strip = 0.17 * -(-entries // (1 if narrow else 2)) + 0.15
    fig = plt.figure(figsize=(width_inches(width), side + 0.55 + strip + 0.2), dpi=SCREEN_DPI, layout="constrained")
    grid = fig.add_gridspec(2, 1, height_ratios=[side + 0.55, strip])
    ax, legend = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    ax.set_box_aspect(1)
    legend.axis("off")
    return fig, ax, legend


def _curve_legend(ax, strip, narrow: bool) -> None:
    handles, labels = ax.get_legend_handles_labels()
    strip.legend(handles, labels, loc="upper center", ncol=1 if narrow else 2, handlelength=3.4, columnspacing=1.0,
                 borderaxespad=0.0)


def roc_column(pred: pd.DataFrame, name: str) -> str:
    """The column an ROC curve of model `name` ranks: its base-rate-adjusted score when the evaluation wrote one (the
    same score its AUC ranks), else its probability."""
    return f"score_{name}" if f"score_{name}" in pred else f"p_{name}"


def figure_3_roc(a, width=DEFAULT_WIDTH):
    """ROC curves of the held-out base-rate-adjusted scores (`roc_column`) of every model with features, one dash
    pattern and colour each, with its AUC and that AUC's p-value in the legend; a near-constant model
    (`auc_meaningful` False, a diagnostic only) is drawn in its own colour like the others, "near-constant" in its
    legend entry."""
    models = _curve_models(a)
    if not models:
        return None
    pred, m = a.result["predictions"], a.result["metrics"].set_index("model")
    with _context():
        fig, ax, strip = _curve_figure(width, 1 + len(models))
        ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.8, linestyle=(0, (1, 2)), label="chance (AUC 0.50)")
        for name, colour, dash in zip(models, LINE_COLOURS, LINE_DASHES):
            fpr, tpr = roc_points(pred["y"], pred[roc_column(pred, name)], pred["weight"])
            ax.plot(fpr, tpr, color=colour, linestyle=dash, linewidth=1.4, label=f"{name} ({_auc_words(m, name)})")
        ax.set_xlabel("False-positive rate\n(share of pitches without an offer)")
        ax.set_ylabel("True-positive rate\n(share of pitches with an offer)")
        for set_lim, set_ticks in ((ax.set_xlim, ax.set_xticks), (ax.set_ylim, ax.set_yticks)):
            set_lim(0, 1)
            set_ticks([0, 0.5, 1])
        _curve_legend(ax, strip, _stacked(width))
    return fig


def calibration_table(a, bins: int = 4) -> pd.DataFrame:
    """What figure 6 draws: `pitch_report.calibration_points` of the models with features."""
    models = _curve_models(a)
    return (pr.calibration_points(a.result, models, bins) if models else
            pd.DataFrame(columns=["model", "group", "p", "y", "n"]))


def figure_6_calibration(a, width=DEFAULT_WIDTH, bins: int = 4):
    """Predicted against observed offer rate: per model the pitches are sorted by held-out prediction into `bins`
    equal groups; a point is a group's mean prediction against its share of pitches with an offer."""
    table = calibration_table(a, bins)
    if table.empty:
        return None
    models = _curve_models(a)
    with _context():
        fig, ax, strip = _curve_figure(width, 1 + len(models))
        ax.plot([0, 1], [0, 1], color=GREY, linewidth=0.8, linestyle=(0, (1, 2)), label="perfect calibration")
        for name, colour, dash, marker in zip(models, LINE_COLOURS, LINE_DASHES, LINE_MARKERS):
            points = table[table.model == name]
            ax.plot(points.p, points.y, color=colour, linestyle=dash, linewidth=1.1, marker=marker, markersize=4.5,
                    markerfacecolor="white", markeredgewidth=1.1, label=name)
        ax.set_xlabel("Mean predicted probability of an offer")
        ax.set_ylabel("Observed share of pitches with an offer")
        for set_lim, set_ticks in ((ax.set_xlim, ax.set_xticks), (ax.set_ylim, ax.set_yticks)):
            set_lim(0, 1)
            set_ticks([0, 0.5, 1])
        _curve_legend(ax, strip, _stacked(width))
    return fig


# ---- figures 4 and 8: forests of features ---------------------------------------------------------

def _top_n(table: dict, width) -> int:
    return table["narrow" if _stacked(width) else "wide"]


def forest_rows(a, top_n: int = FOREST_TOP_N["wide"]) -> pd.DataFrame:
    """The rows of the single-feature table figure 4 draws: the `top_n` features with the largest correlation in
    size (counts behind a mean left out: they repeat the number of questions asked)."""
    table = a.single
    if table is None or not len(table):
        return pd.DataFrame(columns=getattr(table, "columns", []))
    kept = table[(table.note != NOTE_COUNT) & table.correlation.notna()]
    order = kept.correlation.abs().sort_values(ascending=False, kind="stable").index
    return kept.loc[order].head(max(0, int(top_n))).reset_index(drop=True)


def coefficient_rows(a, top_n: int = COEFFICIENT_TOP_N["wide"]) -> pd.DataFrame:
    """The rows of C5's coefficient table figure 8 draws: the `top_n` largest in size."""
    table = a.coefficients
    if table is None or not len(table):
        return pd.DataFrame(columns=getattr(table, "columns", []))
    order = table.coefficient.abs().sort_values(ascending=False, kind="stable").index
    return table.loc[order].head(max(0, int(top_n))).reset_index(drop=True)


def family_entries(rows: pd.DataFrame, value: str, chars: int) -> list[tuple[str, str, object]]:
    """Group headings and rows in drawing order: families in the order of the feature dictionary, within a family
    the largest `value` in size first."""
    entries = []
    for family in [f for f in FAMILIES if f in set(rows.family)] + sorted(set(rows.family) - set(FAMILIES)):
        entries.append(("header", _wrap(FAMILY_WORDS.get(family, plain(family)), chars), None))
        part = rows[rows.family == family]
        for row in part.reindex(part[value].abs().sort_values(ascending=False, kind="stable").index).itertuples():
            entries.append(("row", _wrap(plain(row.label), chars), row))
    return entries


def _forest(width, entries, draw, xlabel: str, handles, title: str):
    import matplotlib.pyplot as plt

    centres, total = _layout(entries)
    stack = _stacked(width)
    with _context():
        fig, ax = plt.subplots(figsize=(width_inches(width), total + (2.0 if stack else 1.4)), dpi=SCREEN_DPI,
                               layout="constrained")
        # the title spans the figure, not the plot area, so it never runs past the right edge
        fig.suptitle(_wrap(title, 44 if stack else 90), x=0.01, ha="left", fontweight="bold", fontsize=FONT_SIZE + 1)
        ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
        for (kind, _, row), y in zip(entries, centres):
            if kind == "row":
                draw(ax, row, y)
        _label_rows(ax, entries, centres, total)
        ax.set_xlabel(xlabel)
        ax.margins(x=0.08)
        fig.legend(handles=handles, loc="outside lower center", ncol=1 if stack else 2)
    return fig


def figure_4_concepts(a, width=DEFAULT_WIDTH, top_n: int | None = None):
    """Forest plot: the correlation of each pitch feature with any offer and its 95% interval, one row per feature,
    grouped by family and sorted by size within it. Direction is in the colour and the marker; filled survives the
    false-discovery adjustment, hollow has an unadjusted p below 0.05 only, grey has neither."""
    stack = _stacked(width)
    rows = forest_rows(a, _top_n(FOREST_TOP_N, width) if top_n is None else top_n)
    if not len(rows):
        return None
    entries = family_entries(rows, "correlation", 30 if stack else 60)
    return _forest(width, entries,
                   lambda ax, r, y: _interval_point(ax, r.correlation, r.corr_low, r.corr_high, y, r.p_value,
                                                    r.significant_fdr),
                   "Correlation with any offer\n(right of 0: goes with an offer)" if stack else
                   "Correlation with any offer (right of 0: more of it goes with an offer)",
                   _significance_handles(), f"The {len(rows)} features with the largest correlation")


def figure_8_coefficients(a, width=DEFAULT_WIDTH, top_n: int | None = None):
    """Forest plot of the largest standardised coefficients of C5 refitted on all pitches with a known outcome,
    each with a 95% pitch-bootstrap interval, grouped by feature family. A hollow triangle has an unadjusted
    bootstrap p below 0.05 (colour and direction say the sign), a grey circle has not."""
    stack = _stacked(width)
    rows = coefficient_rows(a, _top_n(COEFFICIENT_TOP_N, width) if top_n is None else top_n)
    if not len(rows):
        return None
    entries = family_entries(rows, "coefficient", 30 if stack else 60)
    return _forest(width, entries,
                   lambda ax, r, y: _interval_point(ax, r.coefficient, r.ci_low, r.ci_high, y, r.p_value),
                   "Coefficient (log-odds per 1 SD,\npenalised, conditional)" if stack else
                   "Coefficient (log-odds of an offer per 1 SD; penalised, conditional)",
                   _significance_handles(UNADJUSTED_WORDS), f"C5 refitted on all pitches: the {len(rows)} largest")


# ---- figure 5: answer quality by any offer --------------------------------------------------------

BY_OUTCOME_PANELS = (("substance", "means", "Substance", "Mean substance of\nthe answers (1 to 5)"),
                     ("substance", "correlation", "Correlation", "Correlation of substance\nwith any offer"),
                     ("support", "means", "Support", "Answers with checkable\nsupport (%)"),
                     ("support", "correlation", "Correlation", "Correlation of support\nwith any offer"))


def by_outcome_table(a) -> pd.DataFrame:
    """What figure 5 draws: `answer_quality_by_offer` of the pitch rows (pooled over all sharks' questions), with
    the kind of measure (substance, support)."""
    t = getattr(a, "by_offer", None)
    if t is None or not len(t):
        return pd.DataFrame()
    return t.assign(kind=["support" if str(m).startswith("share") else "substance" for m in t.measure])


def by_outcome_entries(table: pd.DataFrame) -> list[tuple[str, str, object]]:
    entries = []
    for questions in dict.fromkeys(table.questions):
        part = table[table.questions == questions]
        n = " / ".join(str(int(part[part.kind == k].n.iloc[0])) for k in ("substance", "support") if (part.kind == k).any())
        entries.append(("row", f"{QUESTION_WORDS.get(questions, plain(questions))}\nn = {n}", part.set_index("kind")))
    return entries


def figure_5_answer_quality(a, width=DEFAULT_WIDTH):
    """Answer quality by any offer, for all and for hard questions, pooled over all sharks' questions before the
    cut. Panels A and C: the mean among pitches with an offer (filled blue circle) and without (hollow vermillion
    square), for substance and for checkable support. Panels B and D: the correlation with any offer and its 95%
    interval, direction in colour and marker."""
    import matplotlib.pyplot as plt

    table = by_outcome_table(a)
    if table.empty or not table.correlation.notna().any():
        return None
    stack = _stacked(width)
    entries = by_outcome_entries(table)
    centres, total = _layout(entries)
    total += 0.05
    with _context():
        rows, cols = (2, 2) if stack else (1, 4)
        fig, grid = plt.subplots(rows, cols, figsize=(width_inches(width), rows * (total + 0.8) + 0.95 + 0.4 * stack),
                                 dpi=SCREEN_DPI, layout="constrained", sharey=True, squeeze=False)
        for index, (ax, (kind, what, title, xlabel)) in enumerate(zip(grid.ravel(), BY_OUTCOME_PANELS)):
            _letter(ax, index, title)
            scale = 100.0 if kind == "support" and what == "means" else 1.0
            if what == "correlation":
                ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
            for (_, _, part), y in zip(entries, centres):
                if kind not in part.index:
                    continue
                r = part.loc[kind]
                if what == "correlation":
                    if r.correlation == r.correlation:
                        _interval_point(ax, r.correlation, r.corr_low, r.corr_high, y, r.p_value, r.significant_fdr)
                    continue
                ax.plot([r.mean_not_offered * scale, r.mean_offered * scale], [y, y], color=GREY, linewidth=0.9, zorder=1)
                for column, look in (("mean_not_offered", OFFER_STYLE["not_offered"]),
                                     ("mean_offered", OFFER_STYLE["offered"])):
                    _point(ax, getattr(r, column) * scale, y, look["color"], look["marker"], look["filled"], size=4.5)
            _label_rows(ax, entries, centres, total, labels=index % cols == 0)
            ax.set_xlabel(xlabel)
            ax.margins(x=0.25)
        handles = [_handle(s["color"], s["marker"], s["filled"], "mean, pitches with " + ("an " if key == "offered" else "")
                           + s["label"], line=False) for key, s in OFFER_STYLE.items()] + _significance_handles()
        fig.legend(handles=handles, loc="outside lower center", ncol=1 if stack else 3)
    return fig


# ---- figure 9: H9 -------------------------------------------------------------------------------------

H9_ROW_COLUMNS = ["questions", "kind", "model", "outcome", "predictor", "coef", "low", "high", "outcome_words",
                  "predictor_words", "why", "reason"]


def _h9_frame(a) -> pd.DataFrame:
    if a.h9 is None:
        return pd.DataFrame(columns=H9_ROW_COLUMNS)
    parts = []
    for questions, frame in a.h9.items():
        if frame is None or not len(frame):
            continue
        parts.append(frame.assign(questions=questions))
    if not parts:
        return pd.DataFrame(columns=H9_ROW_COLUMNS)
    t = pd.concat(parts, ignore_index=True)
    t = t[~t.predictor.isin(H9_COVARIATES)].reset_index(drop=True)     # a covariate is not a predictor of interest
    kind = {o: k for k, outcomes, _, _ in H9_KINDS for o in outcomes}

    def why(r) -> str:
        if bool(r.separated):
            return "separated"
        if str(r.estimable) != "yes":
            return str(r.estimable)
        if bool(r.main_failed):
            return "the main fit failed"
        if not (np.isfinite(r.low) and np.isfinite(r.high)):
            failed = int(r.n_failed) if r.n_failed == r.n_failed else 0
            return f"too few draws estimate it ({failed} failed)" if failed else "no interval given"
        return ""

    def reason(text: str) -> str:
        return ("" if not text else H9_SEPARATED if text == "separated" else
                H9_TOO_FEW if text.startswith("too few draws") else H9_NOT_ESTIMABLE)
    whys = [why(r) for r in t.itertuples()]
    return t.assign(kind=t.outcome.map(kind), outcome_words=t.outcome.map(lambda o: H9_OUTCOME_WORDS.get(o, plain(o))),
                    predictor_words=t.predictor.map(lambda p: H9_PREDICTOR_WORDS.get(p, plain(p))),
                    why=whys, reason=[reason(w) for w in whys])[H9_ROW_COLUMNS]


def h9_rows(a) -> pd.DataFrame:
    """The H9 rows figure 9 draws: every (questions, model, outcome, predictor) with a finite interval."""
    t = _h9_frame(a)
    return t[t.why == ""].reset_index(drop=True)


def h9_left_out(a) -> pd.DataFrame:
    """The H9 rows figure 9 does not draw, with why (`why`) and its kind (`reason`: too few estimable draws,
    separated, not estimable); the caption counts them per panel and kind."""
    t = _h9_frame(a)
    return t[t.why != ""].reset_index(drop=True)


def _h9_layout(rows: pd.DataFrame, outcomes: tuple[str, ...], narrow: bool = False):
    """Rows of one kind of outcome, shared by both panels: a heading per outcome, a label per predictor and a
    sub-row per nested model that has a drawn interval in either panel. (entries, centres, total, positions)
    with positions[(outcome, predictor, model)] = y."""
    step, pad, head, least = 0.05, 0.05, 0.2, 0.13
    entries, centres, positions, top = [], [], {}, 0.0
    for outcome in outcomes:
        part = rows[rows.outcome == outcome]
        if part.empty:
            continue
        words = H9_NARROW_OUTCOME_WORDS if narrow else H9_OUTCOME_WORDS
        entries.append(("header", words.get(outcome, plain(outcome)), None))
        centres.append(top + head / 2 + 0.03)
        top += head
        for predictor in dict.fromkeys(p for spec in H9_MODELS.values() for p in spec):
            models = [m for m in H9_MODELS if predictor in H9_MODELS[m] and
                      ((part.predictor == predictor) & (part.model == m)).any()]
            if not models:
                continue
            height = max(least, step * len(models) + pad)
            first = top + (height - step * (len(models) - 1)) / 2
            for k, model in enumerate(models):
                positions[(outcome, predictor, model)] = first + step * k
            words = H9_NARROW_PREDICTOR_WORDS if narrow else H9_PREDICTOR_WORDS
            entries.append(("row", words.get(predictor, plain(predictor)), None))
            centres.append(top + height / 2)
            top += height
    return entries, centres, top, positions


def figure_9_h9(a, width=DEFAULT_WIDTH):
    """H9: the coefficients of each nested answer-level model with their 95% pitch-bootstrap intervals. Panel A:
    answers to all questions; panel B: to hard questions. One axis row per unit (H9_KINDS): the binary next moves
    (log-odds, conditional logit within pitch), the next question's severity and the number of further questions
    (each its own pitch fixed-effects scale); within a row A and B share the x axis. One sub-row per (model,
    outcome, predictor), marker by model; filled and coloured by sign when the interval excludes zero, hollow grey
    when it does not. Rows without an interval are not drawn; the caption counts them."""
    import matplotlib.pyplot as plt

    rows = h9_rows(a)
    if rows.empty:
        return None
    stack = _stacked(width)
    layouts = [(kind, short if stack else words, _h9_layout(rows[rows.kind == kind], outcomes, stack))
               for kind, outcomes, words, short in H9_KINDS]
    layouts = [(kind, words, lay) for kind, words, lay in layouts if lay[0]]
    heights = [lay[2] + 0.6 for _, _, lay in layouts]
    with _context():
        fig = plt.figure(figsize=(width_inches(width), sum(heights) + 0.85 + 0.45 * stack), dpi=SCREEN_DPI,
                         layout="constrained")
        grid = fig.add_gridspec(len(layouts), 2, height_ratios=heights)
        first = {}
        for r, (kind, words, (entries, centres, total, positions)) in enumerate(layouts):
            for c, (questions, title) in enumerate(H9_PANELS):
                ax = fig.add_subplot(grid[r, c], sharey=first.get(r), sharex=first.get(r))   # one scale per row
                first.setdefault(r, ax)
                if r == 0:
                    _letter(ax, c, H9_NARROW_PANELS[questions] if stack else title)
                ax.axvline(0, color=INK, linewidth=0.8, zorder=1)
                part = rows[(rows.kind == kind) & (rows.questions == questions)]
                for row in part.itertuples():
                    y = positions[(row.outcome, row.predictor, row.model)]
                    clear = row.low > 0 or row.high < 0
                    colour = (BLUE if row.low > 0 else VERMILLION) if clear else GREY
                    ax.plot([row.low, row.high], [y, y], color=colour, linewidth=1.0, solid_capstyle="butt", zorder=2)
                    _point(ax, row.coef, y, colour, H9_MARKERS.get(row.model, "o"), clear, size=4.0)
                if part.empty:
                    ax.text(0.5, 0.5, "no interval", transform=ax.transAxes, ha="center", va="center", color=GREY)
                _label_rows(ax, entries, centres, total, labels=c == 0)
                ax.set_xlabel(_wrap(words, 36))          # two panels side by side: a long unit wraps
                ax.margins(x=0.1)
        handles = [_handle(INK, marker, True, model if stack else f"{model} model", line=False)
                   for model, marker in H9_MARKERS.items() if model in set(rows.model)]
        handles += [_handle(BLUE, "o", True, "above 0" if stack else "interval above zero (filled)"),
                    _handle(VERMILLION, "o", True, "below 0" if stack else "interval below zero (filled)"),
                    _handle(GREY, "o", False, "includes 0" if stack else "interval includes zero (hollow grey)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2 if stack else 4, handlelength=1.4,
                   columnspacing=1.0)
    return fig


# ---- S1: the archived shark-level ladder ----------------------------------------------------------

# the shark-level families (analysis.archive.shark_level_models), as what a rung adds
SHARK_FAMILY_WORDS = {"control": "venture", "exposure": "exposure", "stage_a": "Stage A",
                      "own": "the shark's own Q and A", "panel_content": "the other sharks' Q and A",
                      "panel_reactions": "reactions", "rapport_repair": "rapport and repair",
                      "decision_history": "order of deciding"}


def shark_labels(metrics: pd.DataFrame) -> list[str]:
    """Each shark-level model named by what it adds to the model before it (the families of its metrics row)."""
    labels, before = [], []
    for row in metrics.itertuples():
        families = [] if row.families == "(base rate)" else str(row.families).split("+")
        added = [f for f in families if f not in before] or families
        words = ", ".join(SHARK_FAMILY_WORDS.get(f, plain(f)) for f in added)
        labels.append(f"{row.model}  " + (f"+ {words}" if families else "offer rate only"))
        before = families
    return labels


def figure_s1_shark_level(a, width=DEFAULT_WIDTH):
    """Skill over the base rate (%) with its 95% interval of each model of the archived shark-level ladder (one row
    per investor, B0 to B5d), from `analysis.archive.shark_level.run`; drawn only when that path ran."""
    import matplotlib.pyplot as plt

    shark = getattr(a, "shark_level", None)
    if shark is None:
        return None
    m = shark.get("metrics")
    if m is None or not len(m) or "skill_pct" not in m:
        return None
    stack = _stacked(width)
    with _context():
        fig, ax = plt.subplots(figsize=(width_inches(width), 0.32 * len(m) + (1.7 if stack else 1.3)), dpi=SCREEN_DPI,
                               layout="constrained")
        for i, row in enumerate(m.itertuples()):
            ax.plot([row.skill_low, row.skill_high], [i, i], color=PURPLE, linewidth=1.2, solid_capstyle="butt", zorder=2)
            _point(ax, row.skill_pct, i, PURPLE, "s", True)
        _ladder_axis(ax, [_wrap(t, 26 if stack else 64) for t in shark_labels(m)], 0.0, "offer rate",
                     "Skill over the base rate, %\n(each shark's offer)" if stack else
                     "Skill over the base rate, % (each shark's offer)")
        ax.margins(x=0.1)
        fig.legend(handles=[_handle(PURPLE, "s", True, "estimate with its 95% interval")], loc="outside lower center")
    return fig


# ---- the figures, their captions, and the files -----------------------------------------------------

BUILDERS = {1: figure_1_model_ladder, 2: figure_2_auc, 3: figure_3_roc, 4: figure_4_concepts,
            5: figure_5_answer_quality, 6: figure_6_calibration, 7: figure_7_after_an_answer,
            8: figure_8_coefficients, 9: figure_9_h9, 10: figure_10_gains, "S1": figure_s1_shark_level,
            "S2": figure_s2_gains_skill}


def _number(number):
    if number not in FIGURES:
        raise ValueError(f"there is no figure {number!r}. Figures: {', '.join(str(n) for n in FIGURES)}.")
    return number


def not_drawn(a, number) -> str:
    """Why figure `number` cannot be drawn from this analysis; an empty text when it can."""
    number = _number(number)
    status = a.result.get("status", "not estimated")
    if number in (1, 2, 3, 6) and _estimated(a) and _degenerate(a):
        return _degenerate(a)
    if number in (1, 2):
        return "" if _estimated(a) else f"no model was estimated ({status})"
    if number in (10, "S2"):
        return "" if len(baseline_gain_table(a)) else f"no baseline comparison was estimated ({status})"
    if number in (3, 6):
        return "" if _curve_models(a) else f"no model with features was estimated ({status})"
    if number == 4:
        return "" if len(forest_rows(a)) else "no feature could be ranked"
    if number == 5:
        table = by_outcome_table(a)
        return "" if len(table) and table.correlation.notna().any() else "no row has both a value and a varying outcome"
    if number == 7:
        return "" if a.h4_shares is not None and len(a.h4_shares) else "no answer to an investor question in this selection"
    if number == 8:
        return "" if len(coefficient_rows(a)) else "the full model C5 was not fitted"
    if number == 9:
        if a.h9 is None:
            return H9_SKIPPED
        return "" if len(h9_rows(a)) else "no H9 coefficient has an interval"
    if getattr(a, "shark_level", None) is None:
        return NO_SHARK_LEVEL
    m = a.shark_level.get("metrics")
    return "" if m is not None and len(m) and "skill_pct" in m else "the shark-level ladder was not estimated"


def figure(a, number, width=DEFAULT_WIDTH):
    """Figure `number` (1 to 9, "S1") at a width preset, or None when it cannot be drawn (`not_drawn` says why)."""
    number = _number(number)
    if not_drawn(a, number):
        return None
    return BUILDERS[number](a, width)


def _models_list(a) -> str:
    return ", ".join(f"{name} ({'+'.join(spec) or 'no features'})" for name, spec in a.models.items())


def h9_left_out_counts(a) -> dict[str, dict[str, int]]:
    """{questions: {reason: n}} of the H9 coefficients figure 9 does not draw."""
    left = h9_left_out(a)
    return {q: {reason: int(((left.questions == q) & (left.reason == reason)).sum())
                for reason in (H9_TOO_FEW, H9_SEPARATED, H9_NOT_ESTIMABLE)} for q, _ in H9_PANELS}


def _h9_left_out_text(a) -> str:
    counts = h9_left_out_counts(a)
    if not any(sum(c.values()) for c in counts.values()):
        return "Every coefficient has an interval and is drawn."
    parts = [f"panel {letter} ({title.lower()}): " + ", ".join(f"{n} {reason}" for reason, n in counts[q].items())
             for letter, (q, title) in zip("AB", H9_PANELS)]
    return ("Not drawn, having no interval: " + "; ".join(parts) + ". Every row and its reason is in h9_all.csv / "
            "h9_hard.csv (the responsiveness model in h9_responsiveness.csv).")


def caption(a, number, width=DEFAULT_WIDTH) -> str:
    """The caption of figure `number`: what it shows, the sample, the coding, the intervals, the test behind the
    marks, and how to read it."""
    number = _number(number)
    name = f"Supplementary Figure {number}" if number in SUPPLEMENTARY else f"Figure {number}"
    title = f"{name}. {FIGURES[number][1]}."
    why = not_drawn(a, number)
    if why:
        return f"{title} Not drawn: {why}."
    sample = sample_sentence(a)
    if number == 1:
        settings = [s for s in SENSITIVITY_STYLE if s in set(sensitivity_table(a).setting)]
        words = sensitivity_words(a)
        sens = "; ".join(f"{words[s]} ({SENSITIVITY_STYLE[s][1]})" for s in settings) or "none estimated"
        absent = "".join(f" Without a marker: {words.get(s, s)} ({why})." for s, why in not_run_sensitivities(a).items())
        stage_a = ("the primary Stage A rating (the run has no ranking, so no rank score)" if rating_fallback(a)
                   else "the primary Stage A rank score")
        body = (f"Skill over the base rate, 100 x (1 - log loss / log loss of always predicting the offer rate), of "
                f"each model of the pre-registered ladder, fitted leaving one pitch out; one row per pitch, outcome "
                f"any offer. The ladder adds one step at a time: C1 the venture, ask and exposure; C2 {stage_a}; "
                f"C2p the prepared presentation; C3q the sharks' questions; C3 the answers; C4 the "
                f"reactions; C5 rapport and repair. Models: {_models_list(a)}. Hollow markers below each filled one "
                f"are the same model under each sensitivity (point estimates only): {sens}.{absent} {sample} "
                f"{interval_sentence(a, 'Bars', fixed=True)} {test_sentence(a)} How to read it: the vertical line at 0 "
                "is a model that knows only the offer rate; a point to its right predicts held-out pitches better, and "
                "a bar that crosses the line means this sample does not show a difference from the offer rate. The "
                "verdicts of H1' (C3 against C2p) and H8 (C3 against C3q) rest on the paired differences in "
                f"report.md, not on overlapping bars. {EXPLORATORY_SENTENCE}")
    elif number == 2:
        body = (f"Area under the ROC curve of the held-out scores of the models of Figure 1 (C0 to C5 with C2p "
                f"and C3q). {sample} {interval_sentence(a, 'Bars', fixed=True)} {test_sentence(a)} How to read it: the "
                "vertical line at 0.5 is chance; right of it a model ranks pitches with an offer above pitches without. "
                "Every finite AUC is drawn; a hollow grey marker is a model whose held-out predictions stay within "
                "about 3 points of the offer rate, a diagnostic mark only (its AUC still counts); the base-rate model "
                f"ranks nothing and is named in the right margin. {AUC_SCORE_SENTENCE}. Read the skill "
                f"figure first; AUC is not the primary metric. {EXPLORATORY_SENTENCE}")
    elif number == 3:
        body = (f"ROC curves of every model with features, drawn on the base-rate-adjusted held-out score (the "
                f"score its AUC ranks; {AUC_SCORE_SENTENCE}), pitches weighted equally. Each "
                f"model has its own dash pattern and colour, with its AUC and the p-value of that AUC in the legend; "
                f"the dotted diagonal is chance; 'near-constant' in the legend marks a model whose held-out "
                f"predictions stay within about 3 points of the offer rate (a diagnostic only; see Figure 2). {sample} The AUC intervals are in Figure 2. {test_sentence(a)} How to read it: a curve "
                "above the diagonal ranks pitches with an offer above pitches without at every threshold; the further "
                f"towards the top-left, the better the ranking. {EXPLORATORY_SENTENCE}")
    elif number == 4:
        rows, tested = forest_rows(a, _top_n(FOREST_TOP_N, width)), a.single[a.single.p_value.notna()]
        survive = int(a.single.significant_fdr.astype(bool).sum())
        body = (f"Each pitch feature on its own: its correlation with any offer, one row per feature, grouped by "
                f"family and sorted by size within a family; the {len(rows)} features with the largest correlation "
                f"are shown, columns that only count the observations behind a mean are left out. {sample} "
                f"{interval_sentence(a, draws=a.single_draws)} Markers show significance: two-sided pitch-level "
                f"permutation p-values of the correlation, adjusted for the {len(tested)} features ranked with the "
                f"Benjamini-Hochberg false-discovery rate; a filled triangle survives the adjustment (q < "
                f"{FDR_LEVEL:g}), a hollow triangle has p < 0.05 before adjustment only, a grey circle has neither "
                f"({survive} of {len(a.single)} features survive). How to read it: right of 0, more of the feature "
                "goes with an offer. Each feature is looked at alone and features overlap; a lead here is not a "
                f"model result. {EXPLORATORY_SENTENCE}")
    elif number == 5:
        table = by_outcome_table(a)
        body = (f"Answer quality by any offer, from the answers to all sharks' questions before the pitch's cut, "
                f"pooled per pitch: the mean substance (1 to 5) and the percentage of answers with checkable support, "
                f"for answers to all questions and to hard questions only (severity 2 to 3); n is the number of "
                f"pitches with a value (substance / support). Panels A and C: the mean among pitches with an offer "
                f"(filled blue circle) and without (hollow vermillion square). Panels B and D: the correlation of each "
                f"measure with any offer. {sample} {interval_sentence(a, 'Bars in B and D', draws=a.single_draws)} "
                f"Markers: two-sided pitch-level permutation p-values, Benjamini-Hochberg-adjusted over the "
                f"{int(table.p_value.notna().sum())} rows of this table (filled: q < {FDR_LEVEL:g}; hollow: p < 0.05 "
                "before adjustment only; grey: neither). How to read it: right of 0, better answers go with offers. "
                f"Descriptive, not causal. {EXPLORATORY_SENTENCE}")
    elif number == 6:
        body = (f"Calibration of the held-out predictions of every model with features. Per model the pitches are "
                f"sorted by predicted probability into four equal groups; a point is a group's mean prediction against "
                f"the share of its pitches with an offer. Each model has its own marker and dash pattern; the dotted "
                f"diagonal is perfect calibration. {sample} No intervals or tests are drawn: each point rests on a "
                f"quarter of the pitches (the models' intervals, from {a.n_boot} resamples of whole pitches, are in "
                "Figure 1). How to read it: points on the diagonal mean that a predicted 30% came true for about 30% "
                "of the pitches; a flat line means the predictions hardly separate the groups.")
    elif number == 7:
        first = a.h4_shares[a.h4_shares.questions == ALL_QUESTIONS].iloc[0]
        hard = a.h4_shares[a.h4_shares.questions == HARD_QUESTIONS].iloc[0]
        body = (f"What the panel does after an answer (H4): the percentage of answers whose first reaction by any "
                f"shark is a receipt (an accepts turn), another question, or a challenge, for answers that give "
                f"checkable support (filled blue circle) and answers that give none (hollow vermillion square). Panel "
                f"A: answers to all questions ({int(first.n_yes)} with support, {int(first.n_no)} without); panel B: "
                f"answers to hard questions, severity 2 to 3 ({int(hard.n_yes)} and {int(hard.n_no)}). One unit is an "
                f"answer and a shark's question it responds to, over the whole encounter of {int(first.n_pitches)} "
                f"pitches (no cut). {sample} {interval_sentence(a)} The numbers above each pair are the difference, "
                "with support minus without, in percentage points, with its interval; its stars are the two-sided "
                f"bootstrap p-value of the difference ({pr.STARS_NOTE}; six differences, not adjusted). How to read "
                "it: when the bars of a pair overlap and the difference has no star, this sample does not show that "
                "support changes the panel's reaction. Descriptive counts; they do not say that an answer caused the "
                f"reaction. {EXPLORATORY_SENTENCE}")
    elif number == 8:
        t = a.coefficients
        rows = coefficient_rows(a, _top_n(COEFFICIENT_TOP_N, width))
        body = (f"Coefficients of the full model C5 refitted on all pitches with a known outcome: the {len(rows)} "
                f"largest of {len(t)}, grouped by feature family. A coefficient is the change in the log-odds of an "
                f"offer per one standard deviation of the feature (per level of a category) in a ridge-penalised "
                f"logistic model whose penalty is fixed at {t.penalty.iloc[0]:g}, the value the leave-one-pitch-out run "
                f"chose most often. {sample} Bars are 95% percentile intervals from resampling whole pitches and "
                f"refitting the model ({int(t.n_boot.max())} draws); a hollow triangle has a two-sided bootstrap p "
                "below 0.05 (blue pointing right: positive; vermillion pointing left: negative), a grey circle has "
                "not; these p-values are not adjusted for the number of coefficients. How to read it: these are "
                "conditional associations within one penalised model, not independent effects: each depends on which "
                "other features are in the model, the penalty shrinks all of them towards zero, and correlated "
                f"features share their weight. Compare with Figure 4. {EXPLORATORY_SENTENCE}")
    elif number == 9:
        rows = h9_rows(a)
        body = (f"H9, the answer-level test: what follows an answer, by the answer's ratings, within pitch. Panel A: "
                f"answers to all questions; panel B: answers to hard questions (severity 2 or 3). Each nested model "
                f"has its own marker: support (checkable support and admitted uncertainty, all answers), substance "
                f"(+ substance, answers with a substance rating), full (+ coherence and context, answers rated on all "
                f"five; the context rating is Gemini only, not human-coded) and responsiveness (admitted uncertainty and repair before the next turn, all answers). One row of "
                f"axes per unit: the binary next moves, among them whether the next investor question keeps the "
                f"promotion/prevention frame of the question answered (log-odds from a conditional logit within "
                f"pitch), the severity of the next question, and the number of further questions in turn order that "
                f"keep that frame (each a change in that outcome from pitch fixed effects, on its own scale); within a "
                f"row panels A and B share the scale. Binary predictors are per 0/1, ratings per standard deviation. "
                f"For the two outcomes about the frame, the answered question's frame (promotion or prevention) is "
                f"entered as a covariate and not drawn; answers to a question with neither frame are not in those "
                f"models."
                + (" Outcomes read on the model's grouping of turns into exchanges were computed as a sensitivity "
                   "only and are not drawn (h9_exchange_sensitivity.csv)."
                   if getattr(a, "h9_sensitivity", None) is not None else "")
                + (" At this width the rows are named briefly: support = checkable support, uncertainty = admits "
                   "uncertainty, repair = repair before the next turn, SD = per standard deviation; 'Same frame "
                   "next' = the next question keeps the frame, 'Same-frame run' = the further questions that keep "
                   "it." if _stacked(width) else "")
                + f" {len(rows)} coefficients "
                f"are drawn. {_h9_left_out_text(a)} {sample} Bars are 95% percentile intervals from resampling whole "
                f"pitches and refitting ({a.n_boot} draws); filled and coloured when the interval excludes zero (blue "
                "above, vermillion below), hollow grey when it includes zero. How to read it: H9 has no decision rule; "
                "a filled marker is a within-pitch association in this sample, not a causal effect, and the rows are "
                f"not adjusted for their number. {EXPLORATORY_SENTENCE}")
    elif number in (10, "S2"):
        from analysis.panel_bottleneck import BASELINE

        table = baseline_gain_table(a)
        m = a.result["metrics"].set_index("model")
        ranking = number == 10
        part = table[table.metric == ("auc" if ranking else "skill")].set_index("family")
        scale, digits = (100, 0) if ranking else (1, 1)
        gains = "; ".join(f"{GAIN_FAMILY_WORDS[f].lower()} alone {scale * part.alone_delta[f]:+.{digits}f}% (p = {part.alone_p[f]:.3f})"
                          + ("" if part.alone_model[f] == part.cumulative_model[f] else
                             f", cumulative {scale * part.cumulative_delta[f]:+.{digits}f}% (p = {part.cumulative_p[f]:.3f})")
                          for f in GAIN_FAMILY_WORDS if f in part.index)
        baseline = (f"{100 * m.auc[BASELINE]:.0f} pairs ranked correctly per 100" if ranking else
                    f"skill {m.skill_pct[BASELINE]:+.1f}%")
        what = ("Of 100 pairs of one funded and one unfunded pitch, how many each model ranks correctly (AUC x 100; "
                "the dotted line at 50 is chance), with the gains written as percentage points of pairs."
                if ranking else
                f"Skill over the base rate, in percent (the dotted line at 0 is guessing the offer rate for every "
                f"pitch); the axis is held to {SKILL_AXIS_LIMITS[0]:g} to {SKILL_AXIS_LIMITS[1]:g} so that the models "
                "can be told apart, and an interval that runs past an edge is cut there and marked with an arrowhead "
                "(the full intervals are in metrics.csv).")
        body = (f"The baseline model {BASELINE} (the venture, the ask, exposure, the Stage A rank score and the prepared "
                f"presentation; {baseline}) in the first row as a black diamond, and below it each family's models at "
                f"their own level, each with its 95% interval. {what} A filled blue circle is the family entering alone "
                f"({BASELINE} plus that family: C3q, C3a, C4r, C5r; spec amendment 6.9 of 2026-10-25); a hollow "
                f"vermillion square is the same family entering the cumulative ladder on top of the families before it "
                f"(C3q, C3, C4, C5). For the questions the two are the same model and are drawn once. The solid "
                f"vertical line is the baseline's value, so a point's distance from it is the family's gain over the "
                f"baseline; beside each point, that gain (later minus earlier, paired on the same held-out pitches) and "
                f"the bootstrap p of the gain. Gains: {gains}. {sample} {interval_sentence(a, 'Bars', fixed=True)} How "
                f"to read it: a family whose bar lies wholly to the right of the solid line adds something to the "
                f"baseline that this sample can tell from zero. "
                + ("The skill version of this figure is Supplementary Figure S2. " if ranking else
                   "The ranking version of this figure is Figure 10. ")
                + "The pre-registered verdicts (H1', H8) are in report.md; the one-family models were added after the "
                f"first run and are reported, not tested. {EXPLORATORY_SENTENCE}")
    else:
        m = a.shark_level["metrics"]
        label = m.label.iloc[0] if "label" in m and len(m) else ""
        body = (f"The archived shark-level design (one row per investor, outcome: does this shark offer), rerun on the "
                f"same records with --with-shark-level for the F1 paragraph: skill over the base rate of each model "
                f"B0 to B5d, fitted leaving one pitch out, with 95% percentile intervals from resampling whole pitches "
                f"({a.n_boot} draws) of the fixed held-out predictions. {int(m.n_pitches.iloc[0]) if 'n_pitches' in m else ''} "
                f"pitches, {int(m.n_rows.iloc[0]) if 'n_rows' in m else ''} shark rows{f' ({label})' if label else ''}. "
                f"{sample} This figure is reported, not tested: the pre-registered test is the pitch-level ladder of "
                "Figure 1. How to read it: a point right of 0 predicts each held-out shark's decision better than the "
                "offer rate; sharks of one pitch are not independent, so the intervals resample pitches.")
    return f"{title} {body}"


# what a builder raises on an analysis that lacks a field, a key of a table or result, or a table
SKIPPED_ERRORS = (AttributeError, KeyError, TypeError)


def file_names(number) -> dict[str, str]:
    stem = f"figure_{_number(number)}_{FIGURES[number][0]}"
    return {"png": f"{stem}.png", "pdf": f"{stem}.pdf", "caption": f"figure_{number}_caption.md"}


def write_figures(a, folder: Path | str, width=DEFAULT_WIDTH, skipped: dict | None = None) -> list[str]:
    """Save every figure that can be drawn into `folder` as PNG (300 dpi) and vector PDF, each with its caption file,
    and return the file names written, in order. Writes nowhere else.

    With `skipped` (a dict), each figure that is not drawn is recorded there as {figure: reason}: the `not_drawn`
    reason (S1 without the shark-level path: NO_SHARK_LEVEL), or "builder failed (<exception type>)" when its builder
    or caption raises one of SKIPPED_ERRORS, which then does not stop the others. Without `skipped` nothing is
    swallowed."""
    import matplotlib.pyplot as plt

    width_inches(width)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for number in FIGURES:
        try:
            fig = figure(a, number, width)
            text = caption(a, number, width) if fig is not None else ""
        except SKIPPED_ERRORS as exc:
            plt.close("all")
            if skipped is None:
                raise
            skipped[str(number)] = f"builder failed ({type(exc).__name__})"
            continue
        if fig is None:
            if skipped is not None:
                skipped[str(number)] = not_drawn(a, number)
            continue
        written += save_figure(fig, folder, file_names(number), text)
    return written
