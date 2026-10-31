"""The paper's validation table. It opens with the summary the paper reports (the researcher's call of 2026-10-31):
one row per code with its scale, Gwet's AC1 (categorical and binary codes) or AC2 (ordinal codes) with Krippendorff's
alpha in parentheses, Gemini against the human coders and coder against coder. The detail tables follow (plan D,
2026-10-30, the layout the demo notebook reads): every reported code with the statistics of its kind, raw agreement
leading and alpha in parentheses, alpha's Landis & Koch band and the flags in words; then an appendix with every other
figure and the definitions, and a methods note generated from the reporting configuration
(validation_reporting.yaml), which this module also loads. Nothing is passed or failed: no row carries a verdict and
none is dropped."""
from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable

from pipeline.errors import ConfigError
from pipeline.instrument import VALIDATION_REPORTING_FILE, InstrumentRelease, load_release
from pipeline.reliability.agreement import MIN_DEFINED_SHARE, landis_koch_band
from pipeline.stage_b.panel_books import PanelBooks, frame_of
from pipeline.stage_b.review_apply import MIN_CLUSTER_VIDEOS, N_BOOT, CUT_TOLERANCE_S
from pipeline.stage_b.review_items import (
    ASK, CUT, EVID_BANDS, EVID_CODE, OUTCOME, OUTCOME_VALUES, PAIR, VENTURE_CODES)

NOTE = ("Every figure is agreement given the model's segmentation of the conversation into turns; the structural "
        "rate in the appendix is the only human evidence on segmentation. Coder against model uses each coder's own "
        "blind codes, never adjudicated values: each coder's codes are the last line that coder imported for each "
        "field.")
OUTCOME_NOTE = "Outcomes are sampled rows, not a census."
STAGE_A_NOTE = ("Stage A was rated over the prepared-presentation window the model recorded, which coders were "
                "shown; the window itself was not validated.")
HEADLINE_NOTE = ("Raw agreement leads each row: the share of units on which the two sides give the identical value, "
                 "and for an ordinal code also the share within one scale point. Krippendorff's α follows in "
                 "parentheses because it answers another question, how reliably the categories are told apart beyond "
                 "chance, and it falls when one value dominates the answers even where the raters mostly agree; the "
                 "prevalence flag marks those rows. Gwet's AC1 and AC2 and the agreement on each value are in the "
                 "appendix.")
PASSES_NOTE = ("Coder against coder uses the videos exactly two coders judged and the units both gave a definite "
               "value. Model pass against pass compares the model's two identical passes (primary and replicate) on "
               "the sampled units that have both, each unit once: it is the model's consistency with itself, not "
               "agreement with a person.")
OFF_SCALE_NOTE = ("A unit on which either side gave no value on the code's scale (an empty or missing value) is left "
                  "out of the row; the appendix counts them.")
NO_PAIRS = "— (no comparisons imported)"
DASH = "—"
SIBLINGS = ("human_human", "pass_to_pass")          # pooled["reporting"]'s keys that are not variables
COMPARISONS = (("coder_model", "Coder against model"), ("human_human", "Coder against coder"),
               ("pass_to_pass", "Model pass against pass"))
# the figures that lead a row's Agreement cell, by kind
HEADLINE = {"categorical": ("exact",), "binary": ("exact",), "ordinal": ("exact", "within_one"),
            "clock": ("same_turn", "median_abs"), "exact": ("exact",), "comparison": ("agreement", "p_vs_half")}
ALPHA_KINDS = ("categorical", "binary", "ordinal")
KIND_WORDS = {"categorical": "Categorical codes", "binary": "Binary codes", "ordinal": "Ordinal codes",
              "clock": "Clock codes", "exact": "Exact codes", "comparison": "Comparisons"}
HEAD = "| Field | Kind | Units (videos) | Agreement (α) | Band of α | Flags | Without unsure: point (n) |"


# ---- the rows --------------------------------------------------------------------------------------------------

def _section(pooled: dict, which: str = "coder_model") -> dict:
    """{name: row} of one comparison in pooled["reporting"]: coder against model (the variable keys), "human_human"
    or "pass_to_pass"."""
    reported = pooled.get("reporting") or {}
    if which == "coder_model":
        return {k: v for k, v in reported.items() if k not in SIBLINGS}
    return reported.get(which) or {}


def _rank(reporting) -> list[str]:
    """The order of the rows: the derived rows (the frame of the question concept), then the codes as the
    reporting file lists them; a pair dimension takes the place of `pair`."""
    return list(reporting.derived) + list(reporting.codes) if reporting is not None else []


def _ordered(names, reporting) -> list[str]:
    rank = _rank(reporting)

    def key(name: str):
        # a pair dimension sits at `pair`'s place, its dimensions in the order of their Stage A items
        base, _, item = name.partition(":") if name.startswith(f"{PAIR}:") else (name, "", "")
        place = rank.index(base) if base in rank else len(rank)
        return (place, rank.index(item) if item in rank else len(rank), name)
    return sorted(dict.fromkeys(names), key=key)


def _finite(value) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _alpha_of(row: dict) -> dict | None:
    """The alpha the row shows in parentheses (nominal for categorical and binary codes, ordinal for ordinal ones);
    None for a kind without one."""
    alpha = (row.get("figures") or {}).get("alpha") if row["kind"] in ALPHA_KINDS else None
    return dict(alpha) if alpha is not None else None


def _prevalence_read(row: dict, band_of) -> dict | None:
    """On a prevalence-flagged row, the statistic to read rather than alpha (AC2 for an ordinal code, else AC1), with
    its value and its band."""
    if "prevalence" not in (row.get("flags") or []):
        return None
    name = "ac2" if row["kind"] == "ordinal" else "ac1"
    value = _finite(((row.get("figures") or {}).get(name) or {}).get("value"))
    return {"statistic": name, "value": value, "band": band_of(value) if value is not None else None}


def _row(name: str, row: dict | None, which: str, kind: str | None, against: float) -> dict:
    if row is None:
        return {"field": name, "comparison": which, "kind": kind, "missing": True, "raters": None, "n": None,
                "units": None, "videos": None, "n_off_scale": None, "figures": {}, "counts": {}, "alpha": None,
                "band": None, "flags": [], "without_unsure": {}, "coder_values": {}, "model_values": {}, "strata": {},
                "against": against}
    alpha = _alpha_of(row)
    return {"field": name, "comparison": which, "kind": row["kind"], "missing": False, "raters": row.get("raters"),
            "n": row.get("n"), "units": row.get("units", row.get("n")), "videos": row.get("videos"),
            "n_off_scale": row.get("n_off_scale", 0), "figures": row.get("figures") or {},
            "counts": row.get("counts") or {}, "alpha": alpha,
            "band": landis_koch_band(alpha["value"]) if alpha else None, "flags": list(row.get("flags") or []),
            "without_unsure": row.get("without_unsure") or {}, "coder_values": row.get("coder_values") or {},
            "model_values": row.get("model_values") or {}, "strata": row.get("strata") or {}, "against": against}


def _names(pooled: dict, reporting) -> list[str]:
    """The rows of every table: the coder-against-model rows, then any row only another comparison has; `pair` when
    the reporting has comparisons and none was imported."""
    names = _ordered([*_section(pooled), *_section(pooled, "human_human"), *_section(pooled, "pass_to_pass")],
                     reporting)
    wants_pairs = reporting is None or reporting.kind_of(PAIR) == "comparison"
    if wants_pairs and not any(n.startswith(f"{PAIR}:") for n in names):
        names = _ordered([*names, PAIR], reporting)
    return names


def paper_rows(pooled: dict, reporting=None, which: str = "coder_model") -> list[dict]:
    """One row per reported code of one comparison (coder against model, "human_human", "pass_to_pass"), in the
    reporting file's order, every comparison with the same rows: a row the comparison lacks is `missing` (its cells
    "—"). Each row carries its figures as pooled["reporting"] has them, the alpha shown in parentheses with its
    Landis & Koch band, and on a prevalence-flagged row the statistic to read instead ("read": AC1 or AC2, its value
    and band); no verdict."""
    against = (reporting.comparisons["against"] if reporting is not None else 0.5)
    section = _section(pooled, which)
    kinds = {n: r["kind"] for w, _ in COMPARISONS for n, r in _section(pooled, w).items()}
    out = []
    for name in _names(pooled, reporting):
        kind = kinds.get(name) or (reporting.kind_of(name) if reporting is not None else None)
        row = _row(name, section.get(name), which, kind, against)
        row["positive"] = reporting.positive_of(name) if reporting is not None and kind == "binary" else None
        band_of = reporting.band_of if reporting is not None else landis_koch_band
        if row["alpha"]:
            row["band"] = band_of(row["alpha"]["value"])
        row["read"] = _prevalence_read(row, band_of)
        out.append(row)
    return out


# ---- the cells -------------------------------------------------------------------------------------------------

def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def _num(value) -> str:
    value = _finite(value)
    return DASH if value is None else f"{value:.2f}"


def _secs(value) -> str:
    value = _finite(value)
    return DASH if value is None else f"{value:.1f} s"


def _p(value) -> str:
    value = _finite(value)
    if value is None:
        return DASH
    return "< 0.001" if value < 0.001 else f"{value:.3f}"


def _interval(f: dict | None) -> str:
    low, high = (_finite((f or {}).get("low")), _finite((f or {}).get("high")))
    return f"[{DASH}]" if low is None or high is None else f"[{low:.2f}, {high:.2f}]"


def _figure(f: dict | None, count: str = "") -> str:
    """A share or a chance-corrected statistic with its interval ("[—]" when it has none); "—" when undefined."""
    if f is None or _finite(f.get("value")) is None:
        return DASH
    return f"{f['value']:.2f}{count} {_interval(f)}"


def _count(row: dict, name: str = "exact") -> str:
    """The units behind the leading share, shown on a small-n row: " (k/n)"."""
    if "small_n" not in row["flags"] or not row["n"]:
        return ""
    if row["kind"] == "clock":
        c = row["counts"]
        return f" ({c.get('same_turn_agree')}/{c.get('same_turn_n')})" if c.get('same_turn_n') else ""
    value = _finite((row["figures"].get(name) or {}).get("value"))
    return "" if value is None else f" ({round(value * row['n'])}/{row['n']})"


def agreement_cell(row: dict) -> str:
    """The Agreement (α) cell: the identical share (and within one for an ordinal code) with its interval, alpha in
    parentheses; the clock's same-turn share and median seconds; the ask's identical share; a comparison's k/n with its
    interval resampling videos and the exact p."""
    if row["missing"]:
        return NO_PAIRS if row["field"] == PAIR else DASH
    f, kind = row["figures"], row["kind"]
    if kind == "comparison":
        c = row["counts"]
        if not c.get("n"):
            return DASH
        return (f"{c['agree']}/{c['n']} = {_figure(f.get('agreement'))}; p = {_p((f.get('p_vs_half') or {}).get('value'))}"
                f" against {row['against']:g}")
    if kind == "clock":
        median = (f.get("median_abs") or {}).get("value")
        missing = row['counts'].get('same_turn_undetermined', 0)
        suffix = f"; {missing} undetermined" if missing else ''
        return f"same turn {_figure(f.get('same_turn'), _count(row))}; median {_secs(median)}{suffix}"
    cell = _figure(f.get("exact"), _count(row))
    if kind == "ordinal":
        cell = f"identical {cell}; within one {_figure(f.get('within_one'), _count(row, 'within_one'))}"
    if kind in ALPHA_KINDS:
        cell += f" (α {_figure(row['alpha']) if row['alpha'] else DASH})"
    return cell


def _flag_words(row: dict) -> list[str]:
    """A row's flags in words: what to read instead, never a verdict."""
    words = []
    for flag in row["flags"]:
        if flag == "prevalence":
            read = row.get("read") or {}
            label = (read.get("statistic") or "ac1").upper()
            said = "" if read.get("value") is None else f" ({read['value']:.2f}, {read['band']})"
            words.append(f"prevalence: read {label}{said} rather than α")
        elif flag == "small_n":
            words.append(f"small n ({row['units']} units)")
        else:
            words.append(flag)
    return words


def _unsure_cell(row: dict) -> str:
    sure = row["without_unsure"]
    if row["missing"] or not sure or row["kind"] == "comparison":
        return DASH
    value = sure.get(HEADLINE[row["kind"]][0])
    n = sure.get('same_turn_n', 0) if row['kind'] == 'clock' else sure.get('n', 0)
    return f"{_num(value)} ({n})"


def _n_cell(row: dict) -> str:
    """Units (videos), and the judgements when several coders judged a unit: "5 (5); 10 judgements". A unit is a
    pitch for a pitch-level item, a question-answer pair for a turn code, a pair of clips for a comparison."""
    if row["missing"] or row["n"] is None:
        return DASH
    units = row["units"] if row.get("units") is not None else row["n"]
    cell = f"{units} ({row['videos'] if row['videos'] is not None else DASH})"
    return cell + (f"; {row['n']} judgements" if row["n"] != units else "")


def table_lines(rows: list[dict], extra: tuple[str, Callable[[dict], str]] | None = None) -> list[str]:
    """The table of `paper_rows`: one Markdown line per row under HEAD; `extra` (a heading and a cell of each row)
    adds a last column, as `review_panel.py calibrate` adds its bar."""
    head = HEAD + (f" {extra[0]} |" if extra else "")
    out = [head, "|" + "---|" * head.count(" | ") + "---|"]
    for r in rows:
        flags = "; ".join(_flag_words(r)) or DASH
        band = r["band"] or DASH
        out.append(f"| {_cell(r['field'])} | {r['kind'] or DASH} | {_n_cell(r)} | {_cell(agreement_cell(r))} | "
                   f"{band} | {_cell(DASH if r['missing'] else flags)} | {_unsure_cell(r)} |"
                   + (f" {_cell(extra[1](r))} |" if extra else ""))
    return out


# ---- the summary table the paper reports (2026-10-31) ---------------------------------------------------------------

SUMMARY_HEAD = "| Code | Scale | Gemini vs. human: AC1/AC2 (α) | Human 1 vs. human 2: AC1/AC2 (α) |"
# the chance-corrected statistic that leads a summary cell, by kind
SUMMARY_STATISTIC = {"categorical": "ac1", "binary": "ac1", "ordinal": "ac2"}
SMALL_N_MARK = "†"
# each reported code's name in words and its scale; a code missing here is shown by its id and its values
CODE_WORDS = {
    "PR-FRAME": ("Question frame", "promotion / prevention"),
    "RS-RESP": ("Answer addresses the question", "addresses / partial / deflects"),
    "RS-EVID": ("Answer gives a checkable specific", "yes / no"),
    "PR-SEV": ("Question severity", "1–3"),
    "PREP-SUBS": ("Answer substance", "1–5"),
    "PREP-COH": ("Answer coherence", "1–5"),
    "PRES-SUBS": ("Presentation substance", "1–5"),
    "PRES-COH": ("Presentation coherence", "1–5"),
    "PRES-EVID": ("Presentation checkable specifics", "count in bands: 0 / 1–2 / 3–5 / 6+"),
    "pitch_positivity": ("Delivery: pitch positivity", "1–5"),
    "expressed_enthusiasm": ("Delivery: expressed enthusiasm", "1–5"),
    "interpersonal_warmth": ("Delivery: interpersonal warmth", "1–5"),
    "cut": ("First decision", "time (same turn)"),
    "venture:ask": ("The ask", "amount and equity"),
    "outcome": ("Any offer", "yes / no"),
    PAIR: ("Paired comparisons", "which of two clips is higher"),
}
SUMMARY_NOTE = ("Each cell gives Gwet's AC1 (Gwet 2008) for a categorical or binary code and Gwet's AC2 with quadratic "
                "weights (Gwet 2014) for an ordinal code, with Krippendorff's α (nominal or ordinal) in parentheses. "
                "AC1 and AC2 stay interpretable when one value dominates the answers, where α falls even though the "
                "raters mostly agree; α is given beside them because it is the field's common reliability statistic. "
                "Gemini vs. human pools each coder's blind codes against the model's primary pass (a unit both coders "
                "judged counts twice); human 1 vs. human 2 uses the units both coders gave a definite value. The first "
                "decision gives the share of decisions placed in the same turn, the ask the share of identical asks, "
                "and a paired comparison the share of pairs ranked alike, against a coin flip of 0.5 with the exact "
                "binomial p; none has an α. The intervals, raw agreement and the other figures are in the detail "
                "tables below and in the appendix.")


def _code_words(name: str, reporting) -> tuple[str, str]:
    """(the Code cell, the Scale cell) of one row: the code in words with its id, and its values."""
    if name.startswith(f"{PAIR}:"):
        item = name.partition(":")[2]
        words = CODE_WORDS.get(item, (item, ""))[0].removeprefix("Delivery: ")
        return f"Paired comparison: {words} ({name})", CODE_WORDS[PAIR][1]
    words, scale = CODE_WORDS.get(name, (None, None))
    if scale is None:
        values = reporting.values_of(name) if reporting is not None else None
        scale = " / ".join(values) if values else (reporting.kind_of(name) if reporting is not None else None) or DASH
    return (f"{words} ({name})" if words else name), scale


def summary_cell(row: dict, small_n: int | None = None) -> str:
    """A summary cell: "AC1 0.74 (α 0.25)" (AC2 for an ordinal code); a clock's same-turn share, the ask's identical
    share and a comparison's share against 0.5, each with its counts and no α; "—" for a row the comparison lacks.
    A row with the small-n flag ends with †."""
    if row["missing"]:
        return DASH
    f, kind, c = row["figures"], row["kind"], row["counts"]
    if kind in SUMMARY_STATISTIC:
        statistic = SUMMARY_STATISTIC[kind]
        value = _finite((f.get(statistic) or {}).get("value"))
        alpha = _finite((row["alpha"] or {}).get("value"))
        if value is None and alpha is None:
            return DASH
        cell = f"{statistic.upper()} {_num(value)} (α {_num(alpha)})"
    elif kind == "clock":
        k, n = c.get("same_turn_agree"), c.get("same_turn_n")
        cell = f"same turn {_num((f.get('same_turn') or {}).get('value'))}" + (f" ({k}/{n})" if n else "") + "; no α"
    elif kind == "exact":
        k, n = c.get("agree"), c.get("n")
        cell = f"identical {_num((f.get('exact') or {}).get('value'))}" + (f" ({k}/{n})" if n else "") + "; no α"
    elif kind == "comparison":
        if not c.get("n"):
            return DASH
        cell = (f"{_num((f.get('agreement') or {}).get('value'))} ({c['agree']}/{c['n']}) against {row['against']:g}, "
                f"p = {_p((f.get('p_vs_half') or {}).get('value'))}")
    else:
        return DASH
    return cell + (f" {SMALL_N_MARK}" if "small_n" in row["flags"] else "")


def summary_rows(pooled: dict, reporting=None) -> list[dict]:
    """The paper's summary: one row per reported code, in the detail tables' order, with its Code and Scale cells,
    Gemini against the human coders (each coder's blind codes against the model) and human 1 against human 2."""
    model = paper_rows(pooled, reporting)
    human = {r["field"]: r for r in paper_rows(pooled, reporting, "human_human")}
    out = []
    for r in model:
        code, scale = _code_words(r["field"], reporting)
        hh = human.get(r["field"])
        out.append({"field": r["field"], "kind": r["kind"], "code": code, "scale": scale,
                    "gemini_human": summary_cell(r),
                    "human_human": summary_cell(hh) if hh is not None else DASH})
    return out


def summary_markdown(pooled: dict, reporting=None) -> list[str]:
    """The summary section: its table, the coders named, and its note."""
    rows = summary_rows(pooled, reporting)
    raters = next((r["raters"] for r in paper_rows(pooled, reporting, "human_human") if not r["missing"]), None)
    small = reporting.flags["small_n"] if reporting is not None else None
    lines = ["## Agreement by code", "", SUMMARY_HEAD, "|---|---|---|---|"]
    lines += [f"| {_cell(r['code'])} | {_cell(r['scale'])} | {_cell(r['gemini_human'])} | {_cell(r['human_human'])} |"
              for r in rows]
    said = SUMMARY_NOTE
    if raters:
        said += f" Human 1 and human 2 are {' and '.join(raters)}."
    if small is not None:
        said += f" {SMALL_N_MARK} fewer than {small} units, too few for a stable chance-corrected statistic."
    return [*lines, "", f"Note. {said}"]


# ---- a sample stratified by outcome (plan C Task 7f, 2026-10-21) ---------------------------------------------------

STRATUM_WORDS = {("any_offer", "yes"): "with an offer", ("any_offer", "no"): "without an offer"}


def stratum_word(stratify: str | None, stratum: str) -> str:
    return STRATUM_WORDS.get((stratify, stratum)) or f"{stratify} = {stratum}"


def stratum_rows(pooled: dict, reporting=None) -> list[dict]:
    """One row per coder-against-model row of the table (comparisons aside: a pair spans both strata) when the sample
    is stratified: each stratum's raw agreement (identical; within one for an ordinal code; same turn for a clock),
    alpha and judgements, and over the stratified videos the same reweighted to the run's mix beside the pooled
    (unweighted) ones of the main table. Every figure here is a point. Empty when the sample is not stratified."""
    strata = pooled.get("strata")
    if not strata:
        return []
    out = []
    for r in paper_rows(pooled, reporting):
        if r["missing"] or r["kind"] == "comparison":
            continue
        lead = HEADLINE[r["kind"]][0]
        near = "within_one" if r["kind"] == "ordinal" else None

        def point(f: dict) -> dict:
            return {"n": f.get('same_turn_n', 0) if r['kind'] == 'clock' else f.get("n", 0),
                    "share": f.get(lead), "within_one": f.get(near) if near else None,
                    "alpha": f.get("alpha")}
        by_stratum = {s: point(r["strata"].get(s) or {}) for s in strata["strata"]}
        w = r["strata"].get("reweighted") or {}
        out.append({"field": r["field"], "kind": r["kind"], "figure": lead, "by_stratum": by_stratum,
                    "rows": w.get("n", 0), "reweighted_share": w.get(lead),
                    "reweighted_within_one": w.get(near) if near else None, "reweighted_alpha": w.get("alpha"),
                    "unweighted_share": (r["figures"].get(lead) or {}).get("value"),
                    "unweighted_within_one": (r["figures"].get(near) or {}).get("value") if near else None,
                    "unweighted_alpha": (r["alpha"] or {}).get("value")})
    return out


def strata_statement(strata: dict) -> str:
    """What the stratified sample is and how its figures are weighted."""
    info, stratify = strata["strata"], strata.get("stratify")
    # "5 with an offer, 5 without"
    drawn = ", ".join(f"{f['sample']} {stratum_word(stratify, s).removesuffix(' an offer') if i else stratum_word(stratify, s)}"
                      for i, (s, f) in enumerate(info.items()))
    planned = [f.get("planned", f["sample"]) for f in info.values()]
    if planned != [f["sample"] for f in info.values()]:      # a summary before every sampled video is imported
        drawn += f" imported (of {' and '.join(map(str, planned))} drawn)"
    eligible = " and ".join(str(f["run"]) for f in info.values())
    weights = " and ".join(DASH if f["weight"] is None else f"{f['weight']:.2f}" for f in info.values())
    return (f"The sample is stratified by the outcome ({stratify}): {drawn}, drawn from {eligible} eligible pitches "
            f"of the run. A reweighted figure weights each judgement by its stratum's share of the run's eligible "
            f"pitches over its share of the videos pooled here (weights {weights}), so it describes the run's mix; "
            "the pooled figure is the main table's, unweighted. The main table's figures are unweighted.")


def _point_cell(kind: str, share, within_one, alpha, n=None) -> str:
    """A stratum's figures as the main table leads with them, as points: "0.62; within one 1.00 (α 0.29), n 16"."""
    if _finite(share) is None:
        return DASH
    cell = _num(share) + (f"; within one {_num(within_one)}" if kind == "ordinal" else "")
    if kind in ALPHA_KINDS:
        cell += f" (α {_num(alpha)})"
    return cell + (f", n {n}" if n is not None else "")


def strata_markdown(pooled: dict, reporting=None) -> list[str]:
    """The stratum section of the paper table and the summary; no lines when the sample is not stratified."""
    strata = pooled.get("strata")
    if not strata:
        return []
    rows = stratum_rows(pooled, reporting)
    names = list(strata["strata"])
    words = [stratum_word(strata.get("stratify"), s) for s in names]
    head = (["Field"] + [f"{w.capitalize()}: agreement (α), point" for w in words]
            + ["Reweighted: agreement (α), point", "Pooled (main table): agreement (α), point"])
    lines = ["", "## By outcome stratum", "", strata_statement(strata), "",
             "Each cell leads with raw agreement (identical; within one for an ordinal code; same turn for the first "
             "decision) with Krippendorff's α in parentheses for categorical, binary and ordinal codes, and the "
             "judgements (n) in the stratum. Every figure in this section is a point, without an interval. The "
             "comparisons are left out: a pair's two clips can come from both strata.", "",
             f"| {' | '.join(head)} |", "|" + "---|" * len(head)]
    for r in rows:
        k = r["kind"]
        cells = ([_cell(r["field"])]
                 + [_point_cell(k, r["by_stratum"][s]["share"], r["by_stratum"][s]["within_one"],
                                r["by_stratum"][s]["alpha"], r["by_stratum"][s]["n"]) for s in names]
                 + [_point_cell(k, r["reweighted_share"], r["reweighted_within_one"], r["reweighted_alpha"], r["rows"]),
                    _point_cell(k, r["unweighted_share"], r["unweighted_within_one"], r["unweighted_alpha"])])
        lines.append(f"| {' | '.join(cells)} |")
    if not rows:
        lines.append(f"| {' | '.join([DASH] * len(head))} |")
    return lines


# ---- the note and the methods note ---------------------------------------------------------------------------------

def note_of(pooled: dict, adjudicator_only: int = 0) -> dict:
    """The corpus facts the table's note states: coders, videos, the structural rate overall, how many videos
    were left out because only an adjudicator reviewed them, and how many answers to the outcome and clocks of the
    first decision differ from the record's (outcome and first-decision overrides, counted apart) and the seconds
    within which two clocks are one moment."""
    structural = pooled.get("structural") or {}
    return {"coders": len(pooled.get("by_reviewer") or {}), "videos": pooled.get("n_videos"),
            "structural_rows": sum(f["n_rows"] for f in structural.values()),
            "structural_marked": sum(f["n_structural"] for f in structural.values()),
            "adjudicator_only": adjudicator_only,
            "outcome_overrides": pooled.get("n_outcome_overrides"), "cut_overrides": pooled.get("n_cut_overrides"),
            "cut_seconds": CUT_TOLERANCE_S,  # import override counts are timing diagnostics, not turn disagreements
            "rows": pooled.get("n_rows"), "unsure_rows": (pooled.get("without_unsure") or {}).get("n_unsure")}


def _plural(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def _overrides(fields: set[str], note: dict) -> str | None:
    """The outcome and first-decision overrides, counted apart: coder answers that differ from the record's."""
    said = []
    if OUTCOME in fields and note.get("outcome_overrides") is not None:
        n = note["outcome_overrides"]
        said.append(f"{_plural(n, 'coder answer', 'coder answers')} to the outcome "
                    f"{'differs' if n == 1 else 'differ'} from the record's")
    if CUT in fields and note.get("cut_overrides") is not None:
        n = note["cut_overrides"]
        apart = f"more than {note['cut_seconds']:g} s" if note.get("cut_seconds") is not None else "apart"
        said.append(f"{_plural(n, 'first-decision clock', 'first-decision clocks')} "
                    f"{'falls' if n == 1 else 'fall'} {apart} from the record's")
    return (", and ".join(said) + "; none of these changes the record.") if said else None


def _facts(pooled: dict, note: dict) -> str:
    """The note under the main table: what the figures are given, and the corpus."""
    fields = {r["field"] for r in paper_rows(pooled) if not r["missing"]}
    parts = [NOTE]
    if OUTCOME in fields:
        parts.append(OUTCOME_NOTE)
    overrides = _overrides(fields, note)
    if overrides:
        parts.append(overrides[0].upper() + overrides[1:])
    facts = []
    if note.get("videos") is not None:
        facts.append(f"{note['videos']} videos")
    if note.get("coders"):
        facts.append(_plural(note["coders"], "coder", "coders"))
    if facts:
        parts.append(f"Pooled over {' and '.join(facts)}, each coding independently and blind to the model's values.")
    if note.get("structural_rows"):
        parts.append(f"Structural rate overall: {note['structural_marked']} of {note['structural_rows']} reviewed rows "
                     f"({note['structural_marked'] / note['structural_rows']:.1%}).")
    if note.get("adjudicator_only"):
        n = note["adjudicator_only"]
        parts.append(f"{n} further video{' was' if n == 1 else 's were'} reviewed by an adjudicator only and "
                     f"{'is' if n == 1 else 'are'} not in this table.")
    return " ".join(parts)


def methods_note(pooled: dict, reporting, note: dict | None = None) -> str:
    """The methods note, generated, in paragraphs: why raw agreement leads and alpha is in parentheses; the reporting
    file's reason for each kind of code present (with the rows of that kind); the bands' source, cut points and
    caveat; which figures carry an interval and which are points; the flags; coder against coder, pass against pass,
    the unsure and off-scale conventions; each code not reported with its reason."""
    note = note or {}
    rows = [r for which, _ in COMPARISONS for r in paper_rows(pooled, reporting, which) if not r["missing"]]
    present: dict[str, list[str]] = {}
    for r in rows:
        present.setdefault(r["kind"], [])
        if r["field"] not in present[r["kind"]]:
            present[r["kind"]].append(r["field"])
    paragraphs = [HEADLINE_NOTE]
    for kind in ("categorical", "binary", "ordinal", "clock", "exact", "comparison"):
        if kind in present:
            paragraphs.append(f"{KIND_WORDS[kind]} ({', '.join(present[kind])}): {reporting.reason_of(kind)}")
    paragraphs.append(reporting.bands_reason() + " A band describes the value.")
    resampling = (f"The Agreement and α cells of the tables carry a 95% interval from resampling whole videos with "
                  f"replacement (a seeded cluster bootstrap of {N_BOOT:,} draws), because the units of one video are "
                  f"not independent; so do AC1, AC2, the positive and negative agreement and the modal value's share "
                  f"and the clock's share within 30 s in the appendix. A row of fewer than {MIN_CLUSTER_VIDEOS} videos, or a figure undefined in more "
                  f"than {1 - MIN_DEFINED_SHARE:.0%} of the draws, has none ([{DASH}]). Every other figure is a point: "
                  "the column without unsure units, the figures per stratum and reweighted, the agreement on each "
                  "value, the modal share per side, the chance values, the indices, the mean difference, ρ and the "
                  "other seconds figures.")
    if "comparison" in present:
        resampling += (" A comparison's interval resamples videos too, each pair weighted by how often each of its "
                       "two videos is drawn; its figures per third of the gaps are counts.")
    paragraphs.append(resampling)
    paragraphs.append(f"Flags describe a row and change nothing in it. Prevalence: the most common value is at least "
                      f"{reporting.flags['prevalence']:.0%} of the two raters' answers taken together; the flag "
                      "gives AC1 (AC2 for an ordinal code) with its band, to read rather than α. Small n: a row has "
                      f"fewer than {reporting.flags['small_n']} units, counting each pitch, question–answer pair or "
                      "pair of clips once however many coders judged it; the counts are shown beside the leading "
                      "share, and a chance-corrected statistic on this few units is unstable. When two coders' "
                      "judgements are pooled, the units column gives the judgements beside the units, and every "
                      "figure counts each judgement.")
    conventions = [PASSES_NOTE]
    if note.get("unsure_rows") and note.get("rows"):
        conventions.append(f"The column without unsure repeats the row's leading figure, as a point, without the units "
                           f"a coder answered with a best guess and ticked not sure ([unsure] in the note): "
                           f"{note['unsure_rows']} of {note['rows']} reviewed rows. Every other figure uses every unit.")
    conventions.append(OFF_SCALE_NOTE)
    if any(r["field"] in _stage_a_items(reporting) for r in rows):
        conventions.append(STAGE_A_NOTE)
    paragraphs.append(" ".join(conventions))
    if reporting.not_reported:
        paragraphs.append(" ".join(reporting.reason_not_reported(code) for code in reporting.not_reported))
    return "\n\n".join(paragraphs)


def _stage_a_items(reporting) -> set[str]:
    """The Stage A items among the reported codes (aliased as `stage_a:<item>`)."""
    return {code for key, code in reporting.aliases.items() if key.startswith("stage_a:")}


def paper_table_markdown(pooled: dict, reporting=None, note: dict | None = None) -> str:
    """The paper table: the summary (AC1 or AC2 with alpha, Gemini vs human and human vs human), then the detail
    tables: coder against model with its note, coder against coder and model pass against pass with the same rows, the
    stratum section of a stratified sample, then the methods note."""
    note = note or {}
    rows = paper_rows(pooled, reporting)
    lines = ["# Human validation of the instrument", "", *summary_markdown(pooled, reporting), "",
             "## Coder against model", "", *table_lines(rows), "",
             f"Note. {_facts(pooled, note)}"]
    hh = paper_rows(pooled, reporting, "human_human")
    raters = next((r["raters"] for r in hh if not r["missing"] and r["kind"] != "comparison"), None) or next(
        (r["raters"] for r in hh if not r["missing"]), None)
    lines += ["", "## Coder against coder", "",
              (f"The videos judged by two coders ({' and '.join(raters)}), on the units both gave a definite value."
               if raters else "No video was judged by two coders."), "", *table_lines(hh)]
    pp = paper_rows(pooled, reporting, "pass_to_pass")
    lines += ["", "## Model pass against pass", "",
              "The model's primary pass against its replicate on the sampled units that have both, each unit once.",
              "", *table_lines(pp)]
    lines += strata_markdown(pooled, reporting)
    if reporting is not None:
        lines += ["", "## Note on the statistics", "", methods_note(pooled, reporting, note)]
    return "\n".join(lines) + "\n"


# ---- the appendix --------------------------------------------------------------------------------------------------

def definitions(reporting=None) -> list[tuple[str, str]]:
    """The appendix's definitions of every statistic, with the flags' levels from the reporting file."""
    prevalence = f"{reporting.flags['prevalence']:.0%}" if reporting is not None else "the reporting file's level"
    small = f"{reporting.flags['small_n']}" if reporting is not None else "the reporting file's level of"
    return [
        ("Identical share", "the share of units on which the two sides give the same value; for the ask, amounts "
                            "within 1% and the same equity."),
        ("Within-one share", "the share of units whose two values are at most one step apart on the scale."),
        ("Chance identical share, chance within-one share", "the share two raters with these same value "
                                                            "distributions would reach answering independently."),
        ("Krippendorff's α", "1 minus observed over expected disagreement (nominal: any difference; ordinal: "
                             "Krippendorff's ordinal distance); how reliably the values are told apart beyond chance."),
        ("Gwet's AC1", "chance-corrected agreement whose chance term stays small when one value dominates (Gwet "
                       "2008)."),
        ("Gwet's AC2", "AC1 for an ordered scale with quadratic weights, crediting near misses (Gwet 2014)."),
        ("Agreement on each value", "for a value v, twice the units both gave v over the two sides' counts of v "
                                    "(specific agreement); for a binary code these are the positive and negative "
                                    "agreement (Cicchetti & Feinstein 1990)."),
        ("Modal value's share", "the share of the two sides' answers, taken together, that are the most common "
                                "value; also given per side."),
        ("Prevalence index", "|a − d| / n of the 2×2 table (Byrt, Bishop & Carlin 1993): how uneven the yes/no split "
                             "is."),
        ("Bias index", "|b − c| / n of the 2×2 table (Byrt, Bishop & Carlin 1993), with the side that says yes "
                       "more."),
        ("Mean difference", "the first side minus the second, on the scale's positions."),
        ("Spearman's ρ", "the rank correlation of the two sides' values: whether they order the units alike."),
        ("Same-turn agreement", "both times identify the same turn in the sheet's pinned reference pass. Turn "
                                "starts are included and ends excluded; missing context, gaps and overlaps are "
                                "undetermined and excluded from this share's denominator."),
        ("Share within 5 s and within 30 s", "timing diagnostics: the share of clocks at most 5 s (30 s) apart."),
        ("Median absolute difference, mean signed difference", "of the clocks, in seconds (first side minus "
                                                               "second)."),
        ("Limits of agreement", "the mean signed difference ± 1.96 standard deviations (Bland & Altman 1986)."),
        ("Comparison agreement", "k of n judgements that rank the two clips as the model's rank scores do, with the "
                                 "95% interval resampling videos (a pair weighted by how often each of its two videos "
                                 "is drawn) and the two-sided exact binomial p against 0.5; the appendix adds the "
                                 "Clopper–Pearson exact interval, which, like the p, treats the judgements as "
                                 "independent and so, since pairs share clips, is usually the narrower; the same "
                                 "counts per third of the pairs ordered by the model's gap; the close ticks."),
        ("Interval", "the 95% percentile interval of the figure over resamples of whole videos (seeded); none with "
                     f"fewer than {MIN_CLUSTER_VIDEOS} videos, or when the figure is undefined in more than "
                     f"{1 - MIN_DEFINED_SHARE:.0%} of the resamples. The main tables' Agreement and α cells carry "
                     "one, and in the appendix AC1, AC2, the positive and negative agreement the modal value's "
                     "share and the clock's share within 30 s; every other figure (without unsure units, per stratum, reweighted, the agreement on each "
                     "value, the modal share per side, the chance values, the indices, the mean difference, ρ, the "
                     "other seconds figures) is a point."),
        ("Band", "the Landis & Koch (1977) name of a value: below 0 poor, 0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 "
                 "moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect; the table's column names α's, the "
                 "prevalence flag AC1's or AC2's. A descriptive name, not a calibrated benchmark."),
        ("Prevalence flag", f"the most common value is at least {prevalence} of the two sides' answers taken "
                            "together: read AC1 or AC2 (given in the flag with its band) rather than α."),
        ("Small-n flag", f"fewer than {small} units (distinct pitches, question–answer pairs or pairs of clips; a "
                         "unit two coders judged counts once): the counts are shown; a chance-corrected statistic on "
                         "this few units is unstable."),
    ]


FIGURE_WORDS = {"exact": "identical share", "within_one": "within-one share", "chance_exact": "chance identical share",
                "chance_within_one": "chance within-one share", "alpha": "Krippendorff's α", "ac1": "Gwet's AC1",
                "ac2": "Gwet's AC2", "modal_share": "modal value's share", "positive_agreement": "positive agreement",
                "negative_agreement": "negative agreement", "prevalence_index": "prevalence index",
                "bias_index": "bias index", "mean_difference": "mean difference", "spearman": "Spearman's ρ",
                "same_turn": "same-turn agreement", "share_within_5": "share within 5 s", "share_within_30": "share within 30 s",
                "median_abs": "median absolute difference", "mean_signed": "mean signed difference",
                "agreement": "agreement", "p_vs_half": "p"}
POINTS = ("chance_exact", "chance_within_one", "mean_difference", "spearman", "prevalence_index", "bias_index")
SECONDS = ("median_abs", "mean_signed")


def _specific(confusion: dict) -> dict[str, float | None]:
    """The agreement on each value of a confusion matrix ({first: {second: n}}): 2 n_vv / (n_v. + n_.v)."""
    out = {}
    for v in confusion:
        row = sum(confusion[v].values())
        col = sum(confusion[u].get(v, 0) for u in confusion)
        out[v] = (2 * confusion[v].get(v, 0) / (row + col)) if row + col else None
    return out


def _matrix(confusion: dict, raters: list[str]) -> list[str]:
    values = list(confusion)
    first, second = raters
    lines = [f"| {_cell(first)} \\ {_cell(second)} | {' | '.join(_cell(v) for v in values)} |",
             "|" + "---|" * (len(values) + 1)]
    lines += [f"| {_cell(v)} | {' | '.join(str(confusion[v].get(u, 0)) for u in values)} |" for v in values]
    return lines


def _detail(r: dict, clocks: list[dict] | None) -> list[str]:
    """Every figure of one row, the counts behind it and its tables."""
    f, kind = r["figures"], r["kind"]
    first, second = r["raters"] or ["first", "second"]
    units = r["units"] if r.get("units") is not None else r["n"]
    unit = "pairs of clips" if kind == "comparison" else "units"
    lines = ["", f"### {_cell(r['field'])} ({kind}; {first} against {second})", "",
             f"- {unit}: {units} in {r['videos']} videos"
             + (f"; {r['n']} judgements (a unit judged by two coders counts twice in every figure below, once in "
                "the small-n flag)" if r["n"] != units else "")
             + (f"; {r['n_off_scale']} left out off the scale" if r["n_off_scale"] else "")]
    if kind == "comparison":
        c = r["counts"]
        binomial = f.get("binomial_interval") or {}
        lines.append(f"- agreement: {c.get('agree')}/{c.get('n')} = {_num((f.get('agreement') or {}).get('value'))}; "
                     f"p = {_p((f.get('p_vs_half') or {}).get('value'))} against {r['against']:g} (exact binomial)")
        lines.append(f"- 95% interval resampling videos (the table's): {_interval(f.get('agreement'))}; exact "
                     f"(Clopper–Pearson): {_interval(binomial)}. The exact interval and p treat the judgements as "
                     "independent; pairs that share a clip are not, so the interval resampling videos is usually "
                     "the wider, and it is the one the table gives.")
        lines.append(f"- close ticks: {DASH if c.get('close') is None else c['close']}")
        terciles = (f.get("by_gap_tercile") or {}).get("value") or []
        if terciles:
            lines += ["", "| Third of the model's gaps | Agreeing | Judged | Gaps |", "|---|---|---|---|"]
            for t in terciles:
                gap = t.get("gap_range")
                span = DASH if not gap else f"{gap[0]:.2f}–{gap[1]:.2f}"
                lines.append(f"| {t.get('tercile')} | {DASH if t.get('agree') is None else t['agree']} | "
                             f"{DASH if t.get('n') is None else t['n']} | {span} |")
        return lines
    for name, figure in f.items():
        if name in ("confusion", "counts"):
            continue
        if name == "bias_index":
            lines.append(f"- bias index: {_num(figure.get('value'))} ({figure.get('direction', DASH)} says yes more)")
        elif name == "modal_share":
            by = figure.get("by_rater") or [None, None]
            lines.append(f"- modal value's share: {_figure(figure)}; {first} {_num(by[0])}, {second} {_num(by[1])}")
        elif name in SECONDS:
            lines.append(f"- {FIGURE_WORDS[name]}: {_secs(figure.get('value'))}")
        elif name in ("loa_low", "loa_high"):
            continue
        elif name in POINTS:
            lines.append(f"- {FIGURE_WORDS[name]}: {_num(figure.get('value'))}")
        else:
            lines.append(f"- {FIGURE_WORDS.get(name, name)}: {_figure(figure)}")
    if kind == "clock":
        lines.append(f"- limits of agreement: {_secs((f.get('loa_low') or {}).get('value'))} to "
                     f"{_secs((f.get('loa_high') or {}).get('value'))} (Bland & Altman)")
        c = r["counts"]
        lines.append(f"- same turn: {c.get('same_turn_agree', 0)} of {c.get('same_turn_n', 0)} mapped comparisons; "
                     f"{c.get('same_turn_undetermined', 0)} undetermined")
        lines.append(f"- within 5 s: {c.get('within_5')} of {c.get('n')}; within 30 s: {c.get('within_30')} of "
                     f"{c.get('n')}")
        mine = [x for x in clocks or []] if r["comparison"] == "coder_model" else []
        if mine:
            lines += ["", "| Video | Coder | Coder's clock | Model's clock | Difference (s) |", "|---|---|---|---|---|"]
            lines += [f"| {_cell(x['video'])} | {_cell(x['coder'] or DASH)} | {x['coder_clock']} | {x['model_clock']} | "
                      f"{x['difference']:g} |" for x in mine]
    if kind == "exact":
        c = r["counts"]
        lines.append(f"- identical: {c.get('agree')} of {c.get('n')}")
    if kind in ALPHA_KINDS:
        values = list(r["coder_values"])
        lines.append(f"- {first}'s values: " + ", ".join(f"{_cell(v)} {r['coder_values'][v]}" for v in values))
        lines.append(f"- {second}'s values: " + ", ".join(f"{_cell(v)} {r['model_values'].get(v, 0)}" for v in values))
    if kind in ("categorical", "ordinal") and isinstance(r["counts"], dict) and r["counts"]:
        spec = _specific(r["counts"])
        lines.append("- agreement on each value: " + ", ".join(f"{_cell(v)} {_num(x)}" for v, x in spec.items()))
        lines += ["", *_matrix(r["counts"], [first, second])]
    if kind == "binary" and r["counts"]:
        c = r["counts"]
        positive = r.get("positive") or next(iter(r["coder_values"]), "yes")
        negative = next((v for v in r["coder_values"] if v != positive), "no")
        lines += ["", *_matrix({positive: {positive: c["a"], negative: c["b"]},
                                negative: {positive: c["c"], negative: c["d"]}}, [first, second])]
    return lines


def clock_differences(videos: dict[str, list[dict]]) -> list[dict]:
    """The first decision per video and coder, as the table pools it: the coder's clock (the one typed beside a
    confirmation, when the line carries it) against the model's, and their difference in seconds (coder minus
    model). Lines that are not a sampled, definite first decision, or whose clocks do not read, are left out."""
    from pipeline.stage_b.review_apply import Correction, _judgment, _seconds, _text
    from pipeline.stage_b.review_items import is_cut
    out = []
    for video_id, lines in sorted(videos.items()):
        for line in lines:
            if not is_cut(line.get("path")):
                continue
            judged = _judgment(Correction(line["path"], line.get("new"), line["resolution"], line.get("note") or "",
                                          ""), line)
            if judged[0] != "item" or judged[1] != "sampled":
                continue
            human, primary = judged[3][0], judged[3][1]
            if line["resolution"] == "confirm_primary" and _text(line.get("typed")) is not None:
                human = _text(line.get("typed"))
            a, b = _seconds(human), _seconds(primary)
            if a is None or b is None:
                continue
            out.append({"video": video_id, "coder": line.get("reviewer"), "coder_clock": human,
                        "model_clock": primary, "difference": a - b})
    return out


def paper_appendix_markdown(pooled: dict, reporting=None, clocks: list[dict] | None = None) -> str:
    """Every figure the table leaves out, per comparison and row (Gwet's AC1/AC2, the agreement on each value, the
    chance values, the mean difference and ρ, both sides' values, the confusion matrix or 2×2 table, the comparisons
    by third of the gaps, the clocks per video and the limits of agreement), the structural rate, and the
    definitions of every statistic."""
    lines = ["# Validation appendix"]
    for which, title in COMPARISONS:
        rows = [r for r in paper_rows(pooled, reporting, which) if not r["missing"]]
        lines += ["", f"## {title}"]
        if not rows:
            lines += ["", DASH]
        for r in rows:
            lines += _detail(r, clocks)
    lines += ["", "## Structural rate per field class", "", "| Field class | Rows | Structural | Share |",
              "|---|---|---|---|"]
    for cls, f in (pooled.get("structural") or {}).items():
        lines.append(f"| {_cell(cls)} | {f['n_rows']} | {f['n_structural']} | {_num(f['share'])} |")
    lines += ["", "## Statistics and their definitions", ""]
    lines += [f"- **{name}**: {text}" for name, text in definitions(reporting)]
    return "\n".join(lines) + "\n"



# ---- the reporting configuration: the statistics of each kind of code (plan D, the researcher's design of 2026-10-30)

REPORTING_KINDS = ("categorical", "binary", "ordinal", "clock", "exact", "comparison")
# the statistics the summary knows how to compute for each kind; a name outside them would print nothing, so the
# loader refuses it
REPORTING_STATISTICS = {
    "categorical": ("exact", "alpha", "ac1", "modal_share", "confusion"),
    "binary": ("counts", "exact", "alpha", "positive_agreement", "negative_agreement", "ac1", "modal_share",
               "prevalence_index", "bias_index"),
    "ordinal": ("exact", "chance_exact", "within_one", "chance_within_one", "alpha", "ac2", "modal_share",
                "mean_difference", "spearman"),
    "clock": ("same_turn", "share_within_5", "share_within_30", "median_abs", "mean_signed", "loa_low", "loa_high"),
    "exact": ("exact",),
    "comparison": ("agreement", "binomial_interval", "p_vs_half", "by_gap_tercile", "close"),
}
REPORTING_DERIVATIONS = {"frame_of": frame_of}     # `via:` of a derived row: the function applied to both sides' values
# the band scheme `bands:` names, and the sentence the methods note prints for it (with its cut points)
REPORTING_BANDS = {
    "landis_koch": ("Bands follow Landis & Koch (1977): below 0 poor, 0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 "
                    "moderate, 0.61–0.80 substantial, 0.81–1.00 almost perfect, applied to α, the statistic in "
                    "parentheses; on a prevalence-flagged row the flag gives AC1's or AC2's band. Landis & Koch "
                    "proposed them for κ-type statistics, and they are used here as descriptive names, not calibrated "
                    "benchmarks (Gwet 2014 argues for benchmarking agreement coefficients probabilistically)."),
}
# each scheme's function from a value to its band, and the lowest two-decimal value of each band
REPORTING_BAND_OF = {"landis_koch": landis_koch_band}
REPORTING_BAND_FLOORS = {"landis_koch": {"slight": 0.0, "fair": 0.21, "moderate": 0.41, "substantial": 0.61,
                                         "almost perfect": 0.81}}
AGAINST = "{against}"           # the comparison reason's placeholder for `comparisons.against`
REPORTING_KEYS = ("kinds", "codes", "derived", "not_reported", "flags", "bands", "comparisons")
BINARY_VALUES = ("true", "false")
# the kinds a code of each instrument type can be reported as; a count only in the bands a coder's sheet takes
_KINDS_OF_TYPE = {"categorical": ("categorical",), "binary": ("binary",), "ordinal": ("ordinal",)}


@dataclass(frozen=True)
class Reporting:
    """validation_reporting.yaml, checked against the instrument. `codes` and `derived` map a key to
    {"kind", "values", "positive"} ("values" the ordered or categorical values, None for a clock, the ask and the
    comparisons; "positive" the yes of a binary code); `derived` entries also carry "from" and "via". The lookups
    take the variable keys of the pooled summary: a Stage A item as `stage_a:<item>`, a prepared code as
    `prep_pres:<code>`, a pair dimension as `pair:<item>` (reported as `pair` unless it has a row of its own)."""
    kinds: dict[str, dict]
    codes: dict[str, dict]
    derived: dict[str, dict]
    not_reported: list[str]
    not_reported_reasons: dict[str, str]
    flags: dict[str, float]
    bands: str
    comparisons: dict[str, float]
    release_id: str | None = None
    source: str | None = None
    sha256: str | None = None          # of the file read, so a table records which reporting file it was built from
    aliases: dict[str, str] = field(default_factory=dict, repr=False)

    def _entry(self, variable: str) -> dict | None:
        key = self.aliases.get(variable, variable)
        return self.codes.get(key) or self.derived.get(key)

    def kind_of(self, variable: str) -> str | None:
        """The kind a variable is reported as; None for a variable not reported (PR-CONCEPT, a class with no row)."""
        entry = self._entry(variable)
        return entry["kind"] if entry else None

    def values_of(self, variable: str) -> list[str] | None:
        entry = self._entry(variable)
        return list(entry["values"]) if entry and entry["values"] is not None else None

    def positive_of(self, variable: str) -> str | None:
        entry = self._entry(variable)
        return entry["positive"] if entry else None

    def reason_of(self, kind: str) -> str:
        """The methods note's sentences for a kind of code, the comparison's with `comparisons.against` filled in."""
        return self.kinds[kind]["reason"].replace(AGAINST, f"{self.comparisons['against']:g}")

    def bands_reason(self) -> str:
        """The methods note's sentence for the band scheme: its source and its cut points."""
        return REPORTING_BANDS[self.bands]

    def band_of(self, value: float | None) -> str | None:
        """The band a value falls in (None for no value)."""
        return REPORTING_BAND_OF[self.bands](value)

    def band_floor(self, band: str) -> float:
        """The lowest value of a named band (Landis & Koch "substantial": 0.61); KeyError for a band the scheme lacks.
        `review_panel.py calibrate` reads its default bar here."""
        return REPORTING_BAND_FLOORS[self.bands][band]

    def statistics_of(self, kind: str) -> list[str]:
        return list(self.kinds[kind]["statistics"])

    def reason_not_reported(self, code: str) -> str:
        return self.not_reported_reasons[code]


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _scale_values(entry) -> list[str] | None:
    """The ordered values an ordinal code is answered in: its scale's whole steps, or the bands of the evidence
    count."""
    if entry.id == EVID_CODE:
        return list(EVID_BANDS)
    if entry.type == "ordinal" and entry.scale:
        low, high = entry.scale
        return [str(v) for v in range(int(low), int(high) + 1)]
    return None


def _instrument_code(key: str, books: PanelBooks, source) -> tuple[tuple[str, ...], list[str] | None]:
    """(the kinds `key` can be reported as, its values in the instrument) for a code or sheet field the instrument
    has; refuses any other key."""
    stage_a = set(books.stage_a_ids())
    panel = {e.id: e for e in books.panel.entries}
    if key in panel or key in stage_a:
        entry = panel[key] if key in panel else books.stage_a.by_id(key)
        if entry.id == EVID_CODE:
            return ("ordinal",), list(EVID_BANDS)
        kinds = _KINDS_OF_TYPE.get(entry.type, ())
        values = (_scale_values(entry) if entry.type == "ordinal" else list(BINARY_VALUES) if entry.type == "binary"
                  else list(entry.values or []) or None)
        return kinds, values
    unit, _, item = key.partition(":")
    if unit == "venture" and item in VENTURE_CODES:
        return (("exact",), None) if item == ASK else (("categorical",), list(books.venture_values(VENTURE_CODES[item])))
    if key == CUT:
        return ("clock",), None
    if key == OUTCOME:
        return ("categorical",), list(OUTCOME_VALUES)
    if key == PAIR or (unit == PAIR and item in stage_a):
        return ("comparison",), None
    hint = (f"; a Stage A item and a prepared presentation code are keyed by their bare id ({item!r})"
            if unit in ("stage_a", "prep_pres") else "")
    raise ConfigError(f"{source}: {key!r} names no code of this instrument{hint}")


def _code_entry(key: str, raw, books: PanelBooks, kinds: dict, source) -> dict:
    """One `codes:` entry (a kind, or a mapping with `kind` and its extras) checked against the instrument."""
    entry = {"kind": raw} if isinstance(raw, str) else raw
    if not isinstance(entry, dict) or "kind" not in entry:
        raise ConfigError(f"{source}: code {key!r} must be a kind or a mapping with `kind`")
    extra = set(entry) - {"kind", "values", "positive"}
    if extra:
        raise ConfigError(f"{source}: code {key!r} has {sorted(extra)}; a code takes kind, values and positive")
    kind = entry["kind"]
    if not isinstance(kind, str) or kind not in kinds:
        raise ConfigError(f"{source}: code {key!r} has kind {kind!r}; choose from {list(REPORTING_KINDS)}")
    allowed, known = _instrument_code(key, books, source)
    if kind not in allowed:
        raise ConfigError(f"{source}: code {key!r} cannot be reported as {kind!r}: the instrument's code is "
                          f"{' or '.join(allowed) or 'not judged on a coder sheet'}")
    values = entry.get("values")
    if values is not None:
        if not isinstance(values, list) or not values or not all(isinstance(v, str) for v in values):
            raise ConfigError(f"{source}: code {key!r}: values must be a non-empty list of strings")
        if len(set(values)) != len(values):
            raise ConfigError(f"{source}: code {key!r}: values {values} repeat a value")
    if kind == "ordinal":
        if values is None:
            raise ConfigError(f"{source}: ordinal code {key!r} lacks `values`, the scale's values in order")
        if values != known:
            raise ConfigError(f"{source}: ordinal code {key!r} has values {values}; the instrument answers it in "
                              f"{known}")
    elif kind == "categorical":
        if values is not None and set(values) != set(known or []):
            raise ConfigError(f"{source}: categorical code {key!r} has values {values}; the instrument's are {known}")
        values = values if values is not None else known
    elif kind == "binary":
        if entry.get("positive") not in BINARY_VALUES:
            raise ConfigError(f"{source}: binary code {key!r} needs `positive`, the value that counts as yes: one of "
                              f"{list(BINARY_VALUES)} (got {entry.get('positive')!r})")
        values = list(BINARY_VALUES)
    elif values is not None:
        raise ConfigError(f"{source}: code {key!r}: a {kind} code takes no values")
    if "positive" in entry and kind != "binary":
        raise ConfigError(f"{source}: code {key!r}: `positive` belongs to a binary code")
    return {"kind": kind, "values": values, "positive": entry.get("positive")}


def check_reporting(data, books: PanelBooks, source, release_id: str | None = None) -> Reporting:
    """validation_reporting.yaml as a Reporting, refusing what the summary cannot apply: an unknown kind, a kind
    without statistics or reason, a statistic the kind does not have, a code the instrument lacks or one reported as a
    kind its type does not allow, an ordinal code without its ordered values (or with values the sheet does not take),
    a binary code without its positive value, a derived row that is not the derivation's own values, and flags,
    bands or a comparison level out of range."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: must be a mapping")
    unknown = set(data) - set(REPORTING_KEYS)
    if unknown:
        raise ConfigError(f"{source}: unknown keys {sorted(unknown)}; the file has {list(REPORTING_KEYS)}")
    kinds = data.get("kinds")
    if not isinstance(kinds, dict):
        raise ConfigError(f"{source}: `kinds` must be a mapping")
    for kind, spec in kinds.items():
        if not isinstance(kind, str) or kind not in REPORTING_KINDS:
            raise ConfigError(f"{source}: kind {kind!r} is not one the summary computes; choose from "
                              f"{list(REPORTING_KINDS)}")
        if not isinstance(spec, dict) or set(spec) != {"statistics", "reason"}:
            raise ConfigError(f"{source}: kind {kind!r} takes `statistics` and `reason`")
        stats = spec["statistics"]
        if not isinstance(stats, list) or not stats:
            raise ConfigError(f"{source}: kind {kind!r}: statistics must be a non-empty list")
        foreign = [s for s in stats if s not in REPORTING_STATISTICS[kind]]
        if foreign:
            raise ConfigError(f"{source}: kind {kind!r} has statistics {foreign} it does not compute; choose from "
                              f"{list(REPORTING_STATISTICS[kind])}")
        if not isinstance(spec["reason"], str) or not spec["reason"].strip():
            raise ConfigError(f"{source}: kind {kind!r}: reason must be the methods note's text")
        if kind == "comparison" and AGAINST not in spec["reason"]:
            raise ConfigError(f"{source}: kind 'comparison': the reason names the level it is tested against as "
                              f"{AGAINST} (filled from comparisons.against), not as a number of its own")
    missing = [k for k in REPORTING_KINDS if k not in kinds]
    if missing:
        raise ConfigError(f"{source}: kinds {missing} have no statistics or reason")
    raw_codes = data.get("codes")
    if not isinstance(raw_codes, dict) or not raw_codes:
        raise ConfigError(f"{source}: `codes` must be a non-empty mapping")
    codes = {str(k): _code_entry(str(k), v, books, kinds, source) for k, v in raw_codes.items()}
    derived = {}
    for key, raw in (data.get("derived") or {}).items():
        key = str(key)
        if not isinstance(raw, dict) or set(raw) != {"from", "via", "kind", "values"}:
            raise ConfigError(f"{source}: derived {key!r} takes from, via, kind and values")
        if key in codes or key in {e.id for e in books.panel.entries} or key in books.stage_a_ids():
            raise ConfigError(f"{source}: derived {key!r} is already a code; a derived row needs a name of its own")
        via = REPORTING_DERIVATIONS.get(raw["via"]) if isinstance(raw["via"], str) else None
        if via is None:
            raise ConfigError(f"{source}: derived {key!r} has via {raw['via']!r}; choose from "
                              f"{list(REPORTING_DERIVATIONS)}")
        if not isinstance(raw["kind"], str) or raw["kind"] not in kinds:
            raise ConfigError(f"{source}: derived {key!r} has kind {raw['kind']!r}; choose from {list(REPORTING_KINDS)}")
        _, of = _instrument_code(str(raw["from"]), books, source)
        produced = {via(v) for v in of or []} - {None}
        if not isinstance(raw["values"], list) or set(raw["values"]) != produced or len(raw["values"]) != len(produced):
            raise ConfigError(f"{source}: derived {key!r} has values {raw['values']}; {raw['via']} of "
                              f"{raw['from']}'s values gives {sorted(produced)}")
        derived[key] = {"kind": raw["kind"], "values": list(raw["values"]), "positive": None,
                        "from": str(raw["from"]), "via": raw["via"]}
    not_reported = data.get("not_reported") or {}
    if not isinstance(not_reported, dict):
        raise ConfigError(f"{source}: `not_reported` must map each code to the reason it has no row")
    for code, reason in not_reported.items():
        _instrument_code(str(code), books, source)
        if code in codes:
            raise ConfigError(f"{source}: {code!r} is both reported (codes) and not reported")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"{source}: not_reported {code!r} needs the reason it has no row")
    flags = data.get("flags")
    if not isinstance(flags, dict) or set(flags) != {"prevalence", "small_n"}:
        raise ConfigError(f"{source}: `flags` takes prevalence and small_n (got {sorted(flags or {})})")
    if not _is_number(flags["prevalence"]) or not 0 < flags["prevalence"] <= 1:
        raise ConfigError(f"{source}: flags prevalence {flags['prevalence']!r} must be a share in (0, 1]")
    if isinstance(flags["small_n"], bool) or not isinstance(flags["small_n"], int) or flags["small_n"] < 1:
        raise ConfigError(f"{source}: flags small_n {flags['small_n']!r} must be a whole number of units, 1 or more")
    if not isinstance(data.get("bands"), str) or data["bands"] not in REPORTING_BANDS:
        raise ConfigError(f"{source}: bands {data.get('bands')!r}; choose from {list(REPORTING_BANDS)}")
    comparisons = data.get("comparisons")
    if (not isinstance(comparisons, dict) or set(comparisons) != {"against"} or not _is_number(comparisons["against"])
            or not 0 < comparisons["against"] < 1):
        raise ConfigError(f"{source}: comparisons takes `against`, a share in (0, 1) (got {comparisons!r})")
    if comparisons["against"] != 0.5:
        # the comparison's p is reported as p_vs_half (review_apply._comparison_row): only a coin flip is tested
        raise ConfigError(f"{source}: comparisons against {comparisons['against']!r}: the comparisons are read against "
                          "0.5 only (the figure is p_vs_half)")
    stage_a = set(books.stage_a_ids())
    prepared = {e.id for e in books.prepared_codes()}
    aliases = {**{f"stage_a:{k}": k for k in codes if k in stage_a},
               **{f"prep_pres:{k}": k for k in codes if k in prepared},
               **({f"{PAIR}:{item}": PAIR for item in stage_a if f"{PAIR}:{item}" not in codes} if PAIR in codes else {})}
    return Reporting(kinds={k: {"statistics": list(v["statistics"]), "reason": v["reason"].strip()}
                            for k, v in kinds.items()},
                     codes=codes, derived=derived, not_reported=[str(c) for c in not_reported],
                     not_reported_reasons={str(c): r.strip() for c, r in not_reported.items()},
                     flags=dict(flags), bands=data["bands"], comparisons=dict(comparisons),
                     release_id=release_id, source=getattr(source, "name", None), aliases=aliases)


def load_reporting(books: PanelBooks, release: InstrumentRelease | None = None) -> Reporting:
    """The instrument's validation_reporting.yaml (beside the release manifest), checked by `check_reporting`. A
    release without the file is refused: the summary cannot tell which statistics a code is reported with."""
    import yaml
    release = release or load_release()
    source = release.validation_reporting
    if source is None or not source.exists():
        raise ConfigError(f"instrument release {release.release_id!r} has no {VALIDATION_REPORTING_FILE} beside its "
                          "manifest: it names the statistics each code is reported with")
    raw = source.read_bytes()
    try:
        data = yaml.load(raw.decode("utf-8"), Loader=_unique_key_loader()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: not valid YAML ({exc})") from exc
    return dataclasses.replace(check_reporting(data, books, source, release.release_id),
                               sha256=hashlib.sha256(raw).hexdigest())


def _unique_mapping(loader, node, deep=False):
    """A YAML mapping that refuses a key written twice (the later one would silently win)."""
    import yaml
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError:           # a complex key (`? [a, b]`): no name of this file is one
            raise yaml.constructor.ConstructorError(None, None, f"key {key!r} is not a plain name",
                                                    key_node.start_mark) from None
        if key in seen:
            raise yaml.constructor.ConstructorError(None, None, f"key {key!r} is written twice",
                                                    key_node.start_mark)
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


def _unique_key_loader():
    import yaml

    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)
    return Loader
