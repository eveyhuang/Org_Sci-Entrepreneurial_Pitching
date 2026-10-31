"""report.md of the pitch-level analysis (scripts/analysis/analyze_panel.py), and the small helpers the pitch-level
figures (analysis.pitch_figures) and the analysis notebook share: p-values as text and stars, which p a figure shows,
the models an ROC figure draws, the points of a calibration plot.

Nothing here reads the archived shark-level design (analysis/archive): importing `analyze_panel` or
`analysis.pitch_figures` loads no archive module (plan B task 9 split this module out of analysis.panel_report).
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.panel_explorer import REPO, ranking_stability_reading


def analyze_panel():
    """The pitch-level analysis script as a module (scripts/analysis/analyze_panel.py), loaded once."""
    if "analyze_panel" not in sys.modules:
        spec = importlib.util.spec_from_file_location("analyze_panel", REPO / "scripts" / "analysis" / "analyze_panel.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["analyze_panel"] = module
        spec.loader.exec_module(module)
    return sys.modules["analyze_panel"]



STAR_LEVELS = ((0.001, "***"), (0.01, "**"), (0.05, "*"))
STARS_NOTE = "* p < .05, ** p < .01, *** p < .001"
P_HEADER, Q_HEADER = "p", "q (FDR)"
PERM_LOSS_HEADER = "p, beats the offer rate (permutation, log loss)"
PERM_AUC_HEADER = "p, AUC (permutation)"
BOOT_SKILL_HEADER = "p, skill above 0 (bootstrap)"


def p_text(p) -> str:
    """A p- or q-value as text: "<0.001", three decimals below 0.1 ("0.012"), two above ("0.31"); "" when missing."""
    if p is None or p != p:
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def p_phrase(p) -> str:
    """ "p = 0.012" or "p < 0.001"; "" when missing."""
    text = p_text(p)
    return "" if not text else f"p {text.replace('<', '< ')}" if text.startswith("<") else f"p = {text}"


def stars(p) -> str:
    """Stars for a p-value (STARS_NOTE); "" when missing or 0.05 and above."""
    if p is None or p != p:
        return ""
    return next((mark for level, mark in STAR_LEVELS if p < level), "")


def model_p(row, metric: str = "log_loss") -> tuple[float, str]:
    """(p-value, kind) that goes with a model's skill (`metric` "log_loss") or AUC ("auc") in a figure: the
    permutation p when the permutation test was run, else the bootstrap p of the same quantity."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: getattr(row, k, d)
    perm = get("perm_p_log_loss" if metric == "log_loss" else "perm_p_auc", np.nan)
    if perm is not None and perm == perm:
        return float(perm), "permutation"
    boot = get("skill_p_value" if metric == "log_loss" else "auc_p_value", np.nan)
    return (float(boot), "bootstrap") if boot is not None and boot == boot else (np.nan, "")


def p_kind(result: dict) -> str:
    """Which p-value the figures of an evaluation show: "permutation" when the test was run, else "bootstrap"."""
    m = result.get("metrics")
    ran = m is not None and len(m) and "perm_p_log_loss" in m.columns and m.perm_p_log_loss.notna().any()
    return "permutation" if ran else "bootstrap"



ROC_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")


def roc_models(result: dict, models: list[str] | None = None) -> list[str]:
    """The models an ROC figure draws: the main models that have features (a base-rate model has no
    ranking to draw), at most as many as there are line styles; or `models` as given."""
    if result["status"] != "estimated":
        return []
    from analysis.panel_bottleneck import ONE_FAMILY

    m = result["metrics"].set_index("model")
    if models is None:          # the cumulative ladder; the one-family models (C3a, C4r, C5r) are in figure 10
        models = [name for name in m.index if m.role[name] == "main" and m.families[name] != "(base rate)"
                  and (name not in ONE_FAMILY or name == "C3q")]
    return list(models)[:len(ROC_COLOURS)]



def calibration_points(result: dict, models: list[str] | None = None, bins: int = 4) -> pd.DataFrame:
    """The points of a calibration plot: per model, the rows sorted by held-out prediction into `bins` equal
    groups, with each group's mean prediction (`p`), its share of offers (`y`) and its size (`n`). By
    default the main models (the sensitivity model is left out). Empty when nothing was estimated."""
    columns = ["model", "group", "p", "y", "n"]
    if result["status"] != "estimated":
        return pd.DataFrame(columns=columns)
    pred = result["predictions"]
    shown = models or [m for m in result["metrics"].model if result["metrics"].set_index("model").role[m] == "main"]
    parts = []
    for name in shown:
        p = pred[f"p_{name}"]
        groups = pd.qcut(p.rank(method="first"), q=min(bins, len(p)), labels=False)
        grouped = pred.assign(p=p, g=groups).groupby("g").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        parts.append(grouped.reset_index().rename(columns={"g": "group"}).assign(model=name)[columns])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)



# ---- report.md of the pitch-level analysis (scripts/analysis/analyze_panel.py) ------------------------

SPEC = REPO / "docs" / "plans" / "design docs" / "2026-10-19-pitch-level-refocus.md"
PLAN_V4 = REPO / "docs" / "plans" / "project_plan_v4.md"
# report section 2: what the ladder's AUC ranks (analysis.panel_bottleneck.BASE_RATE_ADJUSTED)
AUC_ADJUSTED_SENTENCE = ("AUCs (and their intervals, differences and permutation p-values) are computed on the "
                         "base-rate-adjusted held-out score logit(p) − logit(training base rate of the pitch's fold), "
                         "so they are not comparable with the AUCs of earlier (v3-dev8) analyses, which ranked the raw "
                         "probabilities; log loss, Brier score and skill read the probabilities.")
REPORT_HEADINGS = ("## 0. The pre-registered test", "## 1. Sample and outcomes", "## 2. The ladder", "## 3. Verdicts",
                   "## 4. Sensitivities", "## 5. H9", "## 6. H4 and H5", "## 7. Self-consistency", "## 8. Limits",
                   "## 9. Provenance")
H9_SKIPPED_TEXT = "H9 was skipped in this run (`--skip-h9`): the h9 tables hold their headers only."
H9_UNITS_TEXT = ("Units differ across the nested models: a rating is standardised over the answers of the model's "
                 "own population (all answers, those with a substance rating, those rated on all five), so the "
                 "same predictor's coefficient is per a slightly different SD in each model; compare signs and "
                 "intervals across models, not sizes.")
# report section 8, after the bullets of plan v4 section 11: the limits of the method that plan B's reviews named
METHOD_LIMITS = (
    "The leave-one-pitch-out AUC. The base-rate adjustment removes the shift of the training base rate against the "
    "held-out pitch, but a residual leave-one-out bias against weak models can remain; the bootstrap intervals of "
    "the AUC differences are not immune to it, the permutation p is (it repeats the same leave-one-out under "
    "shuffled outcomes).",
    "H9 runs about 120 coefficients of interest with no multiple-test correction; a direction claimed where the "
    "intervals for all and for hard questions agree is an exploratory pattern, not a finding.",
)
NON_DEVELOPMENT_TEXT = ("`non_development` leaves out the pilot videos the codebooks were first written on, named "
                        "by the selection file (spec 6.4 c). In the wider sense of plan v4 section 11 every analysed "
                        "video is a development video, since the instrument was shaped while looking at all of them; "
                        "this sensitivity removes the pilot videos, not the looking, and does not make the analysis "
                        "confirmatory.")
MEDIATION_TEXT = ("The pitch-level path (answer preparedness, then questioning effort, then the offer) is described, "
                  "not tested as a mediation: {n} pitches cannot carry an indirect-effect test.")


def _between(path: Path, first: str, then: str | None) -> list[str]:
    """The lines of `path` after the heading line starting with `first`, up to the next heading starting with `then`
    (or any `## ` heading when `then` is None); a missing heading is an error that names it."""
    lines = Path(path).read_text().splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith(first))
    except StopIteration:
        raise ValueError(f"{path}: no heading starting with {first!r}") from None
    for i in range(start + 1, len(lines)):
        if (then is not None and lines[i].startswith(then)) or (then is None and lines[i].startswith("## ")):
            return lines[start + 1:i]
    if then is not None:
        raise ValueError(f"{path}: no heading starting with {then!r} after {first!r}")
    return lines[start + 1:]


def section_six(path: Path | str = SPEC) -> str:
    """Section 6 of the refocus spec, verbatim: the text between its `## 6.` and `## 7.` headings (the headings and
    the closing rule `---` left out). Raises ValueError when either heading is missing."""
    text = "\n".join(_between(Path(path), "## 6.", "## 7.")).strip()
    if text.endswith("---"):
        text = text[:-3].rstrip()
    return text


def plan_limits(path: Path | str = PLAN_V4) -> list[str]:
    """The bullets of plan v4 section 11 ("What v4 does not fix"), read from the file each time, however many there
    are: a line starting with "- " opens a bullet, an indented line continues it (joined with one space)."""
    bullets: list[str] = []
    for line in _between(Path(path), "## 11.", None):
        if line.startswith("- "):
            bullets.append(line[2:].strip())
        elif line.startswith("  ") and line.strip() and bullets:
            bullets[-1] += " " + line.strip()
    if not bullets:
        raise ValueError(f"{path}: section 11 holds no bullet")
    return bullets


def git_commit(repo: Path | str = REPO) -> str:
    """The commit checked out in `repo` (`git rev-parse HEAD`), or "unknown" outside a repository."""
    import subprocess

    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = done.stdout.strip()
    return commit if done.returncode == 0 and commit else "unknown"


def _num(value, digits: int = 3, signed: bool = False) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _ci(low, high, digits: int = 3, signed: bool = False) -> str:
    a, b = _num(low, digits, signed), _num(high, digits, signed)
    return "—" if a == "—" and b == "—" else f"[{a}, {b}]"


def _with_ci(value, low, high, digits: int = 3, signed: bool = False) -> str:
    """A value and its interval, "—" once (not "— —") when the value and the interval are all missing."""
    shown, interval = _num(value, digits, signed), _ci(low, high, digits, signed)
    if shown == "—" and interval == "—":
        return "—"
    return f"{shown} {interval}"


def _md(rows: list[dict] | pd.DataFrame) -> str:
    """A Markdown table of `rows` (no dependency on tabulate); an empty table is said in words."""
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty:
        return "_(no rows)_"
    def cell(v) -> str:
        if isinstance(v, float) and not np.isfinite(v):
            return "—"
        if v is None or v is pd.NA or v is pd.NaT:          # a nullable integer's missing value prints <NA>
            return "—"
        return str(v).replace("|", "\\|").replace("\n", " ")
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _report_sample(a) -> list[str]:
    ap = analyze_panel()
    counts = a.outcome.any_offer
    coding = ap._counts(a.data.videos.source)
    lines = [f"{len(a.data.videos)} pitches ({ap.coding_text(coding)}); label: {a.label}.",
             f"Outcome `any_offer` (any investor made or joined an offer): {int((counts == 1).sum())} with an offer, "
             f"{int((counts == 0).sum())} without, {int(counts.isna().sum())} unknown.",
             "Unit: one row per pitch, cut at the first decision of any kind by any investor; leave-one-pitch-out, so "
             "every pitch is one independent observation.",
             f"Stage A ranking (the rank score C2 reads): {a.stage_a_ranking}."]
    overrides = int((a.outcome.any_offer_source != "record").sum())
    if overrides:
        lines.append(f"{overrides} pitch outcomes are a reviewer's verdict rather than the record's.")
    still_open = int((a.data.videos.source == ap.SOURCE_OPEN).sum())
    if still_open:
        lines.append(f"{still_open} reviewed records still have something open; they are read, not excluded.")
    return lines


def _ladder_tables(result: dict) -> tuple[str, str]:
    metrics, deltas = result["metrics"], result["deltas"]
    if result["status"] != "estimated" or not len(metrics):
        return f"Not estimable: {result['status']}.", "No comparisons."
    rows = [{"Model": r.model, "Families": r.families, "Skill %": _num(r.skill_pct, 1),
             "Skill 95%": _ci(r.skill_low, r.skill_high, 1), "AUC": _num(r.auc), "AUC 95%": _ci(r.auc_low, r.auc_high),
             "AUC meaningful": "yes" if bool(r.auc_meaningful) else "no", "Log loss": _num(r.log_loss),
             "Penalty": r.penalty, "Permutation p (log loss)": _num(getattr(r, "perm_p_log_loss", np.nan)),
             "Permutation p (AUC)": _num(getattr(r, "perm_p_auc", np.nan))}
            for r in metrics.itertuples()]
    pairs = [{"Comparison": r.comparison, "Metric": r.metric, "Difference": _num(r.delta, 3, True),
              "95% interval": _ci(r.ci_low, r.ci_high, 3, True), "Bootstrap p": _num(r.p_value)}
             for r in deltas.itertuples()]
    return _md(rows), _md(pairs)


BASELINE_GAINS_NOTE = ("Each family's gain over the baseline C2p (the venture, the ask, exposure, delivery and the "
                       "prepared presentation), later minus earlier, with its 95% interval and bootstrap p: once "
                       "when the family enters alone (the one-family model, amendment 6.9 of the spec) and once "
                       "when it enters the cumulative ladder, on top of the families before it. Both are paired "
                       "differences of the table above; the single steps of the cumulative ladder are there too.")


def _baseline_gain_table(result: dict) -> str:
    """Report section 2: `baseline_gains` as a Markdown table (one row per family and metric)."""
    from analysis.panel_bottleneck import baseline_gains

    table = baseline_gains(result.get("deltas"), result.get("metrics"))
    if not len(table):
        return "No baseline comparison."
    rows = [{"Family": r.family, "Metric": r.metric, "Alone": r.alone_model,
             "Gain over C2p": _with_ci(r.alone_delta, r.alone_low, r.alone_high, 3, True),
             "Bootstrap p": _num(r.alone_p), "Cumulative": r.cumulative_model,
             "Gain over C2p (cumulative)": _with_ci(r.cumulative_delta, r.cumulative_low, r.cumulative_high, 3, True),
             "Bootstrap p (cumulative)": _num(r.cumulative_p)} for r in table.itertuples()]
    return _md(rows)


GEMINI_ONLY = "Gemini only, not human-coded"
AUC_MEANINGFUL_NOTE = ("\"AUC meaningful\" is a diagnostic only (ruling I3 of plan B's final review): \"no\" marks a "
                       "model whose held-out predictions stay within about 3 points of the offer rate; it withholds "
                       "nothing, and that model's AUC counts in the verdicts. The AUC route is withheld only when a "
                       "model's AUC interval lies wholly below 0.5.")
# the context ratings: PRES-CTX of the prepared presentation (C2p), PREP-CTX of the answers (C3 and H9's full model)
CONTEXT_COLUMNS = ("pres_ctx", "a_ctx_mean", "ctx")
CONTEXT_SENTENCE = ("The context ratings (PRES-CTX in C2p, `pres_ctx`; PREP-CTX in C3 and H9's full model, "
                    f"`a_ctx_mean`, `ctx`) are {GEMINI_ONLY}: they are not on the human coders' sheet.")


def labelled(name: str) -> str:
    """A column or predictor name, with the Gemini-only label when it is a context rating."""
    return f"{name} ({GEMINI_ONLY})" if name in CONTEXT_COLUMNS else name


def _design_lines(result: dict) -> list[str]:
    """Report section 2: each ladder model's design columns (ruling I4 of plan B's final review) and what is never
    fitted."""
    from analysis.panel_bottleneck import LADDER_NOT_FITTED_NOTE

    columns = result.get("design_columns") or {}
    rows = [{"Model": model, "Columns": len(names), "Design columns": ", ".join(map(labelled, names)) or "(base rate)"}
            for model, names in columns.items()]
    return ["### Design columns", "",
            "The columns each model fits, on the pitches with a known outcome; a column constant in a training fold "
            f"is dropped in that fold. {LADDER_NOT_FITTED_NOTE[0].upper()}{LADDER_NOT_FITTED_NOTE[1:]}. "
            + CONTEXT_SENTENCE, "",
            _md(rows), ""]


def _stage_a_sentence(a) -> str:
    from analysis.panel_bottleneck import PRIMARY_STAGE_A, STAGE_A_FALLBACK_NOTE

    ap = analyze_panel()
    if a.stage_a_ranking == ap.RANKING_ABSENT:
        return (f"Stage A: no pitch has the rank score of `{PRIMARY_STAGE_A}`, so C2 reads its rating instead "
                f"({STAGE_A_FALLBACK_NOTE}).")
    if a.stage_a_ranking == ap.RANKING_COMPLETE:
        return f"Stage A: C2 reads the rank score of `{PRIMARY_STAGE_A}`, which every pitch has (complete)."
    found = re.fullmatch(r"partial: (\d+) of (\d+)", a.stage_a_ranking)
    missing = int(found.group(2)) - int(found.group(1)) if found else 0
    head = (f"Stage A: the ranking is {a.stage_a_ranking} pitches. The design drops the rank-score column from a "
            "training fold in which any pitch lacks it")
    tail = "; the rating fallback applies only when no pitch has a rank score."
    if missing == 1:
        return (head + ". One pitch lacks it, so the rank column is used in one fold only (the fold that holds that "
                "pitch out); in every other fold C2 is C1, and C2 minus C1 is barely a test of Stage A" + tail)
    return (head + f". {missing} pitches lack it, so every training fold lacks it: C2 equals C1 in this run and C2 "
            "minus C1 is no test of Stage A" + tail)


def route_sentence(r) -> str:
    """Which route of the rule passed for verdict row `r` (analysis.verdicts.RULES), in words."""
    from analysis.verdicts import AUC_TOLERANCE, AUC_USED, RULES

    rule = RULES[r.hypothesis]
    auc = (r.auc_route == AUC_USED and r.auc_delta == r.auc_delta and r.auc_low == r.auc_low and r.auc_low > 0
           and r.auc_delta >= rule["auc_min"] - AUC_TOLERANCE)
    skill = rule["skill_alternative"] and r.skill_low == r.skill_low and r.skill_low > 0
    skill_words = ("the skill interval includes zero" if r.skill_low == r.skill_low and r.skill_low <= 0 <= r.skill_high
                   else "the skill interval excludes zero")
    if auc and skill:
        return "Passes on both routes: the AUC gain and the skill gain."
    if auc:
        return (f"Passes on the AUC route only ({skill_words}"
                + ("; the rule has no skill route for this hypothesis" if not rule["skill_alternative"] else "") + ").")
    if skill:
        return "Passes on the skill route only; the AUC route does not pass."
    return "Neither route passes."


def baseline_sentence(r, metrics: pd.DataFrame | None) -> str:
    """When the earlier model of the pair has an AUC below 0.5 (the residual leave-one-out bias against weak models,
    section 8), the gain measured from 0.5 instead, computed from the metrics; "" otherwise."""
    from analysis.verdicts import RULES

    if metrics is None or not len(metrics) or "auc" not in metrics:
        return ""
    m = metrics.set_index("model")
    earlier, later = RULES[r.hypothesis]["pair"]
    if earlier not in m.index or later not in m.index or not m.auc[earlier] < 0.5:
        return ""
    order = list(m.index)
    weak = [f"{name} {m.auc[name]:.3f}" for name in order[:order.index(earlier) + 1]
            if m.families[name] != "(base rate)" and m.auc[name] < 0.5]
    return (f"The earlier models sit below chance under leave-one-pitch-out ({', '.join(weak)}), the residual bias "
            f"against weak models (section 8), so the AUC gain overstates the step: measured from 0.5, {later}'s AUC "
            f"of {m.auc[later]:.3f} is {m.auc[later] - 0.5:+.3f} above chance.")


def _verdict_lines(table: pd.DataFrame, metrics: pd.DataFrame | None = None) -> list[str]:
    lines = []
    for r in table.itertuples():
        notes = " ".join(filter(None, [route_sentence(r), baseline_sentence(r, metrics)]))
        lines += [f"### {r.hypothesis}: {r.comparison}", "",
                  f"**Verdict: {r.verdict}.** AUC difference {_with_ci(r.auc_delta, r.auc_low, r.auc_high, 3, True)}; "
                  f"skill difference {_with_ci(r.skill_delta, r.skill_low, r.skill_high, 2, True)} points. "
                  f"AUC route: {r.auc_route}.", "", notes, "", r.power, ""]
    return lines


def _sensitivity_rows(a) -> list[dict]:
    from analysis.verdicts import RULES

    settings = [("main", "the pre-registered analysis", a.result, a.verdicts)]
    settings += [(name, a.sensitivities[name]["sensitivity_note"], a.sensitivities[name], a.sensitivity_verdicts[name])
                 for name in a.sensitivities]
    rows = []
    for name, note, result, table in settings:
        by = table.set_index("hypothesis")
        row = {"Setting": name, "What changes": note, "Pitches": result["n_groups"]}
        status = str(result.get("status", ""))
        for hypothesis in RULES:
            r = by.loc[hypothesis]
            if status.startswith("not run"):          # final review I2: said in every cell, no number
                row[f"{hypothesis} {r.comparison}: AUC"] = row[f"{hypothesis}: skill"] = "—"
                row[f"{hypothesis} verdict"] = status
                continue
            route = "" if r.auc_route == "used" else f" ({r.auc_route})"
            row[f"{hypothesis} {r.comparison}: AUC"] = _with_ci(r.auc_delta, r.auc_low, r.auc_high, 3, True)
            row[f"{hypothesis}: skill"] = _with_ci(r.skill_delta, r.skill_low, r.skill_high, 2, True)
            row[f"{hypothesis} verdict"] = f"{r.verdict}{route}"
        rows.append(row)
    return rows


def _h9_lines(a) -> list[str]:
    from analysis.answer_level import COVARIATES, MODELS, OUTCOME_COVARIATES, POPULATIONS, SENSITIVITY_OUTCOMES

    if a.h9 is None:
        return [H9_SKIPPED_TEXT, "", MEDIATION_TEXT.format(n=len(a.data.videos))]
    every, hard = a.h9["all"], a.h9["hard"]
    lines = ["Coefficients compare answers within a pitch (conditional logit for the binary next moves, pitch fixed "
             "effects for the counts and the severity); per SD of a rating, per 0 to 1 of a binary predictor. "
             "No decision rule: a direction is claimed only where the intervals for all and for hard questions "
             "both exclude zero on the same side. The outcomes are read on turn order. Every model of "
             f"{' and '.join(OUTCOME_COVARIATES)} also enters {', '.join(sorted(COVARIATES))} (the answered "
             "question's frame, 1 promotion, 0 prevention), a covariate with no direction; answers to a question "
             "with neither frame are not in those models. Where the covariate separates the outcome or cannot be "
             "estimated in a population, the model is fitted without it and the Direction column says why. "
             "A predictor whose log-odds on the data lies beyond the bound the bootstrap draws obey is reported "
             "separated and the model is refitted without it.", "",
             H9_UNITS_TEXT, "", *_repair_lines(a)]
    for model, predictors in MODELS.items():
        mine, mine_hard = every[every.model == model], hard[hard.model == model]
        n_all = int(mine.n_answers.max()) if len(mine) else 0
        n_hard = int(mine_hard.n_answers.max()) if len(mine_hard) else 0
        informative = int(mine.n_pitches_informative.max()) if len(mine) else 0
        withheld = sorted({f"{name} questions / {r.outcome} ({r.estimable})"
                           for name, frame in (("all", mine), ("hard", mine_hard))
                           for r in frame.itertuples() if getattr(r, "estimable", "yes") != "yes"})
        lines.append(f"- **{model}** ({', '.join(predictors)}): population: {POPULATIONS[model]}; {n_all} answers "
                     f"(all questions, the largest of its outcomes), {n_hard} on hard questions; up to {informative} "
                     "informative pitches." + (f" No interval, so no direction: {'; '.join(withheld)}." if withheld else ""))
    claimed = a.h9_summary[a.h9_summary.hard_and_all_agree.astype(bool)] if len(a.h9_summary) else a.h9_summary
    labels = {name: labelled(name) for name in CONTEXT_COLUMNS}
    unclaimed = h9_unclaimed_lines(a, labels)
    lines += ["", H9_EXPLORATORY_TEXT, "", CONTEXT_SENTENCE, "",
              f"Directions claimed ({len(claimed)} of {len(a.h9_summary)} predictor rows):"
              + ("" if len(claimed) else " none."), *[f"- {line}." for line in h9_claim_lines(a, labels)], "",
              "Not claimed: the all-questions interval excludes zero but the hard-questions interval does not "
              f"(or there is none) in {len(unclaimed)} rows" + (":" if unclaimed else "."),
              *[f"- {line}." for line in unclaimed], ""]
    keys = ["model", "outcome", "predictor"]
    both = every.merge(hard, on=keys, how="left", suffixes=("_all", "_hard")).merge(
        a.h9_summary[keys + ["direction"]], on=keys, how="left")
    def estimate(coef, low, high, separated, estimable) -> str:
        flagged = separated == separated and bool(separated)          # NaN (no hard row) is not separated
        if flagged:
            return "separated"
        why = "" if not isinstance(estimable, str) or estimable == "yes" else f" ({estimable})"
        return f"{_with_ci(coef, low, high, 2, True)}{why}"
    rows = [{"Model": r.model, "Outcome": r.outcome, "Predictor": labelled(r.predictor),
             "All questions": estimate(r.coef_all, r.low_all, r.high_all, r.separated_all, r.estimable_all),
             "n (all)": int(r.n_answers_all),
             "Hard questions": estimate(r.coef_hard, r.low_hard, r.high_hard, r.separated_hard, r.estimable_hard),
             "n (hard)": "—" if r.n_answers_hard != r.n_answers_hard else int(r.n_answers_hard),
             "Direction": _covariate_cell(r) if r.predictor in COVARIATES else r.direction}
            for r in both.itertuples()]
    pointer = getattr(a, "h9_sensitivity", None)
    pointer = [] if pointer is None else [
        f"The exchange-based outcomes ({', '.join(SENSITIVITY_OUTCOMES)}) were computed as an unreported "
        f"sensitivity with {a.h9_sensitivity_draws} draws and are in h9_exchange_sensitivity.csv.", ""]
    return lines + [_md(rows), "", *pointer, MEDIATION_TEXT.format(n=len(a.data.videos))]


def _repair_lines(a) -> list[str]:
    """M4 of plan B's final review: how many answers the responsiveness model's repair predictor rests on (nothing
    when the analysis holds no turn table)."""
    from analysis.answer_level import answer_table
    from analysis.panel_dataset import pitch_cut

    if getattr(a.data, "turns", None) is None:
        return []
    table = answer_table(a.data, pitch_cut(a.data))
    repaired = table[table.repair_before_next.astype(bool)]
    hard = int(repaired.hard_question.astype(bool).sum())
    share = len(repaired) / len(table) if len(table) else float("nan")
    return [f"Repairs are rare: a founder repair turn comes between the answer and the panel's next turn for "
            f"{len(repaired)} of {len(table)} answers ({share:.1%}), in {repaired.video_id.nunique()} pitches, "
            f"{hard} of them after a hard question. The responsiveness model's `repair_before_next` rests on these "
            "and, under hard questions, usually has no interval.", ""]


# H9's hypothesis (plan v4 section 7): after a more prepared answer the panel accepts or switches frame rather than
# challenging or pressing on in the same frame. The sign it expects of a rating that is higher for a more prepared
# answer, per outcome; the hypothesis states none for a further question of any frame, or for admitted uncertainty
# and repairs (the responsiveness model).
H9_PREPARED = ("evid", "subs", "coh", "ctx")
H9_EXPECTED = {"next_accept": (1, "more accepts", "accepts less often"),
               "next_challenge": (-1, "fewer challenges", "challenges more often"),
               "next_question_same_frame": (-1, "a switch of frame", "keeps the frame more often"),
               "next_question_sev": (-1, "a less severe next question", "asks a more severe next question"),
               "further_questions_same_frame": (-1, "fewer consecutive questions in the same frame (less pressing on)",
                                                "asks more consecutive questions in the same frame: it presses on, "
                                                "more work for the panel, not less")}
H9_EXPLORATORY_TEXT = ("H9 is an exploratory pattern, not a finding: about 120 coefficients of interest are estimated "
                       "with no multiple-test correction, and requiring the intervals for all and for hard questions "
                       "to agree reduces false positives but does not remove them.")


def h9_claim_lines(a, words: dict | None = None) -> list[str]:
    """One line per direction H9 claims (h9_summary), saying whether it runs with or against the hypothesis
    (H9_EXPECTED); `words` names the predictors (raw names by default)."""
    words = words or {}
    s = a.h9_summary
    lines = []
    for r in s[s.hard_and_all_agree.astype(bool)].itertuples():
        head = f"{r.model} / {r.outcome} / {words.get(r.predictor, r.predictor)}: {r.direction}"
        expected = H9_EXPECTED.get(r.outcome)
        if r.predictor not in H9_PREPARED or expected is None:
            lines.append(f"{head} (the hypothesis states no direction for it)")
            continue
        sign = 1 if r.direction == "positive" else -1
        lines.append(f"{head}, in the hypothesised direction ({expected[1]} after a more prepared answer)"
                     if sign == expected[0] else
                     f"{head}, against the hypothesis, which expects {expected[1]} after a more prepared answer: "
                     f"here the panel {expected[2]}")
    return lines


def h9_unclaimed_lines(a, words: dict | None = None) -> list[str]:
    """The predictor rows whose all-questions interval excludes zero while the hard-questions one does not (or is
    missing): patterns H9's rule does not claim, listed so a reader does not take them for findings."""
    from analysis.answer_level import COVARIATES, ESTIMABLE

    words = words or {}
    keys = ["model", "outcome", "predictor"]
    both = a.h9["all"].merge(a.h9["hard"], on=keys, how="left", suffixes=("_all", "_hard")).merge(
        a.h9_summary[keys + ["hard_and_all_agree"]], on=keys, how="left")
    lines = []
    for r in both.itertuples():
        agree = r.hard_and_all_agree
        if r.predictor in COVARIATES or (agree == agree and bool(agree)):
            continue
        if r.estimable_all != ESTIMABLE or bool(r.separated_all) or not (r.low_all > 0 or r.high_all < 0):
            continue
        hard = ("no interval" if r.low_hard != r.low_hard else f"{r.coef_hard:+.2f} [{r.low_hard:+.2f}, {r.high_hard:+.2f}]")
        lines.append(f"{r.model} / {r.outcome} / {words.get(r.predictor, r.predictor)}: all questions {r.coef_all:+.2f} "
                     f"[{r.low_all:+.2f}, {r.high_all:+.2f}]; hard questions {hard}")
    return lines


def _covariate_cell(r) -> str:
    """The Direction cell of a covariate row of the H9 table (merged all/hard row `r`): never a direction; when the
    covariate was not entered in a population (separated, the model not estimable, the fit failed, or no
    within-pitch variation, so no coefficient), why, per population when the two differ."""
    reasons = {}
    for name, suffix in (("all", "_all"), ("hard", "_hard")):
        n = getattr(r, f"n_answers{suffix}", np.nan)
        if n != n:                                          # no row for this population
            continue
        separated = getattr(r, f"separated{suffix}", False)
        estimable = getattr(r, f"estimable{suffix}", "yes")
        failed = getattr(r, f"main_failed{suffix}", False)
        coef = getattr(r, f"coef{suffix}", np.nan)
        if separated == separated and bool(separated):
            reasons[name] = "not entered: separated"
        elif isinstance(estimable, str) and estimable != "yes" and coef == coef:
            # the covariate was fitted; only its interval is missing (e.g. too few informative pitches)
            reasons[name] = f"entered, no interval: {estimable}"
        elif isinstance(estimable, str) and estimable != "yes":
            reasons[name] = f"not entered: {estimable}"
        elif failed == failed and bool(failed):
            reasons[name] = "not entered: the main fit failed"
        elif coef != coef:
            reasons[name] = "not entered: no within-pitch variation"
    if not reasons:
        return "covariate, no direction"
    if len(set(reasons.values())) == 1 and len(reasons) == 2:
        return f"covariate, {next(iter(reasons.values()))}"
    return "covariate, " + "; ".join(f"{why} ({name} questions)" for name, why in reasons.items())


def _reviewed_lines(a) -> list[str]:
    """When reviewed records exist: `both_passes` averages each with its own raw replicate pass (Task 5 of plan B)."""
    from analysis.panel_dataset import REVIEWED_SOURCES

    reviewed = int(a.data.videos.source.isin(REVIEWED_SOURCES).sum())
    if not reviewed:
        return []
    videos, verb = ("1 video", "averages") if reviewed == 1 else (f"{reviewed} videos", "average")
    return [f"In `both_passes`, the {videos} with a reviewed record {verb} the reviewed record with the raw "
            "replicate pass of the same video, so a human correction counts half there.", ""]


def _limit_bullets(a) -> list[str]:
    """The bullets of plan v4 section 11, with the number of pitches without an offer set to this run's own count
    (M6 of plan B's final review): plan v4 wrote the v3-dev8 count."""
    without = int((a.outcome.any_offer == 0).sum())

    def own(found):
        stated = int(found.group(1))
        if stated == without:
            return found.group(0)
        return f"{without} pitches without an offer in this run (plan v4 states {stated}, the v3-dev8 count)"
    return [re.sub(r"(\d+) pitches without an offer", own, bullet) for bullet in plan_limits()]


def _h4_h5_lines(a) -> list[str]:
    shares = a.h4_shares
    rows = [{"Questions": r.questions, "Panel's first reaction": r.move,
             "With support": f"{_with_ci(r.share_yes, r.yes_low, r.yes_high, 2)} (n {r.n_yes})",
             "Without support": f"{_with_ci(r.share_no, r.no_low, r.no_high, 2)} (n {r.n_no})",
             "Difference": _with_ci(r.difference, r.diff_low, r.diff_high, 2, True)}
            for r in shares.itertuples()]
    h5 = a.h5
    framed = h5[~h5.question_frame.isin(("no_frame", "no_linked_question")) & (h5.answer_frame != "no_frame")]
    matched = int(framed[framed.matches].n.sum())
    total = int(framed.n.sum())
    match = (f"{matched} of {total} answers with a framed question and a framed answer keep the question's frame "
             f"({matched / total:.0%}); {total - matched} switch." if total else
             "No answer has both a framed question and a frame of its own.")
    return ["### H4: what the panel does after an answer", "",
            "Whole encounter, no history cut; descriptive, never a predictor. Shares of answers followed by each "
            "first reaction of the panel, with and without checkable support, with 95% intervals from resampling "
            "pitches.", "", _md(rows), "", "### H5: frame matching", "",
            f"Descriptive, whole encounter ({int(h5.n.sum())} answers). {match} Frames derive from concepts, so "
            "matching is partly built in; deal-terms and other questions carry no frame (h5_frame_match.csv).", ""]


def _consistency_lines(a) -> list[str]:
    table = a.self_consistency.get("self_consistency.csv", pd.DataFrame())
    n = int(table.n_videos.iloc[0]) if len(table) and "n_videos" in table else 0
    shown = table.drop(columns=[c for c in ("label", "n_videos") if c in table.columns]) if len(table) else table
    return [f"{analyze_panel().SELF_CONSISTENCY_LABEL}: two identical passes of the model compared, over {n} videos. "
            "Stable under re-sampling is not the same as right.", "", _md(shown.round(3) if len(shown) else shown), "",
            "Stage A ranking stability: " + ranking_stability_reading(a.ranking_agreement).split("\n")[0], ""]


def report_markdown(a) -> str:
    """report.md of a pitch-level analysis (`analyze_panel.compute`): the pre-registered test quoted from the spec,
    the sample, the ladder, the verdicts with their AUC route and power sentence, the sensitivities, H9, H4 and H5,
    self-consistency, the limits of plan v4 section 11 and the provenance (REPORT_HEADINGS)."""
    ap = analyze_panel()
    stated = ap.provenance(a)
    metrics, deltas = _ladder_tables(a.result)
    single = a.single_pass_videos
    lines = [f"# Pitch-level analysis of {Path(a.run_root).name}", "",
             "Written by `scripts/analysis/analyze_panel.py` from the tables beside this file; every number here is in "
             "one of them.", "",
             REPORT_HEADINGS[0], "",
             f"Quoted verbatim from `{stated['spec']['path']}`, section 6 (the file's sha256 {stated['spec']['sha256']}). "
             "In the code C2′ is `C2p`.", "",
             section_six(), "",
             REPORT_HEADINGS[1], "", *[f"- {line}" for line in _report_sample(a)], "",
             REPORT_HEADINGS[2], "", _stage_a_sentence(a), "",
             f"Penalty: {stated['penalty']}; {a.n_boot} bootstrap draws; {a.n_perm} permutations "
             "(permutation p: does the model beat the offer rate, in log loss and in AUC).", "",
             AUC_ADJUSTED_SENTENCE, "",
             "### Metrics", "", metrics, "", AUC_MEANINGFUL_NOTE, "", "### Paired differences (later minus earlier)", "", deltas, "",
             "### Gains over the baseline C2p", "", BASELINE_GAINS_NOTE, "", _baseline_gain_table(a.result), "",
             *_design_lines(a.result),
             REPORT_HEADINGS[3], "", *_verdict_lines(a.verdicts, a.result["metrics"]),
             REPORT_HEADINGS[4], "",
             "Each row is a full evaluation of the ladder with the same draws and seed; the verdicts apply the same "
             "rules. The pre-registered verdicts are those of `main`.", "", _md(_sensitivity_rows(a)), "",
             (f"Read from one pass only in `both_passes`: {', '.join(f'{v} ({why})' for v, why in single.items())}."
              if single else "Every video has both passes for `both_passes`."), "",
             *_reviewed_lines(a), NON_DEVELOPMENT_TEXT, "",
             REPORT_HEADINGS[5], "", *_h9_lines(a), "",
             REPORT_HEADINGS[6], "", *_h4_h5_lines(a),
             REPORT_HEADINGS[7], "", *_consistency_lines(a),
             REPORT_HEADINGS[8], "",
             f"Read from `{stated['limits']['path']}`, section 11 (the file's sha256 {stated['limits']['sha256']}); the "
             "number of pitches without an offer is this run's.", "",
             *[f"- {bullet}" for bullet in _limit_bullets(a)], "",
             "Limits of the method:", "", *[f"- {bullet}" for bullet in METHOD_LIMITS], "",
             REPORT_HEADINGS[9], "",
             f"- git commit: {stated['git_commit']}",
             f"- release ids: {', '.join(stated['release_ids']) or 'unknown'}",
             f"- seed: {a.seed}", f"- n_boot: {a.n_boot}", f"- n_perm: {a.n_perm}",
             f"- workers: {a.workers} (processes for H9's bootstrap draws, by default every core but one; the "
             "draws are made before any fit, so no number depends on it)",
             f"- single_draws: {a.single_draws}", f"- penalty: {stated['penalty']}",
             "- passes: main reads the best record of each video; `both_passes` averages both model passes",
             f"- H9: {stated['h9']}", f"- with_shark_level: {a.with_shark_level}",
             f"- spec: {stated['spec']['path']} (sha256 of the whole file {stated['spec']['sha256']})",
             f"- limits: {stated['limits']['path']} (sha256 of the whole file {stated['limits']['sha256']})",
             f"- run: {Path(a.run_root).resolve()}; videos: {len(a.data.videos)}"
             + (" (a selection)" if a.video_ids else " (every video with a result)"),
             "- figures: listed in manifest.json (`figures`, `figures_skipped`)", *_shark_level_lines(a), ""]
    return "\n".join(lines)


def _shark_level_lines(a) -> list[str]:
    """One provenance line for the archived shark-level run (F1), when it was run."""
    if a.shark_level is None:
        return []
    f1 = a.shark_level["f1"]
    first = f1.iloc[0] if len(f1) else None
    alone = (f"; identity alone ranks the sharks of a pitch at {_num(first.value)} {_ci(first.ci_low, first.ci_high)} "
             "within-pitch AUC" if first is not None else "")
    return [f"- shark level (archived design, for F1; reported, not tested): tables in shark_level/{alone}"]
