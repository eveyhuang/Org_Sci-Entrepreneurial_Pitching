"""Full-record descriptive views for the demo, using one analysis pass per episode."""
from __future__ import annotations

import base64
from html import escape
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis import panel_explorer as px
from analysis.panel_bottleneck import LADDER, ladder_columns, stage_a_fallback
from analysis.panel_features import FEATURES
from textwrap import fill


def episode_summary(data):
    """Episode-level spread, including episodes with zero turns of a given kind."""
    ids = data.videos.video_id
    values = {
        "Video duration (minutes)": data.videos.set_index("video_id").duration_s / 60,
        "Turns per episode": data.turns.groupby("video_id").size(),
        "Questions per episode": data.turns[data.turns.function == "question"].groupby("video_id").size(),
        "Answers per episode": data.turns[data.turns.function == "answer"].groupby("video_id").size(),
        "Decisions per episode": data.decisions.groupby("video_id").size(),
        "Investment asked (USD; VC-ASK)": data.videos.set_index("video_id").ask_amount_usd,
        "Equity offered (%; VC-ASK)": data.videos.set_index("video_id").ask_equity_pct,
    }
    rows = []
    for label, series in values.items():
        series = series.reindex(ids, fill_value=0 if "per episode" in label else np.nan)
        known = series.dropna()
        rows.append({"Measure": label, "Episodes with data": len(known), "Missing": int(series.isna().sum()),
                     "Mean": known.mean(), "SD": known.std(), "Minimum": known.min(),
                     "25th percentile": known.quantile(.25), "Median": known.median(),
                     "75th percentile": known.quantile(.75), "Maximum": known.max()})
    return pd.DataFrame(rows)


def overview_figure(counts, features):
    """Outcome frequencies are episodes; turn frequencies are pooled annotated turns."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.7]})
    outcomes = features.any_offer
    labels = ["With an offer", "Without an offer", "Unknown"]
    numbers = [(outcomes == 1).sum(), (outcomes == 0).sum(), outcomes.isna().sum()]
    if numbers[-1] == 0:
        labels, numbers = labels[:-1], numbers[:-1]
    rows = counts[(counts.code == "TURN-FUNC") & (counts.kind == "value")].sort_values("count", ascending=False)
    for ax, names, ns, total, colors, title in [
        (axes[0], labels, numbers, len(outcomes), ["#007bac", "#d97706", "#999999"], "Episode outcomes"),
        (axes[1], rows.value.str.replace("_", " "), rows["count"], int(rows.n.iloc[0]),
         [px.TURN_COLOURS.get(v, "#718096") for v in rows.value], "What happens during the episodes"),
    ]:
        ax.barh(list(names), np.array(ns) / total * 100, color=colors[:len(ns)], height=.65)
        for y, n in enumerate(ns):
            ax.text(n / total * 100 + 1, y, f"{n:,} ({n / total:.1%})", va="center", fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, max(ns) / total * 100 * 1.4)
        ax.set_title(f"{title}\nn = {total:,} {'episodes' if ax is axes[0] else 'turns'}", loc="left", fontsize=12)
        ax.set_xlabel("Share of episodes (%)" if ax is axes[0] else "Share of turns (%)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=3)
    return fig


def ratings_figure(counts):
    """Ordinal distributions, with missing ratings retained in each denominator."""
    groups = [("Preparedness and delivery of the pitch", ["PRES-SUBS", "PRES-COH", "PRES-CTX", "pitch_positivity"]),
              ("The sharks’ questions", ["PR-SEV"]),
              ("The entrepreneurs’ answers", ["PREP-SUBS", "PREP-COH", "PREP-CTX"])]
    colors = ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [4, 1, 3]})
    for ax, (title, codes) in zip(axes, groups):
        for y, code in enumerate(codes):
            part = counts[counts.code == code]
            n = int(part.n.iloc[0])
            left = 0
            for value in range(1, 6):
                width = part.loc[(part.kind == "value") & (part.value == str(value)), "share"].sum() * 100
                ax.barh(y, width, left=left, color=colors[value-1], edgecolor="white", height=.65)
                if width >= 6:
                    ax.text(left + width/2, y, f"{width:.0f}%", ha="center", va="center",
                            color="white" if value >= 4 else "#17202a", fontsize=9)
                left += width
            missing = part.loc[part.kind != "value", "share"].sum() * 100
            ax.barh(y, missing, left=left, color="#bbbbbb", hatch="///", height=.65)
            if missing >= 6:
                ax.text(left + missing/2, y, f"{missing:.0f}%", ha="center", va="center",
                        fontsize=9, bbox={"facecolor": "#dddddd", "edgecolor": "none", "pad": 1})
            ax.text(102, y, f"n = {n:,}", va="center", fontsize=9)
        labels = [counts[counts.code == c].name.iloc[0] + (" (1–3)" if c == "PR-SEV" else "") for c in codes]
        ax.set_yticks(range(len(codes)), labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Share of recorded ratings (%); all scales 1–5 except question severity (1–3)")
    fig.legend(handles=[Patch(color=c, label=str(i)) for i, c in enumerate(colors, 1)] +
               [Patch(facecolor="#bbbbbb", hatch="///", label="Missing")],
               loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, .055, .96, 1), h_pad=1.5)
    return fig


# Match the five rows/families in Figure 2, not the data types in the codebook.
MODEL_GROUPS = (
    ("C2p", "Preparedness and delivery of the pitch + control variables",
     ("PRES-SUBS", "PRES-COH", "PRES-CTX", "PRES-EVID", "pitch_positivity",
      "VC-INDUSTRY", "VC-FORM", "VC-CUSTOMER", "VC-GENDER")),
    ("C3q", "The sharks’ questions", ("PR-CONCEPT", "PR-SEV")),
    ("C3a", "The entrepreneurs’ answers", ("RS-RESP", "RS-CONCEPT", "RS-EVID", "RS-UNCERT",
                                           "PREP-SUBS", "PREP-COH", "PREP-CTX")),
    ("C4r", "The sharks’ reactions to the answers", ()),
    ("C5r", "Rapport and repair", ()),
)


def model_family_columns(features):
    """Use the fitted Figure 2 specification, excluding repeated baseline columns."""
    models, _ = stage_a_fallback(LADDER, features)
    baseline = ladder_columns(features, models["C2p"])
    return {model: [c for c in ladder_columns(features, models[model])
                    if (model == "C2p" or c not in baseline) and not c.endswith("_missing")]
            for model, _, _ in MODEL_GROUPS}


def observed_feature(features, code):
    values = features[code].copy()
    flag = f"{code}_missing"
    if flag in features:
        values = values.mask(features[flag].eq(1))
    return values


def model_family_figure(features, columns):
    """One distribution per fitted concept, one observation per episode."""
    ncols = 3
    nrows = (len(columns) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.9 * nrows), squeeze=False)
    for ax, code in zip(axes.flat, columns):
        values = observed_feature(features, code)
        known = values.dropna()
        if not pd.api.types.is_numeric_dtype(values):
            counts = known.value_counts().sort_values()
            ax.barh(counts.index.str.replace("_", " "), counts.values, color="#007bac")
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("Episodes", fontsize=8)
        else:
            unique = known.nunique()
            bins = np.arange(known.min() - .5, known.max() + 1.5) if unique and unique <= 10 and (known % 1 == 0).all() else 10
            ax.hist(known, bins=bins, color="#007bac", edgecolor="white")
            ax.set_ylabel("Episodes", fontsize=8)
        ax.set_title(fill(FEATURES[code]["label"], 38) + f"\nn = {len(known)}; missing = {values.isna().sum()}", fontsize=9, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout(h_pad=2, w_pad=2)
    return fig


def family_statistics(features, columns):
    rows = []
    for code in columns:
        values = observed_feature(features, code)
        known = values.dropna()
        numeric = pd.api.types.is_numeric_dtype(values)
        rows.append({"Concept": FEATURES[code]["label"], "Feature": code,
                     "Episodes with data": len(known), "Missing": int(values.isna().sum()),
                     "Mean": known.mean() if numeric else np.nan,
                     "SD": known.std() if numeric else np.nan,
                     "Minimum": known.min() if numeric else np.nan,
                     "25th percentile": known.quantile(.25) if numeric else np.nan,
                     "Median": known.median() if numeric else np.nan,
                     "75th percentile": known.quantile(.75) if numeric else np.nan,
                     "Maximum": known.max() if numeric else np.nan,
                     "Definition": FEATURES[code]["description"]})
    return pd.DataFrame(rows)


def distribution_details(counts, features):
    """Figure 2 families: fitted concept distributions and related full-record codes."""
    chunks = []
    families = model_family_columns(features)
    for model, title, codes in MODEL_GROUPS:
        columns = families[model]
        fig = model_family_figure(features, columns)
        stream = BytesIO()
        fig.savefig(stream, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        chart = f'<img alt="{escape(title)} distributions" style="max-width:100%;height:auto" src="data:image/png;base64,{encoded}">'
        stats = family_statistics(features, columns).to_html(index=False, escape=True, float_format=lambda x: f"{x:,.3g}", na_rep="—")
        table = counts[counts.code.isin(codes)][["code", "name", "unit", "value", "kind", "count", "n", "share"]].copy()
        table["share"] = table.share.map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        table.columns = ["Code", "Name", "Unit", "Value", "Status", "Count", "Denominator", "Share"]
        raw = ('<details><summary>Related annotation codes · whole-episode counts</summary>'
               '<p>These are the source annotations, before aggregation into the model concepts above. '
               'They cover the full record, so their denominators differ from the model summaries.</p>'
               f'<div style="max-height:440px;overflow:auto">{table.to_html(index=False, escape=True)}</div></details>') if len(table) else ""
        note = ("Venture, ask, and interaction controls are included here. Delivery uses the fitted positivity rank score; the raw positivity rating is in the annotation table."
                if model == "C2p" else
                "Reactions are derived from investor turns following an answer, using the same definitions as the fitted model."
                if model == "C4r" else
                "Social, hostile, and repair turns are separated by speaker role as in the fitted model."
                if model == "C5r" else "")
        chunks.append(f'<details style="margin:12px 0"><summary><b>{escape(title)}</b> · distributions and statistics</summary>'
                      f'<p>{note}</p>{chart}<div style="max-height:440px;overflow:auto">{stats}</div>{raw}</details>')
    return "".join(chunks)
