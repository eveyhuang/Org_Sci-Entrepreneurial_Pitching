"""Notebook views of the saved, sampled human validation; no agreement is refitted."""
from __future__ import annotations

import pandas as pd


FAMILIES = (
    ("Pitch preparedness and delivery + controls", (
        "PRES-SUBS", "PRES-COH", "PRES-EVID", "pitch_positivity",
        "expressed_enthusiasm", "interpersonal_warmth")),
    ("The sharks’ questions", ("PR-FRAME", "PR-SEV")),
    ("The entrepreneurs’ answers", ("RS-RESP", "RS-EVID", "PREP-SUBS", "PREP-COH")),
)


def _number(value):
    return "—" if value is None else f"{value:.2f}"


def _agreement(row):
    if not row or row.get("missing"):
        return "Not assessed"
    stat = "ac2" if row["kind"] == "ordinal" else "ac1"
    point = row.get("figures", {}).get(stat, {})
    alpha = row.get("alpha", {}).get("value")
    small = " †" if "small_n" in row.get("flags", []) else ""
    return (f"{stat.upper()} {_number(point.get('value'))} (α {_number(alpha)}){small}; "
            f"95% CI [{_number(point.get('low'))}, {_number(point.get('high'))}]")


def agreement_table(report):
    """AC1/AC2, alpha, intervals and denominators, grouped as in Figure 2."""
    summary = {r["field"]: r for r in report["summary_rows"]}
    model = {r["field"]: r for r in report["rows"]}
    human = {r["field"]: r for r in report["human_human_rows"]}
    rows = []
    for family, codes in FAMILIES:
        for code in codes:
            s, m, h = summary[code], model.get(code, {}), human.get(code, {})
            # Include every category actually supplied to the statistic, even if unused.
            scale = " / ".join(m["counts"]) if m.get("kind") == "categorical" else s["scale"]
            rows.append({"Family": family, "Code": s["code"], "Scale": scale,
                         "Gemini vs. human: AC (α), 95% CI": _agreement(m),
                         "Gemini–human: units / judgments": f"{m.get('units', '—')} / {m.get('n', '—')}",
                         "Human1 vs. human2: AC (α), 95% CI": _agreement(h),
                         "Human–human: units": h.get("n")})
    return pd.DataFrame(rows)


def additional_checks(report):
    """Clock, ask and paired ranking use their own statistics, never invented AC/alpha."""
    rows = []
    for row in report["summary_rows"]:
        if row["kind"] in ("clock", "exact", "comparison"):
            rows.append({"Code": row["code"], "Scale": row["scale"],
                         "Gemini vs. human": row["gemini_human"],
                         "Human1 vs. human2": row["human_human"]})
    return pd.DataFrame(rows)


def styled_agreement(table):
    """Separate families visually while preserving ordinary DataFrame access."""
    starts = table.Family.ne(table.Family.shift())
    def borders(row):
        return ["border-top: 2px solid #7b8794" if starts.loc[row.name] else ""] * len(row)
    return (table.style.hide(axis="index").format(escape="html").set_uuid("corpus_validation")
            .apply(borders, axis=1)
            .set_properties(**{"text-align": "left", "vertical-align": "top"})
            .set_properties(subset=["Gemini vs. human: AC (α), 95% CI",
                                    "Human1 vs. human2: AC (α), 95% CI"],
                            **{"font-weight": "600", "min-width": "205px"}))
