"""Draw the paper's agreement figure from the validation table's JSON (`review_panel.py summary --paper-table`).

One row per code with a chance-corrected statistic: Gwet's AC1 (categorical and binary codes) or AC2 (ordinal codes)
for Gemini against the human coders and for human 1 against human 2, each with its 95% interval resampling videos, the
values with Krippendorff's alpha in parentheses at the right, and a line at 0.61, where Landis & Koch's "substantial"
band begins.

    python scripts/plot_agreement.py runs/full-corpus-100/review/agreement/validation_table.json

writes agreement_by_code.png and agreement_by_code.svg beside the JSON (or under --out).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUBSTANTIAL = 0.61
STATISTIC = {"categorical": "ac1", "binary": "ac1", "ordinal": "ac2"}
# the figure's groups and their codes, top to bottom
GROUPS = [
    ("Questions and answers", ["PR-FRAME", "RS-RESP", "RS-EVID", "PR-SEV", "PREP-SUBS", "PREP-COH"]),
    ("Prepared presentation", ["PRES-SUBS", "PRES-COH", "PRES-EVID"]),
    ("Delivery", ["pitch_positivity", "expressed_enthusiasm", "interpersonal_warmth"]),
]
WORDS = {
    "PR-FRAME": "Question frame", "RS-RESP": "Answer addresses the question",
    "RS-EVID": "Answer gives a checkable specific", "PR-SEV": "Question severity", "PREP-SUBS": "Answer substance",
    "PREP-COH": "Answer coherence", "PRES-SUBS": "Substance", "PRES-COH": "Coherence",
    "PRES-EVID": "Checkable specifics", "pitch_positivity": "Pitch positivity",
    "expressed_enthusiasm": "Expressed enthusiasm", "interpersonal_warmth": "Interpersonal warmth",
}
SERIES = [  # (JSON key, legend label, colour, marker): two hues that pass the colour-blindness check, and two shapes
    ("rows", "Gemini vs. human", "#2a78d6", "o"),
    ("human_human_rows", "Human 1 vs. human 2", "#eb6834", "D"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _num(x) -> float | None:
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _point(row: dict | None) -> dict | None:
    if not row or row.get("missing") or row.get("kind") not in STATISTIC:
        return None
    stat = (row.get("figures") or {}).get(STATISTIC[row["kind"]]) or {}
    alpha = (row.get("alpha") or {}).get("value")
    return {"value": _num(stat.get("value")), "low": _num(stat.get("low")), "high": _num(stat.get("high")),
            "alpha": _num(alpha), "statistic": STATISTIC[row["kind"]].upper(),
            "small_n": "small_n" in (row.get("flags") or [])}


def _label(p: dict | None) -> str:
    if p is None or p["value"] is None:
        return "—"
    alpha = "—" if p["alpha"] is None else f"{p['alpha']:.2f}".replace("-", "−")
    return f"{p['value']:.2f} (α {alpha})"


def draw(data: dict, out: Path) -> list[Path]:
    by = {key: {r["field"]: r for r in data.get(key) or []} for key, *_ in SERIES}
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "svg.fonttype": "none"})
    n_rows = sum(len(codes) for _, codes in GROUPS) + len(GROUPS)
    fig = plt.figure(figsize=(9.2, 0.34 * n_rows + 1.3), facecolor="#fcfcfb")
    ax = fig.add_axes([0.29, 0.12, 0.40, 0.78], facecolor="#fcfcfb")
    y, ticks, names, rows = 0.0, [], [], []
    for title, codes in GROUPS:
        ax.text(-0.02, y, title, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=9,
                fontweight="bold", color=INK)
        y += 1
        for code in codes:
            points = [_point(by[key].get(code)) for key, *_ in SERIES]
            kind = next((p["statistic"] for p in points if p), "")
            small = any(p and p["small_n"] for p in points)
            ticks.append(y)
            names.append(f"{WORDS.get(code, code)} · {kind}" + (" †" if small else ""))
            rows.append((y, points))
            y += 1
        y += 0.3
    offset = 0.17
    for i, (key, label, colour, marker) in enumerate(SERIES):
        dy = -offset if i == 0 else offset
        for yy, points in rows:
            p = points[i]
            if p is None or p["value"] is None:
                continue
            if p["low"] is not None and p["high"] is not None:
                ax.plot([p["low"], p["high"]], [yy + dy] * 2, color=colour, lw=1.4, alpha=0.55,
                        solid_capstyle="round", zorder=2)
            ax.scatter([p["value"]], [yy + dy], s=36 if marker == "o" else 28, marker=marker, color=colour,
                       edgecolor="#fcfcfb", linewidth=1.2, zorder=3, label=label if yy == rows[0][0] else None)
    ax.axvline(SUBSTANTIAL, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(SUBSTANTIAL, -0.9, f"substantial ({SUBSTANTIAL:.2f})", ha="center", va="bottom", fontsize=8,
            color=MUTED)
    ax.set_yticks(ticks, names, color=INK)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(y - 0.2, -1.0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="x", colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Gwet's AC1 (categories) or AC2 (rating scales), with 95% interval", color=MUTED)
    # the values with alpha in parentheses, one column per series
    for i, (key, label, colour, marker) in enumerate(SERIES):
        x = 1.06 + 0.33 * i
        ax.text(x + 0.035, -0.55, label, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8,
                color=INK)
        ax.scatter([x + 0.01], [-0.55], marker=marker, s=26 if marker == "o" else 20, color=colour,
                   transform=ax.get_yaxis_transform(), clip_on=False)
        for yy, points in rows:
            ax.text(x, yy, _label(points[i]), transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=8.5, color=INK)
    fig.text(0.02, 0.02, "† fewer than 20 units (10 pitches): too few for a stable chance-corrected statistic.\n"
             "Dashed line: lower edge of Landis & Koch's (1977) \"substantial\" band. Values at right: AC1 or AC2, "
             "with Krippendorff's α in parentheses.", fontsize=7.5, color=MUTED, linespacing=1.4)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "agreement_by_code.png", out / "agreement_by_code.svg"]
    fig.savefig(paths[0], dpi=220)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("table", type=Path, help="validation_table.json from review_panel.py summary --paper-table")
    p.add_argument("--out", type=Path, help="folder for the figure (default: beside the JSON)")
    args = p.parse_args(argv)
    for path in draw(json.loads(args.table.read_text(encoding="utf-8")), args.out or args.table.parent):
        print(f"figure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
