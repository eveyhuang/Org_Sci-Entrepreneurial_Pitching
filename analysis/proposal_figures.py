"""Paper Figure 2 shared by the export script and Demo.ipynb.

Panel A reproduces the approved grouped-means layout and the paper mockup's
2,000-draw percentile bootstrap (seed 42), with saved single-feature p-values.
Panel B reads saved held-out model metrics and permutation p-values.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from analysis.pitch_figures import BLUE, VERMILLION, GREY, INK, GRID, style, _point, _clipped_interval
from analysis.panel_bottleneck import chance_comparisons


PANEL_A_ITEMS = [
    ("substance", "pres_subs", 1), ("coherence", "pres_coh", 1),
    ("positivity in delivery", "sa_pitch_positivity", 1),
    ("substance", "a_subs_mean", 1), ("coherence", "a_coh_mean", 1),
    ("checkable specific", "a_evid_share", 100),
    ("admitted uncertainty", "a_uncert_share", 100),
]


def panel_a_statistics(features, single):
    """Per-episode group means and 95% intervals, matching the paper means mockup."""
    p_values = single.set_index('feature')['p_value']
    rows = []
    for label, feature, scale in PANEL_A_ITEMS:
        known = features
        if f'{feature}_missing' in known:
            known = known[known[f'{feature}_missing'] != 1]
        for outcome in (1, 0):
            values = known.loc[known.any_offer == outcome, feature].dropna().to_numpy(dtype=float) * scale
            if not len(values):
                raise ValueError(f'No observed {feature} values for any_offer={outcome}')
            rng = np.random.default_rng(42)
            draws = [rng.choice(values, len(values)).mean() for _ in range(2000)]
            low, high = np.percentile(draws, [2.5, 97.5])
            rows.append(dict(feature=feature, label=label, any_offer=outcome, n=len(values),
                             mean=values.mean(), low=low, high=high, p=p_values[feature]))
    return pd.DataFrame(rows)


def preparedness_means_panel(fig, features, single):
    """Draw Option 1 into a Figure or SubFigure; keep both answer scales under one heading."""
    stats = panel_a_statistics(features, single).set_index(['feature', 'any_offer'])
    transform = getattr(fig, 'transSubfigure', fig.transFigure)
    fig.text(.045, .972, 'A  Preparedness in the pitch and in the answers, by offer',
             fontsize=11, fontweight='bold', va='top')
    counts = features.any_offer.value_counts()
    handles = [Line2D([], [], marker='o', color=BLUE, ls='none', ms=5.5,
                       label=f'with an offer ({counts.get(1, 0)})'),
               Line2D([], [], marker='s', color=VERMILLION, mfc='white', mew=1.2,
                       ls='none', ms=5.5, label=f'without an offer ({counts.get(0, 0)})')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.96, .923),
               borderaxespad=0, handletextpad=.4, labelspacing=.45, fontsize=9)

    def heading(y, title):
        fig.text(.045, y, title, fontsize=11, fontweight='bold', va='center')
        fig.add_artist(Line2D([.045, .96], [y-.025, y-.025], transform=transform,
                             color='#c9ced3', lw=.65))

    def strip(position, items, limits, ticks, xlabel):
        ax = fig.add_axes(position)
        ys = list(range(len(items)-1, -1, -1))
        for y, (label, feature, scale) in zip(ys, items):
            for outcome, dy, colour, marker, fill in ((1, .13, BLUE, 'o', BLUE),
                                                     (0, -.13, VERMILLION, 's', 'white')):
                row = stats.loc[(feature, outcome)]
                ax.plot([row.low, row.high], [y+dy, y+dy], color=colour, lw=1.2, zorder=2)
                point, = ax.plot(row['mean'], y+dy, marker=marker, color=colour, mfc=fill,
                                 mew=1.2, ms=5.5, zorder=3)
                point.set_gid(f'{feature}-offer-{outcome}')
            p_value = stats.loc[(feature, 1), 'p']
            ax.text(1.025, y, f'p = {p_value:.3f}', transform=ax.get_yaxis_transform(),
                    va='center', fontsize=8, color='black' if p_value < .05 else GREY,
                    fontweight='bold' if p_value < .05 else 'normal')
        ax.set_yticks(ys, labels=[item[0] for item in items])
        ax.tick_params(axis='y', length=0, pad=8, labelsize=9)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks)
        ax.set_ylim(-.5, len(items)-.5)
        ax.set_xlabel(xlabel, fontsize=9, labelpad=5)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#777777')
        ax.set_facecolor('none')
        ax.grid(axis='x', color='#dedede', lw=.55)
        ax.set_axisbelow(True)
        return ax

    heading(.85, 'The pitch')
    pitch = strip([.35, .595, .47, .215], PANEL_A_ITEMS[:3], (1, 5), [1, 2, 3, 4, 5], 'Mean rating (1–5)')
    fig.add_artist(Rectangle((.025, .073), .95, .455, transform=transform,
                             facecolor='#f5f6f8', edgecolor='none', zorder=-10))
    heading(.502, 'Answers to sharks’ questions')
    answers = strip([.35, .325, .47, .135], PANEL_A_ITEMS[3:5], (1, 5), [1, 2, 3, 4, 5], 'Mean rating (1–5)')
    shares = strip([.35, .132, .47, .115], PANEL_A_ITEMS[5:], (0, 80), [0, 20, 40, 60, 80], 'Share of answers (%)')
    fig.text(.045, .032, 'Points show group means; lines show 95% bootstrap intervals.', fontsize=8.5, color=GREY)
    return pitch, answers, shares


def figure2(analysis_dir, *, tables=None):
    """Regenerate the paper's Figure 2 from an analysis export; write nothing."""
    run = Path(analysis_dir)
    def read(name):
        return tables[name].copy() if tables is not None else pd.read_csv(run / f'{name}.csv')
    with matplotlib.rc_context({**style(), 'font.family': 'DejaVu Sans', 'font.size': 9,
                                'axes.linewidth': .6, 'xtick.labelsize': 8.5, 'pdf.fonttype': 42}):
        fig = plt.figure(figsize=(15.5, 7.5), dpi=150, facecolor='white')
        left, right = fig.subfigures(1, 2, width_ratios=[7.2, 8.3], wspace=.015)
        preparedness_means_panel(left, read('features'), read('single_feature'))
        ax = right.add_axes([.37, .132, .43, .70])
        family_chance_panel(ax, read('metrics'))
        ax.set_title('', loc='left')
        ax.tick_params(axis='y', labelsize=9)
        ax.set_xlabel('Pairs of one episode with an offer and one without\nranked correctly, per 100', fontsize=9)
        right.text(.025, .972, 'B  What each family of concepts adds to predicting an offer',
                   fontsize=11, fontweight='bold', va='top')
        right.text(.025, .032, 'Bold labels indicate p < 0.05 in both panels.', fontsize=8.5, color=GREY)
    return fig


def family_chance_panel(ax, metrics, *, panel_letter="B"):
    """Shared Panel B/C renderer: saved AUC and permutation tests against chance."""
    rows = chance_comparisons(metrics)
    ax.set_title(f"{panel_letter}   What each family adds to predicting an offer",
                 loc="left", fontweight="bold")
    ax.axvline(50, color="black", linewidth=1.2, zorder=1)
    for i, r in enumerate(rows.itertuples()):
        colour, marker = (INK, "D") if r.model == "C2p" else (BLUE, "o")
        _clipped_interval(ax, 100 * r.auc_low, 100 * r.auc_high, i, colour, None)
        _point(ax, 100 * r.auc, i, colour, marker, True, size=5.5)
        p_text = "p unavailable" if pd.isna(r.perm_p_auc) else (
            "p < 0.001" if r.perm_p_auc < .001 else f"p = {r.perm_p_auc:.3f}")
        significant = pd.notna(r.perm_p_auc) and r.perm_p_auc < .05
        ax.annotate(f"{100 * r.above_chance:+.0f}, {p_text}", (1.02, i),
                    xycoords=("axes fraction", "data"), va="center", fontsize=7.5,
                    color='black' if significant else GREY,
                    fontweight='bold' if significant else 'normal', clip_on=False)
    ax.set_yticks(range(len(rows)), labels=rows.family)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(len(rows) - .5, -.7)
    ax.set_xlim(25, 85)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
    ax.set_xlabel("Pairs of one pitch with an offer and one without ranked correctly, per 100")
    ax.grid(axis="x", color=GRID, linewidth=.5)
    ax.set_axisbelow(True)
    ax.annotate("above chance\nby, p", (1.02, -.65), xycoords=("axes fraction", "data"),
                va="bottom", fontsize=7.5, color=GREY, clip_on=False)
    ax.text(50.7, -.6, "chance (50)", fontsize=7, color=INK, va="bottom")
    return ax


def chance_table(metrics):
    """Readable table using precisely the values and tests shown in family_chance_panel."""
    rows = chance_comparisons(metrics)
    return pd.DataFrame({
        "Concepts included": rows.family.str.replace("\n", " ", regex=False),
        "Pairs ranked correctly / 100": (100 * rows.auc).round(1),
        "95% interval": [f"{100*r.auc_low:.1f}–{100*r.auc_high:.1f}" for r in rows.itertuples()],
        "Above chance": (100 * rows.above_chance).round(1),
        "Permutation p": rows.perm_p_auc.round(3),
    })
