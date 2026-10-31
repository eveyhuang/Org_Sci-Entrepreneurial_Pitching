"""Guard presentation against changing the estimand or fabricating review/record values."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import demo, panel_explorer as px
from analysis.panel_bottleneck import chance_comparisons
from analysis.proposal_figures import family_chance_panel, figure2, PANEL_A_ITEMS
from pipeline.stage_b.review_page import render_notebook_preview


def test_chance_comparison_uses_permutation_p_not_bootstrap_p():
    metrics = pd.read_csv(demo.REFERENCE / 'analysis/all100/metrics.csv')
    original = metrics.copy(deep=True)
    rows = chance_comparisons(metrics).set_index('model')
    saved = metrics.set_index('model')
    assert list(rows.index) == ['C2p', 'C3q', 'C3a', 'C4r', 'C5r', 'C5']
    np.testing.assert_allclose(rows.above_chance, saved.loc[rows.index, 'auc'] - .5)
    np.testing.assert_allclose(rows.perm_p_auc, saved.loc[rows.index, 'perm_p_auc'])
    assert round(rows.loc['C3a', 'perm_p_auc'], 3) == .014
    assert round(saved.loc['C3a', 'auc_p_value'], 3) == .008
    assert chance_comparisons(metrics.drop(columns='perm_p_auc')).perm_p_auc.isna().all()
    pd.testing.assert_frame_equal(original, metrics)
    fig, ax = plt.subplots()
    family_chance_panel(ax, metrics)
    lines = [line for line in ax.lines if list(line.get_xdata()) == [50, 50]]
    assert len(lines) == 1 and lines[0].get_color() == 'black'
    assert not any('baseline' in t.get_text().lower() for t in ax.texts + ax.get_yticklabels())
    plt.close(fig)


def test_paper_figure_shows_offer_group_means_on_rating_and_percentage_scales():
    folder = demo.REFERENCE / 'analysis/all100'
    tables = {name: pd.read_csv(folder / f'{name}.csv')
              for name in ('features', 'single_feature', 'metrics')}
    fig = figure2(folder, tables=tables)
    assert len(fig.axes) == 4
    assert [ax.get_xlim() for ax in fig.axes[:3]] == [(1, 5), (1, 5), (0, 80)]
    points = {line.get_gid(): line for ax in fig.axes[:3] for line in ax.lines if line.get_gid()}
    assert len(points) == 14
    for _, feature, scale in PANEL_A_ITEMS:
        known = tables['features']
        if f'{feature}_missing' in known:
            known = known[known[f'{feature}_missing'] != 1]
        for outcome in (1, 0):
            expected = known.loc[known.any_offer == outcome, feature].mean() * scale
            np.testing.assert_allclose(points[f'{feature}-offer-{outcome}'].get_xdata(), [expected])
    left, right = fig.subfigs
    assert [t.get_text() for t in left.texts].count('Answers to sharks’ questions') == 1
    assert [t.get_text() for t in left.legends[0].texts] == ['with an offer (73)', 'without an offer (27)']
    assert fig.axes[-1].get_title(loc='left') == ''
    assert 'Bold labels indicate p < 0.05 in both panels.' in [t.get_text() for t in right.texts]
    plt.close(fig)


def test_preview_preserves_submitted_values_and_notes_without_running_scripts():
    source = demo.review_sheet_path().read_text()
    model, _ = json.JSONDecoder().raw_decode(source.split('const PAGE = ', 1)[1])
    html = render_notebook_preview(model)
    for video in model['videos']:
        for section in video['sections']:
            for card in section['cards']:
                assert f'data-path="{card["path"]}"' in html
    first = next(iter(model['completed_answers'][demo.EXAMPLE_ID].values()))
    first.update(note='<script>alert("note")</script>', unsure=True)
    html = render_notebook_preview(model)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html and 'Unsure' in html
    assert '5:52' in html
    blank = demo.review_sheet_preview(completed=False)
    assert 'Not answered' in blank
    assert 'human1’s answer:</b> 5:52' not in blank


def test_detail_timeline_draws_record_links_and_handles_missing_codes():
    _, result, _, _, _ = demo.encounter(demo.REFERENCE, demo.EXAMPLE_ID)
    before = json.dumps(result, sort_keys=True)
    fig = px.annotated_timeline_figure(result, start_s=125, end_s=232)
    actual = {p.get_gid() for p in fig.axes[0].patches if p.get_gid()}
    visible = {t['turn_id']: t for t in result['record']['turns'] if t['end_s'] > 125 and t['start_s'] < 232}
    expected = {f"answer-{t['turn_id']}-to-question-{q}" for t in visible.values()
                if t['function'] == 'answer' for q in t['responds_to']
                if q in visible and visible[q]['function'] == 'question'}
    assert actual == expected and len(actual) > 0
    assert '4 ✓' in [t.get_text() for t in fig.axes[0].texts]
    plt.close(fig)
    assert json.dumps(result, sort_keys=True) == before
    for t in result['record']['turns']:
        t['codes'] = []
    fig = px.annotated_timeline_figure(result, start_s=125, end_s=232)
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert '—' in labels and not any('✓' in label for label in labels)
    plt.close(fig)


def test_open_sheet_uses_system_browser_only_when_called(monkeypatch):
    opened = []
    monkeypatch.setattr('webbrowser.open', lambda url, new: opened.append(url) or True)
    demo.review_sheet_preview()
    assert not opened
    path = demo.open_review_sheet(completed=False)
    assert opened == [path.as_uri()]
    assert path.name == 'human1__tasks.html'
