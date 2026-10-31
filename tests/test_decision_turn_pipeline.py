"""Same-turn decision agreement through the review pipeline, without importing the demo module."""
import json

import pytest

from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.review_apply import pooled_summary
from pipeline.stage_b.review_report import load_reporting, paper_table_markdown
from tests.test_review_cli import make_run, _coder_video, rp


def context():
    return {'pass': 'primary', 'sha256': 'fixture-reference',
            'turns': [[30, 326, 340], [31, 340, 355], [32, 355, 375]]}


def line(human='5:52', model='5:40', reference=True):
    return {'path': 'pitch:first_decision', 'field_class': 'cut', 'primary': model,
            'replicate': '5:41', 'new': human, 'resolution': 'outcome_override',
            'selected_as': 'sampled', 'reviewer': 'ana', 'role': 'coder', 'level': 'interval',
            **({'decision_turn_context': context()} if reference else {})}


def test_shared_reports_bootstrap_and_strata_use_turn_identity():
    books = load_panel_books(load_release())
    reporting = load_reporting(books, load_release())
    # The last video has timestamps only one second apart, but in adjacent turns.
    videos = {f'v{i}': [line() if i < 5 else line('5:55', '5:54')] for i in range(6)}
    strata = {'stratify': 'any_offer', 'of': {v: 'yes' if i < 3 else 'no' for i, v in enumerate(videos)},
              'run': {'yes': 30, 'no': 30}}
    out = pooled_summary(videos, n_boot=99, reporting=reporting, strata=strata,
                         by_reviewer={'ana': videos})
    assert out['sampled']['cut']['human_vs_primary']['agree'] == pytest.approx(5 / 6)
    row = out['reporting']['cut']
    assert row['figures']['same_turn']['value'] == pytest.approx(5 / 6)
    assert row['figures']['share_within_5']['value'] == pytest.approx(1 / 6)
    assert row['figures']['same_turn']['low'] is not None
    assert row['without_unsure']['same_turn'] == pytest.approx(5 / 6)
    assert row['strata']['yes']['same_turn'] == 1
    assert row['strata']['reweighted']['same_turn'] == pytest.approx(5 / 6)
    assert 'same turn 0.83 (5/6)' in paper_table_markdown(out, reporting)


def test_missing_context_and_different_reference_context_are_undetermined():
    books = load_panel_books(load_release())
    reporting = load_reporting(books, load_release())
    a = {'v': [line(reference=False)]}
    out = pooled_summary(a, reporting=reporting)
    assert out['sampled']['cut']['human_vs_primary'] == {'agree': None, 'n': 0}
    assert out['sampled']['cut']['decision_undetermined'] == 1
    assert out['reporting']['cut']['counts']['same_turn_undetermined'] == 1
    a = {'v': [line()]}
    b = {'v': [{**line(), 'reviewer': 'ben', 'decision_turn_context': {**context(), 'sha256': 'other-record'}}]}
    out = pooled_summary({'v': a['v'] + b['v']}, by_reviewer={'ana': a, 'ben': b}, reporting=reporting)
    assert out['reporting']['human_human']['cut']['figures']['same_turn']['value'] is None


@pytest.mark.parametrize('reference_pass', ['primary', 'replicate'])
def test_import_and_summary_cli_use_pinned_turns_and_preserve_typed_time(tmp_path, monkeypatch, reference_pass):
    books = load_panel_books(load_release())
    run, vdir = make_run(tmp_path, books)
    monkeypatch.setattr(rp, 'model_pass', lambda _: reference_pass)
    before = {p: p.read_bytes() for p in (vdir / 'results').glob('*.json')}
    # 1:55 is the next turn, although within 5 seconds of the first decision at 1:51.
    _coder_video(run, vdir, {'pitch:first_decision': {'human_value': '1:55'}})
    path = vdir / 'review/review_log.jsonl'
    logged = json.loads(path.read_text().splitlines()[0])
    assert logged['typed'] == '1:55'
    assert logged['decision_turn_context']['pass'] == reference_pass
    imported = json.loads((vdir / 'review/human_agreement.json').read_text())
    assert imported['by_field_class']['cut']['human_vs_primary'] == {'agree': 0, 'n': 1}
    assert imported['by_field_class']['cut']['clock_diagnostics']['human_vs_primary']['agree'] == 1
    out = tmp_path / 'validation.md'
    assert rp.main(['summary', '--run', str(run), '--paper-table', str(out)]) == 0
    data = json.loads(out.with_suffix('.json').read_text())
    cut = next(r for r in data['rows'] if r['field'] == 'cut')
    assert cut['figures']['same_turn']['value'] == 0
    assert cut['figures']['share_within_5']['value'] == 1
    assert all(p.read_bytes() == saved for p, saved in before.items())
    # Historical logs recover context read-only from the still-pinned export, never from current best-pass choice.
    logged.pop('decision_turn_context')
    path.write_text(json.dumps(logged) + '\n')
    saved_log = path.read_bytes()
    assert rp._log_lines(path)[0]['decision_turn_context']['pass'] == reference_pass
    assert path.read_bytes() == saved_log
    target = vdir / 'results' / f'{reference_pass}.json'
    target.write_text(target.read_text() + '\n')
    assert 'decision_turn_context' not in rp._log_lines(path)[0]  # stale export hash: no guessed agreement
