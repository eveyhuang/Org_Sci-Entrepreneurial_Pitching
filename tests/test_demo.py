"""Contracts for the Figure 1 demo; synthetic human submissions exist only in temp directories."""
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

from analysis import demo
from pipeline.errors import ConfigError
from pipeline.instrument import load_release
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.review_items import banded
from scripts.build_demo_review import prepare, complete, page_model, refresh


def fixture_submission(tmp_path):
    folder = tmp_path / 'example'
    shutil.copytree(demo.REVIEW, folder)
    m = demo.read_json(folder / 'manifest.json')
    # Start a fresh submission only in the temporary fixture; the published example has real answers.
    for name in ('answers', 'derived', 'original_submission'):
        shutil.rmtree(folder / name, ignore_errors=True)
    (folder / 'pages/human1__completed.html').unlink(missing_ok=True)
    m['status'] = 'awaiting_annotation'
    m.pop('answer_files', None)
    m.pop('completed_files', None)
    demo.write_json(folder / 'manifest.json', m)
    submitted = tmp_path / 'synthetic_sheets'
    submitted.mkdir()
    full = demo.read_json(folder / 'exports/export_manifest.json')['videos'][demo.EXAMPLE_ID]
    session = demo.read_json(folder / 'exports/clips/export_manifest.json')
    for name, rows, template in [
        (f'{demo.EXAMPLE_ID}__human1.csv', full['rows'], folder / 'exports' / f'{demo.EXAMPLE_ID}__review.csv'),
        ('clips__human1.csv', session['rows'], folder / 'exports/clips/clips.csv')]:
        with template.open(encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fields, originals = reader.fieldnames, list(reader)
        by_path = {r['path']: r for r in rows}
        for row in originals:
            row.update(human_value=banded(row['path'], by_path[row['path']]['primary']),
                       context=f"design {m['design_id']}", resolution='', note='synthetic test fixture')
        with (submitted / name).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader();writer.writerows(originals)
    return folder, submitted


def test_one_video_has_32_cards_and_no_model_answers(tmp_path):
    m = demo.review_manifest()
    model = page_model(demo.REVIEW)
    cards = [c for v in model['videos'] for s in v['sections'] for c in s['cards']]
    assert len(cards) == m['card_count'] == 32
    assert len(model['videos']) == 1
    assert all('model' not in c and 'primary' not in c and 'replicate' not in c for c in cards)
    assert demo.read_json(demo.REVIEW / 'exports/clips/export_manifest.json')['pairs'] == []
    folder = tmp_path / 'example'
    shutil.copytree(demo.REVIEW, folder)
    before = (folder / 'exports/export_manifest.json').read_bytes()
    prepare(folder)
    assert before == (folder / 'exports/export_manifest.json').read_bytes()


def test_one_video_manifest_needs_no_ranking(tmp_path, monkeypatch):
    from tests.test_review_cli import make_run
    from pipeline.stage_b import review_clips as rc
    books = load_panel_books(load_release())
    run, _ = make_run(tmp_path, books)
    monkeypatch.setattr(rc, 'model_scores', lambda *a: pytest.fail('unneeded ranking'))
    manifest = rc.build_manifest(run, ['vid1'], ['human1'], 0, books, 'v3-dev9', {}, include_pairs=False)
    assert len(manifest['rows']) == 7 and manifest['pairs'] == []
    assert manifest['scores_source'] is None


def test_submitted_answers_drive_both_agreement_and_completed_html(tmp_path):
    folder, submitted = fixture_submission(tmp_path)
    before = demo.review_manifest(folder)['research_files']
    loaded = demo.load_human_review(folder, submitted)
    assert loaded['cards'].human_vs_model.dropna().all()
    assert len(loaded['cards']) == 32
    target = complete(submitted, folder)
    html = target.read_text()
    payload = json.loads(html.split('const PAGE = ', 1)[1].split(';\n', 1)[0])
    assert payload['read_only']
    for path, row in loaded['submitted'].items():
        assert payload['completed_answers'][demo.EXAMPLE_ID][path]['value'] == row['human_value']
    assert demo.read_json(folder / 'manifest.json')['status'] == 'completed'
    preserved = {p: p.read_bytes() for p in (folder / 'answers').glob('*.csv')}
    preserved[target] = target.read_bytes()
    refresh(folder)
    assert all(p.read_bytes() == content for p, content in preserved.items())
    assert demo.review_manifest(folder)['status'] == 'completed'
    assert demo.verify_files(before) == len(before)
    assert not (demo.REFERENCE / demo.EXAMPLE_ID / 'results/reviewed.json').exists()
    with pytest.raises(ConfigError, match='already preserved'):
        complete(submitted, folder)


def test_submission_mismatch_rejected_without_publishing(tmp_path):
    folder, submitted = fixture_submission(tmp_path)
    path = submitted / 'clips__human1.csv'
    path.write_text(path.read_text(encoding='utf-8-sig').replace('design ', 'wrong '))
    with pytest.raises(ConfigError, match='design'):
        complete(submitted, folder)
    assert not (folder / 'answers').exists()
    assert not (folder / 'pages/human1__completed.html').exists()


def test_partial_answers_are_pending_not_zero_reliability(tmp_path):
    folder, submitted = fixture_submission(tmp_path)
    path = submitted / 'clips__human1.csv'
    with path.open(encoding='utf-8-sig') as f:
        r=csv.DictReader(f); fields=r.fieldnames; rows=list(r)
    rows[0].update(human_value='',resolution='',note='')
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    result = demo.load_human_review(folder, submitted)
    assert len(result['cards']) == 32
    assert result['cards'].human_vs_model.isna().sum() == 1
    with pytest.raises(ConfigError, match='still blank'):
        complete(submitted, folder)
    assert not (folder/'answers').exists()


def test_completed_notes_are_escaped_and_flags_preserved():
    from pipeline.stage_b.review_page import render_page
    model = page_model(demo.REVIEW)
    cards = [c for s in model['videos'][0]['sections'] for c in s['cards']]
    submitted = {c['path']: {'human_value':'', 'resolution':'unclear',
                            'note':'[unsure] </script><script>alert("x")</script>'} for c in cards}
    state = demo.completed_state(model, submitted)
    assert all(s['unclear'] for s in state[demo.EXAMPLE_ID].values())
    model.update(read_only=True,completed_answers=state)
    html = render_page(model)
    assert '</script><script>alert' not in html
    assert '<\\/script>' in html


def test_reference_prediction_and_support_trace():
    d,f,s,m=demo.load_reference()
    p,breakdown,fitted=demo.held_out_example(f,m,demo.EXAMPLE_ID)
    np.testing.assert_allclose(p,s['predictions'].set_index('video_id').loc[demo.EXAMPLE_ID,'p_C5'],rtol=1e-8)
    assert demo.EXAMPLE_ID not in fitted['folds'][0]['train_groups']
    np.testing.assert_allclose(1/(1+np.exp(-breakdown.contribution.sum())),p)
    _,record,_,_,_=demo.encounter(demo.REFERENCE,demo.EXAMPLE_ID)
    trace=demo.support_trace(d,demo.EXAMPLE_ID,record)
    row=f.set_index('video_id').loc[demo.EXAMPLE_ID]
    assert trace.iloc[:,2].notna().sum()==row.a_evid_share_n
    assert trace.iloc[:,2].mean()==pytest.approx(row.a_evid_share)


def test_custom_codebook_preserves_contract_and_preflight_never_calls_provider(tmp_path, monkeypatch):
    from scripts.experiments import run_panel_pilot as cli
    release,books,prompt=demo.instrument_preview(demo.CUSTOM_MANIFEST)
    assert release.release_id=='demo-support-v1' and 'a name alone does not count' in prompt
    monkeypatch.setattr(cli,'build_provider',lambda *a: pytest.fail('provider constructed in preflight'))
    media=tmp_path/'example.mp4';media.write_bytes(b'fake preflight media')
    demo.run_live(media,10,'my_pitch',tmp_path/'new_run',manifest=demo.CUSTOM_MANIFEST)
    assert not (tmp_path/'new_run').exists()
    with pytest.raises(ConfigError,match='separate'):
        demo.run_live(media,10,'my_pitch',demo.REFERENCE,execute=True)


def test_offline_notebook_has_no_provider_and_does_not_change_records(monkeypatch):
    import nbformat
    import IPython.display
    from scripts.experiments import run_panel_pilot as cli
    monkeypatch.setattr(cli,'build_provider',lambda *a: pytest.fail('offline provider'))
    monkeypatch.setattr(IPython.display,'display',lambda *a,**kw: None)
    monkeypatch.setattr('webbrowser.open', lambda *a, **kw: True)
    monkeypatch.chdir(demo.ROOT)
    m=demo.review_manifest(); before=dict(m['research_files'])
    namespace={'__name__':'__main__'}
    nb=nbformat.read(demo.ROOT/'Demo.ipynb',as_version=4)
    for cell in nb.cells:
        if cell.cell_type=='code':
            source='\n'.join(l for l in cell.source.splitlines() if not l.startswith('%matplotlib'))
            exec(compile(source,'Demo.ipynb','exec'),namespace)
    assert len(namespace['human']['cards']) == 32
    assert len(demo.human_agreement_table(namespace['human'])) == 6
    assert namespace['episode_statistics']['Episodes with data'].eq(100).all()
    assert not namespace['code_counts'].code.str.startswith('EX-').any()
    demo.verify_files(before)


def test_demo_uses_updated_agreement_statistics(monkeypatch):
    from pipeline.reliability import agreement
    called = set()
    for name in ('gwet_ac1', 'gwet_ac2', 'krippendorff_alpha', 'clock_agreement'):
        original = getattr(agreement, name)
        def tracked(*args, _name=name, _original=original, **kwargs):
            called.add(_name)
            return _original(*args, **kwargs)
        monkeypatch.setattr(agreement, name, tracked)
    human = demo.load_human_review()
    assert called == {'gwet_ac1', 'gwet_ac2', 'krippendorff_alpha', 'clock_agreement'}
    reported = human['pooled']['reporting']
    assert 'PR-FRAME' in reported and 'PR-CONCEPT' not in reported
    # The original seconds differ, but both clocks identify the same reference decision turn.
    assert reported['cut']['figures']['mean_signed']['value'] == 12
    assert human['decision_turn']['agreed'] is True
    assert reported['cut']['figures']['same_turn']['value'] == 1
    assert human['pooled']['sampled']['cut']['human_vs_primary']['agree'] == 1
    assert human['decision_turn']['a_turn_id'] == human['decision_turn']['b_turn_id'] == 31
    clock = human['cards'].set_index('path').loc['pitch:first_decision']
    assert clock.human_vs_model == True
    assert clock.decision_clock_within_tolerance == False
    assert reported['RS-RESP']['figures']['alpha']['value'] is None
    assert reported['RS-RESP']['figures']['ac1']['value'] == 1
    assert all(r['figures']['exact']['low'] is None for k, r in reported.items()
               if k not in ('human_human', 'pass_to_pass', 'cut'))
    payload = json.loads((demo.REVIEW / 'pages/human1__completed.html').read_text()
                         .split('const PAGE = ', 1)[1].split(';\n', 1)[0])
    for path, row in human['submitted'].items():
        assert payload['completed_answers'][demo.EXAMPLE_ID][path]['value'] == row['human_value']
    compact = demo.human_agreement_table(human, include_codes=True).set_index('Code')
    for code, row in compact.iterrows():
        expected = reported[code]
        assert row['Exact'] == f"{round(expected['figures']['exact']['value'] * expected['n'])}/{expected['n']}"


def test_pending_demo_still_loads_without_fabricated_agreement(tmp_path):
    folder, _ = fixture_submission(tmp_path)
    assert demo.load_human_review(folder) is None


def test_live_branch_uses_existing_two_pass_runner_with_fake_provider(tmp_path, monkeypatch):
    from scripts.experiments import run_panel_pilot as cli
    from pipeline.providers.fake import FakeProvider
    from pipeline.execute import TokenBucket
    from tests.panel_fixtures import synthetic_record, to_wire
    monkeypatch.setattr(TokenBucket, 'acquire', lambda *a: None)
    fixtures=tmp_path/'fixtures';fixtures.mkdir()
    for name in ('primary','replicate'):
        (fixtures/f'panel__my_pitch__{name}.json').write_text(json.dumps(to_wire(synthetic_record())))
    provider=FakeProvider(fixtures)
    monkeypatch.setattr(cli,'build_provider',lambda *a:provider)
    media=tmp_path/'video.mp4';media.write_bytes(b'fake video')
    from tests.panel_fixtures import DURATION_S
    out=demo.run_live(media,DURATION_S,'my_pitch',tmp_path/'new_run',execute=True)
    assert len(provider.calls)==2
    assert (out/'my_pitch/agreement.json').exists()
    demo.run_live(media,DURATION_S,'my_pitch',out,execute=True)
    assert len(provider.calls)==2  # existing cache reused
