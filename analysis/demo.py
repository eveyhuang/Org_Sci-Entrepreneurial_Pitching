"""Assembly for Demo.ipynb; measurements, agreement and fitting stay in their owning modules."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path

import pandas as pd

from analysis import human_agreement as ha, panel_explorer as px
from analysis.panel_dataset import load_run
from analysis.panel_features import pitch_features, history_table
from pipeline.errors import ConfigError
from pipeline.ingest import sha256_file
from pipeline.instrument import load_release, check_release
from pipeline.stage_b.panel_books import load_panel_books, render_panel_prompt
from pipeline.stage_b.review_apply import Tolerances, agrees, within_one, is_unsure, pooled_summary
from pipeline.stage_b.review_items import normalize_path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'runs/full-corpus-100'
EXAMPLE_ID = 'st_8a6fd25840'
SOURCE_URL = 'https://www.youtube.com/watch?v=dxsV1p3Hue0'
REVIEW = ROOT / 'examples/firefighter1_review'
CUSTOM_MANIFEST = ROOT / 'examples/demo_custom_instrument/codebooks/current_instrument.yaml'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def verify_files(files, root=ROOT):
    wrong = [name for name, digest in files.items()
             if not (Path(root) / name).is_file() or sha256_file(Path(root) / name) != digest]
    if wrong:
        raise ConfigError('Saved inputs differ or are missing: ' + ', '.join(wrong[:8]))
    return len(files)


def load_reference():
    verify_files(read_json(ROOT / 'examples/demo_release.json')['files'])
    m = read_json(REFERENCE / 'analysis/all100/manifest.json')
    verify_files(m['inputs'], REFERENCE)
    saved = {name: pd.read_csv(REFERENCE / 'analysis/all100' / f'{name}.csv')
             for name in ('features', 'metrics', 'deltas', 'predictions', 'single_feature')}
    # Guard load_run's dynamic best-pass rule against later human adjudications.
    for row in saved['features'].itertuples():
        if px.analysis_pass(REFERENCE / row.video_id) != row.source:
            raise ConfigError(f'{row.video_id}: selected pass changed since the reported analysis')
    data = load_run(REFERENCE, m['video_ids'])
    features = pitch_features(data)
    compare_export(features, saved['features'], 'features')
    return data, features, saved, m


def compare_export(actual, saved, name, atol=1e-6):
    a, b = actual.reset_index(drop=True).copy(), saved.reset_index(drop=True).copy()
    for col in b:
        if col in a and (a[col].dtype == object or b[col].dtype == object):
            a[col], b[col] = a[col].fillna(''), b[col].fillna('')
    try:
        pd.testing.assert_frame_equal(a[b.columns], b, check_dtype=False, check_exact=False,
                                      rtol=1e-8, atol=atol)
    except (AssertionError, KeyError) as exc:
        raise ConfigError(f'{name} differs from its saved export: {exc}') from exc


def encounter(run, video_id, pass_name=None):
    vdir = Path(run) / video_id
    result, source, note = px.load_pass(vdir, pass_name or px.analysis_pass(vdir))
    inst = px.instrument(vdir)
    if inst.source != 'bundle':
        raise ConfigError('Restore the archived instrument bundle for this encounter')
    return vdir, result, inst, source, note


def annotation_record(result, books):
    """Figure 1's compact view of every turn in one pass, with selected codes."""
    record = result['record']
    turns = px.turn_table(result, books, group_by_exchange=False).set_index('Turn')
    names = {}
    for role, label in [('founder', 'founder'), ('investor', 'shark')]:
        actors = [a for a in record['actors'] if a['role'] == role]
        for number, actor in enumerate(actors, 1):
            names[actor['actor_id']] = (label if role == 'founder' and len(actors) == 1
                                       else f'{label} {number}')
    decisions = {}
    decision_labels = {'exit': 'out', 'offer': 'offer', 'joins_offer': 'joins the offer',
                       'founder_counters': 'counters the offer', 'founder_accepts': 'accepts the offer'}
    for decision in record['decisions']:
        decisions.setdefault(decision['turn_id'], []).append(
            decision_labels.get(decision['kind'], decision['kind'].replace('_', ' ')))
    rows = []
    for turn in sorted(record['turns'], key=lambda t: (t['start_s'], t['turn_id'])):
        values = {c['code']: c for c in turn['codes']}
        codes = []
        if turn['function'] == 'presents':
            if record.get('stage_a'):
                codes.append('delivery rated')
            if record.get('prepared'):
                codes.append('preparedness rated')
        for code, label in [('PR-SEV', 'severity'), ('PREP-SUBS', 'substance'),
                            ('PREP-COH', 'coherence')]:
            if code in values:
                item = values[code]
                value = item['value']
                if value is None:
                    value = f"missing ({item.get('missing') or 'unspecified'})"
                elif code == 'PR-SEV' and str(value) == '3':
                    value = '3 (hard)'
                codes.append(f'{label} {value}')
        if 'RS-EVID' in values:
            item = values['RS-EVID']
            support = str(item['value']).lower()
            if support in ('true', '1'):
                codes.append('checkable specific ✓')
            elif support in ('false', '0'):
                codes.append('no specific')
            else:
                codes.append(f"checkable specific: missing ({item.get('missing') or 'unspecified'})")
        codes.extend(decisions.get(turn['turn_id'], []))
        detail = turns.loc[turn['turn_id']]
        concept_frame = ', '.join(str(detail[c]).replace('_', ' ')
                                  for c in ('Concept', 'Frame (derived)') if detail[c])
        rows.append({'time': px.clock(turn['start_s']),
                     'who': names.get(turn['speaker'], turn['speaker']),
                     'turn': turn['function'], 'concept, frame': concept_frame,
                     'codes': ' · '.join(codes)})
    return pd.DataFrame(rows, columns=['time', 'who', 'turn', 'concept, frame', 'codes'])


def support_trace(data, video_id, result):
    turns = history_table(data, video_id)
    answers = turns[(turns.function == 'answer') & turns.used].copy()
    summaries = {t['turn_id']: t['summary'] for t in result['record']['turns']}
    answers['clock'] = answers.apply(lambda t: px.clock_range(t.start_s, t.end_s), axis=1)
    answers['summary'] = answers.turn_id.map(summaries)
    return answers[['turn_id', 'clock', 'RS-EVID', 'summary']].rename(
        columns={'RS-EVID': 'checkable_support (1=yes, 0=no)'})


def held_out_example(features, manifest, video_id, model='C5'):
    from analysis.panel_bottleneck import fit_head, prediction_breakdown
    rows = features[features.any_offer.notna()].copy()
    train, test = rows[rows.video_id != video_id], rows[rows.video_id == video_id]
    if len(test) != 1 or len(train) < 2:
        raise ConfigError('Expected one example and a separate training corpus')
    penalty = None if manifest['penalty'] == 'inner_cv' else float(manifest['penalty'])
    probabilities, detail = fit_head(train, test, 'any_offer', tuple(manifest['models'][model]),
                                    penalty=penalty, penalty_grid=tuple(manifest['penalty_grid']),
                                    not_fitted=tuple(manifest['not_fitted']['columns']))
    fitted = {'status': 'estimated', 'folds': [{'held_out': video_id,
              'train_groups': sorted(train.video_id.unique()), 'models': {model: detail}}]}
    return float(probabilities[0]), prediction_breakdown(fitted, features, model, video_id), fitted


def review_manifest(folder=REVIEW):
    folder = Path(folder)
    m = read_json(folder / 'manifest.json')
    verify_files(m['research_files'])
    verify_files(m['export_files'], folder)
    if m.get('status') == 'completed':
        verify_files(m['answer_files'], folder)
        verify_files(m['completed_files'], folder)
    release = load_release()
    if release.release_id != m['release_id']:
        raise ConfigError('Review instrument release differs from the frozen example')
    for book in release.books.values():
        for path in (book.markdown, book.yaml):
            archived = REFERENCE / m['video_id'] / 'bundle' / path.name
            if not archived.is_file() or sha256_file(path) != sha256_file(archived):
                raise ConfigError(f'{path.name}: review codebook differs from the archived instrument')
    return m


def validate_submission(folder, sheets):
    """Demo-specific identity checks; the existing importer validates values and resolutions."""
    rp = ha.review_panel()
    folder, sheets = Path(folder), Path(sheets)
    m = review_manifest(folder)
    expected = {f"{m['video_id']}__{m['coder']}.csv", f"clips__{m['coder']}.csv"}
    found = {p.name for p in sheets.glob('*.csv')}
    if found != expected:
        raise ConfigError(f'Expected exactly {sorted(expected)}; received {sorted(found)}')
    all_rows = {}
    session = read_json(folder / 'exports/clips/export_manifest.json')
    full = read_json(folder / 'exports/export_manifest.json')['videos'][m['video_id']]
    for name in sorted(expected):
        rows = rp.read_sheet(sheets / name)
        wanted = session['rows'] if name.startswith('clips__') else full['rows']
        paths = [normalize_path(r.get('path')) for r in rows]
        if len(paths) != len(set(paths)) or set(paths) != {r['path'] for r in wanted}:
            raise ConfigError(f'{name}: paths must match the frozen selection exactly, without duplicates')
        for r in rows:
            if r.get('context', '').strip() != f"design {m['design_id']}":
                raise ConfigError(f'{name}: wrong or missing demo design id; download from this demo sheet')
        if name.startswith('clips__'):
            _, problems = rp._session_rows(sheets / name, session)
            if problems:
                raise ConfigError('; '.join(problems))
        all_rows.update({normalize_path(r['path']): r for r in rows})
    return m, all_rows


def load_human_review(folder=REVIEW, sheets=None):
    """Import real human sheets into the existing throwaway run; never adjudicate the corpus."""
    folder = Path(folder)
    sheets = Path(sheets) if sheets is not None else folder / 'answers'
    if not sheets.is_dir() or not list(sheets.glob('*.csv')):
        return None
    m, submitted = validate_submission(folder, sheets)
    rp = ha.review_panel()
    release = load_release()
    if release.release_id != m['release_id']:
        raise ConfigError('Review tooling release differs from the exported demo instrument')
    books = load_panel_books(release, sync=False)
    messages = io.StringIO()
    with contextlib.redirect_stdout(messages), contextlib.redirect_stderr(messages):
        logs, status, agreements, _ = rp.training_imports(REFERENCE, sheets, books,
                                                         training_dir=folder / 'exports')
    if status:
        raise ConfigError('Sheet import failed:\n' + messages.getvalue())
    videos, tolerances, by_reviewer = rp.pool_training(logs, agreements)
    if not by_reviewer:
        raise ConfigError('The sheets contain no human answers yet')
    # Use the current per-kind reporting, whose statistics call pipeline.reliability.agreement.
    # Fail on an incompatible reporting file instead of silently reverting to the older metrics.
    reporting = rp.load_reporting(books)
    pooled = pooled_summary(videos, tolerances, by_reviewer=by_reviewer, strata=None,
                            reporting=reporting)
    data = ha.TrainingAgreement(run=REFERENCE, coders=sorted(by_reviewer), videos=sorted(videos),
                               logs=logs, agreements=agreements, pooled=pooled, books=books,
                               bar=reporting.band_floor(rp.CALIBRATION_BAND), import_failed=False,
                               import_log=messages.getvalue(), reporting=reporting)
    cards = ha.cards(data)
    # Include unanswered selected cards too, rather than losing them from the display.
    selected = read_json(folder / 'exports/export_manifest.json')['videos'][m['video_id']]['rows']
    selected += read_json(folder / 'exports/clips/export_manifest.json')['rows']
    base = pd.DataFrame([{'path': r['path'], 'clock': r.get('clock_start', 'presentation')} for r in selected])
    cards = base.merge(cards, on='path', how='left', validate='one_to_one')
    tol = Tolerances.of(agreements[m['video_id']])
    cards['human_vs_model'] = cards.apply(lambda r: agrees(r.path, r[m['coder']], r.model, tol)
        if pd.notna(r.get(m['coder'])) and pd.notna(r.get('model')) else None, axis=1)
    cards['human_vs_model_within_one'] = cards.apply(lambda r: within_one(r[m['coder']], r.model, r.path)
        if r.get('level') == 'ordinal' and pd.notna(r.get(m['coder'])) and pd.notna(r.get('model')) else None, axis=1)
    summary = ha.agreement_by_code(data)
    summary = summary[[c for c in summary if not c.startswith('coder vs coder')
                       and c not in ('rows (coder pairs)', 'α units', 'reached the bar')]].copy()
    summary.loc[summary['class'] == 'cut', 'code'] = 'First investor decision time'
    by_variable = pooled.get('by_variable', {})
    summary['usable_items'] = summary['key'].map(lambda k: by_variable.get(k, {}).get('n'))
    # One presentation judgement or one decision clock is a direct comparison, not reliability.
    for col in summary:
        if 'α' in col:
            summary.loc[summary['class'] != 'turn_code', col] = None
    # This single clock is a direct difference, not an across-video reliability estimate.
    from pipeline.schemas.stage_b_panel import clock_to_seconds
    from pipeline.stage_b.review_apply import decision_turn_comparison
    context = next((line.get('decision_turn_context') for line in videos[m['video_id']]
                    if line['path'] == 'pitch:first_decision'), None)
    decision_turn = None
    cards['clock_difference_s'] = None
    cards['decision_same_turn'] = None
    cards['decision_clock_within_tolerance'] = None
    cards['agreement_rule'] = 'code_specific'
    for idx, r in cards[cards.path == 'pitch:first_decision'].iterrows():
        human_seconds = model_seconds = None
        if pd.notna(r.get(m['coder'])) and pd.notna(r.get('model')):
            try:
                human_seconds = clock_to_seconds(str(r[m['coder']]))
                model_seconds = clock_to_seconds(str(r.model))
                cards.loc[idx, 'clock_difference_s'] = human_seconds - model_seconds
            except ValueError:
                pass  # e.g. a human explicitly records no decision
        decision_turn = decision_turn_comparison(r.get(m['coder']), r.get('model'), context)
        decision_turn.update(video_id=m['video_id'],
                             rule_provenance='Author requested same-turn agreement after inspecting this example; '
                                             'raw clock differences remain timing diagnostics.')
        cards.loc[idx, 'decision_clock_within_tolerance'] = r.human_vs_model
        cards.loc[idx, 'decision_same_turn'] = decision_turn['agreed']
        cards.loc[idx, 'human_vs_model'] = decision_turn['agreed']
        cards.loc[idx, 'agreement_rule'] = 'same_reference_turn'
    # Both the detailed pipeline report and this display now use the shared turn-identity check.
    summary.loc[summary['class'] == 'cut', 'code'] = 'First investor decision: same turn'
    return {'data': data, 'cards': cards, 'summary': summary, 'submitted': submitted,
            'manifest': m, 'pooled': pooled, 'decision_turn': decision_turn}


def human_agreement_table(human, *, include_codes=False):
    """A compact view of the existing per-kind results for the sampled Q&A codes; no new statistics."""
    from pipeline.stage_b.review_report import paper_rows
    reporting = human['data'].reporting
    variables = human['cards'].loc[human['cards']['class'] == 'turn_code', 'variable'].dropna()
    codes = {reporting.aliases.get(v, v) for v in variables}
    codes.update(name for name, entry in reporting.derived.items() if entry['from'] in codes)
    rows = []
    for row in paper_rows(human['pooled'], reporting):
        if row['field'] not in codes or row['missing']:
            continue
        figures = human['pooled']['reporting'][row['field']]['figures']
        n = row['n']
        def count(name):
            value = figures.get(name, {}).get('value')
            return '—' if value is None else f'{round(value * n)}/{n}'
        def number(name):
            value = figures.get(name, {}).get('value')
            return '—' if value is None else f'{value:.2f}'
        ac = 'ac2' if row['kind'] == 'ordinal' else 'ac1'
        # PR-FRAME is derived by the reporting specification; the other names are in the codebook.
        entry = human['data'].books.panel.by_id(row['field']) if row['field'] != 'PR-FRAME' else None
        name = (entry.name or entry.id) if entry is not None else 'Question frame (from concept)'
        rows.append({'Measure': name, **({'Code': row['field']} if include_codes else {}), 'Exact': count('exact'),
                     'Within one': count('within_one'), 'α': number('alpha'),
                     'AC1/AC2': f'{ac.upper()} {number(ac)}',
                     'Dominant category': 'yes' if 'prevalence' in row['flags'] else 'no'})
    return pd.DataFrame(rows)


def review_sheet_path(completed=True, folder=REVIEW):
    """Locate a verified review artifact; do not launch a browser during Run All."""
    m = review_manifest(folder)
    kind = 'completed' if completed else 'tasks'
    path = Path(folder) / 'pages' / f"{m['coder']}__{kind}.html"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def review_sheet_preview(completed=True, folder=REVIEW):
    """Render the very same page payload without JavaScript, for notebook output renderers."""
    from pipeline.stage_b.review_page import render_notebook_preview
    source = review_sheet_path(completed, folder).read_text(encoding='utf-8')
    model, _ = json.JSONDecoder().raw_decode(source.split('const PAGE = ', 1)[1])
    return render_notebook_preview(model)


def open_review_sheet(completed=True, folder=REVIEW):
    """Explicit user action for a local kernel: open the full sheet in the system browser."""
    import webbrowser
    path = review_sheet_path(completed, folder)
    if not webbrowser.open(path.as_uri(), new=2):
        raise RuntimeError(f'Open this file in your browser: {path}')
    return path


def completed_state(model, submitted):
    """Map the existing CSV values into the existing browser card state; no inferred answers."""
    state = {}
    for vd in model['videos']:
        state[vd['video_id']] = {}
        for section in vd['sections']:
            for card in section['cards']:
                row = submitted[card['path']]
                value = row.get('human_value') or ''
                note = row.get('note') or ''
                entry = {'value': value, 'note': note, 'unsure': is_unsure(row),
                         'unclear': row.get('resolution') == 'unclear',
                         'structural': row.get('resolution') == 'structural'}
                if card['input'] == 'ask' and ';' in value:
                    entry['usd'], entry['pct'] = value.split(';', 1)
                state[vd['video_id']][card['path']] = entry
    return state


def instrument_preview(manifest=None):
    release = load_release(manifest)
    problems = check_release(release)
    if problems:
        raise ConfigError('Instrument is not ready: ' + '; '.join(problems))
    books = load_panel_books(release, sync=False)
    reference = load_panel_books(load_release(), sync=False)
    # Same record/feature contract for the definition-only example.
    for attr in ('stage_a', 'panel', 'venture'):
        book, base = getattr(books, attr), getattr(reference, attr)
        own = {e.id: e for e in book.entries}
        old = {e.id: e for e in base.entries}
        if own.keys() != old.keys():
            raise ConfigError('Changed code IDs require feature/schema mapping before this demo can use them')
        for key in old:
            for field in ('type', 'values', 'scale', 'applies_to', 'unit', 'side'):
                if getattr(own[key], field, None) != getattr(old[key], field, None):
                    raise ConfigError(f'{key}: changed {field}; adapt the feature/schema mapping')
    return release, books, render_panel_prompt(books, release.templates_dir)


def run_live(video_path, duration_s, video_id, output_dir, *, manifest=None, execute=False):
    """Use the existing command runner, with an explicit opt-in and a separate run."""
    from scripts.experiments import run_panel_pilot as cli
    out, path = Path(output_dir).resolve(), Path(video_path).expanduser().resolve()
    if out == REFERENCE or REFERENCE in out.parents or out in REFERENCE.parents:
        raise ConfigError('Choose a separate output directory outside the reference run')
    if not path.is_file() or float(duration_s) <= 0:
        raise ConfigError('Provide an existing local video and its positive duration in seconds')
    instrument_preview(manifest)
    with tempfile.TemporaryDirectory() as tmp:
        selection = Path(tmp) / 'selection.json'
        write_json(selection, {'videos': [{'video_id': video_id, 'path': str(path), 'duration_s': float(duration_s)}]})
        args = ['--selection', str(selection), '--runs-root', str(out.parent), '--run-name', out.name]
        if manifest:
            args += ['--instrument', str(manifest)]
        if cli.main(args + ['--phase', 'preflight']):
            raise ConfigError('Annotation preflight failed')
        if execute and cli.main(args + ['--phase', 'annotate', '--execute']):
            raise ConfigError('Live annotation failed; inspect its run, no saved-record substitution was made')
    return out


def recompute_reference(manifest, workers=1):
    from scripts.analysis.analyze_panel import compute, assemble
    sensitivity = manifest.get('h9_exchange_sensitivity')
    analysis = compute(REFERENCE, manifest['video_ids'],
        penalty=None if manifest['penalty'] == 'inner_cv' else float(manifest['penalty']),
        n_boot=manifest['n_boot'], seed=manifest['seed'], n_perm=manifest['n_perm'],
        single_draws=manifest['single_draws'], workers=workers,
        skip_h9=manifest['h9'] != 'run', with_shark_level=manifest['with_shark_level'],
        h9_exchange_outcomes=bool(sensitivity), h9_sensitivity_draws=sensitivity['draws'] if sensitivity else 499)
    return analysis, assemble(analysis)
