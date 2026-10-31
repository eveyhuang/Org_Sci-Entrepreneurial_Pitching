#!/usr/bin/env python3
"""Prepare the Firefighter1 coding page; finalize it from the user's exported CSVs/ZIP."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import demo, human_agreement as ha
from pipeline.errors import ConfigError
from pipeline.ingest import sha256_file
from pipeline.instrument import load_release, check_release
from pipeline.stage_b import review_clips as rc, review_page as page
from pipeline.stage_b.panel_books import load_panel_books
from pipeline.stage_b.review_labels import load_labels
from pipeline.stage_b.review_codebook import codebook_model


def page_model(folder):
    m = demo.review_manifest(folder)
    release = load_release()
    books, labels = load_panel_books(release, sync=False), load_labels(release)
    session = demo.read_json(folder / 'exports/clips/export_manifest.json')
    model = page.tasks_page_model(demo.REFERENCE, session, m['coder'], books, labels,
                                 out_dirs={m['video_id']: folder / 'exports'})
    model['codebook'] = codebook_model(release, books, labels, page.on_sheet(model))
    model.update(storage_namespace='firefighter1-demo', sheet_design=m['design_id'],
                 source_url=m['source_url'], heading='Firefighter1 — example annotation',
                 banner='One example, 32 cards. Watch the footage and enter your own answers. '
                        'Download all sheets when finished; saving this HTML does not save your answers. '
                        'The timestamps and optional locating summaries come from the model record.')
    return model


def prepare(folder=demo.REVIEW, coder='human1'):
    folder = Path(folder)
    rp = ha.review_panel()
    coder = rp.reviewer_name(coder)
    release = load_release()
    if check_release(release):
        raise ConfigError('Resolve instrument validation errors before preparing the sheet')
    books = load_panel_books(release, sync=False)
    if (folder / 'manifest.json').exists():
        m = demo.review_manifest(folder)
        if m['coder'] != coder:
            raise ConfigError('Existing demo has a different coder; use a separate folder')
    else:
        exports = folder / 'exports'
        args = ['export', '--run', str(demo.REFERENCE), '--video-ids', demo.EXAMPLE_ID,
                '--preset', 'exchange', '--blind', '--qa-per-video', '4', '--stage-a-per-video', '0',
                '--venture-per-video', '0', '--exchanges-per-video', '0', '--seed', '0', '--out', str(exports)]
        if rp.main(args):
            raise ConfigError('Review export failed')
        entry = demo.read_json(exports / 'export_manifest.json')['videos'][demo.EXAMPLE_ID]
        session = rc.build_manifest(demo.REFERENCE, [demo.EXAMPLE_ID], [coder], 0, books,
                                    release.release_id, {demo.EXAMPLE_ID: entry['primary_sha256']},
                                    include_pairs=False)
        clips = exports / 'clips'
        clips.mkdir(parents=True, exist_ok=True)
        rc.write_sheet(clips / 'clips.csv', session['rows'])
        demo.write_json(clips / 'export_manifest.json', session)
        vdir = demo.REFERENCE / demo.EXAMPLE_ID
        research = [p for sub in ('results', 'bundle', 'prompts') for p in (vdir / sub).glob('*') if p.is_file()]
        research += [vdir / 'experiment.json', vdir / 'agreement.json']
        m = {'status': 'awaiting_annotation', 'video_id': demo.EXAMPLE_ID, 'source_url': demo.SOURCE_URL,
             'source_url_provenance': 'Shark Tank Global YouTube title match, checked 2026-10-29',
             'coder': coder, 'release_id': release.release_id, 'pass': entry['pass'],
             'design_id': session['design_id'], 'seed': 0, 'sample': entry['sample'],
             'card_count': len(entry['rows']) + len(session['rows']),
             'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'research_files': {str(p.relative_to(ROOT)): sha256_file(p) for p in research},
             'export_files': {str(p.relative_to(folder)): sha256_file(p) for p in exports.rglob('*') if p.is_file()}}
        demo.write_json(folder / 'manifest.json', m)
    model = page_model(folder)
    (folder / 'pages').mkdir(exist_ok=True)
    target = folder / 'pages' / f'{coder}__tasks.html'
    target.write_text(page.render_page(model))
    return target


def complete(submission, folder=demo.REVIEW):
    folder = Path(folder)
    rp = ha.review_panel()
    m = demo.review_manifest(folder)
    if (folder / 'answers').exists() or m['status'] == 'completed':
        raise ConfigError('A submission is already preserved here; use a separate example folder for a revision')
    # Validate completely before preserving the submission or changing completion status.
    with rp._sheets_at(Path(submission)) as source:
        result = demo.load_human_review(folder, source)
        if result is None:
            raise ConfigError('No submitted CSV sheets found')
        unanswered = [p for p, r in result['submitted'].items()
                      if not (r.get('human_value') or r.get('resolution'))]
        if unanswered:
            raise ConfigError(f'{len(unanswered)} cards are still blank; answer or mark unclear/structural before finalizing')
        model = page_model(folder)
        model.update(read_only=True, completed_answers=demo.completed_state(model, result['submitted']),
                     heading='Firefighter1 — completed human annotation',
                     banner=f"Completed by {m['coder']}. A worked example of human–model comparison, not an independent validation sample.")
        model['answer_files'] = [{'name': p.name, 'href': '../answers/' + p.name} for p in sorted(source.glob('*.csv'))]
        html = page.render_page(model)
        answers = folder / 'answers'
        answers.mkdir()
        for p in source.glob('*.csv'):
            shutil.copy2(p, answers / p.name)
        if Path(submission).is_file():
            original = folder / 'original_submission'
            original.mkdir()
            shutil.copy2(submission, original / Path(submission).name)
    (folder / 'pages' / f"{m['coder']}__completed.html").write_text(html)
    write_derived(folder, result)
    m.update(status='completed', answer_files={str(p.relative_to(folder)): sha256_file(p) for p in answers.glob('*.csv')})
    m['completed_files'] = {str(p.relative_to(folder)): sha256_file(p)
                            for p in [folder / 'pages' / f"{m['coder']}__completed.html", *(folder / 'derived').glob('*')]}
    demo.write_json(folder / 'manifest.json', m)
    return folder / 'pages' / f"{m['coder']}__completed.html"


def write_derived(folder, result):
    derived = folder / 'derived'
    derived.mkdir(exist_ok=True)
    result['cards'].to_csv(derived / 'item_comparison.csv', index=False)
    # pandas normalizes missing values to JSON null for portable derived output.
    demo.write_json(derived / 'agreement.json', json.loads(result['summary'].to_json(orient='records')))
    demo.write_json(derived / 'reporting.json', result['pooled']['reporting'])
    demo.write_json(derived / 'decision_turn.json', result['decision_turn'])
    demo.human_agreement_table(result, include_codes=True).to_csv(derived / 'agreement_compact.csv', index=False)


def refresh(folder=demo.REVIEW):
    """Recalculate derived comparisons after a method change; preserve the submitted CSVs and completed HTML."""
    folder = Path(folder)
    m = demo.review_manifest(folder)
    if m['status'] != 'completed':
        raise ConfigError('Complete the example before refreshing its derived comparisons')
    result = demo.load_human_review(folder)
    write_derived(folder, result)
    m['completed_files'].update({str(p.relative_to(folder)): sha256_file(p)
                                 for p in (folder / 'derived').glob('*')})
    demo.write_json(folder / 'manifest.json', m)
    return folder / 'derived'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'complete', 'refresh'])
    p.add_argument('--folder', type=Path, default=demo.REVIEW)
    p.add_argument('--coder', default='human1')
    p.add_argument('--submission', type=Path)
    args = p.parse_args(argv)
    try:
        if args.action == 'complete' and not args.submission:
            p.error('complete requires --submission ZIP_OR_CSV_FOLDER')
        if args.action == 'prepare':
            target = prepare(args.folder, args.coder)
        elif args.action == 'refresh':
            target = refresh(args.folder)
        else:
            target = complete(args.submission, args.folder)
        print(target)
    except (ConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
