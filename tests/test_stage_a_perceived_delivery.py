"""The three-item Stage A instrument is live again and still reproduces its archived prompt;
the ten-item book of v3-dev8 reproduces from its archive."""
from pathlib import Path
import json

import pytest
import yaml
from pydantic import ValidationError

from pipeline.codebook import load_codebook
from pipeline.codebook_md import md_to_entries
from pipeline.config import load_config
from pipeline.instrument import load_release
from pipeline.prompts import render_prompt, prompt_hash
from pipeline.schemas.stage_a import STAGE_A_CONSTRUCTS
from pipeline.schemas.stage_a_perceived_delivery import (
    DeliveryRating, StageADeliveryRecord, STAGE_A_DELIVERY_CODES,
)
from pipeline.stage_b.panel_books import load_panel_books, render_panel_prompt
from tests.conftest import needs_ffmpeg

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / 'docs/codebooks/archive/stage_a_perceived_delivery_v2'
BOOK = ARCHIVE / 'stage_a_perceived_delivery.yaml'
TEMPLATE = 'stage_a_perceived_delivery.j2'
RECORDED_HASH = 'e96edfd4156b6321'   # runs/stage_a_delivery_v2_20260925_012509/manifest.json
V3DEV8 = REPO / 'docs/codebooks/archive/stage_b_panel_v3dev8'


def config_with_local_corpus(filename, tmp_path):
    data = yaml.safe_load((REPO / 'configs' / filename).read_text())
    data['corpus']['video_dir'] = str(tmp_path)
    metadata = tmp_path / 'metadata.csv'
    metadata.write_text('video_id,url\n')
    data['corpus']['metadata_csv'] = str(metadata)
    data['corpus'].pop('dev_set', None)
    for section in [data, *data['passes']]:
        if section.get('codebook'):
            section['codebook'] = str((REPO / 'configs' / section['codebook']).resolve())
    if data.get('prompt_dir'):
        data['prompt_dir'] = str((REPO / 'configs' / data['prompt_dir']).resolve())
    path = tmp_path / filename
    path.write_text(yaml.safe_dump(data))
    return load_config(path)


def test_archived_protocol_and_codebook_render_the_recorded_prompt():
    book = load_codebook(BOOK)
    assert book.ids == STAGE_A_DELIVERY_CODES
    assert tuple(StageADeliveryRecord.model_fields) == (*book.ids, 'overall_notes')
    prompt = render_prompt(TEMPLATE, codebook=book, prompt_dir=ARCHIVE)
    protocol = (ARCHIVE / 'stage_a_perceived_delivery_protocol.md').read_text()
    assert protocol.rstrip() in prompt
    for entry in book.entries:
        assert entry.definition in prompt and entry.provenance not in prompt
    assert prompt_hash(prompt) == RECORDED_HASH


def test_archived_config_reproduces_the_recorded_prompt(tmp_path):
    cfg = config_with_local_corpus('archived_stage_a_perceived_delivery.yaml', tmp_path)
    target = cfg.passes[0]
    assert target.prompt == TEMPLATE and cfg.codebook_for(target) == BOOK
    prompt = render_prompt(target.prompt, codebook=load_codebook(cfg.codebook_for(target)), prompt_dir=cfg.prompt_dir)
    assert prompt_hash(prompt) == RECORDED_HASH


def test_current_stage_a_default_is_the_three_item_instrument(tmp_path):
    cfg = config_with_local_corpus('stage_a.yaml', tmp_path)
    target = cfg.passes[0]
    assert target.prompt == 'stage_a_perceived_delivery.j2'
    assert target.schema.endswith('stage_a_perceived_delivery.StageADeliveryRecord')
    assert cfg.codebook_for(target).name == 'stage_a_perceived_delivery.yaml'
    assert target.k == 5
    assert cfg.historical is False
    assert load_codebook(cfg.codebook_for(target)).ids == STAGE_A_DELIVERY_CODES


def test_live_stage_a_book_is_the_archived_text_plus_provenance_note():
    live = md_to_entries((REPO / 'pipeline/codebooks/stage_a_perceived_delivery.md').read_text())
    archived = md_to_entries((ARCHIVE / 'stage_a_perceived_delivery.md').read_text())
    for a, b in zip(live, archived, strict=True):
        assert {k: v for k, v in a.items() if k != 'provenance'} == {k: v for k, v in b.items() if k != 'provenance'}
        assert a['provenance'].startswith(b['provenance'])
        assert 'Restored 2026-10-20' in a['provenance']


def test_protocol_is_in_both_prompts_and_the_archived_hash_is_unchanged(tmp_path):
    protocol = (REPO / 'pipeline/prompts/stage_a_perceived_delivery_protocol.md').read_text()
    assert protocol == (ARCHIVE / 'stage_a_perceived_delivery_protocol.md').read_text()
    release = load_release()
    books = load_panel_books(release)
    assert protocol.strip() in render_panel_prompt(books, release.templates_dir)
    # The standalone prompt renders the live template, protocol and book from the live prompt dir; provenance is
    # never rendered, so it reproduces the archived prompt exactly.
    cfg = config_with_local_corpus('stage_a.yaml', tmp_path)
    target = cfg.passes[0]
    prompt = render_prompt(target.prompt, codebook=load_codebook(cfg.codebook_for(target)), prompt_dir=cfg.prompt_dir)
    assert protocol.rstrip() in prompt
    assert prompt_hash(prompt) == RECORDED_HASH


def test_archived_ten_item_config_reproduces_the_v3dev8_standalone_book(tmp_path):
    cfg = config_with_local_corpus('archived_stage_a_pitch_factor.yaml', tmp_path)
    target = cfg.passes[0]
    assert cfg.historical is True
    assert cfg.prompt_dir == V3DEV8
    assert target.prompt == 'stage_a_holistic.j2' and target.schema.endswith('stage_a.StageARecord')
    assert cfg.codebook_for(target) == V3DEV8 / 'stage_a_pitch_factor.yaml'
    book = load_codebook(cfg.codebook_for(target), sync=False)
    assert book.ids == STAGE_A_CONSTRUCTS
    prompt = render_prompt(target.prompt, codebook=book, prompt_dir=cfg.prompt_dir)
    assert all(code in prompt for code in STAGE_A_CONSTRUCTS)


def test_archived_stage_b_v1_configs_select_the_archived_three_item_book(tmp_path):
    for filename in ('archived_stage_b_v1.yaml', 'archived_stage_b_v1_dev.yaml'):
        cfg = config_with_local_corpus(filename, tmp_path)
        prepared = cfg.get_pass('prepared')
        assert prepared.prompt == TEMPLATE and cfg.codebook_for(prepared) == BOOK
        assert cfg.prompt_dir == REPO / 'docs/codebooks/archive/stage_b_v1_episode'
        assert cfg.historical is True


def test_rating_requires_real_score_or_missingness_and_locatable_evidence():
    assert DeliveryRating(value=1, uncodable=False, evidence='0:12 Flat delivery.').value == 1
    assert DeliveryRating(value=None, uncodable=True, evidence='No usable audio.').value is None
    for value, missing, evidence in [
        (None, False, '0:12 Flat.'), (2, True, 'No audio.'),
        (3, False, 'Generally positive.'), (3, False, '  '),
        (6, False, '0:12 Positive.'), (3.5, False, '0:12 Positive.'),
        (True, False, '0:12 Positive.'),
    ]:
        with pytest.raises(ValidationError):
            DeliveryRating(value=value, uncodable=missing, evidence=evidence)


@needs_ffmpeg
def test_archived_three_item_instrument_still_runs_end_to_end_on_the_fake_provider(tmp_path):
    from tests.test_cli_end_to_end import build_project, run_cli
    import pandas as pd

    config, fixtures = build_project(tmp_path)
    data = yaml.safe_load(config.read_text())
    data['codebook'] = str(BOOK)
    data['prompt_dir'] = str(ARCHIVE)
    data['historical'] = True
    data['passes'][0]['prompt'] = TEMPLATE
    data['passes'][0]['schema'] = 'pipeline.schemas.stage_a_perceived_delivery.StageADeliveryRecord'
    config.write_text(yaml.safe_dump(data))
    for video in ('abc', 'def'):
        record = {key: {'value': 3, 'uncodable': False, 'evidence': '0:01 Engaged delivery.'}
                  for key in STAGE_A_DELIVERY_CODES}
        (fixtures / f'holistic__{video}.json').write_text(json.dumps(record | {'overall_notes': ''}))
    run_cli('ingest.py', '--config', str(config), cwd=tmp_path)
    run_cli('annotate.py', '--config', str(config), '--provider', 'fake', '--fixtures', str(fixtures), cwd=tmp_path)
    run_cli('aggregate.py', '--config', str(config), cwd=tmp_path)
    run_dir = next((tmp_path / 'runs').glob('e2e_*'))
    table = pd.read_parquet(run_dir / 'tables/annotations.parquet')
    assert set(table.code_id) == set(STAGE_A_DELIVERY_CODES)
    assert len(table) == 12  # two pitches, two replicates, three judgments
    assert table.confidence.isna().all()
