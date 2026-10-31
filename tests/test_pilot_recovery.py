from types import SimpleNamespace
from pipeline.config import ModelConfig
from scripts.experiments.run_structured_pilot import BudgetRecoveryProvider


def test_medium_recovery_is_isolated_and_preserves_output_limit():
    calls=[]
    original=SimpleNamespace(files=object(),models=SimpleNamespace(generate_content=lambda **kw:calls.append(kw) or 'ok'))
    provider=SimpleNamespace(client=original)
    provider.generate=lambda req:provider.client.models.generate_content(config={'max_output_tokens':req.model.max_output_tokens,'temperature':req.model.temperature})
    wrapped=BudgetRecoveryProvider(provider)
    request=SimpleNamespace(unit_key='failed_call_medium',model=ModelConfig('gemini-3.1-pro-preview',0,32768))
    assert wrapped.generate(request)=='ok'
    assert calls[0]['config']=={'max_output_tokens':32768,'temperature':0,'thinking_config':{'thinking_level':'medium'}}
    assert provider.client is original


def test_medium_recovery_restores_client_after_failure():
    original=SimpleNamespace(files=object(),models=object())
    provider=SimpleNamespace(client=original)
    def fail(req):raise ValueError('provider failure')
    provider.generate=fail
    import pytest
    with pytest.raises(ValueError):BudgetRecoveryProvider(provider).generate(SimpleNamespace(unit_key='failed_call_medium'))
    assert provider.client is original


def test_prepared_ids_do_not_overwrite_qa_turns():
    from scripts.experiments.run_structured_pilot import namespace_prepared_turns
    original=[{'turn_idx':1,'start_s':15,'end_s':104}]
    prepared,mapping=namespace_prepared_turns(original,{1,2,3})
    assert prepared[0]['turn_idx'] not in {1,2,3}
    assert mapping[1]==prepared[0]['turn_idx']
    assert prepared[0]['start_s']==15 and prepared[0]['end_s']==104
    assert original[0]['turn_idx']==1
    assert namespace_prepared_turns(original,{2,3})==(original,{})


def _pilot_with(tmp_path, monkeypatch, outcomes):
    """A Pilot whose base call follows `outcomes` {key: 'ok' | error message}; records the keys called."""
    import json
    from scripts.experiments import run_structured_pilot as rsp
    (tmp_path / 'results').mkdir(); (tmp_path / 'raw').mkdir()
    called = []
    def base(self, key, prompt, schema, media, info):
        called.append(key)
        if (tmp_path / 'results' / f'{key}.json').exists() or outcomes.get(key) == 'ok':
            return {'key': key}
        (tmp_path / 'raw' / f'{key}__attempt1_error.json').write_text(json.dumps({'message': outcomes[key]}))
        raise RuntimeError(outcomes[key])
    monkeypatch.setattr(rsp.StructuredExperiment, 'call', base)
    p = object.__new__(rsp.Pilot); p.run = tmp_path
    return p, called


def test_recovery_chain_runs_within_one_run(tmp_path, monkeypatch):
    # MAX_TOKENS, then the 64k retry hits the deadline: medium thinking in the SAME run (was: next run)
    p, called = _pilot_with(tmp_path, monkeypatch, {'r': 'response did not finish cleanly (MAX_TOKENS)',
                                                    'r_budget64': '504 DEADLINE_EXCEEDED', 'r_medium': 'ok'})
    assert p.call('r', '', None, None, {}) == {'key': 'r_medium'} and called == ['r', 'r_budget64', 'r_medium']


def test_read_timeout_goes_to_medium(tmp_path, monkeypatch):
    p, called = _pilot_with(tmp_path, monkeypatch, {'s': 'The read operation timed out', 's_medium': 'ok'})
    assert p.call('s', '', None, None, {}) == {'key': 's_medium'} and called == ['s', 's_medium']


def test_rerun_uses_cached_recovery_result_without_paying_for_the_default_call(tmp_path, monkeypatch):
    p, called = _pilot_with(tmp_path, monkeypatch, {})
    (tmp_path / 'results' / 't_budget64.json').write_text('{}')
    p.call('t', '', None, None, {}); assert called == ['t_budget64']


def test_rerun_after_timeouts_goes_straight_to_medium(tmp_path, monkeypatch):
    p, called = _pilot_with(tmp_path, monkeypatch, {'u_medium': 'ok'})
    (tmp_path / 'raw' / 'u_budget64__attempt2_error.json').write_text('{"message": "504 DEADLINE_EXCEEDED"}')
    p.call('u', '', None, None, {}); assert called == ['u_medium']


def test_copied_recovery_result_is_not_treated_as_the_default_cache(tmp_path, monkeypatch):
    # discovery copies timeline_budget64.json to timeline.json; a rerun must reuse the budget64 cache
    p, called = _pilot_with(tmp_path, monkeypatch, {})
    (tmp_path / 'results' / 'timeline_budget64.json').write_text('{"key": "timeline_budget64"}')
    (tmp_path / 'results' / 'timeline.json').write_text('{"key": "timeline_budget64"}')
    p.call('timeline', '', None, None, {}); assert called == ['timeline_budget64']
