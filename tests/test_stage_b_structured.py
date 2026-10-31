import copy
import pytest
from pipeline.stage_b.structured import (Turn,TimelineReview,Episode,InteractionMap,Issue,
    freeze_bundle,assert_frozen,validate_codes,PitchResult,validate_pitch)

ROSTER=[{'speaker':'speaker-2','role':'founder'},{'speaker':'speaker-4','role':'investor'}]

def fixture():
    tl=TimelineReview(turns=[Turn(turn_idx=i,speaker=s,start_s=a,end_s=b,tag='test') for i,s,a,b in
        [(13,'speaker-4',251,254),(14,'speaker-2',254,256),(15,'speaker-4',255,260),(101,'speaker-2',260,275)]],issues=[],review_notes='source-reviewed test')
    ep=Episode(episode_id='patent',topic='patent',probing_speaker='speaker-4',turn_refs=[13,14,15,101],opening_probe=[13],initial_answer=[14],first_uptake=[15],uptake_status='observed',later_exchanges=[],resolution='unclear',cut=False,rationale='test')
    m=InteractionMap(episodes=[ep],excluded=[],issues=[],review_notes='test')
    return ep,tl,m,freeze_bundle(m,tl,ROSTER,'source',251,278,'test')

def test_later_challenge_cannot_set_opening_severity():
    ep,_,_,bundle=fixture()
    rating={'PR-SEV':{'value':3,'uncodable':False,'evidence':'copying challenge','turn_refs':[15]}}
    assert validate_codes(rating,ep,bundle)==['PR-SEV: out-of-scope evidence refs']
    rating['PR-SEV'].update(value=1,turn_refs=[13]);assert not validate_codes(rating,ep,bundle)

def test_later_analogy_cannot_replace_initial_answer():
    ep,_,_,bundle=fixture()
    r={'PREP-SUBS':{'value':1,'uncodable':False,'evidence':'analogy','turn_refs':[101]}}
    assert validate_codes(r,ep,bundle)

def test_stable_ids_overlap_and_stale_freeze():
    _,tl,_,bundle=fixture();assert_frozen(bundle)
    tl.turns[-1].tag='repaired'
    with pytest.raises(ValueError,match='stale freeze'):assert_frozen(bundle,timeline=tl)

def test_unresolved_repair_blocks_freeze():
    _,tl,m,_=fixture();tl.issues=[Issue(start_s=254,end_s=256,status='needs_repair',detail='missing turn',affected_codes=['RAP-INTERRUPT'])]
    with pytest.raises(ValueError,match='repair'):freeze_bundle(m,tl,ROSTER,'s',251,278,'test')

def test_unobservable_codes_require_missingness():
    ep,tl,m,_=fixture();tl.issues=[Issue(start_s=254,end_s=256,status='unobservable',detail='unresolvable overlap',affected_codes=['RAP-INTERRUPT'])]
    b=freeze_bundle(m,tl,ROSTER,'s',251,278,'test')
    assert validate_codes({'RAP-INTERRUPT':{'value':0,'uncodable':False,'evidence':'none','turn_refs':[14,15]}},ep,b)

def test_team_absence_not_false():
    ep,_,_,b=fixture()
    assert validate_codes({'TM-RESCUE':{'value':False,'uncodable':False,'evidence':'solo','turn_refs':[14]}},ep,b)
    assert not validate_codes({'TM-RESCUE':{'value':None,'uncodable':True,'evidence':'solo','turn_refs':[]}},ep,b)

def test_unassigned_turn_blocks_freeze():
    _,tl,m,_=fixture();m.episodes[0].turn_refs.remove(101)
    with pytest.raises(ValueError,match='assigned'):freeze_bundle(m,tl,ROSTER,'s',251,278,'test')

def test_unknown_or_wrong_role_component_blocks_freeze():
    _,tl,m,_=fixture();m.episodes[0].initial_answer=[15]
    with pytest.raises(ValueError,match='founder'):freeze_bundle(m,tl,ROSTER,'s',251,278,'test')

def pitch():
    claim={'turn_ref':1,'time_s':30,'claim':'inventory','metric':'units','period':'launch day','units':'units','scope':'initial stock'}
    second={**claim,'turn_ref':14,'period':'eventual fulfillment'}
    return PitchResult(value=1,uncodable=False,coverage_complete=True,inspected_turn_refs=[1,14],comparable_claims_present=True,comparisons=[{'first':claim,'second':second,'matching_scope_established':True,'incompatibility':'incompatible','reconciliation':'none','counted':True}],evidence='test',limitations=[])

def test_pitch_same_metric_different_period_rejected():
    assert 'counted pair has unmatched period' in validate_pitch(pitch(),{1,14},{1:'prepared',14:'ep'})

def test_pitch_excluded_source_rejected():
    assert 'ineligible claim' in validate_pitch(pitch(),{1},{1:'prepared'})

def test_incomplete_pitch_is_not_zero():
    p=pitch();p.coverage_complete=False;p.value=0;p.comparisons=[]
    assert 'incomplete coverage requires uncodable' in validate_pitch(p,{1,14},{1:'prepared',14:'ep'})

def test_cannot_skip_first_investor_intervention():
    _,tl,m,_=fixture()
    tl.turns.append(Turn(turn_idx=102,speaker='speaker-4',start_s=276,end_s=277,tag='later follow-up'))
    m.episodes[0].turn_refs.append(102);m.episodes[0].first_uptake=[102]
    with pytest.raises(ValueError,match='skips'):freeze_bundle(m,tl,ROSTER,'s',251,278,'test')

def test_pitch_investor_claim_and_wrong_time_rejected():
    p=pitch()
    errors=validate_pitch(p,{1,14},{1:'prepared',14:'ep'},founder_refs={1},turns={1:{'start_s':50,'end_s':60}})
    assert 'claim is not from a founder turn' in errors
    assert 'claim time outside referenced turn' in errors

def test_later_social_exchange_not_initial_backchannel():
    _,tl,m,_=fixture()
    tl.turns.append(Turn(turn_idx=300,speaker='speaker-4',start_s=276,end_s=277,tag='later listener'))
    m.episodes[0].turn_refs.append(300);m.episodes[0].minimal_backchannels=[300]
    with pytest.raises(ValueError,match='backchannel outside'):freeze_bundle(m,tl,ROSTER,'s',251,278,'test')

def test_probe_alone_cannot_support_response_code():
    ep,_,_,b=fixture()
    errors=validate_codes({'RS-EVID':{'value':False,'uncodable':False,'evidence':'no evidence','turn_refs':[13]}},ep,b)
    assert 'RS-EVID: must cite initial answer' in errors

def test_visibility_cannot_double_count_or_use_context():
    from pipeline.stage_b.structured import VisibilityAudit,visibility_gate
    a=VisibilityAudit(speaker='speaker-3',coverage_reviewed=True,intervals=[{'start_s':115,'end_s':120,'linkage_evidence':'linked'}]*4,limitations=[])
    assert not visibility_gate(a,'speaker-3',115,149)
    a.intervals[0].start_s=105
    assert not visibility_gate(a,'speaker-3',115,149)
    a.intervals=[]
    assert not visibility_gate(a,'speaker-3',115,149)

def test_comparable_claims_require_inspectable_pair():
    p=pitch();p.comparisons=[];p.value=0
    assert 'comparable claims require locatable pair evidence' in validate_pitch(p,{1,14},{1:'prepared',14:'ep'})
