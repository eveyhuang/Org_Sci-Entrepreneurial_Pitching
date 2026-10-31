import copy
import pytest
from pipeline.stage_b.exchanges import *
from pipeline.stage_b.structured import Turn

def fixture():
 roster=[{'speaker':'i1','role':'investor'},{'speaker':'i2','role':'investor'},{'speaker':'f','role':'founder'}]
 timeline=TimelineReview(turns=[Turn(turn_idx=i,speaker=s,start_s=a,end_s=z,tag='test') for i,s,a,z in [(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,6),(4,'f',6,8),(5,'i1',8,9)]],issues=[],review_notes='test')
 m=Map(segments=[Segment(segment_id='s1',kind='issue',topic='costs',turn_refs=[1,2],rationale='test',cut=False),Segment(segment_id='s2',kind='issue',topic='sales',turn_refs=[3,4,5],rationale='test',cut=False)],exchanges=[Exchange(exchange_id='x1',segment_id='s1',investor='i1',probe=[1],answer=[2],next_move=[3],backchannels=[],uptake_status='observed',eligibility='cost request',uptake_reason='different shark question'),Exchange(exchange_id='x2',segment_id='s2',investor='i2',probe=[3],answer=[4],next_move=[5],backchannels=[],uptake_status='observed',eligibility='sales request',uptake_reason='receipt')],investor_moves=[Move(turn_ref=1,kind='probe',reason='cost'),Move(turn_ref=3,kind='probe',reason='sales'),Move(turn_ref=5,kind='receipt',reason='thanks')],repair_requests=[],review_notes='test')
 return m,timeline,roster

def test_cross_issue_uptake_keeps_exclusive_membership():
 m,t,r=fixture();b=freeze(m,t,r,'sha',0,10);check(b)
 assert 3 not in m.segments[0].turn_refs
 assert 3 in allowed('UP-TYPE',m.exchanges[0]) and 3 not in allowed('RS-EVID',m.exchanges[0])

def test_probe_actor_not_episode_opener():
 m,t,r=fixture();m.segments[0].turn_refs=[1,2,3,4,5];m.segments=m.segments[:1];m.exchanges[1].segment_id='s1'
 validate_map(m,t,r,0,10)
 assert [x.investor for x in m.exchanges]==['i1','i2']

def test_skipped_next_investor_fails():
 m,t,r=fixture();m.exchanges[0].next_move=[5]
 with pytest.raises(ValueError,match='skips'):validate_map(m,t,r,0,10)

def test_procedural_opening_fails():
 m,t,r=fixture();m.investor_moves[0].kind='procedural'
 with pytest.raises(ValueError,match='function'):validate_map(m,t,r,0,10)

def test_missing_substantive_exchange_fails():
 m,t,r=fixture();m.exchanges=m.exchanges[:1]
 with pytest.raises(ValueError,match=r'uncovered probes \[3\]'):validate_map(m,t,r,0,10)

def test_visual_sampling_does_not_allow_unlinked_or_outside_bin():
 roster=[{'speaker':'i1','role':'investor'}]
 slot=VisualSlot(investor='i1',bin_index=0,start_s=2,end_s=5,role='speaking',linkage='synchronized_speech',evidence='mouth',missing_reason='none')
 inv=VisualInventory(slots=[slot]);validate_visual(inv,roster,0,10)
 slot.linkage='unavailable'
 with pytest.raises(ValueError):validate_visual(inv,roster,0,10)
 slot.linkage='synchronized_speech';slot.end_s=20
 with pytest.raises(ValueError):validate_visual(inv,roster,0,10)

def test_missing_visual_cannot_be_scored():
 inv=VisualInventory(slots=[VisualSlot(investor='i1',bin_index=0,start_s=None,end_s=None,role='unavailable',linkage='unavailable',evidence='not visible',missing_reason='off_camera')])
 result=VisualRatings(ratings=[VisualRating(investor='i1',bin_index=0,verified=True,value=3,evidence='guess',missing_reason='none')])
 with pytest.raises(ValueError):validate_visual_ratings(result,inv)


def test_generated_evidence_schema_blocks_cross_issue_refs():
 from pathlib import Path
 from pipeline.codebook import load_codebook
 book=load_codebook(Path('docs/codebooks/archive/stage_b_v2_structured_2026-09-27/episode.draft.yaml'))
 refs={e.id:[1,2] for e in book.entries if e.unit=='EPISODE'}
 model=record_schema(book,'EPISODE',refs)
 field=model.model_fields['RP_ACK'].annotation
 with pytest.raises(Exception):
  field.model_validate({'value':False,'uncodable':False,'missing_reason':'none','evidence':'Inspected source','turn_refs':['3']})
 value=field.model_validate({'value':False,'uncodable':False,'missing_reason':'none','evidence':'Inspected source','turn_refs':['1','2']})
 assert value.turn_refs==['1','2']
 assert normalize_rating_refs({'episode':{'RP-ACK':value.model_dump()},'exchanges':[]})['episode']['RP-ACK']['turn_refs']==[1,2]


def test_multi_exchange_wire_schema_keeps_each_answer_scoped():
 from pathlib import Path
 from pipeline.codebook import load_codebook
 book=load_codebook(Path('docs/codebooks/archive/stage_b_v2_structured_2026-09-27/episode.draft.yaml'))
 m,_,_=fixture()
 schema=batch_schema(book,False,exchanges=m.exchanges)
 wire=schema.model_fields['exchanges'].annotation
 first=wire.model_fields['item_0'].annotation
 rating=first.model_fields['RS_EVID'].annotation
 payload={'value':None,'uncodable':True,'missing_reason':'ambiguous_attribution','evidence':'Source uncertain','turn_refs':['2']}
 rating.model_validate(payload)
 with pytest.raises(Exception):rating.model_validate({**payload,'turn_refs':['4']})
 normalized=normalize_rating_refs({'episode':None,'exchanges':{'x1':{'RS-EVID':payload}}})
 assert normalized['exchanges']==[{'exchange_id':'x1','ratings':{'RS-EVID':{**payload,'turn_refs':[2]}}}]


def test_exchange_only_schema_omits_null_only_episode_field():
 from pathlib import Path
 from pipeline.codebook import load_codebook
 book=load_codebook(Path('docs/codebooks/archive/stage_b_v2_structured_2026-09-27/episode.draft.yaml'))
 assert 'episode' not in batch_schema(book,False).model_fields
 assert normalize_rating_refs({'exchanges':[]})=={'exchanges':[],'episode':None}


def test_rating_cannot_call_observed_uptake_a_cutoff():
 m,t,r=fixture();b=freeze(m,t,r,'sha',0,10)
 record={'episode':None,'exchanges':[{'exchange_id':'x1','ratings':{'UP-TYPE':{'uncodable':True,'value':None,'evidence':'Ends early','turn_refs':[2],'missing_reason':'cutoff'}}}]}
 assert any('cutoff contradicts' in e for e in validate_ratings(record,m.segments[0],b,exchange_ids=['x1']))


# --- Map v3: exchanges derived by code from the model's judgments ---------------------------------
import json
from pathlib import Path

ROSTER3=[{'speaker':s,'role':'investor'} for s in ('i1','i2','i3')]+[{'speaker':'f','role':'founder'}]

def timeline3(rows):
 return TimelineReview(turns=[Turn(turn_idx=i,speaker=s,start_s=a,end_s=z,tag='t') for i,s,a,z in rows],issues=[],review_notes='t')

def proposal3(segments,moves,flags=(),groups=()):
 return MapProposal(segments=[Segment(segment_id=sid,kind='issue',topic=sid,turn_refs=refs,rationale='t',cut=False) for sid,refs in segments],
  investor_moves=[Move(turn_ref=i,kind=k,reason='r') for i,k in moves],uptake_flags=list(flags),probe_groups=[list(g) for g in groups],repair_requests=[],review_notes='t')

def test_derivation_backchannel_receipt_trailing_token_and_cutoff():
 t=timeline3([(1,'i1',0,2),(2,'f',2,5),(3,'i2',3,3.5),(4,'f',5,7),(5,'i1',7,8),(6,'i2',8,10),(7,'i3',10,11),(8,'f',11,13),(9,'i2',13,13.5),(10,'i3',14,15),(11,'f',15,17)])
 p=proposal3([('s1',[1,2,3,4,5]),('s2',[6,7,8,9]),('s3',[10,11])],[(1,'probe'),(3,'backchannel'),(5,'receipt'),(6,'probe'),(7,'social'),(9,'backchannel'),(10,'probe')])
 m=derive_map(p,t,ROSTER3);validate_map(m,t,ROSTER3,0,20)
 x1,x2,x3=m.exchanges
 assert (x1.answer,x1.backchannels,x1.next_move,x1.uptake_status)==([2,4],[3],[5],'observed')   # 'mm-hm' skipped; a reaction is the uptake
 assert (x2.answer,x2.next_move)==([8],[10])        # side remark before the answer does not pre-empt it; a trailing 'mm-hm' is never the next move
 assert (x3.answer,x3.next_move,x3.uptake_status)==([11],[],'cutoff')

def test_derivation_no_answer_before_a_new_question():
 t=timeline3([(1,'i1',0,2),(2,'i2',2,4),(3,'f',4,6)])
 m=derive_map(proposal3([('s1',[1,2,3])],[(1,'probe'),(2,'probe')]),t,ROSTER3);validate_map(m,t,ROSTER3,0,10)
 assert [(x.investor,x.answer,x.uptake_status) for x in m.exchanges]==[('i1',[],'no_opportunity'),('i2',[3],'cutoff')]

def test_explicit_probe_group_keeps_the_founder_reply_between_its_turns():
 t=timeline3([(1,'i1',0,2),(2,'f',2,3),(3,'i1',3,5),(4,'f',5,6),(5,'i2',6,8)])
 default=derive_map(proposal3([('s1',[1,2,3,4,5])],[(1,'probe'),(3,'probe'),(5,'receipt')]),t,ROSTER3)
 assert [x.probe for x in default.exchanges]==[[1],[3]]
 grouped=derive_map(proposal3([('s1',[1,2,3,4,5])],[(1,'probe'),(3,'probe'),(5,'receipt')],groups=[(1,3)]),t,ROSTER3)
 validate_map(grouped,t,ROSTER3,0,10)
 assert [(x.probe,x.answer,x.next_move) for x in grouped.exchanges]==[([1,3],[2,4],[5])]

def test_uptake_flag_marks_the_link_unobservable():
 t=timeline3([(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,6)])
 flag=UptakeFlag(probe_turn=1,status='edited_adjacency',reason='cut between answer and reaction')
 m=derive_map(proposal3([('s1',[1,2,3])],[(1,'probe'),(3,'receipt')],flags=[flag]),t,ROSTER3);validate_map(m,t,ROSTER3,0,10)
 assert (m.exchanges[0].uptake_status,m.exchanges[0].next_move,m.exchanges[0].uptake_reason)==('edited_adjacency',[],'cut between answer and reaction')

def test_unassigned_turns_become_other_but_an_unassigned_probe_still_fails():
 t=timeline3([(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,5),(4,'f',5,6)])
 m=derive_map(proposal3([('s1',[1,2])],[(1,'probe'),(3,'social')]),t,ROSTER3);validate_map(m,t,ROSTER3,0,10)
 assert [(s.kind,s.turn_refs) for s in m.segments if s.kind=='other']==[('other',[3,4])]
 bad=derive_map(proposal3([('s1',[1,2])],[(1,'probe'),(3,'probe')]),t,ROSTER3)
 with pytest.raises(ValueError):validate_map(bad,t,ROSTER3,0,10)

def test_review_corrections_relabel_and_resegment():
 t=timeline3([(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,5),(4,'f',5,6)])
 p=proposal3([('s1',[1,2,3,4])],[(1,'probe'),(3,'social')])
 fix=MapCorrections(move_corrections=[Move(turn_ref=3,kind='probe',reason='asks price')],segment_replacements=[Segment(segment_id='s1',kind='issue',topic='a',turn_refs=[1,2],rationale='t',cut=False),Segment(segment_id='s2',kind='issue',topic='b',turn_refs=[3,4],rationale='t',cut=False)],removed_segments=[],probe_groups=[],uptake_flags=[],repair_requests=[],review_notes='split')
 m=derive_map(apply_corrections(p,fix),t,ROSTER3);validate_map(m,t,ROSTER3,0,10)
 assert [(x.segment_id,x.probe,x.answer,x.next_move) for x in m.exchanges]==[('s1',[1],[2],[3]),('s2',[3],[4],[])]
 # a label on a founder turn is accepted by apply_corrections but rejected by the gate
 bad=derive_map(apply_corrections(p,fix.model_copy(update={'move_corrections':[Move(turn_ref=2,kind='probe',reason='x')]})),t,ROSTER3)
 with pytest.raises(ValueError):validate_map(bad,t,ROSTER3,0,10)

def test_review_can_label_investor_turns_the_proposal_omitted():
 t=timeline3([(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,5),(4,'f',5,6)])
 p=proposal3([('s1',[1,2,3,4])],[(1,'probe')])            # turn 3 (an investor turn) left unlabelled
 fix=MapCorrections(move_corrections=[Move(turn_ref=3,kind='probe',reason='asks price')],segment_replacements=[],removed_segments=[],probe_groups=[],uptake_flags=[],repair_requests=[],review_notes='added')
 m=derive_map(apply_corrections(p,fix),t,ROSTER3);validate_map(m,t,ROSTER3,0,10)
 assert [x.turn_ref for x in m.investor_moves]==[1,3] and [x.probe for x in m.exchanges]==[[1],[3]]

_PILOT5=next((d for d in (Path('runs/old_version_runs/stage_b_exchange_pilot5'),Path('runs/stage_b_exchange_pilot5')) if d.is_dir()),Path('runs/stage_b_exchange_pilot5'))
BUNDLES=sorted(_PILOT5.glob('st_*/frozen_bundle.json'))

@pytest.mark.skipif(not BUNDLES,reason='exchange pilot outputs not present')
@pytest.mark.parametrize('path',BUNDLES,ids=lambda p:p.parent.name)
def test_derivation_reproduces_every_accepted_pilot_exchange(path):
 """Given only the model's own judgments from the accepted maps, code reproduces each accepted exchange."""
 b=json.loads(path.read_text());m=Map.model_validate(b['mapping']);t=TimelineReview.model_validate(b['timeline'])
 listed={i for x in m.exchanges for i in x.backchannels}   # the v2 label set had no 'backchannel'
 moves=[mv.model_copy(update={'kind':'backchannel'}) if mv.turn_ref in listed else mv for mv in m.investor_moves]
 flags=[UptakeFlag(probe_turn=x.probe[0],status=x.uptake_status,reason=x.uptake_reason) for x in m.exchanges if x.uptake_status in ('edited_adjacency','ambiguous_attribution')]
 p=MapProposal(segments=m.segments,investor_moves=moves,uptake_flags=flags,probe_groups=[x.probe for x in m.exchanges if len(x.probe)>1],repair_requests=[],review_notes='')
 d=derive_map(p,t,b['roster']);validate_map(d,t,b['roster'],b['start_s'],b['cutoff_s'])
 key=lambda x:(tuple(x.probe),tuple(sorted(x.answer)),tuple(sorted(x.backchannels)),tuple(x.next_move),x.uptake_status,x.segment_id,x.investor)
 assert sorted(map(key,d.exchanges))==sorted(map(key,m.exchanges))


def test_solo_founder_team_codes_are_filled_not_asked_and_pass_the_gate():
 m,t,r=fixture();b=freeze(m,t,r,'sha',0,10);seg=m.segments[0]
 episode=fill_solo_team_codes({},seg,['TM-HANDOFF','TM-RESCUE','TM-CONTRA'])
 assert all(v['uncodable'] and v['missing_reason']=='not_applicable' and v['value'] is None for v in episode.values())
 record={'episode':episode,'exchanges':[]}
 errors=validate_ratings(record,seg,b,exchange_ids=[])
 assert not [e for e in errors if e.startswith('TM-')]

def test_retired_codes_are_not_in_the_archived_v2_draft():
 from pathlib import Path
 from pipeline.codebook import load_codebook
 ids={e.id for e in load_codebook(Path('docs/codebooks/archive/stage_b_v2_structured_2026-09-27/episode.draft.yaml')).entries}
 assert not ids&{'RP-RECOV','VIS-ENGAGE'} and len(ids)==18
 retired={e.id for e in load_codebook(Path('docs/codebooks/archive/stage_b/retired_codes_2026-09-28.yaml')).entries}
 assert retired=={'RP-RECOV','VIS-ENGAGE'}


def test_unassigned_founder_turn_inside_an_issue_is_left_for_the_gate():
 # the map put only investor turns into the issue; the founder's answer (2) sits inside the issue's span
 t=timeline3([(1,'i1',0,2),(2,'f',2,4),(3,'i2',4,5),(4,'f',5,6)])
 p=proposal3([('s1',[1,3])],[(1,'probe'),(3,'probe')])
 m=derive_map(p,t,ROSTER3)
 assert 2 not in {i for s in m.segments for i in s.turn_refs}
 with pytest.raises(ValueError,match='missing turn IDs'):validate_map(m,t,ROSTER3,0,10)
