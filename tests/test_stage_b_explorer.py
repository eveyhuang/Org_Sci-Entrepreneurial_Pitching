import json
from pathlib import Path
import pytest
from analysis.stage_b_explorer import load_run,evaluate,artifact_text
from pipeline.stage_b.exchanges import Map,Segment,Exchange,Move,freeze,TimelineReview
from pipeline.stage_b.structured import Turn

def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))

def sample(tmp_path):
 root=tmp_path/'back';front=tmp_path/'front';vid='st_test';r=root/vid
 roster=[{'speaker':'i','role':'investor'},{'speaker':'j','role':'investor'},{'speaker':'f','role':'founder'}]
 timeline=TimelineReview(turns=[Turn(turn_idx=i,speaker=s,start_s=a,end_s=b,tag='test') for i,s,a,b in [(1,'i',0,1),(2,'f',1,3),(3,'j',3,4)]],issues=[],review_notes='test')
 m=Map(segments=[Segment(segment_id='s1',kind='issue',topic='test',turn_refs=[1,2,3],rationale='test',cut=False)],exchanges=[Exchange(exchange_id='x1',segment_id='s1',investor='i',probe=[1],answer=[2],next_move=[3],backchannels=[],uptake_status='observed',eligibility='test',uptake_reason='receipt')],investor_moves=[Move(turn_ref=1,kind='probe',reason='test'),Move(turn_ref=3,kind='receipt',reason='test')],repair_requests=[],review_notes='test')
 b=freeze(m,timeline,roster,'source',0,5)
 write(r/'frozen_bundle.json',b);write(r/'completion.json',{'status':'complete'})
 write(r/'experiment.json',{'source':'/test/video.mp4','codebook_sha256':{}})
 write(front/'selection.json',{'videos':[{'video_id':vid},{'video_id':'st_pending'}]})
 write(r/'boundaries.json',{'cutoff_s':5,'outcomes_model_reviewed_not_gold':{'sharks':[{'speaker':'i','ever_offer':True},{'speaker':'j','ever_offer':None}]}})
 write(r/'results/accepted_prepared.json',{'pitch_positivity':{'value':4}})
 def rating(value,refs):return {'value':value,'uncodable':False,'missing_reason':'none','turn_refs':refs,'evidence':'source'}
 codes={c:rating(v,[1] if c.startswith('PR-') else [2]) for c,v in {'RS-RESP':'addresses','RS-EVID':True,'PREP-SUBS':3,'PREP-COH':3,'PR-SEV':1,'PR-FRAME':'promotion'}.items()}
 write(r/'results/accepted_s1.json',{'episode':{'RP-ACK':rating(False,[2])},'exchanges':[{'exchange_id':'x1','ratings':codes}],'map_hash':b['map_hash'],'timeline_hash':b['timeline_hash']})
 (r/'prompts').mkdir();(r/'prompts/coding.txt').write_text('Actual archived prompt')
 for stage in [r,front/vid]:
  stage.mkdir(parents=True,exist_ok=True);(stage/'calls.jsonl').write_text(json.dumps({'key':'coding','input_tokens':100,'output_tokens':20,'thinking_tokens':30,'info':{'phase':'coding'}})+'\n')
 return root,front,r

def test_selected_denominator_and_unknown_outcome_are_preserved(tmp_path):
 root,front,r=sample(tmp_path);d=load_run(root,front)
 assert len(d.videos)==2 and d.videos.selected.all()
 assert d.videos.set_index('video_id').loc['st_pending','status']=='no_backend'
 assert d.decisions.offer.isna().sum()==1
 assert len(evaluate(d)[2])==1  # unknown is excluded, never a negative
 assert artifact_text(d,'st_test',contains='coding').endswith('Actual archived prompt')

def test_stale_accepted_ratings_never_enter_features(tmp_path):
 root,front,r=sample(tmp_path);p=r/'results/accepted_s1.json';a=json.loads(p.read_text());a['map_hash']='stale';write(p,a)
 d=load_run(root,front)
 assert d.ratings.empty and d.decisions.n_exchanges.sum()==0
 assert not d.decisions.analysis_eligible.any()
 assert 'accepted_validation' in d.flags.kind.values

def test_flagged_output_is_not_accepted_and_cost_is_not_doubled(tmp_path):
 root,front,r=sample(tmp_path);(r/'results/accepted_s1.json').unlink();write(r/'completion.json',{'status':'complete_with_flags'})
 write(r/'results/needs_human_review_s1.json',{'errors':['source conflict']})
 d=load_run(root,front)
 assert d.ratings.empty and d.decisions.n_exchanges.sum()==0
 assert d.decisions.has_flags.all()
 assert len(d.calls)==2 and d.calls.cost_usd.sum()==pytest.approx(.0016)
 assert d.changes_since_load()==[]
 write(r/'results/accepted_later.json',{})
 assert str(r/'results/accepted_later.json') in d.changes_since_load()
