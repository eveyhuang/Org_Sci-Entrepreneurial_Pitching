#!/usr/bin/env python3
"""HISTORICAL ROUTE (v2 instrument, archived 2026-10-06). Not current execution.
V2 comparison: new real audiovisual calls; fixed videos, Stage A/outcomes/cutoffs."""
from __future__ import annotations
import argparse,concurrent.futures,hashlib,json,shutil,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.errors import SchemaInvalidError
from scripts.experiments.run_structured_pilot import Pilot,namespace_prepared_turns
from pipeline.stage_b.exchanges import *
from pipeline.stage_b.structured import PitchResult,validate_pitch
from pipeline.codebook import Codebook,load_codebook
from pipeline.media import clip
from pipeline.prompts import render_prompt
from pipeline.schemas.stage_b_timeline import normalised_turns
BASE=ROOT/'runs/stage_b_structured_pilot5';RUN=ROOT/'runs/stage_b_exchange_pilot5'
SEED=None  # optional run whose final reviewed timelines seed this one (set by --seed-timelines-from)
def read(p):return json.loads(p.read_text())
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def snapshot(row):
 r=RUN/row['video_id'];old=BASE/row['video_id']
 if (r/'experiment.json').exists():return r
 for name in ['codebooks','templates','prompts','results','raw','media']:(r/name).mkdir(parents=True,exist_ok=True)
 for name in ['segmentation','episode','pitch']:
  load_codebook(ROOT/f'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/{name}.draft.yaml')
  for ext in ['md','yaml']:shutil.copy2(ROOT/f'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/{name}.draft.{ext}',r/'codebooks')
 for name in ['boundaries.json','reviewed_timeline.json']:shutil.copy2(old/name,r/name)
 if SEED is not None:shutil.copy2(SEED/row['video_id']/'reviewed_timeline.json',r/'reviewed_timeline.json')  # identical timelines: differences come from the map/codebook, not re-transcription
 for name in ['timeline.json','accepted_prepared.json']:shutil.copy2(old/'results'/name,r/'results'/name)
 shutil.copy2(ROOT/'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/stage_b_exchanges.j2',r/'templates')
 meta=read(old/'experiment.json');meta.update(schema_version=2,baseline=str(old),timeline_seed=str(SEED) if SEED else None,reuse=['Stage A v2','outcomes/cutoffs','reviewed timeline as source-review seed'],codebook_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'codebooks').iterdir()})
 dump(r/'experiment.json',meta);return r
class Experiment(Pilot):
 def __init__(self,run):
  # Parent loads the legacy EPISODE subset, unused here; new schemas are per-unit.
  super().__init__(run)
 def call(self,key,prompt,schema,media,info):
  prior=list((self.run/'raw').glob(f'{key}__attempt*_error.json'))
  if key.startswith('visual_inventory') and any('MAX_TOKENS' in read(p).get('message','') for p in prior):
   return super().call(key+'_medium',prompt,schema,media,{**info,'recovery_thinking_level':'medium','recovery_reason':'visual inventory exhausted default budget'})
  failed_deadline=any('DEADLINE_EXCEEDED' in read(p).get('message','') or read(p).get('error_type')=='ReadTimeout' for p in prior)
  if failed_deadline and not (self.run/'results'/f'{key}.json').exists():
   return super().call(key+'_medium',prompt,schema,media,{**info,'recovery_thinking_level':'medium'})
  try:return super().call(key,prompt,schema,media,info)
  except Exception as exc:
   if 'DEADLINE_EXCEEDED' not in str(exc) and type(exc).__name__!='ReadTimeout':raise
   return super().call(key+'_medium',prompt,schema,media,{**info,'recovery_thinking_level':'medium'})
 def source_inputs(self):
  bounds=read(self.run/'boundaries.json');full=read(self.run/'results/timeline.json')['record'];tl=TimelineReview.model_validate(read(self.run/'reviewed_timeline.json'));start=bounds['prepared_end_s'];end=bounds['cutoff_s']
  note=f'Source clip {start}–{end}s. Return ORIGINAL SOURCE seconds (clip time + {start}). Fixed roster: '+json.dumps(full['roster'],separators=(',',':'))
  return bounds,full,tl,start,end,clip(self.source,start,end,self.run/'media'),note
 def mapping(self):
  """Map v3: the model decides turn functions, issue membership and (rarely) probe groupings and
  uptake flags; code derives every exchange. The review returns corrections only. A map that still
  fails the gate after one targeted correction is written to needs_human_review.json, not retried."""
  bounds,full,tl,start,end,media,note=self.source_inputs()
  existing=self.run/'frozen_bundle.json'
  if existing.exists():
   cached=read(existing)
   if cached['timeline_hash']==digest(tl):check(cached);return
  revision=read(self.run/'timeline_revision.json')['revision'] if (self.run/'timeline_revision.json').exists() else 0
  suffix=f'_r{revision}' if revision else ''
  defs='\n'.join(e.id+' '+e.definition+'\n'+e.include_if+'\n'+e.exclude_if for e in self.segbook.entries)
  rules=('Label EVERY investor turn with one function, following the eligibility gate in the definitions below: '
   'probe = requests answerable information, explanation or substantiation about the product, venture or founders, or challenges a claim. '
   'A challenge can be phrased as a statement (calling the valuation unrealistic, doubting a number, warning about competitors) and is still a probe '
   'when it invites the founder to answer or defend; '
   'receipt = a reaction that takes the floor but does not ask the founder to answer or defend (praise, surprise, agreement, laughter, thinking aloud); '
   'backchannel = a minimal listener token that does not take the floor while the founder keeps talking ("mm-hm", "right", "okay"); '
   'procedural, social or other otherwise. Assign EVERY timeline turn to exactly one segment. An issue begins at its first '
   'substantive probe, never at demonstration setup, tasting instructions or banter; turns before it belong to a procedural, social or other segment. '
   'Do NOT list exchanges, answers, listener tokens or next moves: the pipeline derives them from your labels (each probe, or run of probe '
   'turns by one investor, is one exchange; its answer runs until the next investor turn that is not a backchannel). '
   'Only if two probe turns by the same investor are ONE question although a founder turn separates them (e.g. a reiteration after a minimal '
   'founder reply), list them in probe_groups. Only if the next investor move after an answer cannot be linked to it because of an edit or an '
   'unidentifiable speaker, add an uptake_flag naming the probe\'s first turn and the source evidence. '
   'Do not use old episode maps or ratings. Return repair_requests if the timeline itself needs changes; never invent IDs.\n')
  p=note+'\n'+rules+defs+'\nTIMELINE\n'+tl.model_dump_json()
  proposal=MapProposal.model_validate(self.call('map_v3'+suffix,p,MapProposal,media,{'phase':'map_v3','timeline_hash':digest(tl)}))
  q=(p+'\nIndependently source-check this proposed map: every investor function (especially probe vs receipt vs backchannel), '
     'every issue boundary, procedural openings and excluded social exchanges. Do not approve by default. Return ONLY corrections: '
     'investor turns whose function changes, complete corrected segments to replace or add, segment IDs to remove, and the complete '
     'probe_groups and uptake_flags lists. Return empty lists where nothing changes.\nPROPOSAL\n'+proposal.model_dump_json())
  review=MapCorrections.model_validate(self.call('map_v3_review'+suffix,q,MapCorrections,media,{'phase':'map_v3_review','timeline_hash':digest(tl)}))
  final=apply_corrections(proposal,review)
  for attempt in range(2):
   if final.repair_requests:
    if revision>=2:raise ValueError('Unresolved source repair after two reviewed revisions')
    repair_prompt=note+'\nCheck these DISPUTED attribution/segmentation claims against source independently; do not assume the reviewer is correct: '+json.dumps(final.repair_requests)+'\nReturn the complete corrected timeline, retaining IDs for retained events, new split IDs >=30000. Explain whether the claims were confirmed or rejected in review_notes. Preserve short speech; roles/IDs follow fixed roster, not inferred names. Resolve reparable errors; truly unavailable evidence needs actual affected code IDs.\nCURRENT TIMELINE\n'+tl.model_dump_json()
    repaired=self.call(f'timeline_source_repair_{revision+1}',repair_prompt,TimelineReview,media,{'phase':'source_repair','previous_hash':digest(tl),'disputes':final.repair_requests})
    verified=TimelineReview.model_validate(repaired);validate_timeline(verified,full['roster'],start,end)
    dump(self.run/f'timeline_before_revision_{revision+1}.json',tl.model_dump())
    dump(self.run/'reviewed_timeline.json',repaired)
    dump(self.run/'timeline_revision.json',{'revision':revision+1,'reason':final.repair_requests,'previous_hash':digest(tl),'new_hash':digest(verified)})
    return self.mapping()
   try:
    mapping=derive_map(final,tl,full['roster'])
    b=freeze(mapping,tl,full['roster'],self.meta['source_sha256'],start,end);break
   except ValueError as exc:
    dump(self.run/f'map_rejection_v3_{attempt}{suffix}.json',{'error':str(exc),'proposal':final.model_dump()})
    if attempt==1:
     dump(self.run/'needs_human_review.json',{'error':str(exc),'proposal':final.model_dump(),'timeline_hash':digest(tl)})
     raise ValueError('Map still fails the gate after one targeted correction; see needs_human_review.json: '+str(exc))
    fix=(p+'\nThis map fails a structural check. Correct ONLY what the errors name (segment membership, issue openings, probe labels); '
         'return corrections in the same format.\nERRORS\n'+str(exc)+'\nCURRENT MAP\n'+final.model_dump_json())
    patch=MapCorrections.model_validate(self.call(f'map_v3_patch{suffix}',fix,MapCorrections,media,{'phase':'map_v3_patch','timeline_hash':digest(tl)}))
    final=apply_corrections(final,patch)
  dump(self.run/'map_proposal.json',final.model_dump())
  dump(self.run/'frozen_bundle.json',b)
 def annotate(self):
  """Code every issue. An issue whose ratings still fail the gate after three attempts is written to
  results/needs_human_review_<segment>.json and skipped; the other issues are still coded."""
  b=read(self.run/'frozen_bundle.json');m,t=check(b);ts={v.turn_idx:v for v in t.turns};flagged=[]
  for seg in m.segments:
   if seg.kind!='issue':continue
   xs=[x for x in m.exchanges if x.segment_id==seg.segment_id];xs.sort(key=lambda x:min(ts[i].start_s for i in x.probe))
   saved=self.run/f'results/accepted_{seg.segment_id}.json'
   if saved.exists():
    a=read(saved);start=min(ts[i].start_s for i in seg.turn_refs);context=[v.turn_idx for v in t.turns if v.start_s<start and v.end_s>max(0,start-10)]
    if a['map_hash']==b['map_hash'] and not validate_ratings(a,seg,b,context):continue
   accepted=[];episode=None;failed=None
   for offset in range(len(xs)):
    batch=xs[offset:offset+1];with_ep=offset==0;ids=[x.exchange_id for x in batch]
    scored=set(seg.turn_refs) if with_ep else {i for x in batch for i in x.probe+x.answer}
    links={i for x in batch for i in x.next_move};refs=scored|links
    lo=min(ts[i].start_s for i in scored);hi=max(ts[i].end_s for i in refs);media_start=lo  # no pre-probe padding: it served only the retired RP-RECOV
    context=[]
    source_context=[]
    if not with_ep and not batch[0].answer:
     source_context=sorted(set(seg.turn_refs)-refs)
     media_start=max(0,min(ts[i].start_s for i in seg.turn_refs)-2)
     hi=max(hi,max(ts[i].end_s for i in seg.turn_refs))
    media=clip(self.source,media_start,hi,self.run/'media')
    solo=sum(r['role']=='founder' for r in b['roster'])<2
    askbook=Codebook(entries=[e for e in self.epbook.entries if not (solo and e.id.startswith('TM-'))])
    selected=Codebook(entries=[e for e in askbook.entries if e.unit=='EXCHANGE' or with_ep and e.unit=='EPISODE'])
    ctx={'source_offset':media_start,'score_start':lo,'score_end':max(ts[i].end_s for i in scored),'roster':b['roster'],'issue':seg.model_dump(),'score_issue_codes':with_ep,'exchanges':[x.model_dump() for x in batch],'context_only_refs':context,'timeline':[ts[i].model_dump() for i in sorted(refs|set(context)|set(source_context),key=lambda i:ts[i].start_s)]}
    if source_context:ctx['no_answer_context_only_refs']=source_context
    unresolved=[z for z in b['timeline']['issues'] if z['status']=='unobservable']
    if unresolved:ctx['source_issues']=unresolved
    p=render_prompt('stage_b_exchanges.j2',codebook=selected,context={'context_json':json.dumps(ctx,separators=(',',':'))},prompt_dir=self.run/'templates')
    if source_context:p+='\nAdditional issue footage verifies interruption/no answer only; do not substitute its answers or cite its context-only IDs for exchange ratings. With no founder answer, RS-RESP may be NONE; every other RS/PREP code and UP-TYPE must have uncodable=true, value=null, missing_reason=no_opportunity.'
    episode_refs={e.id:sorted(seg.turn_refs) for e in askbook.entries if e.unit=='EPISODE'}
    exchange_refs={code:sorted(set().union(*(allowed(code,x) for x in batch))) for code in EXCHANGE_CODES}
    schema=batch_schema(askbook,with_ep,episode_refs,exchange_refs,batch)
    for n in range(3):
     version='single_context_v7' if source_context else ('single_v4' if with_ep else 'single_v5')
     key=f'codes_scoped_{version}_{seg.segment_id}_{offset}_try{n}'
     try:out=self.call(key,p,schema,media,{'phase':'exchange_codes','map_hash':b['map_hash'],'exchange_ids':ids,'source_start':media_start,'source_end':hi})
     except SchemaInvalidError as exc:  # malformed output (e.g. a value with uncodable=false) is retried like a gate failure
      out=None;errors=['output failed the schema: '+str(exc).split(': ',1)[-1][:800]];dump(self.run/f'results/{key}_qc.json',errors)
      p+='\nReinspect source under the SAME windows. Rejected: '+json.dumps(errors);continue
     out=normalize_rating_refs(out)
     if with_ep and solo:out['episode']=fill_solo_team_codes(out['episode'],seg,[e.id for e in self.epbook.entries if e.id.startswith('TM-')])
     errors=validate_ratings(out,seg,b,context,ids);dump(self.run/f'results/{key}_qc.json',errors)
     if not errors:break
     p+='\nReinspect source under the SAME windows. Rejected: '+json.dumps(errors)
    else:
     failed={'segment_id':seg.segment_id,'exchange_ids':ids,'errors':errors,'last_output':out,'map_hash':b['map_hash']};break
    if with_ep:episode=out['episode']
    accepted.extend(out['exchanges'])
   if failed:dump(self.run/f'results/needs_human_review_{seg.segment_id}.json',failed);flagged.append(seg.segment_id);continue
   dump(self.run/f'results/accepted_{seg.segment_id}.json',{'episode':episode,'exchanges':accepted,'map_hash':b['map_hash'],'timeline_hash':b['timeline_hash']})
 def visuals(self):
  if (self.run/'visual_bin_verification.json').exists():
   inv=VisualInventory.model_validate(read(self.run/'visual_inventory.json'))
   if read(self.run/'visual_bin_verification.json')['inventory_hash']==digest(inv):
    validate_visual_ratings(VisualRatings.model_validate(read(self.run/'results/accepted_visual.json')),inv);return
  bounds,full,tl,start,end,media,note=self.source_inputs()
  code=next(e for e in self.epbook.entries if e.id=='VIS-ENGAGE')
  p=note+'\nBuild the fixed investor × 60-second-bin visual sampling grid. Bin 0 starts at '+str(start)+'. For every investor in each bin, select the FIRST qualifying contiguous readable interval >=2s, capped at5s and ending within that bin. Use source footage, not inferred turn timing. Synchronized speaking footage or continuous shared-frame listening only; exclude detached reactions. Return explicit missing slots too. Do NOT rate warmth or select for expression.\n'+code.include_if+'\n'+code.exclude_if
  for n in range(3):
   raw=self.call(f'visual_inventory_{n}',p,VisualInventory,media,{'phase':'visual_inventory','start':start,'end':end})
   inv=VisualInventory.model_validate(raw)
   try:validate_visual(inv,full['roster'],start,end);break
   except ValueError as exc:
    if n==2:raise
    p+='\nCorrect source-grounded sampling; error: '+str(exc)+'\n'+json.dumps(raw,separators=(',',':'))
  dump(self.run/'visual_inventory.json',inv.model_dump())
  q=note+'\nVerify every selected interval and its identity, role, linkage, FIRST eligible position and readable duration against source. An invalid selection must be unverified with null value; do not substitute a better interval. For valid observations rate VIS-ENGAGE only during the listed interval. Preserve missing slots.\n'+json.dumps(code.model_dump(exclude={'provenance','examples'}),separators=(',',':'))+'\nSLOTS\n'+inv.model_dump_json()
  for n in range(3):
   raw=self.call(f'visual_ratings_{n}',q,VisualRatings,media,{'phase':'visual_ratings','inventory_hash':digest(inv)})
   result=VisualRatings.model_validate(raw)
   try:validate_visual_ratings(result,inv);break
   except ValueError as exc:
    if n==2:raise
    q+='\nGate rejected: '+str(exc)+'\n'+json.dumps(raw,separators=(',',':'))
  dump(self.run/'results/accepted_visual.json',result.model_dump())
 def pitch_v2(self):
  b=read(self.run/'frozen_bundle.json');m,tl=check(b);bounds=read(self.run/'boundaries.json');full=read(self.run/'results/timeline.json')['record']
  prep=[t for t in normalised_turns(full) if bounds['prepared_start_s']<=t['start_s'] and t['end_s']<=bounds['prepared_end_s']]
  prep,remap=namespace_prepared_turns(prep,{t.turn_idx for t in tl.turns});dump(self.run/'prepared_id_namespace.json',remap)
  turns={t['turn_idx']:t for t in prep};turns.update({t.turn_idx:t.model_dump() for t in tl.turns})
  units={t['turn_idx']:'prepared' for t in prep}
  for s in m.segments:
   if s.kind=='issue':units.update({i:s.segment_id for i in s.turn_refs})
  founders={r['speaker'] for r in b['roster'] if r['role']=='founder'};frefs={i for i,t in turns.items() if t['speaker'] in founders}
  p='Watch pre-decision source. Code PREP-CONSIST only over eligible founder statements; original source timestamps. Next-move context does not confer membership. Report inspected IDs including eligible investor context. A repeated claim needs locatable pair evidence even if compatible; count only incompatible cross-issue/ prepared-to-issue pairs with matching metric, period, units, scope and no reconciliation.\n'+json.dumps([e.model_dump(exclude={'provenance','examples'}) for e in self.pitchbook.entries],separators=(',',':'))+'\n'+json.dumps({'roster':b['roster'],'timeline':list(turns.values()),'eligible_units':units},separators=(',',':'))
  media=clip(self.source,0,bounds['cutoff_s'],self.run/'media')
  for n in range(3):
   raw=self.call(f'pitch_v2_{n}',p,PitchResult,media,{'phase':'pitch_v2','map_hash':b['map_hash'],'eligible':sorted(units)})
   errors=validate_pitch(PitchResult.model_validate(raw),set(units),units,frefs,turns)
   dump(self.run/f'results/pitch_v2_{n}_qc.json',errors)
   if not errors:dump(self.run/'results/accepted_pitch.json',raw);return
   p+='\nReinspect source; rejected: '+json.dumps(errors)
  dump(self.run/'results/needs_human_review_pitch.json',{'code':'PREP-CONSIST','errors':errors,'last_output':raw,'map_hash':b['map_hash']})  # flagged like an issue; the video still completes
 def all(self):
  self.mapping();self.annotate();self.pitch_v2()  # visuals() retired 2026-09-28 (VIS-ENGAGE); kept only to reproduce the exchange pilot
  b=read(self.run/'frozen_bundle.json');m,t=check(b)
  flagged=sorted(p.stem.removeprefix('needs_human_review_') for p in (self.run/'results').glob('needs_human_review_*.json') if not (self.run/'results'/('accepted_'+p.stem.removeprefix('needs_human_review_')+'.json')).exists())
  flagged=[x for x in flagged if not (x=='pitch' and (self.run/'results/accepted_pitch.json').exists())]
  dump(self.run/'completion.json',{'status':'complete_with_flags' if flagged else 'complete','needs_human_review':flagged,'episodes':sum(s.kind=='issue' for s in m.segments),'exchanges':len(m.exchanges),'qa_turns':len(t.turns)})
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['mapping','annotate','visuals','pitch_v2','all'],default='all');ap.add_argument('--video-ids',nargs='*');ap.add_argument('--workers',type=int,default=3);ap.add_argument('--run-name',default='stage_b_exchange_pilot5');ap.add_argument('--seed-timelines-from',default=None);ap.add_argument('--base-run',default='stage_b_structured_pilot5',help='front-end run to read boundaries, timelines and Stage A from');a=ap.parse_args()
 global RUN,SEED,BASE;RUN=ROOT/'runs'/a.run_name;SEED=ROOT/'runs'/a.seed_timelines_from if a.seed_timelines_from else None;BASE=ROOT/'runs'/a.base_run
 rows=read(BASE/'selection.json')['videos'];rows=[r for r in rows if not a.video_ids or r['video_id'] in a.video_ids]
 ready=lambda r:all((BASE/r['video_id']/f).exists() for f in ('boundaries.json','reviewed_timeline.json','results/timeline.json','results/accepted_prepared.json'))
 for r in rows:
  if not ready(r):print('SKIPPED (front end not complete)',r['video_id'],flush=True)
 rows=[r for r in rows if ready(r)]
 def work(row):
  x=Experiment(snapshot(row))
  try:getattr(x,a.phase)();print('COMPLETE',row['video_id'],a.phase,flush=True);return True
  except Exception as exc:dump(x.run/f'failure_{a.phase}.json',{'error':str(exc),'traceback':traceback.format_exc()});print('FAILED',row['video_id'],str(exc),flush=True);return False
  finally:x.close()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:ok=list(pool.map(work,rows))
 if not all(ok):raise SystemExit(1)
if __name__=='__main__':main()
