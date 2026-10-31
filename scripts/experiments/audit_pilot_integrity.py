#!/usr/bin/env python3
"""Verify finalized pilot artifacts without making API calls."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.stage_b.structured import assert_frozen,TimelineReview,InteractionMap,validate_codes,validate_pitch,PitchResult
from pipeline.schemas.stage_b_timeline import normalised_turns
from scripts.experiments.run_structured_pilot import namespace_prepared_turns
RUN=ROOT/'runs/stage_b_structured_pilot5'
def read(p):return json.loads(p.read_text())
def main():
 results=[]
 for row in read(RUN/'selection.json')['videos']:
  r=RUN/row['video_id']
  if not (r/'completion.json').exists():results.append({'video_id':r.name,'status':'incomplete'});continue
  b=read(r/'frozen_bundle.json');assert_frozen(b);meta=read(r/'experiment.json');bounds=read(r/'boundaries.json')
  assert hashlib.sha256(Path(meta['source']).read_bytes()).hexdigest()==meta['source_sha256']
  assert all(hashlib.sha256((r/'codebooks'/name).read_bytes()).hexdigest()==sha for name,sha in meta['codebook_sha256'].items())
  tl=TimelineReview.model_validate(b['timeline']);mapping=InteractionMap.model_validate(b['mapping']);byid={t.turn_idx:t for t in tl.turns}
  for e in mapping.episodes:
   path=r/'results'/f'accepted_{e.episode_id}.json';a=read(path);key=a['source_call']
   if not (r/'results'/f'{key}.json').exists():
    # Correct only provenance for old-process budget recoveries; preserve all ratings.
    key=next(k for k in [key+'_budget64',key+'_medium'] if (r/'results'/f'{k}.json').exists())
    a['source_call']=key;path.write_text(json.dumps(a,indent=2)+'\n')
   call=read(r/'results'/f'{key}.json');start=min(byid[i].start_s for i in e.turn_refs)
   context=[t.turn_idx for t in tl.turns if t.start_s<start and t.end_s>max(0,start-10)]
   assert not validate_codes(a['record'],e,b,context),(r.name,e.episode_id)
   assert call['info']['end']<=bounds['cutoff_s']
   assert call['info']['map_hash']==b['map_hash'] and a['map_hash']==b['map_hash']
  full=read(r/'results/timeline.json')['record'];prepared=[t for t in normalised_turns(full) if t['start_s']>=bounds['prepared_start_s'] and t['end_s']<=bounds['prepared_end_s']]
  prepared,_=namespace_prepared_turns(prepared,{t.turn_idx for t in tl.turns})
  turns={t['turn_idx']:t for t in prepared};turns.update({t.turn_idx:t.model_dump() for t in tl.turns})
  units={t['turn_idx']:'prepared' for t in prepared}
  for e in mapping.episodes:units.update({i:e.episode_id for i in e.turn_refs})
  founders={s['speaker'] for s in b['roster'] if s['role']=='founder'};refs={i for i,t in turns.items() if t['speaker'] in founders}
  assert not validate_pitch(PitchResult.model_validate(read(r/'results/accepted_pitch.json')),set(units),units,refs,turns)
  results.append({'video_id':r.name,'status':'integrity_checks_passed','episodes':len(mapping.episodes),'prepared_turns':len(prepared),'prepared_covered_start':min(t['start_s'] for t in prepared),'prepared_covered_end':max(t['end_s'] for t in prepared)})
 (RUN/'integrity_audit.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
