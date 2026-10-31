#!/usr/bin/env python3
"""Verify codable affect scores against the draft's measurable visibility gate."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.run_structured_copy_keyboard import StructuredExperiment,RUN,read,dump
from pipeline.stage_b.structured import VisibilityAudit,visibility_gate,assert_frozen
from pipeline.media import clip

def main():
 exp=StructuredExperiment(RUN)
 try:
  b=read(RUN/'frozen_bundle.json');assert_frozen(b);turns={t['turn_idx']:t for t in b['timeline']['turns']}
  for e in b['mapping']['episodes']:
   path=RUN/f"results/accepted_{e['episode_id']}.json";result=read(path)
   if result['record']['UP-AFFECT']['uncodable']:continue
   start=min(turns[i]['start_s'] for i in e['turn_refs']);end=max(turns[i]['end_s'] for i in e['turn_refs'])
   media=clip(exp.source,start,end,RUN/'media')
   p=f'Inspect ONLY visual observability of {e["probing_speaker"]} in this episode. Do not assign warmth or any other label. Clip second zero = source {start}; return original SOURCE seconds. Roster: '+json.dumps(b['roster'])+'\nList every qualifying interval in which this specific investor is visibly readable and demonstrably linked to the ongoing interaction. Exclude ambiguous reaction cutaways, distant unreadable panel shots, other people, and offscreen speech. Each interval needs linkage evidence. Do not merge separate shots across time without visible continuity. Do not target a desired coverage percentage. Return coverage_reviewed=false if you cannot inspect the full clip, plus limitations. An empty interval list is allowed.'
   raw=exp.call(f'visibility_{e["episode_id"]}',p,VisibilityAudit,media,{'episode_id':e['episode_id'],'source_start_s':start,'source_end_s':end,'map_hash':b['map_hash']})
   if any(not start<=v['start_s']<v['end_s']<=end for v in raw['intervals']):
    dump(RUN/f'results/visibility_{e["episode_id"]}_coordinate_rejection.json',{'reason':'Returned intervals outside scored source window','record':raw})
    p+='\nYour previous response used invalid times outside the source window: '+json.dumps(raw)+'\nReinspect and return ORIGINAL SOURCE SECONDS, not clip-relative seconds. Source time = clip time + '+str(start)+'. For example clip second 1 is source second '+str(start+1)+'. Do not extend intervals to meet a threshold.'
    raw=exp.call(f'visibility_{e["episode_id"]}_coordinate_repair',p,VisibilityAudit,media,{'episode_id':e['episode_id'],'source_start_s':start,'source_end_s':end,'map_hash':b['map_hash'],'repair':'coordinate system'})
   ok=visibility_gate(VisibilityAudit.model_validate(raw),e['probing_speaker'],start,end)
   dump(RUN/f'results/visibility_{e["episode_id"]}_gate.json',{'passed':ok,'source_start_s':start,'source_end_s':end})
   if not ok:
    dump(RUN/f'results/{e["episode_id"]}_before_visibility_gate.json',result)
    result['record']['UP-AFFECT']={'uncodable':True,'value':None,'turn_refs':e['turn_refs'],'evidence':'Source-based visibility audit failed the required half-episode and continuous-five-second gate; see visibility audit artifact.'}
   result['visibility_audit']=f'visibility_{e["episode_id"]}';result['visibility_gate_passed']=ok;dump(path,result)
 finally:exp.close()
if __name__=='__main__':main()
