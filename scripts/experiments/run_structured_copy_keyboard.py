#!/usr/bin/env python3
"""Resumable real-API test of the structured Stage B workflow, isolated from main registry."""
from __future__ import annotations
import argparse, concurrent.futures, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.compare_stage_b_copy_keyboard import Experiment, snapshot, read, dump
from pipeline.stage_b.structured import (TimelineReview,InteractionMap,PitchResult,freeze_bundle,
    validate_timeline,validate_map,validate_codes,validate_pitch,assert_frozen,allowed_refs)
from pipeline.schemas.stage_b_timeline import normalised_turns
from pipeline.media import clip
from pipeline.prompts import render_prompt, prompt_hash

RUN=ROOT/'runs/stage_b_copy_keyboard_structured_test'
PREVIOUS=ROOT/'runs/stage_b_copy_keyboard_revision_test'

class StructuredExperiment(Experiment):
 def call(self,key,prompt,schema,media,info):
  path=self.run/'results'/f'{key}.json'
  if path.exists():
   old=read(path)
   if old['prompt_hash']!=prompt_hash(prompt) or old['info']!=info or old['media']!=str(media):
    raise ValueError(f'Stale cached result {key}; use a new versioned call key/run, never reuse changed inputs')
  return super().call(key,prompt,schema,media,info)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['timeline','map','annotate'],required=True);a=ap.parse_args()
 snapshot(RUN);exp=StructuredExperiment(RUN)
 roster=read(PREVIOUS/'results/timeline.json')['record']['roster']
 qa=clip(exp.source,115,278,RUN/'media')
 source_note='Video clip starts at source second 115 (1:55) and ends at 278 (4:38), before the first decision. ALL returned times must be ORIGINAL SOURCE seconds: add 115 to clip time. Fixed speaker roster: '+json.dumps(roster)
 try:
  if a.phase=='timeline':
   seed=[t for t in normalised_turns(read(PREVIOUS/'results/timeline.json')['record']) if 115<=t['start_s']<278]
   p=source_note+'\nRepair this PROVISIONAL speaking timeline by inspecting the entire supplied audiovisual clip. Record all short spoken exchanges, responses to compliments, receipts, overlaps and unfinished continuations; exclude non-speech laughter as a speaking turn. Split founder speech at an explicit unrelated topic shift, including admission followed by a new agenda. Do not transcribe or quote; use short action tags. Preserve existing turn IDs for retained events. New/split events receive unused integer IDs >=100; NEVER renumber existing events. Keep source times within 115..278. Sort by onset. Attribution must follow voice/mouth evidence, not the camera cut. If genuinely unknowable, list an unobservable issue and affected code IDs; if reparable but unresolved, needs_repair. Do not invent another speaker.\nPROVISIONAL TURNS\n'+json.dumps(seed)+'\nReview particularly dense speech around 2:20–2:35, 3:28–3:40, 3:50–4:11 and 4:11–4:38 without assuming any expected event or code value. Return the complete repaired timeline, issues, and a concise repair log.'
   proposal=exp.call('timeline_repair',p,TimelineReview,qa,{'phase':'timeline','source_offset':115})
   p=source_note+'\nIndependently check every proposed turn against the audiovisual source. This is a verification pass, not approval by default. Return a corrected COMPLETE timeline with the SAME stable IDs for retained events and new unused IDs >=200 for additions. Do not quote dialogue. Confirm short speech, same-speaker topic splits, overlap, speaker identity, and timing. Remove non-speech-only turns. Flag unresolved issues as needs_repair or genuinely unobservable with affected codes. Do not claim human validation.\nPROPOSAL\n'+json.dumps(proposal)
   final=exp.call('timeline_review',p,TimelineReview,qa,{'phase':'timeline_review'})
   tl=TimelineReview.model_validate(final);validate_timeline(tl,roster,115,278)
   dump(RUN/'reviewed_timeline.json',final)
  elif a.phase=='map':
   tl=TimelineReview.model_validate(read(RUN/'reviewed_timeline.json'))
   code=exp.segbook.entries
   definitions='\n'.join(e.id+'\n'+e.definition+'\n'+e.include_if+'\n'+e.exclude_if for e in code)
   p=source_note+'\nBuild issue episodes AND explicit exchange spans from the repaired timeline. Do not change or invent turn IDs. Every timeline turn must belong to one episode or an explicit exclusion. No target episode count. Challenge to a claim made in the preceding answer may remain a same-issue follow-up; document rationale. Unrelated founder monologue after concession must be excluded, not carried into an episode. If a supplied turn combines eligible and excluded material, flag needs_repair rather than guessing. Opening probe=investor question; initial answer=founder speech before substantive investor intervention OR unrelated topic shift. first_uptake=the FIRST subsequent investor move (may be a different investor); preserve later cycles separately, do not overwrite the first answer with a later better/longer answer. Identify any minimal_backchannels by turn ID explicitly; ONLY these may be skipped. Minimal backchannels during an ongoing answer do not end it. Short receipts after a completed answer do qualify. Never use a decision as an uptake. Return all episode turns, components, later_exchanges and exclusions.\nCODEBOOK\n'+definitions+'\nTIMELINE\n'+tl.model_dump_json()
   proposal=exp.call('interaction_map',p,InteractionMap,qa,{'phase':'map'})
   p=source_note+'\nVerify and correct this complete interaction map against the video and fixed timeline. Return COMPLETE corrected map. Ensure issue continuity, exclude unrelated monologues, preserve short social exchanges inside their issue, and distinguish opening/initial/first uptake from later follow-ups. Explicitly check that first_uptake does not skip a substantive investor intervention. Every supplied turn must be assigned or excluded exactly once. If the timeline needs repair, list needs_repair and do not pretend the map can be frozen. No desired episode count or ratings.\nRULES\n'+definitions+'\nTIMELINE\n'+tl.model_dump_json()+'\nPROPOSED MAP\n'+json.dumps(proposal)
   final=exp.call('map_review',p,InteractionMap,qa,{'phase':'map_review'})
   for attempt in range(3):
    mapping=InteractionMap.model_validate(final)
    try:
     bundle=freeze_bundle(mapping,tl,roster,exp.meta['source_sha256'],115,278,'Gemini source-review pass + deterministic structural checks; development only')
     dump(RUN/'frozen_bundle.json',bundle);break
    except ValueError as err:
     dump(RUN/f'map_gate_rejection_{attempt}.json',{'error':str(err),'proposal':final})
     if attempt==2:raise
     correction=source_note+'\nThe map failed its deterministic gate: '+str(err)+'\nRecheck source and return the complete corrected map. first_uptake means ONE earliest investor move, not a group of sequential receipts. minimal_backchannels lists ONLY listener tokens DURING THE INITIAL ANSWER that are skipped to locate the first uptake; later social exchanges remain ordinary episode turns. Do not relabel a substantive intervention as a listener token to pass a check. If attribution/order is genuinely inseparable, use uncertain uptake with empty refs. Preserve substantive boundaries unless source warrants change.\nCODEBOOK\n'+definitions+'\nTIMELINE\n'+tl.model_dump_json()+'\nREJECTED MAP\n'+json.dumps(final)
     final=exp.call(f'map_gate_repair_{attempt+1}',correction,InteractionMap,qa,{'phase':'map_gate_repair','attempt':attempt+1})
  else:
   bundle=read(RUN/'frozen_bundle.json');assert_frozen(bundle)
   mapping=InteractionMap.model_validate(bundle['mapping']);tl=TimelineReview.model_validate(bundle['timeline']);turns={t.turn_idx:t for t in tl.turns}
   def episode(ep):
    start=min(turns[i].start_s for i in ep.turn_refs);end=max(turns[i].end_s for i in ep.turn_refs)
    media_start=max(0,start-10);media=clip(exp.source,media_start,end,RUN/'media')
    context_turns=[t for t in tl.turns if t.start_s<start and t.end_s>media_start]
    ctx={'roster':json.dumps(roster),'pitch_summary':"Copy Keyboard: a two-button copy/paste keyboard. Scored issue: "+ep.topic,
         'timeline_slice':'\n'.join(f't{i} source {turns[i].start_s}–{turns[i].end_s} clip {max(0,turns[i].start_s-media_start):.2f}–{turns[i].end_s-media_start:.2f} {turns[i].speaker}: {turns[i].tag}' for i in ep.turn_refs)}
    p=render_prompt('stage_b_episode.j2',codebook=exp.epbook,context=ctx,prompt_dir=RUN/'templates')
    p+='\nBINDING STRUCTURED WINDOWS\n'+ep.model_dump_json()+'\nALLOWED REFERENCE IDS BY CODE\n'+json.dumps({e.id:sorted(allowed_refs(e.id,ep)) for e in exp.epbook.entries})
    p+=f'\nThe first {start-media_start} seconds are context ONLY; scored episode starts at clip second {start-media_start} and ends at {end-media_start}. Context may only support RP-RECOV baseline. Context turns: '+json.dumps([t.model_dump() for t in context_turns])
    p+='\nFor a codable negative value cite inspected permitted turns. Solo-founder TM codes MUST be uncodable. If an answer is brief apply PREP minimum/applicability BEFORE scoring. Use source video for substantive judgments, but never substitute later answer evidence for the designated initial answer. If the binding map contradicts source, mark affected code uncodable and explain; do not silently choose another window. Unobservable issues: '+json.dumps(bundle['timeline']['issues']+bundle['mapping']['issues'])
    for attempt in range(1,4):
     key=f'episode_{ep.episode_id}_try{attempt}'
     record=exp.call(key,p,exp.epschema,media,{'episode_id':ep.episode_id,'source_start_s':start,'source_end_s':end,'media_start_s':media_start,'timeline_hash':bundle['timeline_hash'],'map_hash':bundle['map_hash']})
     errors=validate_codes(record,ep,bundle,[t.turn_idx for t in context_turns]);dump(RUN/'results'/f'{key}_qc.json',errors)
     if not errors:dump(RUN/'results'/f'accepted_{ep.episode_id}.json',{'record':record,'source_call':key,'status':'structural_checks_passed_not_human_validated'});return
     p+='\nPREVIOUS RESULT REJECTED BY VALIDATOR. Recode from source under the binding windows; do not merely change reference IDs. Violations: '+json.dumps(errors)
    raise ValueError('Episode remains rejected: '+ep.episode_id)
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(episode,mapping.episodes))
   prepared={'turn_idx':1,'speaker':'speaker-2','start_s':16,'end_s':115,'tag':'prepared presentation'}
   eligible={1};units={1:'prepared'}
   for ep in mapping.episodes:
    eligible.update(ep.turn_refs);units.update({i:ep.episode_id for i in ep.turn_refs})
   p='Code PREP-CONSIST only from prepared founder presentation (source 16–115 seconds, t1) and FOUNDER claims in eligible episode turns listed below. Full video is pre-decision 0–278; ignore all excluded gaps. No outside knowledge. Claims need original-video times and speaker/turn eligibility.\n'+json.dumps([e.model_dump(exclude={'provenance','examples'}) for e in exp.pitchbook.entries])+'\nTIMELINE\n'+json.dumps([prepared]+[t.model_dump() for t in tl.turns])+'\nELIGIBLE TURN IDS AND UNIT IDS\n'+json.dumps(units)+'\nEXCLUSIONS\n'+json.dumps([e.model_dump() for e in mapping.excluded])+'\nFor EACH candidate pair give metric, period, units, and scope separately for BOTH claims. Count only if matching scope is established and same normalized fields can be used truthfully. Explain logical incompatibility and check correction/reconciliation. Different initial inventory and eventual shipments are not inherently contradictory. Do not assume a period not stated. Only cross-unit pairs count. Unknown scope preventing an exact total requires uncodable. Return coverage, all inspected eligible turn IDs, comparable-claims flag, rejected as well as counted candidates, limitations. Coverage is model-reported, not human validation.'
   media=clip(exp.source,0,278,RUN/'media')
   for attempt in range(1,3):
    record=exp.call(f'pitch_try{attempt}',p,PitchResult,media,{'eligible_turns':sorted(eligible),'timeline_hash':bundle['timeline_hash'],'map_hash':bundle['map_hash']})
    result=PitchResult.model_validate(record);errors=validate_pitch(result,eligible,units,{1}|{t.turn_idx for t in tl.turns if next(r for r in roster if r["speaker"]==t.speaker)["role"]=="founder"},{1:prepared,**{t.turn_idx:t.model_dump() for t in tl.turns}});dump(RUN/'results'/f'pitch_try{attempt}_qc.json',errors)
    if not errors:dump(RUN/'results/accepted_pitch.json',record);break
    p+='\nValidator rejection: '+json.dumps(errors)
   else:raise ValueError('Pitch remains rejected')
   from scripts.experiments.check_structured_visibility import main as verify_visibility
   verify_visibility()
 finally:exp.close()
if __name__=='__main__':main()
