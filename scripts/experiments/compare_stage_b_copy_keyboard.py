#!/usr/bin/env python3
"""HISTORICAL ROUTE (v2 instrument, archived 2026-10-06). Not current execution.
Isolated, resumable real-API development comparison; never writes the main registry."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, shutil, sys, threading, time
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pydantic import BaseModel,ConfigDict,Field,model_validator
from pipeline.codebook import load_codebook
from pipeline.config import ModelConfig
from pipeline.media import clip
from pipeline.prompts import render_prompt,prompt_hash
from pipeline.providers.base import GenerateRequest
from pipeline.providers.gemini import GeminiProvider
from pipeline.registry import Registry
from pipeline.schemas.generate import build_record_model
from pipeline.schemas.stage_b import SegmentationRecord
from pipeline.schemas.stage_b_timeline import TimelineRecord,normalised_turns,roles,clock_to_seconds,seconds_to_clock
from pipeline.stage_b.context import format_turn

VIDEO_ID='st_1f094ea0ad'
OLD_TIMELINE=ROOT/'runs/stage_b_dev_20260922_011758/parsed/timeline/st_1f094ea0ad__k0__b58e6682.json'
OLD_SEG=ROOT/'runs/stage_b_dev_20260922_023102/parsed/segmentation/st_1f094ea0ad__k0__74d9ab68.json'
OLD_EP=ROOT/'runs/stage_b_fullbook_preview_20260923_010648/parsed/episode'
MODEL=ModelConfig('gemini-3.1-pro-preview',0.0,32768)
LOCK=threading.Lock()
def read(path):return json.loads(Path(path).read_text())
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')
def snapshot(run):
 if (run/'experiment.json').exists():return
 for name in ['codebooks','templates','prompts','results','raw','baseline','media']:(run/name).mkdir(parents=True,exist_ok=True)
 source=next((ROOT/'data/shark tank/downloads').glob('*Copy Keyboard*.mp4'))
 for name in ['segmentation','episode','pitch']:
  for ext in ['md','yaml']:shutil.copy2(ROOT/f'docs/codebooks/archive/stage_b_v2_structured_2026-09-27/{name}.draft.{ext}',run/'codebooks')
 for name in ['stage_b_episode.j2','stage_b_segmentation.j2','stage_b_timeline.j2']:
  shutil.copy2(ROOT/'docs/codebooks/archive/stage_b_v1_episode'/name,run/'templates'/name)
 ep=run/'templates/stage_b_episode.j2';text=ep.read_text().replace('Answer true if the inclusion rule is met anywhere in this episode, otherwise false.','Apply this code only within its specified evidence window. Use true/false when applicable and observable; otherwise mark uncodable.').replace('Answer with a whole number: how many separate times this occurs in the episode (0 if never).','Count only the events eligible under this code. Zero requires adequate observation; use uncodable when the total cannot be established.')
 ep.write_text(text)
 shutil.copy2(OLD_TIMELINE,run/'baseline/timeline.json');shutil.copy2(OLD_SEG,run/'baseline/segmentation.json')
 for p in OLD_EP.glob(f'{VIDEO_ID}*.json'):
  d=read(p);dump(run/f"baseline/episode_{d['episode_idx']}.json",d)
 dump(run/'experiment.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'video_id':VIDEO_ID,'model':MODEL.name,'temperature':0,'replicates':1,'arms':{'matched':'Original four spans and original timeline; revised episode draft and scope-consistent experimental template. No pre-probe padding.','rebuilt':'Fresh timeline, revised segmentation, revised episode draft; labeled 10-second pre-probe context solely for RP-RECOV.','pitch':'Revised pitch draft, prepared segment plus eligible episodes only, pre-decision video; experimental structured pair-evidence wrapper.'},'limitations':['Single development video, k=1; not reliability or accuracy validation.','Matched arm retains old timeline defects intentionally.','Rebuilt arm changes timeline/prompt/media context too; differences cannot be attributed solely to codebook.','No human-corrected reference labels or adjudicated boundary gold standard.'],'codebook_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'codebooks').iterdir()}})
 print('Prepared isolated experiment:',run,flush=True)

class Pair(BaseModel):
 model_config=ConfigDict(extra='forbid')
 first_turn:int;second_turn:int;first_time:str;second_time:str
 referent:str;scope:str;first_claim:str;second_claim:str
 mismatch_type:Literal['numeric','date','negation','other','none']
 counted:bool;reason:str
class Range(BaseModel):
 model_config=ConfigDict(extra='forbid')
 start:str;end:str;kind:Literal['prepared','episode'];episode_index:int|None

def pitch_schema(book):
 rating=build_record_model(book,name='PitchCodeRecord',unit='PITCH')
 class PitchAudit(BaseModel):
  model_config=ConfigDict(extra='forbid')
  record:rating
  coverage_complete:bool
  comparable_claims_present:bool
  inspected_ranges:list[Range]
  comparisons:list[Pair]
  limitations:list[str]
  @model_validator(mode='after')
  def counts(self):
   value=self.record.model_dump(by_alias=True)['PREP-CONSIST']
   if not self.coverage_complete and not value['uncodable']:raise ValueError('Incomplete coverage requires uncodable')
   if not value['uncodable'] and value['value']!=sum(p.counted for p in self.comparisons):raise ValueError('Count differs from listed pairs')
   return self
 return PitchAudit

class Experiment:
 def __init__(self,run):
  from dotenv import load_dotenv
  from google import genai
  load_dotenv(ROOT/'.env');self.run=run;self.meta=read(run/'experiment.json');self.source=Path(self.meta['source'])
  self.reg=Registry(run/'registry.sqlite')
  self.provider=GeminiProvider(genai.Client(api_key=os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY'),http_options={'timeout':300000}),self.reg)
  self.epbook=load_codebook(run/'codebooks/episode.draft.yaml');self.segbook=load_codebook(run/'codebooks/segmentation.draft.yaml');self.pitchbook=load_codebook(run/'codebooks/pitch.draft.yaml')
  self.epschema=build_record_model(self.epbook)
 def call(self,key,prompt,schema,media,info):
  result=self.run/'results'/f'{key}.json'
  if result.exists():return read(result)['record']
  (self.run/'prompts'/f'{key}.txt').write_text(prompt)
  dump(self.run/'prompts'/f'{key}.schema.json',schema.model_json_schema())
  handle=self.provider.upload_media(media)
  for attempt in range(1,4):  # 2 attempts; a 3rd, after a pause, only for service-unavailable / rate-limit errors
   started=time.monotonic();print('Calling',key,'attempt',attempt,flush=True)
   try:
    response=self.provider.generate(GenerateRequest(prompt,handle,schema,MODEL,key))
    record=response.parsed.model_dump(mode='json',by_alias=True)
    (self.run/'raw'/f'{key}__attempt{attempt}.txt').write_text(response.raw_text)
    entry={'key':key,'model_requested':MODEL.name,'model_version':response.model_version,'prompt_hash':prompt_hash(prompt),'media':str(media),'input_tokens':response.input_tokens,'output_tokens':response.output_tokens,'thinking_tokens':response.thinking_tokens,'seconds':round(time.monotonic()-started,2),'info':info,'record':record}
    dump(result,entry)
    with LOCK:
     with (self.run/'calls.jsonl').open('a') as f:f.write(json.dumps({k:v for k,v in entry.items() if k!='record'})+'\n')
    print('Completed',key,'tokens',response.input_tokens,response.output_tokens,response.thinking_tokens,flush=True)
    return record
   except Exception as exc:
    err={'key':key,'attempt':attempt,'error_type':type(exc).__name__,'message':str(exc)[:1800],'seconds':round(time.monotonic()-started,2)}
    dump(self.run/'raw'/f'{key}__attempt{attempt}_error.json',err)
    raw=getattr(exc,'raw_text',None)
    if raw:(self.run/'raw'/f'{key}__attempt{attempt}.txt').write_text(raw)
    print('Failed',key,type(exc).__name__,str(exc)[:120],flush=True)
    transient=any(x in str(exc).lower() for x in ['503','unavailable','429','resource_exhausted'])
    if attempt==3 or (attempt==2 and not transient) or any(x in str(exc).lower() for x in ['api key','permission','billing','recitation']):raise
    time.sleep(60 if transient else 5)
 def timeline(self):
  old=read(self.run/'baseline/timeline.json')['record'];p=(self.run/'templates/stage_b_timeline.j2').read_text()
  p=p.replace('Leave out brief back-channels that do not\n  take the floor ("mm-hm", "right") and non-speech sound.','Retain brief acknowledgments, substantive interjections and overlaps. Non-speech sounds need not become turns, but audible short speech must not disappear.')
  p=p.replace('Start a new turn\n  whenever, and only when, the speaker changes.','Start a new turn whenever the speaker changes, including overlapping entries. Also split a continuous speaker turn at an explicit shift to a different substantive topic, to allow issue boundaries.')
  p+='\nEXPERIMENT ROSTER OVERRIDE\nUse the fixed roster below; do not renumber by first appearance. Update first_heard from the source if needed, but preserve IDs, roles and seat identity. No old timeline, annotations or expected corrections are supplied. Go through the entire source, including short off-camera speech. Overlap may have overlapping times; order turn indices by onset. Do not quote dialogue.\n'+json.dumps(old['roster'],ensure_ascii=False)
  return self.call('timeline',p,TimelineRecord,self.source,{'arm':'rebuilt','fixed_roster':True})
 def segmentation(self):
  tl=read(self.run/'results/timeline.json')['record'];ctx={'roster':'\n'.join(f"{r['speaker']} — {r['role']} — {r['seat']} — {r['descriptor']}" for r in tl['roster']),'timeline':'\n'.join(format_turn(t,roles(tl)) for t in normalised_turns(tl))}
  p=render_prompt('stage_b_segmentation.j2',codebook=self.segbook,context=ctx,prompt_dir=self.run/'templates')
  return self.call('segmentation',p,SegmentationRecord,self.source,{'arm':'rebuilt','timeline_prompt_hash':read(self.run/'results/timeline.json')['prompt_hash']})
 def episode(self,arm,idx):
  if arm=='matched':
   tl=read(self.run/'baseline/timeline.json')['record'];seg=read(self.run/'baseline/segmentation.json')['record']
  else:
   tl=read(self.run/'results/timeline.json')['record'];seg=read(self.run/'results/segmentation.json')['record']
  ep=seg['episodes'][idx];turns=normalised_turns(tl);byid={t['turn_idx']:t for t in turns}
  start=byid[ep['start_turn']]['start_s'];end=byid[ep['end_turn']]['end_s']
  if arm=='rebuilt':end=max(t['end_s'] for t in turns if ep['start_turn']<=t['turn_idx']<=ep['end_turn'])
  decisions=[byid[d['turn']]['start_s'] for d in seg['decisions'] if d['turn'] is not None]
  cutoff=min(decisions) if decisions else clock_to_seconds(tl['video_length'])
  if end>cutoff or start>=cutoff:raise ValueError(f'{arm} episode {idx} overlaps decision cutoff')
  media_start=start if arm=='matched' else max(0,start-10)
  media=clip(self.source,media_start,end,self.run/'media')
  inside=[t for t in turns if ep['start_turn']<=t['turn_idx']<=ep['end_turn']]
  context={'roster':'\n'.join(f"{r['speaker']} — {r['role']} — {r['seat']} — {r['descriptor']}" for r in tl['roster']),'timeline_slice':'\n'.join(format_turn(t,roles(tl),offset=media_start) for t in inside),'pitch_summary':seg['summary']+f"\nEpisode topic: {ep['topic']}\nEP-CUT: {ep['cut']}\nOpening investor: {ep['probing_speaker']}"}
  p=render_prompt('stage_b_episode.j2',codebook=self.epbook,context=context,prompt_dir=self.run/'templates')
  if arm=='rebuilt':
   baseline=[t for t in turns if t['end_s']>media_start and t['start_s']<start]
   p+='\nMEDIA WINDOW OVERRIDE\nThis video begins with CONTEXT ONLY. Scored episode begins at clip '+seconds_to_clock(start-media_start)+' and ends at clip '+seconds_to_clock(end-media_start)+'. All preceding footage is solely potential RP-RECOV baseline, not scored episode behavior. It is not automatically a valid baseline; apply continuity and comparability rules. RP-RECOV alone may cite context turns below. Other codes may cite only episode turns. Turn timestamps are clip-relative; partial context turns may have begun before this clip.\nCONTEXT-ONLY TIMELINE\n'+'\n'.join(format_turn(t,roles(tl),offset=media_start) for t in baseline)
  out=self.call(f'{arm}_episode_{idx}',p,self.epschema,media,{'arm':arm,'episode_index':idx,'episode':ep,'source_start_s':start,'source_end_s':end,'media_start_s':media_start,'context_seconds':start-media_start,'cutoff_s':cutoff})
  valid={t['turn_idx'] for t in inside}
  violations=[]
  for code,val in out.items():
   allowed=valid|({t['turn_idx'] for t in baseline} if arm=='rebuilt' and code=='RP-RECOV' else set())
   if set(val['turn_refs'])-allowed:violations.append({'code':code,'out_of_window_refs':list(set(val['turn_refs'])-allowed)})
  dump(self.run/'results'/f'{arm}_episode_{idx}_reference_qc.json',violations)
  return out
 def pitch(self):
  tl=read(self.run/'results/timeline.json')['record'];seg=read(self.run/'results/segmentation.json')['record'];turns=normalised_turns(tl);byid={t['turn_idx']:t for t in turns}
  cutoff=min(byid[d['turn']]['start_s'] for d in seg['decisions'] if d['turn'] is not None)
  media=clip(self.source,0,cutoff,self.run/'media')
  prep=clock_to_seconds(seg['prepared_end']);eligible=[{'start':'0:00','end':seg['prepared_end'],'kind':'prepared','episode_index':None}]
  for i,e in enumerate(seg['episodes']):eligible.append({'start':seconds_to_clock(byid[e['start_turn']]['start_s']),'end':seconds_to_clock(max(t['end_s'] for t in turns if e['start_turn']<=t['turn_idx']<=e['end_turn'])),'kind':'episode','episode_index':i})
  e=self.pitchbook.by_id('PREP-CONSIST')
  p='You are coding content consistency in the supplied pre-decision source video. Do not quote long dialogue, infer outcomes, use outside knowledge, or assign other episode codes. The video contains some excluded gaps; ONLY the prepared segment and listed episode ranges are eligible.\nCODE PREP-CONSIST\nDefinition: '+e.definition+'\nInclude if: '+e.include_if+'\nExclude if: '+e.exclude_if+'\nELIGIBLE RANGES\n'+json.dumps(eligible)+'\nROSTER\n'+json.dumps(tl['roster'])+'\nSOURCE TIMELINE (original-video times; action tags, not dialogue)\n'+'\n'.join(format_turn(t,roles(tl)) for t in turns if t['start_s']<cutoff)+'\nOUTPUT\nReturn record containing PREP-CONSIST with value/uncodable/evidence/turn_refs, plus coverage_complete, comparable_claims_present, inspected_ranges, comparisons, limitations. For each comparison give both turn IDs and original times, claim paraphrases, referent, scope, mismatch_type, counted and reason. Include informative rejected candidates as well as counted pairs. Counted pairs must sum to a codable count; incomplete coverage requires uncodable. An empty comparison list is permitted if no comparable repeated claims exist. Do not fabricate a pair to populate the list. No annotations or outcomes are supplied.'
  return self.call('pitch',p,pitch_schema(self.pitchbook),media,{'arm':'pitch','cutoff_s':cutoff,'eligible_ranges':eligible})
 def close(self):self.reg.close()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True,type=Path);ap.add_argument('--phase',required=True,choices=['prepare','discovery','segmentation','rebuilt','pitch']);a=ap.parse_args();run=a.run_dir.resolve();snapshot(run)
 if a.phase=='prepare':return
 exp=Experiment(run)
 try:
  if a.phase=='discovery':
   jobs=[('timeline',exp.timeline)]+[(f'matched_episode_{i}',lambda i=i:exp.episode('matched',i)) for i in range(4)]
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures={pool.submit(fn):key for key,fn in jobs};errors=[]
    for f in concurrent.futures.as_completed(futures):
     try:f.result()
     except Exception as exc:errors.append(futures[f]);print('JOB FAILED',futures[f],type(exc).__name__,flush=True)
    if errors:raise RuntimeError('Failed jobs: '+','.join(errors))
  elif a.phase=='segmentation':exp.segmentation()
  elif a.phase=='rebuilt':
   count=len(read(run/'results/segmentation.json')['record']['episodes'])
   if count>12:raise ValueError('Unexpected >12 episodes; inspect segmentation before increasing scope')
   with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(lambda i:exp.episode('rebuilt',i),range(count)))
  elif a.phase=='pitch':exp.pitch()
 finally:exp.close()
if __name__=='__main__':main()
