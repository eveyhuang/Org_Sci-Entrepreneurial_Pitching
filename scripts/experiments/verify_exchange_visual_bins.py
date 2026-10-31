#!/usr/bin/env python3
"""Recheck fixed selected visual slots on short source bins; never resample them."""
import argparse,concurrent.futures,json,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.run_exchange_pilot import Experiment,RUN,read,dump
from pipeline.stage_b.exchanges import VisualInventory,VisualRatings,validate_visual_ratings,digest
from pipeline.media import clip

def work(vid):
 r=RUN/vid;x=Experiment(r)
 try:
  b=read(r/'frozen_bundle.json');inv=VisualInventory.model_validate(read(r/'visual_inventory.json'));code=x.epbook.by_id('VIS-ENGAGE');out=[]
  for idx in sorted({s.bin_index for s in inv.slots}):
   start=b['start_s']+60*idx;end=min(start+60,b['cutoff_s']);slots=[s for s in inv.slots if s.bin_index==idx];subset=VisualInventory(slots=slots)
   shown=[{**s.model_dump(),'clip_start_s':None if s.start_s is None else s.start_s-start,'clip_end_s':None if s.end_s is None else s.end_s-start} for s in slots]
   p='Watch this entire short audiovisual bin. Clip time ZERO equals source '+str(start)+' seconds. Independently verify the fixed selected intervals below, using CLIP_START/END to locate them in this clip. Previous long-video verification was unreliable; no previous ratings supplied. Check actual visible face against roster, continuous readable duration >=2s, synchronized speaking or continuous shared-frame listening, and FIRST qualifying position in this bin. Detached reaction shots are ineligible. Never infer identity from offscreen voice; the named investor must actually be visible throughout the selected interval. Any failed selection: verified=false, value=null, appropriate missing_reason and concrete evidence. Do not replace intervals. Preserve originally missing slots as missing. Rate VIS-ENGAGE only for verified selections. Return one result per slot.\nROSTER\n'+json.dumps(b['roster'])+'\nCODE\n'+json.dumps(code.model_dump(exclude={'provenance','examples'}))+'\nFIXED SLOTS\n'+json.dumps(shown)
   for attempt in range(3):
    raw=x.call(f'visual_bin_verify_{idx}_try{attempt}',p,VisualRatings,clip(x.source,start,end,r/'media'),{'phase':'visual_bin_verification','inventory_hash':digest(inv),'source_start':start,'source_end':end})
    ratings=VisualRatings.model_validate(raw)
    try:validate_visual_ratings(ratings,subset);break
    except ValueError as exc:
     if attempt==2:raise
     p+='\nCorrect schema/linkage error: '+str(exc)
   out.extend(ratings.ratings)
  result=VisualRatings(ratings=out);validate_visual_ratings(result,inv)
  old=r/'results/accepted_visual.json';backup=r/'results/accepted_visual_long_context.json'
  if not backup.exists():
   if old.exists():shutil.copy2(old,backup)
   else:
    candidates=list((r/'results').glob('visual_ratings_*.json'))
    dump(backup,read(max(candidates,key=lambda p:p.stat().st_mtime))['record'])
  dump(old,result.model_dump());dump(r/'visual_bin_verification.json',{'complete':True,'inventory_hash':digest(inv),'bins':len({s.bin_index for s in inv.slots})})
  print('VERIFIED',vid,flush=True)
 finally:x.close()
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--video-ids',nargs='+',required=True);ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(work,a.video_ids))
