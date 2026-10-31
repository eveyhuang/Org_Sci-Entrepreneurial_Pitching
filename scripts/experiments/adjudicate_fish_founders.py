"""Bounded source recheck after repeated conflicting founder-attribution reviews."""
import sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.run_exchange_pilot import Experiment,RUN,read,dump,TimelineReview,validate_timeline,digest,clip
r=RUN/'st_ea076612ce';p=r/'results/timeline.json';raw=read(p)
if not (r/'timeline_roster_before_anchors.json').exists():shutil.copy2(p,r/'timeline_roster_before_anchors.json')
for person in raw['record']['roster']:
 if person['speaker']=='speaker-2':person['descriptor']='At source 18s shared frontal shot: VIEWER-LEFT founder; shorter wavy highlighted hair, V-neck white graphic tee, slim vertical pendant necklace. Keep this ID in close-ups; hair color alone is ambiguous.'
 if person['speaker']=='speaker-3':person['descriptor']='At source 18s shared frontal shot: VIEWER-RIGHT founder; longer straighter blonde hair, round/crew-neck white graphic tee, layered round-neck necklaces. Keep this ID in close-ups; hair color alone is ambiguous.'
dump(p,raw)
x=Experiment(r)
try:
 bounds=read(r/'boundaries.json');old=read(r/'reviewed_timeline.json');media=clip(x.source,0,bounds['cutoff_s'],r/'media')
 prompt='Resolve founder speaker attribution from this ORIGINAL SOURCE video, beginning at second 0. Prior reviews swapped the founders; do not infer identity from hair color or a camera cut. Anchor both IDs using the shared frontal shot at source 18s and clothing/necklace details in this roster. Inspect synchronized mouth movement/voice continuity for each founder turn; retain investor IDs. Return the complete Q&A timeline only, '+str(bounds['prepared_end_s'])+'–'+str(bounds['cutoff_s'])+' source seconds. Retain IDs/timing unless source warrants a correction. Explain direct visual evidence for disputed intervals in review_notes. If identity really cannot be resolved, retain the provisional role/ID and explicitly flag unobservable affected TM codes with source intervals; do not invent certainty. No dialogue transcription.\nROSTER\n'+str(raw['record']['roster'])+'\nPROVISIONAL TIMELINE\n'+str(old)
 result=x.call('founder_anchor_adjudication',prompt,TimelineReview,media,{'phase':'identity_adjudication','roster':raw['record']['roster'],'previous_timeline_hash':digest(old)})
 tl=TimelineReview.model_validate(result);validate_timeline(tl,raw['record']['roster'],bounds['prepared_end_s'],bounds['cutoff_s'])
 dump(r/'timeline_before_revision_3.json',old);dump(r/'reviewed_timeline.json',result);dump(r/'timeline_revision.json',{'revision':3,'reason':'Repeated conflicting founder labels; source recheck anchored to clothing and shared opening shot','previous_hash':digest(old),'new_hash':digest(result)})
 print('Anchored founder timeline saved',flush=True)
finally:x.close()
