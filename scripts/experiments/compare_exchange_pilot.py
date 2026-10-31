#!/usr/bin/env python3
"""Compare V1 opener-only and V2 exchange-level outputs without outcome-driven selection."""
from __future__ import annotations
import csv,json,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments.analyze_structured_pilot import grouped_auc,mean,feature_value
from pipeline.stage_b.exchanges import check,validate_ratings,VisualInventory,VisualRatings,validate_visual,validate_visual_ratings
BASE=ROOT/'runs/stage_b_structured_pilot5';RUN=ROOT/'runs/stage_b_exchange_pilot5'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def csvwrite(p,rows):
 if rows:
  with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def n(x):return 'NA' if x is None else f'{round(x,12):.3f}'
def esc(x):return str(x).replace('|','\\|').replace('\n',' ')
def models(rows):
 response=['response','evidence','substance','coherence'];full=response+['repair','n_exchanges']
 return {name:grouped_auc(rows,features) for name,features in {'interaction':full,'plus_positivity':['positivity']+full,'probe_control':['positivity']+full+['probe_severity','promotion_probe'],'response_only':response+['n_exchanges']}.items()}
def main():
 old=read(BASE/'analysis.json');videos=read(BASE/'selection.json')['videos'];rows=[];bridge=[];summaries=[];matched=[];reclassified=[];exlong=[];eplong=[];vislong=[];calls=[];codability=defaultdict(list)
 fcodes={'response':'RS-RESP','evidence':'RS-EVID','substance':'PREP-SUBS','coherence':'PREP-COH','probe_severity':'PR-SEV','promotion_probe':'PR-FRAME'}
 for v in videos:
  r=RUN/v['video_id'];prev=BASE/v['video_id'];assert read(r/'completion.json')['status']=='complete'
  b=read(r/'frozen_bundle.json');m,t=check(b);ts={x.turn_idx:x for x in t.turns};bm=read(prev/'frozen_bundle.json')['mapping'];bounds=read(r/'boundaries.json');assert bounds==read(prev/'boundaries.json')
  assert read(r/'results/accepted_prepared.json')==read(prev/'results/accepted_prepared.json')
  meta=read(r/'experiment.json');source_hash=hashlib.file_digest(Path(meta['source']).open('rb'),'sha256').hexdigest();assert source_hash==b['source_sha256']==meta['source_sha256']
  for name,h in read(r/'experiment.json')['codebook_sha256'].items():assert hashlib.sha256((r/'codebooks'/name).read_bytes()).hexdigest()==h
  records={};erecords={};openers=[]
  for s in m.segments:
   if s.kind!='issue':continue
   a=read(r/f'results/accepted_{s.segment_id}.json');assert a['map_hash']==b['map_hash']
   start=min(ts[i].start_s for i in s.turn_refs);ctx=[x.turn_idx for x in t.turns if x.start_s<start and x.end_s>max(0,start-10)]
   assert not validate_ratings(a,s,b,ctx),s.segment_id
   records.update({x['exchange_id']:x['ratings'] for x in a['exchanges']});erecords[s.segment_id]=a['episode']
   sx=[x for x in m.exchanges if x.segment_id==s.segment_id];openers.append(min(sx,key=lambda x:min(ts[i].start_s for i in x.probe)).exchange_id)
   for code,value in a['episode'].items():
    codability[code].append(value);eplong.append({'video_id':r.name,'segment_id':s.segment_id,'code':code,**value,'turn_refs':json.dumps(value['turn_refs'])})
  for x in m.exchanges:
   for code,value in records[x.exchange_id].items():
    codability[code].append(value);exlong.append({'video_id':r.name,'segment_id':x.segment_id,'exchange_id':x.exchange_id,'investor':x.investor,'probe':json.dumps(x.probe),'answer':json.dumps(x.answer),'next_move':json.dumps(x.next_move),'code':code,**value,'turn_refs':json.dumps(value['turn_refs'])})
  # Match the same probe, then separate answer-window changes from rating changes.
  for e in bm['episodes']:
   functions={z.turn_ref:z.kind for z in m.investor_moves}
   labels=[functions.get(i,'retired_or_founder') for i in e['opening_probe']]
   if set(labels)!={'probe'}:reclassified.append({'video_id':r.name,'old_episode':e['episode_id'],'opening_probe':e['opening_probe'],'new_functions':labels})
   matches=[x for x in m.exchanges if set(x.probe)==set(e['opening_probe'])]
   if len(matches)!=1:continue
   x=matches[0];before=read(prev/f"results/accepted_{e['episode_id']}.json")['record']
   for code,after in records[x.exchange_id].items():
    bv=before[code];matched.append({'video_id':r.name,'old_episode':e['episode_id'],'new_exchange':x.exchange_id,'code':code,'same_answer':set(x.answer)==set(e['initial_answer']),'same_investor':x.investor==e['probing_speaker'],'old_uncodable':bv['uncodable'],'new_uncodable':after['uncodable'],'old_value':bv['value'],'new_value':after['value']})
  outcomes=bounds['outcomes_model_reviewed_not_gold']['sharks'];positivity=read(r/'results/accepted_prepared.json')['pitch_positivity']['value']
  def decision_rows(only_openers):
   result=[]
   for shark in outcomes:
    xs=[x for x in m.exchanges if x.investor==shark['speaker'] and (not only_openers or x.exchange_id in openers)]
    own={x.segment_id for x in xs};row={'video_id':r.name,'speaker':shark['speaker'],'offer':None if shark['ever_offer'] is None else int(shark['ever_offer']),'n_exchanges':len(xs),'n_issues':len(own),'positivity':positivity}
    for name,code in fcodes.items():row[name]=mean([feature_value(code,records[x.exchange_id][code]) for x in xs])
    row['repair']=mean([feature_value('RP-ACK',erecords[i]['RP-ACK']) for i in sorted(own)])
    row['participation_status']='substantive_probes' if xs else ('other_participation' if any(ts[z.turn_ref].speaker==shark['speaker'] for z in m.investor_moves) else 'no_observed_participation')
    result.append(row)
   return result
  newrows=decision_rows(False);rows.extend(newrows);bridge.extend(decision_rows(True))
  assert read(r/'visual_bin_verification.json')['complete']
  inv=VisualInventory.model_validate(read(r/'visual_inventory.json'));vr=VisualRatings.model_validate(read(r/'results/accepted_visual.json'));validate_visual(inv,b['roster'],b['start_s'],b['cutoff_s']);validate_visual_ratings(vr,inv)
  slots={(x.investor,x.bin_index):x for x in inv.slots}
  for x in vr.ratings:
   s=slots[x.investor,x.bin_index];vislong.append({'video_id':r.name,**s.model_dump(),'selection_evidence':s.evidence,'verified':x.verified,'value':x.value,'rating_evidence':x.evidence,'rating_missing_reason':x.missing_reason})
  vc=[x for x in vr.ratings if x.value is not None];covered={x.investor for x in vc};uptake=[records[x.exchange_id]['UP-TYPE'] for x in m.exchanges]
  oldv=next(z for z in old['videos'] if z['video_id']==r.name)
  sm={'video_id':r.name,'title':v['path'].split('/')[-1].removesuffix('.mp4'),'old_episodes':oldv['episodes'],'new_episodes':len(erecords),'exchanges':len(m.exchanges),'old_uptake_missing':sum(read(prev/f"results/accepted_{e['episode_id']}.json")['record']['UP-TYPE']['uncodable'] for e in bm['episodes']),'new_uptake_missing':sum(z['uncodable'] for z in uptake),'cross_issue_uptake_links':sum(bool(set(x.next_move)-set(next(s.turn_refs for s in m.segments if s.segment_id==x.segment_id))) for x in m.exchanges),'old_uncovered_sharks':sum(z['n_episodes']==0 for z in old['decision_rows'] if z['video_id']==r.name),'new_uncovered_sharks':sum(z['n_exchanges']==0 for z in newrows),'old_uncovered_offer_makers':sum(z['n_episodes']==0 and z['offer']==1 for z in old['decision_rows'] if z['video_id']==r.name),'new_uncovered_offer_makers':sum(z['n_exchanges']==0 and z['offer']==1 for z in newrows),'visual_long_context_codable':sum(z['value'] is not None for z in read(r/'results/accepted_visual_long_context.json')['ratings']),'visual_slots':len(vr.ratings),'visual_codable':len(vc),'visual_covered_sharks':len(covered),'functions':dict(Counter(x.kind for x in m.investor_moves)),'source_revision':read(r/'timeline_revision.json')['revision'] if (r/'timeline_revision.json').exists() else 0,'pitch_consistency':read(r/'results/accepted_pitch.json')['value']}
  summaries.append(sm);calls.extend(json.loads(l) for l in (r/'calls.jsonl').read_text().splitlines())
  # Human-readable complete record with explicit actor-specific windows.
  lines=['# '+sm['title'],'','V2 development annotations; model-reviewed, not human gold.','']
  for s in m.segments:
   lines += [f'## {s.segment_id}: {s.topic} [{s.kind}]','',s.rationale,'']
   for x in [x for x in m.exchanges if x.segment_id==s.segment_id]:
    lines += [f'### {x.exchange_id} — {x.investor}',f'Probe {x.probe}; answer {x.answer}; next move {x.next_move} ({x.uptake_status}).',x.eligibility,'','| Code | Value | Missing reason | Refs | Evidence |','|---|---|---|---|---|']
    for k,z in records[x.exchange_id].items():lines.append('| '+' | '.join(map(esc,[k,z['value'],z['missing_reason'],z['turn_refs'],z['evidence']]))+' |')
   if s.segment_id in erecords:
    lines+=['','Issue-wide ratings:','', '| Code | Value | Missing reason | Refs | Evidence |','|---|---|---|---|---|']
    for k,z in erecords[s.segment_id].items():lines.append('| '+' | '.join(map(esc,[k,z['value'],z['missing_reason'],z['turn_refs'],z['evidence']]))+' |')
  (r/'annotation.md').write_text('\n'.join(lines)+'\n')
 phase_costs=defaultdict(float)
 for call in calls:phase_costs[call['info']['phase']]+=(2*call['input_tokens']+12*(call['output_tokens']+call['thinking_tokens']))/1e6
 usage={k:sum(x[k] for x in calls) for k in ['input_tokens','output_tokens','thinking_tokens']};cost=(2*usage['input_tokens']+12*(usage['output_tokens']+usage['thinking_tokens']))/1e6
 lengths={}
 for name in ['segmentation','episode','pitch']:
  before=(RUN/f'baseline/{name}.draft.md').read_text();after=(RUN/videos[0]['video_id']/f'codebooks/{name}.draft.md').read_text();lengths[name]={'before_words':len(before.split()),'after_words':len(after.split())}
 prompt_lengths={}
 for label,root,pattern in [('old_episode',BASE,'episode_*try1.txt'),('new_batch',RUN,'codes_*try0.txt')]:
  ps=[p for v in videos for p in (root/v['video_id']/'prompts').glob(pattern)];prompt_lengths[label]={'n':len(ps),'mean_words':mean([len(p.read_text().split()) for p in ps]),'mean_characters':mean([len(p.read_text()) for p in ps]),'total_words':sum(len(p.read_text().split()) for p in ps)}
 res={'videos':summaries,'rows':rows,'bridge_rows':bridge,'models_all_exchanges':models(rows),'models_openers_bridge':models(bridge),'old_models':old['models'],'matched':matched,'reclassified_old_openers':reclassified,'missingness':{k:{'n':len(v),'missing':sum(z['uncodable'] for z in v),'reasons':dict(Counter(z['missing_reason'] for z in v if z['uncodable']))} for k,v in codability.items()},'visual_selected_retention':{str(sec):sum(z['value'] is not None and z['end_s']-z['start_s']>=sec for z in vislong) for sec in [2,3,5]},'codebook_lengths':lengths,'prompt_lengths':prompt_lengths,'usage':usage,'successful_calls':len(calls),'cost_usd_successful_only':cost,'successful_cost_by_phase':dict(phase_costs),'failed_attempts':sum(len(list((RUN/v['video_id']/'raw').glob('*_error.json'))) for v in videos),'failure_types':dict(Counter(read(p)['error_type'] for v in videos for p in (RUN/v['video_id']/'raw').glob('*_error.json')))}
 dump(RUN/'comparison.json',res);csvwrite(RUN/'decision_features.csv',rows);csvwrite(RUN/'exchange_codes.csv',exlong);csvwrite(RUN/'episode_codes.csv',eplong);csvwrite(RUN/'visual_observations.csv',vislong);csvwrite(RUN/'matched_probe_comparison.csv',matched)
 lines=['# Five-video exchange revision comparison','','Same videos, Stage A v2, outcome labels and pre-first-decision cutoffs. New real audiovisual map, coding, visual and consistency calls. Old results are preserved. This is development on previously examined videos, not independent accuracy validation.','','## Coverage','','| Video | Old issues → new issues | Exchanges | Uptake missing: old / new | Cross-issue links | Sharks without probes: old → new | Visual slots scored |','|---|---|---|---|---|---|---|']
 for v in summaries:lines.append('| '+' | '.join(map(esc,[v['title'],f"{v['old_episodes']} → {v['new_episodes']}",v['exchanges'],f"{v['old_uptake_missing']}/{v['old_episodes']} → {v['new_uptake_missing']}/{v['exchanges']}",v['cross_issue_uptake_links'],f"{v['old_uncovered_sharks']} → {v['new_uncovered_sharks']}",f"{v['visual_codable']}/{v['visual_slots']}"]))+' |')
 lines+=['','Issue and exchange denominators differ. Matched-probe comparisons below isolate changes for the same opening question. A source-frame audit rejected several Wendy’s Gnome Shop long-context visual verifications. Final visual values therefore use a separate 60-second-bin audiovisual recheck of fixed selections, preserving invalid slots as missing without resampling. Local VIS-ENGAGE measures sampled speaking/listening displays; increased codability is not validation of the old episode-wide UP-AFFECT construct.','','## AUC: fixed whole-video holdout','','| Model | V1 opener-only | V2 new issue openers | V2 all substantive exchanges | V2 all: mean within-video |','|---|---|---|---|---|']
 oldkeys={'interaction':'founder_interaction','plus_positivity':'positivity_plus_interaction','probe_control':'conditional_probe_control'}
 for k,m in res['models_all_exchanges'].items():lines.append('| '+' | '.join([k,n(old['models'][oldkeys[k]]['pooled_oof_auc']) if k in oldkeys else 'not prespecified in V1',n(res['models_openers_bridge'][k]['pooled_oof_auc']),n(m['pooled_oof_auc']),n(m['mean_within_video_auc'])])+' |')
 lines+=['','Same fixed ridge penalty and training-only imputation/scaling. Exposure now counts exchanges. Repair is averaged over unique issues in which the investor probes; it remains shared issue context, not necessarily repair directed to that shark. The response-only model removes this attribution ambiguity. Positivity is 5/5 throughout, so adding it supplies no information. No uptake/visual/decision features enter models. Five pitch clusters and six offers are insufficient for validated performance or causal claims.','','## Matched opening probes','']
 um=[z for z in matched if z['code']=='UP-TYPE'];recovered=sum(z['old_uncodable'] and not z['new_uncodable'] for z in um);lost=sum(not z['old_uncodable'] and z['new_uncodable'] for z in um)
 lines += [f'Matched by identical probe turn-ID set: {len(um)} old openings. Uptake recovered: {recovered}; newly missing: {lost}. Same-answer matches: {sum(z["same_answer"] for z in um)}; same-investor matches: {sum(z["same_investor"] for z in um)}. Values and changed answer windows are retained in matched_probe_comparison.csv.','', '| Code | Missing / observations | Reasons |','|---|---|---|']
 for k,v in res['missingness'].items():lines.append(f"| {k} | {v['missing']}/{v['n']} | {esc(v['reasons'])} |")
 lines+=['','## Visual verification audit','','| Video | Long-context candidate scores | Short-bin verified scores | Fixed slots |','|---|---|---|---|']
 for v in summaries:lines.append('| '+' | '.join(map(esc,[v['title'],v['visual_long_context_codable'],v['visual_codable'],v['visual_slots']]))+' |')
 lines+=['','The long-context column includes the final rejected candidate when no valid long-context output existed. Short-bin verification can reject a selected interval but cannot search for a replacement. These are conservative results from a failed selection/verification workflow, not estimates of all usable visual footage in the videos.']
 lines+=['','## Length and usage','','| Codebook | Before words | After words |','|---|---|---|']
 for k,v in lengths.items():lines.append(f"| {k} | {v['before_words']} | {v['after_words']} |")
 lines += ['',f'Actual first-attempt rating prompt lengths: {prompt_lengths}. Early prompts batched up to four exchanges; later recovery calls use one exchange with exact evidence constraints. These counts include attempted first prompts, not only accepted results; totals reflect expanded coverage and development overhead rather than a one-to-one comparison.',f'Successful calls: {len(calls)}; usage: {usage}. Approximate cost at standard $2/M input and $12/M output including thinking: ${cost:.2f}; failed-call usage is not included. Failed attempts recorded: {res["failed_attempts"]}, including client-side schema failures: {res["failure_types"]}. Interrupted requests may also have unreported usage.', '',f'Fixed selected-observation retention at 2/3/5 seconds: {res["visual_selected_retention"]}. This does not resample later intervals at higher thresholds and is not a validated threshold choice.','','## Artifacts','']
 for v in summaries:lines.append(f'- [{v["title"]}]({v["video_id"]}/annotation.md)')
 lines+=['','[Structured comparison](comparison.json), [exchange codes](exchange_codes.csv), [visual observations](visual_observations.csv), [matched probes](matched_probe_comparison.csv), [analysis plan](comparison_plan.md).']
 notes=RUN/'comparison_interpretation.md'
 if notes.exists():lines+=['',notes.read_text()]
 (RUN/'report.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'videos':summaries,'models':{k:v['pooled_oof_auc'] for k,v in res['models_all_exchanges'].items()},'cost':cost},indent=2))
if __name__=='__main__':main()
