#!/usr/bin/env python3
"""Recompute v2 versus latest v3b AUC from accepted records; no API calls."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pipeline.stage_b.exchanges import check,validate_ratings
from scripts.experiments.analyze_structured_pilot import mean,feature_value,grouped_auc
RUNS={'v2':ROOT/'runs/stage_b_exchange_pilot5','v3b':ROOT/'runs/stage_b_exchange_pilot5_v3b'}
FEATURES={'response':'RS-RESP','evidence':'RS-EVID','substance':'PREP-SUBS','coherence':'PREP-COH','probe_severity':'PR-SEV','promotion_probe':'PR-FRAME'}
MODELS={'interaction':['response','evidence','substance','coherence','repair','n_exchanges'],'plus_positivity':['positivity','response','evidence','substance','coherence','repair','n_exchanges'],'probe_control':['positivity','response','evidence','substance','coherence','repair','n_exchanges','probe_severity','promotion_probe'],'response_only':['response','evidence','substance','coherence','n_exchanges']}
def read(p):return json.loads(p.read_text())
def load(root,vid):
 r=root/vid;completion=read(r/'completion.json');assert completion['status'] in ('complete','complete_with_flags')
 b=read(r/'frozen_bundle.json');m,t=check(b);records={};issues={};missing=[]
 for s in m.segments:
  if s.kind!='issue':continue
  p=r/f'results/accepted_{s.segment_id}.json'
  if not p.exists():
   flag=read(r/f'results/needs_human_review_{s.segment_id}.json');assert flag['map_hash']==b['map_hash'];missing.append({'segment':s.segment_id,'exchanges':flag['exchange_ids'],'errors':flag['errors']});continue
  a=read(p);assert a['map_hash']==b['map_hash'] and a['timeline_hash']==b['timeline_hash']
  start=min(z.start_s for z in t.turns if z.turn_idx in s.turn_refs);ctx=[z.turn_idx for z in t.turns if z.start_s<start and z.end_s>max(0,start-10)]
  errors=validate_ratings(a,s,b,ctx);assert not errors,(vid,s.segment_id,errors)
  assert len(a['exchanges'])==len([x for x in m.exchanges if x.segment_id==s.segment_id])
  records.update({x['exchange_id']:x['ratings'] for x in a['exchanges']});issues[s.segment_id]=a['episode']
 bounds=read(r/'boundaries.json');prep=read(r/'results/accepted_prepared.json');rows=[]
 for shark in bounds['outcomes_model_reviewed_not_gold']['sharks']:
  xs=[x for x in m.exchanges if x.investor==shark['speaker'] and x.exchange_id in records];own={x.segment_id for x in xs}
  row={'video_id':vid,'speaker':shark['speaker'],'offer':None if shark['ever_offer'] is None else int(shark['ever_offer']),'positivity':prep['pitch_positivity']['value'],'n_exchanges':len(xs),'n_issues':len(own)}
  row.update({k:mean([feature_value(code,records[x.exchange_id][code]) for x in xs]) for k,code in FEATURES.items()});row['repair']=mean([feature_value('RP-ACK',issues[s]['RP-ACK']) for s in sorted(own)]);rows.append(row)
 return rows,{'completion':completion,'accepted_issues':len(issues),'accepted_exchanges':len(records),'mapped_exchanges':len(m.exchanges),'flags':missing,'uncovered_sharks':sum(x['n_exchanges']==0 for x in rows)},b,bounds,prep

def main():
 vids=[v['video_id'] for v in read(ROOT/'runs/stage_b_structured_pilot5/selection.json')['videos']];allrows={};audit={};old={}
 for label,root in RUNS.items():
  rows=[];audit[label]={}
  for vid in vids:
   rr,aa,b,bounds,prep=load(root,vid);rows+=rr;audit[label][vid]=aa
   if label=='v2':old[vid]=(b,bounds,prep)
   else:
    ob,oo,op=old[vid];assert bounds==oo and prep==op;assert b['source_sha256']==ob['source_sha256'];aa['same_timeline']=b['timeline_hash']==ob['timeline_hash'];aa['same_roster']=b['roster']==ob['roster']
  assert len(rows)==25 and sum(r['offer']==1 for r in rows)==6;allrows[label]=rows
 flagged={vid for v in audit.values() for vid,a in v.items() if a['flags']}
 scores={label:{k:grouped_auc(rows,features) for k,features in MODELS.items()} for label,rows in allrows.items()}
 sensitivity={label:{k:grouped_auc([r for r in rows if r['video_id'] not in flagged],features) for k,features in MODELS.items()} for label,rows in allrows.items()}
 out={'runs':{k:str(v) for k,v in RUNS.items()},'audit':audit,'rows':allrows,'models':scores,'exclude_flagged_videos':sorted(flagged),'sensitivity_without_flagged_videos':sensitivity,'method':'Leave-one-video-out ridge logistic L2=1; training-only median imputation and standardization; accepted records only; all 25 outcome rows retained. No parameter tuning.'}
 dest=RUNS['v3b'];(dest/'auc_vs_v2.json').write_text(json.dumps(out,indent=2)+'\n')
 for label,rows in allrows.items():
  with (dest/f'auc_audit_features_{label}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# V2 versus newest v3b: recomputed AUC','','All five videos have finished processing. V3b Slate is complete_with_flags: seg_2 was rejected after x1 UP-TYPE reported a source/map attribution conflict. The whole issue has no accepted ratings, excluding both of its mapped exchanges (x1 and x2) from feature aggregation. Rejected output is not treated as accepted. All 25 shark outcome rows remain (6 offers, 19 no offers).','','V2 = stage_b_exchange_pilot5; newest = stage_b_exchange_pilot5_v3b. Features reconstructed from accepted JSON records, map/timeline hashes and rating gates checked; Stage A, outcomes and cutoffs are identical. All five timeline hashes match, but the Fish Fixe roster differs: v3b lacks the revised visual identity anchors.','','Fixed whole-video holdout, training-only imputation/scaling, ridge penalty 1. Interaction features: responsiveness, evidence, substance, coherence, issue-level repair and exchange count. Response-only omits repair. Probe controls add severity and promotion-probe share. Uptake and retired visual/recovery measures are not predictors.','','| Model | V2 pooled AUC | V3b pooled AUC | Difference | V2 mean within-video | V3b mean within-video |','|---|---|---|---|---|---|']
 for k in MODELS:
  a=scores['v2'][k];b=scores['v3b'][k];lines.append(f"| {k} | {a['pooled_oof_auc']:.3f} | {b['pooled_oof_auc']:.3f} | {b['pooled_oof_auc']-a['pooled_oof_auc']:+.3f} | {a['mean_within_video_auc']:.3f} | {b['mean_within_video_auc']:.3f} |")
 lines+=['','Mean within-video AUC uses four evaluable videos; Slate has no positive outcomes, so its within-video AUC is undefined. It still contributes to pooled AUC and the training folds. Positivity is constant at 5 and adds no discrimination.','','## Coverage','','| Video | V2 accepted exchanges | V3b accepted / mapped | V3b status |','|---|---|---|---|']
 for vid in vids:
  a=audit['v2'][vid];b=audit['v3b'][vid];lines.append(f"| {vid} | {a['accepted_exchanges']} | {b['accepted_exchanges']}/{b['mapped_exchanges']} | {b['completion']['status']} |")
 lines+=['','## Sensitivity: exclude Slate from both runs','','Re-fit the same models on the same remaining four videos; this is a sensitivity check, not a replacement chosen for higher AUC.','','| Model | V2 AUC | V3b AUC |','|---|---|---|']
 for k in MODELS:lines.append(f"| {k} | {sensitivity['v2'][k]['pooled_oof_auc']:.3f} | {sensitivity['v3b'][k]['pooled_oof_auc']:.3f} |")
 lines+=['','Only five pitch clusters and six positive outcomes: changes are descriptive, not evidence of statistically established improvement. New maps/ratings and roster differences prevent attributing AUC changes solely to pruning two codes. The v3b analysis remains provisional pending the flagged source issue. No new API calls were made.','','Full model coefficients, fold predictions, feature rows and validation status: [auc_vs_v2.json](auc_vs_v2.json).']
 (dest/'auc_vs_v2.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
