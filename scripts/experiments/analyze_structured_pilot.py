#!/usr/bin/env python3
"""Auditable descriptive summaries and video-grouped exploratory ROC AUC for the pilot."""
from __future__ import annotations
import csv,json,sys
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
RUN=ROOT/'runs/stage_b_structured_pilot5'

def read(p):return json.loads(p.read_text())
from analysis.auc import auc, fit_predict, grouped_auc, mean, feature_value

def main():
    selected=read(RUN/'selection.json')['videos'];rows=[];summaries=[];code_values={};global_calls=[]
    founder={'response':'RS-RESP','evidence':'RS-EVID','substance':'PREP-SUBS','coherence':'PREP-COH','repair':'RP-ACK'}
    controls={'probe_severity':'PR-SEV','promotion_probe':'PR-FRAME'}
    for video in selected:
        r=RUN/video['video_id']
        if not (r/'completion.json').exists():raise ValueError(f'Incomplete video: {r.name}')
        b=read(r/'frozen_bundle.json');bounds=read(r/'boundaries.json');out=bounds['outcomes_model_reviewed_not_gold'];prep=read(r/'results/accepted_prepared.json');pitch=read(r/'results/accepted_pitch.json')
        records={e['episode_id']:read(r/f"results/accepted_{e['episode_id']}.json")['record'] for e in b['mapping']['episodes']}
        for rec in records.values():
            for k,v in rec.items():code_values.setdefault(k,[]).append(v)
        for shark in out['sharks']:
            ep=[e for e in b['mapping']['episodes'] if e['probing_speaker']==shark['speaker']]
            row={'video_id':video['video_id'],'speaker':shark['speaker'],'offer':None if shark['ever_offer'] is None else int(shark['ever_offer']),'n_episodes':len(ep),'positivity':prep['pitch_positivity']['value']}
            for name,code in {**founder,**controls}.items():row[name]=mean([feature_value(code,records[e['episode_id']][code]) for e in ep])
            rows.append(row)
        calls=[json.loads(s) for s in (r/'calls.jsonl').read_text().splitlines()];global_calls.extend(calls)
        qc=[read(p) for p in (r/'results').glob('episode_*_qc.json')]
        error_files=list((r/'raw').glob('*_error.json'))
        # Archived pre-resume errors are separate attempts even when a resumed call reused its key.
        error_files+=list((r/'raw_before_resume').glob('*_budget64__attempt*_error.json'))
        audit={'initial_episode_count':len(read(r/'results/segmentation.json')['record']['episodes']),
               'uptake_status_counts':dict(Counter(e['uptake_status'] for e in b['mapping']['episodes'])),
               'episode_qc_rejected_outputs':sum(bool(x) for x in qc),
               'visibility_rejected':sum(read(p).get('visibility_gate_passed') is False for p in (r/'results').glob('accepted_ep*.json')),
               'provider_failed_attempts':len(error_files),
               'provider_errors':dict(Counter(read(p)['error_type'] for p in error_files)),
               'comparisons_present':pitch['comparable_claims_present'],
               'pitch_uncodable':pitch['uncodable']}

        summaries.append({**audit,'video_id':video['video_id'],'title':Path(video['path']).stem,'duration_s':video['duration_s'],'prepared_start':bounds['prepared_start_s'],'prepared_end':bounds['prepared_end_s'],'cutoff':bounds['cutoff_s'],'episodes':len(records),'qa_turns':len(b['timeline']['turns']),'offers':sum(s['ever_offer'] is True for s in out['sharks']),'out':sum(s['ever_offer'] is False for s in out['sharks']),'unknown':sum(s['ever_offer'] is None for s in out['sharks']),'positivity':prep['pitch_positivity']['value'],'pitch_consistency':pitch['value'],'excluded':b['mapping']['excluded'],'calls':len(calls),'gate_rejections':len(list(r.glob('*gate_rejection*.json'))),'source_review_issues':b['timeline']['issues']+b['mapping']['issues']})
    models={'positivity_baseline':['positivity'],'founder_interaction':list(founder)+['n_episodes'],'positivity_plus_interaction':['positivity']+list(founder)+['n_episodes'],'conditional_probe_control':['positivity']+list(founder)+['n_episodes']+list(controls)}
    analysis={name:grouped_auc(rows,features) for name,features in models.items()}
    missing={k:{'n':len(v),'uncodable':sum(x['uncodable'] for x in v),'uncodable_fraction':sum(x['uncodable'] for x in v)/len(v),'values':[x['value'] for x in v if not x['uncodable']]} for k,v in code_values.items()}
    usage={k:sum(c[k] for c in global_calls) for k in ['input_tokens','output_tokens','thinking_tokens']}
    diagnostics={'exposure_only_not_prespecified':grouped_auc(rows,['n_episodes']),'raw_positivity_auc':auc([r['offer'] for r in rows if r['offer'] is not None],[r['positivity'] for r in rows if r['offer'] is not None]),'positivity_unique_values':sorted({r['positivity'] for r in rows})}
    result={'diagnostics':diagnostics,'videos':summaries,'decision_rows':rows,'models':analysis,'code_missingness':missing,'api_calls':len(global_calls),'usage':usage,'cost_at_2_12_usd_per_million':(2*usage['input_tokens']+12*(usage['output_tokens']+usage['thinking_tokens']))/1e6,'limitations':['Five pitch clusters only; exploratory grouped CV, not validated performance.','Outcomes and features extracted by same model family, not independently human-coded.','Missing shark-associated episodes require training-only imputation; coverage may drive discrimination.','Episode-wide repair belongs to opener-associated episode context, not necessarily a repair directed at that shark.','No venture controls, canonical shark identity fixed effects, pitch random effect or causal interpretation.']}
    (RUN/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (RUN/'decision_features.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (RUN/'oof_predictions.csv').open('w',newline='') as f:
        preds=[{'model':name,**p} for name,a in analysis.items() for p in a['predictions']];w=csv.DictWriter(f,fieldnames=list(preds[0]));w.writeheader();w.writerows(preds)
    print(json.dumps({'video_count':len(summaries),'rows':len(rows),'models':{k:{'pooled':v['pooled_oof_auc'],'within':v['mean_within_video_auc']} for k,v in analysis.items()},'usage':usage},indent=2))
if __name__=='__main__':main()
