"""Offline diagnostics before scaling Stage B; no annotation calls or source edits.

Run from the repository root with Python, numpy, pandas, and scipy installed:
    python scripts/analysis/audit_pre_scale.py
Exploratory sensitivity checks, not independent validation or model selection.
"""
from pathlib import Path
import sys, json, hashlib, argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.auc import fit_predict


def loss(y, p):
    p = np.clip(np.asarray(p), 1e-10, 1-1e-10)
    y = np.asarray(y)
    return -(y*np.log(p)+(1-y)*np.log(1-p))


def fit(d, tr, te, features, penalty):
    return fit_predict(d.iloc[tr][features].to_numpy(float), d.iloc[tr].offer.to_numpy(float),
                       d.iloc[te][features].to_numpy(float), penalty)


def independent_predict(d, tr, te, features, penalty=1.):
    x=d.iloc[tr][features].to_numpy(float);z=d.iloc[te][features].to_numpy(float);y=d.iloc[tr].offer.to_numpy(float)
    med=np.array([np.nanmedian(v) if np.isfinite(v).any() else 0 for v in x.T])
    x=np.where(np.isfinite(x),x,med);z=np.where(np.isfinite(z),z,med)
    mu=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1
    x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);z=np.column_stack([np.ones(len(z)),(z-mu)/sd])
    def objective(b):
        eta=x@b;pred=1/(1+np.exp(-np.clip(eta,-35,35)))
        value=np.logaddexp(0,eta).sum()-y@eta+penalty/2*(b[1:]@b[1:])
        gradient=x.T@(pred-y);gradient[1:]+=penalty*b[1:]
        return value,gradient
    b0=np.zeros(x.shape[1]);b0[0]=np.log(y.mean()/(1-y.mean()))
    result=minimize(objective,b0,jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    return 1/(1+np.exp(-np.clip(z@result.x,-35,35))),float(np.max(np.abs(objective(result.x)[1])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='stage_b_sample50');ap.add_argument('--output',default='pre_scale_audit_20261005');args=ap.parse_args()
    run=ROOT/'runs'/args.run;base=run/'analysis/project_v2';out=run/'analysis'/args.output;out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    def read_csv(name):
        p=base/(name+'.csv');inputs.append(p);return pd.read_csv(p)
    d=read_csv('decision_cohort').reset_index(drop=True);e=read_csv('exchanges');v=read_csv('coverage');v=v[v.selected & v.analysis_eligible].copy()
    flags=read_csv('flags');prep=read_csv('prepared')
    features=['positivity','response','evidence','substance','response_missing','evidence_missing','substance_missing']
    masks=features[-3:];y=d.offer.to_numpy(float)
    folds=[(np.flatnonzero(d.video_id.ne(vid)),np.flatnonzero(d.video_id.eq(vid))) for vid in sorted(d.video_id.unique())]
    results=[];predictions=d[['video_id','speaker','offer']].copy();foldrows=[];maxdiff=0.;maxgradient=0.
    models={'intercept':([],1.),'missing_indicators':(masks,1.),'substance_and_missing':(['substance']+masks,1.)}
    for penalty in [.1,1.,10.,100.,1000.]:models['concepts_penalty_'+str(penalty)]=(features,penalty)
    for name,(fs,penalty) in models.items():
        pred=np.empty(len(d))
        for tr,te in folds:
            pred[te],detail=fit(d,tr,te,fs,penalty)
            if name=='concepts_penalty_1.0':
                ref,grad=independent_predict(d,tr,te,fs,penalty)
                maxdiff=max(maxdiff,float(np.max(np.abs(pred[te]-ref))));maxgradient=max(maxgradient,grad)
                foldrows.append({'video_id':d.iloc[te[0]].video_id,'iterations':detail['iterations'],**dict(zip(['intercept']+fs,detail['coefficients_standardized']))})
        predictions[name]=pred
        train_pred,_=fit(d,np.arange(len(d)),np.arange(len(d)),fs,penalty)
        results.append({'model':name,'penalty':penalty,'features':','.join(fs),'held_out_log_loss':loss(y,pred).mean(),'in_sample_log_loss':loss(y,train_pred).mean(),'p_min':pred.min(),'p_max':pred.max()})
    pd.DataFrame(results).to_csv(out/'model_sensitivity.csv',index=False);predictions.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(foldrows).to_csv(out/'fold_coefficients.csv',index=False)
    saved=read_csv('concept_bottleneck/predictions').merge(predictions,on=['video_id','speaker','offer'],validate='one_to_one')
    saved_diff=float(np.max(np.abs(saved.concept_bottleneck-saved['concepts_penalty_1.0'])))
    delta=loss(y,predictions['concepts_penalty_1.0'])-loss(y,predictions.intercept)
    rng=np.random.default_rng(20261005);groups=[te for tr,te in folds]
    draws=[np.mean(delta[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])]) for _ in range(3999)]
    pitch=predictions[['video_id']].copy();pitch['delta']=delta;pitch['cbm_loss']=loss(y,predictions['concepts_penalty_1.0']);pitch['baseline_loss']=loss(y,predictions.intercept)
    pitch=pitch.groupby('video_id').agg(n=('delta','size'),delta_sum=('delta','sum'),delta_mean=('delta','mean'),cbm_loss=('cbm_loss','mean'),baseline_loss=('baseline_loss','mean')).reset_index()
    pitch.sort_values('delta_sum',ascending=False).to_csv(out/'pitch_loss_contributions.csv',index=False)
    # Repeated five-group validation is a split sensitivity check, not a new test set.
    cv=[]
    for repeat in range(20):
        vids=rng.permutation(d.video_id.unique());blocks=np.array_split(vids,5)
        pred=np.empty(len(d));bench=np.empty(len(d))
        for block in blocks:
            te=np.flatnonzero(d.video_id.isin(block));tr=np.flatnonzero(~d.video_id.isin(block))
            pred[te],_=fit(d,tr,te,features,1.);bench[te],_=fit(d,tr,te,[],1.)
        cv.append({'repeat':repeat,'concept_log_loss':loss(y,pred).mean(),'benchmark_log_loss':loss(y,bench).mean(),'difference':np.mean(loss(y,pred)-loss(y,bench))})
    pd.DataFrame(cv).to_csv(out/'group_split_sensitivity.csv',index=False)
    # Subgroup scores retain the original predictions; they do not refit a favorable cohort.
    subgroup=[]
    for name,mask in [('no_accepted_exchange',d.n_exchanges.eq(0)),('has_accepted_exchange',d.n_exchanges.gt(0)),('all_concepts_observed',d[features[:4]].notna().all(axis=1)),('flagged',d.has_flags),('unflagged',~d.has_flags)]:
        subgroup.append({'group':name,'n':int(mask.sum()),'offers':int(d.loc[mask,'offer'].sum()),'concept_log_loss':loss(y[mask],predictions.loc[mask,'concepts_penalty_1.0']).mean(),'benchmark_log_loss':loss(y[mask],predictions.loc[mask,'intercept']).mean()})
    pd.DataFrame(subgroup).to_csv(out/'subgroup_scores.csv',index=False)
    # Source/aggregation audit from accepted JSON; no model inference and no video interpretation.
    inventory=[];countrows=[];aggregation_errors=[];structural=[];prompt_versions={};books={};discarded_firsts=[]
    for row in v.itertuples():
        folder=run/row.video_id
        bp=folder/'frozen_bundle.json';inputs.append(bp);b=json.loads(bp.read_text());ts={t['turn_idx']:t for t in b['timeline']['turns']};accepted={}
        inputs.extend([folder/'boundaries.json',folder/'experiment.json'])
        bounds=json.loads((folder/'boundaries.json').read_text());meta=json.loads((folder/'experiment.json').read_text())
        for name,expected in meta.get('codebook_sha256',{}).items():
            path=folder/'codebooks'/name;inputs.append(path);actual=hashlib.sha256(path.read_bytes()).hexdigest()
            books.setdefault(name,set()).add(actual)
            if actual!=expected:structural.append({'video_id':row.video_id,'problem':'codebook hash mismatch','item':name})
        for template in (folder/'templates').glob('stage_b_exchanges.j2'):
            inputs.append(template);h=hashlib.sha256(template.read_bytes()).hexdigest();prompt_versions[h]=prompt_versions.get(h,0)+1
        for seg in b['mapping']['segments']:
            if seg['kind']!='issue':continue
            path=folder/'results'/('accepted_'+seg['segment_id']+'.json')
            xs=[x for x in b['mapping']['exchanges'] if x['segment_id']==seg['segment_id']]
            if path.exists():
                inputs.append(path);a=json.loads(path.read_text())
                if a['map_hash']!=b['map_hash'] or a['timeline_hash']!=b['timeline_hash']:structural.append({'video_id':row.video_id,'problem':'accepted hash mismatch','item':seg['segment_id']})
                accepted.update({x['exchange_id']:x['ratings'] for x in a['exchanges']})
            elif list((folder/'results').glob('codes_scoped_single_v4_'+seg['segment_id']+'_0_try*.json')):
                discarded_firsts.append({'video_id':row.video_id,'issue':seg['segment_id'],'mapped_exchanges':len(xs)})
            for x in xs:
                refs=x['probe']+x['answer']+x['next_move']
                for ref in refs:
                    if ref not in ts:structural.append({'video_id':row.video_id,'problem':'unknown turn','item':ref})
                answer_seconds=sum(ts[i]['end_s']-ts[i]['start_s'] for i in x['answer'])
                next_different=bool(x['next_move']) and ts[x['next_move'][0]]['speaker']!=x['investor']
                cross=bool(x['next_move']) and x['next_move'][0] not in seg['turn_refs']
                inventory.append({'video_id':row.video_id,'exchange_id':x['exchange_id'],'accepted':path.exists(),'answer_seconds_sum':answer_seconds,'answer_turns':len(x['answer']),'probe_turns':len(x['probe']),'next_different_investor':next_different,'next_cross_issue':cross,'status':x['uptake_status'],'first_in_issue':x['exchange_id']==xs[0]['exchange_id']})
        for investor in d[d.video_id.eq(row.video_id)].itertuples():
            xs=[x for x in b['mapping']['exchanges'] if x['investor']==investor.speaker and x['exchange_id'] in accepted]
            vals={}
            for column,code in [('response','RS-RESP'),('evidence','RS-EVID'),('substance','PREP-SUBS')]:
                vv=[]
                for x in xs:
                    value=accepted[x['exchange_id']][code]
                    if value['uncodable']:continue
                    val=value['value']
                    if code=='RS-RESP':val={'addresses':1.,'partial':.5,'deflects':0.}.get(val)
                    if val is not None:vv.append(float(val))
                val=float(np.mean(vv)) if vv else np.nan;vals[column+'_n']=len(vv)
                if not np.isclose(val,getattr(investor,column),equal_nan=True):aggregation_errors.append((row.video_id,investor.speaker,column))
            countrows.append({'video_id':row.video_id,'speaker':investor.speaker,**vals})
        # Retained observation window versus available discussion after prepared delivery.
        denominator=meta.get('duration_s',bounds['cutoff_s'])-bounds['prepared_end_s']
        v.loc[v.video_id.eq(row.video_id),'observed_qa_fraction']=(bounds['cutoff_s']-bounds['prepared_end_s'])/denominator if denominator>0 else np.nan
        v.loc[v.video_id.eq(row.video_id),'first_decision_kind']=next((event['kind'] for shark in bounds['outcomes_model_reviewed_not_gold']['sharks'] for event in shark['events'] if abs(event['source_s']-bounds['cutoff_s'])<2),'unmatched')
    inv=pd.DataFrame(inventory);inv.to_csv(out/'exchange_structure_inventory.csv',index=False)
    den=pd.DataFrame(countrows);den.to_csv(out/'concept_denominators.csv',index=False)
    v[['video_id','observed_qa_fraction','first_decision_kind','issues_mapped','issues_accepted','exchanges_mapped','exchanges_accepted','active_flags']].to_csv(out/'coverage_audit.csv',index=False)
    pw=prep.pivot(index='video_id',columns='code',values='value');pw.corr(method='spearman').to_csv(out/'shark_tank_delivery_correlations.csv')
    distributions=[]
    for code in ['RS-RESP','RS-EVID','PREP-SUBS','PR-SEV']:
        for value,n in e[code].fillna('MISSING').value_counts().items():distributions.append({'code':code,'value':value,'n':int(n)})
    pd.DataFrame(distributions).to_csv(out/'exchange_distributions.csv',index=False)
    active=flags[flags.active & flags.video_id.isin(v.video_id)]
    summary={'python_version':sys.version.split()[0],'numpy_version':np.__version__,'pandas_version':pd.__version__,'eligible_pitches':len(v),'investors':len(d),'offers':int(y.sum()),'positivity_pitch_counts':{str(k):int(val) for k,val in v.positivity.value_counts().sort_index().items()},'prepared_means':pw.mean().to_dict(),'no_accepted_exchange':int(d.n_exchanges.eq(0).sum()),'missing':d[features[:4]].isna().sum().to_dict(),'perfect_directness':int(d.response.eq(1).sum()),'denominators_at_most_two_including_zero':{c:int(den[c].le(2).sum()) for c in den if c.endswith('_n')},'mapped_issues':int(v.issues_mapped.sum()),'accepted_issues':int(v.issues_accepted.sum()),'mapped_exchanges':int(v.exchanges_mapped.sum()),'accepted_exchanges':len(e),'active_flagged_pitches':int(v.active_flags.gt(0).sum()),'active_flag_records':len(active),'flags_mentioning_source_map':int(active.message.str.contains('source/map').sum()),'accepted_next_different_investor':int(inv.loc[inv.accepted,'next_different_investor'].sum()),'accepted_next_cross_issue':int(inv.loc[inv.accepted,'next_cross_issue'].sum()),'median_observed_qa_fraction':float(v.observed_qa_fraction.median()),'first_decision_kinds':v.first_decision_kind.value_counts().to_dict(),'aggregation_mismatches':aggregation_errors,'archive_basic_structural_errors':structural,'archived_template_versions':prompt_versions,'archived_codebook_versions':{k:len(val) for k,val in books.items()},'independent_scipy_max_probability_difference':maxdiff,'independent_scipy_max_abs_gradient':maxgradient,'saved_model_max_probability_difference':saved_diff,'concept_minus_benchmark_log_loss':float(delta.mean()),'conditional_fixed_prediction_pitch_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),'pitches_where_concept_loss_lower':int(pitch.delta_mean.lt(0).sum()),'repeated_group_cv_mean_difference':float(pd.DataFrame(cv).difference.mean()),'repeated_group_cv_range_difference':[float(pd.DataFrame(cv).difference.min()),float(pd.DataFrame(cv).difference.max())],'first_batch_attempts_in_unaccepted_issues':len(discarded_firsts),'limitations':['No source-video human verification.','Basic reference/hash audit is not a semantic validation of maps.','Penalty and split checks are exploratory, not nested selection or independent confirmation.','Bootstrap fixes predictions and does not refit models.','QA fraction denominator includes all remaining source video, possibly closing narration.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'input_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},indent=2)+'\n')
    print(json.dumps(summary,indent=2));print(pd.DataFrame(results).to_string(index=False));print('Wrote',out)

if __name__=='__main__':main()
