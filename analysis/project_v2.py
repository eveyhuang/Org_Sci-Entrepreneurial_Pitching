"""Read-only, exploratory analyses for project_plan_v2. No annotation API calls."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.auc import grouped_auc, auc

# Exposure/missingness is held constant across the two H1 models.
BASE = ['positivity', 'n_exchanges', 'no_exchange', 'response_missing', 'evidence_missing', 'substance_missing']
RESPONSE = ['response', 'evidence', 'substance']
MODELS = {'intercept': [], 'baseline': BASE, 'response_added': BASE + RESPONSE,
          'probe_baseline': BASE + ['probe_severity', 'promotion_probe'],
          'probe_response_added': BASE + ['probe_severity', 'promotion_probe'] + RESPONSE}

def tables(data, exclude_flagged=False):
    v=data.videos.query('selected and analysis_eligible').copy()
    if exclude_flagged and not data.flags.empty:
        v=v[~v.video_id.isin(data.flags.loc[data.flags.active,'video_id'])]
    ids=set(v.video_id)
    r=data.ratings[data.ratings.video_id.isin(ids)].copy()
    e=data.exchanges[data.exchanges.video_id.isin(ids)&data.exchanges.accepted].copy()
    if e.empty:raise ValueError('No accepted exchanges in the chosen analysis cohort.')
    required={'RS-RESP','RS-EVID','RS-FRAME','PR-FRAME','PR-SEV','UP-TYPE','PREP-SUBS'}
    missing=required-set(r.code)
    if missing:raise ValueError(f'Run lacks required v2 construct codes: {sorted(missing)}')
    values=r[r.unit.eq('EXCHANGE')].copy()
    values.loc[values.uncodable,'value']=None
    wide=values.pivot(index=['video_id','exchange_id'],columns='code',values='value').reset_index()
    e=e.merge(wide,on=['video_id','exchange_id'],validate='one_to_one')
    e['direct']=e['RS-RESP'].map({'addresses':1.,'partial':0.,'deflects':0.})
    e['evidence_binary']=pd.to_numeric(e['RS-EVID'],errors='coerce')
    e['promotion_answer']=e['RS-FRAME'].map({'promotion':1.,'prevention':0.})
    e['promotion_question']=e['PR-FRAME'].map({'promotion':1.,'prevention':0.})
    e['severity']=pd.to_numeric(e['PR-SEV'],errors='coerce')
    e['accept']=e['UP-TYPE'].map({'accepts':1.,'follows_up':0.})
    d=data.decisions[data.decisions.video_id.isin(ids)&data.decisions.offer.notna()].copy()
    d['no_exchange']=d.n_exchanges.eq(0).astype(float)
    for code in RESPONSE:d[code+'_missing']=d[code].isna().astype(float)
    return v,r,e,d

def score(y,p):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),1e-10,1-1e-10)
    return {'auc':auc(y,p), 'brier':float(np.mean((p-y)**2)),
            'log_loss':float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'mean_prediction':float(p.mean()),'observed_rate':float(y.mean())}

def prediction_analysis(d, reps=1999, seed=42):
    if d.video_id.nunique()<3 or d.offer.nunique()<2:
        return pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),{'status':'Insufficient pitches or outcome classes'}
    rows=d.where(pd.notna(d),None).to_dict('records');pred=d[['video_id','speaker','offer']].copy();details={};metrics=[]
    for name,features in MODELS.items():
        result=grouped_auc(rows,features);details[name]=result
        pp=pd.DataFrame(result['predictions']).rename(columns={'prediction':name})
        pred=pred.merge(pp[['video_id','speaker',name]],on=['video_id','speaker'],validate='one_to_one')
        metrics.append(dict(model=name,n=len(d),pitches=d.video_id.nunique(),**score(pred.offer,pred[name])))
    groups=[g.index.to_numpy() for _,g in pred.groupby('video_id')];rng=np.random.default_rng(seed)
    pairs=[('baseline','response_added'),('probe_baseline','probe_response_added')];out=[]
    for baseline,augmented in pairs:
        obs0=score(pred.offer,pred[baseline]);obs1=score(pred.offer,pred[augmented]);samples={k:[] for k in ['auc','brier','log_loss']}
        for _ in range(reps):
            ix=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))]);g=pred.iloc[ix]
            a,b=score(g.offer,g[baseline]),score(g.offer,g[augmented])
            for key in samples:
                if a[key] is not None and b[key] is not None:samples[key].append(b[key]-a[key])
        for key,arr in samples.items():
            low,high=np.quantile(arr,[.025,.975]) if arr else (np.nan,np.nan)
            out.append(dict(comparison=augmented+' minus '+baseline,metric=key,delta=obs1[key]-obs0[key] if obs1[key] is not None and obs0[key] is not None else np.nan,ci_low=low,ci_high=high,bootstrap_valid=len(arr),better='positive' if key=='auc' else 'negative',uncertainty='pitch bootstrap of FIXED OOF predictions; conditional, does not refit training'))
    return pd.DataFrame(metrics),pd.DataFrame(out),pred,details

def _cluster_fit(x,y,cluster):
    n,k=x.shape;g=len(np.unique(cluster));rank=np.linalg.matrix_rank(x)
    if rank<k or n-g-k<=0:raise ValueError('Within-pitch design is rank deficient or lacks residual degrees of freedom')
    inv=np.linalg.inv(x.T@x);b=inv@x.T@y;u=y-x@b
    sums=np.zeros((g,k));np.add.at(sums,cluster,x*u[:,None])
    cov=inv@(sums.T@sums)@inv * g/(g-1)*(n-1)/(n-g-k)
    return b,np.sqrt(np.maximum(np.diag(cov),0)),u

def fe_test(frame,outcome,predictors,target,reps=1999,seed=42):
    """Linear probability association with pitch FE and restricted wild-cluster bootstrap-t.

    Rademacher weights by pitch. Null imposed for the target coefficient, nuisance
    coefficients refitted. Other issues/sharks within a pitch remain together.
    Estimates are probability differences, not odds ratios or causal effects.
    """
    cols=['video_id',outcome]+predictors
    d=frame[cols].dropna().copy()
    for c in [outcome]+predictors:d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna();counts=d.groupby('video_id')[target].nunique()
    result={'outcome':outcome,'target':target,'controls':', '.join(p for p in predictors if p!=target),
            'n':len(d),'pitches':d.video_id.nunique(),'target_varying_pitches':int((counts>1).sum()),
            'events':int(d[outcome].sum()),'nonevents':int((1-d[outcome]).sum()),'method':'pitch FE LPM; restricted wild-cluster bootstrap-t'}
    if d.video_id.nunique()<10 or (counts>1).sum()<5 or min(result['events'],result['nonevents'])<10:
        return dict(result,status='not estimable: fewer than 10 pitches/events/non-events or 5 within-pitch varying target clusters')
    centered=d[[outcome]+predictors]-d.groupby('video_id')[[outcome]+predictors].transform('mean')
    x=centered[predictors].to_numpy(float);y=centered[outcome].to_numpy(float);cluster=pd.factorize(d.video_id)[0];j=predictors.index(target)
    try:b,se,u=_cluster_fit(x,y,cluster)
    except ValueError as err:return dict(result,status=str(err))
    if se[j]<1e-12:return dict(result,status='not estimable: zero cluster standard error')
    nuisance=np.delete(x,j,axis=1);fit0=nuisance@np.linalg.lstsq(nuisance,y,rcond=None)[0] if nuisance.shape[1] else np.zeros(len(y));u0=y-fit0
    rng=np.random.default_rng(seed);null_t=[];ci_t=[]
    for _ in range(reps):
        w=rng.choice([-1.,1.],size=len(np.unique(cluster)))[cluster]
        bb,ss,_=_cluster_fit(x,fit0+u0*w,cluster)
        if ss[j]>1e-12:null_t.append(bb[j]/ss[j])
        bc,sc,_=_cluster_fit(x,x@b+u*w,cluster)
        if sc[j]>1e-12:ci_t.append((bc[j]-b[j])/sc[j])
    qlo,qhi=np.quantile(ci_t,[.025,.975]);t=b[j]/se[j]
    return dict(result,status='estimated',estimate=b[j],cluster_se=se[j],ci_low=b[j]-qhi*se[j],ci_high=b[j]-qlo*se[j],p_two_sided=(1+np.sum(np.abs(null_t)>=abs(t)))/(len(null_t)+1),bootstrap_valid=len(null_t))

def association_tests(e,reps=1999,seed=42):
    # No original-questioner uptake is inferred from NONE or another investor's move.
    uptake=e[e.answered & e.next_same_investor & e.uptake_status.eq('observed') & e['UP-TYPE'].isin(['accepts','follows_up'])]
    tests=[]
    for target in ['direct','evidence_binary']:
        z=fe_test(uptake,'accept',['direct','evidence_binary','severity'],'%s'%target,reps,seed)
        z['test']='RQ2_'+target;tests.append(z)
    z=fe_test(e,'promotion_answer',['promotion_question','severity'],'promotion_question',reps,seed);z['test']='H3_non_neutral';tests.append(z)
    # Sensitivity explicitly changes the answer estimand: promotion vs all codable other frames.
    sensitivity=e.copy();sensitivity['promotion_any']=sensitivity['RS-FRAME'].map({'promotion':1.,'prevention':0.,'NEUTRAL':0.})
    z=fe_test(sensitivity,'promotion_any',['promotion_question','severity'],'promotion_question',reps,seed);z['test']='H3_neutral_answers_included';tests.append(z)
    out=pd.DataFrame(tests);out['p_holm_primary']=np.nan
    idx=out.index[out.test.isin(['RQ2_direct','RQ2_evidence_binary','H3_non_neutral']) & out.status.eq('estimated')]
    if len(idx):
        order=out.loc[idx,'p_two_sided'].sort_values().index;adjusted=np.maximum.accumulate([min(1.,out.loc[i,'p_two_sided']*(len(order)-j)) for j,i in enumerate(order)])
        out.loc[order,'p_holm_primary']=adjusted
    return out,uptake

def stage_a_repeatability(run):
    """Resolve cached replicates from exported annotations table, not parsed-file counts."""
    if run is None:return pd.DataFrame()
    p=Path(run)/'tables/annotations.parquet'
    if not p.exists():return pd.DataFrame([{'status':'annotations.parquet missing','run':str(run)}])
    a=pd.read_parquet(p);wanted=['pitch_positivity','expressed_enthusiasm','interpersonal_warmth'];a=a[a.code_id.isin(wanted)]
    if a.empty:return pd.DataFrame([{'status':'No current perceived-delivery codes in configured Stage A run'}])
    # Different prompt versions are reported separately, never silently pooled.
    out=[]
    for (prompt,code),g in a.groupby(['prompt_hash','code_id']):
        g=g.copy();g['numeric']=pd.to_numeric(g.value,errors='coerce');g.loc[g.uncodable,'numeric']=np.nan
        v=g.groupby('video_id').numeric.agg(['count','mean','std','min','max']);multi=v[v['count']>=2]
        out.append(dict(prompt_hash=prompt,code=code,pitches=len(v),ratings=len(g),uncodable=int(g.uncodable.sum()),pitches_with_repeats=len(multi),mean_pitch_score=v['mean'].mean(),mean_within_pitch_sd=multi['std'].mean(),exact_repeat_fraction=(multi['min']==multi['max']).mean(),mean_range=(multi['max']-multi['min']).mean(),status='repeatability only; human validity not assessed'))
    return pd.DataFrame(out)
