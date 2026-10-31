"""Exploratory held-out utility of concept values with availability held fixed."""
import numpy as np
import pandas as pd
from analysis.auc import grouped_auc
from analysis.concept_bottleneck import CONCEPTS, MASKS
from analysis.project_v2 import score


def rank_concepts(d,reps=1999,seed=20260928):
    d=d.reset_index(drop=True)
    specs={'availability_baseline':MASKS,'full':list(CONCEPTS)+MASKS,'offer_rate':[]}
    for c in CONCEPTS:
        specs['only_'+c]=[c]+MASKS
        specs['without_'+c]=[f for f in CONCEPTS if f!=c]+MASKS
    predictions=d[['video_id','speaker','offer']].copy()
    metrics=[]
    for name,features in specs.items():
        r=grouped_auc(d.to_dict('records'),features)
        p=pd.DataFrame(r['predictions']).rename(columns={'prediction':name})
        predictions=predictions.merge(p[['video_id','speaker',name]],on=['video_id','speaker'],validate='one_to_one')
        metrics.append(dict(model=name,**score(predictions.offer,predictions[name])))
    y=predictions.offer.to_numpy()
    def loss(col):
        p=np.clip(predictions[col].to_numpy(),1e-10,1-1e-10)
        return -(y*np.log(p)+(1-y)*np.log(1-p))
    groups=[g.index.to_numpy() for _,g in predictions.groupby('video_id')]
    rng=np.random.default_rng(seed)
    draws=[np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))]) for _ in range(reps)]
    rows=[]
    for c,(label,_,_) in CONCEPTS.items():
        gain=loss('without_'+c)-loss('full')
        solo=loss('availability_baseline')-loss('only_'+c)
        lo,hi=np.quantile([gain[ix].mean() for ix in draws],[.025,.975])
        slo,shi=np.quantile([solo[ix].mean() for ix in draws],[.025,.975])
        rows.append(dict(concept=c,name=label,incremental_gain=gain.mean(),ci_low=lo,ci_high=hi,
                         standalone_gain=solo.mean(),standalone_ci_low=slo,standalone_ci_high=shi,
                         log_loss_without=loss('without_'+c).mean(),log_loss_alone=loss('only_'+c).mean(),
                         pitches=d.video_id.nunique(),outcomes=len(d)))
    ranking=pd.DataFrame(rows).sort_values('incremental_gain',ascending=False).reset_index(drop=True)
    ranking.insert(0,'rank',np.arange(1,len(ranking)+1))
    return ranking,pd.DataFrame(metrics),predictions

