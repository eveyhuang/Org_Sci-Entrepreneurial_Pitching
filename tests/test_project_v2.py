"""Statistical safety checks for the reusable v2 analyses."""
import numpy as np
import pandas as pd
from analysis.project_v2 import fe_test, prediction_analysis, BASE, MODELS


def test_fixed_effect_estimate_is_invariant_to_pitch_constant_outcome_shift():
    rng=np.random.default_rng(25);g=np.repeat(np.arange(24),16);x=rng.binomial(1,.5,len(g));z=rng.normal(size=len(g))
    y=.2+.35*x+.1*z+rng.normal(0,.15,len(g))
    d=pd.DataFrame({'video_id':g,'x':x,'z':z,'y':y})
    a=fe_test(d,'y',['x','z'],'x',99,5)
    d['y']+=np.repeat(rng.normal(0,.03,24),16)
    b=fe_test(d,'y',['x','z'],'x',99,5)
    assert a['status']=='estimated'
    assert .25<a['estimate']<.45
    assert np.isclose(a['estimate'],b['estimate'])
    assert np.isclose(a['p_two_sided'],b['p_two_sided'])


def test_pitch_constant_predictor_does_not_create_pseudoreplication():
    d=pd.DataFrame({'video_id':np.repeat(np.arange(20),20),'x':np.repeat(np.arange(20)%2,20),'y':np.tile([0,1],200)})
    r=fe_test(d,'y',['x'],'x',19)
    assert r['status'].startswith('not estimable')
    assert r['target_varying_pitches']==0


def test_sparse_outcome_skips_inference():
    d=pd.DataFrame({'video_id':np.repeat(np.arange(20),20),'x':np.tile([0,1],200),'y':[1]+[0]*399})
    assert fe_test(d,'y',['x'],'x',19)['status'].startswith('not estimable')


def test_prediction_baselines_hold_exposure_fixed_and_intercept_is_constant_within_pitch():
    rng=np.random.default_rng(3);rows=[]
    for video in range(8):
        for shark in range(5):
            rows.append(dict(video_id=str(video),speaker=str(shark),offer=int(shark<1+video%3),positivity=3+video%2,n_exchanges=shark,no_exchange=int(shark==0),response_missing=0,evidence_missing=0,substance_missing=0,response=rng.random(),evidence=rng.random(),substance=rng.random(),probe_severity=1,promotion_probe=.5))
    metrics,deltas,pred,details=prediction_analysis(pd.DataFrame(rows),19,3)
    assert set(BASE)<=set(MODELS['response_added'])
    assert pred.groupby('video_id')['intercept'].nunique().eq(1).all()
    assert len(metrics)==5 and len(deltas)==6
    for result in details.values():
        assert all(f['train_n']==35 and f['test_n']==5 for f in result['folds'])
