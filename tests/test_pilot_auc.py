import numpy as np
from scripts.experiments.analyze_structured_pilot import auc,fit_predict,grouped_auc

def test_auc_ties_and_single_class():
    assert auc([0,1],[0,1])==1
    assert auc([0,1],[1,0])==0
    assert auc([0,1],[.5,.5])==.5
    assert auc([1,1],[0,1]) is None

def test_training_only_imputation():
    a,meta=fit_predict([[0],[2],[np.nan],[1]],[0,1,0,1],[[100],[np.nan]])
    assert meta['training_medians']==[1.]
    assert np.all((0<a)&(a<1))
    assert a[0]>a[1]

def test_single_class_train_prior():
    p,info=fit_predict([[0],[1]],[1,1],[[100]])
    assert info['single_class_training'] and p[0]==.75

def test_grouped_prediction_holds_out_complete_video():
    rows=[{'video_id':str(i),'speaker':str(j),'offer':j,'x':j} for i in range(5) for j in [0,1]]
    result=grouped_auc(rows,['x'])
    assert len(result['predictions'])==10
    assert all(f['train_n']==8 and f['test_n']==2 for f in result['folds'])
    assert result['pooled_oof_auc']==1

def test_no_response_opportunity_is_not_zero_quality():
    from scripts.experiments.analyze_structured_pilot import feature_value
    assert feature_value('RS-RESP',{'uncodable':False,'value':'NONE'}) is None
