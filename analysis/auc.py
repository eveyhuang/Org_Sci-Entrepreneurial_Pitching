"""Shared fixed-penalty, video-held-out pilot estimators."""
import numpy as np

def auc(y,score):
    y=np.asarray(y);score=np.asarray(score,dtype=float);positive=score[y==1];negative=score[y==0]
    if not len(positive) or not len(negative):return None
    return float(((positive[:,None]>negative[None,:]).sum()+.5*(positive[:,None]==negative[None,:]).sum())/(len(positive)*len(negative)))

def fit_predict(train_x,train_y,test_x,penalty=1.):
    """Training-only median imputation/scaling; fixed ridge logistic, unpenalized intercept."""
    train_x=np.asarray(train_x,dtype=float);test_x=np.asarray(test_x,dtype=float);y=np.asarray(train_y,dtype=float)
    med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in train_x.T])
    x=np.where(np.isfinite(train_x),train_x,med);z=np.where(np.isfinite(test_x),test_x,med)
    mean=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>1e-10,sd,1)
    x=np.column_stack([np.ones(len(x)),(x-mean)/sd]);z=np.column_stack([np.ones(len(z)),(z-mean)/sd])
    if len(np.unique(y))<2:return np.full(len(z),(y.sum()+1)/(len(y)+2)),{'single_class_training':True}
    beta=np.zeros(x.shape[1]);beta[0]=np.log(y.mean()/(1-y.mean()));regular=np.ones(len(beta))*penalty;regular[0]=0
    def loss(b):
        logits=x@b
        return np.logaddexp(0,logits).sum()-y@logits+.5*np.sum(regular*b*b)
    for iteration in range(100):
        prob=1/(1+np.exp(-np.clip(x@beta,-35,35)));w=prob*(1-prob)
        grad=x.T@(prob-y)+regular*beta
        h=(x.T*w)@x+np.diag(regular)+np.eye(len(beta))*1e-9
        step=np.linalg.solve(h,grad);rate=1.;old=loss(beta)
        while rate>1e-8 and loss(beta-rate*step)>old:rate*=.5
        beta-=rate*step
        if np.max(np.abs(rate*step))<1e-8:break
    return 1/(1+np.exp(-np.clip(z@beta,-35,35))),{'single_class_training':False,'iterations':iteration+1,'coefficients_standardized':beta.tolist(),'training_medians':med.tolist(),'training_means_after_imputation':mean.tolist(),'training_sd':sd.tolist()}

def grouped_auc(rows,features):
    data=[r for r in rows if r['offer'] is not None];videos=sorted({r['video_id'] for r in data});preds=[];folds=[]
    for video in videos:
        tr=[r for r in data if r['video_id']!=video];te=[r for r in data if r['video_id']==video]
        if not tr:continue
        def x(rs):return [[r.get(k,np.nan) if r.get(k) is not None else np.nan for k in features] for r in rs]
        pred,detail=fit_predict(x(tr),[r['offer'] for r in tr],x(te))
        folds.append({'held_out_video':video,'train_n':len(tr),'test_n':len(te),'within_video_auc':auc([r['offer'] for r in te],pred),**detail})
        for r,p in zip(te,pred):preds.append({'video_id':video,'speaker':r['speaker'],'offer':r['offer'],'prediction':float(p)})
    within=[f['within_video_auc'] for f in folds if f['within_video_auc'] is not None]
    return {'features':features,'pooled_oof_auc':auc([r['offer'] for r in preds],[r['prediction'] for r in preds]),'mean_within_video_auc':float(np.mean(within)) if within else None,'folds':folds,'predictions':preds}

def mean(vals):
    x=[v for v in vals if v is not None];return float(np.mean(x)) if x else None

def feature_value(code,v):
    if v['uncodable']:return None
    val=v['value']
    if code=='RS-RESP':return {'addresses':1.,'partial':.5,'deflects':0.}.get(val)
    if code=='PR-FRAME':return float(val=='promotion')
    if isinstance(val,(bool,int,float)):return float(val)
    return None
