"""Stage A: v1 (Pitch Factor, 10 codes) vs v2 (perceived delivery, 3 codes) on the same 50 Hu & Ma videos.

Read-only over the two runs' tables; makes no API calls. Reproduces every number in
docs/results/stage_A/Prog_sep25_v2.md. Run from the repo root:
    .venv/bin/python scripts/analysis/compare_v1_v2.py
"""
import numpy as np, pandas as pd, itertools, json
V1 = pd.read_parquet("runs/stage_a_v1_20260903_222907/tables/annotations.parquet")
V2 = pd.read_parquet("runs/stage_a_delivery_v2_20260925_012509/tables/annotations.parquet")
C1 = ["visual_positive","vocal_positive","verbal_ability","verbal_warmth","verbal_positive","vocal_arousal","verbal_negative","vocal_valence","visual_negative","vocal_negative"]
C2 = ["pitch_positivity","expressed_enthusiasm","interpersonal_warmth"]
vids = sorted(set(V1.video_id) & set(V2.video_id)); assert len(vids)==50
V2k2 = V2[V2.replicate_k<=1]
def means(df, codes): return df.pivot_table(index="video_id", columns="code_id", values="value", aggfunc="mean").loc[vids, codes]
M1, M2, M2k2 = means(V1,C1), means(V2,C2), means(V2k2,C2)
rk = lambda s: s.rank()
out = {}
def stats(M, codes):
    rows=[]
    for c in codes:
        s = M[c].dropna(); vc = s.round(2).value_counts()
        rows.append(dict(code=c, mean=s.mean(), sd=s.std(), distinct=int(vc.size), modal=vc.iloc[0]/len(s)))
    return pd.DataFrame(rows)
print("=== 4.1 STATS  v1 k=2 ==="); print(stats(M1,C1).round(2).to_string(index=False))
print("=== 4.1 STATS  v2 k=2 (replicates 0-1) ==="); print(stats(M2k2,C2).round(2).to_string(index=False))
print("=== 4.1 STATS  v2 k=5 ==="); print(stats(M2,C2).round(2).to_string(index=False))
print("\n=== 4.2 TEST-RETEST exact agreement k0 vs k1 ===")
def agree(df, codes):
    r={}
    for c in codes:
        w = df[df.code_id==c].pivot_table(index="video_id", columns="replicate_k", values="value").loc[vids]
        w = w.dropna(subset=[0,1]); r[c]=((w[0]==w[1]).mean(), len(w))
    return r
for c,(a,n) in agree(V1,C1).items(): print(f"  v1 {c:22s} {a:.2f} (n={n})")
for c,(a,n) in agree(V2,C2).items(): print(f"  v2 {c:22s} {a:.2f} (n={n})")
print("\n=== ICC: share of rating variance that is BETWEEN videos (one-way random effects) ===")
def icc(df, c, reps):
    w = df[(df.code_id==c)&(df.replicate_k.isin(reps))].pivot_table(index="video_id", columns="replicate_k", values="value").loc[vids].dropna()
    k = w.shape[1]; n = len(w); gm = w.values.mean()
    msb = k*((w.mean(axis=1)-gm)**2).sum()/(n-1); msw = ((w.values - w.mean(axis=1).values[:,None])**2).sum()/(n*(k-1))
    return (msb-msw)/(msb+(k-1)*msw), (msb-msw)/msb, n
for c in C1: a,b,n = icc(V1,c,[0,1]); print(f"  v1 {c:22s} ICC(1)={a:.2f}  ICC(1,k=2)={b:.2f}")
for c in C2:
    a,b,n = icc(V2,c,[0,1]); a5,b5,n5 = icc(V2,c,[0,1,2,3,4]); print(f"  v2 {c:22s} k=2: ICC(1)={a:.2f} ICC(1,2)={b:.2f} | k=5: ICC(1)={a5:.2f} ICC(1,5)={b5:.2f} (n={n5})")
print("\n=== 4.3 SCALE USE (all ratings) ===")
for name, df in (("v1",V1),("v2 k=5",V2),("v2 k=2",V2k2)):
    v = df.value.dropna().astype(int).value_counts().reindex([1,2,3,4,5], fill_value=0); n=v.sum()
    ent = -(v[v>0]/n*np.log2(v[v>0]/n)).sum()
    print(f"  {name:7s} n={n} ", {k:int(x) for k,x in v.items()}, " %:", {k:round(100*x/n) for k,x in v.items()}, f" entropy={ent:.2f} bits (max 2.32); uncodable={int(df.uncodable.sum())}")
print("\n=== 4.4 PAIRWISE SPEARMAN ===")
def pairs(M, codes):
    R = M[codes].apply(rk).corr(); ps = [(a,b,R.loc[a,b]) for a,b in itertools.combinations(codes,2)]
    return R, sorted(ps, key=lambda x: -abs(x[2]))
R1,p1 = pairs(M1,C1); R2,p2 = pairs(M2,C2)
print("  v1 top 5:", [(a,b,round(r,2)) for a,b,r in p1[:5]]); print("  v1 pairs |rho|>0.7:", sum(abs(r)>0.7 for *_,r in p1), "of", len(p1), " >0.9:", sum(abs(r)>0.9 for *_,r in p1))
print("  v2 pairs:", [(a,b,round(r,2)) for a,b,r in p2]); print("  v2 identical k2-averaged values: pos-warm", int((M2k2.pitch_positivity==M2k2.interpersonal_warmth).sum()), "pos-enth", int((M2k2.pitch_positivity==M2k2.expressed_enthusiasm).sum()))
print("  v1 identical k2 visual/vocal positive:", int((M1.visual_positive==M1.vocal_positive).sum()))
print("\n=== 4.5 PCA (standardized video means) ===")
def pca(M):
    X = M.dropna(); Z = (X-X.mean())/X.std(ddof=1); C = np.corrcoef(Z.values, rowvar=False)
    ev, evec = np.linalg.eigh(C); o = ev.argsort()[::-1]; ev, evec = ev[o], evec[:,o]
    share = ev/ev.sum(); cum = np.cumsum(share); p=len(ev)
    load = evec[:,0]*np.sqrt(ev[0]); load = load if load.sum()>0 else -load
    auc = cum.mean()                             # area under cumulative-variance curve, normalised to [1/p, 1]
    auc_norm = (auc - 1/p)/(1 - 1/p) if p>1 else 1  # 0 = variance spread evenly, 1 = all on PC1
    pr = ev.sum()**2/(ev**2).sum()               # effective number of dimensions (participation ratio)
    n80 = int(np.searchsorted(cum, 0.80)+1)
    pc1 = Z.values @ evec[:,0]; pc1 = pc1 if np.corrcoef(pc1, Z.values.mean(axis=1))[0,1]>0 else -pc1
    return dict(share=share, cum=cum, load=dict(zip(X.columns, load)), auc=auc, auc_norm=auc_norm, pr=pr, n80=n80, p=p, pc1=pd.Series(pc1, index=X.index))
P1, P2 = pca(M1), pca(M2); P2k2 = pca(M2k2)
for name,P in (("v1 (10 codes, k=2)",P1),("v2 (3 codes, k=5)",P2),("v2 (3 codes, k=2)",P2k2)):
    print(f"  {name}: variance share {np.round(100*P['share'],1)} cum {np.round(100*P['cum'],1)}")
    print(f"     AUC of cumulative curve={P['auc']:.3f}  normalised={P['auc_norm']:.3f}  effective dims={P['pr']:.2f}  PCs for 80%={P['n80']}/{P['p']}")
    print("     PC1 loadings:", {k:round(v,2) for k,v in P['load'].items()})
# v1 restricted to 3 counterparts, for a fair 3-vs-3 comparison
P1c = pca(M1[["vocal_positive","vocal_arousal","verbal_warmth"]]); print(f"  v1 3-code counterpart set (vocal_positive, vocal_arousal, verbal_warmth): share {np.round(100*P1c['share'],1)} auc_norm={P1c['auc_norm']:.3f} eff dims={P1c['pr']:.2f}")
print("\n=== CROSS-INSTRUMENT: how much of each instrument's between-video variance the other explains ===")
def cv_r2(y, X):
    X = np.column_stack([np.ones(len(X)), X]); pred = np.empty(len(y))
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i]=False; b,*_ = np.linalg.lstsq(X[m], y[m], rcond=None); pred[i] = X[i]@b
    return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
def r2(y, X):
    X = np.column_stack([np.ones(len(X)), X]); b,*_ = np.linalg.lstsq(X,y,rcond=None); return 1-((y-X@b)**2).sum()/((y-y.mean())**2).sum()
Mb = pd.concat([M1,M2],axis=1).dropna()
for c in C2:
    y = Mb[c].values; X = Mb[C1].values; print(f"  v2 {c:22s} <- all 10 v1 codes: R2={r2(y,X):.2f}  LOO-CV R2={cv_r2(y,X):.2f}")
y = P1['pc1'].loc[Mb.index].values
print(f"  v1 PC1 (Pitch Factor) <- 3 v2 codes: R2={r2(y,Mb[C2].values):.2f} LOO-CV={cv_r2(y,Mb[C2].values):.2f};  <- pitch_positivity alone: R2={r2(y,Mb[['pitch_positivity']].values):.2f}")
for c in C1:
    y = Mb[c].values; print(f"  v1 {c:18s} <- 3 v2 codes: LOO-CV R2={cv_r2(y,Mb[C2].values):5.2f}")
print(f"\n  rho(v2 PC1, v1 PC1) = {rk(P2['pc1']).corr(rk(P1['pc1'].loc[P2['pc1'].index])):.2f}")
print("\n=== v2 vs v1 Spearman, all pairs ===")
print(pd.DataFrame({n:{o: rk(M2[n]).corr(rk(M1[o])) for o in C1} for n in C2}).round(2).to_string())

print("\n=== PARTIAL: enthusiasm vs warmth after regressing positivity out of both (v2, k=5 means) ===")
Mp = M2.dropna()
def resid(y, x): return y - np.polyval(np.polyfit(x, y, 1), x)
print("  rho =", round(rk(pd.Series(resid(Mp.expressed_enthusiasm.values, Mp.pitch_positivity.values))).corr(rk(pd.Series(resid(Mp.interpersonal_warmth.values, Mp.pitch_positivity.values)))), 2))
