"""Stage B pilot (5 videos): compare sharks WITHIN a pitch.

Outcome: shark ever made/joined an on-air offer. Only offer/no-offer pairs inside the same
video are compared, so video-level differences (and each video's offer count) cancel out.
Read-only over <run>/decision_features.csv; no API calls.
    .venv/bin/python scripts/analysis/pilot5_within_video.py [runs/<pilot run>]
Exposure is opened episodes (structured pilot) or substantive exchanges (exchange pilot).
"""
import itertools, math
import numpy as np, pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.within_pitch import informative, stratified_auc, clogit
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/stage_b_structured_pilot5"
D = pd.read_csv(f"{RUN}/decision_features.csv")
if "n_exchanges" in D and "n_episodes" not in D: D["n_episodes"] = D["n_exchanges"]   # exposure: exchanges (v2) / opened episodes (v1)
FOUNDER = ["response", "evidence", "substance", "coherence", "repair"]
OTHER = ["positivity", "probe_severity", "promotion_probe", "n_episodes"]
VIDS = sorted(D.video_id.unique())

def exact_perm_p(groups, col):
    """Exact within-video permutation test: every re-assignment of each video's offers."""
    obs, _ = stratified_auc(groups, col)
    per_video = []
    for g in groups:
        x, m = g[col].values, int(g.offer.sum()); opts = []
        for S in itertools.combinations(range(len(x)), m):
            y = np.zeros(len(x)); y[list(S)] = 1; opts.append((x, y))
        per_video.append(opts)
    stats = []
    for combo in itertools.product(*per_video):
        num = pairs = 0.0
        for x, y in combo:
            for p in x[y == 1]:
                for n in x[y == 0]: num += (p > n) + 0.5 * (p == n); pairs += 1
        stats.append(num / pairs)
    stats = np.array(stats)
    return float(np.mean(np.abs(stats - 0.5) >= abs(obs - 0.5) - 1e-12)), len(stats)

def lr_p(ll1, ll0, df=1):
    x = max(0.0, 2 * (ll1 - ll0)); return math.erfc(math.sqrt(x / 2)) if df == 1 else float("nan")

def lovo_auc(df, cols, ridge=1.0):
    """Leave one video out: fit clogit on the other videos (training-median imputation,
    training-SD scaling), score the held-out video's sharks, then pool within-video pairs."""
    scored = []
    for v in VIDS:
        tr, te = df[df.video_id != v].copy(), df[df.video_id == v].copy()
        if not (0 < te.offer.sum() < len(te)): continue
        med, sd = tr[cols].median(), tr[cols].std().replace(0, 1).fillna(1)
        tr[cols] = ((tr[cols].fillna(med) - med) / sd); te[cols] = ((te[cols].fillna(med) - med) / sd)
        b, *_ = clogit(informative(tr, cols), cols, ridge=ridge)
        te["score"] = te[cols].values @ b; scored.append(te)
    return stratified_auc(scored, "score")

print(f"{len(D)} shark rows, {int(D.offer.sum())} offers, {len(VIDS)} videos; offers per video:",
      D.groupby('video_id').offer.sum().to_dict())
print("positivity is constant within every video (all 5 prepared pitches = 5): no within-video comparison possible\n")

rows = []
for f in FOUNDER + OTHER:
    if f == "positivity": continue
    G = informative(D, [f]); auc, pairs = stratified_auc(G, f)
    p, nperm = exact_perm_p(G, f)
    varies = [g for g in G if g[f].nunique() > 1]
    distinct_pairs = sum(1 for g in G for a in g[g.offer == 1][f] for b in g[g.offer == 0][f] if a != b)
    if not varies:
        or_txt, lrp = "no within-pitch variation", "—"
    else:
        Gs = [g.assign(**{f: (g[f] - D[f].mean()) / (D[f].std() or 1)}) for g in G]
        b, l1, l0 = clogit(Gs, [f], ridge=0.25)          # light ridge keeps separated features finite
        sep = distinct_pairs <= 2
        or_txt = f"{np.exp(b[0]):.2f}" + (" (separation: <=2 untied pairs)" if sep else "")
        lrp = "—" if sep else round(lr_p(l1, l0), 2)
    rows.append(dict(feature=f, videos=len(G), sharks=sum(len(g) for g in G), pairs=pairs,
                     untied_pairs=distinct_pairs, within_AUC=round(auc, 2), exact_perm_p=round(p, 2),
                     clogit_OR_per_SD=or_txt, LR_p_asymptotic=lrp))
print("UNIVARIATE, complete cases, sharks compared only within their own pitch")
print(pd.DataFrame(rows).to_string(index=False))

print("\nMULTIVARIATE conditional logit, leave-one-video-out (ridge=1, training-median imputation):")
for name, cols in [("founder interaction (prespecified 5 features + n_episodes)", FOUNDER + ["n_episodes"]),
                   ("founder interaction without exposure", FOUNDER),
                   ("+ probe controls", FOUNDER + ["n_episodes", "probe_severity", "promotion_probe"]),
                   ("exposure only (n_episodes)", ["n_episodes"])]:
    auc, pairs = lovo_auc(D, cols); print(f"  {name:62s} held-out within-video AUC={auc:.2f} over {pairs} pairs")

# how often a random within-video ranking reaches a given AUC, for the 20-pair design
G = informative(D, ["n_episodes"]); rng = np.random.default_rng(0); null = []
for _ in range(20000):
    num = pairs = 0.0
    for g in G:
        s = rng.random(len(g)); y = g.offer.values
        for p in s[y == 1]:
            for n in s[y == 0]: num += p > n; pairs += 1
    null.append(num / pairs)
print(f"\nrandom within-video scores ({int(sum(len(g[g.offer==1])*len(g[g.offer==0]) for g in G))} pairs): 95% interval {np.percentile(null,2.5):.2f}–{np.percentile(null,97.5):.2f}")
