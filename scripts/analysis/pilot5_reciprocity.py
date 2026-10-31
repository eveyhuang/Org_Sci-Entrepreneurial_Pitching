"""Stage B pilot: separate shark interest from founder answer quality.

1. Offer prediction within a pitch, controlling for how many substantive questions the shark asked.
2. Exchange-level reciprocity that needs no outcome: probe severity -> answer quality, answer
   quality -> the same investor's next move, and change across one shark's repeated questions.
All comparisons are within a pitch. Read-only; no API calls.
    .venv/bin/python scripts/analysis/pilot5_reciprocity.py [runs/<exchange-level run>]
"""
import itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from pipeline.stage_b.exchanges import check
from analysis.within_pitch import clogit, informative, stratified_auc

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs/stage_b_exchange_pilot5_v3b"
RESP = {"addresses": 1.0, "partial": 0.5, "deflects": 0.0, "NONE": 0.0}

def value(r):
    if r["uncodable"]: return np.nan
    v = r["value"]
    return RESP.get(v, np.nan) if isinstance(v, str) and v in RESP else (float(v) if isinstance(v, (int, float, bool)) else v)

rows = []
for vdir in sorted(RUN.glob("st_*")):
    b = json.loads((vdir / "frozen_bundle.json").read_text()); m, t = check(b); ts = {x.turn_idx: x for x in t.turns}
    ratings = {}
    for f in (vdir / "results").glob("accepted_*.json"):
        a = json.loads(f.read_text())
        if "exchanges" in a: ratings.update({e["exchange_id"]: e["ratings"] for e in a["exchanges"]})
    offers = {s["speaker"]: s["ever_offer"] for s in json.loads((vdir / "boundaries.json").read_text())["outcomes_model_reviewed_not_gold"]["sharks"]}
    xs = sorted(m.exchanges, key=lambda x: min(ts[i].start_s for i in x.probe))
    for order, x in enumerate(xs):
        if x.exchange_id not in ratings: continue           # issue flagged for human review
        r = ratings[x.exchange_id]
        nxt = ts[x.next_move[0]].speaker if x.next_move else None
        rows.append(dict(video=vdir.name, investor=x.investor, order=order, offer=offers.get(x.investor),
                         severity=value(r["PR-SEV"]), response=value(r["RS-RESP"]), evidence=value(r["RS-EVID"]),
                         substance=value(r["PREP-SUBS"]), uptake=None if r["UP-TYPE"]["uncodable"] else r["UP-TYPE"]["value"],
                         answered=bool(x.answer), next_same_investor=(nxt == x.investor) if nxt else None))
E = pd.DataFrame(rows)
print(f"{len(E)} coded exchanges in {E.video.nunique()} pitches ({E.answered.sum()} with a founder answer)\n")

def within_auc(df, group_col, pos, score):
    """AUC of `score` for rows with group==pos vs other rows, pairs formed only within a video."""
    num = pairs = 0.0
    for _, g in df.dropna(subset=[score]).groupby("video"):
        P = g[g[group_col] == pos][score].values; N = g[g[group_col] != pos][score].values
        for p in P:
            for n in N: num += (p > n) + 0.5 * (p == n); pairs += 1
    return (num / pairs if pairs else np.nan), int(pairs)

def perm_p(df, group_col, pos, score, n=5000, seed=0):
    obs, _ = within_auc(df, group_col, pos, score); rng = np.random.default_rng(seed); hits = 0
    for _ in range(n):
        d = df.copy(); d[group_col] = d.groupby("video")[group_col].transform(lambda s: rng.permutation(s.values))
        a, _ = within_auc(d, group_col, pos, score); hits += abs(a - 0.5) >= abs(obs - 0.5) - 1e-12
    return hits / n

# ---- 1. Offer prediction, controlling for questions asked --------------------------------------
print("1. OFFER, WITHIN PITCH, CONTROLLING FOR QUESTIONS ASKED (leave-one-pitch-out conditional logit, 20 pairs)")
D = pd.read_csv(RUN / "decision_features.csv"); D["n_episodes"] = D["n_exchanges"]
def lovo(cols, ridge=1.0):
    scored = []
    for v in sorted(D.video_id.unique()):
        tr, te = D[D.video_id != v].copy(), D[D.video_id == v].copy()
        if not (0 < te.offer.sum() < len(te)): continue
        med, sd = tr[cols].median(), tr[cols].std().replace(0, 1).fillna(1)
        tr[cols] = (tr[cols].fillna(med) - med) / sd; te[cols] = (te[cols].fillna(med) - med) / sd
        b, *_ = clogit(informative(tr, cols), cols, ridge=ridge); te["score"] = te[cols].values @ b; scored.append(te)
    return stratified_auc(scored, "score")[0]
base = lovo(["n_episodes"]); print(f"   questions asked only: {base:.2f}")
for f in ["response", "evidence", "substance", "coherence", "probe_severity", "promotion_probe"]:
    print(f"   + {f:16s} {lovo(['n_episodes', f]):.2f}  (change {lovo(['n_episodes', f]) - base:+.2f})")

# ---- 2. Reciprocity at the exchange level ------------------------------------------------------
A = E[E.answered].copy(); A["tough"] = A.severity >= 2
print(f"\n2a. TOUGHER PROBES -> WEAKER ANSWERS?  answers to severity>=2 vs severity 1, within pitch ({int(A.tough.sum())} tough, {int((~A.tough).sum())} easy)")
for s in ["response", "evidence", "substance"]:
    auc, pairs = within_auc(A, "tough", True, s); print(f"   {s:10s} AUC={auc:.2f} over {pairs} pairs  (<0.5 = tough probes get weaker answers)  perm p={perm_p(A, 'tough', True, s):.2f}")

U = A[A.uptake.isin(["accepts", "follows_up"])].copy()
print(f"\n2b. ANSWER QUALITY -> SAME INVESTOR'S NEXT MOVE?  'accepts' vs 'follows_up' ({(U.uptake=='accepts').sum()} vs {(U.uptake=='follows_up').sum()})")
for s in ["response", "evidence", "substance"]:
    auc, pairs = within_auc(U, "uptake", "accepts", s); print(f"   {s:10s} AUC={auc:.2f} over {pairs} pairs  (>0.5 = accepted answers were stronger)  perm p={perm_p(U, 'uptake', 'accepts', s):.2f}")
print(f"   next move by the SAME investor after {A.next_same_investor.eq(True).sum()} answers, another investor after {A.next_same_investor.eq(False).sum()}")

print("\n2c. REPEATED QUESTIONS FROM ONE SHARK: first vs last answer to that shark")
T = A.sort_values("order").groupby(["video", "investor"])
traj = []
for (v, inv), g in T:
    if len(g) < 2: continue
    first, last = g.iloc[0], g.iloc[-1]
    traj.append(dict(video=v, investor=inv, n=len(g), offer=g.offer.iloc[0], d_response=last.response - first.response, d_substance=last.substance - first.substance,
                     d_evidence=last.evidence - first.evidence, severity_first=first.severity, severity_last=last.severity))
T = pd.DataFrame(traj)
print(f"   {len(T)} shark-pitch pairs with 2+ answered questions (median {T.n.median():.0f} questions)")
for c in ["d_response", "d_evidence", "d_substance"]:
    s = T[c].dropna(); print(f"   {c:12s} improved {int((s>0).sum())}, same {int((s==0).sum())}, declined {int((s<0).sum())}; mean change {s.mean():+.2f}")
print(f"   probe severity first -> last: mean {T.severity_first.mean():.2f} -> {T.severity_last.mean():.2f}")
for label, g in T.groupby(T.offer.map({True: "offer-makers", False: "went out"})):
    print(f"   {label:12s} (n={len(g)}): mean change response {g.d_response.mean():+.2f}, substance {g.d_substance.mean():+.2f}")
