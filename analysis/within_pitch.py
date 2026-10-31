"""Shared within-pitch estimators for the Stage B pilot analyses."""
import itertools
import numpy as np

def informative(df, cols):
    """Videos with >=1 offer and >=1 non-offer among sharks observed on `cols`."""
    out = []
    for v, g in df.groupby("video_id"):
        g = g.dropna(subset=cols)
        if 0 < g.offer.sum() < len(g): out.append(g)
    return out

def stratified_auc(groups, col):
    num = pairs = 0.0
    for g in groups:
        P, N = g[g.offer == 1][col].values, g[g.offer == 0][col].values
        for p in P:
            for n in N: num += (p > n) + 0.5 * (p == n); pairs += 1
    return (num / pairs if pairs else np.nan), int(pairs)

def clogit(groups, cols, ridge=0.0, iters=50):
    """Exact conditional logit (stratified by video), optional ridge. Returns beta, loglik, loglik0."""
    strata = []
    for g in groups:
        X = g[cols].values.astype(float); m = int(g.offer.sum())
        subsets = [list(S) for S in itertools.combinations(range(len(X)), m)]
        strata.append((X, X[g.offer.values == 1].sum(0), np.array([X[S].sum(0) for S in subsets])))
    def ll(b): return sum(xc @ b - np.log(np.exp(Z @ b).sum()) for _, xc, Z in strata) - ridge * b @ b / 2
    b = np.zeros(len(cols))
    for _ in range(iters):
        grad = -ridge * b; H = -ridge * np.eye(len(b))
        for _, xc, Z in strata:
            w = np.exp(Z @ b - (Z @ b).max()); w /= w.sum(); mu = w @ Z
            grad += xc - mu; H -= (Z - mu).T @ (w[:, None] * (Z - mu))
        step = np.linalg.solve(H, grad); b = b - step
        if np.abs(step).max() < 1e-8: break
    return b, ll(b), ll(np.zeros(len(cols)))

