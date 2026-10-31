"""Stage A on the Hu & Ma accelerator videos against the accelerator's funding decision.

Follows Hu & Ma (2025, Journal of Finance), Tables II to IV, as far as 50 videos allow:

* per-video measures: the mean of each of the ten ratings over replicates, and the
  Plackett-Luce rank score of each dimension from the forced ranking;
* a Pitch Factor per measurement method: the first principal component of the ten
  standardized measures (principal component factor analysis, as in their §II.D.2),
  signed so that the positive dimensions load positively; loadings and uniqueness reported;
* a logit of I(invested) on the standardized Pitch Factor with average marginal effects
  (their Table III, column 1), one logit per feature (their Table IV);
* when Hu & Ma's own Pitch Factor and features, computed with their code on the same videos,
  are given: how far each Stage A Pitch Factor agrees with theirs (Pearson with a Fisher
  interval, Spearman, Kendall, same-tercile share), the same per dimension, and how the three
  factors weight the ten dimensions (loading vectors and their correlations);
* what their design cannot give here: accelerator fixed effects and accelerator-year
  clustering need many clusters and many acceptances, so beside the plain logit this file
  reports a Firth penalized logit (finite under separation) and an exact permutation test
  of the AUC, which stay honest with a handful of acceptances.

Consumes only exported tables (annotations.parquet, ranking/scores.csv, funding_outcomes.csv,
and optionally data/hu-ma/pitch_factor.csv).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

POSITIVE_CODES = ("visual_positive", "vocal_arousal", "vocal_valence", "vocal_positive",
                  "verbal_positive", "verbal_warmth", "verbal_ability")
NEGATIVE_CODES = ("visual_negative", "vocal_negative", "verbal_negative")
CODES = POSITIVE_CODES[:1] + NEGATIVE_CODES[:1] + POSITIVE_CODES[1:4] + NEGATIVE_CODES[1:2] \
    + POSITIVE_CODES[4:5] + NEGATIVE_CODES[2:] + POSITIVE_CODES[5:]
EXACT_PERMUTATION_LIMIT = 300_000
# Hu & Ma's feature names (their code's output) under this project's code names. The features
# are machine-extracted estimators (face, audio and word models), not ratings: only the names
# are shared.
HU_MA_COLUMNS = {
    "Visual-Positive": "visual_positive", "Visual-Negative": "visual_negative",
    "Vocal-Positive": "vocal_positive", "Vocal-Negative": "vocal_negative",
    "Vocal-Arousal": "vocal_arousal", "Vocal-Valence": "vocal_valence",
    "Verbal-Positive": "verbal_positive", "Verbal-Negative": "verbal_negative",
    "Verbal-Warmth": "verbal_warmth", "Verbal-Ability": "verbal_ability",
}


# --- inputs -------------------------------------------------------------------------------

def accelerator_family(applied_to: str) -> str:
    """Hu & Ma's accelerator fixed effect, coarsened: YC, Techstars, other, or missing."""
    text = (applied_to or "").strip()
    low = text.lower()
    if not text or low == "na":
        return "missing"
    if low.startswith("yc") or "combinator" in low:
        return "YC"
    if "techstars" in low or "tech stars" in low:
        return "Techstars"
    return "other"


def load_outcomes(path: Path | str) -> pd.DataFrame:
    """funding_outcomes.csv as video_id, accepted (0/1, NaN when unknown), accelerator, notes."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("video_id", "Accepted?", "Applied to"):
        if col not in raw.columns:
            raise ValueError(f"{path} has no {col!r} column")
    accepted = pd.to_numeric(raw["Accepted?"].str.strip().replace({"NA": None, "": None}), errors="coerce")
    out = pd.DataFrame({
        "video_id": raw["video_id"].str.strip(),
        "accepted": accepted.astype(float),
        "accelerator": raw["Applied to"].map(accelerator_family),
        "applied_to": raw["Applied to"].str.strip(),
        "notes": raw.get("Notes", pd.Series([""] * len(raw))).fillna(""),
    })
    bad = out.loc[~out["accepted"].isin([0.0, 1.0]) & out["accepted"].notna(), "video_id"].tolist()
    if bad:
        raise ValueError(f"Accepted? must be 0, 1 or NA; offending videos: {bad}")
    return out.drop_duplicates("video_id").reset_index(drop=True)


def rating_means(annotations: pd.DataFrame, codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    """video_id × code: the mean rating over replicates, uncodable replicates left out."""
    a = annotations[annotations["code_id"].isin(codes)].copy()
    a["numeric"] = pd.to_numeric(a["value"], errors="coerce")
    a.loc[a["uncodable"].astype(bool), "numeric"] = np.nan
    wide = a.pivot_table(index="video_id", columns="code_id", values="numeric", aggfunc="mean")
    return wide.reindex(columns=[c for c in codes if c in wide.columns])


def rating_repeatability(annotations: pd.DataFrame, codes: tuple[str, ...] = CODES) -> pd.DataFrame:
    """Per code: how the replicates of one video agree with each other, and ICC(1) from a one-way
    ANOVA of ratings on videos. Model self-consistency, not validity."""
    a = annotations[annotations["code_id"].isin(codes)].copy()
    a["numeric"] = pd.to_numeric(a["value"], errors="coerce")
    a.loc[a["uncodable"].astype(bool), "numeric"] = np.nan
    rows = []
    for code in codes:
        g = a[a["code_id"] == code].dropna(subset=["numeric"])
        if g.empty:
            continue
        per_video = g.groupby("video_id")["numeric"]
        n_videos = per_video.ngroups
        k = per_video.size()
        means = per_video.mean()
        grand = g["numeric"].mean()
        ss_between = float((k * (means - grand) ** 2).sum())
        ss_within = float(((g["numeric"] - g["video_id"].map(means)) ** 2).sum())
        df_between = n_videos - 1
        df_within = len(g) - n_videos
        icc = float("nan")
        if df_between > 0 and df_within > 0:
            ms_between = ss_between / df_between
            ms_within = ss_within / df_within
            k0 = (len(g) - float((k ** 2).sum()) / len(g)) / df_between
            denom = ms_between + (k0 - 1) * ms_within
            icc = (ms_between - ms_within) / denom if denom > 0 else float("nan")
        multi = k[k >= 2].index
        spread = per_video.agg(["min", "max", "std"]).loc[multi] if len(multi) else None
        rows.append({
            "code": code,
            "videos": int(n_videos),
            "ratings": int(len(g)),
            "uncodable": int(a.loc[a["code_id"] == code, "uncodable"].astype(bool).sum()),
            "mean": float(g["numeric"].mean()),
            "sd_between_videos": float(means.std(ddof=1)) if n_videos > 1 else float("nan"),
            "mean_within_video_sd": float(spread["std"].mean()) if spread is not None else float("nan"),
            "exact_repeat_fraction": float((spread["min"] == spread["max"]).mean()) if spread is not None else float("nan"),
            "icc1": float(icc),
            "distinct_values": int(g["numeric"].nunique()),
        })
    return pd.DataFrame(rows)


def rank_scores(scores_csv: Path | str, codes: tuple[str, ...] = CODES, column: str = "pl_score") -> pd.DataFrame:
    """video_id × code: one relative score per video per dimension from ranking/scores.csv."""
    s = pd.read_csv(scores_csv)
    wide = s.pivot(index="video_id", columns="code", values=column)
    return wide.reindex(columns=[c for c in codes if c in wide.columns]).astype(float)


# --- Pitch Factor -------------------------------------------------------------------------

@dataclass
class PitchFactor:
    scores: pd.Series                 # standardized, mean 0 sd 1, indexed by video_id
    loadings: pd.DataFrame            # code, loading, uniqueness
    explained_variance_share: float
    dropped: dict[str, str] = field(default_factory=dict)


def pitch_factor(wide: pd.DataFrame) -> PitchFactor:
    """First principal component of the standardized measures (Hu & Ma §II.D.2), signed so that
    the positive dimensions load positively on balance. Columns without variance are dropped and
    named; videos with any missing measure are left out of the scores."""
    data = wide.dropna(axis=0, how="any").astype(float)
    dropped: dict[str, str] = {}
    keep = []
    for code in data.columns:
        sd = data[code].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            dropped[code] = "no variance across videos"
        else:
            keep.append(code)
    if len(keep) < 2:
        raise ValueError(f"the Pitch Factor needs at least two measures with variance, got {keep}")
    z = (data[keep] - data[keep].mean()) / data[keep].std(ddof=1)
    corr = np.corrcoef(z.to_numpy(), rowvar=False)
    values, vectors = np.linalg.eigh(corr)
    first = vectors[:, -1]
    share = float(values[-1] / values.sum())
    sign = sum(first[i] for i, c in enumerate(keep) if c in POSITIVE_CODES) \
        - sum(first[i] for i, c in enumerate(keep) if c in NEGATIVE_CODES)
    if sign < 0:
        first = -first
    # loadings as correlations between each measure and the component
    loadings = first * math.sqrt(values[-1])
    raw = z.to_numpy() @ first
    scores = pd.Series((raw - raw.mean()) / raw.std(ddof=1), index=data.index, name="pitch_factor")
    table = pd.DataFrame({"code": keep, "loading": loadings, "uniqueness": 1 - loadings ** 2})
    return PitchFactor(scores=scores, loadings=table, explained_variance_share=share, dropped=dropped)


# --- Hu & Ma's own Pitch Factor (convergent validity) -------------------------------------

def load_hu_ma_pitch_factor(path: Path | str) -> pd.DataFrame:
    """video_id × (pitch_factor, the ten features under this project's code names) from a file
    written by Hu & Ma's code: one row per video, their Pitch Factor and their ten features."""
    raw = pd.read_csv(path)
    for col in ("video_id", "pitch_factor"):
        if col not in raw.columns:
            raise ValueError(f"{path} has no {col!r} column")
    missing = [c for c in HU_MA_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path} lacks Hu & Ma feature columns: {missing}")
    out = raw.rename(columns=HU_MA_COLUMNS).copy()
    out["video_id"] = out["video_id"].astype(str).str.strip()
    out = out.drop_duplicates("video_id").set_index("video_id")
    return out[["pitch_factor", *[c for c in CODES if c in out.columns]]].astype(float)


def _fisher_interval(r: float, n: int, level: float = 0.95) -> tuple[float, float]:
    if n < 4 or not np.isfinite(r) or abs(r) >= 1:
        return float("nan"), float("nan")
    z, se = math.atanh(r), 1 / math.sqrt(n - 3)
    q = stats.norm.ppf(0.5 + level / 2)
    return float(math.tanh(z - q * se)), float(math.tanh(z + q * se))


def _tercile(x: pd.Series) -> pd.Series:
    return pd.qcut(x.rank(method="first"), 3, labels=False)


def _agreement_row(name: str, a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    a, b, n = a[ok], b[ok], int(ok.sum())
    row = {"method": name, "videos": n, "pearson": float("nan"), "pearson_low": float("nan"),
           "pearson_high": float("nan"), "pearson_p": float("nan"), "spearman": float("nan"), "spearman_p": float("nan"),
           "kendall_tau": float("nan"), "kendall_p": float("nan"), "same_tercile_share": float("nan")}
    if n < 4 or a.nunique() < 2 or b.nunique() < 2:
        return row
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    kt, kp = stats.kendalltau(a, b)
    lo, hi = _fisher_interval(float(pr), n)
    row.update({"pearson": float(pr), "pearson_low": lo, "pearson_high": hi, "pearson_p": float(pp),
                "spearman": float(sr), "spearman_p": float(sp), "kendall_tau": float(kt), "kendall_p": float(kp),
                "same_tercile_share": float((_tercile(a) == _tercile(b)).mean())})
    return row


def external_agreement(pf_scores: pd.DataFrame, hu_ma: pd.DataFrame) -> pd.DataFrame:
    """Per Stage A method, how its Pitch Factor agrees with Hu & Ma's over the videos both have:
    Pearson with a Fisher 95% interval, Spearman, Kendall's tau, and the share of videos the two
    put in the same tercile (chance: one third)."""
    common = pf_scores.index.intersection(hu_ma.index)
    rows = [_agreement_row(name, pf_scores.loc[common, f"pitch_factor_{name}"], hu_ma.loc[common, "pitch_factor"])
            for name in ("ratings", "ranks") if f"pitch_factor_{name}" in pf_scores.columns]
    return pd.DataFrame(rows)


def external_feature_agreement(ratings: pd.DataFrame, ranks: pd.DataFrame, hu_ma: pd.DataFrame) -> pd.DataFrame:
    """Per dimension, Spearman between each Stage A measure (mean rating, rank score) and Hu & Ma's
    feature of the same name, over the videos both have. Where the two instruments agree and
    where they see different things."""
    rows = []
    for code in CODES:
        if code not in hu_ma.columns:
            continue
        row = {"code": code, "videos": 0}
        for name, wide in (("rating", ratings), ("rank_score", ranks)):
            rho = p = float("nan")
            if code in wide.columns:
                common = wide.index.intersection(hu_ma.index)
                a, b = wide.loc[common, code], hu_ma.loc[common, code]
                ok = a.notna() & b.notna()
                row["videos"] = max(row["videos"], int(ok.sum()))
                if ok.sum() >= 3 and a[ok].nunique() > 1 and b[ok].nunique() > 1:
                    rho, p = stats.spearmanr(a[ok], b[ok])
            row[f"{name}_spearman"], row[f"{name}_p"] = float(rho), float(p)
        row["hu_ma_feature_sd"] = float(hu_ma[code].std(ddof=1))
        row["hu_ma_distinct_values"] = int(hu_ma[code].nunique())
        rows.append(row)
    return pd.DataFrame(rows)


def loading_agreement(factors: dict[str, PitchFactor], hu_ma_factor: PitchFactor | None) -> tuple[pd.DataFrame, dict]:
    """Per code, the loading of each Pitch Factor (Stage A ratings, Stage A ranks, Hu & Ma's
    features re-factored on the same videos), and the Pearson correlation between every pair of
    loading vectors over the codes: do the instruments weight the ten dimensions the same way?"""
    table = pd.DataFrame(index=pd.Index(list(CODES), name="code"))
    for name, pf in factors.items():
        table[f"loading_{name}"] = pf.loadings.set_index("code")["loading"]
    if hu_ma_factor is not None:
        table["loading_hu_ma"] = hu_ma_factor.loadings.set_index("code")["loading"]
    pairs = {}
    for a, b in combinations(list(table.columns), 2):
        both = table[[a, b]].dropna()
        if len(both) >= 3:
            key = f"{a.removeprefix('loading_')}_vs_{b.removeprefix('loading_')}"
            pairs[key] = float(np.corrcoef(both[a], both[b])[0, 1])
    return table.reset_index(), pairs


# --- models -------------------------------------------------------------------------------

def standardize(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=1)
    return (x - x.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0


def logit_ame(y: np.ndarray, x: np.ndarray, controls: np.ndarray | None = None) -> dict:
    """Maximum-likelihood logit; the average marginal effect of the first regressor with a
    heteroskedasticity-robust (HC1) standard error, McFadden pseudo R². Hu & Ma's Table III
    column 1, minus the accelerator fixed effects and clustering a sample this size cannot carry."""
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    X = np.column_stack([x] + ([controls] if controls is not None and controls.size else []))
    X = sm.add_constant(X, has_constant="add")
    out = {"n": int(len(y)), "events": int(y.sum()), "method": "logit_ml"}
    try:
        # separation and non-convergence are reported in `note`, not as warnings
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(y, X).fit(disp=0, maxiter=200, cov_type="HC1")
            me = res.get_margeff(at="overall", method="dydx")
        out.update({
            "coef": float(res.params[1]), "coef_se": float(res.bse[1]),
            "ame": float(me.margeff[0]), "ame_se": float(me.margeff_se[0]), "p": float(me.pvalues[0]),
            "pseudo_r2": float(res.prsquared), "converged": bool(res.mle_retvals.get("converged", True)),
            "note": "" if res.mle_retvals.get("converged", True) else "did not converge",
        })
        if not np.isfinite(out["ame_se"]) or abs(out["coef"]) > 20 or not out["converged"]:
            # the maximum does not exist or was not reached: no number from this fit is a result
            out["note"] = (out["note"] + "; " if out["note"] else "") + "quasi-complete separation; ML estimates withheld"
            out.update({"coef": float("nan"), "coef_se": float("nan"), "ame": float("nan"), "ame_se": float("nan"),
                        "p": float("nan"), "pseudo_r2": float("nan")})
    except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
        out.update({"coef": float("nan"), "coef_se": float("nan"), "ame": float("nan"), "ame_se": float("nan"),
                    "p": float("nan"), "pseudo_r2": float("nan"), "converged": False,
                    "note": f"not estimable: {type(exc).__name__}"})
    return out


def firth_logit(y: np.ndarray, X: np.ndarray, max_iter: int = 200, tol: float = 1e-9) -> dict:
    """Logit with Firth's (1993) Jeffreys-prior penalty, by Newton steps on the modified score
    U(β) + ½·tr(I⁻¹ ∂I/∂β) = Xᵀ(y − p + h(½ − p)); finite under separation. Returns the
    coefficients (constant first), standard errors from the penalized information, Wald p-values,
    the average marginal effect of each regressor at the fitted probabilities, and the penalized
    log-likelihood."""
    Xc = np.column_stack([np.ones(len(y)), X]).astype(float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(Xc.shape[1])

    def penalised_loglik(b):
        eta = Xc @ b
        p = expit(eta)
        w = p * (1 - p)
        info = Xc.T @ (Xc * w[:, None])
        sign, logdet = np.linalg.slogdet(info)
        ll = float(np.sum(y * eta - np.logaddexp(0, eta)))
        return ll + 0.5 * logdet if sign > 0 else -np.inf

    current = penalised_loglik(beta)
    converged = False
    for _ in range(max_iter):
        p = expit(Xc @ beta)
        w = p * (1 - p)
        info = Xc.T @ (Xc * w[:, None])
        info_inv = np.linalg.pinv(info)
        hat = np.einsum("ij,jk,ik->i", Xc, info_inv, Xc) * w
        score = Xc.T @ (y - p + hat * (0.5 - p))
        step = info_inv @ score
        candidate = beta + step
        new = penalised_loglik(candidate)
        halvings = 0
        while new < current and halvings < 30:
            step /= 2
            candidate = beta + step
            new = penalised_loglik(candidate)
            halvings += 1
        beta = candidate
        if abs(new - current) < tol and np.max(np.abs(step)) < 1e-6:
            converged = True
            current = new
            break
        current = new
    p = expit(Xc @ beta)
    w = p * (1 - p)
    info = Xc.T @ (Xc * w[:, None])
    cov = np.linalg.pinv(info)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return {
        "coef": beta, "se": se, "p": 2 * stats.norm.sf(np.abs(z)),
        "ame": beta * float(w.mean()), "penalised_loglik": float(current), "converged": converged,
        "method": "logit_firth",
    }


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC: the probability a funded video scores above an unfunded one, ties half."""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auc_permutation(y: np.ndarray, score: np.ndarray, reps: int = 20_000, seed: int = 0) -> dict:
    """The AUC and the two-sided probability of an AUC at least as far from ½ under random
    relabelling. Enumerated exactly when the number of ways to choose the funded videos is small
    enough (it is, with a handful of acceptances); Monte Carlo otherwise."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    observed = auc(y, score)
    n, k = len(y), int(y.sum())
    if not np.isfinite(observed) or k == 0 or k == n:
        return {"auc": observed, "p_two_sided": float("nan"), "exact": True, "permutations": 0}
    distance = abs(observed - 0.5) - 1e-12
    total = math.comb(n, k)
    if total <= EXACT_PERMUTATION_LIMIT:
        extreme = 0
        labels = np.zeros(n, dtype=int)
        for chosen in combinations(range(n), k):
            labels[:] = 0
            labels[list(chosen)] = 1
            if abs(auc(labels, score) - 0.5) >= distance:
                extreme += 1
        return {"auc": observed, "p_two_sided": extreme / total, "exact": True, "permutations": total}
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(reps):
        labels = rng.permutation(y)
        if abs(auc(labels, score) - 0.5) >= distance:
            extreme += 1
    return {"auc": observed, "p_two_sided": (extreme + 1) / (reps + 1), "exact": False, "permutations": reps}


def accelerator_dummies(accelerator: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Indicator columns for every accelerator family but the most common one (the reference)."""
    counts = accelerator.value_counts()
    if len(counts) < 2:
        return np.empty((len(accelerator), 0)), []
    reference = counts.index[0]
    levels = [lvl for lvl in counts.index if lvl != reference]
    cols = np.column_stack([(accelerator == lvl).astype(float).to_numpy() for lvl in levels])
    return cols, [f"accelerator={lvl} (ref {reference})" for lvl in levels]


def predictor_row(name: str, y: np.ndarray, x: pd.Series, accelerator: pd.Series | None = None) -> dict:
    """One line of a Table III/IV analogue for a standardized predictor: ML logit AME, Firth
    logit with and without accelerator indicators, AUC with its exact permutation p."""
    z = standardize(x).to_numpy(dtype=float)
    ml = logit_ame(y, z)
    firth = firth_logit(y, z[:, None])
    row = {
        "predictor": name, "n": int(len(y)), "events": int(y.sum()),
        "ml_ame": ml["ame"], "ml_ame_se": ml["ame_se"], "ml_p": ml["p"], "ml_pseudo_r2": ml["pseudo_r2"],
        "ml_note": ml["note"],
        "firth_coef": float(firth["coef"][1]), "firth_se": float(firth["se"][1]), "firth_p": float(firth["p"][1]),
        "firth_ame": float(firth["ame"][1]),
    }
    if accelerator is not None:
        dummies, names = accelerator_dummies(accelerator)
        if names:
            with_fe = firth_logit(y, np.column_stack([z, dummies]))
            row.update({"firth_fe_coef": float(with_fe["coef"][1]), "firth_fe_se": float(with_fe["se"][1]),
                        "firth_fe_p": float(with_fe["p"][1]), "firth_fe_ame": float(with_fe["ame"][1]),
                        "firth_fe_controls": "; ".join(names)})
    perm = auc_permutation(y, z)
    row.update({"auc": perm["auc"], "auc_p_exact": perm["p_two_sided"], "auc_permutations": perm["permutations"],
                "auc_p_is_exact": perm["exact"]})
    mw = stats.mannwhitneyu(z[y == 1], z[y == 0], alternative="two-sided") if 0 < y.sum() < len(y) else None
    row["mann_whitney_p"] = float(mw.pvalue) if mw is not None else float("nan")
    row["mean_funded"] = float(x[y == 1].mean()) if y.sum() else float("nan")
    row["mean_not_funded"] = float(x[y == 0].mean()) if (y == 0).sum() else float("nan")
    return row


def method_convergence(ratings: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    """Per dimension, Spearman correlation between the mean rating and the rank score over the
    videos both methods measured: do the two Stage A methods see the same ordering?"""
    rows = []
    common = ratings.index.intersection(ranks.index)
    for code in CODES:
        if code not in ratings.columns or code not in ranks.columns:
            continue
        a = ratings.loc[common, code]
        b = ranks.loc[common, code]
        ok = a.notna() & b.notna()
        rho, p = (stats.spearmanr(a[ok], b[ok]) if ok.sum() >= 3 and a[ok].nunique() > 1 and b[ok].nunique() > 1
                  else (float("nan"), float("nan")))
        rows.append({"code": code, "videos": int(ok.sum()), "spearman": float(rho), "p": float(p),
                     "rating_distinct_values": int(a[ok].nunique()), "rating_sd": float(a[ok].std(ddof=1)),
                     "rank_score_sd": float(b[ok].std(ddof=1))})
    return pd.DataFrame(rows)


# --- the whole analysis ---------------------------------------------------------------------

def _fmt(x, digits=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if isinstance(x, (bool, np.bool_)):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{digits}f}"


def _md_table(frame: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 3) -> str:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_fmt(row[col], digits) if not isinstance(row[col], str) else row[col]
                              for col, _ in columns) + " |" for _, row in frame.iterrows()]
    return "\n".join([head, sep, *body])


@dataclass
class Analysis:
    """Everything the analysis computes, before anything is written."""
    outcomes: pd.DataFrame
    ratings: pd.DataFrame
    repeatability: pd.DataFrame
    ranks: pd.DataFrame
    factors: dict[str, PitchFactor]
    pf_scores: pd.DataFrame
    convergence: pd.DataFrame
    table3: pd.DataFrame
    table4: pd.DataFrame
    summary: dict
    hu_ma: pd.DataFrame | None = None                                   # Hu & Ma's factor and features, if given
    external: pd.DataFrame = field(default_factory=pd.DataFrame)        # Stage A factors against Hu & Ma's
    external_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    loading_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def compute(annotations_parquet: Path | str, scores_csv: Path | str | None, outcomes_csv: Path | str,
            provenance: dict | None = None, hu_ma_csv: Path | str | None = None) -> Analysis:
    """Compute every table from the exported files; writes nothing. hu_ma_csv, when given, is the
    file Hu & Ma's code wrote for the same videos (video_id, pitch_factor, their ten features)."""
    annotations = pd.read_parquet(annotations_parquet)
    outcomes = load_outcomes(outcomes_csv)
    hu_ma = load_hu_ma_pitch_factor(hu_ma_csv) if hu_ma_csv else None
    ratings = rating_means(annotations)
    repeat = rating_repeatability(annotations)
    ranks = rank_scores(scores_csv) if scores_csv else pd.DataFrame(index=pd.Index([], name="video_id"))

    labelled = outcomes.dropna(subset=["accepted"]).set_index("video_id")
    result: dict = {
        "videos_with_outcome_label": int(len(labelled)),
        "videos_in_outcomes_file": int(len(outcomes)),
        "acceptances": int(labelled["accepted"].sum()),
        "videos_rated": int(ratings.shape[0]),
        "videos_ranked": int(ranks.shape[0]),
        "provenance": provenance or {},
    }

    factors: dict[str, PitchFactor] = {}
    pf_scores = pd.DataFrame(index=ratings.index.union(ranks.index))
    for name, wide in (("ratings", ratings), ("ranks", ranks)):
        if wide.empty:
            continue
        try:
            pf = pitch_factor(wide)
        except ValueError as exc:
            result[f"pitch_factor_{name}"] = {"error": str(exc)}
            continue
        factors[name] = pf
        pf_scores[f"pitch_factor_{name}"] = pf.scores
        result[f"pitch_factor_{name}"] = {
            "explained_variance_share": pf.explained_variance_share,
            "dropped": pf.dropped,
            "videos": int(len(pf.scores)),
        }
    pf_scores.index.name = "video_id"

    convergence = method_convergence(ratings, ranks) if not ranks.empty else pd.DataFrame()
    if len(factors) == 2:
        both = pf_scores[["pitch_factor_ratings", "pitch_factor_ranks"]].dropna()
        if len(both) >= 3:
            rho, p = stats.spearmanr(both["pitch_factor_ratings"], both["pitch_factor_ranks"])
            result["pitch_factor_methods_spearman"] = {"spearman": float(rho), "p": float(p), "videos": int(len(both))}

    # Hu & Ma's own Pitch Factor: convergent validity of each Stage A factor, per dimension, and the loadings
    external, external_features, hu_ma_factor = pd.DataFrame(), pd.DataFrame(), None
    if hu_ma is not None:
        matched = hu_ma.index.intersection(pf_scores.index)
        info: dict = {"videos_in_file": int(len(hu_ma)), "videos_matched": int(len(matched))}
        pf_scores["pitch_factor_hu_ma"] = standardize(hu_ma["pitch_factor"]).reindex(pf_scores.index)
        try:
            hu_ma_factor = pitch_factor(hu_ma[[c for c in CODES if c in hu_ma.columns]])
            on_both = hu_ma_factor.scores.index.intersection(hu_ma.index)
            info["reproduced_from_features_corr"] = float(np.corrcoef(hu_ma_factor.scores.loc[on_both],
                                                                        hu_ma.loc[on_both, "pitch_factor"])[0, 1])
            info["explained_variance_share"] = hu_ma_factor.explained_variance_share
        except ValueError as exc:
            info["reproduced_error"] = str(exc)
        if len(matched) >= 4:
            external = external_agreement(pf_scores, hu_ma)
            external_features = external_feature_agreement(ratings, ranks, hu_ma)
        info["agreement"] = external.to_dict(orient="records")
        result["hu_ma_pitch_factor"] = info
    loading_table, loading_pairs = loading_agreement(factors, hu_ma_factor)
    result["loading_correlations"] = loading_pairs

    # Table III analogue: the Pitch Factor(s); Table IV analogue: every feature
    table3_rows, table4_rows = [], []
    if result["acceptances"] > 0 and result["acceptances"] < len(labelled):
        predictors = [(f"Pitch Factor ({name})", pf.scores) for name, pf in factors.items()]
        if hu_ma is not None:
            predictors.append(("Pitch Factor (Hu & Ma)", hu_ma["pitch_factor"]))
        for label, scores in predictors:
            joined = labelled.join(scores.rename("pitch_factor"), how="inner").dropna(subset=["pitch_factor"])
            y = joined["accepted"].to_numpy(dtype=int)
            if 0 < y.sum() < len(y):
                table3_rows.append(predictor_row(label, y, joined["pitch_factor"], joined["accelerator"]))
        methods = [("rating", ratings), ("rank score", ranks)]
        if hu_ma is not None:
            methods.append(("Hu & Ma feature", hu_ma.drop(columns="pitch_factor")))
        for name, wide in methods:
            for code in [c for c in CODES if c in wide.columns]:
                joined = labelled.join(wide[code], how="inner").dropna(subset=[code])
                y = joined["accepted"].to_numpy(dtype=int)
                if 0 < y.sum() < len(y) and joined[code].nunique() > 1:
                    table4_rows.append({"method": name, "code": code, **predictor_row(f"{code} ({name})", y, joined[code], joined["accelerator"])})
    table3 = pd.DataFrame(table3_rows)
    table4 = pd.DataFrame(table4_rows)
    result["table3"] = table3.to_dict(orient="records")
    result["table4_rows"] = int(len(table4))
    return Analysis(outcomes=outcomes, ratings=ratings, repeatability=repeat, ranks=ranks, factors=factors,
                    pf_scores=pf_scores, convergence=convergence, table3=table3, table4=table4, summary=result,
                    hu_ma=hu_ma, external=external, external_features=external_features, loading_table=loading_table)


def write(a: Analysis, out_dir: Path | str) -> dict:
    """Write every table and the summary of a computed analysis under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    a.ratings.to_csv(out / "rating_means.csv")
    a.repeatability.to_csv(out / "rating_repeatability.csv", index=False)
    if not a.ranks.empty:
        a.ranks.to_csv(out / "rank_scores_wide.csv")
    for name, pf in a.factors.items():
        pf.loadings.to_csv(out / f"pitch_factor_loadings_{name}.csv", index=False)
    a.pf_scores.to_csv(out / "pitch_factor_scores.csv")
    if not a.convergence.empty:
        a.convergence.to_csv(out / "method_convergence.csv", index=False)
    if not a.external.empty:
        a.external.to_csv(out / "hu_ma_agreement.csv", index=False)
        a.external_features.to_csv(out / "hu_ma_feature_agreement.csv", index=False)
    if not a.loading_table.empty:
        a.loading_table.to_csv(out / "loading_agreement.csv", index=False)
    a.table3.to_csv(out / "table3_pitch_factor.csv", index=False)
    a.table4.to_csv(out / "table4_features.csv", index=False)
    (out / "summary.json").write_text(json.dumps(a.summary, indent=2, default=str))
    (out / "summary.md").write_text(summary_markdown(a))
    return a.summary


def run_analysis(annotations_parquet: Path | str, scores_csv: Path | str | None, outcomes_csv: Path | str,
                 out_dir: Path | str, provenance: dict | None = None, hu_ma_csv: Path | str | None = None) -> dict:
    """Compute and write; return the headline numbers."""
    return write(compute(annotations_parquet, scores_csv, outcomes_csv, provenance, hu_ma_csv=hu_ma_csv), out_dir)


def summary_markdown(a: Analysis) -> str:
    return _summary_markdown(a.summary, a.repeatability, a.factors, a.convergence, a.table3, a.table4, a.outcomes,
                             a.external, a.external_features, a.loading_table)


def _summary_markdown(result, repeat, factors, convergence, table3, table4, outcomes,
                      external=None, external_features=None, loading_table=None) -> str:
    acc = outcomes.groupby("accelerator")["accepted"].agg(["count", "sum"]).rename(columns={"count": "videos", "sum": "accepted"})
    lines = ["# Stage A on the Hu & Ma videos and the funding decision", ""]
    lines += [f"Videos in the outcomes file: {result['videos_in_outcomes_file']}; with a 0/1 label: "
              f"{result['videos_with_outcome_label']}; accepted: {result['acceptances']}; rated: "
              f"{result['videos_rated']}; ranked: {result['videos_ranked']}.", ""]
    lines += ["Acceptances by accelerator family (label missing videos counted in `videos` only):", ""]
    acc = acc.reset_index()
    lines += [_md_table(acc, [("accelerator", "accelerator"), ("videos", "videos"), ("accepted", "accepted")], 0), ""]
    lines += ["## Ratings: self-consistency over replicates", ""]
    lines += [_md_table(repeat, [("code", "code"), ("videos", "videos"), ("ratings", "ratings"), ("mean", "mean"),
                                 ("sd_between_videos", "sd between videos"), ("mean_within_video_sd", "mean within-video sd"),
                                 ("exact_repeat_fraction", "identical replicates"), ("icc1", "ICC(1)"),
                                 ("distinct_values", "distinct values")]), ""]
    for name, pf in factors.items():
        lines += [f"## Pitch Factor from the {name}", "",
                  f"First principal component of the standardized measures; share of variance "
                  f"{pf.explained_variance_share:.3f}; videos {len(pf.scores)}."
                  + (f" Dropped: {pf.dropped}." if pf.dropped else ""), ""]
        lines += [_md_table(pf.loadings, [("code", "code"), ("loading", "loading"), ("uniqueness", "uniqueness")]), ""]
    if "pitch_factor_methods_spearman" in result:
        s = result["pitch_factor_methods_spearman"]
        lines += [f"Spearman correlation between the two Pitch Factors over {s['videos']} videos: "
                  f"{s['spearman']:.3f} (p = {s['p']:.3f}).", ""]
    if not convergence.empty:
        lines += ["## Do the ratings and the forced ranking order the videos the same way?", ""]
        lines += [_md_table(convergence, [("code", "code"), ("videos", "videos"), ("spearman", "Spearman"), ("p", "p"),
                                          ("rating_distinct_values", "distinct rating values"),
                                          ("rating_sd", "rating sd"), ("rank_score_sd", "rank score sd")]), ""]
    if "hu_ma_pitch_factor" in result:
        h = result["hu_ma_pitch_factor"]
        lines += ["## Against Hu & Ma's own Pitch Factor (convergent validity)", "",
                  f"Hu & Ma's Pitch Factor and features, computed with their code: {h['videos_in_file']} videos, "
                  f"{h['videos_matched']} matched to the Stage A runs."
                  + (f" Their file's factor is the first component of its ten features (r = "
                     f"{h['reproduced_from_features_corr']:.3f}; variance share {h['explained_variance_share']:.3f})."
                     if "reproduced_from_features_corr" in h else ""), ""]
        if external is not None and not external.empty:
            lines += [_md_table(external, [("method", "Stage A method"), ("videos", "videos"), ("pearson", "Pearson"),
                                           ("pearson_low", "95% low"), ("pearson_high", "95% high"),
                                           ("spearman", "Spearman"), ("spearman_p", "p"), ("kendall_tau", "Kendall tau"),
                                           ("same_tercile_share", "same tercile")]), ""]
            lines += ["Per dimension (Spearman with Hu & Ma's feature of the same name):", ""]
            lines += [_md_table(external_features, [("code", "code"), ("videos", "videos"), ("rating_spearman", "rating"),
                                                    ("rank_score_spearman", "rank score"),
                                                    ("hu_ma_distinct_values", "Hu & Ma distinct values")]), ""]
    if loading_table is not None and not loading_table.empty and len(loading_table.columns) > 2:
        lines += ["## Loadings of each Pitch Factor", ""]
        cols = [("code", "code")] + [(c, c.removeprefix("loading_")) for c in loading_table.columns if c != "code"]
        lines += [_md_table(loading_table, cols), ""]
        pairs = result.get("loading_correlations", {})
        if pairs:
            lines += ["Correlation between loading vectors over the codes: "
                      + ", ".join(f"{k.replace('_vs_', ' vs ')} {v:.2f}" for k, v in pairs.items()) + ".", ""]
    lines += ["## Table III analogue: I(invested) on the standardized Pitch Factor", ""]
    if table3.empty:
        lines += ["Not estimable: no variation in the outcome among the measured videos.", ""]
    else:
        lines += [_md_table(table3, [("predictor", "predictor"), ("n", "n"), ("events", "accepted"),
                                     ("ml_ame", "ML logit AME"), ("ml_ame_se", "SE"), ("ml_p", "p"),
                                     ("ml_pseudo_r2", "pseudo R²"), ("firth_coef", "Firth coef"), ("firth_se", "SE"),
                                     ("firth_p", "p"), ("firth_ame", "Firth AME"),
                                     ("firth_fe_coef", "Firth coef + accelerator"), ("firth_fe_p", "p"),
                                     ("auc", "AUC"), ("auc_p_exact", "exact p"), ("ml_note", "note")]), ""]
    lines += ["## Table IV analogue: one model per feature", ""]
    if table4.empty:
        lines += ["Not estimable.", ""]
    else:
        lines += [_md_table(table4, [("method", "method"), ("code", "code"), ("ml_ame", "ML logit AME"), ("ml_ame_se", "SE"),
                                     ("ml_p", "p"), ("firth_coef", "Firth coef"), ("firth_p", "p"), ("auc", "AUC"),
                                     ("auc_p_exact", "exact p"), ("mean_funded", "mean, funded"),
                                     ("mean_not_funded", "mean, not funded"), ("ml_note", "note")]), ""]
    lines += ["## Reading these numbers", "",
              "The ML logit column is Hu & Ma's Table III specification without accelerator fixed effects "
              "and without accelerator-year clustering: both need many clusters and many acceptances. "
              "The Firth column is the same model with Jeffreys' penalty, which stays finite under "
              "separation; the `+ accelerator` column adds accelerator-family indicators to it. The AUC p-value "
              "enumerates every assignment of the acceptances to the videos, so it is exact. With a handful "
              "of acceptances none of these can detect an effect of Hu & Ma's size (3 percentage points per "
              "standard deviation); the tables show the procedure and the point estimates, not evidence either way. "
              "Every measure is model-coded and not human-validated.", ""]
    return "\n".join(lines)


# --- for the notebook and the script --------------------------------------------------------

def run_provenance(stage_a_run: Path | str, rank_run: Path | str | None) -> dict:
    """What the two runs recorded about themselves: model, prompt hash, calls, design, completion."""
    stage_a_run = Path(stage_a_run)
    info: dict = {"stage_a_run": stage_a_run.name}
    manifest = stage_a_run / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        info["stage_a_model"] = m.get("model_requested")
        info["stage_a_passes"] = {k: {kk: v.get(kk) for kk in ("done", "failed", "prompt_hash", "model",
                                                              "input_tokens", "output_tokens")}
                                  for k, v in (m.get("passes") or {}).items()}
        info["stage_a_git_sha"] = m.get("git_sha")
    if rank_run is not None:
        rank_run = Path(rank_run)
        info["rank_run"] = rank_run.name
        design = rank_run / "ranking" / "design.json"
        if design.exists():
            d = json.loads(design.read_text())
            info["rank_settings"] = d.get("settings")
            info["rank_windows_from"] = d.get("windows_from")
            info["rank_release_id"] = d.get("release_id")
            info["rank_model"] = d.get("model")
            info["rank_summary"] = d.get("summary")
        completion = rank_run / "ranking" / "completion.json"
        if completion.exists():
            info["rank_completion"] = json.loads(completion.read_text())
        calls = rank_run / "ranking" / "calls.jsonl"
        if calls.exists():
            rows = [json.loads(ln) for ln in calls.read_text().splitlines() if ln.strip()]
            info["rank_calls"] = {"calls": len(rows), "ok": sum(1 for r in rows if r.get("ok")),
                                  "input_tokens": sum(int(r.get("input_tokens") or 0) for r in rows),
                                  "output_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
                                  "thinking_tokens": sum(int(r.get("thinking_tokens") or 0) for r in rows)}
    return info


def provenance_table(info: dict) -> pd.DataFrame:
    """The provenance dict as one two-column table, nested values flattened with dots."""
    rows = []

    def walk(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)
        else:
            rows.append({"item": prefix, "value": value if not isinstance(value, list) else json.dumps(value)})
    walk("", info)
    return pd.DataFrame(rows)


def acceptance_table(a: Analysis) -> pd.DataFrame:
    """Videos and acceptances per accelerator family; label-missing videos count in `videos` only."""
    t = a.outcomes.groupby("accelerator")["accepted"].agg(videos="count", labelled=lambda s: int(s.notna().sum()),
                                                           accepted=lambda s: int(s.fillna(0).sum()))
    t["videos"] = a.outcomes.groupby("accelerator").size()
    return t.reset_index()


def pitch_factor_figure(a: Analysis):
    """One scatter: the rating-based against the rank-based Pitch Factor, every video a dot, the
    accepted ones marked and labelled so identity does not rest on colour. None without both factors."""
    if len(a.factors) < 2:
        return None
    import matplotlib.pyplot as plt

    both = a.pf_scores.dropna().join(a.outcomes.set_index("video_id")[["accepted"]], how="left")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    rest = both[both["accepted"] != 1]
    funded = both[both["accepted"] == 1]
    ax.scatter(rest["pitch_factor_ratings"], rest["pitch_factor_ranks"], s=36, color="#9a9a96",
               edgecolor="#fcfcfb", linewidth=1.5, label="not accepted or unknown")
    ax.scatter(funded["pitch_factor_ratings"], funded["pitch_factor_ranks"], s=90, marker="D", color="#2a78d6",
               edgecolor="#fcfcfb", linewidth=1.5, label="accepted by the accelerator")
    for video_id, row in funded.iterrows():
        ax.annotate(video_id, (row["pitch_factor_ratings"], row["pitch_factor_ranks"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#3a3a38")
    lo = float(min(both.min().min(), -3)) - 0.2
    hi = float(max(both.max().max(), 3)) + 0.2
    ax.plot([lo, hi], [lo, hi], color="#d4d4d0", linewidth=1, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Pitch Factor from the ratings (standardized)")
    ax.set_ylabel("Pitch Factor from the rank scores (standardized)")
    s = a.summary.get("pitch_factor_methods_spearman", {})
    title = "The two Pitch Factors, one dot per video"
    if s:
        title += f" (Spearman {s['spearman']:.2f}, n = {s['videos']})"
    ax.set_title(title, fontsize=11)
    ax.grid(color="#ececea", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def external_readings(a: Analysis) -> list[str]:
    """One sentence per Stage A method on its agreement with Hu & Ma's Pitch Factor, from the numbers."""
    out = []
    for _, r in a.external.iterrows():
        if not np.isfinite(r["pearson"]):
            out.append(f"Pitch Factor ({r['method']}) against Hu & Ma's: not estimable on {int(r['videos'])} videos.")
            continue
        includes = r["pearson_low"] <= 0 <= r["pearson_high"]
        out.append(f"Pitch Factor ({r['method']}) against Hu & Ma's: Pearson {r['pearson']:+.2f} "
                   f"(95% interval {r['pearson_low']:+.2f} to {r['pearson_high']:+.2f}), Spearman {r['spearman']:+.2f}, "
                   f"same tercile for {r['same_tercile_share']:.0%} of {int(r['videos'])} videos (chance 33%): "
                   + ("the interval includes zero, so this sample shows no agreement at the factor level."
                      if includes else "the interval excludes zero."))
    return out


def external_feature_readings(a: Analysis, high: float = 0.5, low: float = 0.2) -> list[str]:
    """Which dimensions the Stage A measures and Hu & Ma's features order the same way, and which not."""
    if a.external_features.empty:
        return []
    out = []
    for name, col in (("mean rating", "rating_spearman"), ("rank score", "rank_score_spearman")):
        f = a.external_features.dropna(subset=[col])
        agree = f[f[col].abs() >= high]["code"].tolist()
        apart = f[f[col].abs() < low]["code"].tolist()
        out.append(f"Dimensions where the {name} and Hu & Ma's feature agree at |Spearman| of {high} or more: "
                   f"{', '.join(agree) or 'none'}; below {low}: {', '.join(apart) or 'none'}.")
    return out


def external_figure(a: Analysis):
    """One panel per Stage A Pitch Factor: Hu & Ma's factor on the x axis, the Stage A factor on the
    y axis, every video a dot, accepted videos marked and labelled. None without Hu & Ma's file."""
    if a.hu_ma is None or "pitch_factor_hu_ma" not in a.pf_scores.columns or not a.factors:
        return None
    import matplotlib.pyplot as plt

    names = [n for n in ("ratings", "ranks") if n in a.factors]
    both = a.pf_scores.join(a.outcomes.set_index("video_id")[["accepted"]], how="left")
    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 5.2), squeeze=False)
    rows = {r["method"]: r for _, r in a.external.iterrows()} if not a.external.empty else {}
    for ax, name in zip(axes[0], names):
        d = both.dropna(subset=["pitch_factor_hu_ma", f"pitch_factor_{name}"])
        rest, funded = d[d["accepted"] != 1], d[d["accepted"] == 1]
        ax.scatter(rest["pitch_factor_hu_ma"], rest[f"pitch_factor_{name}"], s=36, color="#9a9a96",
                   edgecolor="#fcfcfb", linewidth=1.5, label="not accepted or unknown")
        ax.scatter(funded["pitch_factor_hu_ma"], funded[f"pitch_factor_{name}"], s=90, marker="D", color="#2a78d6",
                   edgecolor="#fcfcfb", linewidth=1.5, label="accepted by the accelerator")
        for video_id, row in funded.iterrows():
            ax.annotate(video_id, (row["pitch_factor_hu_ma"], row[f"pitch_factor_{name}"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=8, color="#3a3a38")
        lo, hi = -3.4, 3.4
        ax.plot([lo, hi], [lo, hi], color="#d4d4d0", linewidth=1, zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Hu & Ma's Pitch Factor, their code (standardized)")
        ax.set_ylabel(f"Pitch Factor from the Stage A {name} (standardized)")
        r = rows.get(name)
        title = f"Stage A {name} against Hu & Ma"
        if r is not None and np.isfinite(r["pearson"]):
            title += f" (Pearson {r['pearson']:.2f}, Spearman {r['spearman']:.2f}, n = {int(r['videos'])})"
        ax.set_title(title, fontsize=10)
        ax.grid(color="#ececea", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    plt.close(fig)
    return fig


def loading_figure(a: Analysis):
    """Two panels over the ten codes: the loading of each Pitch Factor (Stage A ratings, Stage A
    ranks, Hu & Ma's features), and the per-dimension Spearman between each Stage A measure and
    Hu & Ma's feature of the same name. None without Hu & Ma's file."""
    if a.hu_ma is None or a.loading_table.empty or a.external_features.empty:
        return None
    import matplotlib.pyplot as plt

    codes = list(a.loading_table["code"])
    y = np.arange(len(codes))[::-1]
    series = [("loading_ratings", "Stage A ratings", "o", "#2a78d6"), ("loading_ranks", "Stage A ranks", "s", "#1d9e75"),
              ("loading_hu_ma", "Hu & Ma features", "D", "#d85a30")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for col, label, marker, color in series:
        if col in a.loading_table.columns:
            ax1.scatter(a.loading_table[col], y, marker=marker, s=48, color=color, edgecolor="#fcfcfb",
                        linewidth=1.0, label=label, zorder=3)
    ax1.axvline(0, color="#d4d4d0", linewidth=1, zorder=0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(codes)
    ax1.set_xlim(-1.05, 1.05)
    ax1.set_xlabel("loading on the Pitch Factor")
    ax1.set_title("How each factor weights the ten dimensions", fontsize=10)
    ax1.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    feats = a.external_features.set_index("code").reindex(codes)
    ax2.scatter(feats["rating_spearman"], y, marker="o", s=48, color="#2a78d6", edgecolor="#fcfcfb",
                linewidth=1.0, label="mean rating vs Hu & Ma feature", zorder=3)
    ax2.scatter(feats["rank_score_spearman"], y, marker="s", s=48, color="#1d9e75", edgecolor="#fcfcfb",
                linewidth=1.0, label="rank score vs Hu & Ma feature", zorder=3)
    ax2.axvline(0, color="#d4d4d0", linewidth=1, zorder=0)
    ax2.set_xlim(-1.05, 1.05)
    ax2.set_xlabel("Spearman with Hu & Ma's feature of the same name")
    ax2.set_title("Where the instruments order the videos alike", fontsize=10)
    ax2.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    for ax in (ax1, ax2):
        ax.grid(color="#ececea", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    plt.close(fig)
    return fig
