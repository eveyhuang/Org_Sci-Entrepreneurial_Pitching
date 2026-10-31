"""Krippendorff's alpha over units coded by several coders, a bootstrap that resamples videos, and the agreement
statistics reported beside alpha per kind of code (Gwet's AC1/AC2, specific agreement, prevalence and bias indices,
shares beside their chance values, clock agreement, the exact binomial, Landis & Koch bands).

Pure functions. Follows Krippendorff (2011), Computing Krippendorff's Alpha-Reliability: a coincidence matrix
over values, observed disagreement against expected disagreement under the same value frequencies.

`stratified_alpha` keeps the variables of a mixed set apart: each variable (stratum) has its own coincidence
matrix, marginals and metric, and the disagreements add up across strata before the ratio is taken. Pooling units
of different variables into one matrix would put the variance between variables into the expected disagreement
and inflate alpha.
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.stats import beta, binomtest, spearmanr

Level = Literal["nominal", "ordinal", "interval"]
MIN_DEFINED_SHARE = 0.9         # a bootstrap interval is given when at least this share of the draws has an alpha
_CELLS = 4_000_000              # cells of the matrices held at once when draws are computed in chunks


def _numeric(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare(units: list[list[object | None]], level: Level, weights: list[float] | None = None):
    """The pairable units (two values readable at the level), and with `weights` (one per unit) the pairable units'
    weights beside them as a second list."""
    out, kept = [], []
    for i, unit in enumerate(units):
        values = [v for v in unit if v is not None and str(v) != ""]
        if level != "nominal":
            values = [n for n in (_numeric(v) for v in values) if n is not None]
        else:
            values = [str(v) for v in values]
        if len(values) >= 2:
            out.append(values)
            kept.append(1.0 if weights is None else float(weights[i]))
    return out if weights is None else (out, kept)


def _index(units: list[list[object]]) -> list:
    return sorted({v for unit in units for v in unit}, key=lambda v: (isinstance(v, str), v))


def _coincidences(units: list[list[object]], index: dict, k: int, weights: list[float] | None = None) -> np.ndarray:
    """The coincidence matrix of prepared units: every ordered pair of values of a unit, weighted 1/(m-1), and by the
    unit's weight when `weights` are given (a unit of weight 2 counts as that unit twice)."""
    o = np.zeros((k, k))
    for n, unit in enumerate(units):
        counts = np.zeros(k)
        for v in unit:
            counts[index[v]] += 1
        o += (1.0 if weights is None else weights[n]) * (np.outer(counts, counts) - np.diag(counts)) / (len(unit) - 1)
    return o


def _terms(o: np.ndarray, level: Level, values: list) -> tuple[np.ndarray, np.ndarray]:
    """(observed, expected) disagreement of coincidence matrices o of shape (b, k, k), one stratum each.

    The ordinal metric is built from each matrix's own marginals: (S_c - S_k)^2 with S the cumulative frequency
    up to the value less half its own, which is Krippendorff's (sum of n_g from c to k) - (n_c + n_k)/2.
    """
    n_c = o.sum(axis=2)
    n = n_c.sum(axis=1)
    if level == "nominal":
        delta = np.broadcast_to(1.0 - np.eye(o.shape[1]), o.shape)
    elif level == "interval":
        vals = np.array(values, dtype=float)
        delta = np.broadcast_to((vals[:, None] - vals[None, :]) ** 2, o.shape)
    else:
        s = np.cumsum(n_c, axis=1) - n_c / 2
        delta = (s[:, :, None] - s[:, None, :]) ** 2
    observed = (o * delta).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(n > 1, np.einsum("bi,bj,bij->b", n_c, n_c, delta) / (n - 1), 0.0)
    return observed, expected


def stratified_alpha(strata: dict[str, list[list[object | None]]], level: Level,
                     weights: dict[str, list[float]] | None = None) -> float | None:
    """Alpha over units grouped by variable: 1 - sum of observed disagreements / sum of expected ones, each
    stratum with its own coincidence matrix, marginals and metric. A stratum with fewer than two pairable units
    cannot estimate its own expected disagreement and adds to neither sum. None when no stratum qualifies or
    no stratum shows any variation.

    `weights` (per stratum, one weight per unit) weight each unit's pairs in the coincidence matrix, as frequency
    weights: a unit of weight w counts w times, the marginals and the n - 1 of the expected disagreement included.
    A validation sample stratified by outcome is reweighted to the run's mix so (review_apply, plan C Task 7f); the
    caller scales the weights to a mean of 1 over its rows, so n is close to the unweighted count (exactly so only
    when every stratum's rows carry the same mean weight).
    """
    observed = expected = 0.0
    qualifying = 0
    for name, units in strata.items():
        if weights is None:
            usable, unit_weights = _prepare(units, level), None
        else:
            usable, unit_weights = _prepare(units, level, weights[name])
        if len(usable) < 2:
            continue
        qualifying += 1
        values = _index(usable)
        o = _coincidences(usable, {v: i for i, v in enumerate(values)}, len(values), unit_weights)
        d_o, d_e = _terms(o[None], level, values)
        observed += float(d_o[0])
        expected += float(d_e[0])
    if not qualifying or expected == 0:
        return None
    return float(1.0 - observed / expected)


def alpha_units(strata: dict[str, list[list[object | None]]], level: Level) -> int:
    """The units `stratified_alpha` computes from: the pairable units (two values readable at the level) of the
    strata that hold at least two of them. A unit with a value that is no number at the ordinal or interval
    level, and a stratum with one pairable unit, are not among them."""
    usable = (_prepare(units, level) for units in strata.values())
    return sum(len(units) for units in usable if len(units) >= 2)


def krippendorff_alpha(units: list[list[object | None]], level: Level) -> float | None:
    """Alpha of `units` (one list per unit, one entry per coder, None = not coded) at a measurement level.

    Units with fewer than two values are dropped; at the ordinal and interval levels so are values that are
    not numbers. None when fewer than two units remain or when the values show no variation at all (the
    expected disagreement is then zero and alpha is undefined).
    """
    return stratified_alpha({"": units}, level)


def bootstrap_alpha_detail(strata_by_video: dict[str, dict[str, list[list[object | None]]]], level: Level,
                           n_boot: int = 1999, seed: int = 0) -> dict | None:
    """The 95% percentile interval of stratified alpha, resampling whole videos with replacement. Seeded.

    Returns {"ci95": [low, high] or None, "n_boot": draws, "n_defined": draws that had an alpha}; draws with no
    alpha (too few units, or no variation) are left out of the interval, so ci95 is None when fewer than
    MIN_DEFINED_SHARE of the draws are defined. None when no video has a unit. The coincidence matrix is additive
    over videos: one matrix per video and variable is built once and each draw sums the picked ones.
    """
    videos = [{s: _prepare(us, level) for s, us in per.items()} for per in strata_by_video.values()]
    videos = [v for v in videos if any(v.values())]
    if not videos:
        return None
    names = sorted({s for v in videos for s, us in v.items() if us})
    stacks = []
    for s in names:
        per_video = [v.get(s, []) for v in videos]
        values = _index([u for us in per_video for u in us])
        index = {v: i for i, v in enumerate(values)}
        mats = np.stack([_coincidences(us, index, len(values)) for us in per_video])
        stacks.append((values, mats.reshape(len(videos), -1), len(values), np.array([len(us) for us in per_video])))
    rng = np.random.default_rng(seed)
    picked = rng.integers(len(videos), size=(n_boot, len(videos)))
    biggest = max(k for _, _, k, _ in stacks)
    chunk = max(1, _CELLS // (biggest * biggest))
    alphas = []
    for start in range(0, n_boot, chunk):
        draws = picked[start:start + chunk]
        weights = np.zeros((len(draws), len(videos)))
        for row, got in enumerate(draws):
            weights[row] = np.bincount(got, minlength=len(videos))
        observed = np.zeros(len(draws))
        expected = np.zeros(len(draws))
        for values, flat, k, counts in stacks:
            d_o, d_e = _terms((weights @ flat).reshape(len(draws), k, k), level, values)
            counts_in_draw = weights @ counts
            counts_in_draw = counts_in_draw >= 2        # a stratum counts in a draw with two pairable units
            observed += np.where(counts_in_draw, d_o, 0.0)
            expected += np.where(counts_in_draw, d_e, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            alpha = 1.0 - observed / expected
        alphas.append(alpha[expected > 0])
    defined = np.concatenate(alphas) if alphas else np.array([])
    ci95 = None
    if len(defined) and len(defined) >= MIN_DEFINED_SHARE * n_boot:
        low, high = np.quantile(defined, [0.025, 0.975])
        ci95 = [float(low), float(high)]
    return {"ci95": ci95, "n_boot": n_boot, "n_defined": int(len(defined))}


def bootstrap_alpha(units_by_video: dict[str, list[list[object | None]]], level: Level,
                    n_boot: int = 1999, seed: int = 0) -> list[float] | None:
    """The 95% percentile interval [low, high] of alpha over one variable, resampling videos. Seeded."""
    detail = bootstrap_alpha_detail({v: {"": us} for v, us in units_by_video.items()}, level, n_boot, seed)
    return detail["ci95"] if detail else None


# --- Agreement statistics per kind of code (plan D) ------------------------------------------------------------------
# `pairs` is a list of (a, b) values, one per unit, rater a first (the coder, against the model as b); units with a
# missing value are dropped by the caller. A statistic that returns a plain float is NaN when there are no pairs.

Pair = tuple[object, object]


def _positions(pairs: list[Pair], categories: list) -> tuple[np.ndarray, np.ndarray]:
    """Each rater's values as positions in `categories`; ValueError for a value that is not among them or for a
    category listed twice. Values match by equality, so 4.0 is the category 4 but the string "4" is not."""
    where = {c: i for i, c in enumerate(categories)}
    if len(where) != len(categories):
        raise ValueError(f"a category is listed twice in {categories!r}")
    try:
        a = np.array([where[x] for x, _ in pairs], dtype=int)
        b = np.array([where[y] for _, y in pairs], dtype=int)
    except KeyError as missing:
        raise ValueError(f"value {missing.args[0]!r} is not among the categories {categories!r}") from None
    return a, b


def _table(pairs: list[Pair], categories: list) -> np.ndarray:
    """The q x q table of proportions p_ij: rater a's category i (rows) against rater b's j (columns)."""
    a, b = _positions(pairs, categories)
    q = len(categories)
    table = np.zeros((q, q))
    np.add.at(table, (a, b), 1.0)
    return table / len(pairs)


def _gwet(pairs: list[Pair], categories: list, weights: np.ndarray) -> float | None:
    """Gwet's weighted agreement coefficient: pa = sum w_ij p_ij; pe = sum_k pi_k (1 - pi_k) * sum w_ij / (q(q - 1)),
    pi_k the mean of the two raters' shares of category k (Gwet 2014, ch. 4)."""
    q = len(categories)
    if len(pairs) < 2 or q < 2:
        return None
    p = _table(pairs, categories)
    pi = (p.sum(axis=1) + p.sum(axis=0)) / 2
    pa = float((weights * p).sum())
    pe = float((pi * (1 - pi)).sum() * weights.sum() / (q * (q - 1)))     # at most 1/q, so 1 - pe > 0
    return (pa - pe) / (1 - pe)


def gwet_ac1(pairs: list[Pair], categories: list) -> float | None:
    """Gwet's AC1 for two raters on nominal categories (Gwet 2008, Br J Math Stat Psychol 61:29-48).

    pa is the share of identical pairs; pe = (1/(q - 1)) sum_k pi_k (1 - pi_k), q = len(categories) and pi_k the
    mean of the two raters' shares of category k; AC1 = (pa - pe)/(1 - pe). Unlike kappa or alpha, pe falls as one
    category dominates, so AC1 stays near the observed agreement under high prevalence. q is the length of the scale
    passed, not the number of values observed: pass the codebook's full list, unused categories included. None with
    fewer than two pairs or fewer than two categories; ValueError for a value outside `categories` or a category
    listed twice.
    """
    return _gwet(pairs, categories, np.eye(len(categories)))


def gwet_ac2(pairs: list[Pair], ordered_categories: list) -> float | None:
    """Gwet's AC2 with quadratic weights w_ij = 1 - ((i - j)/(q - 1))^2 over the ordered categories (Gwet 2014,
    Handbook of Inter-Rater Reliability, 4th ed., ch. 4): pa = sum w_ij p_ij; pe = (sum_k pi_k (1 - pi_k)) *
    (sum w_ij)/(q(q - 1)). A near miss earns partial credit. With two categories the weights are the identity and
    AC2 equals AC1. None with fewer than two pairs or fewer than two categories."""
    q = len(ordered_categories)
    if q < 2:
        return None
    i = np.arange(q)
    weights = 1.0 - ((i[:, None] - i[None, :]) / (q - 1)) ** 2
    return _gwet(pairs, ordered_categories, weights)


def _same(value, category) -> bool:
    """Equal and of the same type: True is not 1, and the string "true" is not True."""
    return type(value) is type(category) and value == category


def _two_by_two(pairs: list[Pair], positive, negative=None) -> tuple[int, int, int, int]:
    """(a, b, c, d): both positive; rater a positive and b negative; rater a negative and b positive; both negative.

    `negative` defaults to the one value present other than `positive` (none present: no negative answers). ValueError
    when `positive` and `negative` are the same, when no `negative` is given and two other values are present, or
    when a value is neither `positive` nor `negative` (a value of another type counts as another value)."""
    if negative is not None and _same(negative, positive):
        raise ValueError(f"positive and negative are the same value {positive!r}")
    others = []
    for v in (v for pair in pairs for v in pair):
        if not _same(v, positive) and not any(_same(v, o) for o in others):
            others.append(v)
    if negative is None:
        if len(others) > 1:
            raise ValueError(f"more than one value besides {positive!r}: {others!r}; name the negative value")
        negative = others[0] if others else None
    stray = [v for v in others if not _same(v, negative)]
    if stray:
        raise ValueError(f"value {stray[0]!r} is neither positive {positive!r} nor negative {negative!r}")
    a = sum(_same(x, positive) and _same(y, positive) for x, y in pairs)
    b = sum(_same(x, positive) and not _same(y, positive) for x, y in pairs)
    c = sum(not _same(x, positive) and _same(y, positive) for x, y in pairs)
    return a, b, c, len(pairs) - a - b - c


def specific_agreement(pairs: list[Pair], positive, negative=None) -> tuple[float | None, float | None]:
    """(positive agreement 2a/(2a + b + c), negative agreement 2d/(2d + b + c)) of a 2x2 table (Cicchetti & Feinstein
    1990, J Clin Epidemiol 43:551-558). Each is None when neither rater gave that answer. `negative` defaults to the
    one other value present; a value that is neither `positive` nor `negative`, of another type included (True is not
    1 or "true"), raises ValueError."""
    a, b, c, d = _two_by_two(pairs, positive, negative)
    pos = 2 * a / (2 * a + b + c) if 2 * a + b + c else None
    neg = 2 * d / (2 * d + b + c) if 2 * d + b + c else None
    return pos, neg


def prevalence_index(pairs: list[Pair], positive, negative=None) -> float:
    """|a - d|/n of the 2x2 table (Byrt, Bishop & Carlin 1993, J Clin Epidemiol 46:423-429): 0 when the two answers
    agreed on are equally common, near 1 when one dominates. NaN with no pairs; `negative` as in
    `specific_agreement`."""
    a, _, _, d = _two_by_two(pairs, positive, negative)
    return abs(a - d) / len(pairs) if pairs else float("nan")


def bias_index(pairs: list[Pair], positive, negative=None) -> tuple[float, str]:
    """(|b - c|/n, which rater says `positive` more: "a", "b" or "none") of the 2x2 table (Byrt, Bishop & Carlin
    1993, J Clin Epidemiol 46:423-429). b > c means rater a gives the positive answer more often. (NaN, "none") with
    no pairs; `negative` as in `specific_agreement`."""
    _, b, c, _ = _two_by_two(pairs, positive, negative)
    who = "a" if b > c else "b" if c > b else "none"
    return (abs(b - c) / len(pairs) if pairs else float("nan")), who


def modal_share(pairs: list[Pair]) -> float:
    """The most common value's share of all 2n values given by both raters (the prevalence the flag reads)."""
    if not pairs:
        return float("nan")
    counts: dict[object, int] = {}
    for x, y in pairs:
        counts[x] = counts.get(x, 0) + 1
        counts[y] = counts.get(y, 0) + 1
    return max(counts.values()) / (2 * len(pairs))


def modal_shares_by_rater(pairs: list[Pair]) -> tuple[float, float]:
    """(rater a's, rater b's) share of their own most common value. NaN with no pairs."""
    if not pairs:
        return float("nan"), float("nan")
    return modal_share([(x, x) for x, _ in pairs]), modal_share([(y, y) for _, y in pairs])


def within_one_share(pairs: list[Pair], ordered: list) -> float:
    """The share of pairs at most one step apart on the ordered scale."""
    if not pairs:
        return float("nan")
    a, b = _positions(pairs, ordered)
    return float(np.mean(np.abs(a - b) <= 1))


def _marginals(pairs: list[Pair], categories: list) -> tuple[np.ndarray, np.ndarray]:
    p = _table(pairs, categories)
    return p.sum(axis=1), p.sum(axis=0)


def chance_within_one(pairs: list[Pair], ordered: list) -> float:
    """The within-one share expected if the raters answered independently, each with their own observed
    distribution: sum over value pairs at most one step apart of (rater a's share of i x rater b's share of j)."""
    if not pairs:
        return float("nan")
    pa, pb = _marginals(pairs, ordered)
    i = np.arange(len(ordered))
    near = np.abs(i[:, None] - i[None, :]) <= 1
    return float((np.outer(pa, pb) * near).sum())


def chance_exact(pairs: list[Pair], categories: list | None = None) -> float:
    """The exact-agreement share expected if the raters answered independently, each with their own observed
    distribution: sum over values v of rater a's share of v times rater b's share of v (Cohen's pe). With
    `categories`, a value outside them raises ValueError as in the scale's other statistics; without, every distinct
    value is its own category. NaN with no pairs."""
    if not pairs:
        return float("nan")
    values = categories if categories is not None else list({v: None for pair in pairs for v in pair})
    pa, pb = _marginals(pairs, values)
    return float((pa * pb).sum())


def mean_difference(pairs: list[Pair], ordered: list) -> float:
    """The mean of (position of a's value - position of b's value) on the ordered scale; positive when rater a
    rates higher (the systematic difference of Bland & Altman 1986, on the scale's index)."""
    if not pairs:
        return float("nan")
    a, b = _positions(pairs, ordered)
    return float(np.mean(a - b))


def spearman(pairs: list[Pair], ordered: list) -> float | None:
    """Spearman's rho between the two raters' positions on the ordered scale (scipy.stats.spearmanr). None with
    fewer than two pairs or when either rater gives a single value (rho is undefined)."""
    if len(pairs) < 2:
        return None
    a, b = _positions(pairs, ordered)
    if len(set(a.tolist())) < 2 or len(set(b.tolist())) < 2:
        return None
    rho = float(spearmanr(a, b).statistic)
    return None if np.isnan(rho) else rho


def same_turn_agreement(a_seconds: float | None, b_seconds: float | None,
                        turns: list[tuple[object, float, float]]) -> dict:
    """Compare two times by their turn in one reference segmentation: (id, start, end) tuples.

    Intervals include their start and exclude their end, so a shared boundary belongs to the next turn.
    Both times must map to exactly one turn. Missing/nonfinite times, gaps and overlapping turns give None,
    not agreement or disagreement. This checks turn identity, not exact sentence onset or segmentation validity.
    """
    def locate(seconds):
        if seconds is None or not math.isfinite(seconds):
            return None
        matches = [tid for tid, start, end in turns if start <= seconds < end]
        return matches[0] if len(matches) == 1 else None

    a, b = locate(a_seconds), locate(b_seconds)
    return {'agreed': a == b if a is not None and b is not None else None,
            'a_turn_id': a, 'b_turn_id': b, 'rule': 'same_reference_turn',
            'interval': 'start_inclusive_end_exclusive'}


def turn_agreement_summary(matches: list[bool | None]) -> dict:
    """Share identifying the same turn, excluding and explicitly counting undetermined mappings."""
    known = [m for m in matches if m is not None]
    return {'same_turn': sum(known) / len(known) if known else None,
            'same_turn_n': len(known), 'same_turn_agree': sum(known),
            'same_turn_undetermined': len(matches) - len(known)}


def clock_agreement(diffs_seconds: list[float]) -> dict:
    """Agreement of two clock readings, from their differences in seconds (rater a minus rater b): the shares
    within 5 s and within 30 s (inclusive), the median absolute difference, the mean signed difference and the
    Bland & Altman (1986, Lancet 327:307-310) limits of agreement, mean +/- 1.96 x sample sd (ddof 1). Fields are
    None when there are no differences; the limits are None with fewer than two. ValueError for a difference that
    is NaN or infinite (the caller drops unreadable times before calling)."""
    d = np.asarray([float(x) for x in diffs_seconds], dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("a clock difference is NaN or infinite")
    out = {"share_within_5": None, "share_within_30": None, "median_abs": None, "mean_signed": None,
           "loa_low": None, "loa_high": None, "n": int(len(d))}
    if not len(d):
        return out
    mean = float(d.mean())
    out.update(share_within_5=float(np.mean(np.abs(d) <= 5)), share_within_30=float(np.mean(np.abs(d) <= 30)),
               median_abs=float(np.median(np.abs(d))), mean_signed=mean)
    if len(d) >= 2:
        sd = float(d.std(ddof=1))
        out.update(loa_low=mean - 1.96 * sd, loa_high=mean + 1.96 * sd)
    return out


def _check_binomial(k: int, n: int) -> None:
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")


def binomial_interval(k: int, n: int) -> tuple[float, float]:
    """The Clopper & Pearson (1934, Biometrika 26:404-413) exact 95% interval of a proportion k/n, from beta
    quantiles: lower = Beta(0.025; k, n - k + 1) (0 when k = 0), upper = Beta(0.975; k + 1, n - k) (1 when k = n).
    (NaN, NaN) when n = 0; ValueError unless 0 <= k <= n."""
    _check_binomial(k, n)
    if n == 0:
        return float("nan"), float("nan")
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return low, high


def binomial_p_vs_half(k: int, n: int) -> float:
    """The two-sided exact binomial p of k successes in n against 0.5 (scipy.stats.binomtest). NaN when n = 0;
    ValueError unless 0 <= k <= n."""
    _check_binomial(k, n)
    if n == 0:
        return float("nan")
    return float(binomtest(k, n, 0.5).pvalue)


LANDIS_KOCH = ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial"))


def landis_koch_band(value: float | None) -> str | None:
    """The Landis & Koch (1977, Biometrics 33:159-174) benchmark band of a chance-corrected statistic: < 0 poor,
    0-0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, 0.81-1.00 almost perfect. The value is
    rounded to two decimals first, as the tables print it, so the band always agrees with the printed number:
    0.204 and 0.2001 are slight, 0.8001 substantial, and -0.004 (printed -0.00) slight. None for None or NaN."""
    if value is None or np.isnan(value):
        return None
    value = round(float(value), 2) + 0.0          # + 0.0 turns -0.0 into 0.0
    if value < 0:
        return "poor"
    for upper, band in LANDIS_KOCH:
        if value <= upper:
            return band
    return "almost perfect"
