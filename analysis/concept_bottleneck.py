"""Exploratory frozen-MLLM concept bottleneck with pitch-held-out prediction heads.

No video/embedding path bypasses the named concepts. Availability and exposure
controls are explicit; interventions are model sensitivity, not causal effects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.auc import grouped_auc
from analysis.project_v2 import BASE, RESPONSE, score

CONCEPTS = {
    'positivity': ('Prepared-presentation positivity', 1., 5.),
    'response': ('Answer directness', 0., 1.),
    'evidence': ('Share of answers with checkable support', 0., 1.),
    'substance': ('Answer substance and specificity', 1., 5.),
}
MASKS = [c + '_missing' for c in RESPONSE]
MODELS = {
    'intercept': [],
    'prepared_exposure_baseline': BASE,
    'concept_bottleneck': list(CONCEPTS) + MASKS,
    'concepts_plus_exposure': BASE + RESPONSE,
    'probe_adjusted_baseline': BASE + ['probe_severity', 'promotion_probe'],
    'probe_adjusted_concepts': BASE + RESPONSE + ['probe_severity', 'promotion_probe'],
}


def predict_head(values, features, fold):
    """Apply the training-fold preprocessing and fitted head to concept vectors."""
    x = np.asarray(values, dtype=float).reshape(-1, len(features))
    if fold['single_class_training']:
        return np.full(len(x), fold['constant_prediction'])
    med = np.asarray(fold['training_medians'])
    x = np.where(np.isfinite(x), x, med)
    z = (x - fold['training_means_after_imputation']) / fold['training_sd']
    b = np.asarray(fold['coefficients_standardized'])
    logits = b[0] + z @ b[1:]
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def evaluate(decisions, reps=1999, seed=20260928):
    """Same outcome rows/folds for all heads; no concept fitting on outcome labels."""
    d = decisions.reset_index(drop=True).copy()
    if d.duplicated(['video_id', 'speaker']).any():
        raise ValueError('Investor rows must be unique within a pitch')
    if d.offer.isna().any() or not d.offer.isin([0, 1]).all():
        raise ValueError('Use only known binary on-air outcomes')
    if d.video_id.nunique() < 3 or d.offer.nunique() != 2:
        raise ValueError('At least three pitches and both outcomes are required')
    required = set().union(*map(set, MODELS.values()))
    if required - set(d):
        raise ValueError(f'Missing predictors: {sorted(required-set(d))}')
    for c, (_, low, high) in CONCEPTS.items():
        x = d[c].dropna()
        if not x.between(low, high).all():
            raise ValueError(f'{c} is outside the codebook range')
    if d.positivity.isna().any():
        raise ValueError('Missing prepared positivity requires an explicit availability policy')
    for c in RESPONSE:
        if not np.array_equal(d[c].isna().astype(int), d[c+'_missing']):
            raise ValueError(f'Inconsistent missingness flag for {c}')
    rows = d.to_dict('records')
    predictions = d[['video_id', 'speaker', 'offer']].copy()
    models, metrics = {}, []
    for name, features in MODELS.items():
        result = grouped_auc(rows, features)
        for fold in result['folds']:
            train = d[d.video_id.ne(fold['held_out_video'])]
            fold['training_video_ids'] = sorted(train.video_id.unique())
            fold['constant_prediction'] = float((train.offer.sum()+1)/(len(train)+2))
        p = pd.DataFrame(result['predictions']).rename(columns={'prediction': name})
        predictions = predictions.merge(p[['video_id', 'speaker', name]], on=['video_id', 'speaker'], validate='one_to_one')
        metrics.append(dict(model=name, n=len(d), pitches=d.video_id.nunique(),
                            **score(predictions.offer, predictions[name]),
                            mean_within_pitch_auc=result['mean_within_video_auc']))
        models[name] = result
    # Paired conditional uncertainty for the controlled comparison already in H1.
    rng = np.random.default_rng(seed)
    groups = [g.index.to_numpy() for _, g in predictions.groupby('video_id')]
    pairs = [('prepared_exposure_baseline', 'concepts_plus_exposure'),
             ('probe_adjusted_baseline', 'probe_adjusted_concepts')]
    deltas = []
    for base, added in pairs:
        samples = []
        for _ in range(reps):
            ix = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
            pp = predictions.iloc[ix]
            samples.append(score(pp.offer, pp[added])['log_loss']-score(pp.offer, pp[base])['log_loss'])
        lo, hi = np.quantile(samples, [.025, .975])
        deltas.append(dict(comparison=f'{added} minus {base}',
                           delta_log_loss=score(predictions.offer, predictions[added])['log_loss']-score(predictions.offer, predictions[base])['log_loss'],
                           ci_low=lo, ci_high=hi, reps=reps,
                           uncertainty='Conditional pitch bootstrap of fixed held-out predictions; no model refitting'))
    coverage = pd.DataFrame([dict(concept=c, name=label, codable=int(d[c].notna().sum()),
                                 missing=int(d[c].isna().sum()), scale_min=lo, scale_max=hi)
                             for c, (label, lo, hi) in CONCEPTS.items()])
    return dict(cohort=d, coverage=coverage, metrics=pd.DataFrame(metrics),
                predictions=predictions, deltas=pd.DataFrame(deltas), models=models)


def edit_concept(result, video_id, speaker, concept, value, model='concept_bottleneck'):
    """Hypothetical edit to one codable concept using that pitch's held-out head."""
    if concept not in CONCEPTS:
        raise ValueError('Only named bottleneck concepts can be edited')
    label, lo, hi = CONCEPTS[concept]
    if not np.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f'{label} must lie between {lo} and {hi}')
    rows = result['cohort'].query('video_id == @video_id and speaker == @speaker')
    if len(rows) != 1:
        raise ValueError('Select exactly one pitch/investor row')
    row = rows.iloc[0]
    if pd.isna(row[concept]):
        raise ValueError('Cannot edit an unobserved concept as if it were measured')
    head = result['models'][model]
    features = head['features']
    if concept not in features:
        raise ValueError('Concept is not used by the selected head')
    fold = next(f for f in head['folds'] if f['held_out_video'] == video_id)
    before = row[features].to_numpy(float)
    after = before.copy(); after[features.index(concept)] = value
    p = predict_head(np.vstack([before, after]), features, fold)
    return dict(video_id=video_id, speaker=speaker, concept=concept, name=label,
                observed_value=float(row[concept]), hypothetical_value=float(value),
                original_probability=float(p[0]), edited_probability=float(p[1]),
                change_percentage_points=float(100*(p[1]-p[0])))


def contributions(result, video_id, speaker, model='concept_bottleneck'):
    """Exact additive decomposition of the held-out prediction in log-odds units."""
    head = result['models'][model]; features = head['features']
    fold = next(f for f in head['folds'] if f['held_out_video'] == video_id)
    row = result['cohort'].set_index(['video_id', 'speaker']).loc[(video_id, speaker)]
    if fold['single_class_training']:
        raise ValueError('No feature attribution for single-class training')
    raw = row[features].to_numpy(float)
    imputed = np.where(np.isfinite(raw), raw, fold['training_medians'])
    z = (imputed-fold['training_means_after_imputation'])/fold['training_sd']
    b = fold['coefficients_standardized']
    out = [dict(feature='intercept', name='Training-fold intercept', log_odds=b[0], imputed=False)]
    for i, f in enumerate(features):
        name = CONCEPTS[f][0] if f in CONCEPTS else CONCEPTS[f.removesuffix('_missing')][0]+' unavailable'
        out.append(dict(feature=f, name=name, log_odds=float(z[i]*b[i+1]), imputed=not np.isfinite(raw[i])))
    return pd.DataFrame(out)
