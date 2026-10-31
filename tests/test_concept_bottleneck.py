"""Prediction-head isolation and intervention semantics, with synthetic concepts."""
import unittest
import numpy as np
import pandas as pd
from analysis.concept_bottleneck import evaluate, edit_concept, contributions

class ConceptBottleneckTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        rows=[]
        for v in range(6):
            for j in range(4):
                rows.append(dict(video_id=f'v{v}',speaker=f's{j}',offer=(v+j)%2,
                                 positivity=1+v%5,response=rng.random(),evidence=rng.random(),
                                 substance=1+4*rng.random(),n_exchanges=j,no_exchange=int(j==0),
                                 response_missing=0,evidence_missing=0,substance_missing=0,
                                 probe_severity=1+v%3,promotion_probe=rng.random()))
        self.d=pd.DataFrame(rows)
        self.result=evaluate(self.d,reps=19)

    def test_held_out_outcomes_cannot_change_their_own_head(self):
        changed=self.d.copy()
        changed.loc[changed.video_id.eq('v0'),'offer']=1-changed.loc[changed.video_id.eq('v0'),'offer']
        other=evaluate(changed,reps=19)
        for name,model in self.result['models'].items():
            fold=next(f for f in model['folds'] if f['held_out_video']=='v0')
            self.assertNotIn('v0',fold['training_video_ids'])
            p=self.result['predictions'].query("video_id == 'v0'")[name]
            q=other['predictions'].query("video_id == 'v0'")[name]
            np.testing.assert_allclose(p,q)

    def test_training_median_excludes_held_out_pitch(self):
        d=self.d.copy();d.loc[d.video_id.eq('v0'),'substance']=5
        r=evaluate(d,reps=19);m=r['models']['concept_bottleneck']
        f=next(f for f in m['folds'] if f['held_out_video']=='v0')
        i=m['features'].index('substance')
        self.assertAlmostEqual(f['training_medians'][i],d.loc[d.video_id.ne('v0'),'substance'].median())

    def test_unchanged_edit_and_contributions_recover_held_out_prediction(self):
        row=self.d.iloc[0]
        z=edit_concept(self.result,row.video_id,row.speaker,'response',row.response)
        p=self.result['predictions'].iloc[0].concept_bottleneck
        self.assertAlmostEqual(z['original_probability'],p)
        self.assertAlmostEqual(z['edited_probability'],p)
        c=contributions(self.result,row.video_id,row.speaker)
        self.assertAlmostEqual(1/(1+np.exp(-c.log_odds.sum())),p)
        with self.assertRaises(ValueError):edit_concept(self.result,'v0','s0','response',2)

    def test_missing_concept_cannot_be_presented_as_observed_edit(self):
        d=self.d.copy();d.loc[0,'response']=np.nan;d.loc[0,'response_missing']=1
        r=evaluate(d,reps=19)
        with self.assertRaises(ValueError):edit_concept(r,'v0','s0','response',1)
        d.loc[0,'response_missing']=0
        with self.assertRaises(ValueError):evaluate(d,reps=19)

if __name__=='__main__':unittest.main()
