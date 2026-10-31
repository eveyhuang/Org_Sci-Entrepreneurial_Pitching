import unittest
import numpy as np
import pandas as pd
from analysis.concept_ranking import rank_concepts

class ConceptRankingTests(unittest.TestCase):
    def test_gain_sign_and_availability_matched_baseline(self):
        rows=[]
        for v in range(6):
            for j in range(4):
                y=j%2
                rows.append(dict(video_id=f'v{v}',speaker=f's{j}',offer=y,
                                 positivity=3,response=.5,evidence=.5,substance=1+4*y,
                                 response_missing=0,evidence_missing=0,substance_missing=0))
        ranking,metrics,predictions=rank_concepts(pd.DataFrame(rows),reps=49)
        self.assertEqual(ranking.iloc[0].concept,'substance')
        self.assertGreater(ranking.iloc[0].incremental_gain,0)
        metrics=metrics.set_index('model')
        for r in ranking.itertuples():
            self.assertAlmostEqual(r.incremental_gain,metrics.loc['without_'+r.concept,'log_loss']-metrics.loc['full','log_loss'])
            self.assertAlmostEqual(r.standalone_gain,metrics.loc['availability_baseline','log_loss']-metrics.loc['only_'+r.concept,'log_loss'])
        np.testing.assert_allclose(predictions.availability_baseline,predictions.offer_rate)
        self.assertEqual(len(predictions),24)

if __name__=='__main__':unittest.main()
