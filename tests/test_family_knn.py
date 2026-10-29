from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
import numpy as np
READY=importlib.util.find_spec('torch') is not None
if READY:from family_knn import family_knn_margin


@unittest.skipUnless(READY,'Optional training Torch environment required')
class FamilyKnnTests(unittest.TestCase):
    def test_duplicate_views_cannot_multiply_family_votes(self):
        score=family_knn_margin([[1],[1],[1],[-1],[1]],[1,1,1,0,0],[0,0,0,1,2],[1]*5,[[0]],k=3)
        self.assertAlmostEqual(float(1/(1+np.exp(-score[0]))),1/3,places=6)

    def test_nearer_supported_family_changes_margin_in_the_expected_direction(self):
        score=family_knn_margin([[0],[2]],[1,0],[0,1],[1,1],[[.1],[1.9]],k=2)
        self.assertGreater(score[0],0);self.assertLess(score[1],0)

    def test_insufficient_distinct_family_evidence_is_rejected(self):
        with self.assertRaises(ValueError):family_knn_margin([[0],[1]],[1,0],[0,0],[1,1],[[.5]],k=2)
