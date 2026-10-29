from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('scipy') is not None
if READY:
    from committee_feasibility import optimistic_committee


@unittest.skipUnless(READY,'Optional training SciPy environment required')
class CommitteeFeasibilityTests(unittest.TestCase):
    def test_committee_can_cancel_different_experts_false_positives(self):
        r=optimistic_committee([[.99,.99]],[[1.,.1],[.1,1.]],.95)
        self.assertTrue(r['optimal']);self.assertEqual(r['admitted_healthy'],1);self.assertLess(r['maximum_negative'],.95)

    def test_identical_positive_and_negative_cannot_be_separated(self):
        r=optimistic_committee([[.9,.8,.2],[.8,.9,.2]],[[.9,.8,.2],[.8,.9,.2]],.85)
        self.assertTrue(r['optimal']);self.assertEqual(r['admitted_healthy'],0);self.assertEqual(r['optimistic_healthy_upper_bound'],0)

    def test_no_safe_convex_weights_is_reported_infeasible(self):
        r=optimistic_committee([[1.,1.]],[[.99,.98]],.95)
        self.assertFalse(r['feasible']);self.assertEqual(r['status'],2)
