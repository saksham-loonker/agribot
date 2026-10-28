from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('scipy') is not None
if READY:
    from committee_outer_bound import relaxed_committee_upper_bound


@unittest.skipUnless(READY,'Optional training SciPy environment required')
class CommitteeOuterBoundTests(unittest.TestCase):
    def test_relaxation_does_not_exclude_a_safe_complementary_committee(self):
        r=relaxed_committee_upper_bound([[.99,.99]],[[1.,.1],[.1,1.]],.95)
        self.assertTrue(r['optimal']);self.assertEqual(r['relaxed_healthy_upper_bound'],1)

    def test_boundary_ties_are_counted_optimistically(self):
        r=relaxed_committee_upper_bound([[.9,.8,.2],[.8,.9,.2]],[[.9,.8,.2],[.8,.9,.2]],.85)
        self.assertTrue(r['optimal']);self.assertEqual(r['relaxed_healthy_upper_bound'],2)

    def test_uniformly_unsafe_experts_have_no_feasible_committee(self):
        r=relaxed_committee_upper_bound([[1.,1.]],[[.99,.98]],.95)
        self.assertEqual(r['status'],2);self.assertEqual(r['relaxed_healthy_upper_bound'],0)
