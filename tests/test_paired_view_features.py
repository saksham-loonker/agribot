import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from paired_view_features import paired_view_features
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class PairedViewTests(unittest.TestCase):
    def test_swapping_inputs_swaps_outputs_and_preserves_minimum(self):
        rng=np.random.default_rng(23);v=rng.normal(size=(4,2,7));a=paired_view_features(v);b=paired_view_features(v[:,::-1]);np.testing.assert_array_equal(a[:,::-1],b);w=rng.normal(size=28);np.testing.assert_array_equal((a@w).min(1),(b@w).min(1))
    def test_changing_other_view_changes_own_conditional_input(self):
        v=np.array([[[1.,0.],[1.,0.]]]);a=paired_view_features(v);v[:,1]=[0.,1.];b=paired_view_features(v);np.testing.assert_array_equal(a[:,0,:2],b[:,0,:2]);self.assertGreater(float(abs(a[:,0,2:]-b[:,0,2:]).sum()),0.)
    def test_zero_and_invalid_views(self):
        self.assertTrue(np.isfinite(paired_view_features(np.zeros((3,2,5)))).all())
        with self.assertRaises(ValueError):paired_view_features(np.ones((3,3,5)))
