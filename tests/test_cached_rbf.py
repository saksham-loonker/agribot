import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from sklearn.svm import SVC
    from cached_rbf import training_gamma,restored_margin
except ImportError:np=None

@unittest.skipIf(np is None,'Scientific training dependencies unavailable')
class CachedRBFTests(unittest.TestCase):
    def test_explicit_arrays_reproduce_sklearn_binary_margin(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(24,2,5));y=np.arange(24)%2;gamma=training_gamma(x);m=SVC(C=1,gamma=gamma).fit(x.reshape(-1,5),np.repeat(y,2));q=rng.normal(size=(13,5));state=dict(classes=m.classes_,support_vectors=m.support_vectors_,dual_coef=m.dual_coef_,intercept=m.intercept_,gamma=gamma)
        np.testing.assert_allclose(restored_margin(q,state),m.decision_function(q),rtol=1e-12,atol=1e-12)
    def test_degenerate_bandwidth_rejected(self):
        with self.assertRaises(ValueError):training_gamma(np.zeros((4,2,3)))
    def test_bandwidth_depends_only_on_supplied_training_rows(self):
        x=np.arange(12,dtype=float).reshape(3,2,2);self.assertAlmostEqual(training_gamma(x),1/32)
