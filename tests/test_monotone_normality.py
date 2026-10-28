from pathlib import Path
import importlib.util,sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('scipy') is not None
if READY:
    from monotone_normality import fit_normality,predict_normality


@unittest.skipUnless(READY,'Optional training SciPy environment required')
class NormalityCalibrationTests(unittest.TestCase):
    def test_more_anomalous_features_cannot_raise_healthy_probability(self):
        x=np.array([[0.,.1],[.1,.2],[.2,.1],[1.,1.2],[1.1,1.3],[1.2,1.1]])
        model=fit_normality(x,np.array([1,1,1,0,0,0]),np.ones(6))
        low=predict_normality(model,np.array([[.2,.2]]));high=predict_normality(model,np.array([[.9,.9]]))
        self.assertGreater(float(low[0]),float(high[0]));self.assertTrue((model['coefficient']<=0).all())

    def test_single_class_calibration_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'cohort'):fit_normality(np.zeros((3,2)),np.ones(3),np.ones(3))
