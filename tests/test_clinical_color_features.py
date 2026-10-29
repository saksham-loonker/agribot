from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
import numpy as np
READY=importlib.util.find_spec('cv2') is not None
if READY:from clinical_color_features import clinical_color_features


@unittest.skipUnless(READY,'Optional training OpenCV environment required')
class ClinicalColorFeaturesTests(unittest.TestCase):
    def test_neutral_image_has_finite_fallback_without_fake_tissue_labels(self):
        v=clinical_color_features(np.full((32,32,3),.5,np.float32));self.assertEqual(v.shape,(75,));self.assertTrue(np.isfinite(v).all());self.assertEqual(v[64],0)

    def test_chromatic_difference_is_retained(self):
        a=np.empty((32,32,3),np.float32);a[:]=[.2,.6,.1];b=a.copy();b[:]=[.6,.6,.1]
        self.assertGreater(float(np.linalg.norm(clinical_color_features(a)-clinical_color_features(b))),.1)

    def test_wrong_pixel_scale_is_rejected(self):
        with self.assertRaises(ValueError):clinical_color_features(np.full((32,32,3),128,np.float32))
        with self.assertRaises(ValueError):clinical_color_features(np.full((32,32,3),np.nan,np.float32))
