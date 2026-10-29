import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from structured_leaf_features import structured_leaf_features
except ImportError:np=None

@unittest.skipIf(np is None,'Image/scientific dependencies unavailable')
class StructuredFeaturesTests(unittest.TestCase):
    def image(self,lines):
        y,x=np.ogrid[:64,:64];mask=(x-31.5)**2+(y-31.5)**2<25**2;rgb=np.full((64,64,3),.03,np.float32);rgb[mask]=[.2,.6,.2]
        if lines:rgb[mask&((abs(x-31.5)<2)|(abs(y-31.5)<1))]=[.55,.8,.5]
        return rgb
    def test_finite_descriptor_responds_to_internal_lines(self):
        a=structured_leaf_features(self.image(False));b=structured_leaf_features(self.image(True))
        self.assertEqual(b.shape,(69,));self.assertTrue(np.isfinite(b).all());self.assertGreater(float(np.abs(a[12:54]-b[12:54]).max()),.01)
    def test_empty_image_has_no_foreground_support(self):
        np.testing.assert_array_equal(structured_leaf_features(np.zeros((32,32,3),np.float32)),np.zeros(69))
    def test_rgb_units_and_nan_rejected(self):
        for x in (np.ones((32,32,3))*255,np.full((32,32,3),np.nan)):
            with self.assertRaises(ValueError):structured_leaf_features(x)
    def test_quarter_turn_preserves_sorted_spatial_summary(self):
        x=self.image(True);a=structured_leaf_features(x);b=structured_leaf_features(np.rot90(x))
        np.testing.assert_allclose(a,b,atol=.01,rtol=.01)
