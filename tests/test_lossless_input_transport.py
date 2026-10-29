import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import numpy as np
    from lossless_input_transport import pack_views,unpack_views
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class LosslessTransportTests(unittest.TestCase):
    def test_exact_float_bits_survive_including_signed_zero(self):
        rng=np.random.default_rng(23);x=rng.normal(size=(3,5,7,3)).astype(np.float32);x[0,0,0]=[-0.,0.,np.finfo(np.float32).tiny];planes,shape=pack_views(x);actual=unpack_views(planes,shape);np.testing.assert_array_equal(actual.view(np.uint32),x.view(np.uint32))
    def test_corrupt_shape_and_nonfinite_rejected(self):
        p,s=pack_views(np.ones((2,4,4,3),np.float32))
        with self.assertRaises(ValueError):unpack_views(p,[1,4,4,3])
        with self.assertRaises(ValueError):pack_views(np.full((1,4,4,3),np.nan))
