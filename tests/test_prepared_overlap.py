import hashlib,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import cv2,numpy as np
    from prepared_overlap import prepare_gray,prepared_features
    from phase106_cross_source_overlap import extract
except ImportError:np=None

@unittest.skipIf(np is None,'Image/scientific dependencies unavailable')
class PreparedOverlapTests(unittest.TestCase):
    def test_streamed_gray_matches_existing_original_and_mirrored_extractor(self):
        cv2.setNumThreads(1);rng=np.random.default_rng(23);rgb=rng.integers(0,256,(340,520,3),dtype=np.uint8);ok,encoded=cv2.imencode('.png',rgb);self.assertTrue(ok);payload=encoded.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'source.png';path.write_bytes(payload)
            for mirror in (False,True):
                old=extract(dict(path=str(path),file_sha256=hashlib.sha256(payload).hexdigest(),box=None,mirror=mirror));new=prepared_features(prepare_gray(payload,mirror))
                self.assertEqual(old[2],new[2]);np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
