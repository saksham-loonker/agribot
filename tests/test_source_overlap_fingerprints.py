"""Evidence-integrity checks for the provenance audit, independent of labels."""
from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util
import hashlib,zlib
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('cv2') is not None
if READY:
    from phase106_cross_source_overlap import Cache,verify
    from phase112_development_source_audit import profile


@unittest.skipUnless(READY,'Optional training OpenCV environment required')
class SourceOverlapTests(unittest.TestCase):
    def test_identity_tracks_bytes_crop_mirror_and_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Cache(Path(directory)/'features.sqlite3')
            row=dict(file_sha256='a'*64,box=None)
            key=cache.key(row)
            for change in [dict(file_sha256='b'*64),dict(box=[0,0,100,100]),dict(mirror=True)]:
                self.assertNotEqual(key,cache.key(dict(row,**change)))
            self.assertEqual(key,cache.key(dict(row,path='/another/location.jpg')))
            cache.version='changed-preprocessing'
            self.assertNotEqual(key,cache.key(row))
            cache.db.close()

    def test_lossless_cache_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            cache=Cache(Path(directory)/'features.sqlite3');row=dict(file_sha256='a'*64)
            feature=(np.array([[12.,32.]],np.float32),np.ones((1,128),np.float32),(200,300))
            cache.put(row,feature)
            for actual,expected in zip(cache.get(row),feature):np.testing.assert_array_equal(actual,expected)
            cache.db.execute('UPDATE features SET payload=?',(b'corrupted',))
            with self.assertRaisesRegex(AssertionError,'corrupted'):cache.get(row)
            cache.db.close()

    def test_geometry_accepts_transform_and_rejects_insufficient_support(self):
        rng=np.random.default_rng(2106)
        points=rng.uniform(20,180,(40,2)).astype(np.float32)
        descriptors=rng.uniform(0,255,(40,128)).astype(np.float32)
        original=(points,descriptors,(200,200))
        translated=(points+np.array([3,5],np.float32),descriptors.copy(),(210,210))
        result=verify(original,translated,2106)
        self.assertIsNotNone(result)
        self.assertEqual(result['inliers'],40)
        self.assertIsNone(verify(original,(points[:10],descriptors[:10],(200,200)),2106))

    def test_repeated_texture_is_not_independent_geometric_support(self):
        rng=np.random.default_rng(6109)
        target=rng.uniform(0,255,(20,128)).astype(np.float32)
        repeated=target[np.arange(40)%7]+rng.normal(0,.01,(40,128)).astype(np.float32)
        points=rng.uniform(0,200,(40,2)).astype(np.float32)
        target_points=rng.uniform(0,200,(20,2)).astype(np.float32)
        self.assertIsNone(verify((points,repeated,(200,200)),(target_points,target,(200,200)),2106))

    def test_profile_keeps_encoded_and_decoded_identities_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'photo.png'
            rgb=np.random.default_rng(112).integers(0,256,(48,32,3),dtype=np.uint8)
            Image.fromarray(rgb).save(path);encoded=path.read_bytes()
            result=profile(dict(path=str(path),crc32=zlib.crc32(encoded)))
            self.assertEqual(result['file_sha256'],hashlib.sha256(encoded).hexdigest())
            self.assertEqual(result['decoded_rgb_sha256'],hashlib.sha256(rgb.tobytes()).hexdigest())
            self.assertEqual(result['orientation'],'portrait')

    def test_profile_rejects_changed_member_before_admitting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'photo.png';path.write_bytes(b'changed-source-bytes')
            with self.assertRaisesRegex(AssertionError,'CRC mismatch'):
                profile(dict(path=str(path),crc32=zlib.crc32(path.read_bytes())^1))


if __name__=='__main__':unittest.main()
