from pathlib import Path
import sys,tempfile,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from safe_feature_memmap import feature_memmap


class SafeFeatureMemmapTests(unittest.TestCase):
    def test_list_shape_round_trips_through_normal_numpy_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.npy';value=feature_memmap(path,[3,4,5]);value[:]=np.arange(60).reshape(3,4,5);value.flush()
            np.testing.assert_array_equal(np.load(path),np.arange(60).reshape(3,4,5))
            value._mmap.close()

    def test_wrong_existing_shape_cannot_be_reinterpreted(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.npy';feature_memmap(path,[3,4]).flush()
            with self.assertRaisesRegex(ValueError,'shape/dtype'):feature_memmap(path,[4,3])
