"""Model range reuse must preserve offsets, selected rows and byte integrity."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from hf_range_tensors import RangeTensors


class RangeTensorTests(unittest.TestCase):
    def fixture(self,root):
        model=RangeTensors.__new__(RangeTensors);model.root=Path(root);model.data_start=64
        value=np.arange(60,dtype='<f4').reshape(20,3)
        model.header={'vocabulary':dict(dtype='F32',shape=[20,3],data_offsets=[16,256])}
        payload=b'x'*80+value.tobytes()+b'z'*12
        path=Path(root)/f'0-{len(payload)-1}.bin';path.write_bytes(payload)
        model.ranges=[(0,len(payload)-1,path)]
        return model,value,path

    def test_superset_range_uses_tensor_offset(self):
        with tempfile.TemporaryDirectory() as root:
            model,value,_=self.fixture(root)
            np.testing.assert_array_equal(model.array('vocabulary'),value)

    def test_sparse_vocabulary_rows_use_original_ids(self):
        with tempfile.TemporaryDirectory() as root:
            model,value,_=self.fixture(root)
            np.testing.assert_array_equal(model.rows('vocabulary',[19,1,10,1]),value[[1,10,19]])
            for indices in ([-1],[20],[]):
                with self.assertRaises(AssertionError):model.rows('vocabulary',indices)

    def test_on_disk_range_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            model,_,path=self.fixture(root);payload=path.read_bytes()
            path.with_suffix('.json').write_text(json.dumps(dict(sha256=hashlib.sha256(payload).hexdigest())))
            model.ranges=[]
            self.assertEqual(model.blob(0,len(payload)-1),path)
            model.ranges=[];path.write_bytes(b'!'+payload[1:])
            with self.assertRaises(AssertionError):model.blob(0,len(payload)-1)
