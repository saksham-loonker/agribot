from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from native_joint_features import join_native_cnn


class NativeFeatureJoinTests(unittest.TestCase):
    def test_native_and_parent_namespaces_can_have_different_orders(self):
        native=np.stack((np.full(1024,2,np.float32),np.full(1024,1,np.float32)))
        links=[dict(input_sha256='a'*64,parent_224_input_sha256='d'*64),dict(input_sha256='b'*64,parent_224_input_sha256='c'*64)]
        cnn={'c'*64:np.full(768,30,np.float32),'d'*64:np.full(768,40,np.float32)}
        result=join_native_cnn(native,['b'*64,'a'*64],links,cnn)
        np.testing.assert_array_equal(result[0,:1024],1.);np.testing.assert_array_equal(result[0,1024:],40.);np.testing.assert_array_equal(result[1,:1024],2.)

    def test_duplicate_native_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity table'):join_native_cnn(np.zeros((2,1024),np.float32),['a'*64]*2,[],{})

    def test_missing_parent_is_not_replaced_by_a_local_row_number(self):
        with self.assertRaisesRegex(ValueError,'Missing linked'):join_native_cnn(np.zeros((1,1024),np.float32),['a'*64],[dict(input_sha256='a'*64,parent_224_input_sha256='b'*64)],{})
