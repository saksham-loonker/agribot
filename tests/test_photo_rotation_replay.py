from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from photo_rotation_replay import select_photo_rotation


@unittest.skipUnless(READY,'Optional training Torch environment required')
class PhotoRotationReplayTests(unittest.TestCase):
    def test_real_rotation_is_shared_by_both_crops_and_all_leaves(self):
        index=torch.arange(2*4*2*3).reshape(2,4,2,3);mask=torch.zeros_like(index);mask[0,2]=1;mask[1,3]=1
        selected,_,rotation=select_photo_rotation(index,mask,True)
        self.assertEqual(rotation.tolist(),[2,3]);self.assertTrue(torch.equal(selected[0],index[0,2]));self.assertTrue(torch.equal(selected[1],index[1,3]))

    def test_evaluation_always_uses_original_orientation(self):
        index=torch.arange(48).reshape(2,4,2,3);selected,_,rotation=select_photo_rotation(index,torch.ones_like(index),False)
        self.assertEqual(rotation.tolist(),[0,0]);self.assertTrue(torch.equal(selected,index[:,0]))

    def test_missing_real_or_original_evaluation_view_is_rejected(self):
        index=torch.zeros((1,4,2,3),dtype=torch.long)
        with self.assertRaises(ValueError):select_photo_rotation(index,index,True)
        mask=index.clone();mask[:,1]=1
        with self.assertRaises(ValueError):select_photo_rotation(index,mask,False)
