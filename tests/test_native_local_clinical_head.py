from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from joint_clinical_head import JointClinicalHead
    from native_local_clinical_head import NativeLocalClinicalHead


@unittest.skipUnless(READY,'Optional training Torch environment required')
class NativeLocalClinicalHeadTests(unittest.TestCase):
    def test_zero_local_residual_preserves_every_warm_class_logit(self):
        torch.manual_seed(2175);warm=JointClinicalHead().eval();model=NativeLocalClinicalHead(warm.state_dict()).eval();value=torch.randn(2,1792);local=torch.randn(2,36,1024)
        torch.testing.assert_close(model(value,local),warm(value),rtol=0,atol=0)

    def test_wrong_local_grid_is_not_silently_reinterpreted(self):
        model=NativeLocalClinicalHead(JointClinicalHead().state_dict())
        with self.assertRaisesRegex(ValueError,'6x6'):model(torch.zeros(2,1792),torch.zeros(2,49,1024))
