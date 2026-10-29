from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from torch import nn
    from joint_clinical_head import JointClinicalHead
    from native_adapted_patch_tail import NativePatchClinicalHead,NativeAdaptedPatchTail


@unittest.skipUnless(READY,'Optional training Torch environment required')
class NativeAdaptedPatchTailTests(unittest.TestCase):
    def test_zero_local_residual_preserves_global_warm_verdict_logits(self):
        torch.manual_seed(191);warm=JointClinicalHead().eval();head=NativePatchClinicalHead(warm.state_dict(),patches=4).eval()
        tokens=torch.randn(2,5,1024);texture=torch.randn(2,768)
        expected=warm(torch.cat((tokens[:,0],texture),1));actual=head(torch.cat((tokens.flatten(1),texture),1))
        self.assertTrue(torch.equal(actual,expected))

    def test_image_loss_reaches_local_attention_without_patch_targets(self):
        torch.manual_seed(191);warm=JointClinicalHead();head=NativePatchClinicalHead(warm.state_dict(),patches=4)
        nn.init.normal_(head.local_class.weight,std=.01)
        features=torch.randn(2,5*1024+768,requires_grad=True);head(features).square().mean().backward()
        self.assertGreater(float(head.gate_weight.weight.grad.abs().sum()),0)
        self.assertGreater(float(features.grad[:,1024:5*1024].abs().sum()),0)

    def test_incompatible_token_count_is_rejected(self):
        head=NativePatchClinicalHead(JointClinicalHead().state_dict(),patches=4)
        with self.assertRaises(ValueError):head(torch.zeros(1,1024+768))
