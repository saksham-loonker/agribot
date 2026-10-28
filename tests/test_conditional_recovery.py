from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from conditional_recovery_policy import recover_only
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from conditional_recovery_losses import ReverseGradient,cross_source_contrastive


class RecoveryPolicyTests(unittest.TestCase):
    def test_healthy_verdict_and_evidence_are_exactly_preserved(self):
        retained=dict(app_replay=dict(kind='HEALTHY',confidence=.83,leaves=3,frames=7))
        self.assertIs(recover_only({},retained,{},0.,{}),retained)

    def test_named_disease_cannot_be_replaced_by_high_healthy_probability(self):
        retained=dict(app_replay=dict(kind='DISEASE',label='Leaf Miner',confidence=.91))
        self.assertIs(recover_only({},retained,{},1.,{}),retained)


@unittest.skipUnless(READY,'Optional training Torch environment required')
class RecoveryLossTests(unittest.TestCase):
    def test_gradient_reversal_reverses_feature_gradient(self):
        x=torch.tensor([1.,-2.],requires_grad=True)
        ReverseGradient.apply(x).square().sum().backward()
        torch.testing.assert_close(x.grad,torch.tensor([-2.,4.]))

    def test_no_cross_source_positive_has_finite_zero_loss(self):
        h=torch.randn(4,8,requires_grad=True)
        loss=cross_source_contrastive(h,torch.tensor([0,0,1,1]),torch.zeros(4,dtype=torch.long),torch.arange(4))
        self.assertEqual(float(loss.detach()),0.)
        loss.backward();self.assertTrue(torch.isfinite(h.grad).all())

    def test_cross_source_same_class_alignment_reduces_loss(self):
        labels=torch.tensor([0,0,1,1]);sources=torch.tensor([0,1,0,1]);ids=torch.arange(4)
        correct=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        wrong=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]])
        self.assertLess(float(cross_source_contrastive(correct,labels,sources,ids)),float(cross_source_contrastive(wrong,labels,sources,ids)))


if __name__=='__main__':unittest.main()
