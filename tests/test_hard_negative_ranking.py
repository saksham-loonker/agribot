from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from hard_negative_ranking import hard_negative_rank_loss


@unittest.skipUnless(READY,'Optional training Torch environment required')
class HardNegativeRankingTests(unittest.TestCase):
    def test_pulls_worst_positive_view_above_highest_negative_view(self):
        z=torch.tensor([[0.,.5],[1.,.5],[-1.,-2.]],requires_grad=True)
        hard_negative_rank_loss(z,torch.tensor([1.,0.,0.]),maximum_negatives=1).backward()
        self.assertLess(float(z.grad[0,0]),0.);self.assertGreater(float(z.grad[1,0]),0.)
        self.assertEqual(float(z.grad[2].abs().sum()),0.)

    def test_single_class_batch_has_finite_zero_gradient(self):
        z=torch.randn(3,2,requires_grad=True);loss=hard_negative_rank_loss(z,torch.ones(3));loss.backward()
        self.assertEqual(float(loss.detach()),0.);self.assertEqual(float(z.grad.abs().sum()),0.)
