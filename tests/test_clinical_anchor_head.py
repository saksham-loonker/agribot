from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from clinical_anchor_head import image_condition_loss


@unittest.skipUnless(READY,'Optional training Torch environment required')
class ClinicalAnchorLossTests(unittest.TestCase):
    def test_coarse_disease_does_not_force_an_invented_disease_class(self):
        a=torch.tensor([[[4.,-4.,-4.],[4.,-4.,-4.]]])
        b=torch.tensor([[[-4.,-4.,4.],[-4.,-4.,4.]]])
        torch.testing.assert_close(image_condition_loss(a,torch.tensor([-1]),1),image_condition_loss(b,torch.tensor([-1]),1))
        healthy=torch.tensor([[[-4.,4.,-4.],[-4.,4.,-4.]]])
        self.assertGreater(float(image_condition_loss(healthy,torch.tensor([-1]),1)),float(image_condition_loss(a,torch.tensor([-1]),1)))

    def test_exact_condition_uses_both_actual_crop_views(self):
        logits=torch.tensor([[[3.,0.,0.],[0.,0.,3.]]])
        expected=(torch.nn.functional.cross_entropy(logits[:,0],torch.tensor([0]))+torch.nn.functional.cross_entropy(logits[:,1],torch.tensor([0])))/2
        torch.testing.assert_close(image_condition_loss(logits,torch.tensor([0]),1),expected)
