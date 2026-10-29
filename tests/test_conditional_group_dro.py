import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
try:
    import torch
    from conditional_group_dro import ConditionalGroupDRO
except ImportError:torch=None

@unittest.skipIf(torch is None,'PyTorch unavailable')
class GroupDROTests(unittest.TestCase):
    def test_harder_condition_gets_more_mass_without_dropping_other_class(self):
        dro=ConditionalGroupDRO([0,0,1],.1)
        losses=torch.tensor([1.,4.,2.],requires_grad=True)
        loss=dro(losses,torch.tensor([0,1,2]));loss.backward()
        self.assertGreater(float(dro.log_weights[1]),float(dro.log_weights[0]))
        self.assertAlmostEqual(float(losses.grad[2]),.5,places=6)
        self.assertAlmostEqual(float(losses.grad[:2].sum()),.5,places=6)

    def test_evaluation_does_not_update_and_state_roundtrips(self):
        dro=ConditionalGroupDRO([0,0,1]);dro(torch.tensor([1.,2.,3.]),torch.tensor([0,1,2]))
        clone=ConditionalGroupDRO([0,0,1]);clone.load_state_dict(dro.state_dict());clone.eval()
        before=clone.log_weights.clone();clone(torch.tensor([5.,1.]),torch.tensor([0,2]))
        self.assertTrue(torch.equal(before,clone.log_weights))
        self.assertTrue(torch.equal(before,dro.log_weights))

    def test_unknown_groups_cannot_update_state(self):
        dro=ConditionalGroupDRO([0,1]);before=dro.log_weights.clone()
        with self.assertRaises(ValueError):dro(torch.tensor([1.]),torch.tensor([-1]))
        self.assertTrue(torch.equal(before,dro.log_weights))
