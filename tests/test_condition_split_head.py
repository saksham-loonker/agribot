from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from condition_split_head import expanded_healthy_margin,split_other_head,split_condition_loss
    from partial_dino_healthy import healthy_margin


@unittest.skipUnless(READY,'Optional training Torch environment required')
class ConditionSplitTests(unittest.TestCase):
    def test_expansion_conserves_original_healthy_margin(self):
        torch.manual_seed(135);original=torch.nn.Linear(384,9);x=torch.randn(11,384)
        for n in (1,4,15):
            torch.testing.assert_close(expanded_healthy_margin(split_other_head(original,n)(x)),healthy_margin(original(x)),atol=2e-6,rtol=2e-6)

    def test_partial_other_label_accepts_subtype_but_excludes_supported_disease(self):
        subtype=torch.full((1,2,11),-5.);subtype[:,:,10]=5.
        supported=torch.full((1,2,11),-5.);supported[:,:,2]=5.
        self.assertLess(float(split_condition_loss(subtype,torch.tensor([-2]))),float(split_condition_loss(supported,torch.tensor([-2]))))
        self.assertLess(float(split_condition_loss(supported,torch.tensor([-1]))),.01)
