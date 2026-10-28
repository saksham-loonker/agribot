import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
from joint_clinical_head import JointClinicalHead


class JointClinicalHeadTests(unittest.TestCase):
    def test_zero_residual_preserves_warm_clinical_logits_for_bags(self):
        torch.manual_seed(2158);warm=torch.nn.Linear(1024,9);head=JointClinicalHead();head.load_warm_head(warm.state_dict());head.eval()
        value=torch.randn(3,2,4,1792)
        torch.testing.assert_close(head(value),warm(value[...,:1024]),rtol=0,atol=0)

    def test_wrong_encoder_dimension_is_rejected(self):
        with self.assertRaises(ValueError):JointClinicalHead()(torch.zeros(2,1408))
