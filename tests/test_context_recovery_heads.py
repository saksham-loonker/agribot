from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from leaf_feature_expert import LeafFeatureHead
    from context_recovery_heads import ContextRecoveryHead


@unittest.skipUnless(READY,'Optional training Torch environment required')
class ContextRecoveryHeadTests(unittest.TestCase):
    def test_initial_context_cannot_change_existing_roi_scores(self):
        torch.manual_seed(122);base=LeafFeatureHead('mlp',1408).eval();leaves=torch.randn(2,2,3,1408);whole=torch.randn(2,768)
        expected=(base(leaves.reshape(-1,1408))*.9-.2).reshape(2,2,3)
        for kind in ('additive_context','bilinear_context'):
            model=ContextRecoveryHead(kind,base.state_dict(),[.9,-.2]).eval()
            torch.testing.assert_close(model(leaves,whole),expected)
            torch.testing.assert_close(model(leaves,whole*100),expected)


if __name__=='__main__':unittest.main()
