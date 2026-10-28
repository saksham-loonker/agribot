from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import math,torch
    from torch.nn import functional as F
    from leaf_feature_expert import LeafFeatureHead
    from local_damage_heads import LocalRecoveryHead


@unittest.skipUnless(READY,'Optional training Torch environment required')
class LocalDamageHeadTests(unittest.TestCase):
    def test_zero_residual_preserves_both_initial_classifier_scores(self):
        torch.manual_seed(121);base=LeafFeatureHead('mlp',1408).eval();tokens=torch.randn(2,257,384);cnn=torch.randn(2,1024)
        expected=base(torch.cat([F.normalize(tokens[:,0],dim=1)*math.sqrt(384),cnn],1))*.8+.1
        for kind in ('gated_attention_mil','topk_damage_mil'):
            model=LocalRecoveryHead(kind,base.state_dict(),[.8,.1]).eval()
            torch.testing.assert_close(model(tokens,cnn),expected)

    def test_learned_local_pooling_is_invariant_to_patch_order(self):
        torch.manual_seed(2121);base=LeafFeatureHead('mlp',1408);tokens=torch.randn(2,257,384);cnn=torch.randn(2,1024)
        permuted=torch.cat([tokens[:,:1],tokens[:,1:][:,torch.randperm(256)]],1)
        for kind in ('gated_attention_mil','topk_damage_mil'):
            model=LocalRecoveryHead(kind,base.state_dict(),[1.,0.]).eval()
            with torch.no_grad():
                if kind=='gated_attention_mil':model.local_projection.weight.normal_(0,.02)
                else:model.damage[-1].weight.normal_(0,.02)
            torch.testing.assert_close(model(tokens,cnn),model(permuted,cnn),atol=1e-5,rtol=1e-5)


if __name__=='__main__':unittest.main()
