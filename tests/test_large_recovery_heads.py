from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import math,torch
    from torch.nn import functional as F
    from leaf_feature_expert import LeafFeatureHead
    from large_roi_heads import LargeRecoveryHead
    from native_local_damage_heads import NativeLocalRecoveryHead


@unittest.skipUnless(READY,'Optional training Torch environment required')
class LargerRepresentationTests(unittest.TestCase):
    def test_larger_encoder_residual_preserves_original_score_at_initialization(self):
        torch.manual_seed(132);base=LeafFeatureHead('mlp',1408).eval();old=torch.randn(3,1408)
        for kind in ('large_linear','large_mlp'):
            model=LargeRecoveryHead(kind,base.state_dict(),[.8,.1]).eval()
            torch.testing.assert_close(model(torch.cat([old,torch.randn(3,1024)*100],1)),base(old)*.8+.1)

    def test_native_resolution_patch_head_preserves_matching_224_cls_base(self):
        torch.manual_seed(129);base=LeafFeatureHead('mlp',1408).eval();tokens=torch.randn(2,1025,384);cnn=torch.randn(2,1024)
        expected=base(torch.cat([F.normalize(tokens[:,0],dim=1)*math.sqrt(384),cnn],1))
        for kind in ('gated_attention_mil','topk_damage_mil'):
            model=NativeLocalRecoveryHead(kind,base.state_dict(),[1.,0.]).eval()
            torch.testing.assert_close(model(tokens,cnn),expected)
