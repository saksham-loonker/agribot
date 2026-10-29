from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from torch import nn
    from native_full_patch_tail import NativeFullPatchTail


@unittest.skipUnless(READY,'Optional training Torch environment required')
class NativeFullPatchTailTests(unittest.TestCase):
    def backbone(self):
        b=nn.Module();b.global_pool='token';b.num_features=4;b.blocks=nn.Sequential(nn.Linear(4,4),nn.GELU(),nn.Linear(4,4));b.norm=nn.LayerNorm(4);b.fc_norm=nn.Identity();return b

    def test_full_tail_updates_with_frozen_input_and_keeps_all_patch_tokens(self):
        torch.manual_seed(199);b=self.backbone();model=NativeFullPatchTail(b,nn.Linear(22,2));prefix=torch.randn(2,5,4)
        model.train();self.assertEqual(model.features(prefix,checkpointed=True).shape,(2,20))
        model(prefix,torch.randn(2,2),checkpointed=True).square().mean().backward()
        self.assertGreater(float(model.blocks[-1].weight.grad.abs().sum()),0);self.assertIsNone(b.blocks[-1].weight.grad)

    def test_nonidentity_final_class_norm_is_rejected(self):
        b=self.backbone();b.fc_norm=nn.LayerNorm(4);model=NativeFullPatchTail(b,nn.Linear(22,2))
        with self.assertRaises(ValueError):model.features(torch.randn(2,5,4))
