"""Copied CNN image-bag expert must preserve inference and original weights."""
from pathlib import Path
import importlib.util
import sys
import unittest
READY=all(importlib.util.find_spec(name) is not None for name in ("torch","timm"))
if READY:
    import torch
    import timm
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"model_training/v3"))
    from cnn_bag_cache import CopiedCnnBagTail
    from partial_cnn_healthy import prefix_features


@unittest.skipUnless(READY,"Optional training environment required")
class CnnBagTailTests(unittest.TestCase):
    def test_inference_matches_original_margin_and_training_cannot_change_it(self):
        torch.set_num_threads(2);torch.manual_seed(101)
        original=timm.create_model("mobilenetv3_small_100",pretrained=False,num_classes=2).eval()
        copied=CopiedCnnBagTail(original).eval()
        before={key:value.clone() for key,value in original.state_dict().items()}
        x=torch.randn(2,3,224,224)
        with torch.no_grad():
            prefix=prefix_features(original,x)
            logits=original(x)
            expected=logits[:,0]-logits[:,1]
            torch.testing.assert_close(copied(prefix,None),expected,rtol=1e-5,atol=1e-5)
        copied.train()
        optimizer=torch.optim.AdamW(copied.parameters(),lr=.001)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(copied(prefix,None),torch.ones(2))
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(copied(prefix,None).detach(),expected))
        self.assertTrue(all(torch.equal(value,original.state_dict()[key]) for key,value in before.items()))
        self.assertTrue(all(not module.training for module in copied.modules() if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)))


if __name__=="__main__":unittest.main()
