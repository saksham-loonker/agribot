from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model_training/v3'))
READY=importlib.util.find_spec('torch') is not None
if READY:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from leaf_feature_expert import LeafFeatureHead
    from full_mobile_clinical_expert import FullMobileClinicalExpert

    class TinyCnn(nn.Module):
        def __init__(self):super().__init__();self.projection=nn.Linear(3,1024);self.bn=nn.BatchNorm1d(1024);self.classifier=nn.Linear(1024,2)
        def get_classifier(self):return self.classifier

    class TinyFeatures(nn.Module):
        def __init__(self):
            super().__init__();self.primary=nn.Module();self.primary.backbone=nn.Module();self.primary.backbone.projection=nn.Linear(3,384);self.primary.backbone.head=nn.Linear(384,9);self.primary.cnn=TinyCnn();self.primary.confirmation=None
        def components(self,x):
            x=x.mean((1,2));d=self.primary.backbone.projection(x);c=self.primary.cnn.bn(self.primary.cnn.projection(x));return torch.cat((F.normalize(d,dim=1)*(384**.5),F.normalize(c,dim=1)*(1024**.5)),1),self.primary.backbone.head(d)


@unittest.skipUnless(READY,'Optional training Torch environment required')
class FullMobileClinicalExpertTests(unittest.TestCase):
    def test_original_is_not_mutated_and_both_feature_encoders_receive_gradients(self):
        torch.manual_seed(206);original=TinyFeatures().eval().requires_grad_(False);warm=LeafFeatureHead('mlp',1408).eval();model=FullMobileClinicalExpert(original,warm.state_dict(),[1.,0.]);x=torch.randn(3,8,8,3)
        model.eval();expected=warm(original.components(x)[0]).ravel();self.assertTrue(torch.equal(model(x)[0],expected))
        model.train();running=model.features.primary.cnn.bn.running_mean.clone();binary,clinical=model(x);(binary.square().mean()+clinical.square().mean()).backward()
        self.assertGreater(float(model.features.primary.backbone.projection.weight.grad.abs().sum()),0);self.assertGreater(float(model.features.primary.cnn.projection.weight.grad.abs().sum()),0)
        self.assertIsNone(model.features.primary.backbone.head.weight.grad);self.assertIsNone(model.features.primary.cnn.classifier.weight.grad);self.assertTrue(torch.equal(running,model.features.primary.cnn.bn.running_mean))
        self.assertIsNone(original.primary.backbone.projection.weight.grad)
