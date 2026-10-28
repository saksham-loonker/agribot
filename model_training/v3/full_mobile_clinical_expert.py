"""Copied full mobile feature encoders, with fixed original clinical classifiers."""
import copy
from torch import nn
from leaf_feature_expert import LeafFeatureHead


class FullMobileClinicalExpert(nn.Module):
    def __init__(self,original_features,warm_head,calibration):
        super().__init__();self.features=copy.deepcopy(original_features).requires_grad_(False)
        self.features.primary.backbone.requires_grad_(True);self.features.primary.backbone.head.requires_grad_(False)
        self.features.primary.cnn.requires_grad_(True);self.features.primary.cnn.get_classifier().requires_grad_(False)
        self.head=LeafFeatureHead('mlp',1408);self.head.load_state_dict(warm_head,strict=True)
        self.scale=float(calibration[0]);self.bias=float(calibration[1])

    def train(self,mode=True):
        super().train(mode)
        for module in self.features.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        self.features.primary.backbone.head.eval();self.features.primary.cnn.get_classifier().eval()
        confirmation=getattr(self.features.primary,'confirmation',None)
        if confirmation is not None:confirmation.eval()
        return self

    def forward(self,nhwc):
        features,reference=self.features.components(nhwc)
        return self.head(features).ravel()*self.scale+self.bias,reference
