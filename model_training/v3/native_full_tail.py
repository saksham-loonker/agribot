"""Full final-block adaptation with an untouched, independently cached prefix."""
import copy
from torch import nn
from native_lora_tail import NativeLoraTail


class NativeFullTail(NativeLoraTail):
    def __init__(self,backbone,head):
        nn.Module.__init__(self)
        if backbone.global_pool!='token':raise ValueError('Native tail requires CLS pooling')
        self.dimension=backbone.num_features
        self.blocks=copy.deepcopy(backbone.blocks[-2:]).requires_grad_(True)
        self.norm=copy.deepcopy(backbone.norm).requires_grad_(True)
        self.fc_norm=copy.deepcopy(backbone.fc_norm).requires_grad_(True)
        self.head=copy.deepcopy(head).requires_grad_(True)
