import torch
#from efficientnet_pytorch import EfficientNet
from .efficientnetv2.efficientnet_v2 import get_efficientnet_v2


class QualityModel(torch.nn.Module):
    def __init__(self, backbone='efficientnet-b4', num_classes=3, dropout=0.1, stochastic_depth=0.2):
        super().__init__()
        if 'efficientnet_v2' in backbone:
            self.backbone = get_efficientnet_v2(backbone, pretrained=True, num_classes=num_classes, dropout=dropout, stochastic_depth=stochastic_depth)
        else:
            self.backbone = EfficientNet.from_pretrained(backbone, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)