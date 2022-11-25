import torch
from torch import nn
from torchvision import models

# class GradingModel(torch.nn.Module):
#     def __init__(self, backbone='efficientnet_v2_m', num_classes=3, dropout=0.1, stochastic_depth=0.2):
#         super().__init__()
#         self.backbone = get_efficientnet_v2(backbone, pretrained=True, num_classes=num_classes, dropout=dropout, stochastic_depth=stochastic_depth)
#     def forward(self, x):
#         return self.backbone(x)

class GradingModel(torch.nn.Module):
    def __init__(self, backbone='efficientnet-b4', num_classes=3):
        super().__init__()
        
        self.network = getattr(models, backbone)(pretrained=True)
        dropout_p = self.network.classifier[0].p
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
        )
        
    def forward(self, x):
        return self.network(x)