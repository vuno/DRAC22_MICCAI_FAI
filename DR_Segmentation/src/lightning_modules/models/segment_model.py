from torch import nn

from .unet import UNet
from .u2net import U2NET_full, U2NET_lite
from .sa_unet import SA_UNet
from .hr_ocr_net import get_hr_ocr_model

class SegmentModel(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        
        if backbone == 'unet':
            self.network = UNet(3, num_classes, False)
        elif backbone == 'u2net_full':
            self.network = U2NET_full(num_classes)
        elif backbone == 'u2net_lite':
            self.network = U2NET_lite(num_classes)
        elif backbone == 'sa_unet':
            self.network = SA_UNet(num_classes=num_classes)
        elif backbone == 'hr_net':
            self.network = get_hr_ocr_model()
        else:
            raise NotImplementedError
    
    def forward(self, x):
        return self.network(x)