import torch
from torch.nn import functional as F

def quadratic_kappa_coefficient(output, target):
    n_classes = target.shape[-1]
    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum() # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK

def quadratic_kappa_loss(output, target, scale=2.0):
    QWK = quadratic_kappa_coefficient(output, target)
    loss = -torch.log(torch.sigmoid(scale * QWK))
    return loss

class QWKLoss2(torch.nn.Module):
    def __init__(self, scale=2.0, n_classes=3):
        super().__init__()
        self.scale = scale
        self.n_classes = n_classes

    def forward(self, output, target):
        # Keep trace of output dtype for half precision training
        target = F.one_hot(target.squeeze(), num_classes=self.n_classes).to(target.device).type(output.dtype)
        output = torch.softmax(output, dim=1)
        return quadratic_kappa_loss(output, target, self.scale)