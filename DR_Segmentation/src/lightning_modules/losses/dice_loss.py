import torch
from torch import Tensor
import numpy as np
from torch import Tensor, einsum
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs) # sigmoid를 통과한 출력이면 주석처리
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.mul(input.reshape(-1), target.reshape(-1)).sum()
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    total_positive = target.sum()
    for channel in range(input.shape[1]):
        w = 1. - (target[:, channel, ...].sum() / total_positive)
        w = w.item()
        if np.isnan(w):
            w = 0.
        
        #print(f"CLASS {channel} : {w}")
        dice += w * dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def calc_dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


# class GeneralizedDice():
#     def __init__(self, num_classes=3):
#         self.num_classes = num_classes

#     def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
#         total_loss = 0
#         probs = torch.sigmoid(probs)
#         for c in range(self.num_classes):
#             #assert simplex(probs) and simplex(target)

#             # probs = torch.softmax(probs, dim=1)
#             c_probs = probs[:, c, ...]
#             c_target = target[:, c, ...]

#             # pc = probs[:, self.idc, ...].type(torch.float32)
#             # tc = target[:, self.idc, ...].type(torch.float32)
#             pc, tc = torch.unsqueeze(c_probs, 1).type(torch.float32), torch.unsqueeze(c_target, 1).type(torch.float32)

#             w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
#             intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
#             union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

#             divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

#             loss = divided.mean()
#             total_loss += loss
#         return total_loss

# class GeneralizedDice(nn.Module):
#     def __init__(self, **kwargs):
#         super(GeneralizedDice, self).__init__()
#         # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
#         self.idc: List[int] = [0, 1, 2] #kwargs["idc"]
#         #print(f"Initialized {self.__class__.__name__} with {kwargs}")

#     def forward(self, probs: Tensor, target: Tensor) -> Tensor:
#         probs = torch.sigmoid(probs)
#         batch_size = probs.size(0)

#         pc = probs.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
#         tc = target.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
#         #torch.sum(tc.view(batch_size, -1), dim=1)

#         #w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
#         w: Tensor = 1 / (einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) #.unsqueeze(0)
#         # TODO: torch.log 잘못구현
#         #w = torch.log(w + 1e-10) 
#         #print(w)
#         #print(w)
#         #print(w)
#         intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
#         #print(intersection)
#         #print("----------")
#         #print(intersection)
#         #print(np.unique(tc.detach().cpu().numpy()))
#         union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))
#         #print(union)

#         dice = 1 - 2 * intersection / (union + 1e-10)
#         #dice = dice.view(-1)
#         mask = torch.sum(tc.view(batch_size, 3, -1), dim=2) > 0
#         mask = mask.to(torch.float32)

#         loss = dice * mask
#         loss = loss.sum() / (mask.sum() + 1e-10)

#         return loss 

class GeneralizedDice(nn.Module):
    def __init__(self, **kwargs):
        super(GeneralizedDice, self).__init__()
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [0, 1, 2] #kwargs["idc"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def forward(self, probs: Tensor, target: Tensor) -> Tensor:
        probs = torch.sigmoid(probs)
        batch_size = probs.size(0)

        pc = torch.reshape(probs, (batch_size, 3, 1024 * 1024)) #.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
        tc = torch.reshape(target, (batch_size, 3, 1024 * 1024))#.to(torch.float32)#[:, self.idc, ...].type(torch.float32)
        #torch.sum(tc.view(batch_size, -1), dim=1)
        w = 1. / (torch.sum(tc, dim=-1) + 1e-3)
        #w = w.unsqueeze(-1)

        #w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
        #w: Tensor = 1 / (einsum("bkwh->bk", tc) + 1e-3) #.unsqueeze(0)
        # TODO: torch.log 잘못구현
        #w = torch.log(w + 1e-10) 
        #print(w)
        #print(w)
        #print(w)
        #print(pc.size())
        #print(tc.size())
        
        intersection = w * torch.sum(pc * tc, dim=2)
        #intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
        #print(intersection)
        #print("----------")
        #print(intersection)
        #print(np.unique(tc.detach().cpu().numpy()))
        union = w * (torch.sum(pc, dim=2) + torch.sum(tc, dim=2))
        #union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))
        #print(union)

        dice = 1. - 2. * intersection / (union + 1e-3)
        #print("DICE", dice)
        #dice = dice.view(-1)
        mask = torch.sum(tc, dim=2) > 0
        #mask = mask.unsqueeze(-1)
        #mask = mask.to(torch.float32)

        loss = dice * mask
        loss = loss.sum() / (mask.sum() + 1e-3)

        return loss 

        # #print(mask.size())
        # #exit()
        # loss = 0
        # n_count = 0
        # for _ in range(len(dice)):
        #     if mask[_] != 0:
        #         loss += dice[_]
        #         n_count += 1
        #     #else:
        #     #    print("TEST")
        # #print(w)
        # #print(dice)
        # #print("----------")

        # #divided: Tensor = 1 - 2 * (einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        # #loss = divided.mean()
        # #print(loss)
        # #exit()
        # if n_count > 0:
        #     return loss / (n_count + 1e-10)
        # else:
        #     return 0. * dice.mean() #torch.FloatTensor([0.])