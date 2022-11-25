import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics import Dice
from pytorch_lightning import LightningModule
import os
import cv2
import numpy as np
from .models import get_drac_model
from .losses.dice_loss import calc_dice_loss, GeneralizedDice, DiceLoss
from .losses.focal_loss import FocalLossMultiLabel


class Task1LM(LightningModule):
    def __init__(self, lr, backbone, num_classes=3, target=2, epochs=500):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = get_drac_model('segment', backbone, num_classes)
        
        self.criterion = GeneralizedDice()
        self.metrics = nn.ModuleDict({
            f'{split}_{idx_class+1}': Dice()
            for idx_class in range(3)
            for split in ['metric_train', 'metric_val', 'metric_test']
        })
        self.target = int(target)
        if self.target == 2:
            self.aux_criterion = FocalLossMultiLabel()
        else:
            self.aux_criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, split='train', batch_idx=None):
        imgs, labels = batch
        batch_size = imgs.size(0)
        ds = self.model(imgs)
        if self.target == 2:
            focal_loss_1 = self.aux_criterion(ds[0][:, 0], labels[:, 0]) 
            focal_loss_2 = self.aux_criterion(ds[0][:, 1], labels[:, 1]) 
            focal_loss_3 = self.aux_criterion(ds[0][:, 2], labels[:, 2])
            dice_loss = self.criterion(ds[0], labels)
            loss = 0.5 * (focal_loss_1 + focal_loss_2 + focal_loss_3) + dice_loss
        else:
            loss = 0.5 * self.aux_criterion(ds[0], labels) + self.criterion(ds[0], labels)
        self.update_metrics(torch.sigmoid(ds[0]), labels.long(), split)
        self.log(f'{split}/loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, split='train', batch_idx=batch_idx)
    
    def training_epoch_end(self, outputs):
        self.compute_metrics(split='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, split='val', batch_idx=batch_idx)
    
    def validation_epoch_end(self, outputs):
        self.compute_metrics(split='val')
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,200,300,400])
        return [optimizer]#, [scheduler]

    def update_metrics(self, preds, labels, split):
        for idx_class in range(3):
            self.metrics[f'metric_{split}_{idx_class+1}'].update(preds[:,idx_class,:,:], labels[:,idx_class,:,:])
    
    def compute_metrics(self, split):
        score_sum = 0
        for idx_class in range(3):
            score = self.metrics[f'metric_{split}_{idx_class+1}'].compute()
            self.metrics[f'metric_{split}_{idx_class+1}'].reset()
            self.log(f'{split}/Dice_{idx_class+1}', score)
            score_sum += score
            if idx_class + 1 == self.target:
                print(f'{split}/Dice_{self.target}', score)


        self.log(f'{split}/Dice_avg', score_sum/3)
        #print(f'{split}/Dice_avg', score_sum/3)

# class Task1LM(LightningModule):
#     def __init__(self, lr, backbone, num_classes=3, target=2):
#         super().__init__()
#         self.save_hyperparameters()
        
#         self.model = get_drac_model('segment', backbone, num_classes)
        
#         self.criterion = GeneralizedDice() 
#         self.bce_criterion = nn.BCEWithLogitsLoss()
#         self.metrics = nn.ModuleDict({
#             f'{split}_{idx_class+1}': Dice()
#             for idx_class in range(3)
#             for split in ['metric_train', 'metric_val', 'metric_test']
#         })

#         self.target = int(target)
#         if not os.path.exists(f"/home/kimjaeyoung/DRAC/seg_vis2/{self.target}"):
#             os.makedirs(f"/home/kimjaeyoung/DRAC/seg_vis2/{self.target}")
    
#     def forward(self, x):
#         return self.model(x)
    
#     def step(self, batch, split='train', batch_idx=None):
#         imgs, labels = batch
#         batch_size = imgs.size(0)

#         if self.hparams.backbone in ['u2net_full', 'u2net_lite', 'hr_net']:
#             ds = self.model(imgs)
#             # if split == 'val':
#             #     origin_img = imgs.squeeze().detach().cpu().numpy()
#             #     origin_img = np.uint8(np.transpose(origin_img, [1,2,0]) * 255.)
#             #     #origin_img = origin_img[..., 0]
#             #     pred = torch.sigmoid(ds[0]) >= 0.5
#             #     pred = pred.to(torch.float32).squeeze().detach().cpu().numpy()
#             #     gt = labels.to(torch.float32).squeeze().detach().cpu().numpy()
#             #     pred_img = np.copy(origin_img)
#             #     gt_img = np.copy(origin_img)
#             #     pred_img[np.where(pred[0, ...] > 0)] = (0, 0, 255)
#             #     pred_img[np.where(pred[1, ...] > 0)] = (0, 255, 255)
#             #     pred_img[np.where(pred[2, ...] > 0)] = (255, 0, 0)
#             #     gt_img[np.where(gt[0, ...] > 0)] = (0, 0, 255)
#             #     gt_img[np.where(gt[1, ...] > 0)] = (0, 255, 255)
#             #     gt_img[np.where(gt[2, ...] > 0)] = (255, 0, 0)
#             #     result = np.concatenate([origin_img, pred_img, gt_img], axis=1)
#             #     cv2.imwrite(f"/home/kimjaeyoung/DRAC/seg_vis2/{self.target}/{batch_idx}.png", result[..., ::-1])
#             #loss = 0.5 * sum(self.bce_criterion(p, labels) for p in ds) + sum(self.criterion(d, labels) for d in ds)
#             loss = 0.5 * self.bce_criterion(ds[0], labels) + self.criterion(ds[0], labels)
#             #loss = 0.5 * sum(self.bce_criterion(p, labels) for p in ds[:2]) + sum(self.criterion(d, labels) for d in ds[:2])
#             self.update_metrics(torch.sigmoid(ds[0]), labels.long(), split)
#             #self.update_metrics(metrics_pred.to(torch.float32).view(batch_size, -1), labels.long().view(batch_size, -1), split)
        
#         else:
#             preds = self.model(imgs)
            
#             loss = 0.5 * self.criterion(preds, labels) + calc_dice_loss(torch.sigmoid(preds), labels.float(), multiclass=True)
#             self.update_metrics(torch.sigmoid(preds), labels.long(), split)
            
#         self.log(f'{split}/loss', loss)
#         return loss
    
#     def training_step(self, batch, batch_idx):
#         return self.step(batch, split='train', batch_idx=batch_idx)
    
#     def training_epoch_end(self, outputs):
#         self.compute_metrics(split='train')
    
#     def validation_step(self, batch, batch_idx):
#         self.step(batch, split='val', batch_idx=batch_idx)
    
#     def validation_epoch_end(self, outputs):
#         self.compute_metrics(split='val')
    
#     def configure_optimizers(self):
#         # optimizer = optim.RMSprop(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-8, momentum=0.9)
#         # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
#         # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/Dice'}
        
#         optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,200,300,400])
#         return [optimizer], [scheduler]

#     def update_metrics(self, preds, labels, split):
#         for idx_class in range(3):
#             self.metrics[f'metric_{split}_{idx_class+1}'].update(preds[:,idx_class,:,:], labels[:,idx_class,:,:])
    
#     def compute_metrics(self, split):
#         score_sum = 0
#         for idx_class in range(3):
#             score = self.metrics[f'metric_{split}_{idx_class+1}'].compute()
#             self.metrics[f'metric_{split}_{idx_class+1}'].reset()
#             self.log(f'{split}/Dice_{idx_class+1}', score)
#             score_sum += score

#         self.log(f'{split}/Dice_avg', score_sum/3)
#         print(f'{split}/Dice_avg', score_sum/3)


class Task1LM2(LightningModule):
    def __init__(self, lr, backbone, num_classes=3, target=2):
        super().__init__()
        self.save_hyperparameters()
        self.target = int(target)
        num_classes = 1
        self.model = get_drac_model('segment', backbone, num_classes)
        if int(target) == 2:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
            self.dice_criterion = DiceLoss() 
        #self.bce_criterion = nn.BCEWithLogitsLoss()

        self.metrics = nn.ModuleDict({
            f'{split}_{target}': Dice()
            for split in ['metric_train', 'metric_val', 'metric_test']
        })

    
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, split='train', batch_idx=None):
        imgs, labels = batch
        batch_size = imgs.size(0)

        if self.hparams.backbone in ['u2net_full', 'u2net_lite']:
            if split == 'train':
                imgs = imgs.reshape(-1, 3, 256, 256)
                labels = labels.reshape(-1, 1, 256, 256)
                ds = self.model(imgs)
                if self.target == 2:
                    loss = self.criterion(ds[0], labels)
                else:
                    loss = 0.1 * self.criterion(ds[0], labels) + self.dice_criterion(ds[0], labels)

                self.update_metrics(torch.sigmoid(ds[0]), labels.long(), split)
            else:
                rows = np.int(np.ceil(1.0 * (1024 - 256) / 128)) + 1
                cols = np.int(np.ceil(1.0 * (1024 - 256) / 128)) + 1
                preds = torch.zeros([1, 1, 1024, 1024]).cuda()
                count = torch.zeros([1, 1, 1024, 1024]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * 128
                        w0 = c * 128
                        h1 = min(h0 + 256, 1024)
                        w1 = min(w0 + 256, 1024)
                        h0 = max(int(h1 - 256), 0)
                        w0 = max(int(w1 - 256), 0)
                        crop_img = imgs[:, :, h0:h1, w0:w1]
                        pred = self.model(crop_img.cuda())
                        pred = torch.sigmoid(pred[0])
                        preds[:,:, h0:h1, w0:w1] += pred[:, :, 0:h1-h0, 0:w1-w0]
                        count[:,:, h0:h1, w0:w1] += 1
                preds = preds / count
                #preds = preds[:,:,:height,:width]
                #ds = self.model(imgs)
                loss = self.criterion(preds, labels)
                self.update_metrics(preds, labels.long(), split)

        else:
            preds = self.model(imgs)
            
            loss = 0.5 * self.criterion(preds, labels) + calc_dice_loss(torch.sigmoid(preds), labels.float(), multiclass=True)
            self.update_metrics(torch.sigmoid(preds), labels.long(), split)
            
        self.log(f'{split}/loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, split='train', batch_idx=batch_idx)
    
    def training_epoch_end(self, outputs):
        self.compute_metrics(split='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, split='val', batch_idx=batch_idx)
    
    def validation_epoch_end(self, outputs):
        self.compute_metrics(split='val')
    
    def configure_optimizers(self):
        # optimizer = optim.RMSprop(self.model.parameters(), lr=self.hparams.lr, weight_decay=1e-8, momentum=0.9)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/Dice'}
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(0.6, 0.999))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100])
        return [optimizer], [scheduler]

    def update_metrics(self, preds, labels, split):
        for idx_class in range(1):
            self.metrics[f'metric_{split}_{self.target}'].update(preds[:,0,:,:], labels[:,0,:,:])
    
    def compute_metrics(self, split):
        score = self.metrics[f'metric_{split}_{self.target}'].compute()
        self.metrics[f'metric_{split}_{self.target}'].reset()
        self.log(f'{split}/Dice_{self.target}', score)
        print(f'{split}/Dice_{self.target}', score)
            
