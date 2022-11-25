import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchmetrics import AUROC, CohenKappa, MetricCollection
from pytorch_lightning import LightningModule
from .models import get_drac_model
from .losses.focal_loss import FocalLossAdaptive, GeneralizedLabelSmooth
from .losses.qwk_loss import QWKLoss
from .losses.qwk_loss2 import QWKLoss2
#from cosine_annealing_warmup import CosineAnnealingWarmupRestarts


class DRACLM(LightningModule):
    def __init__(self, task, backbone, lr, optimizer, loss, beta1, decay_step, mix_up, num_classes=3):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_drac_model(task, backbone, num_classes)
        self.best_score = -100000
        
        if loss == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        elif loss == 'mse':
            self.criterion = nn.MSELoss()
        elif loss == 'smoothl1loss':
            self.criterion = nn.SmoothL1Loss()
        elif loss == 'ls':
            self.criterion = GeneralizedLabelSmooth(smooth_rate=-0.8)
        elif loss == 'focal':
            self.criterion = FocalLossAdaptive()
        elif loss == 'qwk':
            self.criterion = QWKLoss()
        elif loss == 'qwk2':
            self.criterion = QWKLoss2()
        else:
            raise NotImplementedError
        
        self.metrics = nn.ModuleDict({
            split: MetricCollection([
                CohenKappa(num_classes=3, weights='quadratic'),
                AUROC(num_classes=3),
            ])
            for split in ['metric_train', 'metric_val', 'metric_test']
        })
    
    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, split='train'):
        imgs, labels = batch
        preds = self.model(imgs)
        
        if self.hparams.loss in ['mse', 'smoothl1loss']:
            loss = self.criterion(preds.squeeze(dim=1), labels.float())
            #print(loss)
            metric_preds = torch.round(torch.clamp(preds, min=0., max=2.)).view(-1).long()
            metric_preds = F.one_hot(metric_preds, num_classes=3).float()
            self.update_metrics(metric_preds, labels, split=split)
        else:
            loss = self.criterion(preds, labels)
            if self.hparams.loss == 'qwk' or self.hparams.loss == 'qwk2':
                self.update_metrics(preds.softmax(dim=-1), labels, split=split)
            else:
                self.update_metrics(preds.log_softmax(dim=-1), labels, split=split)
        
        self.log(f'{split}/loss', loss)
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, split='train')
    
    def training_epoch_end(self, outputs):
        self.compute_metrics(split='train')
    
    def validation_step(self, batch, batch_idx):
        self.step(batch, split='val')
    
    def validation_epoch_end(self, outputs):
        self.compute_metrics(split='val')
    
    def configure_optimizers(self):
        beta1 = self.hparams.beta1
        
        if self.hparams.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.hparams.lr, momentum=0.9)
        elif self.hparams.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(beta1, 0.999))
        elif self.hparams.optimizer == 'radam':
            optimizer = optim.RAdam(self.model.parameters(), lr=self.hparams.lr, betas=(beta1, 0.999))
        elif self.hparams.optimizer == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(beta1, 0.999))
        else:
            raise NotImplementedError
        
        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        
        return [optimizer]#, [lr_scheduler]

    def update_metrics(self, preds, labels, split):
        self.metrics[f'metric_{split}'].update(preds, labels)
    
    def compute_metrics(self, split):
        score = self.metrics[f'metric_{split}'].compute()
        self.metrics[f'metric_{split}'].reset()
        
        for k, v in score.items():
            self.log(f'{split}/{k}', v)
            print(f'{split}/{k}', v)
