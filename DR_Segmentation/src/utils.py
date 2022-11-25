import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torchmetrics import Metric
from sklearn.metrics import cohen_kappa_score
from torchmetrics.utilities.data import dim_zero_cat



class QuadraticCohenKappa(Metric):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = np.arange(num_classes).tolist()
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        #preds = preds.detach().cpu().numpy().flatten()
        #target = target.detach().cpu().numpy().flatten()
        assert preds.shape == target.shape
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        #try:
        try:
            true = dim_zero_cat(self.target)
            pred = dim_zero_cat(self.preds)
            true = true.detach().cpu().numpy().flatten()
            pred = pred.detach().cpu().numpy().flatten()
            score = cohen_kappa_score(true, pred, labels=self.num_classes, weights='quadratic')
        except:
            score = 0.0
        return score


def set_seed(random_seed=42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def construct_kfold_dataset(conf):
    if conf.task == 'quality':
        label_path = glob.glob(os.path.join(conf.data_dir, "B. Image Quality Assessment", "2. Groundtruths", "*.csv"))[0]
        label_col_name = 'image quality level'
    elif conf.task == 'grading':
        label_path = glob.glob(os.path.join(conf.data_dir, "C. Diabetic Retinopathy Grading", "2. Groundtruths", "*.csv"))[0]
        label_col_name = 'DR grade'
    else:
        raise NotImplementedError
    
    save_path = label_path.replace('Labels.csv', 'Labels_Fold.csv')
    if os.path.isfile(save_path):
        return
    
    df = pd.read_csv(label_path)
    x_total = df['image name'].values
    y_total = df[label_col_name].values

    # 8 : 1 : 1
    sss = StratifiedShuffleSplit(n_splits=conf.num_kfold, test_size=0.2)
    fold_idx = 0
    for train_idx, val_idx in sss.split(x_total, y_total):
        split = []
        for idx in range(len(df)):
            if idx in train_idx:
                split.append("train")
            else:
                split.append("val")
        df[f'fold_{fold_idx}'] = split
        fold_idx += 1
    df.to_csv(save_path, index=False)


