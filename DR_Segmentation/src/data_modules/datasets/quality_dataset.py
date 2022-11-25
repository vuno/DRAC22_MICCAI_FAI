import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class QualityDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, fold_idx=0):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform
        
        self.task_tag = 'B. Image Quality Assessment'
        df = pd.read_csv(os.path.join(data_dir, self.task_tag, '2. Groundtruths', 'a. DRAC2022_ Image Quality Assessment_Training Labels.csv'))
        df = df[df[f'fold_{fold_idx}']==split]
        df.reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        # image
        filename = info['image name']
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        img = Image.open(img_path).convert('RGB')
        
        # label
        lbl = info['image quality level']
        
        if self.transform is not None:
            img = self.transform(img)
        
        img = np.array(img)
        img = img - np.mean(img)
        img = np.transpose(img, [2, 0, 1])
        img = torch.FloatTensor(img)
        lbl = torch.LongTensor(np.array([lbl]))
        
        return img, lbl