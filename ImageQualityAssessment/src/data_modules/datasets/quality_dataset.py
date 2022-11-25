import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class QualityDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, fold_idx=0, input_size=1024, u_df=None):
        super().__init__()
        
        self.input_size = input_size
        self.data_dir = data_dir
        self.transform = transform
        
        self.task_tag = 'B. Image Quality Assessment'
        df = pd.read_csv(os.path.join(data_dir, self.task_tag, '2. Groundtruths', 'a. DRAC2022_ Image Quality Assessment_Training Labels.csv'))
        df = df[df[f'fold_{fold_idx}']==split]
        df.reset_index(drop=True)
        self.df = df[['image name', 'image quality level']]
        if u_df is not None:
            self.df = self.df.append(u_df[['image name', 'image quality level']])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        # image
        filename = info['image name']
        if self.input_size == 1024:
            img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        else:
            raise NotImplementedError
        # img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set_512_ben40', filename)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # label
        lbl = info['image quality level']
        
        if self.transform is not None:
            img = self.transform(image=img)['image']
        
        return img, lbl