import os
import cv2
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class GradingDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, fold_idx=0):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform
        
        self.task_tag = 'C. Diabetic Retinopathy Grading'
        df = pd.read_csv(os.path.join(data_dir, self.task_tag, '2. Groundtruths', 'a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels_Fold.csv'))
        if fold_idx != 'pre':
            df = df[df[f'fold_{fold_idx}']==split]
            df.reset_index(drop=True)
        self.df = df
        self.split = split

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        # image
        filename = info['image name']
        # img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set_512_ben40', filename)
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # label
        lbl = info['DR grade']

        if self.transform is not None:
            # augmix
            if self.split == 'train':
                augmix_img = []
                origin_img = np.copy(img)
                for i in range(3):
                    img = self.transform(image=origin_img)['image']
                    img = img[i, ...].unsqueeze(0)
                    augmix_img.append(img)
                img = torch.cat(augmix_img, dim=0)
            else:
                img = self.transform(image=img)['image']
        return img, lbl