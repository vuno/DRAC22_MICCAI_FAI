import os
import cv2
import numpy as np
import pandas as pd

import glob
import torch
from torch.utils.data import Dataset

# 0909 현재까지 베스트
class Task1Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None, target=2):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform
        
        self.task_tag = 'A. Segmentation'
        df = pd.read_csv(os.path.join(data_dir, 'segmentation_split.csv'))
        self.df = df
        #self.df = df[df['split'] == split]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        # image
        filename = info['filename']
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # label
        lbl = []
        for c in ['1. Intraretinal Microvascular Abnormalities', '2. Nonperfusion Areas', '3. Neovascularization']:
            lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', c, filename)
            if os.path.exists(lbl_path):
                mask = cv2.imread(lbl_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                assert (np.unique(mask) == [0, 255]).all(), np.unique(mask)
                lbl.append(mask/255.)
            else:
                lbl.append(np.zeros((1024,1024),np.float))
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
            img = aug['image']
            lbl[0] = aug['mask']
            lbl[1] = aug['mask1']
            lbl[2] = aug['mask2']
        lbl = torch.stack(lbl)
        #print(img.size())
        return img, lbl


class Task1Dataset2(Dataset):
    def __init__(self, data_dir, split, transform=None, target=2, pl_mask=None, patch_size=256, stride=128):
        super().__init__()
        
        self.data_dir = data_dir
        self.transform = transform
        
        self.task_tag = 'A. Segmentation'
        df = pd.read_csv(os.path.join(data_dir, 'segmentation_split.csv'))
        self.df = df
        #self.df = df[df['split'] == split]
        self.label_name = {
            1: ['1. Intraretinal Microvascular Abnormalities'],
            2: ['2. Nonperfusion Areas'],
            3: ['3. Neovascularization'],
        }
        self.target = int(target)
        self.pl_mask = pl_mask
        self.patch_size = patch_size
        self.stride = stride
        self.rows = np.int(np.ceil(1.0 * (1024 - self.patch_size) / self.stride)) + 1
        self.cols = np.int(np.ceil(1.0 * (1024 - self.patch_size) / self.stride)) + 1
        self.split = split

        if int(target) != 2:
            data = {"filename": []}
            for i, row in enumerate(self.df['filename'].values):
                lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', f'{self.label_name[self.target][0]}', row)
                if os.path.isfile(lbl_path):
                    data['filename'].append(row)
            self.df = pd.DataFrame(data=data)



    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        info = self.df.iloc[index]
        
        # image
        filename = info['filename']
        img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (512, 512))

        pl_mask = self.pl_mask[img_path]
        pl_mask = pl_mask / 255.
        
        # label
        lbl = []
        for c in self.label_name[self.target]:
            lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', c, filename)
            if os.path.exists(lbl_path):
                mask = cv2.imread(lbl_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                #mask = np.uint8(cv2.resize(mask, (512, 512)) > 0) * 255
                assert (np.unique(mask) == [0, 255]).all(), np.unique(mask)
                lbl.append(mask/255.)
            else:
                lbl.append(np.zeros((1024, 1024),np.float))
        
        if self.transform is not None:
            aug = self.transform(image=img, mask=lbl[0], mask1=pl_mask)
            img = aug['image']
            lbl[0] = aug['mask']
            pl_mask = aug['mask1'] #.unsqueeze(0)
            img[2, ...] = pl_mask
        lbl = lbl[0].unsqueeze(0)

        if self.split == 'train':
            crop_imgs = []
            crop_masks = []
            for r in range(self.rows):
                for c in range(self.cols):
                    h0 = r * self.stride
                    w0 = c * self.stride
                    h1 = min(h0 + self.patch_size, 1024)
                    w1 = min(w0 + self.patch_size, 1024)
                    h0 = max(int(h1 - self.patch_size), 0)
                    w0 = max(int(w1 - self.patch_size), 0)
                    crop_img = img[:, h0:h1, w0:w1] #new_img[h0:h1, w0:w1, :]
                    crop_mask = lbl[:, h0:h1, w0:w1]
                    crop_imgs.append(crop_img)
                    crop_masks.append(crop_mask)
            crop_imgs = torch.stack(crop_imgs, dim=0)
            crop_masks = torch.stack(crop_masks, dim=0)
            return crop_imgs, crop_masks
        else:
            return img, lbl


# class 3 용도로만 사용
# class Task1Dataset(Dataset):
#     def __init__(self, data_dir, split, transform=None, target=2):
#         super().__init__()
        
#         self.data_dir = data_dir
#         self.transform = transform
        
#         self.task_tag = 'A. Segmentation'
#         df = pd.read_csv(os.path.join(data_dir, 'segmentation_split.csv'))
#         self.df = df
#         #print("TEST")
#         #exit()
#         self.split = split
#         if self.split != 'train':
#             img_paths = glob.glob("/home/kimjaeyoung/DRAC_task1_best/class3_validation/image/*_0.png")
#             mask_paths = [x.replace("/image/", "/mask/").replace("_0.png", "_3.png") for x in img_paths]
#             print(img_paths)
#             print(mask_paths)
#             #exit()
#             self.df = img_paths
#             self.df_label = mask_paths
#         #self.df = df[df['split'] == split]

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         if self.split == 'train':
#             info = self.df.iloc[index]
#             #print(index)
#             # image
#             filename = info['filename']
#             img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             # label
#             lbl = []
#             for c in ['1. Intraretinal Microvascular Abnormalities', '2. Nonperfusion Areas', '3. Neovascularization']:
#                 lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', c, filename)
#                 if os.path.exists(lbl_path):
#                     mask = cv2.imread(lbl_path)
#                     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                     assert (np.unique(mask) == [0, 255]).all(), np.unique(mask)
#                     lbl.append(mask/255.)
#                 else:
#                     lbl.append(np.zeros((1024,1024),np.float))
            
#             if self.transform is not None:
#                 aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
#                 img = aug['image']
#                 lbl[0] = aug['mask']
#                 lbl[1] = aug['mask1']
#                 lbl[2] = aug['mask2']
#             lbl = torch.stack(lbl)
#             #print(img.size())
#             return img, lbl
#         else:
#             # image
#             img_path = self.df[index]
#             msk_path = self.df_label[index]
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
#             lbl = [mask/255., mask/255., mask/255.]
#             if self.transform is not None:
#                 aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
#                 img = aug['image']
#                 lbl[0] = aug['mask']
#                 lbl[1] = aug['mask1']
#                 lbl[2] = aug['mask2']
#             lbl = torch.stack(lbl)
#             #print(img.size())
#             return img, lbl


# class Task1Dataset(Dataset):
#     def __init__(self, data_dir, split, transform=None, target=2):
#         super().__init__()
        
#         self.data_dir = data_dir
#         self.transform = transform
        
#         self.task_tag = 'A. Segmentation'
#         df = pd.read_csv(os.path.join(data_dir, 'segmentation_split.csv'))

#         data = {'filename':[]}
#         for i, row in df.iterrows():
#             filename =row['filename']
#             lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', '3. Neovascularization', filename)
#             if os.path.exists(lbl_path):
#                 data['filename'].append(filename)
#         df = pd.DataFrame(data)

#         self.df = df
#         #print("TEST")
#         #exit()
#         self.split = split
#         if self.split != 'train':
#             img_paths = glob.glob("/home/kimjaeyoung/DRAC_task1_best/class3_validation/image/*_3.png")
#             mask_paths = [x.replace("/image/", "/mask/") for x in img_paths]
#             print(img_paths)
#             print(mask_paths)
#             #exit()
#             self.df = img_paths
#             self.df_label = mask_paths
#         #self.df = df[df['split'] == split]

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, index):
#         if self.split == 'train':
#             info = self.df.iloc[index]
#             #print(index)
#             # image
#             filename = info['filename']
#             img_path = os.path.join(self.data_dir, self.task_tag, '1. Original Images', 'a. Training Set', filename)
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
#             # label
#             lbl = []
#             for c in ['1. Intraretinal Microvascular Abnormalities', '2. Nonperfusion Areas', '3. Neovascularization']:
#                 lbl_path = os.path.join(self.data_dir, 'A. Segmentation', '2. Groundtruths', 'a. Training Set', c, filename)
#                 if os.path.exists(lbl_path):
#                     mask = cv2.imread(lbl_path)
#                     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                     assert (np.unique(mask) == [0, 255]).all(), np.unique(mask)
#                     lbl.append(mask/255.)
#                 else:
#                     lbl.append(np.zeros((1024,1024),np.float))
            
#             if self.transform is not None:
#                 aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
#                 img = aug['image']
#                 lbl[0] = aug['mask']
#                 lbl[1] = aug['mask1']
#                 lbl[2] = aug['mask2']
#             lbl = torch.stack(lbl)
#             #print(img.size())
#             return img, lbl
#         else:
#             # image
#             img_path = self.df[index]
#             msk_path = self.df_label[index]
#             img = cv2.imread(img_path)
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
#             lbl = [mask/255., mask/255., mask/255.]
#             if self.transform is not None:
#                 aug = self.transform(image=img, mask=lbl[0], mask1=lbl[1], mask2=lbl[2])
#                 img = aug['image']
#                 lbl[0] = aug['mask']
#                 lbl[1] = aug['mask1']
#                 lbl[2] = aug['mask2']
#             lbl = torch.stack(lbl)
#             #print(img.size())
#             return img, lbl