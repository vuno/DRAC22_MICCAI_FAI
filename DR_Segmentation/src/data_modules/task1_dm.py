import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .samplers.cycle_sampler import CycleSampler
from .datasets.task1_dataset import Task1Dataset


class Task1DM(LightningDataModule):
    def __init__(
        self,
        task,
        data_dir,
        input_size,
        batch_size,
        num_workers,
        balanced_sampling,
        target,
    ):
        super().__init__()
        self.task = task
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling
        self.target = target

        # self.train_transform = A.Compose([
        #     # geometric
        #     A.OneOf([
        #         A.Flip(p=1),
        #         A.RandomRotate90(p=1),
        #         A.RandomResizedCrop(height=1024, width=1024, p=1),
        #         #A.RandomScale(p=1),
        #         A.NoOp(p=1)
        #     ]),
        #     # distort
        #     A.OneOf([
        #         A.RandomBrightnessContrast(p=1),
        #         A.CoarseDropout(max_height=1024, min_height=1000, max_width=24, min_width=1, max_holes=2, p=1, mask_fill_value=0.),
        #         A.CoarseDropout(max_height=24, min_height=1, max_width=1024, min_width=1000, max_holes=2, p=1, mask_fill_value=0.),
        #         A.NoOp(p=1),
        #         A.Blur(p=1),
        #         A.GaussNoise(p=1),
        #         A.RandomShadow(shadow_roi=(0, 0, 1, 1), p=1)
        #     ]),
        #     A.OneOf([
        #         A.CLAHE(p=1),
        #         A.Equalize(p=1),
        #         A.NoOp(p=1)
        #     ]),
        #     A.Normalize(mean=(0,0,0), std=(1,1,1)),
        #     ToTensorV2(),
        # ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
        
        self.train_transform = A.Compose([
            # A.GaussianBlur(p=0.3),
            # A.HorizontalFlip(p=0.3),
            # A.RandomBrightnessContrast(p=0.3),
            # A.ShiftScaleRotate(p=0.3),
            
            A.Flip(),
            A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=90), # default = A.ShiftScaleRotate()
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ]),
            ##A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51, mask_fill_value=0),
            A.OneOf([
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
            ]),
            #A.RandomResizedCrop(512, 512, p=0.2),
            A.GridDistortion(p=0.2),
            A.CoarseDropout(max_height=128, min_height=32, max_width=128, min_width=32, max_holes=3, p=0.2, mask_fill_value=0.),
            
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
        
        self.test_transform = A.Compose([
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ], additional_targets={'mask1': 'mask', 'mask2': 'mask'})
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Task1Dataset(self.data_dir, 'train', self.train_transform, self.target)
            self.val_dataset = Task1Dataset(self.data_dir, 'val', self.test_transform, self.target)

        if stage == 'test' or stage is None:
            self.test_dataset = Task1Dataset(self.task, self.data_dir, 'test', self.test_transform)
    
    def train_dataloader(self):
        if self.balanced_sampling:
            train_sampler = CycleSampler(self.train_dataset)
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


# class Task1DM2(LightningDataModule):
#     def __init__(
#         self,
#         task,
#         data_dir,
#         input_size,
#         batch_size,
#         num_workers,
#         balanced_sampling,
#         target,
#         pl_mask,
#     ):
#         super().__init__()
#         self.task = task
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.balanced_sampling = balanced_sampling
#         self.target = target
#         self.pl_mask = pl_mask
        
#         self.train_transform = A.Compose([
#             # A.GaussianBlur(p=0.3),
#             # A.HorizontalFlip(p=0.3),
#             # A.RandomBrightnessContrast(p=0.3),
#             # A.ShiftScaleRotate(p=0.3),
            
#             A.Flip(),
#             A.ShiftScaleRotate(shift_limit=0.2, rotate_limit=90), # default = A.ShiftScaleRotate()
#             A.OneOf([
#                 A.RandomBrightnessContrast(p=1),
#                 A.RandomGamma(p=1),
#             ]),
#             ##A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51, mask_fill_value=0),
#             A.OneOf([
#                 A.Sharpen(p=1),
#                 A.Blur(blur_limit=3, p=1),
#                 A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
#             ]),
#             #A.RandomResizedCrop(512, 512, p=0.2),
#             A.GridDistortion(p=0.2),
#             A.CoarseDropout(max_height=128, min_height=32, max_width=128, min_width=32, max_holes=3, p=0.2, mask_fill_value=0.),
#             A.Affine(scale=[0.8, 1.2]),
            
#             A.Normalize(mean=(0,0,0), std=(1,1,1)),
#             ToTensorV2(),
#         ], additional_targets={'mask1': 'mask'})
        
#         self.test_transform = A.Compose([
#             A.Normalize(mean=(0,0,0), std=(1,1,1)),
#             ToTensorV2(),
#         ], additional_targets={'mask1': 'mask'})
    
#     def setup(self, stage=None):
#         if stage == 'fit' or stage is None:
#             self.train_dataset = Task1Dataset2(self.data_dir, 'train', self.train_transform, self.target, self.pl_mask)
#             self.val_dataset = Task1Dataset2(self.data_dir, 'test', self.test_transform, self.target, self.pl_mask)

#         if stage == 'test' or stage is None:
#             self.test_dataset = Task1Dataset2(self.task, self.data_dir, 'test', self.test_transform)
    
#     def train_dataloader(self):
#         if self.balanced_sampling:
#             train_sampler = CycleSampler(self.train_dataset)
#             return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)
#         else:
#             return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle=True)
    
#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)
    
#     def test_dataloader(self):
#         return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)