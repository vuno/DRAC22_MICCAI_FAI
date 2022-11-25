import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import transforms as T
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .augmentations.augments import RandAugment
from .augmentations.augments_v2 import RandAugmentFUNDUS
from .samplers.cycle_sampler import CycleSampler
from .datasets import get_drac_dataset
from .augmentations.bens import BensPreprocessing


class DRACDM(LightningDataModule):
    def __init__(
        self,
        task,
        data_dir,
        input_size,
        batch_size,
        num_workers,
        balanced_sampling,
        fold_idx,
        auto_aug=None,
        u_df=None,
    ):
        super().__init__()
        self.task = task
        self.data_dir = data_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balanced_sampling = balanced_sampling
        self.fold_idx = fold_idx
        self.u_df = u_df
        
        if auto_aug is None:
            self.train_transform = A.Compose([
                # A.Resize(input_size, input_size),
                # BensPreprocessing(sigmaX=40),
                # # base
                A.Flip(),
                A.ShiftScaleRotate(),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ]),
                A.CoarseDropout(max_height=5, min_height=1, max_width=512, min_width=51),
                A.OneOf([
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                ]),
                
                # trial
                # A.HorizontalFlip(),
                # A.VerticalFlip(),
                # A.ShiftScaleRotate(border_mode=0, value=0),
                # A.RandomBrightnessContrast(),
                # A.CoarseDropout(max_height=3, min_height=1, max_width=512, max_holes=5),
                # A.HorizontalFlip(),
                # A.ShiftScaleRotate(rotate_limit=15, border_mode=0),
                # A.RandomBrightnessContrast(p=0.5),
                # A.OneOf([
                #     A.CoarseDropout(max_height=5, min_height=1, max_width=512, max_holes=5, p=1),
                #     A.Blur(blur_limit=3, p=1),
                # ], p=0.5)
                
                # # from fai_2022
                # A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(rotate_limit=180, border_mode=0, value=0, mask_value=0, p=0.8),
                # A.RGBShift(r_shift_limit=30, g_shift_limit=20, b_shift_limit=10, p=0.2),
                # A.OneOf([
                #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, p=1),
                #     A.RandomGamma(gamma_limit=(50, 150), p=1),
                # ], p=0.8),
                # A.CoarseDropout(max_height=40, max_width=40, p=0.2),
                # A.OneOf([
                #     A.Sharpen(p=1),
                #     A.Blur(blur_limit=10, p=1),
                #     A.ImageCompression(quality_lower=60, quality_upper=100, p=1),
                #     A.Downscale(scale_max=0.5, p=1),
                # ], p=0.2),
                
                # randaug
                # A.SomeOf([
                #     # A.RandomContrast(limit=0.4, p=1),
                #     A.Equalize(p=1),
                #     A.InvertImg(p=1),
                #     A.Rotate(limit=180, border_mode=0, value=0, p=1),
                #     A.Posterize(num_bits=[0,4], p=1),
                #     A.Solarize(p=1),
                #     A.ColorJitter(),
                #     A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0, p=1),
                #     A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.4, p=1),
                #     A.Sharpen(p=1),
                #     A.Affine(scale=(0.9, 1.1), p=1),
                #     A.Affine(translate_percent=(0, 0.1), p=1),
                #     A.Affine(shear=(-30,30), p=1),
                #     A.CoarseDropout(max_height=40, max_width=40, p=1)
                #     ], n=n_randaugment, p=1),
                
                # A.Normalize(mean=(0.4128,0.4128,0.4128), std=(0.2331,0.2331,0.2331)),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
                ToTensorV2(),
            ])
        else:
            self.train_transform = A.load(auto_aug)
        
        self.test_transform = A.Compose([
            # A.Resize(input_size, input_size),
            # BensPreprocessing(sigmaX=40),
            # A.Normalize(mean=(0.4128,0.4128,0.4128), std=(0.2331,0.2331,0.2331)),
            A.Normalize(mean=(0,0,0), std=(1,1,1)),
            ToTensorV2(),
        ])
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = get_drac_dataset(self.task, self.data_dir, 'train', self.train_transform, self.fold_idx, self.input_size, self.u_df)
            self.val_dataset = get_drac_dataset(self.task, self.data_dir, 'val', self.test_transform, self.fold_idx, self.input_size)

        if stage == 'test' or stage is None:
            self.test_dataset = get_drac_dataset(self.task, self.data_dir, 'test', self.test_transform, self.fold_idx, self.input_size)
    
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