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
                A.Flip(),
                A.ShiftScaleRotate(),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ]),
                A.OneOf([
                    A.Sharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.Downscale(scale_min=0.7, scale_max=0.9, p=1),
                ]),
                A.Normalize(mean=(0,0,0), std=(1,1,1)),
                ToTensorV2(),
            ])
        else:
            self.train_transform = A.load(auto_aug)
        
        self.test_transform = A.Compose([
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