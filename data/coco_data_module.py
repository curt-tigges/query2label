import os
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from data.coco_dataset import CoCoDataset


class COCODataModule(pl.LightningDataModule):
    '''
    '''
    def __init__(
        self, 
        data_dir,
        img_size,
        batch_size=128, 
        num_workers=0
        ) -> None:

        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        '''Loads metadata file and subsamples it if requested.
        '''
        pass

    def setup(self, stage=None) -> None:

        normalize = transforms.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1])

        train_transforms = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)),
            RandAugment(),
            transforms.ToTensor(),
            normalize]
        )

        test_transforms = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            normalize]
        )

        self.train_set = CoCoDataset(
            image_dir=(self.data_dir+"images/train2014"),
            anno_path=(self.data_dir+"annotations/instances_train2014.json"),
            input_transform=train_transforms,
            labels_path=(self.data_dir+"annotations/labels_train2014.npy"),
        )
        self.val_set = CoCoDataset(
            image_dir=(self.data_dir+"images/val2014"),
            anno_path=(self.data_dir+"annotations/instances_val2014.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir+"annotations/labels_val2014.npy")
        )
        '''
        self.test_set = CoCoDataset(
            image_dir=(self.data_dir+"images/test2014"),
            anno_path=(self.data_dir+"annotations/instances_test2014.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir+"annotations/labels_test2014.npy"))
        '''

    def get_num_classes(self):
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)