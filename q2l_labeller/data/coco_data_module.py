import os
import torch
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.utils.data import DataLoader
from q2l_labeller.data.coco_dataset import CoCoDataset
from q2l_labeller.data.cutmix import CutMixCollator


class COCODataModule(pl.LightningDataModule):
    """Datamodule for Lightning Trainer"""

    def __init__(
        self,
        data_dir,
        img_size,
        batch_size=128,
        num_workers=0,
        use_cutmix=False,
        cutmix_alpha=1.0,
    ) -> None:
        """_summary_

        Args:
            data_dir (str): Location of data.
            img_size (int): Desired size for transformed images.
            batch_size (int, optional): Dataloader batch size. Defaults to 128.
            num_workers (int, optional): Number of CPU threads to use. Defaults to 0.
            use_cutmix (bool, optional): Flag to enable Cutmix augmentation. Defaults to False.
            cutmix_alpha (float, optional): Defaults to 1.0.
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.collator = torch.utils.data.dataloader.default_collate

    def prepare_data(self) -> None:
        """Loads metadata file and subsamples it if requested"""
        pass

    def setup(self, stage=None) -> None:
        """Creates train, validation, test datasets

        Args:
            stage (str, optional): Stage. Defaults to None.
        """
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # mean=[0, 0, 0],
        # std=[1, 1, 1])

        train_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                RandAugment(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.train_set = CoCoDataset(
            image_dir=(self.data_dir + "images/train2014"),
            anno_path=(self.data_dir + "annotations/instances_train2014.json"),
            input_transform=train_transforms,
            labels_path=(self.data_dir + "annotations/labels_train2014.npy"),
        )
        self.val_set = CoCoDataset(
            image_dir=(self.data_dir + "images/val2014"),
            anno_path=(self.data_dir + "annotations/instances_val2014.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir + "annotations/labels_val2014.npy"),
        )
        """
        self.test_set = CoCoDataset(
            image_dir=(self.data_dir+"images/test2014"),
            anno_path=(self.data_dir+"annotations/instances_test2014.json"),
            input_transform=test_transforms,
            labels_path=(self.data_dir+"annotations/labels_test2014.npy"))
        """

        if self.use_cutmix:
            self.collator = CutMixCollator(self.cutmix_alpha)

    def get_num_classes(self):
        """Returns number of classes

        Returns:
            int: number of classes
        """
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        """Creates and returns training dataloader

        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        """Creates and returns validation dataloader

        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Creates and returns test dataloader

        Returns:
            DataLoader: Test dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
