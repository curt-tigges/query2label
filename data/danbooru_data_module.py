import os
import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data.danbooru_dataset import DanbooruDataset


class DanbooruDataModule(pl.LightningDataModule):
    '''
    '''
    def __init__(
        self, 
        meta_path, 
        img_dir=None,
        subsample=None,
        min_support=100,
        batch_size=128, 
        num_workers=0
        ) -> None:

        super().__init__()
        self.meta_path = meta_path
        self.img_dir = img_dir
        self.subsample = subsample
        self.min_support = min_support
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_dir = (os.getcwd() if img_dir is None else img_dir)

    def prepare_data(self) -> None:
        '''Loads metadata file and subsamples it if requested.
        '''
        pass

    def setup(self, stage=None) -> None:

        df = pd.read_pickle(self.meta_path)

        if self.subsample != None:
            df = df.sample(self.subsample, random_state=42)

        self.metadata = df
        
        tag_dict = self.get_tag_dict(self.metadata, 10)
        self.classes = self.get_top_tags(tag_dict, self.min_support)
        self.num_classes = len(self.classes)

        train_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(360),
            transforms.ToTensor()]
        )

        test_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        train_val_df = self.metadata.sample(frac=0.9, random_state=42)
        test_df = self.metadata.drop(train_val_df.index)

        train_df = train_val_df.sample(frac=0.08, random_state=42)
        val_df = train_val_df.drop(train_df.index)

        self.train_set = DanbooruDataset(train_df, self.classes, self.img_dir, train_transforms)
        self.val_set = DanbooruDataset(val_df, self.classes, self.img_dir, test_transforms)
        self.test_set = DanbooruDataset(test_df, self.classes, self.img_dir, test_transforms)

    def get_class_mapping(self) -> dict:
        class_map = {n:c for n, c in enumerate(self.classes)}
        return class_map

    def get_num_classes(self):
        return len(self.classes)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def get_tag_dict(self, df, n=10):
        tag_dict = {}
        for list in df.tagslist:
            for tag in list:
                if tag in tag_dict:
                    tag_dict[tag] += 1
                else:
                    tag_dict[tag] = 1

        item = sorted(tag_dict.items(), key = lambda x:x[1],reverse = True)
        print(f"{n} top tags:")
        for i in range(0,n):
            print(item[i])

        return tag_dict

    def get_top_tags(self, tag_dict, min_support):
        reduced_dict = {t for t in tag_dict if tag_dict[t] > min_support}
        print(f"Dictionary contains {len(reduced_dict)} tags.")
        tag_list = [l for l in reduced_dict]
        return tag_list