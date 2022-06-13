from torch.utils.data.dataset import Dataset
import pandas as pd
import numpy as np
from PIL import Image

class DanbooruDataset(Dataset):
    def __init__(
        self, metadata, classes, img_path, transforms):

        self.transforms = transforms

        self.img_path = img_path
        self.images = metadata['paths'].tolist()

        self.labels = metadata['tagslist'].tolist()

        self.classes = classes

    def __getitem__(self, item):

        # load and transform image
        img = Image.open(self.img_path+self.images[item])
        
        im_arr = np.array(img)

        if im_arr.ndim == 2:
            channels = 1        
        else:
            channels = im_arr.shape[-1]

        if channels == 4:
            img = img.convert("RGB")
    
        if self.transforms is not None:
            img = self.transforms(img)

        if img.shape[0] == 1:
            img = img.expand(3, *img.shape[1:])
        
        

        # check for labels and create label list
        labels = np.array([int(l in self.labels[item]) for l in self.classes], dtype=float)

        return img, labels

    def __len__(self):
        return len(self.images)