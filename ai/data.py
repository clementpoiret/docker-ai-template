from pathlib import Path
from typing import List

import lightning as L
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import ai.constants as C
from ai.transforms import PadIfNeeded, ResizeAndCrop


class DummyDataset(Dataset):

    def __init__(self, labels_file: Path, transforms: List = []):
        super().__init__()
        self.transform = T.Compose(transforms)

        labels = pd.read_csv(labels_file)
        self.data = labels

        self.N = len(labels)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        img = Image.open(row["path"]).convert("RGB")
        img = ImageOps.exif_transpose(img)

        img = self.transform(img)

        label = row["label"]

        return img, label


class DummyDataModule(L.LightningDataModule):

    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 shuffle: bool = True,
                 use_all_data: bool = False):
        super().__init__()

        self.imgs_path = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.use_all_data = use_all_data

        # Preprocessing
        self.transformation_pipeline = [
            ResizeAndCrop(C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
            PadIfNeeded(C.IMAGE_HEIGHT, C.IMAGE_WIDTH),
        ]

        # Normalize
        self.transformation_pipeline.extend([
            T.ToTensor(),
            T.Normalize(mean=C.MEAN, std=C.STD),
        ])

        # self.dims is returned when you call dm.size()
        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()
        # self.dims = (1, 28, 28)
        # self.num_metrics = 6

    # def prepare_data(self):
    #    Ellipsis

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = DummyDataset(self.imgs_path,
                                      transforms=self.transformation_pipeline)
            self.val = DummyDataset(self.imgs_path,
                                    transforms=self.transformation_pipeline)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=self.shuffle,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
