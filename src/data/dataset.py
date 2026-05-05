"""PyTorch Dataset and DataModule for EuroSAT CSV-based loading."""

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class EurosatDataset(Dataset):
    """Loads EuroSAT samples from a CSV manifest.

    Args:
        csv_path: Path to train.csv / val.csv / test.csv.
        transform: torchvision transform pipeline.
    """

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["filename"]).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


class EurosatDataModule:
    """Lightweight DataModule that wraps train/val/test DataLoaders.

    Args:
        config: Application config dict (from config.yaml).
        train_transforms: Augmentation pipeline for training.
        val_transforms: Preprocessing pipeline for validation and test.
    """

    def __init__(
        self,
        config: dict,
        train_transforms=None,
        val_transforms=None,
    ):
        self.cfg = config
        self.train_tf = train_transforms
        self.val_tf = val_transforms

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            EurosatDataset(self.cfg["splits"]["train"], transform=self.train_tf),
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["training"]["num_workers"],
            pin_memory=self.cfg["training"]["pin_memory"],
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            EurosatDataset(self.cfg["splits"]["val"], transform=self.val_tf),
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["training"]["num_workers"],
            pin_memory=self.cfg["training"]["pin_memory"],
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            EurosatDataset(self.cfg["splits"]["test"], transform=self.val_tf),
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["training"]["num_workers"],
            pin_memory=self.cfg["training"]["pin_memory"],
        )
