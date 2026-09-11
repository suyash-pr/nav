import os

import numpy as np
import pandas as pd
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset

from nav.rgb_loader import CAMERAS, RgbDataset
from nav.utils.data_utils import stretch_ids


def chronological_split(timestamps: np.ndarray, ranges: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Splits sample indices into (train, val) by holding out whole navigating stretches from the end of the session."""

    sample_stretch = stretch_ids(timestamps, ranges)
    counts = np.bincount(sample_stretch, minlength=len(ranges))
    cumulative = np.cumsum(counts)
    n_train = round(len(timestamps) * (1 - val_fraction))
    boundary_stretch = np.searchsorted(cumulative, n_train, side="right")

    train_mask = sample_stretch < boundary_stretch
    return np.flatnonzero(train_mask), np.flatnonzero(~train_mask)


class RgbDataModule(LightningDataModule):
    """Wraps RgbDataset for use with a Lightning Trainer."""

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        cameras: tuple[str, ...] = CAMERAS,
        size: tuple[int, int] = (253, 190),
        tolerance_us: int = 100_000,
        val_fraction: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.cameras = cameras
        self.size = size
        self.tolerance_us = tolerance_us
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        nav_cmd = pd.read_csv(self.csv_path, sep=";")
        dataset = RgbDataset(self.data_dir, nav_cmd, self.cameras, self.size, self.tolerance_us)

        # Every camera column is within tolerance_us of the others, so any one of them identifies
        # the stretch a sample belongs to.
        train_idx, val_idx = chronological_split(dataset.timestamps[:, 0], dataset.ranges, self.val_fraction)
        self.train_dataset = Subset(dataset, train_idx.tolist())
        self.val_dataset = Subset(dataset, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


if __name__ == "__main__":

    data_dir = r"/data/model-training"
    csv_path = os.path.join(data_dir, r"sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv")

    dm = RgbDataModule(data_dir, csv_path, cameras=("oakd_back",), batch_size=8, num_workers=8)
    dm.setup()
    print(f"train={len(dm.train_dataset)} val={len(dm.val_dataset)}")

    images, timestamps = next(iter(dm.train_dataloader()))
    print(f"batch shape={tuple(images.shape)} dtype={images.dtype}")
