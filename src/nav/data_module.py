import os

import numpy as np
import pandas as pd
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from nav.depth_loader import DEPTH_MAX_M, DEPTH_SIZE, list_depth_frames, load_depth
from nav.lidar_loader import open_lidar_bev
from nav.rgb_loader import CAMERAS, RgbDataset, list_camera_frames, load_image
from nav.utils.data_utils import (
    anchor_times,
    navigating_ranges,
    nearest_within,
    stretch_ids,
)

STEP_US = 200_000
LATENCY_US = {"rgb": 130_000, "depth": 150_000, "lidar": 100_000}
TOLERANCE_US = {"rgb": 100_000, "depth": 260_000, "lidar": 60_000}


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


def window_actions(nav_cmd: pd.DataFrame, t_us: np.ndarray, step_us: int) -> tuple[np.ndarray, np.ndarray]:
    """Averages desired_linear/angular_velocity over [t, t + step_us) for each anchor time."""
    nav_cmd = nav_cmd.sort_values("timestamp_us")
    ts = nav_cmd["timestamp_us"].to_numpy()
    lin = nav_cmd["desired_linear_velocity"].to_numpy(dtype=np.float32)
    ang = nav_cmd["desired_angular_velocity"].to_numpy(dtype=np.float32)

    start_idx = np.searchsorted(ts, t_us, side="left")
    end_idx = np.searchsorted(ts, t_us + step_us, side="left")
    count = np.maximum(end_idx - start_idx, 1).astype(np.float32)

    lin_cum = np.concatenate([[0.0], np.cumsum(lin)])
    ang_cum = np.concatenate([[0.0], np.cumsum(ang)])
    lin_mean = (lin_cum[end_idx] - lin_cum[start_idx]) / count
    ang_mean = (ang_cum[end_idx] - ang_cum[start_idx]) / count

    actions = np.stack([lin_mean, ang_mean], axis=1).astype(np.float32)
    valid = end_idx > start_idx
    return actions, valid


class WorldModelDataset(Dataset):
    """A dataset of (t, t+step_us) anchor pairs with per-modality latency-aligned RGB, depth and lidar BEV."""

    def __init__(
        self,
        data_dir: str,
        nav_cmd: pd.DataFrame,
        bev_path: str,
        bev_ts_path: str,
        cameras: tuple[str, ...] = ("oakd_back",),
        rgb_size: tuple[int, int] = (253, 190),
        depth_size: tuple[int, int] = DEPTH_SIZE,
        step_us: int = STEP_US,
        latency_us: dict[str, int] = LATENCY_US,
        tolerance_us: dict[str, int] = TOLERANCE_US,
    ):
        self.cameras = cameras
        self.rgb_size = rgb_size
        self.depth_size = depth_size
        self.bev_path = bev_path
        self._bev = None
        self.action_mean = np.zeros(2, dtype=np.float32)
        self.action_std = np.ones(2, dtype=np.float32)

        self.ranges = navigating_ranges(nav_cmd)
        end_us = int(nav_cmd["timestamp_us"].max())
        t_us, stretch_id = anchor_times(self.ranges, end_us, step_us)
        actions, action_ok = window_actions(nav_cmd, t_us[:, 0], step_us)

        self.rgb_paths = []
        self.depth_paths = []
        rgb_idx = np.zeros((len(t_us), 2, len(cameras)), dtype=np.int64)
        depth_idx = np.zeros((len(t_us), 2, len(cameras)), dtype=np.int64)
        ok = action_ok.copy()
        for cam_i, camera in enumerate(cameras):
            camera_dir = os.path.join(data_dir, camera)
            rgb_ts, rgb_paths = list_camera_frames(camera_dir)
            depth_ts, depth_paths = list_depth_frames(data_dir, camera)
            self.rgb_paths.append(rgb_paths)
            self.depth_paths.append(depth_paths)
            for step in range(2):
                idx, cam_ok = nearest_within(rgb_ts, t_us[:, step] - latency_us["rgb"], tolerance_us["rgb"])
                rgb_idx[:, step, cam_i] = idx
                ok &= cam_ok
                idx, cam_ok = nearest_within(depth_ts, t_us[:, step] - latency_us["depth"], tolerance_us["depth"])
                depth_idx[:, step, cam_i] = idx
                ok &= cam_ok

        lidar_ts = np.load(bev_ts_path)
        lidar_idx = np.zeros((len(t_us), 2), dtype=np.int64)
        for step in range(2):
            idx, lidar_ok = nearest_within(lidar_ts, t_us[:, step] - latency_us["lidar"], tolerance_us["lidar"])
            lidar_idx[:, step] = idx
            ok &= lidar_ok

        self.t_us = t_us[ok]
        self.stretch_id = stretch_id[ok]
        self.actions = actions[ok]
        self.rgb_idx = rgb_idx[ok]
        self.depth_idx = depth_idx[ok]
        self.lidar_idx = lidar_idx[ok]

    def _lidar_bev(self) -> np.ndarray:
        if self._bev is None:
            self._bev = open_lidar_bev(self.bev_path)
        return self._bev

    def __len__(self) -> int:
        return len(self.t_us)

    def __getitem__(self, index: int) -> dict:
        n_cam = len(self.cameras)
        width, height = self.rgb_size
        depth_width, depth_height = self.depth_size

        rgb = torch.empty((2, n_cam, 3, height, width), dtype=torch.float32)
        depth = torch.empty((2, n_cam, 2, depth_height, depth_width), dtype=torch.float32)
        for step in range(2):
            for cam in range(n_cam):
                image = load_image(self.rgb_paths[cam][self.rgb_idx[index, step, cam]], self.rgb_size)
                rgb[step, cam] = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

                d = load_depth(self.depth_paths[cam][self.depth_idx[index, step, cam]], self.depth_size)
                depth[step, cam, 0] = torch.from_numpy(d / DEPTH_MAX_M)
                depth[step, cam, 1] = torch.from_numpy((d > 0).astype(np.float32))

        bev = self._lidar_bev()
        lidar = np.stack([bev[self.lidar_idx[index, step]] for step in range(2)])
        lidar = torch.from_numpy(lidar).float().unsqueeze(1)

        action = (self.actions[index] - self.action_mean) / self.action_std

        return {
            "rgb": rgb,
            "depth": depth,
            "bev": lidar,
            "action": torch.from_numpy(action),
            "t_us": torch.from_numpy(self.t_us[index]),
        }


class WorldModelDataModule(LightningDataModule):
    """Wraps WorldModelDataset for use with a Lightning Trainer."""

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        bev_path: str,
        bev_ts_path: str,
        cameras: tuple[str, ...] = ("oakd_back",),
        val_fraction: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 12,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.bev_path = bev_path
        self.bev_ts_path = bev_ts_path
        self.cameras = cameras
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        nav_cmd = pd.read_csv(self.csv_path, sep=";")
        dataset = WorldModelDataset(self.data_dir, nav_cmd, self.bev_path, self.bev_ts_path, self.cameras)

        train_idx, val_idx = chronological_split(dataset.t_us[:, 0], dataset.ranges, self.val_fraction)
        dataset.action_mean = dataset.actions[train_idx].mean(axis=0)
        dataset.action_std = np.maximum(dataset.actions[train_idx].std(axis=0), 1e-6)

        self.dataset = dataset
        self.train_dataset = Subset(dataset, train_idx.tolist())
        self.val_dataset = Subset(dataset, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )


if __name__ == "__main__":

    data_dir = r"/data/model-training"
    csv_path = os.path.join(data_dir, r"sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv")

    dm = RgbDataModule(data_dir, csv_path, cameras=("oakd_back",), batch_size=8, num_workers=8)
    dm.setup()
    print(f"train={len(dm.train_dataset)} val={len(dm.val_dataset)}")

    images, timestamps = next(iter(dm.train_dataloader()))
    print(f"batch shape={tuple(images.shape)} dtype={images.dtype}")

    wm_dm = WorldModelDataModule(
        data_dir, csv_path,
        os.path.join(data_dir, "derived", "lidar_bev.npy"),
        os.path.join(data_dir, "derived", "lidar_bev_ts.npy"),
        cameras=("oakd_back",), batch_size=8, num_workers=8,
    )
    wm_dm.setup()
    print(f"train={len(wm_dm.train_dataset)} val={len(wm_dm.val_dataset)}")

    batch = next(iter(wm_dm.train_dataloader()))
    for key, value in batch.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
