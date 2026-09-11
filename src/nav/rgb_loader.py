import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

from nav.utils.data_utils import navigating_mask, navigating_ranges

ImageFile.LOAD_TRUNCATED_IMAGES = True

CAMERAS = ("oakd_front", "oakd_left", "oakd_right", "oakd_back")
IMAGE_DIRS = ("images_0", "images_1", "images_2", "images_3")


def list_camera_frames(camera_dir: str) -> tuple[np.ndarray, list[str]]:
    """Lists a camera's frames across its images_0..3 dirs, returning (timestamps_us, paths), sorted and deduplicated by timestamp."""
    timestamps = []
    paths = []
    for image_dir in IMAGE_DIRS:
        dir_path = os.path.join(camera_dir, image_dir)
        for name in os.listdir(dir_path):
            timestamps.append(int(name.split("_", 1)[0]))
            paths.append(os.path.join(dir_path, name))

    timestamps_arr, unique_idx = np.unique(np.array(timestamps, dtype=np.int64), return_index=True)
    unique_paths = [paths[i] for i in unique_idx]
    return timestamps_arr, unique_paths


def align_cameras(timestamps: list[np.ndarray], tolerance_us: int = 100_000) -> np.ndarray:
    """Groups frames across cameras by nearest timestamp, returning an (n_samples, n_cameras) index matrix.

    The reference camera is the one with the fewest frames. Groups whose timestamp spread exceeds
    tolerance_us are dropped.
    """
    ref_cam = min(range(len(timestamps)), key=lambda i: len(timestamps[i]))
    ref_timestamps = timestamps[ref_cam]

    indices = np.zeros((len(ref_timestamps), len(timestamps)), dtype=np.int64)
    indices[:, ref_cam] = np.arange(len(ref_timestamps))

    for cam, cam_timestamps in enumerate(timestamps):
        if cam == ref_cam:
            continue
        idx = np.searchsorted(cam_timestamps, ref_timestamps)
        idx = np.clip(idx, 1, len(cam_timestamps) - 1)
        left, right = idx - 1, idx
        use_left = np.abs(cam_timestamps[left] - ref_timestamps) <= np.abs(cam_timestamps[right] - ref_timestamps)
        indices[:, cam] = np.where(use_left, left, right)

    aligned_ts = np.stack([timestamps[cam][indices[:, cam]] for cam in range(len(timestamps))], axis=1)
    spread = aligned_ts.max(axis=1) - aligned_ts.min(axis=1)
    return indices[spread <= tolerance_us]


def load_image(path: str, size: tuple[int, int]) -> np.ndarray:
    """Loads an image, decoding it directly at (or near) the target size, as an (H, W, 3) uint8 array."""
    with Image.open(path) as im:
        im.draft("RGB", size)
        rgb = im.convert("RGB")
        if rgb.size != size:
            rgb = rgb.resize(size, Image.Resampling.BILINEAR)
        return np.array(rgb)


class RgbDataset(Dataset):
    """A dataset of aligned multi-camera frames, filtered to navigating stretches.

    Each sample is a (n_cameras, 4, H, W) float32 tensor: channels are R, G, B (in [0, 1]) and a
    constant camera-id channel (cam_index / (n_cameras - 1)).
    """

    def __init__(
        self,
        data_dir: str,
        nav_cmd: pd.DataFrame,
        cameras: tuple[str, ...] = CAMERAS,
        size: tuple[int, int] = (253, 190),
        tolerance_us: int = 100_000,
    ):
        self.cameras = cameras
        self.size = size

        ranges = navigating_ranges(nav_cmd)
        filtered_timestamps = []
        filtered_paths = []
        for camera in cameras:
            timestamps, paths = list_camera_frames(os.path.join(data_dir, camera))
            mask = navigating_mask(timestamps, ranges)
            filtered_timestamps.append(timestamps[mask])
            filtered_paths.append([p for p, keep in zip(paths, mask) if keep])

        indices = align_cameras(filtered_timestamps, tolerance_us)
        self.timestamps = np.stack(
            [filtered_timestamps[cam][indices[:, cam]] for cam in range(len(cameras))], axis=1
        )
        self.paths: list[list[str]] = [
            [filtered_paths[cam][i] for cam, i in enumerate(row)] for row in indices
        ]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_cameras = len(self.cameras)
        width, height = self.size
        images = torch.empty((n_cameras, 4, height, width), dtype=torch.float32)
        for cam, path in enumerate(self.paths[index]):
            rgb = load_image(path, self.size)
            images[cam, :3] = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            images[cam, 3] = cam / (n_cameras - 1) if n_cameras > 1 else 0.0
        timestamps = torch.from_numpy(self.timestamps[index])
        return images, timestamps


if __name__ == "__main__":

    data_dir = r"/data/model-training"
    nav_cmd = pd.read_csv(os.path.join(data_dir, r"sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv"), sep=";")

    dataset = RgbDataset(data_dir, nav_cmd, cameras=("oakd_back",))
    print(f"{len(dataset)} aligned samples")

    images, timestamps = dataset[0]
    print(f"sample shape={tuple(images.shape)} dtype={images.dtype} min={images.min():.3f} max={images.max():.3f}")
    print(f"timestamps={timestamps.tolist()}")

    loader = DataLoader(dataset, batch_size=8, num_workers=8)
    start = time.time()
    n = 0
    for batch_images, _ in loader:
        n += batch_images.shape[0]
        if n >= 32:
            break
    elapsed = time.time() - start
    print(f"{n} samples in {elapsed:.2f}s ({elapsed / n:.4f}s/sample)")
