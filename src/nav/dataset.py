from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from nav.preprocess.store import SessionManifest, read_array

STEP_CYCLES = 2  # cycles are 10 Hz, so 2 cycles == 200 ms, matching the model's original step
MAX_TICK_JITTER_US = 30_000
CYCLE_PERIOD_US = 100_000

# The arch cameras (the default rgb_cameras/depth_cameras) quantise depth to a 5.0m horizon
# (karter topics.yaml: depth_quantisation.horizon_m); the blind-spot pair uses 2.0m. This is a
# single global normalizer, matching the old pipeline's DEPTH_MAX_M -- correct for the default
# camera set, an approximation if blind-spot cameras are added to depth_cameras.
DEPTH_MAX_M = 5.0


class CycleDataset(Dataset):
    """A dataset of (cycle_i, cycle_i + step_cycles) pairs over one preprocessed session store.

    Every alignment question the old pipeline answered by guessing a latency is now answered by
    construction: the pair's endpoints are the actual planning cycles, and each modality's frame
    index was resolved from the recorded sensor frontier at preprocessing time. The only thing
    still checked here is that nothing sits between the endpoints -- a gap, a restart, a missing
    modality -- because `store.dropped` already removed the untrainable rows individually and a
    valid *pair* additionally needs both its ends, and nothing between them, to have survived.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        step_cycles: int = STEP_CYCLES,
        max_tick_jitter_us: int = MAX_TICK_JITTER_US,
        rgb_cameras: tuple[str, ...] | None = None,
        depth_cameras: tuple[str, ...] | None = None,
    ):
        self.store_dir = Path(store_dir)
        self.manifest = SessionManifest.read(self.store_dir)
        self.step_cycles = step_cycles

        self.rgb_cameras = tuple(rgb_cameras) if rgb_cameras is not None else tuple(self.manifest.rgb_cameras)
        self.depth_cameras = tuple(depth_cameras) if depth_cameras is not None else tuple(self.manifest.depth_cameras)
        self.rgb_size = tuple(self.manifest.rgb_size)
        self.depth_size = tuple(self.manifest.depth_size)

        self.run_id = read_array(self.store_dir, "cycles/run_id.npy", mmap=False)
        self.cycle_id = read_array(self.store_dir, "cycles/cycle_id.npy", mmap=False)
        self.tick_us = read_array(self.store_dir, "cycles/tick_us.npy", mmap=False)
        self.action = read_array(self.store_dir, "cycles/action.npy", mmap=False)
        self.proprio = read_array(self.store_dir, "cycles/proprio.npy", mmap=False)
        self.lidar_valid = read_array(self.store_dir, "lidar/valid.npy", mmap=False)

        self.action_mean = np.zeros(2, dtype=np.float32)
        self.action_std = np.ones(2, dtype=np.float32)

        self._rgb_index = {cam: read_array(self.store_dir, f"index/rgb_{cam}.npy", mmap=False) for cam in self.rgb_cameras}
        self._depth_index = {
            cam: read_array(self.store_dir, f"index/depth_{cam}.npy", mmap=False) for cam in self.depth_cameras
        }
        self._rgb_arrays = None
        self._depth_arrays = None
        self._bev = None

        self.pairs = self._build_pairs(max_tick_jitter_us)

    def _build_pairs(self, max_tick_jitter_us: int) -> np.ndarray:
        n = len(self.cycle_id)
        step = self.step_cycles
        if n <= step:
            return np.zeros((0, 2), dtype=np.int64)

        i = np.arange(n - step, dtype=np.int64)
        j = i + step

        ok = self.run_id[i] == self.run_id[j]
        ok &= (self.cycle_id[j] - self.cycle_id[i]) == step
        expected_dt = step * CYCLE_PERIOD_US
        ok &= np.abs((self.tick_us[j] - self.tick_us[i]) - expected_dt) <= max_tick_jitter_us
        ok &= self.lidar_valid[i] & self.lidar_valid[j]

        for index in self._rgb_index.values():
            ok &= (index[i] >= 0) & (index[j] >= 0)
        for index in self._depth_index.values():
            ok &= (index[i] >= 0) & (index[j] >= 0)

        return np.stack([i[ok], j[ok]], axis=1)

    def _open_arrays(self) -> None:
        # Deferred past __init__ so a DataLoader worker opens its own memmaps rather than
        # inheriting file handles across the fork.
        if self._rgb_arrays is None:
            self._rgb_arrays = {cam: read_array(self.store_dir, f"rgb/{cam}.npy") for cam in self.rgb_cameras}
            self._depth_arrays = {cam: read_array(self.store_dir, f"depth/{cam}.npy") for cam in self.depth_cameras}
            self._bev = read_array(self.store_dir, "lidar/bev.npy")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        self._open_arrays()
        i, j = self.pairs[index]
        steps = (i, j)

        width, height = self.rgb_size
        depth_width, depth_height = self.depth_size
        n_rgb = max(len(self.rgb_cameras), 1)
        n_depth = max(len(self.depth_cameras), 1)

        rgb = torch.empty((2, n_rgb, 3, height, width), dtype=torch.float32)
        depth = torch.empty((2, n_depth, 2, depth_height, depth_width), dtype=torch.float32)

        for step, cycle in enumerate(steps):
            for cam_i, cam in enumerate(self.rgb_cameras):
                frame = self._rgb_arrays[cam][self._rgb_index[cam][cycle]]
                rgb[step, cam_i] = torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0
            for cam_i, cam in enumerate(self.depth_cameras):
                depth_mm = self._depth_arrays[cam][self._depth_index[cam][cycle]]
                depth_m = depth_mm.astype(np.float32) / 1000.0
                depth[step, cam_i, 0] = torch.from_numpy(depth_m / DEPTH_MAX_M)
                depth[step, cam_i, 1] = torch.from_numpy((depth_m > 0).astype(np.float32))

        lidar = np.stack([self._bev[cycle] for cycle in steps])
        lidar = torch.from_numpy(lidar.astype(np.float32)).unsqueeze(1)

        action = (self.action[i] - self.action_mean) / self.action_std
        proprio = np.stack([self.proprio[cycle] for cycle in steps])

        return {
            "rgb": rgb,
            "depth": depth,
            "bev": lidar,
            "action": torch.from_numpy(action),
            "proprio": torch.from_numpy(proprio.astype(np.float32)),
            "t_us": torch.from_numpy(np.array([self.tick_us[i], self.tick_us[j]], dtype=np.int64)),
        }
