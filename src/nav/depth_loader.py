import os

import numpy as np
from PIL import Image

DEPTH_DIRS = ("depth_0", "depth_1", "depth_2", "depth_3")
DEPTH_SCALE_M = 0.001
DEPTH_MAX_M = 10.0
DEPTH_SIZE = (160, 100)


def load_depth(path: str, size: tuple[int, int] = DEPTH_SIZE) -> np.ndarray:
    """Loads a 16-bit depth PNG (millimetres) as an (H, W) float32 array in metres. 0 = invalid."""
    with Image.open(path) as im:
        if im.size != size:
            im = im.resize(size, Image.Resampling.NEAREST)
        depth_mm = np.array(im, dtype=np.float32)
    return depth_mm * DEPTH_SCALE_M


def list_depth_frames(data_dir: str, camera: str) -> tuple[np.ndarray, list[str]]:
    """Lists a camera's depth frames from its flat top-level topic dir (`camera_<name>_mono_r`).

    Unlike RGB, depth frames aren't sharded under the camera dir in this export - they sit directly
    in a sibling topic dir. Returns (timestamps_us, paths), sorted and deduplicated by timestamp.
    """
    dir_path = os.path.join(data_dir, f"camera_{camera}_mono_r")
    timestamps = []
    paths = []
    for name in os.listdir(dir_path):
        prefix = name.split("_", 1)[0]
        if not prefix.isdigit():
            continue
        timestamps.append(int(prefix))
        paths.append(os.path.join(dir_path, name))
    timestamps_arr, unique_idx = np.unique(np.array(timestamps, dtype=np.int64), return_index=True)
    unique_paths = [paths[i] for i in unique_idx]
    return timestamps_arr, unique_paths


if __name__ == "__main__":
    from nav.rgb_loader import list_camera_frames

    data_dir = r"/data/model-training"
    timestamps, paths = list_camera_frames(os.path.join(data_dir, "oakd_back"), DEPTH_DIRS)
    print(f"{len(timestamps)} depth frames")

    depth = load_depth(paths[0])
    print(f"shape={depth.shape} dtype={depth.dtype} max={depth.max():.3f}m")
