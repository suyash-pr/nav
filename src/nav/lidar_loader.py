from pathlib import Path

import numpy as np

from nav.frame_transformer import LidarFrameTransformer

BEV_EXTENT_M = 6.0
BEV_RESOLUTION_M = 0.1


def load_lidar_csv(path: Path, transformer: LidarFrameTransformer) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Streams a single lidar CSV chunk, returning (timestamps_us, seqs, points_xy_robot_frame).

    points_xy_robot_frame[i] is the (N_i, 2) array of points for timestamps_us[i], already
    transformed into the robot frame via the frame_id each row carries.
    """
    timestamps = []
    seqs = []
    points = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            timestamp_us, seq, frame_id, x, y, _z = line.rstrip("\n").split(";")
            xs = np.fromstring(x, dtype=np.float32, sep=",")
            ys = np.fromstring(y, dtype=np.float32, sep=",")
            valid = (xs != 0.0) | (ys != 0.0)
            xs = xs[valid]
            ys = ys[valid]
            xyz = np.zeros((xs.shape[0], 3), dtype=np.float32)
            xyz[:, 0] = xs
            xyz[:, 1] = ys
            timestamps.append(int(timestamp_us))
            seqs.append(int(seq))
            points.append(transformer.lidar_to_robot(frame_id, xyz)[:, :2].astype(np.float32))
    return np.array(timestamps, dtype=np.int64), np.array(seqs, dtype=np.int64), points


def load_lidar_channel(paths: list[Path], transformer: LidarFrameTransformer) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Loads one channel's rotated CSV chunks, concatenated in write order."""
    all_ts, all_seq, all_points = [], [], []
    for path in paths:
        ts, seq, points = load_lidar_csv(path, transformer)
        all_ts.append(ts)
        all_seq.append(seq)
        all_points.extend(points)
    if not all_ts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), []
    return np.concatenate(all_ts), np.concatenate(all_seq), all_points


def rasterize_bev(points_xy: np.ndarray, extent_m: float = BEV_EXTENT_M, resolution_m: float = BEV_RESOLUTION_M) -> np.ndarray:
    """Rasterizes robot-frame XY points into a centered top-down occupancy grid."""
    size = round(2 * extent_m / resolution_m)
    grid = np.zeros((size, size), dtype=np.uint8)
    if points_xy.shape[0] == 0:
        return grid
    row = np.floor((points_xy[:, 0] + extent_m) / resolution_m).astype(np.int64)
    col = np.floor((points_xy[:, 1] + extent_m) / resolution_m).astype(np.int64)
    valid = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    grid[row[valid], col[valid]] = 1
    return grid


def open_lidar_bev(bev_path: Path) -> np.ndarray:
    """Opens preprocessed BEV grids as a read-only memmap."""
    return np.load(bev_path, mmap_mode="r")
