import os
from collections.abc import Iterator

import numpy as np
import pandas as pd

from nav.frame_transformer import lidar_to_robot
from nav.utils.data_utils import navigating_mask, navigating_ranges, nearest_within

BEV_EXTENT_M = 6.0
BEV_RESOLUTION_M = 0.1


def load_lidar_csv(path: str) -> tuple[np.ndarray, list[np.ndarray]]:
    """Streams a single lidar's CSV, returning (timestamps_us, points_xy_robot_frame).

    points_xy_robot_frame[i] is the (N_i, 2) array of points for timestamps_us[i],
    already transformed into the robot frame. N_i varies per timestamp.
    """
    timestamps = []
    points = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            timestamp_us, _seq, frame_id, x, y, _z = line.rstrip("\n").split(";")
            xs = np.fromstring(x, dtype=np.float32, sep=",")
            ys = np.fromstring(y, dtype=np.float32, sep=",")
            valid = (xs != 0.0) | (ys != 0.0)
            xs = xs[valid]
            ys = ys[valid]
            xyz = np.zeros((xs.shape[0], 3), dtype=np.float32)
            xyz[:, 0] = xs
            xyz[:, 1] = ys
            timestamps.append(int(timestamp_us))
            points.append(lidar_to_robot(frame_id, xyz)[:, :2].astype(np.float32))
    return np.array(timestamps, dtype=np.int64), points


def filter_timestamps(timestamps: np.ndarray, points: list[np.ndarray], nav_cmd: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Filters timestamps and points to only those in ranges where navigation commands were issued."""
    mask = navigating_mask(timestamps, navigating_ranges(nav_cmd))
    filtered_points = [p for p, keep in zip(points, mask) if keep]
    return timestamps[mask], filtered_points


def load_lidars(csv_paths: list[str]) -> dict[str, tuple[np.ndarray, list[np.ndarray]]]:
    """Loads multiple lidar CSVs, keyed by each file's frame_id."""
    result = {}
    for path in csv_paths:
        timestamps, points = load_lidar_csv(path)
        with open(path) as f:
            next(f)
            frame_id = next(f).split(";")[2]
        result[frame_id] = (timestamps, points)
    return result


def rasterize_bev(points_xy: np.ndarray, extent_m: float = BEV_EXTENT_M, resolution_m: float = BEV_RESOLUTION_M) -> np.ndarray:
    """Rasterizes robot-frame XY points into a centered top-down occupancy grid."""
    size = round(2 * extent_m / resolution_m)
    grid = np.zeros((size, size), dtype=np.uint8)
    row = np.floor((points_xy[:, 0] + extent_m) / resolution_m).astype(np.int64)
    col = np.floor((points_xy[:, 1] + extent_m) / resolution_m).astype(np.int64)
    valid = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    grid[row[valid], col[valid]] = 1
    return grid


def merge_scans(
    ref_ts: np.ndarray,
    ref_points: list[np.ndarray],
    others: list[tuple[np.ndarray, list[np.ndarray]]],
    tolerance_us: int = 60_000,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yields (timestamp_us, occupancy_grid), merging ref with the nearest scan within tolerance from each other lidar."""
    other_lookup = [(nearest_within(ts, ref_ts, tolerance_us), pts) for ts, pts in others]
    for i, t in enumerate(ref_ts):
        points = [ref_points[i]]
        for (idx, ok), pts in other_lookup:
            if ok[i]:
                points.append(pts[idx[i]])
        merged = np.concatenate(points, axis=0)
        yield int(t), rasterize_bev(merged)


def open_lidar_bev(bev_path: str) -> np.ndarray:
    """Opens preprocessed BEV grids as a read-only memmap."""
    return np.load(bev_path, mmap_mode="r")


if __name__ == "__main__":

    data_dir = r"/data/model-training"
    nav_cmd = pd.read_csv(os.path.join(data_dir, r"sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv"), sep=";")


    for d in os.listdir(data_dir):
        if not d.startswith("lidar"):
            continue
        path = os.path.join(data_dir, d, d + "_points.csv")
        timestamps, points = load_lidar_csv(path)
        timestamps, points = filter_timestamps(timestamps, points, nav_cmd)
        print(f"{d}: {len(timestamps)} timestamps, {sum(len(p) for p in points)} total points")
