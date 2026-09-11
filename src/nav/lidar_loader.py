import os

import numpy as np
import pandas as pd

from nav.frame_transformer import lidar_to_robot
from nav.utils.data_utils import navigating_mask, navigating_ranges


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
