import numpy as np
import pandas as pd


def navigating_ranges(nav_cmd: pd.DataFrame) -> np.ndarray:
    """Finds the (start, end) timestamp of each stretch where navigation_status == "navigating"."""
    nav_cmd = nav_cmd.sort_values("timestamp_us")
    is_navigating = nav_cmd["navigation_status"] == "navigating"

    ranges = []
    start = None
    for timestamp_us, navigating in zip(nav_cmd["timestamp_us"], is_navigating):
        if navigating and start is None:
            start = timestamp_us
        elif not navigating and start is not None:
            ranges.append((start, timestamp_us))
            start = None
    if start is not None:
        ranges.append((start, np.iinfo(np.int64).max))

    return np.array(ranges, dtype=np.int64).reshape(-1, 2)


def navigating_mask(timestamps: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Returns a bool mask of which timestamps fall inside a navigating range."""
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, end in ranges:
        mask |= (timestamps >= start) & (timestamps < end)
    return mask


def stretch_ids(timestamps: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Maps each timestamp to the index (into ranges, chronological) of the navigating stretch it falls in.

    Every timestamp must already fall inside some range (e.g. have passed navigating_mask).
    """
    return np.searchsorted(ranges[:, 0], timestamps, side="right") - 1


def nearest_within(sensor_ts: np.ndarray, query_us: np.ndarray, tolerance_us: int) -> tuple[np.ndarray, np.ndarray]:
    """For each query timestamp, finds the nearest sensor timestamp and whether it's within tolerance_us."""
    idx = np.searchsorted(sensor_ts, query_us)
    idx = np.clip(idx, 1, len(sensor_ts) - 1)
    left, right = idx - 1, idx
    use_left = np.abs(sensor_ts[left] - query_us) <= np.abs(sensor_ts[right] - query_us)
    nearest = np.where(use_left, left, right)
    ok = np.abs(sensor_ts[nearest] - query_us) <= tolerance_us
    return nearest, ok


def anchor_times(ranges: np.ndarray, end_us: int, step_us: int = 200_000, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Builds a synthetic step_us grid seeded at each stretch start, clamped to end_us.

    Returns (t_us, stretch_id): t_us is (n, horizon + 1) int64, stretch_id is (n,) int64.
    """
    bases = []
    stretch_id = []
    for i, (start, end) in enumerate(ranges):
        end = min(end, end_us)
        n_steps = int((end - start) // step_us) - horizon
        if n_steps < 1:
            continue
        bases.append(start + np.arange(n_steps, dtype=np.int64) * step_us)
        stretch_id.append(np.full(n_steps, i, dtype=np.int64))

    if not bases:
        return np.zeros((0, horizon + 1), dtype=np.int64), np.zeros(0, dtype=np.int64)

    base = np.concatenate(bases)
    offsets = np.arange(horizon + 1, dtype=np.int64) * step_us
    t_us = base[:, None] + offsets[None, :]
    return t_us, np.concatenate(stretch_id)
