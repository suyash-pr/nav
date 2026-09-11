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
