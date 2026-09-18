"""Resolving a planning cycle's lidar frontier roles onto recorded scans.

Lidar is `lcm_protocol: direct` (karter orchestrator/run_configs/topics.yaml:142-153), so unlike
RGB its sensor_frontier role is the same stream that lands on disk -- there is no encoded/raw
split to reconcile. Matching is by `seq` first, since it is an exact per-message counter, with a
causal (never-forward) timestamp fallback for the rare row a UDP drop removed from one side.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MATCH_EPSILON_US = 50_000


def resolve_by_seq_or_asof(
    channel_seq: np.ndarray,
    channel_ts: np.ndarray,
    want_seq: np.ndarray,
    want_capture_us: np.ndarray,
    *,
    epsilon_us: int = DEFAULT_MATCH_EPSILON_US,
) -> np.ndarray:
    """Row index into the channel's arrays for each requested (seq, capture_us), -1 if unresolved.

    channel_seq/channel_ts must be sorted by write order (== time order for a rotated-chunk
    stream). want_seq of -1 (NO_FRAME_SEQ, or the role absent) skips straight to the timestamp
    fallback for that entry.
    """
    n = len(want_seq)
    out = np.full(n, -1, dtype=np.int64)
    if len(channel_seq) == 0:
        return out

    seq_lookup = {int(s): i for i, s in enumerate(channel_seq)}
    for i in range(n):
        if want_seq[i] >= 0:
            row = seq_lookup.get(int(want_seq[i]))
            if row is not None:
                out[i] = row
                continue
        if want_capture_us[i] < 0:
            continue
        out[i] = _causal_asof(channel_ts, int(want_capture_us[i]), epsilon_us)
    return out


def _causal_asof(channel_ts: np.ndarray, capture_us: int, epsilon_us: int) -> int:
    """Newest channel timestamp at or before capture_us, within epsilon_us. Never forward."""
    position = np.searchsorted(channel_ts, capture_us, side="right") - 1
    if position < 0:
        return -1
    if capture_us - channel_ts[position] > epsilon_us:
        return -1
    return int(position)
