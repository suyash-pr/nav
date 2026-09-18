"""Deduplicating which encoded frames a session's cycles actually need, before decoding any.

Many cycles legitimately reuse the same frame (navigation ticks faster than a camera in some
configurations, and DataSyncManager uses get_latest() rather than take_latest()), so decoding
once per cycle would decode the same group repeatedly. This collapses a role's per-cycle capture
timestamps to the distinct set actually referenced, resolves each against the recorded frame
index, and hands back a per-cycle row into a *compacted* output array holding only the frames
that both are referenced and resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nav.session.encoded_index import (
    DEFAULT_MATCH_EPSILON_US,
    GroupIndex,
    resolve_frames,
)


@dataclass(frozen=True)
class FrameGatherResult:
    cycle_row: np.ndarray  # (n_cycles,) int64, -1 where the role is absent, unresolved, or undecodable
    frames: pd.DataFrame  # the frames gather_and_resolve chose, in their ORIGINAL compacted order
    # (i.e. not reindexed by a later decode-failure compaction -- kept for diagnostics/debugging,
    # not for indexing into a caller's decoded array once decode_rate < 1.0).
    strategy: str
    match_rate: float
    decode_rate: float = 1.0  # fraction of `frames` that actually decoded; see compact_after_decode_failures


def compact_after_decode_failures(
    cycle_row: np.ndarray, n_frames: int, decoded: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """A frame resolving to a *byte range that exists* is not the same as it being decodable --
    a broken H.265 reference chain can leave a sidecar-declared frame permanently undecodable
    (see nav.session.decode.decode_group). `gather_and_resolve`'s compacted row order assumed
    every gathered frame would decode; this re-compacts around the ones that actually did, and
    remaps `cycle_row` so a cycle pointing at an undecodable frame becomes -1 (absent) rather
    than an out-of-bounds index.

    `decoded` is keyed by position *within `gathered.frames`* (0..n_frames-1), matching what a
    caller gets back after concatenating each bin_path group's `decode_*_group` dict -- see
    rgb.py/depth.py's `build_camera_array` for how that key is constructed.

    Returns (new_cycle_row, decoded_order), where `decoded_order` lists the original compacted
    positions in their new output order, for a caller to gather images by index.
    """
    decoded_order = np.array(sorted(decoded.keys()), dtype=np.int64)
    if n_frames == 0:
        return np.full(len(cycle_row), -1, dtype=np.int64), decoded_order

    new_position = np.full(n_frames, -1, dtype=np.int64)
    new_position[decoded_order] = np.arange(len(decoded_order))

    new_cycle_row = np.where(cycle_row >= 0, new_position[np.clip(cycle_row, 0, n_frames - 1)], -1)
    return new_cycle_row, decoded_order


def gather_and_resolve(
    capture_us: np.ndarray,
    present: np.ndarray,
    index: GroupIndex,
    *,
    epsilon_us: int = DEFAULT_MATCH_EPSILON_US,
) -> FrameGatherResult:
    n = len(capture_us)
    cycle_row = np.full(n, -1, dtype=np.int64)
    present_idx = np.flatnonzero(present)

    if len(present_idx) == 0 or len(index) == 0:
        return FrameGatherResult(cycle_row=cycle_row, frames=index.frames.iloc[0:0], strategy="approximate", match_rate=0.0)

    unique_us, inverse = np.unique(capture_us[present_idx], return_inverse=True)
    resolution = resolve_frames(index, unique_us, epsilon_us=epsilon_us)

    resolved = resolution.row >= 0
    compacted_position = np.full(len(unique_us), -1, dtype=np.int64)
    compacted_position[resolved] = np.cumsum(resolved)[resolved] - 1

    cycle_row[present_idx] = compacted_position[inverse]

    frames = index.frames.iloc[resolution.row[resolved]].reset_index(drop=True)
    return FrameGatherResult(cycle_row=cycle_row, frames=frames, strategy=resolution.strategy, match_rate=resolution.match_rate)
