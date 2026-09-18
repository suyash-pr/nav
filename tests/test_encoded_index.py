from pathlib import Path

import numpy as np
import pandas as pd

from nav.session.encoded_index import (
    JOIN_APPROXIMATE,
    JOIN_CONSTANT_OFFSET,
    JOIN_EXACT,
    GroupIndex,
    resolve_frames,
)


def _index(timestamps_us: list[int]) -> GroupIndex:
    bin_path = Path("group_0.bin")
    frames = pd.DataFrame(
        {
            "timestamp_us": pd.Series(timestamps_us, dtype="int64"),
            "seq": pd.Series(range(len(timestamps_us)), dtype="int64"),
            "bin_path": [bin_path] * len(timestamps_us),
            "frame_index": pd.Series(range(len(timestamps_us)), dtype="int64"),
            "length": pd.Series([100] * len(timestamps_us), dtype="int64"),
        }
    )
    return GroupIndex(frames=frames, meta={bin_path: {"width": 4, "height": 4, "encoding": "h265", "fps": 10}})


def test_resolve_frames_exact_when_every_capture_us_is_indexed():
    index = _index([0, 100_000, 200_000, 300_000])
    capture_us = np.array([100_000, 300_000, 0])

    resolution = resolve_frames(index, capture_us)

    assert resolution.strategy == JOIN_EXACT
    assert resolution.match_rate == 1.0
    assert resolution.row.tolist() == [1, 3, 0]


def test_resolve_frames_constant_offset_when_encoder_shifts_timestamps():
    # This is exactly karter e926a211f's unverified-on-hardware scenario: the encoder stamps a
    # frame with a timestamp that differs from the raw stream's by a fixed amount.
    offset = 3_000
    index = _index([offset, 100_000 + offset, 200_000 + offset])
    capture_us = np.array([0, 100_000, 200_000])

    resolution = resolve_frames(index, capture_us)

    assert resolution.strategy == JOIN_CONSTANT_OFFSET
    assert resolution.offset_us == offset
    assert resolution.match_rate == 1.0
    assert resolution.row.tolist() == [0, 1, 2]


def test_resolve_frames_approximate_falls_back_to_causal_asof_never_forward():
    # No exact or constant-offset match exists; the fallback must pick the newest frame at or
    # before the capture time, never one recorded after it -- a forward match is the exact
    # leakage defect this rewrite exists to remove.
    index = _index([0, 47_000, 103_000, 251_000])
    capture_us = np.array([50_000])  # nearest neighbour would be 47_000 OR 103_000; asof is 47_000 only

    resolution = resolve_frames(index, capture_us, epsilon_us=50_000)

    assert resolution.strategy == JOIN_APPROXIMATE
    assert resolution.row.tolist() == [1]  # the 47_000 frame, not the closer-in-absolute-terms 103_000


def test_resolve_frames_approximate_respects_epsilon():
    index = _index([0])
    capture_us = np.array([200_000])  # far outside any reasonable epsilon

    resolution = resolve_frames(index, capture_us, epsilon_us=50_000)

    assert resolution.row.tolist() == [-1]
    assert resolution.match_rate == 0.0


def test_resolve_frames_unwanted_entries_are_never_resolved():
    index = _index([0, 100_000])
    capture_us = np.array([-1, 100_000])  # -1 means "role absent for this cycle"

    resolution = resolve_frames(index, capture_us)

    assert resolution.row.tolist() == [-1, 1]


def test_resolve_frames_empty_index():
    index = _index([])
    resolution = resolve_frames(index, np.array([0, 100]))
    assert resolution.row.tolist() == [-1, -1]
    assert resolution.match_rate == 0.0
