"""Decoding the H.265 groups a session's cycles reference into a training-resolution memmap.

RGB is recorded to `camera_oakd_*_rgb/image_encoded/*.bin` + `.meta.json`; the sensor frontier's
`rgb_*` roles record arrival on `image_raw`, which is *not* itself recorded (the recorder drops
it as a duplicate of the H.265 stream). So the join is capture time -> the encoded stream's own
sidecar index, via `nav.session.encoded_index` -- see that module's docstring for why this rests
on an assumption karter flags as unverified on hardware, and how a failure of that assumption is
detected and reported rather than silently producing a misaligned dataset.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from nav.preprocess.frame_gather import FrameGatherResult, compact_after_decode_failures, gather_and_resolve
from nav.session import layout
from nav.session.decode import decode_rgb_group
from nav.session.encoded_index import index_groups

IMAGE_ENCODED_SUBTOPIC = "image_encoded"


def channel_dir_for_camera(session_dir: Path, camera: str) -> Path:
    return layout.channel_dir(session_dir, f"camera_{camera}_rgb/{IMAGE_ENCODED_SUBTOPIC}")


def build_camera_array(
    session_dir: Path, camera: str, capture_us: np.ndarray, present: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, FrameGatherResult]:
    """Decode every RGB frame `camera`'s cycles reference, resized to `size` (W, H).

    A frame `gather_and_resolve` matched to a byte range is not guaranteed to actually decode --
    real recordings have dropped a frame the recorder never received (a `seq` gap, LCM is UDP),
    which poisons every frame chained to it in the group's linear GOP until the next keyframe.
    Measured on one real session: ~38% of sidecar-declared frames on one channel were unusable
    this way (scripts/diagnose_recorder_frame_loss.py). Frames that fail are dropped, not
    crashed on, and the survival rate is reported back as `decode_rate` rather than assumed.

    Returns ((n_decoded, H, W, 3) uint8, the gather result -- callers write `gather_result.
    cycle_row` into the store's per-cycle index so a memmap row can be found back from a cycle).
    """
    index = index_groups(channel_dir_for_camera(session_dir, camera))
    gathered = gather_and_resolve(capture_us, present, index)

    width, height = size
    n = len(gathered.frames)
    buffer = np.zeros((n, height, width, 3), dtype=np.uint8)
    decoded_positions: dict[int, None] = {}
    for bin_path, group in gathered.frames.groupby("bin_path", sort=False):
        decoded = decode_rgb_group(bin_path, index.meta[bin_path])
        for row_position, frame_index in zip(group.index, group["frame_index"]):
            frame = decoded.get(frame_index)
            if frame is None:
                continue
            if frame.shape[1::-1] != (width, height):
                frame = np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR))
            buffer[row_position] = frame
            decoded_positions[row_position] = None

    new_cycle_row, order = compact_after_decode_failures(gathered.cycle_row, n, decoded_positions)
    out = buffer[order]
    decode_rate = len(order) / n if n else 1.0
    result = replace(gathered, cycle_row=new_cycle_row, decode_rate=decode_rate)
    return out, result
