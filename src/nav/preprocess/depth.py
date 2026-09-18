"""Decoding the FFV1 depth groups a session's cycles reference into a training-resolution memmap.

Depth's frontier roles point at `camera_oakd_*_mono_l/depth_image` -- **not**
`camera_*_mono_r`, which is what nav's pre-rewrite `depth_loader.list_depth_frames` assumed and
is actually the right IR camera of the stereo pair, never validated because that data was never
even unzipped. `_mono_l/depth_image` is `lcm_protocol: meta`, and unlike RGB there is no encoded/raw split to
reconcile -- the frontier's depth role and the recorded stream are the same channel, so the
timestamp-based join in `nav.session.encoded_index` (shared with RGB) is expected to resolve
`exact` reliably, without needing the constant-offset or as-of fallbacks RGB may require.

Depth is recorded quantised (see the topic's `depth_quantisation: {horizon_m, step_mm}` in
karter's topics.yaml) -- the driver quantises *before* publishing, so what's on disk is exactly
what the robot planned with. Dequantised here back to millimetres so the stored array is
inspectable and the model-facing normalisation (metres / DEPTH_MAX_M) stays a `__getitem__`
concern rather than baked into the store.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from nav.preprocess.frame_gather import FrameGatherResult, compact_after_decode_failures, gather_and_resolve
from nav.session import layout
from nav.session.decode import decode_depth_group as _decode_depth_group
from nav.session.decode import dequantise_depth
from nav.session.encoded_index import index_groups

DEPTH_IMAGE_SUBTOPIC = "depth_image"


def channel_dir_for_camera(session_dir: Path, camera: str) -> Path:
    return layout.channel_dir(session_dir, f"camera_{camera}_mono_l/{DEPTH_IMAGE_SUBTOPIC}")


def build_camera_array(
    session_dir: Path, camera: str, capture_us: np.ndarray, present: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, FrameGatherResult]:
    """Decode every depth frame `camera`'s cycles reference, dequantised to mm and resized.

    See `nav.preprocess.rgb.build_camera_array` for why a frame `gather_and_resolve` matched to
    a byte range is not guaranteed to decode, and why failures here are dropped and reported via
    `decode_rate` rather than crashing. FFV1 is lossless and every packet independently keyed, so
    in practice this should only lose frames to a byte-level tear, not RGB's reference cascade --
    `decode_rate` is what would tell you if that assumption stops holding on a given session.

    Returns ((n_decoded, H, W) uint16 millimetres, 0 = no return, the gather result).
    """
    index = index_groups(channel_dir_for_camera(session_dir, camera))
    gathered = gather_and_resolve(capture_us, present, index)

    width, height = size
    n = len(gathered.frames)
    buffer = np.zeros((n, height, width), dtype=np.uint16)
    decoded_positions: dict[int, None] = {}
    for bin_path, group in gathered.frames.groupby("bin_path", sort=False):
        meta = index.meta[bin_path]
        quantisation = index.quantisation(bin_path)
        decoded = _decode_depth_group(bin_path, meta, index.extradata(bin_path))
        for row_position, frame_index in zip(group.index, group["frame_index"]):
            units = decoded.get(frame_index)
            if units is None:
                continue
            depth_mm = dequantise_depth(units, quantisation[1]) if quantisation else units
            if depth_mm.shape[1::-1] != (width, height):
                depth_mm = np.asarray(
                    Image.fromarray(depth_mm).resize((width, height), Image.Resampling.NEAREST)
                )
            buffer[row_position] = depth_mm
            decoded_positions[row_position] = None

    new_cycle_row, order = compact_after_decode_failures(gathered.cycle_row, n, decoded_positions)
    out = buffer[order]
    decode_rate = len(order) / n if n else 1.0
    result = replace(gathered, cycle_row=new_cycle_row, decode_rate=decode_rate)
    return out, result
