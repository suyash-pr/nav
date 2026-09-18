"""Decoding recorded video groups, and undoing the recorder's depth quantisation.

Mirrors karter/src/python/utils/mono16_codec.py and the LUTs in
karter/src/python/dtypes/depth_image.py (pinned at 208a6fce9). Reimplemented rather than
imported because karter's file-reading entry points sit on classes whose modules pull in `lcm`,
`foxglove` and `cv2` -- karter itself duplicates rather than imports across this seam for the
same reason (depth_image.py:40).

Only the offline preprocessing pass calls any of this; the training loop reads memmaps.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import av
import numpy as np

CODEC_FFV1 = "ffv1"
FFV1_PIX_FMT = "gray16le"
RGB_PIX_FMT = "rgb24"

# The 10-bit ceiling the depth quantiser targets, and the stereo blind zone below which a
# decoded value is codec ringing around an invalid pixel rather than a return.
MAX_TEN_BIT = (1 << 10) - 1
MIN_PLAUSIBLE_DEPTH_MM = 200


def codec_name(encoding: str) -> str:
    """Map a sidecar's encoding name onto the decoder PyAV knows it by."""
    name = (encoding or "").lower()
    if name in ("h265", "hevc"):
        return "hevc"
    if name in ("h264", "avc", "avc1"):
        return "h264"
    return name or "h264"


def decode_group(
    blob: bytes,
    lengths: list[int],
    *,
    codec: str,
    width: int,
    height: int,
    pix_fmt: str,
    extradata: bytes | None = None,
) -> dict[int, np.ndarray]:
    """Decode one group's concatenated packets, sliced apart by the sidecar's recorded lengths.

    Returns {sidecar frame position: decoded array} -- a dict, not a list, and a position
    missing from it is a confirmed, permanent decode failure, not a positional artifact.

    This matters on real recordings: a linearly-chained GOP (one keyframe, every P-frame
    referencing its predecessor) has no recovery point, so a single frame the recorder never
    received (a gap in the sidecar's own `seq` column -- LCM is UDP) poisons every frame chained
    to it until the group's next keyframe. Measured on one real session's RGB channel: ~38% of
    sidecar-declared frames were unusable this way (scripts/diagnose_recorder_frame_loss.py).
    Decoding without position tracking would either crash (IndexError against a shorter list) or
    -- worse -- silently misattribute a decoded frame to the wrong sidecar position, since the
    decoder's own internal buffering delay means output order is not simply input order minus
    the missing entries. Tagging each fed packet with `pts = its sidecar position` is what lets
    a later-emitted frame still be matched back to the right position after that delay.

    A group truncated mid-packet by a killed recorder (a byte-level tear, distinct from the
    reference-chain gaps above) still yields every whole packet before the tear.
    """
    decoder = av.CodecContext.create(codec, "r")
    decoder.width = width
    decoder.height = height
    decoder.pix_fmt = pix_fmt
    if extradata:
        decoder.extradata = extradata

    out: dict[int, np.ndarray] = {}
    offset = 0
    for position, length in enumerate(lengths):
        chunk = blob[offset : offset + length]
        offset += length
        if len(chunk) < length:
            break
        packet = av.Packet(chunk)
        packet.pts = position
        packet.dts = position
        try:
            for frame in decoder.decode(packet):
                out[frame.pts] = _to_array(frame, pix_fmt)
        except av.error.FFmpegError:
            continue  # this packet's position simply won't appear in the output

    try:
        for frame in decoder.decode(None):  # flush frames the decoder was still holding
            out[frame.pts] = _to_array(frame, pix_fmt)
    except av.error.FFmpegError:
        pass
    return out


def _to_array(frame: av.VideoFrame, pix_fmt: str) -> np.ndarray:
    if pix_fmt == FFV1_PIX_FMT:
        return np.ascontiguousarray(frame.to_ndarray())
    return np.ascontiguousarray(frame.to_ndarray(format=RGB_PIX_FMT))


def decode_rgb_group(bin_path: Path, meta: dict) -> dict[int, np.ndarray]:
    """Decode an `image_encoded` group. {sidecar position: (H, W, 3) uint8 RGB frame}, missing
    a position where the frame is permanently undecodable -- see `decode_group`."""
    lengths = [int(entry["length"]) for entry in meta.get("frames", [])]
    return decode_group(
        Path(bin_path).read_bytes(),
        lengths,
        codec=codec_name(meta.get("encoding", "h265")),
        width=int(meta["width"]),
        height=int(meta["height"]),
        pix_fmt=RGB_PIX_FMT,
    )


def decode_depth_group(bin_path: Path, meta: dict, extradata: bytes | None) -> dict[int, np.ndarray]:
    """Decode a `depth_image` group. {sidecar position: (H, W) uint16 frame, still quantised},
    missing a position where the frame is permanently undecodable -- see `decode_group`. FFV1
    is lossless and (per karter) every packet is independently keyed, so in practice this should
    only lose frames to a byte-level tear, not to the RGB stream's reference-chain cascade."""
    lengths = [int(entry["length"]) for entry in meta.get("frames", [])]
    return decode_group(
        Path(bin_path).read_bytes(),
        lengths,
        codec=meta.get("codec", CODEC_FFV1),
        width=int(meta["width"]),
        height=int(meta["height"]),
        pix_fmt=FFV1_PIX_FMT,
        extradata=extradata,
    )


def invalid_threshold(step_mm: int) -> int:
    return max(1, MIN_PLAUSIBLE_DEPTH_MM // step_mm)


@cache
def _dequantise_lut(step_mm: int, threshold: int) -> np.ndarray:
    """Every 10-bit unit's millimetre depth, 0 below the guard band."""
    units = np.arange(MAX_TEN_BIT + 1, dtype=np.int64)
    return np.where(units < threshold, 0, units * step_mm).clip(0, 65535).astype(np.uint16)


def dequantise_depth(units: np.ndarray, step_mm: int) -> np.ndarray:
    """Quantised units -> millimetre depth, 0 where there was no return.

    Values below the guard band collapse to 0 rather than to a very near depth: the gap the
    quantiser opens between 0 and the nearest real return is exactly what stops codec ringing
    around an invalid pixel from reading as an obstacle at the sensor.

    Beyond-horizon pixels carry the quantiser's far sentinel, which dequantises to a distance
    past the horizon. That is intended -- "far" stays large, and the horizon is recorded in the
    sidecar for a caller that wants to mask it.
    """
    if units.dtype != np.uint16:
        raise TypeError(f"expected a uint16 array, got {units.dtype}")
    return _dequantise_lut(int(step_mm), invalid_threshold(int(step_mm)))[np.minimum(units, MAX_TEN_BIT)]
