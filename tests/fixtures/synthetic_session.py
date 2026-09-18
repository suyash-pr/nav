"""Builds a full session directory at the real on-disk contract: rotated ;-CSVs plus real
H.265 and FFV1 `.bin` + `.meta.json` groups, encoded in-process with PyAV. This is the format
pin -- it exercises the actual decode path (nav.session.decode), not a mock of it, so a change
to karter's sidecar schema or codec choice that this pipeline can't handle shows up here.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import av
import numpy as np

from nav.session.layout import channel_dir

RGB_ENCODING = "h265"
DEPTH_HORIZON_M = 5.0
DEPTH_STEP_MM = 16


def _encode_h265_group(frames: list[np.ndarray], width: int, height: int) -> bytes:
    encoder = av.CodecContext.create("hevc", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "yuv420p"
    # No B-frames and every frame independently keyed: decode order must equal encode order,
    # which is what lets frame_index in the sidecar address a specific decoded frame directly.
    encoder.options = {"bf": "0", "x265-params": "keyint=1:min-keyint=1"}

    packets = []
    for i, frame_rgb in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24").reformat(format="yuv420p")
        frame.pts = i
        packets.extend(bytes(p) for p in encoder.encode(frame))
    packets.extend(bytes(p) for p in encoder.encode(None))
    return b"".join(packets), [len(p) for p in packets]


def _encode_ffv1_group(frames: list[np.ndarray], width: int, height: int) -> tuple[bytes, list[int], bytes]:
    encoder = av.CodecContext.create("ffv1", "w")
    encoder.width = width
    encoder.height = height
    encoder.pix_fmt = "gray16le"
    encoder.gop_size = 1

    packets = []
    for i, units in enumerate(frames):
        frame = av.VideoFrame.from_ndarray(units, format="gray16le")
        frame.pts = i
        packets.extend(bytes(p) for p in encoder.encode(frame))
    packets.extend(bytes(p) for p in encoder.encode(None))
    extradata = bytes(encoder.extradata) if encoder.extradata else b""
    return b"".join(packets), [len(p) for p in packets], extradata


def write_rgb_group(
    session_dir: Path,
    camera: str,
    *,
    timestamps_us: list[int],
    width: int = 64,
    height: int = 64,
    start_seq: int = 0,
) -> None:
    """One H.265 group whose Nth frame is a flat RGB value of N * 10, so a test can identify
    which frame it decoded back by its mean pixel value (lossy, so compared with a tolerance)."""
    frames = [np.full((height, width, 3), i * 10, dtype=np.uint8) for i in range(len(timestamps_us))]
    blob, lengths = _encode_h265_group(frames, width, height)

    directory = channel_dir(session_dir, f"camera_{camera}_rgb/image_encoded")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamps_us[0]}_{start_seq}_camera_{camera}_rgb_optical"

    (directory / f"{stem}.bin").write_bytes(blob)
    meta = {
        "encoding": RGB_ENCODING,
        "width": width,
        "height": height,
        "fps": 10,
        "frame_id": f"camera_{camera}_rgb_optical",
        "frames": [
            {"type": "keyframe", "timestamp_us": ts, "seq": start_seq + i, "length": length}
            for i, (ts, length) in enumerate(zip(timestamps_us, lengths))
        ],
    }
    (directory / f"{stem}.meta.json").write_text(json.dumps(meta))


def write_depth_group(
    session_dir: Path,
    camera: str,
    *,
    timestamps_us: list[int],
    depth_mm_values: list[int],
    width: int = 32,
    height: int = 32,
    start_seq: int = 0,
    horizon_m: float = DEPTH_HORIZON_M,
    step_mm: int = DEPTH_STEP_MM,
) -> None:
    """One FFV1 group, quantised the way karter's driver quantises before publishing.

    `depth_mm_values[i]` is the flat depth (mm) every pixel of frame i is set to, pre-quantised
    to the (horizon_m, step_mm) policy so a test can recover the exact value after dequantising.
    """
    horizon_mm = round(horizon_m * 1000)
    horizon_units = round(horizon_mm / step_mm)
    far_sentinel = horizon_units + max(4, horizon_units // 8)

    units_frames = []
    for depth_mm in depth_mm_values:
        unit = far_sentinel if depth_mm > horizon_mm else round(depth_mm / step_mm)
        units_frames.append(np.full((height, width), unit, dtype=np.uint16))

    blob, lengths, extradata = _encode_ffv1_group(units_frames, width, height)

    directory = channel_dir(session_dir, f"camera_{camera}_mono_l/depth_image")
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{timestamps_us[0]}_{start_seq}_camera_{camera}_mono_l_optical_depth"

    (directory / f"{stem}.bin").write_bytes(blob)
    meta = {
        "codec": "ffv1",
        "frame_id": f"camera_{camera}_mono_l_optical",
        "width": width,
        "height": height,
        "horizon_m": horizon_m,
        "step_mm": step_mm,
        "extradata": base64.b64encode(extradata).decode("ascii"),
        "frames": [
            {"timestamp_us": ts, "seq": start_seq + i, "length": length}
            for i, (ts, length) in enumerate(zip(timestamps_us, lengths))
        ],
    }
    (directory / f"{stem}.meta.json").write_text(json.dumps(meta))


def write_lidar_csv(session_dir: Path, channel: str, *, frame_id: str, rows: list[dict]) -> None:
    """rows: [{"timestamp_us": ..., "seq": ..., "x": [..], "y": [..]}] -- z is written as zeros."""
    directory = session_dir / "data" / channel / "points"
    directory.mkdir(parents=True, exist_ok=True)
    lines = ["timestamp_us;seq;frame_id;x;y;z"]
    for row in rows:
        x = ",".join(str(v) for v in row["x"])
        y = ",".join(str(v) for v in row["y"])
        z = ",".join("0.0" for _ in row["x"])
        lines.append(f"{row['timestamp_us']};{row['seq']};{frame_id};{x};{y};{z}")
    (directory / f"{channel.replace('/', '_')}_points_{'0' * 20}.csv").write_text("\n".join(lines) + "\n")
