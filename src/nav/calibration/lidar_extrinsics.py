"""Lidar mount calibration, per sensor suite.

The eight `lidar2d_*_*` channels on v4/v4.1 map to only four physical lidars -- each publishes
two half-scan channels under one `frame_id` (karter orchestrator/run_configs/topics.yaml:144-153;
e.g. `lidar2d_front_left_front` and `lidar2d_front_left_left` both carry `frame_id:
lidar_front_left`). So calibration stays keyed by frame_id, one entry per physical lidar.

v3 -> v4 is not a translation-only change: v3 carries a 180 degree flip about X plus a
45-degree-family Z rotation; v4 and v4_1 are [0, 0] at the front pair and [0, 180] at the back
pair. Reusing nav's old (v3) table against current (v4/v4.1) data mirrors and rotates every
scan into the BEV -- this was the actual bug in frame_transformer.py before this rewrite.

Values transcribed from karter's ansible/inventory/group_vars/{v3,v4,v4_1}/lidars.yml, pinned at
208a6fce9. A karter-gated test (tests/test_lidar_extrinsics_karter.py) asserts these still match
that source.
"""

from __future__ import annotations

LIDAR_CALIBRATION: dict[str, dict[str, dict]] = {
    "v3": {
        "lidar_front_left": {"translation": [0.428, 0.250, 0.173], "rotation": [180, 45], "euler_order": "XZ"},
        "lidar_front_right": {"translation": [0.428, -0.250, 0.173], "rotation": [180, 315], "euler_order": "XZ"},
        "lidar_back_left": {"translation": [-0.428, 0.250, 0.173], "rotation": [180, 135], "euler_order": "XZ"},
        "lidar_back_right": {"translation": [-0.428, -0.250, 0.173], "rotation": [180, 225], "euler_order": "XZ"},
    },
    "v4": {
        "lidar_front_left": {"translation": [0.452, 0.2547, 0.145], "rotation": [0, 0], "euler_order": "XZ"},
        "lidar_front_right": {"translation": [0.452, -0.2547, 0.145], "rotation": [0, 0], "euler_order": "XZ"},
        "lidar_back_left": {"translation": [-0.452, 0.2547, 0.145], "rotation": [0, 180], "euler_order": "XZ"},
        "lidar_back_right": {"translation": [-0.452, -0.2547, 0.145], "rotation": [0, 180], "euler_order": "XZ"},
    },
    "v4_1": {
        "lidar_front_left": {"translation": [0.448, 0.251, 0.1422], "rotation": [0, 0], "euler_order": "XZ"},
        "lidar_front_right": {"translation": [0.448, -0.251, 0.1422], "rotation": [0, 0], "euler_order": "XZ"},
        "lidar_back_left": {"translation": [-0.448, 0.251, 0.1422], "rotation": [0, 180], "euler_order": "XZ"},
        "lidar_back_right": {"translation": [-0.448, -0.251, 0.1422], "rotation": [0, 180], "euler_order": "XZ"},
    },
}

# The eight recorded channel dirs, mapped onto the four calibrated frame_ids they share -- see
# orchestrator/run_configs/topics.yaml:144-153.
LIDAR_CHANNELS_BY_FRAME_ID: dict[str, tuple[str, ...]] = {
    "lidar_front_left": ("lidar2d_front_left_front", "lidar2d_front_left_left"),
    "lidar_front_right": ("lidar2d_front_right_right", "lidar2d_front_right_front"),
    "lidar_back_left": ("lidar2d_back_left_left", "lidar2d_back_left_back"),
    "lidar_back_right": ("lidar2d_back_right_back", "lidar2d_back_right_right"),
}

DEFAULT_SENSOR_VERSION = "v4_1"


def calibration_for(sensor_version: str) -> dict[str, dict]:
    try:
        return LIDAR_CALIBRATION[sensor_version]
    except KeyError as exc:
        raise ValueError(
            f"unknown sensor_version {sensor_version!r}; known: {sorted(LIDAR_CALIBRATION)}"
        ) from exc
