"""Merging a session's per-cycle lidar frontier into robot-frame occupancy BEV grids.

v4/v4.1 record eight half-scan channels (karter orchestrator/run_configs/topics.yaml:142-153),
each independently listed in the sensor frontier (navigation.yaml:52-63). This replaces the old
pipeline's `merge_scans`, which guessed a 60 ms tolerance and picked one arbitrary "reference"
lidar's clock for every sample's timestamp -- here, the *frontier itself* names exactly which
scan on each channel navigation held for a given cycle, so there is nothing left to guess.

Unlike RGB/depth, lidar's per-message payload is small (a CSV row of a few hundred points), so
there is no group-decode cost to amortise here; each cycle's BEV is built directly rather than
through the frame_gather dedup path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav.frame_transformer import LidarFrameTransformer
from nav.lidar_loader import (
    BEV_EXTENT_M,
    BEV_RESOLUTION_M,
    load_lidar_channel,
    rasterize_bev,
)
from nav.session import layout
from nav.session.cycles import RoleFrames
from nav.session.lidar_index import resolve_by_seq_or_asof

# Frontier role -> recorded topic (navigation.yaml sensor_frontier_topics, v4/v4.1 block).
LIDAR_ROLE_TOPICS: dict[str, str] = {
    "lidar_front_left_front": "lidar2d_front_left_front/points",
    "lidar_front_left_left": "lidar2d_front_left_left/points",
    "lidar_front_right_right": "lidar2d_front_right_right/points",
    "lidar_front_right_front": "lidar2d_front_right_front/points",
    "lidar_back_left_left": "lidar2d_back_left_left/points",
    "lidar_back_left_back": "lidar2d_back_left_back/points",
    "lidar_back_right_back": "lidar2d_back_right_back/points",
    "lidar_back_right_right": "lidar2d_back_right_right/points",
}


def build_bev_array(
    session_dir: Path,
    cycle_roles: dict[str, RoleFrames],
    n_cycles: int,
    *,
    sensor_version: str,
    extent_m: float = BEV_EXTENT_M,
    resolution_m: float = BEV_RESOLUTION_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], list[str]]:
    """One merged occupancy BEV per cycle.

    Returns (array, valid -- True where at least one lidar role resolved for that cycle, so an
    all-zero grid from "no obstacle" is distinguishable from "no scan resolved" --, per-role
    match-rate diagnostics, and the subset of LIDAR_ROLE_TOPICS this session's frontier actually
    configured. An empty last element means this session's sensor_frontier carries no lidar
    roles at all -- e.g. an older/differently-configured deployment -- and `valid` will
    therefore be all-False: every cycle in the store, not just the ones with a bad scan.
    """
    transformer = LidarFrameTransformer(sensor_version)
    size = round(2 * extent_m / resolution_m)
    out = np.zeros((n_cycles, size, size), dtype=np.uint8)
    valid = np.zeros(n_cycles, dtype=bool)

    diagnostics: dict[str, float] = {}
    per_role_points: list[list[np.ndarray]] = []
    per_role_index: list[np.ndarray] = []
    configured_roles: list[str] = []

    for role, topic in LIDAR_ROLE_TOPICS.items():
        role_frames = cycle_roles.get(role)
        if role_frames is None:
            continue  # not configured in this session's frontier (e.g. an older sensor suite,
            # or SENSOR_VERSION gated lidar out of sensor_frontier_topics entirely for this
            # deployment -- see build_bev_array's `configured_roles` return value, which is []
            # in exactly that case and means every cycle's lidar_valid will be False)
        configured_roles.append(role)

        chunk_paths = layout.find_topic_csvs(session_dir, topic)
        channel_ts, channel_seq, channel_points = load_lidar_channel(chunk_paths, transformer)

        row = resolve_by_seq_or_asof(channel_seq, channel_ts, role_frames.seq, role_frames.capture_us)
        row = np.where(role_frames.present, row, -1)

        diagnostics[role] = float((row >= 0).mean()) if n_cycles else 0.0
        per_role_points.append(channel_points)
        per_role_index.append(row)

    for cycle_i in range(n_cycles):
        pieces = [
            points[index[cycle_i]] for points, index in zip(per_role_points, per_role_index) if index[cycle_i] >= 0
        ]
        if not pieces:
            continue
        valid[cycle_i] = True
        merged = np.concatenate(pieces, axis=0)
        out[cycle_i] = rasterize_bev(merged, extent_m=extent_m, resolution_m=resolution_m)

    return out, valid, diagnostics, configured_roles
