"""Preprocesses one or more karter sessions (post-planning-cycle-instrumentation format) into
the memmapped cycle stores `nav.dataset.CycleDataset` reads.

Replaces `preprocess_lidar.py`: that script only handled lidar, because the old pipeline decoded
RGB/depth JPEGs/PNGs on the fly. RGB and depth are now H.265/FFV1 video groups, for which
per-sample random access is infeasible, so everything is decoded once here instead.

Usage:
    uv run python scripts/preprocess_session.py --session /path/to/session --out /path/to/store
    uv run python scripts/preprocess_session.py --sessions /path/to/root --out-root /path/to/stores
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from nav.calibration.lidar_extrinsics import DEFAULT_SENSOR_VERSION
from nav.preprocess import depth as depth_pre
from nav.preprocess import lidar as lidar_pre
from nav.preprocess import rgb as rgb_pre
from nav.preprocess.store import SessionManifest, write_array
from nav.session.actions import commanded_action, measured_velocity
from nav.session.cycles import load_cycle_table
from nav.session.layout import find_sessions

DEFAULT_RGB_CAMERAS = ("oakd_front", "oakd_back")
DEFAULT_DEPTH_CAMERAS = ("oakd_front", "oakd_back")
DEFAULT_RGB_SIZE = (253, 190)  # (W, H), matches the encoders' expected input
DEFAULT_DEPTH_SIZE = (160, 100)


def rgb_role(camera: str) -> str:
    return "rgb_" + camera.removeprefix("oakd_")


def depth_role(camera: str) -> str:
    return "depth_" + camera.removeprefix("oakd_")


def preprocess_session(
    session_dir: Path,
    out_dir: Path,
    *,
    sensor_version: str = DEFAULT_SENSOR_VERSION,
    rgb_cameras: tuple[str, ...] = DEFAULT_RGB_CAMERAS,
    depth_cameras: tuple[str, ...] = DEFAULT_DEPTH_CAMERAS,
    rgb_size: tuple[int, int] = DEFAULT_RGB_SIZE,
    depth_size: tuple[int, int] = DEFAULT_DEPTH_SIZE,
) -> SessionManifest:
    out_dir.mkdir(parents=True, exist_ok=True)

    table = load_cycle_table(session_dir)
    n_cycles = len(table)

    manifest = SessionManifest(
        sensor_version=sensor_version,
        session=session_dir.name,
        n_cycles=n_cycles,
        rgb_cameras=list(rgb_cameras),
        depth_cameras=list(depth_cameras),
        rgb_size=rgb_size,
        depth_size=depth_size,
        dropped=table.dropped,
    )

    write_array(out_dir, "cycles/cycle_id.npy", table.cycles["cycle_id"].to_numpy())
    write_array(out_dir, "cycles/run_id.npy", table.cycles["run_id"].to_numpy())
    write_array(out_dir, "cycles/tick_us.npy", table.cycles["tick_us"].to_numpy())
    write_array(out_dir, "cycles/action.npy", commanded_action(table.cycles))
    write_array(out_dir, "cycles/proprio.npy", measured_velocity(session_dir, table.cycles))

    for camera in rgb_cameras:
        role = rgb_role(camera)
        role_frames = table.roles.get(role)
        if role_frames is None:
            continue
        array, gathered = rgb_pre.build_camera_array(
            session_dir, camera, role_frames.capture_us, role_frames.present, rgb_size
        )
        write_array(out_dir, f"rgb/{camera}.npy", array)
        write_array(out_dir, f"index/rgb_{camera}.npy", gathered.cycle_row)
        manifest.rgb_join_strategy[camera] = gathered.strategy
        manifest.rgb_match_rate[camera] = gathered.match_rate
        manifest.rgb_decode_rate[camera] = gathered.decode_rate

    for camera in depth_cameras:
        role = depth_role(camera)
        role_frames = table.roles.get(role)
        if role_frames is None:
            continue
        array, gathered = depth_pre.build_camera_array(
            session_dir, camera, role_frames.capture_us, role_frames.present, depth_size
        )
        write_array(out_dir, f"depth/{camera}.npy", array)
        write_array(out_dir, f"index/depth_{camera}.npy", gathered.cycle_row)
        manifest.depth_join_strategy[camera] = gathered.strategy
        manifest.depth_match_rate[camera] = gathered.match_rate
        manifest.depth_decode_rate[camera] = gathered.decode_rate

    bev, bev_valid, lidar_diagnostics, lidar_roles_configured = lidar_pre.build_bev_array(
        session_dir, table.roles, n_cycles, sensor_version=sensor_version
    )
    write_array(out_dir, "lidar/bev.npy", bev)
    write_array(out_dir, "lidar/valid.npy", bev_valid)
    manifest.bev_shape = bev.shape[1:]
    manifest.lidar_match_rate = lidar_diagnostics
    manifest.lidar_roles_configured = lidar_roles_configured
    manifest.lidar_valid_rate = float(bev_valid.mean()) if n_cycles else 0.0
    if not lidar_roles_configured:
        print(
            f"  WARNING {session_dir}: sensor_frontier carries NO lidar roles at all -- every "
            f"cycle's lidar is invalid, so no CycleDataset pair can form from this store as-is "
            f"(lidar is currently a required modality for pair validity). This session's "
            f"sensor suite/deployment likely predates or is configured differently from the "
            f"v4/v4.1 8-channel lidar frontier this pipeline assumes.",
            file=sys.stderr,
        )

    manifest.write(out_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="a single session directory")
    parser.add_argument("--sessions", type=Path, help="a root directory containing many sessions")
    parser.add_argument("--out", type=Path, help="output store dir (with --session)")
    parser.add_argument("--out-root", type=Path, help="output root, one subdir per session (with --sessions)")
    parser.add_argument("--sensor-version", default=DEFAULT_SENSOR_VERSION, choices=["v3", "v4", "v4_1"])
    args = parser.parse_args()

    if bool(args.session) == bool(args.sessions):
        parser.error("pass exactly one of --session or --sessions")

    if args.session:
        if not args.out:
            parser.error("--out is required with --session")
        targets = [(args.session, args.out)]
    else:
        if not args.out_root:
            parser.error("--out-root is required with --sessions")
        targets = [(session, args.out_root / session.name) for session in find_sessions(args.sessions)]
        if not targets:
            parser.error(f"no session under {args.sessions} carries planning-cycle instrumentation")

    for session_dir, out_dir in targets:
        start = time.time()
        manifest = preprocess_session(session_dir, out_dir, sensor_version=args.sensor_version)
        print(
            f"{session_dir} -> {out_dir}: {manifest.n_cycles} cycles in {time.time() - start:.1f}s "
            f"(dropped: {manifest.dropped})"
        )


if __name__ == "__main__":
    main()
