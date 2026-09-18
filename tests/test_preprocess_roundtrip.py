"""End-to-end: a synthetic session at the real on-disk contract -> preprocess_session.py ->
CycleDataset. This is the regression test for the whole rewrite: every stage the old pipeline
answered by guessing a latency (data_module.py:19-21, before this rewrite) is here answered by
an actual recorded join, and this test is what would catch a break in that chain.
"""

import numpy as np
from fixtures.session_csv import (
    write_command,
    write_planning_cycle,
    write_sensor_frontier,
)
from fixtures.synthetic_session import (
    write_depth_group,
    write_lidar_csv,
    write_rgb_group,
)

from nav.dataset import CycleDataset
from scripts.preprocess_session import preprocess_session

TICKS_US = [1_000_000, 1_100_000, 1_200_000, 1_300_000]
AGE_US = 20_000
CAPTURE_US = [t - AGE_US for t in TICKS_US]
ROLES = ["rgb_front", "depth_front", "lidar_front_left_front"]


def _write_synthetic_session(session_dir):
    cycles = [{"timestamp_us": t, "cycle_id": i + 1, "velocity_seq": i} for i, t in enumerate(TICKS_US)]
    frontier = [
        {
            "timestamp_us": t,
            "cycle_id": i + 1,
            **{f"seq.{role}": i for role in ROLES},
            **{f"age_us.{role}": AGE_US for role in ROLES},
        }
        for i, t in enumerate(TICKS_US)
    ]
    commands = [
        {
            "timestamp_us": t,
            "cycle_id": i + 1,
            "desired_linear_velocity": 0.5 + 0.1 * i,
            "desired_angular_velocity": 0.0,
        }
        for i, t in enumerate(TICKS_US)
    ]
    write_planning_cycle(session_dir, cycles)
    write_sensor_frontier(session_dir, frontier, ROLES)
    write_command(session_dir, commands)

    write_rgb_group(session_dir, "oakd_front", timestamps_us=CAPTURE_US)
    write_depth_group(session_dir, "oakd_front", timestamps_us=CAPTURE_US, depth_mm_values=[500, 1000, 1500, 2000])

    write_lidar_csv(
        session_dir,
        "lidar2d_front_left_front",
        frame_id="lidar_front_left",
        rows=[{"timestamp_us": t, "seq": i, "x": [1.0], "y": [0.5]} for i, t in enumerate(CAPTURE_US)],
    )


def test_preprocess_and_dataset_roundtrip(tmp_path):
    session_dir = tmp_path / "session"
    out_dir = tmp_path / "store"
    _write_synthetic_session(session_dir)

    manifest = preprocess_session(
        session_dir,
        out_dir,
        rgb_cameras=("oakd_front",),
        depth_cameras=("oakd_front",),
    )

    assert manifest.n_cycles == 4
    # The RGB frames were written at exactly the capture timestamps the frontier implies, so
    # the join must resolve 'exact' -- this is the karter e926a211f assumption, verified here
    # for the synthetic case rather than assumed.
    assert manifest.rgb_join_strategy["oakd_front"] == "exact"
    assert manifest.rgb_match_rate["oakd_front"] == 1.0
    assert manifest.depth_join_strategy["oakd_front"] == "exact"
    assert manifest.lidar_match_rate["lidar_front_left_front"] == 1.0

    dataset = CycleDataset(out_dir, rgb_cameras=("oakd_front",), depth_cameras=("oakd_front",))
    # step_cycles=2 over 4 cycles -> pairs (0,2), (1,3)
    assert len(dataset) == 2

    sample = dataset[0]
    assert sample["rgb"].shape == (2, 1, 3, 190, 253)
    assert sample["depth"].shape == (2, 1, 2, 100, 160)
    assert sample["bev"].shape == (2, 1, 120, 120)
    assert sample["action"].shape == (2,)
    assert sample["proprio"].shape == (2, 2)

    # cycle 0's RGB frame was encoded as a flat value of 0*10=0, cycle 2's as 2*10=20 (out of
    # 255, scaled to [0,1] at decode) -- lossy H.265, so compared with a generous tolerance.
    np.testing.assert_allclose(sample["rgb"][0].mean().item(), 0.0, atol=0.05)
    np.testing.assert_allclose(sample["rgb"][1].mean().item(), 20 / 255, atol=0.05)

    # depth is dequantised exactly (FFV1 is lossless): cycle 0 -> 500mm, cycle 2 -> 1500mm.
    depth_channel = sample["depth"][:, 0, 0]  # (2, H, W) normalized depth/DEPTH_MAX_M
    np.testing.assert_allclose(depth_channel[0].mean().item(), 0.5 / 5.0, atol=1e-3)
    np.testing.assert_allclose(depth_channel[1].mean().item(), 1.5 / 5.0, atol=1e-3)

    # the lidar point (1.0, 0.5) in the front-left lidar's frame lands inside the BEV, so the
    # occupancy grid must not be empty for either step.
    assert sample["bev"][0].sum() > 0
    assert sample["bev"][1].sum() > 0

    # action at pair (0, 2) must be exactly cycle 0's command -- no averaging, no windowing.
    np.testing.assert_allclose(sample["action"].numpy(), [0.5, 0.0], atol=1e-5)
