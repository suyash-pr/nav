import numpy as np

from nav.frame_transformer import LidarFrameTransformer
from nav.lidar_loader import load_lidar_csv


def test_load_lidar_csv_drops_zero_zero_dropout_points(tmp_path):
    """(0.0, 0.0) in the raw sensor frame is the lidar's invalid-return sentinel.

    It must be dropped before the frame transform -- otherwise every dropout point lands on
    the lidar's mount point in the robot frame and is mistaken for a real obstacle there.
    """
    csv_path = tmp_path / "lidar2d_front_left_points.csv"
    csv_path.write_text(
        "timestamp_us;seq;frame_id;x;y;z\n"
        "100;0;lidar_front_left;0.0,0.0,1.0,0.0;0.0,0.0,0.0,2.0;0.0,0.0,0.0,0.0\n"
    )
    transformer = LidarFrameTransformer()

    timestamps, seqs, points = load_lidar_csv(csv_path, transformer)

    assert timestamps.tolist() == [100]
    assert seqs.tolist() == [0]
    assert points[0].shape == (2, 2)  # the two (0.0, 0.0) dropout returns are dropped

    expected = transformer.lidar_to_robot(
        "lidar_front_left", np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    )[:, :2]
    np.testing.assert_allclose(points[0], expected, atol=1e-5)

    mount_xy = transformer.lidar_to_robot("lidar_front_left", np.zeros((1, 3), dtype=np.float32))[0, :2]
    assert not np.any(np.all(np.isclose(points[0], mount_xy), axis=1))


def test_v3_and_v4_extrinsics_are_not_interchangeable():
    """Regression for the bug this rewrite fixes: nav's old frame_transformer.py pinned v3's
    extrinsics unconditionally, and v3 -> v4 is a rotation change, not just a translation one.
    """
    point = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    v3 = LidarFrameTransformer("v3").lidar_to_robot("lidar_front_left", point)
    v4 = LidarFrameTransformer("v4").lidar_to_robot("lidar_front_left", point)
    assert not np.allclose(v3, v4, atol=1e-3)
