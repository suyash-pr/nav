import numpy as np

from nav.frame_transformer import lidar_to_robot
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

    timestamps, points = load_lidar_csv(str(csv_path))

    assert timestamps.tolist() == [100]
    assert points[0].shape == (2, 2)  # the two (0.0, 0.0) dropout returns are dropped

    expected = lidar_to_robot(
        "lidar_front_left", np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=np.float32)
    )[:, :2]
    np.testing.assert_allclose(points[0], expected, atol=1e-5)

    mount_xy = lidar_to_robot("lidar_front_left", np.zeros((1, 3), dtype=np.float32))[0, :2]
    assert not np.any(np.all(np.isclose(points[0], mount_xy), axis=1))
