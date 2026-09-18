import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

from nav.calibration.lidar_extrinsics import DEFAULT_SENSOR_VERSION, calibration_for


def make_transform(translation_xyz_m, rotation_deg, euler_order) -> RigidTransform:
    r = Rotation.from_euler(euler_order, rotation_deg, degrees=True)
    return RigidTransform.from_components(np.array(translation_xyz_m, dtype=np.float64), r)


def transform_points(transform: RigidTransform, points_xyz_m: np.ndarray) -> np.ndarray:
    return transform.apply(points_xyz_m)


class LidarFrameTransformer:
    """Robot-frame transforms for one sensor suite's lidars, keyed by `frame_id`.

    A robot's sensor_version (v3/v4/v4_1) determines both the mount geometry and, for v3 vs
    v4/v4_1, the mount *orientation* -- see nav.calibration.lidar_extrinsics for why the tables
    aren't interchangeable. Construct one of these per session rather than using a module-level
    default, so a mixed-fleet dataset can't silently apply the wrong robot's calibration.
    """

    def __init__(self, sensor_version: str = DEFAULT_SENSOR_VERSION):
        self.sensor_version = sensor_version
        calibration = calibration_for(sensor_version)
        self.transforms = {
            frame_id: make_transform(cal["translation"], cal["rotation"], cal["euler_order"])
            for frame_id, cal in calibration.items()
        }

    def lidar_to_robot(self, frame_id: str, points_xyz_m: np.ndarray) -> np.ndarray:
        return transform_points(self.transforms[frame_id], points_xyz_m)
