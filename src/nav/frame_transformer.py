import numpy as np
from scipy.spatial.transform import RigidTransform, Rotation

# translation in meters [x, y, z], rotation in degrees, from
# karter's ansible/inventory/group_vars/v3/lidars.yml
LIDAR_CALIBRATION = {
    "lidar_front_left": {
        "translation": [0.428, 0.250, 0.173],
        "rotation": [180, 45],
        "euler_order": "XZ",
    },
    "lidar_front_right": {
        "translation": [0.428, -0.250, 0.173],
        "rotation": [180, 315],
        "euler_order": "XZ",
    },
    "lidar_back_left": {
        "translation": [-0.428, 0.250, 0.173],
        "rotation": [180, 135],
        "euler_order": "XZ",
    },
    "lidar_back_right": {
        "translation": [-0.428, -0.250, 0.173],
        "rotation": [180, 225],
        "euler_order": "XZ",
    },
}


def make_transform(translation_xyz_m, rotation_deg, euler_order) -> RigidTransform:
    r = Rotation.from_euler(euler_order, rotation_deg, degrees=True)
    return RigidTransform.from_components(np.array(translation_xyz_m, dtype=np.float64), r)


def transform_points(transform: RigidTransform, points_xyz_m: np.ndarray) -> np.ndarray:
    return transform.apply(points_xyz_m)


LIDAR_TRANSFORMS = {
    frame_id: make_transform(cal["translation"], cal["rotation"], cal["euler_order"])
    for frame_id, cal in LIDAR_CALIBRATION.items()
}


def lidar_to_robot(frame_id: str, points_xyz_m: np.ndarray) -> np.ndarray:
    return transform_points(LIDAR_TRANSFORMS[frame_id], points_xyz_m)
