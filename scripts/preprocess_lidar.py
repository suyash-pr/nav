import argparse
import os
import time

import numpy as np

from nav.lidar_loader import load_lidar_csv, merge_scans

LIDAR_DIRS = ("lidar2d_front_left", "lidar2d_front_right", "lidar2d_back_left", "lidar2d_back_right")
REFERENCE = "lidar2d_front_left"


def preprocess(data_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    ref_path = os.path.join(data_dir, REFERENCE, REFERENCE + "_points.csv")
    ref_ts, ref_points = load_lidar_csv(ref_path)

    others = []
    for name in LIDAR_DIRS:
        if name == REFERENCE:
            continue
        path = os.path.join(data_dir, name, name + "_points.csv")
        others.append(load_lidar_csv(path))

    bev = np.lib.format.open_memmap(
        os.path.join(out_dir, "lidar_bev.npy"), mode="w+", dtype=np.uint8, shape=(len(ref_ts), 120, 120)
    )
    for i, (_, grid) in enumerate(merge_scans(ref_ts, ref_points, others)):
        bev[i] = grid
    bev.flush()

    np.save(os.path.join(out_dir, "lidar_bev_ts.npy"), ref_ts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/data/model-training")
    parser.add_argument("--out_dir", default="")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "derived")

    start = time.time()
    preprocess(args.data_dir, out_dir)
    print(f"wrote {out_dir} in {time.time() - start:.1f}s")
