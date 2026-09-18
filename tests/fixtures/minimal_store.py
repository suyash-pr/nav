"""A minimal preprocessed store -- cycle bookkeeping only, no rgb/depth/lidar arrays -- for
tests that exercise CycleDataset's pairing/splitting logic without paying for video decode.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nav.preprocess.store import SessionManifest, write_array


def write_minimal_store(
    out_dir: Path,
    *,
    run_id: list[int],
    cycle_id: list[int],
    tick_us: list[int],
    action: list[tuple[float, float]] | None = None,
) -> Path:
    n = len(run_id)
    manifest = SessionManifest(session=out_dir.name, n_cycles=n, rgb_cameras=[], depth_cameras=[])
    write_array(out_dir, "cycles/run_id.npy", np.array(run_id, dtype=np.int64))
    write_array(out_dir, "cycles/cycle_id.npy", np.array(cycle_id, dtype=np.int64))
    write_array(out_dir, "cycles/tick_us.npy", np.array(tick_us, dtype=np.int64))
    write_array(out_dir, "cycles/action.npy", np.array(action or [(1.0, 0.0)] * n, dtype=np.float32))
    write_array(out_dir, "cycles/proprio.npy", np.zeros((n, 2), dtype=np.float32))
    write_array(out_dir, "lidar/valid.npy", np.ones(n, dtype=bool))
    manifest.write(out_dir)
    return out_dir
