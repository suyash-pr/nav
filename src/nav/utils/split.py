"""Train/val splits over CycleDataset stores.

`chronological_split` (the old pipeline's only split) held out trailing "navigating" stretches
within the single session the pipeline ever handled. With many sessions, within-session frames
are highly correlated -- consecutive cycles overlap almost entirely -- so holding out trailing
stretches of the *same drive* tests memorization more than generalisation. `split_by_session`
holds out whole sessions instead, which is the honest unit once there is more than one.

`split_by_run` is kept for the single-session case, but with a guard band the old
`chronological_split` didn't have: nothing stopped a val sample's `t` and a train sample's `t+1`
from resolving to the very same image, which `split_by_run` fixes by excluding a `step_cycles`
margin on both sides of the boundary.
"""

from __future__ import annotations

import numpy as np

from nav.dataset import CycleDataset


def split_by_session(datasets: list[CycleDataset], val_fraction: float) -> tuple[list[int], list[int]]:
    """Holds out whole sessions (by sample count) from the end of `datasets`.

    `datasets` should already be in a stable, meaningful order (e.g. sorted by session name,
    which for karter sessions is a timestamp) -- this only decides *how many* trailing sessions
    to hold out, not what order they're in.

    Returns (train_dataset_indices, val_dataset_indices): indices into `datasets`, for building
    two ConcatDatasets. A session with zero samples cannot anchor a split by itself; if the
    single trailing session already covers >= val_fraction, it alone is held out.
    """
    sizes = np.array([len(d) for d in datasets], dtype=np.int64)
    total = int(sizes.sum())
    if total == 0 or len(datasets) < 2:
        return list(range(len(datasets))), []

    target_val = round(total * val_fraction)
    cumulative_from_end = np.cumsum(sizes[::-1])
    n_val_sessions = int(np.searchsorted(cumulative_from_end, target_val, side="left")) + 1
    n_val_sessions = min(max(n_val_sessions, 1), len(datasets) - 1)

    boundary = len(datasets) - n_val_sessions
    return list(range(boundary)), list(range(boundary, len(datasets)))


def split_by_run(dataset: CycleDataset, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Holds out whole trailing runs; if there's only one run, trailing pairs within it instead.

    Either way a `step_cycles`-wide guard band is excluded from train, so no train pair's second
    endpoint can be the same frame as a held-out val pair's first endpoint.
    """
    if len(dataset) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)

    pair_run_id = dataset.run_id[dataset.pairs[:, 0]]
    runs = np.unique(pair_run_id)

    if len(runs) > 1:
        counts = np.array([(pair_run_id == r).sum() for r in runs])
        cumulative = np.cumsum(counts[::-1])
        n_val_runs = int(np.searchsorted(cumulative, round(len(dataset) * val_fraction), side="left")) + 1
        n_val_runs = min(max(n_val_runs, 1), len(runs) - 1)
        val_runs = set(runs[-n_val_runs:])
        val_mask = np.isin(pair_run_id, list(val_runs))
        train_mask = ~val_mask
    else:
        order = np.argsort(dataset.pairs[:, 0], kind="stable")
        n_val = max(1, round(len(dataset) * val_fraction))
        boundary = len(dataset) - n_val
        guard = dataset.step_cycles
        train_mask = np.zeros(len(dataset), dtype=bool)
        val_mask = np.zeros(len(dataset), dtype=bool)
        train_mask[order[: max(0, boundary - guard)]] = True
        val_mask[order[boundary:]] = True

    return np.flatnonzero(train_mask), np.flatnonzero(val_mask)
