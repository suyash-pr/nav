from pathlib import Path

import numpy as np
from lightning.pytorch import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Subset

from nav.dataset import CycleDataset
from nav.utils.split import split_by_run, split_by_session


def _compute_action_stats(datasets: list[CycleDataset], pair_indices: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of the action label over exactly the training pairs, matching the old pipeline's
    train-only normalization (data_module.py:241-242 before this rewrite)."""
    actions = [
        dataset.action[dataset.pairs[idx, 0]] for dataset, idx in zip(datasets, pair_indices) if len(idx) > 0
    ]
    if not actions:
        return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
    stacked = np.concatenate(actions, axis=0)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = np.maximum(stacked.std(axis=0), 1e-6).astype(np.float32)
    return mean, std


class WorldModelDataModule(LightningDataModule):
    """Wraps one or more `CycleDataset` stores for use with a Lightning Trainer.

    Each `store_dir` is one preprocessed session (see `scripts/preprocess_session.py`). With more
    than one, whole sessions are held out for validation (`split_by_session`); with exactly one,
    whole trailing runs are held out instead (`split_by_run`) -- see `nav.utils.split` for why
    holding out trailing *stretches of the same drive*, as the old pipeline did, stopped being
    the right unit once there is more than one session to hold out.
    """

    def __init__(
        self,
        store_dirs: list[str | Path],
        *,
        rgb_cameras: tuple[str, ...] | None = None,
        depth_cameras: tuple[str, ...] | None = None,
        val_fraction: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 12,
    ):
        super().__init__()
        self.store_dirs = [Path(d) for d in sorted(store_dirs)]  # session dirnames are timestamps
        self.rgb_cameras = rgb_cameras
        self.depth_cameras = depth_cameras
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        self.datasets = [
            CycleDataset(store_dir, rgb_cameras=self.rgb_cameras, depth_cameras=self.depth_cameras)
            for store_dir in self.store_dirs
        ]

        if len(self.datasets) > 1:
            train_ds_idx, val_ds_idx = split_by_session(self.datasets, self.val_fraction)
            train_pairs = [np.arange(len(self.datasets[i])) for i in train_ds_idx]
            mean, std = _compute_action_stats([self.datasets[i] for i in train_ds_idx], train_pairs)
            for dataset in self.datasets:
                dataset.action_mean, dataset.action_std = mean, std

            self.train_dataset = ConcatDataset([self.datasets[i] for i in train_ds_idx])
            self.val_dataset = ConcatDataset([self.datasets[i] for i in val_ds_idx])
        else:
            dataset = self.datasets[0]
            train_idx, val_idx = split_by_run(dataset, self.val_fraction)
            mean, std = _compute_action_stats([dataset], [train_idx])
            dataset.action_mean, dataset.action_std = mean, std

            self.train_dataset = Subset(dataset, train_idx.tolist())
            self.val_dataset = Subset(dataset, val_idx.tolist())

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
            pin_memory=True,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stores", nargs="+", required=True, help="one or more preprocessed session store dirs")
    args = parser.parse_args()

    dm = WorldModelDataModule(args.stores, batch_size=8, num_workers=0)
    dm.setup()
    print(f"train={len(dm.train_dataset)} val={len(dm.val_dataset)}")

    batch = next(iter(dm.train_dataloader()))
    for key, value in batch.items():
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
