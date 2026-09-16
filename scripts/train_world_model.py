import argparse
import os
from dataclasses import dataclass, fields

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from nav.data_module import WorldModelDataModule
from nav.models.world_model import JepaWorldModel

DATA_DIR = "/data/tdmpc/suyash"
CSV_PATH = os.path.join(r"/data/tdmpc/suyash/sains-ladbroke-grove_karter-01_2026-05-21_03-35-26_ll_navigation_command_ll_navigation_command.csv")
CAMERAS = ("oakd_front", "oakd_back")  # the only cameras with depth coverage


@dataclass
class Config:
    data_dir: str = DATA_DIR
    csv_path: str = CSV_PATH
    bev_path: str = os.path.join(DATA_DIR, "derived", "lidar_bev.npy")
    bev_ts_path: str = os.path.join(DATA_DIR, "derived", "lidar_bev_ts.npy")
    batch_size: int = 32
    num_workers: int = 12
    max_epochs: int = 50
    lr: float = 3e-4
    wandb_project: str = os.environ.get("WANDB_PROJECT", "tdmpc")
    wandb_entity: str = os.environ.get("WANDB_ENTITY", "p9r7")
    run_name: str = ""


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    for field in fields(Config):
        parser.add_argument(f"--{field.name}", type=type(field.default), default=field.default)
    return Config(**vars(parser.parse_args()))


def main() -> None:
    config = parse_args()

    dm = WorldModelDataModule(
        config.data_dir,
        config.csv_path,
        config.bev_path,
        config.bev_ts_path,
        cameras=CAMERAS,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    model = JepaWorldModel(n_cameras=len(CAMERAS), lr=config.lr)

    logger = WandbLogger(
        project=config.wandb_project,
        entity=config.wandb_entity or None,
        name=config.run_name or None,
        log_model=False,
    )
    trainer = Trainer(
        accelerator="gpu",
        precision="bf16-mixed",
        max_epochs=config.max_epochs,
        callbacks=[ModelCheckpoint(monitor="val_loss")],
        logger=logger,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
