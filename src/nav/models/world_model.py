import copy

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn

from nav.models.dynamics import LatentDynamics
from nav.models.encoders import DepthEncoder, LidarEncoder, RgbEncoder


class JepaWorldModel(LightningModule):
    """JEPA world model: per-modality encoders + an EMA target + a latent dynamics MLP."""

    def __init__(
        self,
        n_cameras: int = 1,
        rgb_dim: int = 64,
        depth_dim: int = 32,
        lidar_dim: int = 32,
        hidden: int = 256,
        ema_tau: float = 0.99,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoders = nn.ModuleDict(
            {
                "rgb": RgbEncoder(n_cameras, rgb_dim),
                "depth": DepthEncoder(depth_dim),
                "lidar": LidarEncoder(lidar_dim),
            }
        )
        self.target_encoders = copy.deepcopy(self.encoders)
        self.target_encoders.requires_grad_(False)

        latent_dim = rgb_dim + depth_dim + lidar_dim
        self.dynamics = LatentDynamics(latent_dim, action_dim=2, hidden=hidden)

    def encode(self, encoders: nn.ModuleDict, batch: dict, step: int) -> torch.Tensor:
        rgb = encoders["rgb"](batch["rgb"][:, step])
        depth = encoders["depth"](batch["depth"][:, step])
        lidar = encoders["lidar"](batch["bev"][:, step])
        return torch.cat([rgb, depth, lidar], dim=-1)

    def forward(self, batch: dict) -> torch.Tensor:
        z_t = self.encode(self.encoders, batch, 0)
        return self.dynamics(z_t, batch["action"])

    def shared_step(self, batch: dict) -> dict:
        z_t = self.encode(self.encoders, batch, 0)
        with torch.no_grad():
            z_next = self.encode(self.target_encoders, batch, 1)
        pred = self.dynamics(z_t, batch["action"])

        loss = F.smooth_l1_loss(pred, z_next)
        identity_loss = F.smooth_l1_loss(z_t, z_next)
        z_std = z_t.std(dim=0).mean()
        return {"loss": loss, "identity_loss": identity_loss, "z_std": z_std}

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self.shared_step(batch)
        self.log("train_loss", metrics["loss"])
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self.shared_step(batch)
        self.log("val_loss", metrics["loss"])
        self.log("val_identity_loss", metrics["identity_loss"])
        self.log("val_z_std", metrics["z_std"])

    def on_before_zero_grad(self, optimizer) -> None:
        tau = self.hparams.ema_tau
        for online, target in zip(self.encoders.parameters(), self.target_encoders.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1 - tau)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=self.hparams.lr)
