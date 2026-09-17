import copy

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities import grad_norm
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
        enc_lr_scale: float = 0.3,
        vicreg_std_weight: float = 0.5,
        vicreg_cov_weight: float = 0.5,
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

    def block_vicreg_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Std/cov VICReg losses computed within each modality block, so the cov penalty
        doesn't fight cross-modal correlations the dynamics model relies on.

        Computed in float32 regardless of autocast dtype: variances/covariances involve
        small magnitudes and squared terms that lose precision under bf16 mixed precision.
        """
        z = z.float()
        std_losses = []
        cov_losses = []
        start = 0
        for size in (self.hparams.rgb_dim, self.hparams.depth_dim, self.hparams.lidar_dim):
            block = z[:, start : start + size]
            start += size
            std_losses.append(F.relu(1 - (block.var(dim=0) + 1e-4).sqrt()).mean())
            block_c = block - block.mean(dim=0)
            cov = (block_c.T @ block_c) / (block_c.shape[0] - 1)
            cov_losses.append((cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / size)
        return torch.stack(std_losses).mean(), torch.stack(cov_losses).mean()

    def shared_step(self, batch: dict) -> dict:
        z_t = self.encode(self.encoders, batch, 0)
        with torch.no_grad():
            z_next = self.encode(self.target_encoders, batch, 1)
        pred = self.dynamics(z_t, batch["action"])

        pred_loss = F.smooth_l1_loss(pred, z_next)
        std_loss, cov_loss = self.block_vicreg_loss(z_t)
        loss = (
            pred_loss
            + self.hparams.vicreg_std_weight * std_loss
            + self.hparams.vicreg_cov_weight * cov_loss
        )

        identity_loss = F.smooth_l1_loss(z_t, z_next)
        z_std = z_t.std(dim=0).mean()
        return {
            "loss": loss,
            "pred_loss": pred_loss,
            "std_loss": std_loss,
            "cov_loss": cov_loss,
            "identity_loss": identity_loss,
            "z_std": z_std,
        }

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        metrics = self.shared_step(batch)
        self.log("train_loss", metrics["loss"])
        self.log("train_pred_loss", metrics["pred_loss"])
        self.log("train_std_loss", metrics["std_loss"])
        self.log("train_cov_loss", metrics["cov_loss"])
        return metrics["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        metrics = self.shared_step(batch)
        self.log("val_loss", metrics["loss"])
        self.log("val_pred_loss", metrics["pred_loss"])
        self.log("val_std_loss", metrics["std_loss"])
        self.log("val_cov_loss", metrics["cov_loss"])
        self.log("val_identity_loss", metrics["identity_loss"])
        self.log("val_z_std", metrics["z_std"])

    def on_before_optimizer_step(self, optimizer) -> None:
        norms = grad_norm(self, norm_type=2)
        self.log("grad_norm", norms["grad_2.0_norm_total"])

    def on_before_zero_grad(self, optimizer) -> None:
        tau = self.hparams.ema_tau
        for online, target in zip(self.encoders.parameters(), self.target_encoders.parameters()):
            target.data.mul_(tau).add_(online.data, alpha=1 - tau)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        enc_params = [p for p in self.encoders.parameters() if p.requires_grad]
        other_params = [p for p in self.dynamics.parameters() if p.requires_grad]
        return torch.optim.Adam(
            [
                {"params": enc_params, "lr": self.hparams.lr * self.hparams.enc_lr_scale},
                {"params": other_params, "lr": self.hparams.lr},
            ]
        )
