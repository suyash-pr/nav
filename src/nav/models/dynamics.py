import torch
from torch import nn


class LatentDynamics(nn.Module):
    """Predicts the next latent as a residual over the current latent, conditioned on the action."""

    def __init__(self, latent_dim: int = 128, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, action], dim=-1))
