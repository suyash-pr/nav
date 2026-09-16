from itertools import pairwise

import torch
import torchvision
from torch import nn

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class RgbEncoder(nn.Module):
    """Frozen ResNet-18 trunk + a learned per-camera embedding + a trainable head."""

    def __init__(self, n_cameras: int = 1, out_dim: int = 64, cam_embed_dim: int = 8):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(*list(backbone.children())[:-2])
        self.trunk.requires_grad_(False)
        self.pool = nn.AdaptiveAvgPool2d((2, 3))
        self.cam_embed = nn.Embedding(n_cameras, cam_embed_dim)
        self.head = nn.Sequential(
            nn.Linear(512 * 2 * 3 * n_cameras + cam_embed_dim * n_cameras, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.register_buffer("mean", IMAGENET_MEAN)
        self.register_buffer("std", IMAGENET_STD)

    def train(self, mode: bool = True) -> "RgbEncoder":
        super().train(mode)
        self.trunk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n_cam, c, h, w = x.shape
        x = x.reshape(b * n_cam, c, h, w)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            feat = self.trunk(x)
        feat = self.pool(feat).flatten(1).view(b, n_cam, -1)
        cam_ids = torch.arange(n_cam, device=x.device)
        cam_feat = self.cam_embed(cam_ids).unsqueeze(0).expand(b, -1, -1)
        feat = torch.cat([feat, cam_feat], dim=-1).flatten(1)
        return self.head(feat)


def conv_stack(in_channels: int, depth: int) -> nn.Sequential:
    channels = [in_channels, 16, 32, 64, 64][: depth + 1]
    layers = []
    for c_in, c_out in pairwise(channels):
        layers += [nn.Conv2d(c_in, c_out, 4, stride=2, padding=1), nn.GroupNorm(min(8, c_out), c_out), nn.SiLU()]
    layers.append(nn.AdaptiveAvgPool2d(1))
    return nn.Sequential(*layers)


class DepthEncoder(nn.Module):
    """Small CNN over stacked (depth, validity) channels, averaged across cameras."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = conv_stack(in_channels=2, depth=4)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        return self.head(self.net(x))


class LidarEncoder(nn.Module):
    """Small CNN over the merged BEV occupancy grid."""

    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.net = conv_stack(in_channels=1, depth=3)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(x))
