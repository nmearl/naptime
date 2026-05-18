from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        freqs = torch.arange(1, dim + 1, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1) or (...,)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        angles = x * self.freqs
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.gelu(self.norm1(x)))
        h = self.dropout(h)
        h = self.conv2(F.gelu(self.norm2(h)))
        return x + h


class ConvBackbone1D(nn.Module):
    def __init__(self, channels: int, layers: int, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualConvBlock(channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class GaussianSetConv1D(nn.Module):
    def __init__(self, sigmas: tuple[float, ...]):
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self,
        point_x: torch.Tensor,
        point_feat: torch.Tensor,
        point_mask: torch.Tensor,
        grid_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # point_x: (B, N), point_feat: (B, N, C), point_mask: (B, N), grid_x: (G,)
        # returns: agg (B, G, S*C), density (B, G, S) where S = len(sigmas)
        dx = point_x.unsqueeze(-1) - grid_x.view(1, 1, -1)  # (B, N, G)
        aggs, densities = [], []
        for sigma in self.sigmas:
            weights = torch.exp(-0.5 * (dx / sigma) ** 2) * point_mask.unsqueeze(-1)
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            agg = torch.einsum("bng,bnc->bgc", weights, point_feat) / denom.squeeze(1).unsqueeze(-1)
            aggs.append(agg)
            densities.append(weights.sum(dim=1).unsqueeze(-1))  # (B, G, 1)
        return torch.cat(aggs, dim=-1), torch.cat(densities, dim=-1)


class GlobalLatentEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, N, C), mask: (B, N)
        m = mask.unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        mean = (x * m).sum(dim=1) / denom
        x_masked = x.masked_fill(m == 0, float("-inf"))
        xmax = torch.where(
            torch.isfinite(x_masked).any(dim=1),
            x_masked.max(dim=1).values,
            torch.zeros_like(mean),
        )
        pooled = torch.cat([mean, xmax], dim=-1)
        h = self.mlp(pooled)
        return self.mu(h), self.logvar(h)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers.extend([nn.Linear(d, hidden_dim), nn.GELU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
