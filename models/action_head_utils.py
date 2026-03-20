"""Shared action-head building blocks."""

import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


class ResidualActionMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_blocks: int = 3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.out(x)
