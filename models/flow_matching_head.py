"""Flow-matching policy head for action generation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class FlowMatchingConfig:
    action_dim: int
    cond_dim: int
    t_embed_dim: int = 32
    sample_steps: int = 32

class SinusoidalTime(nn.Module):
    """This is kept same as the diffusion time embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, torch.log(torch.tensor(1000.0)), half, device=t.device))
        args = t.unsqueeze(-1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb

class FlowMatchingModel(nn.Module):
    """
    Predict velocity field for flow matching: v(x_t, t | cond)
    I LOVE THIS BLOG from Federico Sarrocco
    https://federicosarrocco.com/blog/flow-matching
    (hence, the code follows the same structure)
    """

    def __init__(self, cfg: FlowMatchingConfig, hidden_dim=128):
        super().__init__()
        self.time_emb = SinusoidalTime(cfg.t_embed_dim)
        in_dim = cfg.action_dim + cfg.t_embed_dim + cfg.cond_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        t_emb = self.time_emb(t)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        v_pred = self.net(x)
        return v_pred

class FlowMatchingPolicyHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = FlowMatchingModel(cfg)

    def loss(self, actions, cond):
        """
        Conditional flow matching with a linear interpolation path.

        We sample a source point x_0 ~ N(0, I), set x_1 to the demonstrated
        action, and train the model to predict the constant velocity field
        along x_t = (1 - t) * x_0 + t * x_1.
        """
        B = actions.size(0)
        t = torch.rand(B, device=actions.device)

        x_0 = torch.randn_like(actions)
        t_expanded = t.unsqueeze(-1)
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * actions
        target_v = actions - x_0

        v_pred = self.model(x_t, t, cond)
        return F.mse_loss(v_pred, target_v)

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        B = cond.size(0) if n_samples is None else n_samples
        if cond.size(0) != B:
            cond = cond.expand(B, -1)

        # Start at the source distribution and integrate forward to t = 1.
        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)
        dt = 1.0 / self.cfg.sample_steps

        for step in range(self.cfg.sample_steps):
            t = torch.full((B,), step * dt, device=cond.device)
            v = self.model(x_t, t, cond)
            x_t = x_t + v * dt

        return x_t
