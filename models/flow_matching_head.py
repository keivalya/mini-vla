"""Flow matching policy head for action generation."""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_expert.registry import ActionExpert, ActionExpertCfg, register_action_expert


@dataclass
class FlowMatchingConfig:
    T: int = 50  # number of time steps
    action_dim: int = 4
    cond_dim: int = 128 # conditional input dim


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        """
        t: (B,) continuous time values in [0, 1]
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        device = t.device
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                device=device
            )
        )
        # Scale continuous time [0, 1] to [0, 1000] to match frequency range
        t_scaled = t.float() * 1000.0
        # (B, half_dim)
        args = t_scaled.unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

class FlowMatchingModel(nn.Module):
    """
    epsilon_theta(x_t, t, cond)
    x_t:   (B, action_dim)
    t:     (B,) continuous time in [0, 1]
    cond:  (B, cond_dim) fused VLA token
    """
    def __init__(self, cfg: FlowMatchingConfig, time_emb_dim=32, hidden_dim=128):
        super().__init__()
        self.cfg = cfg
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        in_dim = cfg.action_dim + time_emb_dim + cfg.cond_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.action_dim),
        )

    def forward(self, x_t, t, cond):
        """
        x_t: (B, action_dim)
        t:   (B,) continuous time in [0, 1]
        cond: (B, cond_dim)
        """
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        x = torch.cat([x_t, t_emb, cond], dim=-1)
        v_pred = self.net(x)
        return v_pred

@register_action_expert("flow_matching")
class FlowMatchingPolicyHead(ActionExpert):
    def __init__(self, cfg: ActionExpertCfg):
        super().__init__(cfg)
        # Convert ActionExpertCfg to FlowMatchingConfig for internal use
        flow_cfg = FlowMatchingConfig(
            T=cfg.T,
            action_dim=cfg.action_dim,
            cond_dim=cfg.cond_dim,
        )
        self.flow_cfg = flow_cfg
        self.flow_matching_model = FlowMatchingModel(flow_cfg)

    def q_sample(self, x_1, t, x_0):
        """
        Flow matching interpolation: x_t = (1 - t) * x_0 + t * x_1
        x_0: (B, action_dim)  source (noise)
        x_1: (B, action_dim)  target (actions)
        t:   (B,) continuous in [0, 1]
        """
        # unsqueeze t to (B, 1) for broadcasting with (B, action_dim)
        t = t.unsqueeze(-1)  # (B, 1)
        return (1 - t) * x_0 + t * x_1

    def loss(self, actions, cond):
        """
        actions: (B, action_dim)  ground-truth actions
        cond:    (B, cond_dim)    fused VLA token
        """
        B = actions.size(0)
        device = actions.device
        t = torch.rand((B,), device=device) # uniform sampling t
        x_0 = torch.randn_like(actions)
        v_target = actions - x_0
        x_t = self.q_sample(actions, t, x_0)  # noisy actions
        v_pred = self.flow_matching_model(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        """
        cond: (B, cond_dim) or (1, cond_dim)
        returns: (B, action_dim) sampled actions x_0
        """
        self.eval()
        if n_samples is None:
            B = cond.size(0)
        else:
            B = n_samples
            cond = cond.expand(B, -1)

        x_t = torch.randn(B, self.flow_cfg.action_dim, device=cond.device)
        dt = 1.0 / self.flow_cfg.T
        for t_step in range(self.flow_cfg.T):
            t = torch.full((B,), t_step * dt, device=cond.device, dtype=torch.float32)
            v_pred = self.flow_matching_model(x_t, t, cond)

            x_t = x_t + v_pred * dt
        return x_t
