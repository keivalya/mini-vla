# models/flow_matching_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingConfig:
    def __init__(self, action_dim, cond_dim, t_embed_dim=32):
        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.t_embed_dim = t_embed_dim

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
        Conditional flow matching loss:
        Compare predicted velocity vs ideal velocity field - simplified for supervised case.
        """
        B = actions.size(0)
        # uniform random t in [0,1]
        t = torch.rand(B, device=actions.device)
        # construct noisy sample
        noise = torch.randn_like(actions)
        x_t = actions + t.unsqueeze(-1) * noise
        # ideal velocity: here approximate as (actions - x_t) for simplicity
        ideal_v = (actions - x_t) / (t.unsqueeze(-1) + 1e-8)
        v_pred = self.model(x_t, t, cond)
        return F.mse_loss(v_pred, ideal_v)

    @torch.no_grad()
    def sample(self, cond, n_samples=None):
        B = cond.size(0) if n_samples is None else n_samples
        cond = cond.expand(B, -1)
        # start from random noise
        x_t = torch.randn(B, self.cfg.action_dim, device=cond.device)
        # deterministic integration: Euler step
        timesteps = torch.linspace(1, 0, steps=10, device=cond.device)
        for t in timesteps:
            t_batch = torch.full((B,), t, device=cond.device)
            v = self.model(x_t, t_batch, cond)
            x_t = x_t + v * (timesteps[1] - timesteps[0])
        return x_t
