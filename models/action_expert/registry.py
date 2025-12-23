from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Type, Optional, List

import torch
import torch.nn as nn

@dataclass
class ActionExpertCfg:
    name: str = "diffusion"
    action_dim: int = 4
    cond_dim: int = 128
    T: int = 16
    # Diffusion-specific params
    beta_start: float = 1e-4
    beta_end: float = 1e-2
    # Flow matching doesn't need additional params beyond T, action_dim, cond_dim


class ActionExpert(nn.Module):
    """
    Base class for action generation heads (diffusion, flow matching, etc.)
    All action experts must implement:
    - loss(actions, cond) -> loss tensor
    - sample(cond, n_samples=None) -> actions tensor
    """
    def __init__(self, cfg: ActionExpertCfg):
        super().__init__()
        self.cfg = cfg

    def loss(self, actions: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        actions: (B, action_dim)  ground-truth actions
        cond:    (B, cond_dim)    fused VLA token
        returns: loss tensor
        """
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, n_samples: Optional[int] = None) -> torch.Tensor:
        """
        cond: (B, cond_dim) or (1, cond_dim)
        n_samples: optional number of samples to generate
        returns: (B, action_dim) sampled actions
        """
        raise NotImplementedError


_REGISTRY: Dict[str, Type[ActionExpert]] = {}

def register_action_expert(name: str):
    name = name.lower()

    def decorator(cls: Type[ActionExpert]):
        if name in _REGISTRY:
            raise ValueError(f"Action expert '{name}' already registered by {_REGISTRY[name]}.")
        _REGISTRY[name] = cls
        return cls

    return decorator

def available_action_experts() -> List[str]:
    return list(_REGISTRY.keys())

def build_action_expert(cfg: ActionExpertCfg) -> ActionExpert:
    name = cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown action expert '{name}'. Available: {available_action_experts()}")
    return _REGISTRY[name](cfg)

