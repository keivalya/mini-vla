from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Type, Optional, List

import torch
from torch.nn import nn

@dataclass
class VisionEncoderCfg:
    name : str = "tinycnn"
    d_model : int = 128
    pretrained : Optional[str] = None
    trainable : bool = False
    image_size : Optional[int] = None


class VisionEncoder(nn.Module):
    def __init__(self, cfg: VisionEncoderCfg):
        super().__init__()
        self.cfg = cfg

    @property
    def d_model(self) -> int:
        return self.cfg.d_model

_REGISTRY: Dict[str, Type[VisionEncoder]] = {}

def register_vision_encoder(name: str):
    name = name.lower()

    def decorator(cls: Type[VisionEncoder]):
        if name in _REGISTRY:
            raise ValueError(f"Vision encoder '{name}' already registered by {_REGISTRY[name]}.")
        _REGISTRY[name] = cls
        return cls

    return decorator

def available_vision_encoders() -> List[str]:
    return list(_REGISTRY.keys())

def build_vision_encoder(cfg: VisionEncoderCfg) -> VisionEncoder:
    name = cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown vision encoder '{name}'. Available: {available_vision_encoders()}")
    return _REGISTRY[name](cfg)