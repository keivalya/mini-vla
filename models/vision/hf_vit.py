from __future__ import annotations
from typing import Optional, Literal

import torch
from torch.nn import nn
import torch.nn.functional as F

from .registry import VisionEncoder, VisionEncoderCfg

class HFViTVisionEncoder(VisionEncoder):
    def __init__(
        self,
        cfg : VisionEncoderCfg,
        hf_kind : Literal["clip", "siglip"],
        default_pretrained : str,
        mean : tuple[float, float, float],
        std : tuple[float, float, float],
    ):
        try:
            if hf_kind == "clip":
                from transformers import CLIPVisionModel
                model_id = cfg.pretrained or default_pretrained
                self.backbone = CLIPVisionModel.from_pretrained(model_id)
            elif hf_kind == "siglip":
                from transformers import SiglipVisionModel
                model_id = cfg.pretrained or default_pretrained
                self.backbone = SiglipVisionModel.from_pretrained(model_id)
            else:
                raise ValueError(f"Unsupported hf_kind={hf_kind}")
        except ImportError as e:
            raise ImportError(f"Install transformers: pip install transformers") from e

        c = getattr(self.backbone, "config", None)
        image_size = getattr(c, "image_size", None)
        if image_size is None and hasattr(c, "vision_config"):
            image_size = getattr(c.vision_config, "image_size", None)
        self.image_size = int(cfg.image_size or image_size or 84)
        