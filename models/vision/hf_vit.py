# models/vision/hf_vit.py
from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .registry import VisionEncoder, VisionEncoderCfg, register_vision_encoder


class HFViTVisionEncoder(VisionEncoder):
    """
    HuggingFace Vision encoder wrapper (CLIP / SigLIP).

    Input:  x (B,3,H,W) float in [0,1] (or uint8 in [0,255])
    Output: (B,D) pooled CLS token, projected to cfg.d_model if needed
    """
    def __init__(
        self,
        cfg: VisionEncoderCfg,
        hf_kind: Literal["clip", "siglip"],
        default_pretrained: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        super().__init__(cfg)

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
            raise ImportError("Install transformers: pip install transformers") from e

        # project to cfg.d_model
        hidden = int(self.backbone.config.hidden_size)
        out_dim = int(cfg.d_model)
        self.proj = nn.Identity() if out_dim == hidden else nn.Linear(hidden, out_dim)

        # normalize
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self.image_size = int(cfg.image_size or getattr(self.backbone.config, "image_size", 224))

        # cfg has trainable, not freeze
        if not cfg.trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8: x = x.float() / 255.0
        else: x = x.float()

        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

        x = (x - self.mean) / self.std # normalize
        out = self.backbone(pixel_values=x, return_dict=True).last_hidden_state  # (B, N, hidden)
        cls = out[:, 0]
        return self.proj(cls)


@register_vision_encoder("hf_clip_vit")
class HFCLIPViT(HFViTVisionEncoder):
    def __init__(self, cfg: VisionEncoderCfg):
        super().__init__(
            cfg=cfg,
            hf_kind="clip",
            default_pretrained="openai/clip-vit-base-patch32",
            # source for the following mean/std -- default values from HuggingFace documentation
            # https://hf.co/docs/transformers/main/model_doc/clip#transformers.CLIPImageProcessor.image_mean
            mean=(0.48145466, 0.4578275, 0.40821073), # CLIP mean
            std=(0.26862954, 0.26130258, 0.27577711), # CLIP std
            # i know, they look weird, but trust me they work
        )


@register_vision_encoder("hf_siglip_vit")
class HFSiglipViT(HFViTVisionEncoder):
    def __init__(self, cfg: VisionEncoderCfg):
        super().__init__(
            cfg=cfg,
            hf_kind="siglip",
            default_pretrained="google/siglip-base-patch16-224",
            mean=(0.5, 0.5, 0.5), # SigLIP mean
            std=(0.5, 0.5, 0.5), # SigLIP std
        )
