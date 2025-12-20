"""VLA Diffusion Policy Model."""

import torch.nn as nn
from .encoders import TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead

from .vision.registry import VisionEncoderCfg, build_vision_encoder
import models.vision


class VLADiffusionPolicy(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_dim,
        action_dim,
        d_model=128,
        diffusion_T=16,
        vision_cfg: VisionEncoderCfg | None = None,
    ):
        super().__init__()

        # notice how I removed `ImageEncoderTinyCNN` import and now we rely on registry.

        if vision_cfg is None:
            vision_cfg = VisionEncoderCfg(name="tinycnn", d_model=d_model)

        # vision encoder outputs d_model
        vision_cfg.d_model = d_model
        self.vision_cfg = vision_cfg
        self.img_encoder = build_vision_encoder(vision_cfg)

        self.txt_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)

        cfg = DiffusionConfig(T=diffusion_T, action_dim=action_dim, cond_dim=d_model)
        self.diffusion_head = DiffusionPolicyHead(cfg)

    def encode_obs(self, image, text_tokens, state):
        img_token = self.img_encoder(image)  # (B, d_model)
        txt_token = self.txt_encoder(text_tokens)  # (B, d_model)
        state_token = self.state_encoder(state)  # (B, d_model)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context

    def loss(self, image, text_tokens, state, actions):
        """
        Compute the loss of the diffusion policy head given the image, text tokens, state, and actions.
        """
        cond = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.loss(actions, cond)

    def act(self, image, text_tokens, state):
        """
        image: (B, 3, H, W)
        text_tokens: (B, T_text)
        state: (B, state_dim)
        returns: (B, action_dim)
        """
        cond = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.sample(cond)
