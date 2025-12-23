"""VLA Diffusion Policy Model."""

from __future__ import annotations

import torch.nn as nn
from typing import Optional
from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP

from .action_expert.registry import ActionExpertCfg, build_action_expert
import models.action_expert  # Import to trigger registration of action experts


class VLADiffusionPolicy(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_dim,
        action_dim,
        d_model=128,
        diffusion_T=16,
        action_expert_cfg: Optional[ActionExpertCfg] = None,
    ):
        super().__init__()
        self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)


        if action_expert_cfg is None:
            action_expert_cfg = ActionExpertCfg(
                name="diffusion",
                action_dim=action_dim,
                cond_dim=d_model,
                T=diffusion_T,
            )
        else:
            # Ensure action expert config matches provided parameters
            action_expert_cfg.cond_dim = d_model
            action_expert_cfg.action_dim = action_dim
            action_expert_cfg.T = diffusion_T

        self.action_expert_cfg = action_expert_cfg
        self.action_expert = build_action_expert(action_expert_cfg)

    def encode_obs(self, image, text_tokens, state):
        img_token = self.img_encoder(image)  # (B, d_model)
        txt_token = self.txt_encoder(text_tokens)  # (B, d_model)
        state_token = self.state_encoder(state)  # (B, d_model)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context

    def loss(self, image, text_tokens, state, actions):
        """
        Compute the loss of the action expert given the image, text tokens, state, and actions.
        """
        cond = self.encode_obs(image, text_tokens, state)
        return self.action_expert.loss(actions, cond)

    def act(self, image, text_tokens, state):
        """
        image: (B, 3, H, W)
        text_tokens: (B, T_text)
        state: (B, state_dim)
        returns: (B, action_dim)
        """
        cond = self.encode_obs(image, text_tokens, state)
        return self.action_expert.sample(cond)
