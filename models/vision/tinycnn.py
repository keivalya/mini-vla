from torch.nn import nn
import torch.nn.functional as F

from .registry import VisionEncoder, VisionEncoderCfg, register_vision_encoder

@register_vision_encoder("tinycnn")
class TinyCNNEncoder(VisionEncoder):
    def __init__(self, cfg: VisionEncoderCfg):
        super().__init__(cfg)
        d_model = cfg.d_model

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.proj(x)
        x = self.ln(x)
        return x
