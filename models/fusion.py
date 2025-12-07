import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, img_token, txt_token, state_token):
        x = torch.cat([img_token, txt_token, state_token], dim=-1)
        x = self.net(x)
        x = self.ln(x)
        return x  # fused context
