import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])  # GAP
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)


class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # (B, T, d_word)
        _, h_last = self.gru(x)
        x = h_last[0]  # (B, d_model)
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.ln(x)
        return x
