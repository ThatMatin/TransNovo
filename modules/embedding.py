import torch
import torch.nn as nn
from math import log, pi

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.l_min = l_min = 0.001
        l_max = 10000
        len = int(l_max/l_min)
        pe = torch.zeros(len, d_model)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = l_max/l_min * torch.exp(torch.arange(0, d_model, 2).float() * (-log(l_min/2 * pi) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        # x: batch * T_x * mz
        input = (x * 1/self.l_min).int()
        return self.pe[input]


class PeptideEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int,
                 device="cuda"):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(1, d_model, device=device),
                nn.LayerNorm(d_model),
                )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # NOTE: Setting bias to zero here causes large gradient norm
                # if layer.bias is not None:
                #     nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x.unsqueeze(-1))


class SpectrumEmbedding(nn.Module):
    def __init__(self, d_model, device="cuda"):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.intensity_embedding = nn.Linear(1, d_model, device=device)
        self.ln = nn.LayerNorm(d_model, device=device)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.intensity_embedding.weight)
        nn.init.zeros_(self.intensity_embedding.bias)

    def forward(self, x):
        mz_out = self.pos_emb(x[:, :, 0])
        int_out = self.intensity_embedding(x[:, :, 1].unsqueeze(-1))
        out = mz_out + int_out
        return self.ln(out)
