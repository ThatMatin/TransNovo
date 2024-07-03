from typing import Optional
import torch
import torch.nn as nn
from math import log, pi
from tokenizer.aa import get_vocab_size


class MZPositionalEncoding(nn.Module):
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[x]


class PeptideEmbedding(nn.Module):
    def __init__(self, 
                 d_model: int,
                 device="cuda"):
        super().__init__()
        self.pos_emb = PositionalEncoding(d_model)
        self.emb = nn.Embedding(get_vocab_size(), d_model)
        self.ln = nn.LayerNorm(d_model, device=device)
        nn.init.xavier_normal_(self.emb.weight)

    def forward(self, y):
        pos_emb = self.pos_emb(y)
        pep_emb = self.emb(y)
        return self.ln(pos_emb + pep_emb)


class PeptidePrecursorEmbedding(nn.Module):
    def __init__(self,
                 d_model: int,
                 mz_pos_embedding: Optional[MZPositionalEncoding]=None,
                 device="cuda"):
        super().__init__()
        self.pep_emb = PeptideEmbedding(d_model, device)
        self.charge_emb = nn.Embedding(10, d_model)
        if mz_pos_embedding is None:
            self.mz_emb = MZPositionalEncoding(d_model)
        else:
            self.mz_emb = mz_pos_embedding
        self.ln = nn.LayerNorm(d_model)
        nn.init.xavier_normal_(self.charge_emb.weight)

    def forward(self, y:torch.Tensor, charge:torch.Tensor, mz:torch.Tensor):
        pep_emb = self.pep_emb(y)
        ch_emb = self.charge_emb(charge.unsqueeze(-1))
        mz_emb = self.mz_emb(mz.unsqueeze(-1))
        return self.ln(pep_emb + ch_emb + mz_emb)


class SpectrumEmbedding(nn.Module):
    def __init__(self, d_model, device="cuda"):
        super().__init__()
        self.pos_emb = MZPositionalEncoding(d_model)
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
