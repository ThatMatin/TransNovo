import torch
import torch.nn as nn
from numpy import log

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1), :]

class PeptideEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embd = PositionalEncoding(d_model, 100)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        tok_emb = self.embedding(x) + torch.sqrt(torch.tensor(self.embedding.weight.size(1), dtype=torch.float32))
        pos_emb = self.pos_embd(x)
        return self.ln(tok_emb + pos_emb)


class SpectrumEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.embedding(x) * torch.sqrt(torch.tensor(
                self.embedding.weight.size(1), dtype=torch.float32))
        return self.ln(out)
