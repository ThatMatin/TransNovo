import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import data_processor

input_dim = 20
batch_size = 16
d_model = 32
n_heads = 8
d_key = d_model//n_heads
d_val = d_model//n_heads
d_ff = 128
n_enc_layer = 2
n_dec_layer = 3
dropout = 0.2
activation = nn.ReLU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './data/'
aa_csv = './data/amino_acids.csv'

class AttentionHead(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_key, bias=False)
        self.query = nn.Linear(d_model, d_key, bias=False)
        self.value = nn.Linear(d_model, d_val, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_masked = is_masked

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        W = (Q @ K.transpose(-2, -1))/ C**0.5
        if self.is_masked:
            # TODO: Check if it's possible to register the buffer and reuse
            W = W.masked_fill(torch.tril(torch.ones((T, T))) == 0, float('-inf'))
        W = torch.softmax(W, dim=-1)
        W = self.dropout(W)
        out = W @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(is_masked) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.concat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class AttentionBlock(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.MHA = MultiHeadAttention(is_masked)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.MHA(self.ln1(x))
        out = self.ln2(x)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                activation,
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
                )

    def forward(self, x):
        out = self.net(x)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = FeedForward()
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out = x + self.ff(x)
        out = self.ln(out)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.at = AttentionBlock()
        self.ff = FeedForwardBlock()

    def forward(self, x):
        out = self.ff(self.at(x))
        return out


class TransNovo(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, targets=None):
        pass

class MSPLoader(Dataset):
    """Reads msp.gz files, if their sizes are below the max_size(in MB)"""
    def __init__(self, max_size=10):
        super().__init__()
        self._path = Path(data_dir)
        self._spectra = []
        self.X = None
        self.Y = None

        # NOTE: Read in the msp files and add all the samples to spectra
        for p in self._path.glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > max_size:
                continue
            print(f"*> reading file: {p} | size: {os.path.getsize(p)*1e-6:.2f}")
            self._spectra += data_processor.parse_msp_gz(p)

        # NOTE: Tensorify the data: N * S * 2
        # S is the max spectrum length -> M_X
        M_X, M_Y = self.get_maxes()
        names, spectra = [], []
        for name, spectrum in self._spectra:
            # NOTE: Expand tensors to comply with length
            y_offset = M_Y - len(name)
            if y_offset:
                name += y_offset*[0]
            names.append(name)
            offset = M_X - len(spectrum)
            if offset:
                spectrum += offset*[(-1, -1)]
            spectra.append(spectrum)
        self.X = torch.tensor(spectra)
        self.Y = torch.tensor(names)

    def get_maxes(self):
        MAX_X, MAX_Y = 0, 0
        for spectrum in self._spectra:
            MAX_X = max(MAX_X, len(spectrum[1]))
            MAX_Y = max(MAX_Y, len(spectrum[0]))
        return MAX_X, MAX_Y

    def __len__(self):
        if self.X is not None:
            return self.X.shape[0]
        else:
            return len(self._spectra)

    def __getitem__(self, index):
        if self.X and self.Y is not None:
            return self.X[index], self.Y[index]

    def __getitems__(self, index):
        if self.X and self.Y is not None:
            return self.X[index], self.Y[index]


