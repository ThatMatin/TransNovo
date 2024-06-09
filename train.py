import torch
import torch.nn as nn
import torch.nn.functional as F

#|%%--%%| <hVaifVrkF7|BP1Q2E56wH>

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
print(f'using device {device}')

#|%%--%%| <BP1Q2E56wH|yxh8vl6HXU>

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


#|%%--%%| <yxh8vl6HXU|LKAkSrBI89>

enc = Encoder()
x = torch.randn((batch_size, 20, d_model))
out = enc(x)
out.shape
print(sum([p.numel() for p in enc.parameters()]))

#|%%--%%| <LKAkSrBI89|T4hzN4yUll>

ma = MultiHeadAttention()
x = torch.randn((batch_size, 20, d_model))
print(sum([p.numel() for p in ma.parameters()]))
ma(x).shape

#|%%--%%| <T4hzN4yUll|7eG7t7Pu3z>

a = AttentionHead(is_masked=True)
x = torch.randn((batch_size, 20, d_model))
a(x).shape
sum([p.numel() for p in a.parameters()])

#|%%--%%| <7eG7t7Pu3z|mvEmCLtZYf>

t = torch.arange(24).view((3, 4, 2))
u = t.unsqueeze(1)
u.expand(-1, 2, -1, -1)
#|%%--%%| <mvEmCLtZYf|1xR9pd4kTl>

# number of weights in a linear model if a new dimension gets added to the input
i = torch.randn((2, 2, 3, 4))
l = nn.Linear(10, 10, bias=False)
print(sum([p.numel() for p in l.parameters()]))

#|%%--%%| <1xR9pd4kTl|yJB170Kh8I>

heads = [AttentionHead() for _ in range(n_heads)]
keys = torch.stack([head.key.weight for head in heads], dim=0)
keys.shape


#|%%--%%| <yJB170Kh8I|GIgNifeMDM>

import re
line ="Name: IQVR/2"
name = re.split(r'/\d+', line.split('Name: ')[1].strip())[0]
print(name)

#|%%--%%| <GIgNifeMDM|SAvSCv0qOQ>

import gzip
with gzip.open("./data/crap.msp.gz", "rt") as g:
    with open("test_data.msp", "w") as f:
        for _ in range(20000):
            f.write(g.readline())

#|%%--%%| <SAvSCv0qOQ|fGrZEKAYYM>

path = Path("./data/").glob("*Borrelia*")
with gzip.open(next(path), "rt") as g:
    print(g.read()[:1000])

#|%%--%%| <fGrZEKAYYM|8tCeu4nZXE>

import gzip
import os
from pathlib import Path
path = Path("./data/")
for p in path.glob("*.msp.gz"):
    with gzip.open(p, "rt") as g:
        print(f"file: {p.name} | filesize: {os.path.getsize(p)*1e-6:.2f}\nfirst 10 line: {g.readlines(10)}")

#|%%--%%| <8tCeu4nZXE|pDR15LiEhj>

import data_processor
from torch.utils.data import DataLoader, Dataset
from importlib import reload

reload(data_processor)

class MSPLoader(Dataset):
    """Reads msp.gz files, if their sizes are below the max_size(in MB)"""
    def __init__(self, max_size=10):
        super().__init__()
        self.path = Path(data_dir)
        self.spectra = []

        for p in self.path.glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > max_size:
                continue
            print(f"file: {p} | filesize: {os.path.getsize(p)*1e-6:.2f}")
            self.spectra.append(parse_msp_gz(p))

    def __len__(self):
        l = 0
        for s in self.spectra:
            l += len(s)
        return l

#|%%--%%| <pDR15LiEhj|bzqbI6kB7f>

msp = MSPLoader()
print(len(msp))

#|%%--%%| <bzqbI6kB7f|2tZNRNQ72A>
