import os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

import data_processor

# Globals

N_epochs = 10
lr = 1e-4
batch_size = 128
d_model = 32
n_heads = 8
d_key = d_model//n_heads
d_val = d_model//n_heads
d_ff = 128
dropout = 0.2
activation = nn.ReLU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './data/'
aa_csv = './data/amino_acids.csv'
model_save_path = 'transnovo.pth'
vocab_size = 25
print(f'using device {device}')


# Embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
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

    def forward(self, x):
        tok_emb = self.embedding(x) + torch.sqrt(torch.tensor(self.embedding.weight.size(1), dtype=torch.float32))
        pos_emb = self.pos_embd(x)
        return tok_emb + pos_emb


class SpectrumEmbedding(nn.Module):
    """
    Receives a spectrum, discretizes the intensity and 
    creates an embedding
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)

    def forward(self, x):
        x = self.embedding(x) * torch.sqrt(torch.tensor(
                self.embedding.weight.size(1), dtype=torch.float32))
        return x

# Transformer Classes

class AttentionHead(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_key, bias=False)
        self.query = nn.Linear(d_model, d_key, bias=False)
        self.value = nn.Linear(d_model, d_val, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.is_masked = is_masked

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None):
        if kv is None:
            kv = q
        _, T, C = q.shape
        K = self.key(kv)
        V = self.value(kv)
        Q = self.query(q)
        W = (Q @ K.transpose(-2, -1))/ C**0.5
        # TODO: filter zero tokens
        if self.is_masked:
            # TODO: Check if it's possible to register the buffer and reuse
            W = W.masked_fill(torch.tril(torch.ones((T, T), device=device)) == 0, float('-inf'))
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

    def forward(self, q, kv):
        out = torch.concat([h(q, kv) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.MHA = MultiHeadAttention(is_masked)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv=None):
        q = q + self.MHA(self.ln1(q), kv)
        out = self.ln2(q)
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
        return self.ff(self.at(x))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.masked_at = AttentionBlock(is_masked=True)
        self.enc_dec_at = AttentionBlock()
        self.ff = FeedForward()

    def forward(self, dec_in, enc_out):
        at_out = self.masked_at(dec_in)
        out = self.enc_dec_at(at_out, enc_out)
        return self.ff(out)
        

class TransNovo(nn.Module):
    def __init__(self):
        super().__init__()
        self.peptide_emb = PeptideEmbedding(vocab_size, d_model)
        self.spectrum_emb = SpectrumEmbedding()
        self.enc = Encoder()
        self.dec = Decoder()
        self.ll = nn.Linear(d_model, vocab_size)

    def forward(self, x, targets: torch.Tensor):
        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]

        x = self.enc(self.spectrum_emb(x))
        out = self.dec(self.peptide_emb(tgt_input), x)
        logits = self.ll(out)
        probs = F.softmax(logits, dim=-1)

        logits_flat = logits.view(-1, vocab_size)
        tgt_output_flat = tgt_output.reshape(-1)
        loss = F.cross_entropy(logits_flat, tgt_output_flat)
        return probs, loss

# Dataset

# TODO: Improve speed by tensorifying as per file read
class MSPLoader(Dataset):
    """Reads msp.gz files, if their sizes are below the max_size(in MB)"""
    def __init__(self, max_size=10):
        super().__init__()
        self.MAX_X = 0
        self.MAX_Y = 0
        self.X = torch.tensor([])
        self.Y = torch.tensor([])

        # NOTE: Read in the msp files and add all the samples to spectra
        _spectra = []
        for p in Path(data_dir).glob("*.msp.gz"):
            if os.path.getsize(p)*1e-6 > max_size:
                continue
            print(f"*> reading file: {p} | size: {os.path.getsize(p)*1e-6:.2f}")
            _spectra += data_processor.parse_msp_gz(p)

        # NOTE: Get max lenght of spectrum and peptide sequence
        for spectrum in _spectra:
            self.MAX_X = max(self.MAX_X, len(spectrum[1]))
            self.MAX_Y = max(self.MAX_Y, len(spectrum[0]))
        self.MAX_Y += 2 # padding accounted

        names, spectra = [], []
        for name, spectrum in _spectra:
            # NOTE: Pad and expand tensors to comply with length
            name = [0] + name + [0]
            y_offset = self.MAX_Y - len(name)
            if y_offset:
                name += y_offset*[0]
            names.append(name)
            offset = self.MAX_X - len(spectrum)
            if offset:
                spectrum += offset*[(0, 0)]
            spectra.append(spectrum)
        self.X = torch.tensor(spectra).to(device)
        self.Y = torch.tensor(names).to(device)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index]



# Running Script

msp = MSPLoader(30)
print(f"Total length of data: {len(msp)}")
dataloader = DataLoader(msp, batch_size, True, pin_memory=True)

model = TransNovo().to(device)
# TODO: implement paper's lrate formula
optimizer = Adam(model.parameters(), lr)
param_count = sum([p.numel() for p in model.parameters()])
print(f"total number of parameters: {param_count}")

# check all datatypes are float 32 and on cuda
for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        print(f"parameter {name} - {param.dtype}")
    if str(param.device) != "cuda:0":
        print(f"parameter {name} is on device {param.device}")

# train/test split
train_size = int(0.8*len(msp))
test_size = len(msp) - train_size
train_ds, test_ds = random_split(msp, [train_size, test_size])
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)

# training loop
lossi = []
loss = torch.tensor([])

s_time = time.time()
for epoch in tqdm(range(N_epochs), position=0, leave=True):
    model.train()
    for X,Y in train_dl:
        _, loss = model(X, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lossi.append(loss.log10().item())

    if epoch % 10 == 0:
        print(f"epoch: {epoch} | train loss: {loss.item():.4f}")

print(f"training time: {(time.time() - s_time)//60: 0.1f}m")
torch.save(model.state_dict(), model_save_path)
print(f"Saved to {model_save_path}")

plt.plot(torch.tensor(lossi).view(-1, 1000).mean(1) )


