import os, time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from importlib import reload
from pathlib import Path
from typing import Optional
from tqdm.auto import tqdm

import data_processor

#|%%--%%| <hVaifVrkF7|BP1Q2E56wH>

N_epochs = 100
lr = 1e-5
batch_size = 128
d_model = 512
n_heads = 8
d_key = d_model//n_heads
d_val = d_model//n_heads
d_ff = 1024
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_dir = './data/'
aa_csv = './data/amino_acids.csv'
model_save_path = 'transnovo.pth'
vocab_size = 25
print(f'using device {device}')

#|%%--%%| <BP1Q2E56wH|pDR15LiEhj>

reload(data_processor)
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
        self.X = self.discretize(self.X)
        self.Y = torch.tensor(names).to(device)

    def discretize(self, X):
        _, indices = torch.sort(X[:, :, 1], descending=True)
        sorted_indices = indices.unsqueeze(-1).expand(-1, -1, 2)
        X_sorted = X.gather(1, sorted_indices)

        # Non zero elements per spectrum (batch element)
        non_zeros_counts = X_sorted.count_nonzero(1)[:, 0]
        # start indeces of the weakest 33%
        w33_st_idxs = (2/3 * non_zeros_counts).int()

        # pluck the weakest 33% and mean them, then create a tensor from means
        means_list = []
        for b in range(X.size(0)):
            m = X_sorted[b, w33_st_idxs[b]:non_zeros_counts[b], 1].mean()
            means_list.append(m)
        w33_means = torch.stack(means_list)

        w33_means_div = w33_means.unsqueeze(-1).unsqueeze(-1).expand_as(X).clone()
        w33_means_div[:, :, 0] = 1

        X_w33_normalized = X / w33_means_div

        intensities = X_w33_normalized[:, :, 1]
        discretized = torch.zeros_like(intensities)
        discretized[intensities >= 10] = 3
        discretized[(intensities >= 2) & (intensities < 10)] = 2
        discretized[(intensities >= 0.05) & (intensities < 2)] = 1
        discretized[intensities < 0.05] = 0

        X_intens_disc = X_w33_normalized.clone()
        X_intens_disc[:, :, 1] = discretized

        return X_intens_disc


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index], self.Y[index]


#|%%--%%| <pDR15LiEhj|yxh8vl6HXU>

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
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        tok_emb = self.embedding(x) + torch.sqrt(torch.tensor(self.embedding.weight.size(1), dtype=torch.float32))
        pos_emb = self.pos_embd(x)
        return self.ln(tok_emb + pos_emb)


class SpectrumEmbedding(nn.Module):
    """
    Receives a spectrum, discretizes the intensity and 
    creates an embedding
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.embedding(x) * torch.sqrt(torch.tensor(
                self.embedding.weight.size(1), dtype=torch.float32))
        return self.ln(out)


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
        # W = self.dropout(W)
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
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv=None):
        if kv == None:
            kv = q
        res = q + self.MHA(q, kv)
        return self.ln2(res)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
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
        encoded_x = self.enc(self.spectrum_emb(x))
        out = self.dec(self.peptide_emb(tgt_input), encoded_x)
        return self.ll(out)


    @torch.no_grad()
    def generate(self, x):
        self.eval()

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        _, pep_decoder = data_processor.get_AA_Dec_Enc(aa_csv)

        out = []
        dec_in = torch.zeros((1 ,1),device=device, dtype=torch.int64)
        for _ in range(100):
            encoded_x = self.enc(self.spectrum_emb(x))
            dec_out = self.dec(self.peptide_emb(dec_in), encoded_x)
            probs = F.softmax(self.ll(dec_out), dim=-1)
            next_aa = torch.multinomial(probs.squeeze(), num_samples=1, replacement=True).item()
            
            out.append(next_aa)
            if next_aa == 0:
                break

        return pep_decoder(out)
            
#|%%--%%| <yxh8vl6HXU|YvQnWIl8kc>

# Preparing data
msp = MSPLoader(400)
print(f"Total number of data: {len(msp)}")
# TODO: Check memory pinning
dataloader = DataLoader(msp, batch_size, True)

#|%%--%%| <YvQnWIl8kc|EQ8E8fH6Is>

model = TransNovo().to(device)
# TODO: implement paper's lrate formula
# lr = d_model**-0.5 * min(1**-0.5, 10**1-.5)
optimizer = Adam(model.parameters(), lr, (0.9, 0.98), 1e-9)
loss_fn = nn.CrossEntropyLoss()
param_count = sum([p.numel() for p in model.parameters()])
print(f"total number of parameters: {param_count}")

#|%%--%%| <EQ8E8fH6Is|Ixnf9hkxNS>

# check all datatypes are float 32 and on cuda
for name, param in model.named_parameters():
    if param.dtype != torch.float32:
        print(f"parameter {name} - {param.dtype}")
    if str(param.device) != "cuda:0":
        print(f"parameter {name} is on device {param.device}")

#|%%--%%| <Ixnf9hkxNS|MwGEjiuQ9m>

train_size = int(0.8*len(msp))
test_size = len(msp) - train_size
train_ds, test_ds = random_split(msp, [train_size, test_size])
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)

#|%%--%%| <MwGEjiuQ9m|pE2B6PAjh4>

def print_grad(name, grad):
    if torch.isnan(grad).any():
        print(f"NaN detected in gradient for {name}!")

# Register hooks on all parameters with their names
for name, param in model.named_parameters():
    param.register_hook(lambda grad, name=name: print_grad(name, grad))

#|%%--%%| <pE2B6PAjh4|dmifUWL5xz>

# INFO: This block normalizes each spectrum in the batch by the average of 33% weakest intensities

X, Y = msp[100: 113]
sorted, indices = torch.sort(X[:, :, 1], descending=True)
sorted_indices = indices.unsqueeze(-1).expand(-1, -1, 2)
X_sorted = X.gather(1, sorted_indices)

# Non zero elements per spectrum (batch element)
non_zeros_counts = X_sorted.count_nonzero(1)[:, 0]
# start indeces of the weakest 33%
w33_st_idxs = (2/3 * non_zeros_counts).int()

# pluck the weakes 33%
means_list = []
for b in range(X.size(0)):
    m = X_sorted[b, w33_st_idxs[b]:non_zeros_counts[b], 1].mean()
    means_list.append(m)
w33_means = torch.stack(means_list)

w33_means_div = w33_means.unsqueeze(-1).unsqueeze(-1).expand_as(X).clone()
w33_means_div[:, :, 0] = 1

X_w33_normalized = X / w33_means_div

# TEST:
assert (X[:, :, 0] == X_w33_normalized[:, :, 0]).all()
assert torch.allclose(w33_means[1] * X_w33_normalized[1, :, 1][:25], X[1, :, 1][:25])
# print(X[1, :, 1][:25])
# print(w33_means[1] * X_w33_normalized[1, :, 1][:25])
#|%%--%%| <dmifUWL5xz|3C42BFPGyx>

# INFO: Grass intensity discretization

intensities = X_w33_normalized[:, :, 1]
discretized = torch.zeros_like(intensities)
discretized[intensities >= 10] = 3
discretized[(intensities >= 2) & (intensities < 10)] = 2
discretized[(intensities >= 0.05) & (intensities < 2)] = 1
discretized[intensities < 0.05] = 0

X_intens_disc = X_w33_normalized.clone()
X_intens_disc[:, :, 1] = discretized

# view results
# print(X_intens_disc[3][:22])
# print(X_w33_normalized[3][:22])

#|%%--%%| <3C42BFPGyx|x9JPoCiRId>

train_lossi = []
test_lossi = []
train_norms = []

s_time = time.time()
for epoch in tqdm(range(1)):

    model.train()
    loss_list = []
    for X,Y in train_dl:
        logits = model(X, Y)
        # FIX: Implement discretization to address troubly data
        if torch.isnan(logits).any():
            # continue
            break

        optimizer.zero_grad(True)

        tgt_output = Y[:, 1:]
        logits_flat = logits.transpose(-2, -1)

        loss = loss_fn(logits_flat, tgt_output)

        loss.backward()
        optimizer.step()

        train_lossi.append(loss.item())
        loss_list.append(loss.item())

        # norms observation
        norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms.append(param.grad.norm())
        train_norms.append(sum(norms)/len(norms))

    train_batch_loss = torch.mean(torch.tensor(loss_list))

    with torch.inference_mode():
        model.eval()
        loss_list = []
        for X,Y in test_dl:
            logits = model(X, Y)
            # FIX: same as above
            if torch.isnan(logits).any():
                continue

            tgt_output = Y[:, 1:]
            logits_flat = logits.transpose(-2, -1)
            loss = loss_fn(logits_flat, tgt_output)

            test_lossi.append(loss.item())
            loss_list.append(loss.item())

    test_batch_loss = torch.mean(torch.tensor(loss_list))

    # Update learning rate
    lr = 1e-1 * d_model**-0.5 * min((epoch+1)**-0.5, (epoch + 1) * 10**-1.5)
    for p in optimizer.param_groups:
        p['lr'] = lr

    if epoch % 1 == 0:
        print(f"epoch: {epoch} | train loss: {train_batch_loss:.4f} | test loss: {test_batch_loss:.4f}")


print(f"training time: {time.time() - s_time: 0.1f}s")

#|%%--%%| <x9JPoCiRId|RZUjcGa99k>

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 6))
plt.subplot(2, 2, 1)
plt.plot(torch.tensor(train_lossi))
plt.title("train loss")
plt.subplot(2, 2, 2)
plt.plot(torch.tensor(test_lossi))
plt.title("test loss")
plt.subplot(2, 2, 3)
plt.plot(torch.log10(torch.tensor(train_norms)))
plt.title("gradient norms")

#|%%--%%| <RZUjcGa99k|1O2swP2h1L>

torch.save(model.state_dict(), model_save_path)
print(f"Saved to {model_save_path}")

#|%%--%%| <1O2swP2h1L|E2JCCJvk1v>


enc, dec = data_processor.get_AA_Dec_Enc(aa_csv)
X, Y = msp[200]
print(model.generate(X))
dec(Y.tolist())

#|%%--%%| <E2JCCJvk1v|MncYGDHvTS>

# Check if discretization produces nan
for X, Y in train_dl:
    if torch.isnan(msp.discretize(X)).any():
        print("Imposter")

for X, Y in test_dl:
    if torch.isnan(msp.discretize(X).any()):
        print("Test Imposter")

#|%%--%%| <MncYGDHvTS|3va8sjeCus>

with torch.inference_mode():
    for X, Y in train_dl:
        if torch.isnan(model(msp.discretize(X), Y)).any():
            print(X)

#|%%--%%| <3va8sjeCus|E8AEbcolc9>

import matplotlib.pyplot as plt
plt.plot(torch.tensor(train_lossi).view(-1, 1000).mean(1) )

#|%%--%%| <E8AEbcolc9|78sPz0FTOb>

model.load_state_dict(torch.load(model_save_path))

