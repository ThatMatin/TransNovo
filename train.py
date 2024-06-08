import torch
import torch.nn as nn
import torch.nn.functional as F

#|%%--%%| <hVaifVrkF7|BP1Q2E56wH>

batch_size = 32
d_model = 32
n_heads = 8
d_key = d_model//n_heads
d_val = d_model//n_heads
n_enc_layer = 2
n_dec_layer = 3
dim_feed_forward = 128
dropout = 0.2
activation = nn.ReLU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device {device}')

#|%%--%%| <BP1Q2E56wH|yxh8vl6HXU>

class AttentionHead(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_key, bias=False)
        self.query = nn.Linear(d_model, d_key, bias=False)
        self.value = nn.Linear(d_model, d_val, bias=False)
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
        out = W @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        # B, H, T, C
        self.heads = [AttentionHead(is_masked) for _ in range(n_heads)]



class TransNovo(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, targets=None):
        pass


#|%%--%%| <yxh8vl6HXU|7eG7t7Pu3z>

a = AttentionHead(is_masked=True)
x = torch.randn((batch_size, 20, d_model))
a(x).shape
#|%%--%%| <7eG7t7Pu3z|mvEmCLtZYf>

t = torch.arange(24).view((3, 4, 2))
u = t.unsqueeze(1)
u.expand(3, 2, 4, 2)
#|%%--%%| <mvEmCLtZYf|1xR9pd4kTl>

# number of weights in a linear model if a new dimension gets added to the input
i = torch.randn((2, 2, 3, 4))
l = nn.Linear(10, 10, bias=False)
print(sum([p.numel() for p in l.parameters()]))

