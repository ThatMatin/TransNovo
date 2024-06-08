import torch
import torch.nn as nn
import torch.nn.functional as F

#|%%--%%| <hVaifVrkF7|BP1Q2E56wH>

batch_size = 32
d_model = 32
n_head = 8
d_key = d_model//n_head
d_val = d_model//n_head
n_enc_layer = 2
n_dec_layer = 3
dim_feed_forward = 128
dropout = 0.2
activation = nn.ReLU()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using device {device}')

#|%%--%%| <BP1Q2E56wH|yxh8vl6HXU>

class SelfAttentionHead(nn.Module):
    def __init__(self, is_masked=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_key)
        self.query = nn.Linear(d_model, d_key)
        self.value = nn.Linear(d_model, d_val)
        self.is_masked = is_masked

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        W = (Q @ K.transpose(-2, -1))/ C**0.5
        if self.is_masked:
            # TODO: Check if possible to register the buffer and reuse
            W = W.masked_fill(torch.tril(torch.ones((T, T))), float('-inf'))
        W = torch.softmax(W, dim=-1)
        out = W @ V
        return out


class TransNovo(nn.Module):
    def __init__(self):
        pass

    def forward(self, x, targets=None):
        pass


#|%%--%%| <yxh8vl6HXU|7eG7t7Pu3z>

a = SelfAttentionHead()
x = torch.randn((batch_size, 20, d_model))
a(x).shape
