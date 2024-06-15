import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .embedding import PeptideEmbedding, SpectrumEmbedding
from .parameters import Parameters
from tokenizer import decode, get_vocab_size
from tokenizer.aa import END_TOKEN, PAD_TOKEN

MODEL_STATE_DICT = "model_state_dict"
MODEL_HYPERPARAMETERS = "hyper_params"

class AttentionHead(nn.Module):
    def __init__(self, d_model, d_key, d_val, dropout, is_masked=False):
        super().__init__()
        self.key = nn.Linear(d_model, d_key, bias=False)
        self.query = nn.Linear(d_model, d_key, bias=False)
        self.value = nn.Linear(d_model, d_val, bias=False)
        self.dropout = nn.Dropout(dropout)
        # TODO: find a good place for it
        # keep the dimensions large enough to cover max seq lenghth (200 here)
        self.register_buffer("tril", torch.tril(torch.ones(200, 200)))
        self.is_masked = is_masked

    def forward(self, q: torch.Tensor, kv: Optional[torch.Tensor] = None,
                pad_mask: Optional[torch.Tensor] = None):
        if kv is None:
            kv = q
        _, T, C = q.shape
        K = self.key(kv)
        V = self.value(kv)
        Q = self.query(q)
        W = (Q @ K.transpose(-2, -1))/ C**0.5
        if self.is_masked:
            W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        if pad_mask is not None:
            W = W.masked_fill(pad_mask == 1, float('-inf'))

        W = torch.softmax(W, dim=-1)
        W = self.dropout(W)
        out = W @ V
        return out


class MultiHeadAttention(nn.Module):
    """
    Note: d_val == d_model // h
    """
    def __init__(self, d_model, d_key, d_val, n_heads, dropout, is_masked=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_key, d_val, dropout, is_masked) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, kv, pad_mask=None):
        out = torch.concat([h(q, kv=kv, pad_mask=pad_mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_key, d_val, n_heads, dropout, is_masked=False):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, d_key, d_val, n_heads, dropout, is_masked)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, kv=None, pad_mask=None):
        if kv == None:
            kv = q
        res = q + self.MHA(q, kv=kv, pad_mask=pad_mask)
        return self.ln2(res)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
                )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.net(x)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        out = x + self.ff(x)
        out = self.ln(out)
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_key, d_val, n_heads, dropout):
        super().__init__()
        self.at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, False)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, X, pad_mask=None):
        return self.ff(self.at(X, pad_mask=pad_mask))


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_key, d_val, n_heads, dropout):
        super().__init__()
        self.masked_at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, True)
        self.enc_dec_at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, False)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, dec_in, enc_out, pad_mask=None):
        at_out = self.masked_at(dec_in, pad_mask=pad_mask)
        out = self.enc_dec_at(at_out, kv=enc_out)
        return self.ff(out)
        

class TransNovo(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.peptide_emb = PeptideEmbedding(get_vocab_size(), params.d_model)
        self.spectrum_emb = SpectrumEmbedding(params.d_model)
        self.enc = Encoder(params.d_model, params.d_ff, params.d_key, params.d_val, params.n_heads, params.dropout_rate)
        self.dec = Decoder(params.d_model, params.d_ff, params.d_key, params.d_val, params.n_heads, params.dropout_rate)
        self.ll = nn.Linear(params.d_model, get_vocab_size())
        self.N_named_parameters = None
        self.hyper_params = params
        self.introduce()

    def forward(self, X, Y: torch.Tensor):
        Y_input = Y[:, :-1]
        X_mask, Y_mask = self.get_padding_masks(X, Y_input)
        X_enc = self.enc(self.spectrum_emb(X), X_mask)
        model_out = self.dec(self.peptide_emb(Y_input), X_enc, pad_mask=Y_mask)
        return self.ll(model_out)


    def get_padding_masks(self, X, Y):
        with torch.no_grad():
            is_padding = (X == torch.tensor([PAD_TOKEN, 0], device=self.hyper_params.device)).all(dim=2)
            to_bool = is_padding.int()
            X_mask = to_bool.unsqueeze(-1).expand(-1, -1, X.size(1))
            Y_mask = (Y == PAD_TOKEN).unsqueeze(-1).expand(-1, -1, Y.size(1)).int()

        return X_mask.transpose(-2, -1), Y_mask.transpose(-2, -1)


    @torch.no_grad()
    def generate(self, x):
        self.eval()

        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        idx = torch.ones((1 , 1),device=self.hyper_params.device, dtype=torch.int64)
        while True:
            x_m, _ = self.get_padding_masks(x, idx)
            encoded_x = self.enc(self.spectrum_emb(x), x_m)
            dec_out = self.dec(self.peptide_emb(idx), encoded_x)
            probs = F.softmax(self.ll(dec_out), dim=-1)[:, -1, :]
            next_aa = torch.multinomial(probs.squeeze(), num_samples=1, replacement=True).unsqueeze(0)
            
            idx = torch.cat((idx, next_aa), dim=1)
            if next_aa.item() == END_TOKEN:
                break

        return idx.squeeze()


    def finish_training(self, epoch: int, train_result_matrix: torch.Tensor,
                        test_result_matrix: torch.Tensor, optimizer: torch.optim.Optimizer):
        trrm = train_result_matrix[:epoch]
        tsrm = test_result_matrix[:epoch]
        self.hyper_params.n_epochs_sofar += epoch
        self.hyper_params.optimizer_state_dict = optimizer.state_dict()
        self.hyper_params.train_result_matrix = torch.cat((self.hyper_params.train_result_matrix, trrm))
        self.hyper_params.test_result_matrix = torch.cat((self.hyper_params.test_result_matrix, tsrm))

        return self.save_with_metadata()


    def introduce(self):
        p = self.hyper_params
        t = "\n>>>>>>>>>>>>>>>>> TransNovo <<<<<<<<<<<<<<<<<<<"
        t += f"\n\td_model: {p.d_model}\n\tn_heads: {p.n_heads}\n\tlr: {p.learning_rate}"
        t += f"\n\td_key=d_val=d_query: {p.d_key}\n\td_ff: {p.d_ff}\n\tdropout: {p.dropout_rate}"
        t += f"\n\tLen X: {p.max_spectrum_length}\n\tLen Y: {p.max_peptide_lenght}"
        t += f"\nModel parameters: {self.total_param_count()}"
        t += f"\nNum data points: {p.data_point_count}\noptim: Adam"
        t += f"\nBatch size: {p.batch_size}"
        t += f"\nEpochs so far: {p.n_epochs_sofar}\nEpochs this round: {p.n_epochs}"
        print(t)

    # TODO: create and update Metrics Matrix
    def save_with_metadata(self):
        checkpoint = {
                MODEL_HYPERPARAMETERS : self.hyper_params(),
                MODEL_STATE_DICT : self.state_dict(),
                }

        return torch.save(checkpoint, self.hyper_params.model_save_path())


    def load_if_file_exists(self):
        if os.path.exists(self.hyper_params.model_save_path()):
            self.load_model(self.hyper_params.model_save_path())


    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.hyper_params.model_save_path()
        checkpoint = torch.load(model_path)
        self.hyper_params(checkpoint.get(MODEL_HYPERPARAMETERS, {}))
        self.load_state_dict(checkpoint[MODEL_STATE_DICT])
        self.introduce()


    def view_grad_norms(self):
        for n, p in self.named_parameters():
            print(f"{n}: {p.grad.norm()}")

    def grad_norms_mean(self) -> torch.Tensor:
        if self.N_named_parameters is None:
            self.N_named_parameters = len(list(self.named_parameters()))

        norms = torch.zeros(self.N_named_parameters)
        for i, (_, param) in enumerate(self.named_parameters()):
            if param.grad is not None:
                norms[i] = param.grad.norm()

        return norms.mean()


    def check_params(self, device='cuda:0', dtype=torch.float32):
        for name, param in self.named_parameters():
            if param.dtype != dtype:
                print(f"parameter {name} - {param.dtype}")
            if str(param.device) != device:
                print(f"parameter {name} is on device {param.device}")


    def register_grad_hook(self):
        def print_grad(name, grad):
            if torch.isnan(grad).any():
                print(f"NaN detected in gradient for {name}!")
        for name, param in self.named_parameters():
            param.register_hook(lambda grad, name=name: print_grad(name, grad))


    def total_param_count(self) -> int:
        return sum([p.numel() for p in self.parameters()])
