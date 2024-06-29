import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .embedding import PeptideEmbedding, SpectrumEmbedding
from .parameters import Parameters
from tokenizer import get_vocab_size
from tokenizer.aa import END_TOKEN, PAD_TOKEN, mass_tensor

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
        self._init_weights()

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None):
        _, T, C = q.shape
        K = self.key(k)
        V = self.value(v)
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

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.value.weight)


class MultiHeadAttention(nn.Module):
    """
    Note: d_val == d_model // h
    """
    def __init__(self, d_model, d_key, d_val, n_heads, dropout, is_masked=False):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(d_model, d_key, d_val, dropout, is_masked) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, pad_mask=None):
        out = torch.concat([head(k, v, q, pad_mask=pad_mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_key, d_val, n_heads, dropout, is_masked=False):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, d_key, d_val, n_heads, dropout, is_masked)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, k:torch.Tensor, v:torch.Tensor, q:torch.Tensor, pad_mask=None):
        k = self.ln(k)
        v = self.ln(v)
        q = self.ln(q)
        out = q + self.MHA(k, v, q, pad_mask=pad_mask)
        return out


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

    def forward(self, X:torch.Tensor):
        out = self.net(X)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X:torch.Tensor):
        out = X + self.ff(self.ln(X))
        return out


class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, d_key, d_val, n_heads, dropout):
        super().__init__()
        self.at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, False)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, X, pad_mask=None):
        return self.ff(self.at(X, X, X, pad_mask=pad_mask))


class Decoder(nn.Module):
    def __init__(self, d_model, d_ff, d_key, d_val, n_heads, dropout):
        super().__init__()
        self.masked_at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, True)
        self.enc_dec_at = AttentionBlock(d_model, d_key, d_val, n_heads, dropout, False)
        self.ff = FeedForwardBlock(d_model, d_ff, dropout)

    def forward(self, dec_in, enc_out, pad_mask=None):
        at_out = self.masked_at(dec_in, dec_in, dec_in, pad_mask=pad_mask)
        out = self.enc_dec_at(enc_out, enc_out, at_out)
        return self.ff(out)
        

class TransNovo(nn.Module):
    def __init__(self, params: Parameters):
        super().__init__()
        self.peptide_emb = PeptideEmbedding(params.d_model, device=params.device)
        self.spectrum_emb = SpectrumEmbedding(params.d_model)
        self.encoders = nn.ModuleList([Encoder(params.d_model, params.d_ff, params.d_key,
                                              params.d_val, params.n_heads, params.dropout_rate)
                                      for _ in range(params.n_layers)])
        self.decoders = nn.ModuleList([Decoder(params.d_model, params.d_ff, params.d_key,
                                              params.d_val, params.n_heads, params.dropout_rate)
                                      for _ in range(params.n_layers)])
        self.ll = nn.Linear(params.d_model, get_vocab_size())
        nn.init.xavier_uniform_(self.ll.weight)
        nn.init.zeros_(self.ll.bias)
        self.N_named_parameters = None
        self.hyper_params = params
        self.introduce()

    def forward(self, X, Y: torch.Tensor):
        Y_input = Y[:, :-1]
        X_mask, Y_mask = self.get_padding_masks(X, Y_input)
        Y_input = mass_tensor(Y_input)

        enc_out = self.spectrum_emb(X)
        for enc in self.encoders:
            enc_out = enc(enc_out, pad_mask=X_mask)

        dec_out = self.peptide_emb(Y_input)
        for dec in self.decoders:
            dec_out = dec(dec_out, enc_out, pad_mask=Y_mask)

        return self.ll(dec_out)


    def get_padding_masks(self, X, Y):
        with torch.no_grad():
            is_padding = (X == torch.tensor([PAD_TOKEN, 0], device=self.hyper_params.device)).all(dim=2)
            to_bool = is_padding.int()
            X_mask = to_bool.unsqueeze(-1).expand(-1, -1, X.size(1))
            Y_mask = (Y == PAD_TOKEN).unsqueeze(-1).expand(-1, -1, Y.size(1)).int()

        return X_mask.transpose(-2, -1), Y_mask.transpose(-2, -1)


    def generate(self, X):
        self.eval()
        with torch.inference_mode():

            batch_size = X.size(0)
            Y = torch.ones((batch_size , 1), device=self.hyper_params.device, dtype=torch.int64)
            while True:
                X_mask, _ = self.get_padding_masks(X, Y)
                Y_input = mass_tensor(Y)

                enc_out = self.spectrum_emb(X)
                for enc in self.encoders:
                    enc_out = enc(enc_out, pad_mask=X_mask)

                dec_out = self.peptide_emb(Y_input)
                for dec in self.decoders:
                    dec_out = dec(dec_out, enc_out)

                # take output of last token in the sequence
                pre_prob = self.ll(dec_out)[:, -1, :]
                probs = F.softmax(pre_prob, dim=-1)
                next_aa = torch.multinomial(probs, num_samples=1, replacement=True)

                Y = torch.cat((Y, next_aa), dim=1)
                # break out if all in the batch reached <END_TOKEN>
                if Y.size(1) == 100 or (Y == END_TOKEN).count_nonzero() == batch_size:
                    break

            return Y


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
        t += f"\nd_model: {p.d_model}\nn_heads: {p.n_heads}\nn_layers: {p.n_layers}\nlr: {p.learning_rate}"
        t += f"\nd_key=d_val=d_query: {p.d_key}\nd_ff: {p.d_ff}\ndropout: {p.dropout_rate}"
        t += f"\nLen X: {p.max_spectrum_length}\nLen Y: {p.max_peptide_length}"
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
            if p.grad is None:
                print(f"{n}: None grad")
            else:
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
