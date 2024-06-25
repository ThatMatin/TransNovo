import torch.nn as nn

class PeptideEmbedding(nn.Module):
    def __init__(self, 
                 R_dim: int,
                 I_dim: int,
                 d_model: int,
                 h_layer_dim: int=200,
                 device="cuda"):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(R_dim, h_layer_dim, device=device),
                nn.ReLU(),
                nn.Linear(h_layer_dim, 1, device=device),
                nn.Flatten(-2),
                nn.Linear(I_dim, h_layer_dim, device=device),
                nn.ReLU(),
                nn.Linear(h_layer_dim, d_model, device=device),
                nn.LayerNorm(d_model)
                )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class SpectrumEmbedding(nn.Module):
    def __init__(self, d_model, device="cuda"):
        super().__init__()
        self.mz_embedding = nn.Linear(1, d_model, device=device)
        self.intensity_embedding = nn.Linear(1, d_model, device=device)
        self.ln = nn.LayerNorm(d_model, device=device)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.intensity_embedding.weight)
        nn.init.kaiming_normal_(self.mz_embedding.weight)
        nn.init.zeros_(self.intensity_embedding.bias)
        nn.init.zeros_(self.mz_embedding.bias)

    def forward(self, x):
        mz_out = self.mz_embedding(x[:, :, 0].unsqueeze(-1))
        int_out = self.intensity_embedding(x[:, :, 1].unsqueeze(-1))
        out = mz_out + int_out
        return self.ln(out)
