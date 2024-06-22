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
        self.embedding = nn.Linear(2, d_model, device=device)
        self.ln = nn.LayerNorm(d_model, device=device)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.embedding.weight, nonlinearity='relu')
        if self.embedding.bias is not None:
            nn.init.zeros_(self.embedding.bias)

    def forward(self, x):
        out = self.embedding(x)
        return self.ln(out)
