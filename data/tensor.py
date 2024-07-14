import torch
import os
from typing import Tuple
from logger import setup_logger

logger = setup_logger(__name__)
T = torch.Tensor

class TensorBatch():
    def __init__(self, N: int, maxes:Tuple[int, int]):
        max_x = maxes[0]
        max_y = maxes[1]
        self.X = torch.zeros(N, max_x, 2, dtype=torch.float32)
        self.Y = torch.zeros(N, max_y, dtype=torch.int64)
        self.Ch = torch.zeros(N, dtype=torch.int64)
        self.P = torch.zeros(N, dtype=torch.float32)

    def update(self, batch_idx: int, x:torch.Tensor, y: torch.Tensor, ch: int, p: float):
        self.X[batch_idx] = x
        self.Y[batch_idx] = y
        self.Ch[batch_idx] = ch
        self.P[batch_idx] = p

    def trim(self, offset: int):
        self.X = self.X[:offset]
        self.Y = self.Y[:offset]
        self.Ch = self.Ch[:offset]
        self.P = self.P[:offset]

    def save_to_file(self, filename: os.PathLike):
        self.X = self.discretize(self.X)
        checkpoint = {
                "X": self.X,
                "Y": self.Y,
                "Ch": self.Ch,
                "P": self.P,
                }
        torch.save(checkpoint, filename)
        logger.debug(f"saved {self.X.size(0)} spectra to {filename}")

    def load_file(self, filename: os.PathLike):
        checkpoint = torch.load(filename)
        self.X = checkpoint["X"]
        self.Y = checkpoint["Y"]
        self.Ch = checkpoint["Ch"]
        self.P = checkpoint["P"]
        logger.debug(f"loaded from {filename} - N: {self.X.size(0)} - maxes: ({self.X.size(1)}, {self.Y.size(1)})")

    def set_requires_grad_true(self):
        self.X.requires_grad_(True)
        self.P.requires_grad_(True)

    def get_batch_size(self) -> int:
        assert self.X.size(0) == self.Y.size(0) == self.Ch.size(0) == self.P.size(0)
        return self.X.size(0)

    def to(self, device:torch.device|str):
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.Ch = self.Ch.to(device)
        self.P = self.P.to(device)

    def __getitem__(self, index) -> Tuple[T, T, T, T]:
        # NOTE: Use detach for transferring between processes
        if isinstance(index, slice):
            return self.X[index.start:index.stop].detach(), self.Y[index.start:index.stop].detach(), \
                    self.Ch[index.start:index.stop].detach(), self.P[index.start:index.stop].detach()

        else:
            return self.X[index].detach(), self.Y[index].detach(), self.Ch[index].detach(), self.P[index].detach()

    def discretize(self, X: torch.Tensor) -> torch.Tensor:
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


