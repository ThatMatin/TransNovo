import torch
import os
from typing import Tuple

from logger import setup_logger

logger = setup_logger(__name__)

class TensorBatch():
    def __init__(self, N: int, maxes:Tuple[int, int]):
        max_x = maxes[0]
        max_y = maxes[1]
        self.X = torch.zeros(N, max_x, 2, dtype=torch.float64)
        self.Y = torch.zeros(N, max_y, dtype=torch.int64)
        self.Ch = torch.zeros(N, dtype=torch.int64)
        self.P = torch.zeros(N, dtype=torch.float64)

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
        checkpoint = {
                "X": self.X,
                "Y": self.Y,
                "Ch": self.Ch,
                "P": self.P,
                }
        torch.save(checkpoint, filename)
        logger.debug(f"saved {self.X.size(0)} spectra to {filename}")



