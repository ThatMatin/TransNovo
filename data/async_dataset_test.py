from typing import Iterator
import unittest
from pathlib import Path
from data.async_dataset import AsyncDataset
from torch import Tensor as T
from torch import all

from logger import get_gpu_memory_info

class AsyncDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ds = AsyncDataset(Path("datafiles/train"), 32, queue_size=50)

    def test_iterator(self):
        i = 0
        for i, (X, Y, Ch, P) in enumerate(self.ds):
            self.assertIsInstance(X, T)
            self.assertIsInstance(Y, T)
            self.assertIsInstance(Ch, T)
            self.assertIsInstance(P, T)

            self.assertFalse(self.is_all_zeros(X))
            self.assertFalse(self.is_all_zeros(Y))
            self.assertFalse(self.is_all_zeros(Ch))
            self.assertFalse(self.is_all_zeros(P))

            self.assertLessEqual(X.size(0), 32, msg=f"item: {i}")
        self.assertEqual(len(self.ds), i+1)

    def test_iterator_type(self):
        self.assertIsInstance(iter(self.ds), Iterator)

    def test_gpu_allocation(self):
        for X, Y, Ch, P in self.ds:
            total, _, allocated = get_gpu_memory_info()
            tensors_size = self.get_allocated_size(X, Y, Ch, P) * 50
            self.assertLessEqual(tensors_size, total)
            self.assertLessEqual(tensors_size * 1.5, allocated)

    def tearDown(self) -> None:
        super().tearDown()
        self.ds.stop()

    def is_all_zeros(self, tensor: T) -> bool:
        b = all(tensor == 0).item()
        assert isinstance(b, bool)
        return b

    def get_allocated_size(self, X: T, Y: T, Ch: T, P: T) -> float:
        total = 0
        total += X.numel() * X.element_size()
        total += Y.numel() * Y.element_size()
        total += Ch.numel() * Ch.element_size()
        total += P.numel() * P.element_size()
        return total / 1024 ** 2
