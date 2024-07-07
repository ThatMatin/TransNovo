import unittest
from pathlib import Path
from data.async_dataset import AsyncDataset
from torch import Tensor as T
from torch import all

class AsyncDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.ds = AsyncDataset(Path("datafiles"), 32)

    def test_iterator(self):
        i = 0
        for i, (X, Y, Ch, P) in enumerate(self.ds()):
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

    def tearDown(self) -> None:
        super().tearDown()
        self.ds.stop()

    def is_all_zeros(self, tensor: T) -> bool:
        b = all(tensor == 0).item()
        assert isinstance(b, bool)
        return b
