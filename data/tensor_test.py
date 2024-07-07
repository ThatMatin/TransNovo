import unittest
from pathlib import Path
from data.tensor import TensorBatch
from torch import Tensor as T

class TensorTest(unittest.TestCase):
    def setUp(self):
        self.t = TensorBatch(1, (0, 0))
        file = next(Path("datafiles").glob("*.msp.tensor"))
        self.t.load_file(file)

    def test_getitem(self):
        X, Y, Ch, P = self.t[1]
        self.assertIsInstance(X, T)
        self.assertIsInstance(Y, T)
        self.assertIsInstance(Ch, T)
        self.assertIsInstance(P, T)
