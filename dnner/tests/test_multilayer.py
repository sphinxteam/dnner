import warnings
import unittest

import numpy as np

import dnner
from dnner.priors import Normal
from dnner.activations import Linear


class MultilayerTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")

        # Generate random weights
        np.random.seed(42)
        w2 = np.random.randn(20, 50) / np.sqrt(50)
        w1 = np.random.randn(50, 100) / np.sqrt(100)

        alpha2 = float(w2.shape[0]) / w2.shape[1]
        alpha1 = float(w1.shape[0]) / w1.shape[1]
        eigvals2 = np.linalg.eigvalsh(w2.T.dot(w2))
        eigvals1 = np.linalg.eigvalsh(w1.T.dot(w1))
        self.weights = [(alpha1, eigvals1), (alpha2, eigvals2)]

    def tearDown(self):
        pass

    def test_Normal_Linear_Linear(self):
        layers = [Normal(0, 1), Linear(0), Linear(0.01)]
        entropy_expected = 0.244437
        entropy = dnner.compute_entropy(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(entropy, entropy_expected, places=6)

    def test_save_fixed_points(self):
        layers = [Normal(0, 1), Linear(0), Linear(0.01)]

        # Run 2 iterations without saving fixed points
        entropy1 = dnner.compute_entropy(layers=layers,
                                         weights=self.weights,
                                         v0=[(1, 1)],
                                         max_iter=10)

        # Run 1+1 iterations while saving fixed points
        _, extra = dnner.compute_entropy(layers=layers,
                                         weights=self.weights,
                                         return_extra=True,
                                         v0=[(1, 1)],
                                         max_iter=5)

        entropy2, _ = dnner.compute_entropy(layers=layers,
                                            weights=self.weights,
                                            return_extra=True,
                                            start_at=extra,
                                            max_iter=5)

        self.assertEqual(entropy1, entropy2)


if __name__ == "__main__":
        unittest.main()
