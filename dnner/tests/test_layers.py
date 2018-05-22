import warnings
import unittest

import numpy as np

import dnner
from dnner.activations import Linear, Probit, ReLU
from dnner.priors import Normal, Bimodal, SpikeSlab


class LayersTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")

        # Generate random weights
        np.random.seed(42)
        w0 = np.random.randn(100, 50) / np.sqrt(50)

        alpha = w0.shape[0] / w0.shape[1]
        eigvals = np.linalg.eigvalsh(w0.T.dot(w0))
        self.weights = [(alpha, eigvals)]

    def tearDown(self):
        pass

    def test_Normal_Linear(self):
        layers = [Normal(0, 1), Linear(0.01)]
        mi_expected = 2.501166  # could also compare to exact
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_Bimodal_Linear(self):
        layers = [Bimodal(0.5), Linear(0.01)]
        mi_expected = 0.693147
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_SpikeSlab_Linear(self):
        layers = [SpikeSlab(0.2, 0, 1), Linear(0.01)]
        mi_expected = 0.890864
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_Normal_Probit(self):
        layers = [Normal(0, 1), Probit(0.01)]
        mi_expected = 0.897281
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_Bimodal_Probit(self):
        layers = [Bimodal(0.5), Probit(0.01)]
        mi_expected = 0.692927
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_SpikeSlab_Probit(self):
        layers = [SpikeSlab(0.2, 0, 1), Probit(0.01)]
        mi_expected = 0.559198
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_Normal_ReLU(self):
        layers = [Normal(0, 1), ReLU(0.01)]
        mi_expected = 2.021801
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_Bimodal_ReLU(self):
        layers = [Bimodal(0.5), ReLU(0.01)]
        mi_expected = 0.693147
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)

    def test_SpikeSlab_ReLU(self):
        layers = [SpikeSlab(0.2, 0, 1), ReLU(0.01)]
        mi_expected = 0.799361
        mi = dnner.compute_mi(layers=layers, weights=self.weights, verbose=0)
        self.assertAlmostEqual(mi, mi_expected, places=6)


if __name__ == "__main__":
        unittest.main()
