import warnings

import numpy as np

from .. import base
from ..utils.integration import H


class Bimodal(base.Prior):
    """
        Compute relevant quantities for bimodal, $\pm1$ prior $P_0(x) = p
        \delta(x - 1) + (1 - p) \delta(x + 1)$.

        Args:
            prob (float): prior probability of x being positive

        Attributes:
            prob (float): prior probability of x being positive
            rho (float): 2nd moment of P_0 (x)
            b0 (float): auxiliary field used throughout computations
    """

    def __init__(self, prob):
        self.prob = prob
        self.rho = 1.

        self.b0 = .5 * np.log(self.prob / (1. - self.prob))

    # Compute $f_c(A, B)$ for prior
    def __var_x(self, a, b):
        return 1. - np.tanh(self.b0 + b) ** 2

    # Compute $V (\hat{m})$ for prior
    def iter_v(self, a):
        f = lambda s: (lambda z: self.__var_x(a, a * s + np.sqrt(a) * z))
        return self.prob * H(f(+1)) + (1. - self.prob) * H(f(-1))

    # Compute $I_x (\hat{m})$ for prior
    def eval_i(self, a):
        f = lambda s: (lambda z: np.log(2 * np.cosh(self.b0 + a * s + np.sqrt(a) * z)))

        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            try:
                Hp, Hm = H(f(+1)), H(f(-1))
            except RuntimeWarning:  # handling too-large a's
                Hp, Hm = a + self.b0, a - self.b0

        return .5 * np.log(self.prob * (1 - self.prob)) - .5 * a + \
            self.prob * Hp + (1. - self.prob) * Hm

    # Compute $\rho$ for prior
    def eval_rho(self, rho_prev):
        return self.rho
