import numpy as np

from .. import base
from ..utils.integration import H


class SpikeSlab(base.Prior):
    """
        Compute relevant quantities for spike and slab prior $P_0(x) = p
        \mathcal{N} (x; \mu, \sigma^2) + (1 - p) \delta(x)$.

        Args:
            prob (float): prior probability $p$ of x being different from zero
            mean (float): mean $\mu$ of the Normal component
            var(float): variance $\sigma^2$ of the Normal component

        Attributes:
            prob (float): prior probability $p$ of x being different from zero
            mean (float): mean $\mu$ of the Normal component
            var(float): variance $\sigma^2$ of the Normal component
            rho (float): 2nd moment of P_0 (x)
    """

    def __init__(self, prob, mean, var):
        self.prob = prob
        self.mean = mean
        self.var = var
        self.rho = prob * (var + mean ** 2)

    # Compute $f_c(A, B)$ for prior
    def __var_x(self, a, b):
        m_g = (b * self.var + self.mean) / (1. + a * self.var)
        v_g = self.var / (1 + a * self.var)
        p_s = self.prob / (self.prob + (1 - self.prob) *
            np.sqrt(1 + a * self.var) * np.exp(-.5 * m_g ** 2 / v_g +
            .5 * self.mean ** 2 / self.var))
        return p_s * v_g + p_s * (1 - p_s) * m_g ** 2

    # Compute $V (\hat{m})$ for prior
    def iter_v(self, a):
        a_s = 1 + a * self.var
        f = lambda s: (lambda z: self.__var_x(a, s * a * self.mean +
            np.sqrt(a_s ** s * a) * z))
        return (1 - self.prob) * H(f(0)) + self.prob * H(f(1))

    # Compute $I_x (\hat{m})$ for prior
    def eval_i(self, a):
        b_s = self.mean ** 2 / self.var
        a_s = 1 + a * self.var
        f = lambda s: (lambda z: np.log((1 - self.prob) *
            np.exp(-.5 * (np.sqrt(a_s ** s * a) * z + a_s ** s *
            self.mean / self.var) ** 2 / (a + 1 / self.var)) +
            self.prob * np.exp(-.5 * b_s) / np.sqrt(a_s)))
        return (1 - self.prob) * (H(f(0)) + .5 * (a * self.var + b_s) / a_s) + \
            self.prob * (H(f(1)) + .5 * (a * self.var + b_s * a_s))

    # Compute $\rho$ for prior
    def eval_rho(self, rho_prev):
        return self.rho
