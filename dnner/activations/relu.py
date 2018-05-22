import warnings

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import erfc, erfcx

from .. import base
from ..utils.integration import H, H2
from ..utils.preprocessing import check_m
from .relu_fast import _f


class ReLU(base.Activation):
    r"""
        Compute relevant quantities for ReLU likelihood $P(y | z) =
        \theta(y) \mathcal{N} (y; z, \sigma^2) + \delta(y) \frac{1}{2} {\rm
        erfc} (z / \sqrt{2 \sigma^2})$.

        Args:
            var_noise (float): noise variance $\sigma^2$
    """

    def __init__(self, var_noise):
        self.var_noise = var_noise

    def set_mode(self, is_output):
        if is_output:
            return ReLUOutput(self.var_noise)
        else:
            return ReLUInterface(self.var_noise)


class ReLUOutput(ReLU):
    # Compute $\hat{m} (m)$ for channel
    def iter_a(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        def f(z):
            return (2 / np.pi) * np.exp(-z ** 2) / erfcx(-z)
        H_f = H(f, var=v_int)
        res = .5 / v_eff * (H_f + 1)

        return res

    # Compute $I_z (m)$ for channel
    def eval_i(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        def f(z):
            return erfc(z) * np.log(.5 * erfc(z))
        H_f = H(f, var=v_int)
        res = .5 * (H_f - .5 * np.log(2 * np.pi * np.e * v_eff))

        return res


class ReLUInterface(ReLU):
    # Compute expectation of $f(A, b, w, V)$ over measure
    def __average(self, fun, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v

        H_f1 = H2(_f, args=(a, m, v_eff, 0, fun, 0))
        H_f2 = H2(_f, args=(a, m, v_eff, 0, fun, 1))

        return H_f1 + H_f2

    # Compute $A (A, V)$ for interface
    def iter_a(self, a, v, rho):
        return self.__average(0, a, v, rho)

    # Compute $V (A, V)$ for interface
    def iter_v(self, a, v, rho):
        return self.__average(1, a, v, rho)

    # Compute $I_h (A, V, \rho)$ for interface
    def eval_i(self, a, v, rho):
        return self.__average(2, a, v, rho)

    # Compute $\rho$ for interface
    def eval_rho(self, rho_prev):
        return .5 * (self.var_noise + rho_prev)
