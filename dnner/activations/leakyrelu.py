import warnings

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import erfc, erfcx

from .. import base
from ..utils.integration import H2
from ..utils.preprocessing import check_m
from .relu_fast import _f


class LeakyReLU(base.Activation):
    r"""Compute relevant quantities for leaky ReLU likelihood.

    Defined by :math:`P(y | z) = \theta(y) \mathcal{N} (y; z, \sigma^2) +
    \theta(-y) \mathcal{N} (y; \gamma z, \gamma^2 \sigma^2)`.

    Parameters
    ----------
    var_noise : float
        Noise variance :math`\sigma^2`.

    gamma : float
        Leaky ReLU parameter.
    """

    def __init__(self, var_noise, gamma):
        self.var_noise = var_noise
        self.gamma = gamma

    def set_mode(self, is_output):
        if is_output:
            raise NotImplementedError("LeakyReLU output not implemented")
        else:
            return LeakyReLUInterface(self.var_noise, self.gamma)


class LeakyReLUInterface(LeakyReLU):
    # Compute expectation of $f(A, b, w, V)$ over measure
    @base.profile
    def __average(self, fun, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v

        H_f1 = H2(_f, args=(a, m, v_eff, self.gamma, fun, 0), epsrel=1e-7)
        H_f2 = H2(_f, args=(a, m, v_eff, self.gamma, fun, 1), epsrel=1e-7)

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
        return .5 * (1 + self.gamma ** 2) * (self.var_noise + rho_prev)
