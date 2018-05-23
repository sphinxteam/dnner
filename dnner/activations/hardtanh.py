import warnings

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import erf, erfc, erfcx

from .. import base
from ..utils.integration import H, H2
from ..utils.preprocessing import check_m
from .hardtanh_fast import _f


class HardTanh(base.Activation):
    r"""Compute relevant quantities for hard tanh likelihood.

    Parameters
    ----------
    var_noise : float
        Noise variance :math:`\sigma^2`.

    thres : float
        Threshold from linear to constant.
    """

    def __init__(self, var_noise, thres=1):
        self.var_noise = var_noise
        self.thres = thres

    def set_mode(self, is_output):
        if is_output:
            return HardTanhOutput(self.var_noise, self.thres)
        else:
            return HardTanhInterface(self.var_noise, self.thres)


class HardTanhOutput(HardTanh):
    # Compute $\hat{m} (m)$ for channel
    def iter_a(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        # Probit part
        bias = self.thres / np.sqrt(2 * v_eff)
        def f1(z):
            return (2 / np.pi) * np.exp(-(z + bias) ** 2) / erfcx(z + bias)
        H_f1 = H(f1, var=v_int)

        # Linear part
        def f2(z):
            return erf((np.sqrt(m) * z + self.thres) / np.sqrt(2 * v_eff))
        H_f2 = H(f2)

        res = (H_f1 + H_f2) / v_eff
        res -= 2 * self.thres / (self.var_noise + rho) * \
            np.exp(-.5 * self.thres ** 2 / (self.var_noise + rho)) / \
            np.sqrt(2 * np.pi * (self.var_noise + rho))

        return res

    # Compute $I_z (m)$ for channel
    def eval_i(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        # Probit part
        bias = self.thres / np.sqrt(2 * v_eff)
        def f1(z):
            erfc_z = erfc(z + bias)
            return erfc_z * np.log(.5 * erfc_z) if erfc_z > 0 else 0
        H_f1 = H(f1, var=v_int)

        # Linear part
        def f2(z):
            res = np.exp(-.5 * (np.sqrt(m) * z + self.thres) ** 2 / v_eff) / \
                    np.sqrt(2 * np.pi * v_eff) * (np.sqrt(m) * z + self.thres)
            res -= .5 * erf((np.sqrt(m) * z + self.thres) / np.sqrt(2 * v_eff)) * \
                    (1 + np.log(2 * np.pi * v_eff))
            return res
        H_f2 = H(f2)

        return H_f1 + H_f2


class HardTanhInterface(HardTanh):
    # Compute expectation of $f(A, b, w, V)$ over measure
    @base.profile
    def __average(self, fun, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v

        H_f1 = H2(_f, args=(a, m, v_eff, self.thres, fun, 0), epsrel=1e-7)
        H_f2 = H2(_f, args=(a, m, v_eff, self.thres, fun, 1), epsrel=1e-7)
        H_f3 = H2(_f, args=(a, m, v_eff, self.thres, fun, 2), epsrel=1e-7)

        return H_f1 + H_f2 + H_f3

    # Compute $A (A, V) = <-g'>$ for interface
    def iter_a(self, a, v, rho):
        return self.__average(0, a, v, rho)

    # Compute $V (A, V) = <f'>$ for interface
    def iter_v(self, a, v, rho):
        return self.__average(1, a, v, rho)

    # Compute $I_h (A, V, \rho) = <ln Z>$ for interface
    def eval_i(self, a, v, rho):
        return self.__average(2, a, v, rho)

    # Compute $\rho$ for interface
    def eval_rho(self, rho_prev):
        v_eff = self.var_noise + rho_prev

        res = self.thres ** 2 * erfc(self.thres / np.sqrt(2 * v_eff))
        res += v_eff * erf(self.thres / np.sqrt(2 * v_eff))
        res -= self.thres * np.sqrt(2 * v_eff / np.pi) * \
                np.exp(-.5 * self.thres ** 2 / v_eff)

        return res
