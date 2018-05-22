import warnings

import numpy as np
from scipy.special import erfc, erfcx

from .. import base
from ..utils.integration import H
from ..utils.preprocessing import check_m


class Probit(base.Activation):
    r"""
        Compute relevant quantities for probit likelihood $P(y | z) =
        \frac12 \erfc(-yz / \sqrt{\sigma^2})$.

        Args:
            var_noise (float): noise variance $\sigma^2$
    """

    def __init__(self, var_noise):
        self.var_noise = var_noise
        
    def set_mode(self, is_output):
        if is_output:
            return ProbitOutput(self.var_noise)
        else:
            return ProbitInterface(self.var_noise)


class ProbitOutput(Probit):
    # Compute $\hat{m} (m)$ for channel
    def iter_a(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        def f(z):
            return np.exp(-z ** 2) / erfcx(z)
        H_f = H(f, var=v_int)
        res = (2 / np.pi) * H_f / v_eff

        return res

    # Compute $I_z (m)$ for channel
    def eval_i(self, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        v_int = .5 * m / v_eff

        def f(z):
            return erfc(z) * np.log(.5 * erfc(z))
        H_f = H(f, var=v_int)
        res = H_f

        return res


class ProbitInterface(Probit):
    # Compute $A (A, V)$ for interface
    def iter_a(self, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        s_int = np.sqrt(.5 * m / (v_eff + 2 * m))

        # Perform double Gaussian integral in \phi and z
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            def f(phi):
                def g(z):
                    g1 = .5 / (erfc(+s_int * phi) +
                        np.exp(-2 * (a + np.sqrt(a) * z)) * erfc(-s_int * phi))
                    g2 = .5 / (erfc(-s_int * phi) +
                        np.exp(-2 * (a - np.sqrt(a) * z)) * erfc(+s_int * phi))
                    try:
                        g3 = -np.exp(-.5 * a) / \
                            (np.exp(np.sqrt(a) * z) * erfc(s_int * phi) +
                            np.exp(-np.sqrt(a) * z) * erfc(-s_int * phi))
                    except RuntimeWarning:  # handling overflow in exp
                        g3 = 0
                    return g1 + g2 + g3
                return H(g)
            H_f = H(f)
            res = (2 / np.pi) / v_eff * np.sqrt(v_eff / (v_eff + 2 * m)) * H_f

        return res

    # Compute $V (A, V)$ for interface
    def iter_v(self, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        s_int = np.sqrt(.5 * m / v_eff)

        # Perform double Gaussian integral in \phi and z
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            def f(phi):
                def g(z):
                    try:
                        field = .5 * np.log(2 / erfc(-s_int * phi) - 1)
                        res = np.tanh(a + np.sqrt(a) * z + field)
                    except RuntimeWarning:  # handling large v_int
                        res = -np.sign(phi)
                    return res
                return erfc(s_int * phi) * H(g)
            H_f = H(f)
            res = max(0, 1 - H_f)

        return res

    # Compute $I_h (A, V, \rho)$ for interface
    def eval_i(self, a, v, rho):
        m = check_m(v, rho)
        v_eff = self.var_noise + v
        s_int = np.sqrt(.5 * m / v_eff)

        # Perform double Gaussian integral in \phi and z
        def f(phi):
            def g(z):
                res = erfc(-s_int * phi)
                res *= np.exp(-2 * a - 2 * np.sqrt(a) * z)
                res += erfc(s_int * phi)

                if res > 0:
                    res = np.log(res)
                else:
                    res = -2 * a - 2 * np.sqrt(a) * z + np.log(erfc(-s_int * phi))

                return res
            return erfc(s_int * phi) * H(g)
        H_f = H(f)
        res = .5 * a - np.log(2) + H_f

        return res

    # Compute $\rho$ for interface
    def eval_rho(self, rho_prev):
        return 1
