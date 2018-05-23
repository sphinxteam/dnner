import numpy as np

from .. import base


class Linear(base.Activation):
    r"""Compute relevant quantities for noisy linear activation.

    Defined by :math:`P(x | z) = \mathcal{N} (x; z, \sigma^2)`.

    Parameters
    ----------
    var_noise : float
        Noise variance :math:`\sigma^2`
    """

    def __init__(self, var_noise):
        self.var_noise = var_noise
        
    def set_mode(self, is_output):
        if is_output:
            return LinearOutput(self.var_noise)
        else:
            return LinearInterface(self.var_noise)


class LinearOutput(Linear):
    # Compute $\hat{m} (m)$ for channel
    def iter_a(self, v, rho):
        return 1 / (self.var_noise + v)

    # Compute $I_z (m)$ for channel
    def eval_i(self, v, rho):
        return -.5 * np.log(2 * np.pi * np.e * (self.var_noise + v))


class LinearInterface(Linear):
    # Compute $A (A, V)$ for interface
    def iter_a(self, a, v, rho):
        return 1 / (self.var_noise + v + 1 / a)

    # Compute $V (A, V)$ for interface
    def iter_v(self, a, v, rho):
        return 1 / (a + 1 / (self.var_noise + v))

    # Compute $I_h (A, V, \rho)$ for interface
    def eval_i(self, a, v, rho):
        return .5 * a * (rho + self.var_noise) - \
            .5 * np.log(1 + a * (self.var_noise + v))

    # Compute $\rho$ for interface
    def eval_rho(self, rho_prev):
        return self.var_noise + rho_prev
