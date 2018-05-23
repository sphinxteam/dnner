import numpy as np

from .. import base


class Normal(base.Prior):
    r"""Compute relevant quantities for a Gaussian prior.

    Defined by :math:`P_0(x) = N(x; \mu, \sigma^2)`.

    Parameters
    ----------
    mean : float
        Prior mean :math:`\mu`.

    var : float
        Prior variance :math`\sigma^2`.
    """

    def __init__(self, mean, var):
        self.mean = mean
        self.var = var
        self.rho = self.var + self.mean ** 2

    # Compute $V (\hat{m})$ for prior
    def iter_v(self, a):
        return 1. / (a + 1. / self.var)

    # Compute $I_x (\hat{m})$ for prior
    def eval_i(self, a):
        return .5 * (self.rho * a + self.mean ** 2 / self.var) - \
            .5 * np.log(1 + a * self.var)

    # Compute $\rho$ for prior
    def eval_rho(self, rho_prev):
        return self.rho
