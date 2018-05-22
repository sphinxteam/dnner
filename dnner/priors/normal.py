import numpy as np

from .. import base


class Normal(base.Prior):
    """
        Compute relevant quantities for a Gaussian prior $P_0(x) = N(x; \mu,
        \sigma^2)$.

        Args:
            mean (float): prior mean $\mu$
            var (float): prior variance $\sigma^2$

        Attributes:
            mean (float): prior mean $\mu$
            var (float): prior variance $\sigma^2$
            rho (float): 2nd moment of P_0 (x)
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
