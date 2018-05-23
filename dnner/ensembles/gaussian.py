import numpy as np

from .. import base


class Gaussian(base.Ensemble):
    """ Ensemble of Gaussian i.i.d. matrices.

    Parameters
    ----------
    alpha : float
        Rows-to-columns ratio of the matrix.
    """

    def __init__(self, alpha):
        self.alpha = alpha
        self.eig_mean = alpha

    def iter_theta(self, gamma):
        r""" Compute $\vartheta$ in closed form
        """
        theta = gamma
        found_soln = True
        return theta, found_soln

    def iter_llmse(self, a, v):
        """ Compute (`v` times) Stieltjes transform of `-a * v`"""
        z = -a * v
        t = 1 + z - self.alpha
        s = (-t - np.sqrt(t ** 2 - 4 * z)) / (2 * z)
        return v * s

    def eval_f(self, a, v, theta):
        r""" Evaluate $F(A, V, \theta)$ in closed form
        """
        res = self.alpha * a * v
        return res
