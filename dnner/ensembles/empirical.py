import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from .. import base


class Empirical(base.Ensemble):
    """Ensemble defined by the empirical spectrum of a matrix W.

    Parameters
    ----------
    W : ndarray, shape (n_rows, n_cols)
        Matrix which spectrum will be used in computing transforms.
    """

    def __init__(self, w):
        if isinstance(w, np.ndarray):
            self.alpha = w.shape[0] / w.shape[1]
            self.eigvals = np.linalg.eigvalsh(w.T.dot(w))
        elif isinstance(w, tuple) and len(w) == 2:
            self.alpha, self.eigvals = w
        else:
            raise ValueError("input format not recognized")
        self.eig_mean = self.eigvals.mean()
        self.eig_min = self.eigvals.min()

    def iter_theta(self, gamma, tol=1e-4, verbose=0):
        r""" Compute $\vartheta$ by minimizing $(\theta - \psi)^2$
        """
        # Compute theta_max s.t. the arg. of S is smaller than \lambda_min
        discr = max((1 - self.alpha) ** 2 - 4 * self.alpha * gamma * self.eig_min, 0)
        theta_max = ((1 + self.alpha) - np.sqrt(discr)) / (2 * self.alpha)

        # Solve for theta in interval [0, theta_max]
        def psi(_theta):
            arg_S = -(1 - _theta) * (1 - self.alpha * _theta) / gamma
            res = (_theta - (1 - gamma / np.mean(1. / (self.eigvals - arg_S)))) ** 2
            if discr > 0:  # add factor to ignore trivial soln. theta = 1
                res *= _theta
            return res

        theta = minimize_scalar(psi, method="bounded", bounds=[0, theta_max])["x"]

        # Check if solution was found
        found_soln = (psi(theta) < tol)
        if not found_soln:
            warnings.warn("soln. for theta not found", category=RuntimeWarning)

        # Print info on screen
        if verbose > 1:
            if verbose > 2:
                xx = np.linspace(0, 1, 1001)
                plt.plot(xx, [psi(xx_) for xx_ in xx])
                plt.axvline(theta, c="k", ls="-.", lw=2)
                plt.axvline(theta_max, c="r", lw=3, alpha=0.5)
                plt.ylim([0, 1e-4])
                plt.show()

            print(" - theta_max = %g" % (theta_max))
            if not found_soln:
                sys.stdout.write("\033[0;31m")
                print(" - psi(theta) = %g" % (psi(theta)))
                sys.stdout.write("\033[0;0m")
            else:
                print(" - psi(theta) = %g" % (psi(theta)))

        return theta, found_soln

    def iter_llmse(self, a, v):
        """ Compute (`v` times) Stieltjes transform of `-a * v`"""
        return v * np.mean(1 / (self.eigvals + a * v))

    def eval_f(self, a, v, theta):
        r""" Compute $F(A, V, \theta)$ by averaging over empirical spectrum
        """
        res = 2 * self.alpha * theta + (self.alpha - 1) * np.log(1 - theta)
        res += np.mean(np.log(a * v * self.eigvals + \
            (1 - theta) * (1 - self.alpha * theta)))
        return res
