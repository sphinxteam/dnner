import numpy as np
from scipy.integrate import quad


def H(f, var=1, interv=None, epsrel=1e-11):
    """Gaussian integral of 1D function using scipy.integrate.quad.
    """
    if interv is None:  # 10 standard deviations
        limit = 10 * min(1, np.sqrt(var))
        interv = [-limit, limit]

    if var > np.spacing(1.):  # if var < eps, use \delta instead
        def f_gauss(z):
            return np.exp(-.5 * z ** 2 / var) / np.sqrt(2 * np.pi * var) * f(z)
        return quad(f_gauss, interv[0], interv[1], epsrel=epsrel)[0]
    else:
        return f(0)


def __dblquad(f, lims, args=(), epsrel=1e-11):
    def int_x(y, *args):
        return quad(f, lims[0], lims[1], args=(y, *args),
                epsrel=1e-2 * epsrel)[0]
    return quad(int_x, lims[2], lims[3], args=args, epsrel=epsrel)[0]


def H2(f, interv=(-10, 10, -10, 10), epsrel=1e-11, args=()):
    """Gaussian integral of 2D function.
    """
    return __dblquad(f, interv, args=args, epsrel=epsrel)
