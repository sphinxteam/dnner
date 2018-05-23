from time import time

import numpy as np


def compute_gaussian_entropy(weights, var_noise):
    """Compute entropy of a Gaussian channel
    """
    alpha = weights.shape[0] / weights.shape[1]
    eigvals = np.linalg.eigvalsh(weights.dot(weights.T))
    return .5 * alpha * np.mean(np.log(2 * np.pi * np.e * (var_noise + eigvals)))


def binary_search(f, bounds, tol=1e-3, verbose=1):
    """Perform binary search on binary-valued function f in a given interval
    """

    # Compute function at endpoints
    x_l, x_r = bounds
    f_l = f(x_l)
    if verbose > 0:
        print(" - x_l = %g, f_l = %d" % (x_l, f_l))
    f_r = f(x_r)
    if verbose > 0:
        print(" - x_r = %g, f_r = %d" % (x_r, f_r))

    if not (f_l == 0 and f_r == 1):
        raise ValueError("bounds have not been chosen appropriately")

    # Perform bissection
    while x_r - x_l > tol:
        start_time = time()
        x_m = .5 * (x_l + x_r)
        f_m = f(x_m)
        elapsed = time() - start_time
        if verbose > 0:
            print(" - x_m = %g, f_m = %d (%.2fs)" % (x_m, f_m, elapsed))

        if f_m == 0:
            x_l = x_m
        else:
            x_r = x_m

    return x_l, x_r
