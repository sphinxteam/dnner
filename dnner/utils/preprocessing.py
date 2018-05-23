import warnings
from copy import copy

import numpy as np
from scipy.special import erfc
import itertools

from .. import base
from ..ensembles import Empirical


def check_args(layers, weights, v0, fixed_v):
    """Check arguments of extremize and compute_entropy
    """
    # Get layers and weights
    n_layers = len(layers) - 1
    if len(weights) != n_layers:
        raise ValueError("number of layers/weights incompatible")

    # Check and initialize layers
    if not isinstance(layers[0], base.Prior):
        raise ValueError("first layer must be prior")

    for i in range(1, n_layers + 1):
        if not isinstance(layers[i], base.Activation):
            raise ValueError("all layers after first one must be activations")

    # Check if order of weights is correct
    if isinstance(weights[0], np.ndarray):
        for i in range(len(weights) - 1):
            if weights[i].shape[0] != weights[i + 1].shape[1]:
                raise ValueError("weights are ordered incorrectly")

    # Check if v0 is present in case fixed_v is True
    if any(fixed_v) and v0 is None:
        raise ValueError("if fixed_v is True, v0 should not be None")


def initialize(layers_orig, weights_orig):
    """Initialize activations/ensembles.

    Initialize activations as interface/channel depending on position, and
    ensembles depending on dtype.
    """
    n_layers = len(layers_orig) - 1

    layers = copy(layers_orig)
    for i in range(1, n_layers):
        layers[i] = layers[i].set_mode(is_output=False)
    if isinstance(layers[-1], base.Activation):
        layers[-1] = layers[-1].set_mode(is_output=True)

    weights = []
    for w in weights_orig:
        if isinstance(w, base.Ensemble):
            weights.append(w)
        elif isinstance(w, np.ndarray) or (isinstance(w, tuple) and len(w) == 2):
            weights.append(Empirical(w))
        else:
            raise NotImplementedError("other ensembles not yet implemented")

    return layers, weights


def precompute(layers, weights, verbose=0):
    """Precompute relevant quantities for extremize and compute_entropy
    """
    # Compute ratios b/w number of units, \alpha and \tilde{\alpha}
    # NOTE: \tilde{\alpha}_i = \tilde{\alpha}_{i + 1} \alpha_i
    alpha = [w.alpha for w in weights]
    alpha_t = [np.prod(alpha[:i]) for i in range(len(alpha) + 1)]

    # Compute second moments of x and z, \rho and \tilde{\rho}
    rho = []
    rho_t = []
    for i in range(len(layers) - 1):
        rho.append(layers[i].eval_rho(rho_t[-1] if i > 0 else None))
        rho_t.append(weights[i].eig_mean * rho[-1] / alpha[i])

    if verbose > 0:
        print("alpha = %s" % (alpha))
        print("alpha_t = %s" % (alpha_t))
        print("rho = %s" % (rho))
        print("rho_t = %s" % (rho_t))

    return alpha, alpha_t, rho, rho_t


def set_v0(v0, fixed_v, rho, delta=1e-10, many_v0=False):
    """Initialize v0 from argument
    """
    n_layers = len(fixed_v)

    if many_v0:  # for `compute_entropy`
        if v0 is None:
            # [\delta, \delta, ...], [\rho_i, \delta, ...], ...
            v0_i = [delta] * n_layers
            v0_ = [v0_i]
            for i in range(n_layers):
                v0_i = copy(v0_i)
                v0_i[i] = rho[i]
                v0_ += [v0_i]
        elif isinstance(v0, list) and any([v is None for v in v0]):
            v0_ = [v0]
            for i in range(len(v0)):  # replace None's by delta/rho
                if v0[i] is None:
                    v0_new = []
                    for v_i in v0_:
                        v_i[i] = delta
                        v0_new += [v_i]

                        v_r = copy(v_i)
                        v_r[i] = rho[i]
                        v0_new += [v_r]
                    v0_ = v0_new
        elif np.isscalar(v0):
            v0_ = [v0]
        elif not hasattr(v0, "__iter__"):
            raise ValueError("expected v0 to be iterable")
        else:
            v0_ = v0
    else:  # for `extremize`
        if v0 is None:
            v0_ = np.ones(n_layers)
        elif np.isscalar(v0):
            v0_ = np.full(n_layers, v0)
        elif len(v0) != n_layers:
            raise ValueError("length of v0 incompatible with number of layers")
        else:
            v0_ = np.copy(v0)

        # Ensure v0_ is an array of floats
        v0_ = np.asfarray(v0_)

    return v0_


def check_m(v, rho, verbose=0):
    r"""Compute m = \rho - v$
    """
    if rho - v < 0 and verbose:  # handle m = rho - v < 0
        warnings.warn("rho = %.4f smaller than v = %.4f" % (rho, v),
                      category=RuntimeWarning)
    return max(rho - v, 0)
