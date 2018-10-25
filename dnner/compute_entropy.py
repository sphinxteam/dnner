import numpy as np

from .extremize import extremize_fp, extremize_vamp, convert
from .utils.preprocessing import check_args, initialize, precompute, set_v0


def compute_entropy(layers, weights, v0=None, fixed_v=False, start_at=None,
        compute_minimum=True, return_extra=False, max_iter=250, tol=1e-11,
        tol_test=1e-3, verbose=0, use_vamp=False):
    """Compute :math:`H(Y)` for a given choice of layers and weights

    Parameters
    ----------
    layers : list, shape = [n_layers + 1]
        Model specification, i.e. priors and likelihoods

    weights : list, shape = [n_layers]
        Ensemble to which each weight matrix belongs

    Returns
    -------
    entropies : {float, list}, shape = [len(v0s)] if list
        Free energy for different initializations

    extras : list of dicts
        List of dictionaries, each containing mutual information (`mi`),
        MMSE (`mmse`) and fixed points (`fixed_points`) for different
        initializations
    """
    # Pre-processing and initialization
    n_layers = len(layers) - 1

    if isinstance(fixed_v, int):
        fixed_v = n_layers * [fixed_v]

    check_args(layers, weights, v0, fixed_v)
    layers, weights = initialize(layers, weights)
    alpha, alpha_t, rho, rho_t = precompute(layers, weights)
    v0s = set_v0(v0, fixed_v, rho, many_v0=True)

    extremize_ = extremize_vamp if use_vamp else extremize_fp

    entropies = []
    extras = []

    # Check for multiple minima
    if start_at is None and len(v0s) > 1:
        for v0_ in v0s:
            # Run extremization
            fixed_points_ = extremize_(layers=layers, weights=weights, v0=v0_,
                     fixed_v=fixed_v, start_at=None, max_iter=max_iter,
                     tol=tol_test if compute_minimum else tol, verbose=verbose)

            # Convert fixed points to our notation if using ML-VAMP
            fixed_points = convert(fixed_points_, alpha) if use_vamp else fixed_points_

            # Compute $\phi$ at fixed-point
            entropy = __phi(fixed_points, layers, weights, alpha, alpha_t, rho, rho_t)
            mi = entropy + alpha_t[-1] * layers[-1].eval_i(0, rho_t[-1])
            mmse = fixed_points[0][0]

            # Print info on screen, store entropy and extras (MI, MMSE, etc.)
            if verbose > 0:
                print("got entropy = %g starting from v0 = %s" % (entropy, v0_))

            entropies.append(entropy)

            extra = dict()
            extra["mi"] = mi
            extra["mmse"] = mmse
            extra["fixed_points"] = fixed_points_
            extras.append(extra)

    # Compute entropy for given initial condition
    if compute_minimum or start_at is not None or len(v0s) == 1:
        if start_at is None and len(v0s) > 1:  # use test minimum as init cond 
            min_at = np.argmin(entropies)
            v0_ = v0s[min_at]
            start = extras[min_at]["fixed_points"]
        elif start_at is not None:  # use start_at as init cond
            v0_ = None
            start = start_at["fixed_points"]
        else:  # use (sole) v0 as init cond
            v0_ = v0s[0]
            start = None

        fixed_points_ = extremize_(layers=layers, weights=weights, v0=v0_,
                 fixed_v=fixed_v, start_at=start, max_iter=max_iter,
                 tol=tol, verbose=verbose)
        fixed_points = convert(fixed_points_, alpha) if use_vamp else fixed_points_
        entropies = __phi(fixed_points, layers, weights, alpha, alpha_t, rho, rho_t)

        if return_extra:
            mi = entropies + alpha_t[-1] * layers[-1].eval_i(0, rho_t[-1])
            mmse = fixed_points[0][0]

            extras = dict()
            extras["mi"] = mi
            extras["mmse"] = mmse
            extras["fixed_points"] = fixed_points_

    if return_extra:
        return entropies, extras
    else:
        return entropies


def compute_mi(layers, weights, v0=None, fixed_v=False, start_at=None,
               max_iter=250, tol=1e-11, tol_test=1e-3, verbose=0):
    """ Compute :math:`I(X; Y)` for a given choice of layers and weights
    """
    _, extra  = compute_entropy(layers=layers, weights=weights, v0=v0,
            fixed_v=fixed_v, start_at=None, compute_minimum=True,
            return_extra=True, max_iter=max_iter, tol=tol,
            tol_test=tol_test, verbose=verbose)
    return extra["mi"]


def compute_all(layers, weights, always_include=1, start_at=None,
                max_iter=250, tol=1e-11, tol_test=1e-3, verbose=0,
                use_vamp=False):
    """ Compute entropies, MIs and MMSEs for all hidden variables
    """
    entropies = []
    extras = []

    # Handle always_include >= len(layers)
    always_include = min(always_include, len(layers) - 1)

    for i in range(always_include, len(layers)):
        # Pick subset of layers and weights appropriately
        layers_ = layers[:i + 1]
        if i != len(layers):  # borrow var_noise from original output
            var_noise_orig = layers_[-1].var_noise
            layers_[-1].var_noise = layers[-1].var_noise
        weights_ = weights[:i]

        # start_at should be a list of 'extra' dictionaries
        if start_at is not None:
            start_i = start_at[i - always_include]
        else:
            start_i = None

        # Compute entropy for subset of layers and weights
        entropy, extra = compute_entropy(layers=layers_, weights=weights_,
                v0=None, fixed_v=False, start_at=start_i,
                compute_minimum=True, return_extra=True, max_iter=max_iter,
                tol=tol, tol_test=tol_test, verbose=verbose, use_vamp=use_vamp)

        # Reset var_noise
        layers_[-1].var_noise = var_noise_orig

        entropies.append(entropy)
        extras.append(extra)

    return entropies, extras


def __phi(fixed_points, layers, weights, alpha, alpha_t, rho, rho_t):
    r"""Compute :math:`\phi` (see paper for details)

    Notes
    -----
    In the current convention, `alpha_t[0]` = :math:`\tilde{\alpha}_0 = 1`
    and `alpha[0]` = :math:`\alpha_1`
    """
    n_layers = len(layers) - 1
    v, a, v_in, a_in, theta = fixed_points

    # \frac12 \sum_{\ell = 1}^L ...
    entr1 = 0
    for i in range(n_layers):
        term_sum = a_in[i] * (rho[i] - v[i])
        term_sum -= alpha[i] * a[i] * v_in[i]
        term_sum += weights[i].eval_f(a[i], v[i], theta[i])

        entr1 += .5 * alpha_t[i] * term_sum

    # K (\tilde{A}, \tilde{V}; \tilde{\rho})
    entr2 = alpha_t[-1] * layers[-1].eval_i(v_in[-1], rho_t[-1])
    for i in range(1, n_layers):
        entr2 += alpha_t[i] * layers[i].eval_i(a_in[i], v_in[i - 1], rho_t[i - 1])
    entr2 += layers[0].eval_i(a_in[0])

    return entr1 - entr2
