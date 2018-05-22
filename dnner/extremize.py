import warnings

import numpy as np

from .utils.preprocessing import check_args, initialize, precompute, set_v0


def extremize_fp(layers, weights, v0=None, fixed_v=False, start_at=None,
                 max_iter=250, tol=1e-13, verbose=1):
    """Solve saddle-point eqs. using fixed-point iteration

    Parameters
    ----------
    layers : list, shape = [n_layers + 1]
        Model specification, i.e. priors and likelihoods

    weights : list, shape = [n_layers]
        Ensemble to which each weight matrix belongs
    """

    # Pre-processing and initialization
    n_layers = len(layers) - 1

    if isinstance(fixed_v, int):
        fixed_v = n_layers * [fixed_v]

    check_args(layers, weights, v0, fixed_v)
    layers, weights = initialize(layers, weights)
    alpha, alpha_t, rho, rho_t = precompute(layers, weights)
    v0 = set_v0(v0, fixed_v, rho)

    # Initialize remaining order parameters
    if start_at is None:
        v = v0
        a = np.array([.5 / vv if vv != 0 else 1. for vv in v])
        a_in = np.zeros(n_layers)
        v_in = np.zeros(n_layers)
        theta = np.zeros(n_layers)
    else:
        v, a, v_in, a_in, theta = start_at

    found_soln = np.ones(n_layers, dtype=bool)

    # Check if V_\ell == 0 in any of layers
    frozen_layer = np.zeros(n_layers, dtype=bool)
    for i in range(n_layers):
        if v[i] == 0:
            a_in[i] = alpha[i] * a[i]
            v_in[i] = 0
            theta[i] = 0

            frozen_layer[i] = True

    # Store v_diff
    # all_diff = []

    # Iterate scheme
    for t in range(max_iter):
        if any(fixed_v):
            v_old = np.copy(v_in)
        else:
            v_old = np.copy(v)

        # Iterate through layers computing \tilde{V}, A and \tilde{A}
        for i in range(n_layers - 1, -1, -1):
            if not frozen_layer[i]:
                # Compute \vartheta^{(t)} and \tilde{V}
                theta[i], found_soln[i] = weights[i].iter_theta(a[i] * v[i])
                v_in[i] = theta[i] / a[i]

            if i == n_layers - 1:
                a[i] = layers[i + 1].iter_a(v_in[i], rho_t[i])
            else:
                a[i] = layers[i + 1].iter_a(a_in[i + 1], v_in[i], rho_t[i])

            if not frozen_layer[i]:
                # Compute \vartheta^{(t + 1/2)} and \tilde{A}
                theta[i], found_soln[i] = weights[i].iter_theta(a[i] * v[i])
                a_in[i] = alpha[i] * theta[i] / v[i]

        # Compute V_\ell for all layers
        for i in range(n_layers - 1, -1, -1):
            if not fixed_v[i] and not frozen_layer[i]:
                if i == 0:
                    v[i] = layers[i].iter_v(a_in[i])
                else:
                    v[i] = layers[i].iter_v(a_in[i], v_in[i - 1], rho_t[i - 1])

        # Check if V_\ell == 0 in any of the layers
        for i in range(n_layers):
            if v[i] == 0:
                a_in[i] = alpha[i] * a[i]
                v_in[i] = 0
                theta[i] = 0

                frozen_layer[i] = True
                found_soln[i] = True

        # Check for numerical errors
        if any(np.isnan(v)):
            raise RuntimeError("quantities diverging, try increasing noise")

        # Print iteration status on screen
        v_diff = np.sum(np.abs((v_in if any(fixed_v) else v) - v_old))
        if verbose > 0:
            print("t = %d, conv = %g" % (t, v_diff))
            for i in range(n_layers):
                print("layer %d: a = %g, v_in = %g, a_in = %g, v = %g" % \
                    (i, a[i], v_in[i], a_in[i], v[i]))

                if verbose > 1:
                    print(" - gamma = %g" % (a[i] * v[i]))
                    print(" - theta = %g" % (theta[i]))

        # Check for convergence
        if v_diff < tol and all(found_soln):
            break

        # Experimental feature: halt iteration if change is too slow
        # all_diff.append(v_diff)
        # if t > 10:
            # latest_diff = all_diff[-10:]
            # mean_latest = np.mean(np.diff(latest_diff) / latest_diff[1:])
            # if np.abs(mean_latest) < 1e-3:
                # warnings.warn("iter. is not convergent or is slowly converging")
                # break

    # Raise warnings/exceptions if iteration didn't finished as expected
    if v_diff > tol and v[0] > 0:
        warnings.warn("iter. didn't converge to desired precision", RuntimeWarning)

    if not all(found_soln) and v[0] > 0:
        raise RuntimeError("soln. not found, try increasing max_iter")

    return v, a, v_in, a_in, theta


def extremize_vamp(layers, weights, v0=None, fixed_v=False, start_at=None,
                   max_iter=250, tol=1e-13, verbose=1):
    """Solve saddle-point eqs. using the state evolution of ML-VAMP

    Parameters
    ----------
    layers : list, shape = [n_layers + 1]
        Model specification, i.e. priors and likelihoods

    weights : list, shape = [n_layers]
        Ensemble to which each weight matrix belongs
    """

    # Pre-processing and initialization
    n_layers = len(layers) - 1

    if isinstance(fixed_v, int):
        fixed_v = n_layers * [fixed_v]

    check_args(layers, weights, v0, fixed_v)
    layers, weights = initialize(layers, weights)
    alpha, alpha_t, rho, rho_t = precompute(layers, weights)
    v0 = set_v0(v0, fixed_v, rho)

    frozen_layer = np.zeros(n_layers, dtype=bool)

    # Initialize remaining order parameters
    if start_at is None:
        v1x = v0
        v2x = np.empty(n_layers)
        a1x = np.array([1 / vv if vv != 0 else np.inf for vv in v1x])
        a2x = np.empty(n_layers)

        v1z = np.empty(n_layers)
        v2z = np.empty(n_layers)
        a1z_i = np.ones(n_layers)
        a2z = np.empty(n_layers)

        g = np.empty(n_layers)
    else:
        v1x, v2x, a1x, a2x, v1z, v2z, a1z_i, a2z, g = start_at

    # Iterate ML-VAMP state evolution
    for t in range(max_iter):
        v_old = np.copy(v1x)

        # Forward pass
        for i in range(n_layers):
            # Variance on x_-
            if i == 0:
                v1x[i] = layers[i].iter_v(a1x[i])
            else:
                v1x[i] = layers[i].iter_v(a1x[i], 1 / a2z[i - 1], rho_t[i - 1])

            # Check if variance on x_- is zero
            if v1x[i] < np.spacing(1):
                v2x[i] = v1z[i] = v2z[i] = a1z_i[i] = 0
                a1x[i] = a2x[i] = a2z[i] = 1e10

                frozen_layer[i] = True

            # Proceed if variance on x_- is not zero
            if not frozen_layer[i]:
                # Message from x_- to x_+
                a2x[i] = 1 / v1x[i] - a1x[i]

                # Variance on x_+/z_- and message from z_- to z_+
                v2x[i] = weights[i].iter_llmse(a2x[i], a1z_i[i])
                v1z[i] = a1z_i[i] * (1 - a2x[i] * v2x[i]) / alpha[i]
                a2z[i] = 1 / v1z[i] - 1 / a1z_i[i]

        # Backward pass
        for i in range(n_layers - 1, -1, -1):
            # Variance on z_+
            if i == n_layers - 1:
                g[i] = layers[i + 1].iter_a(1 / a2z[i], rho_t[i])
            else:
                g[i] = layers[i + 1].iter_a(a1x[i + 1], 1 / a2z[i], rho_t[i])

            # Proceed if variance on x_- is not zero
            if not frozen_layer[i]:
                # Message from z_+ to z_-
                a1z_i[i] = 1 / g[i] - 1 / a2z[i]

                # Variance on x_+ and message from x_+ to x_-
                v2x[i] = weights[i].iter_llmse(a2x[i], a1z_i[i])
                a1x[i] = 1 / v2x[i] - a2x[i]

        # Compute v2z for displaying purposes only
        for i in range(n_layers):
            v2z[i] = 1 / a2z[i] - g[i] / a2z[i] ** 2

        # Check for numerical errors
        if any(np.isnan(v1x)):
            raise RuntimeError("quantities diverging, try increasing noise")

        # Print iteration status on screen
        v_diff = np.sum(np.abs(v1x - v_old))
        if verbose > 0:
            print("t = %d, conv = %g" % (t, v_diff))
            for i in range(n_layers):
                print("layer %d: v1x = %g, v2x = %g, v1z = %g, v2z = %g" % \
                    (i, v1x[i], v2x[i], v1z[i], v2z[i]))

        # Check for convergence
        if v_diff < tol and t > 0:
            break

    # Raise warnings/exceptions if iteration didn't finished as expected
    if v_diff > tol and v1x[0] > 0:
        warnings.warn("iter. didn't converge to desired precision", RuntimeWarning)

    return v1x, v2x, a1x, a2x, v1z, v2z, a1z_i, a2z, g


def convert(fixed_points, alpha):
    """Convert from ML-VAMP fixed points to ours"""

    v1x, v2x, a1x, a2x, v1z, v2z, a1z_i, a2z, g = fixed_points
    n_layers = len(v1x)

    # Transform fixed points
    v = v1x
    v_in = 1 / a2z
    a = g

    # Deal with zero variances
    a_in = np.zeros(n_layers)
    theta = np.zeros(n_layers)
    for i in range(n_layers):
        if v[i] > np.spacing(1):
            a_in[i] = a1x[i]
            theta[i] = 1 / (1 + a2z[i] * a1z_i[i])
        else:  # FIXME: are we sure this make sense?
            a_in[i] = alpha[i] * a[i]
            theta[i] = 0

    return v, a, v_in, a_in, theta
