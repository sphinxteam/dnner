"""
Compute mutual information as a function of MSE for SpikeSlab-Linear model
"""
import numpy as np
import matplotlib.pyplot as plt

import dnner
from dnner.activations import Linear
from dnner.priors import SpikeSlab


np.random.seed(42)

plt.rc("text", usetex=True)
plt.rc("font", **{"family": "serif", "serif": "Computer Modern"}, size=16)

# Parameters
n_cols = 2000
var_noise = 1e-5

fig, axs = plt.subplots(2, 2, figsize=(7, 7))

alphas = [0.29, 0.37, 0.4, 0.51]
for k, alpha in enumerate(alphas):
    n_rows = int(np.ceil(alpha * n_cols))

    # Compute mutual information for fixed V's
    layers = [SpikeSlab(0.3, 1, 0.5), Linear(var_noise)]
    w0 = np.random.randn(n_rows, n_cols) / np.sqrt(n_cols)
    weights = [(alpha, np.linalg.eigvalsh(w0.T.dot(w0)))]  # save eigenvalues

    vs = np.logspace(-6, 0, 100)
    phis = np.zeros(len(vs))
    for (i, v) in enumerate(vs):
        phis[i] = dnner.compute_entropy(layers=layers, weights=weights,
                v0=v, fixed_v=True, tol=1e-7, max_iter=20)
        print("v = %g, phi = %g" % (v, phis[i]))

    # Compute quantities (MI, MMSE) at fixed-points
    e_r, x_r = dnner.compute_entropy(layers=layers, weights=weights, v0=1,
                                     return_extra=True)
    e_i, x_i = dnner.compute_entropy(layers=layers, weights=weights, v0=1e-3,
                                     return_extra=True)

    # Plot results
    axs[k // 2, k % 2].plot(vs, phis, "k-", lw=2)
    axs[k // 2, k % 2].plot(x_r["mmse"], e_r, "ro", ms=10, alpha=0.5)
    axs[k // 2, k % 2].plot(x_i["mmse"], e_i, "bo", ms=10, alpha=0.5)
    axs[k // 2, k % 2].set_xscale("log")
    axs[k // 2, k % 2].set_title(r"$\alpha = %g$" % (alpha))
    axs[k // 2, k % 2].patch.set_alpha(0.2)

    if k % 2 == 0:
        axs[k // 2, k % 2].set_ylabel(r"$\phi_{\rm RS} ({\rm MSE})$")
    if k // 2 == 1:
        axs[k // 2, k % 2].set_xlabel(r"${\rm MSE}$")

axs[0, 0].patch.set_facecolor("red")
axs[0, 1].patch.set_facecolor("red")
axs[1, 0].patch.set_facecolor("orange")
axs[1, 1].patch.set_facecolor("green")

fig.set_tight_layout(True)
fig.savefig("mmse.pdf", bbox_inches="tight")
