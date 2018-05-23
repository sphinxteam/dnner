"""
Compute entropy of Y in fixed Normal-HardTanh-ReLU model, with both methods
"""
from time import time

import numpy as np

import dnner
from dnner.priors import Normal
from dnner.activations import HardTanh, ReLU


# Parameters
n_cols1 = 1000
alpha1 = 1.5
alpha2 = 0.75
var_noise = 1e-5

n_cols2 = n_rows1 = int(np.ceil(alpha1 * n_cols1))
n_rows2 = int(np.ceil(alpha2 * n_cols2))

# Generate random weights
np.random.seed(42)  # set random seed
w1 = np.random.randn(n_rows1, n_cols1) / np.sqrt(n_cols1)
w2 = np.random.randn(n_rows2, n_cols2) / np.sqrt(n_cols2)

# Pre-compute spectra
start_time = time()
eigvals1 = np.linalg.eigvalsh(w1.T.dot(w1))
eigvals2 = np.linalg.eigvalsh(w2.T.dot(w2))
print("Computed spectra in %gs" % (time() - start_time))

weights = [(alpha1, eigvals1), (alpha2, eigvals2)]  # from x to y
layers = [Normal(0, 1), HardTanh(0), ReLU(var_noise)]

# Compute entropy using both our scheme and ML-VAMP
print("Computing entropy with fixed-point iteration...")
start_time = time()
e1, x1 = dnner.compute_entropy(layers=layers, weights=weights,
        return_extra=True, max_iter=100, tol=1e-7, verbose=0,
        v0=1e-10, use_vamp=False)
elapsed1 = time() - start_time

fp1 = np.hstack(x1["fixed_points"])
print("Entropy = %g (ran in %gs)" % (e1, elapsed1))

print("Computing entropy with the ML-VAMP SE...")
start_time = time()
e2, x2 = dnner.compute_entropy(layers=layers, weights=weights,
        return_extra=True, max_iter=100, tol=1e-7, verbose=0,
        v0=1e-10, use_vamp=True)
elapsed2 = time() - start_time

fp2 = np.hstack(x2["fixed_points"])
print("Entropy = %g (ran in %gs)" % (e2, elapsed2))
