"""
Compute entropy of Y using both empirical/theoretical spectrum
"""
import numpy as np

import dnner
from dnner.activations import Linear
from dnner.priors import SpikeSlab
from dnner.ensembles import Empirical, Gaussian


# Parameters
n_cols = 2000
alpha = 0.8
var_noise = 1e-8
n_runs = 10

n_rows = int(np.ceil(alpha * n_cols))

np.random.seed(42)  # set random seed

mis = []
for i in range(n_runs):
    # Generate random weights
    w0 = np.random.randn(n_rows, n_cols) / np.sqrt(n_cols)
    weights = [Empirical(w0)]

    # Compute mutual information
    layers = [SpikeSlab(0.2, 0, 1), Linear(var_noise)]
    mi = dnner.compute_mi(layers=layers, weights=weights, verbose=0)
    print("run %d, mi = %g" % (i + 1, mi))

    mis.append(mi)
print("mi (empirical distribution) = %g +/- %g" % (np.mean(mis), np.std(mis)))

# Use Gaussian i.i.d. ensemble instead
weights = [Gaussian(alpha)]
mi = dnner.compute_mi(layers=layers, weights=weights, verbose=0)
print("mi (exact distribution) = %g" % (mi))
