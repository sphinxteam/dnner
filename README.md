# dnner - DNNs Entropy from Replicas

The code in this package computes the entropy, mutual information and MMSE
of multi-layer GLMs given orthogonally-invariant matrices of arbitrary
spectrum.

## Instructions

**Install package**

First make sure you have all the requirements installed

- Python 3.x
- Numpy
- Matplotlib
- Scipy
- Cython

Then type

```
python setup.py install --user
```

You can then try the `Demo.ipynb` notebook, and scripts in the `examples`
folder.

**Adding new priors/activations/ensembles**

In order to add new priors/activations, new classes should be written and
added to the `dnner/{priors, activations, ensembles}` folders. Look at the
files already present in these folders for examples; the methods in them
(`iter_a`, `iter_v`, `iter_theta`, `eval_i`, ...) should be reimplemented.

After implementing the new class, you can add it to the `__init__.py` inside
`priors`/`activations`/`ensembles` so that it can be more easily imported.

Currently the following priors are available

- `Normal`
- `Bimodal`
- `SpikeSlab`

as well as the following activations

- `Linear`
- `Probit`
- `ReLU`
- `LeakyReLU`
- `HardTanh`

and the following ensembles

- `Gaussian`
- `Empirical`

## Troubleshooting

**I keep get warnings throughout the iteration, should I be worried about
it?**

Numerical integrations performed for the (leaky) ReLU and the hard tanh are
a bit tricky: the integrator might at occasion complain about lack of
precision. Despite of that, the final result seems in our experience to be
consistent. In any case, a deeper study of the integration procedure should
be performed at some point.

**Do I need to have noise in my activations?**

It is essential to have noise in the outermost activation so that quantities
do not diverge. Usually one can get by with zero noise in the inner layers;
however, if the variables in the model are discrete, noise should also be
added there.

**Can it happen that the iteration does not converge?**

As described in the Supplementary Material of our paper, the fixed-point
iteration we use depends on a solution to a particular equation being found
at each step, and occasionaly this might not happen. In that case, using the
ML-VAMP state evolution instead of the fixed-point iteration should lead
to better results.

The ML-VAMP state evolution is in general more stable, but rarely it might
also happen that variances/precisions become negative. In our experience
however, one of the two schemes will always work.
