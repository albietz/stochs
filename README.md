# stochs - fast stochastic solvers for machine learning

## Introduction
The stochs library provides efficient C++ implementations of stochastic optimization algorithms for common machine learning settings,
including situations with finite datasets augmented with random perturbations (e.g. data augmentation or dropout).
The library is mainly used from Python through a Cython extension.
Currently, SGD, (S-)MISO and (N-)SAGA are supported, for dense and sparse data. See the following reference for details:

A. Bietti and J. Mairal. [Stochastic Optimization with Variance Reduction for Infinite Datasets with Finite-Sum Structure](https://arxiv.org/abs/1610.00970). NIPS, 2017.

## Installation
The library requires Eigen >=3.3 (it will be downloaded automatically in the `setup.py` script unless the folder or symlink `include/Eigen` already exists)
and glog. To install glog on Ubuntu, run:
```
sudo apt-get install libgoogle-glog-dev
```

The Python package can be built with the following command (this requires Cython and a compiler with OpenMP support such as gcc, you might need to change the `CC` and `CXX` environment variables on a mac):
```
python3 setup.py build_ext -if
```
By default, the library is built for double precision floating point numbers (`np.float64`), for single precision (`np.float32`) set `USE_FLOAT = 1` in `setup.py`.

## Usage
Example usage with dropout perturbations:
```py
import numpy as np
import stochs

X, y, Xtest, ytest = load_some_dataset()

solver = stochs.MISO(X.shape[1],  # number of features
                     X.shape[0],  # number of datapoints
                     alpha=1.0,   # initial step-size
                     lmbda=0.01,  # L2 regularization
                     loss=b'squared_hinge', # squared hinge loss
                     prox=b'l1',  # use L1 regularizer (by default 'none')
                     prox_weight=0.1, # L1 regularization weight
                     average=False) # no iterate averaging

n = X.shape[0]
for epoch in range(100):
    if epoch == 2:
        # start decaying the step-size after a few epochs
        # if average=True, this also starts iterate averaging
        solver.start_decay()

    # pick random indexes for one epoch
    idxs = np.random.choice(n, n)

    # apply 10% dropout
    Xperturbed = X[idxs] * np.random.binomial(1, 0.9, size=X.shape) / 0.9

    # run algorithm on batch of perturbed data
    solver.iterate(Xperturbed, y[idxs], idxs)
    # with no perturbations, use: solver.iterate_indexed(X, y, idxs)

    print(solver.compute_loss(Xtest, ytest))  # compute test loss
# access parameter vector with solver.w()
```

A more thorough example for sentiment analysis on the IMDB dataset is given in the `examples` folder, with sparse solvers and non-uniform sampling.
