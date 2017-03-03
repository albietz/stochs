# stochs - fast stochastic solvers for machine learning in C++ and Cython

## Introduction
The stochs library provides efficient C++ implementations of stochastic optimization algorithms for common machine learning settings,
including situations with finite datasets augmented with random perturbations (e.g. data augmentation or dropout).
The library is mainly from Python through a Cython extension.
Currently, SGD, (S-)MISO and (N-)SAGA are supported, for dense and sparse data. See the following reference for details:

A. Bietti and J. Mairal. [Stochastic Optimization with Variance Reduction for Infinite Datasets with Finite-Sum Structure](https://arxiv.org/abs/1610.00970). arXiv 1610.00970, 2017.

## Installation
The library requires Eigen >=3.3 (the `Eigen` headers folder needs to be downloaded or symlinked into an `include` directory at the root of the repository)
and glog (`sudo apt-get install libgoogle-glog-dev`).

The Python library can be built with the following command (this requires Cython and a compiler with OpenMP support):
```
python3 setup.py build_ext -if
(or to install in the current system) python3 setup.py install
```
By default, the library is build for double precision floating points, for single precision set `USE_FLOAT = 1` in `setup.py`.

## Usage
Example usage with dropout
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
                     prox_weight=0.1) # L1 regularization weight

n = X.shape[0]
for epoch in range(100):
    if epoch == 2:
        solver.start_decay()  # start decaying the step-size after a few epochs
    idxs = np.random.choice(n, n)  # pick random indexes for one epoch

    # apply 10% dropout
    Xperturbed = X[idxs] * np.random.binomial(1, 0.9, size=X.shape) / 0.9

    # run algorithm on batch of perturbed data
    solver.iterate(Xperturbed, y[idxs], idxs)
    # with no perturbations, use: solver.iterate_indexed(X, y, idxs)

    print(solver.compute_loss(Xtest, ytest))  # compute test loss
```
