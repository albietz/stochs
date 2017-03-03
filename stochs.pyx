# distutils: include_dirs = /scratch/clear/abietti/local/include
# cython: boundscheck=False

import numpy as np
import scipy.sparse as sp
cimport numpy as np

from cpython.ref cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, int64_t
from libcpp.string cimport string
from libcpp cimport bool

np.import_array()

IF USE_FLOAT:
    ctypedef float Double
    npDOUBLE = np.NPY_FLOAT32
    dtype = np.float32
ELSE:
    ctypedef double Double
    npDOUBLE = np.NPY_FLOAT64
    dtype = np.float64


cdef extern from "numpy/arrayobject.h":
    void PyArray_SetBaseObject(np.ndarray, PyObject*)

cdef extern from "solvers/common.h" namespace "solvers":
    void _center "solvers::center"(
        Double* const XData, const size_t rows, const size_t cols) nogil
    void _normalize "solvers::normalize"(
        Double* const XData, const size_t rows, const size_t cols) nogil
    void _normalizeSparse "solvers::normalize"(
        const size_t rows, const size_t cols, const size_t nnz,
        const int32_t* const indptr, const int32_t* const indices,
        Double* const values) nogil

def center(Double[:,::1] X not None):
    _center(&X[0,0], X.shape[0], X.shape[1])

def normalize_dense(Double[:,::1] X not None):
    _normalize(&X[0,0], X.shape[0], X.shape[1])

def normalize_sparse(X):
    cdef Double[:] values = X.data
    cdef int32_t[:] indptr = X.indptr
    cdef int32_t[:] indices = X.indices
    _normalizeSparse(X.shape[0], X.shape[1], X.nnz,
                     &indptr[0], &indices[0], &values[0])

def normalize(X):
    if isinstance(X, sp.csr_matrix):
        normalize_sparse(X)
    else:
        normalize_dense(X)

cdef extern from "solvers/Solver.h" namespace "solvers":
    void iterateBlock "solvers::Solver::iterateBlock"[SolverT](
            SolverT& solver,
            const size_t blockSize,
            const Double* const XData,
            const Double* const yData,
            const int64_t* const idxData) nogil

    void iterateBlockIndexed "solvers::Solver::iterateBlockIndexed"[SolverT](
            SolverT& solver,
            const size_t dataSize,
            const Double* const XData,
            const Double* const yData,
            const size_t blockSize,
            const int64_t* const idxData) nogil

    void setQ "solvers::Solver::setQ"[SolverT](
            SolverT& solver,
            const size_t n,
            const Double* const qData)

    # for sparse data
    void iterateBlockSparse "solvers::Solver::iterateBlock"[SolverT](
            SolverT& solver,
            const size_t blockSize,
            const size_t nnz,
            const int32_t* const Xindptr,
            const int32_t* const Xindices,
            const Double* const Xvalues,
            const Double* const yData,
            const int64_t* const idxData) nogil

    void iterateBlockIndexedSparse "solvers::Solver::iterateBlockIndexed"[SolverT](
            SolverT& solver,
            const size_t dataSize,
            const size_t nnz,
            const int32_t* const Xindptr,
            const int32_t* const Xindices,
            const Double* const Xvalues,
            const Double* const yData,
            const size_t blockSize,
            const int64_t* const idxData) nogil

    void initFromX "solvers::Solver::initFromX"[SolverT](
            SolverT& solver,
            const size_t dataSize,
            const size_t nnz,
            const int32_t* const Xindptr,
            const int32_t* const Xindices,
            const Double* const Xvalues)

    void initQ "solvers::Solver::initQ"[SolverT](
            SolverT& solver,
            const size_t dataSize,
            const size_t nnz,
            const int32_t* const Xindptr,
            const int32_t* const Xindices,
            const Double* const Xvalues)

    cdef cppclass OneVsRest[SolverT]:
        OneVsRest(size_t nclasses, ...)
        size_t nclasses()
        void startDecay()
        void decay(Double mult)
        void iterateBlock(...)
        void iterateBlockIndexed(...)
        void predict(const size_t sz,
                     int32_t* const out,
                     const Double* const XData)
        Double computeLoss(const size_t sz,
                           const Double* const XData,
                           const int32_t* const yData) nogil
        Double computeSquaredNorm() nogil
        Double computeProxPenalty()

cdef extern from "solvers/Loss.h" namespace "solvers":
    void setGradSigma "solvers::Loss::setGradSigma"(
            const Double gradSigma)
    Double gradSigma "solvers::Loss::gradSigma"()

cdef extern from "solvers/SGD.h" namespace "solvers":
    cdef cppclass _SGD "solvers::SGD":
        _SGD(size_t dim, Double lr, Double lmbda, string loss,
             string prox, Double proxWeight, bool average)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        Double* wdata()
        Double computeLoss(const size_t sz,
                           const Double* const XData,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil
        Double computeProxPenalty() nogil

    cdef cppclass _SparseSGD "solvers::SparseSGD":
        _SparseSGD(size_t dim, Double lr, Double lmbda, string loss)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        Double* wdata()
        Double computeLoss(const size_t sz,
                           const size_t nnz,
                           const int32_t* const Xindptr,
                           const int32_t* const Xindices,
                           const Double* const Xvalues,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil


def set_grad_sigma(Double gradSigma):
    setGradSigma(gradSigma)

def grad_sigma():
    return gradSigma()


cdef class SGD:
    cdef _SGD* solver

    def __cinit__(self, size_t dim, Double lr=0.1,
                  Double lmbda=0., string loss="logistic",
                  string prox="none", Double prox_weight=0., bool average=False):
        self.solver = new _SGD(dim, lr, lmbda, loss, prox, prox_weight, average)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def compute_loss(self, Double[:,::1] X not None, Double[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()

    def iterate(self,
                Double[:,::1] X not None,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        iterateBlock[_SGD](deref(self.solver),
                           X.shape[0],
                           &X[0,0],
                           &y[0],
                           &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        iterateBlockIndexed[_SGD](deref(self.solver),
                                  X.shape[0],
                                  &X[0,0],
                                  &y[0],
                                  idx.shape[0],
                                  &idx[0])


cdef class SparseSGD:
    cdef _SparseSGD* solver

    def __cinit__(self, size_t dim, Double lr=0.1,
                  Double lmbda=0., string loss="logistic"):
        self.solver = new _SparseSGD(dim, lr, lmbda, loss)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def set_q(self, Double[::1] q not None):
        setQ[_SparseSGD](deref(self.solver), q.shape[0], &q[0])

    def compute_loss(self, X, Double[::1] y not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        cdef Double loss
        with nogil:
            loss = self.solver.computeLoss(n, nnz, &indptr[0],
                                           &indices[0], &values[0], &y[0])
        return loss

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def iterate(self, X,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        with nogil:
            iterateBlockSparse[_SparseSGD](deref(self.solver),
                                       n, nnz,
                                       &indptr[0],
                                       &indices[0],
                                       &values[0],
                                       &y[0],
                                       &idx[0])

    def iterate_indexed(self, X,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indices = X.indices
        cdef int32_t[:] indptr = X.indptr
        cdef size_t n = X.shape[0], nnz = X.nnz, blockSize = idx.shape[0]
        with nogil:
            iterateBlockIndexedSparse[_SparseSGD](deref(self.solver),
                                              n, nnz,
                                              &indptr[0],
                                              &indices[0],
                                              &values[0],
                                              &y[0],
                                              blockSize,
                                              &idx[0])


cdef class SGDOneVsRest:
    cdef OneVsRest[_SGD]* solver

    def __cinit__(self, size_t nclasses, size_t dim, Double lr=0.1,
                  Double lmbda=0., string loss="logistic",
                  string prox="none", Double prox_weight=0., bool average=False):
        self.solver = new OneVsRest[_SGD](nclasses, dim, lr, lmbda, loss, prox, prox_weight, average)

    def __dealloc__(self):
        del self.solver

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def iterate(self,
                Double[:,::1] X not None,
                int32_t[::1] y not None,
                int64_t[::1] idx not None):
        self.solver.iterateBlock(X.shape[0], &X[0,0], &y[0], &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        int32_t[::1] y not None,
                        int64_t[::1] idx not None):
        self.solver.iterateBlockIndexed(
                X.shape[0], &X[0,0], &y[0], idx.shape[0], &idx[0])

    def predict(self, Double[:,::1] X not None):
        preds = np.empty(X.shape[0], dtype=np.int32)
        cdef int32_t[:] out = preds
        self.solver.predict(out.shape[0], &out[0], &X[0,0])
        return preds

    def compute_loss(self, Double[:,::1] X not None, int32_t[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()


cdef extern from "solvers/MISO.h" namespace "solvers":
    cdef cppclass _MISO "solvers::MISO":
        _MISO(size_t dim, size_t n, Double alpha, Double lmbda, string loss,
              bool computeLB, string prox, Double proxWeight, bool average)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()
        Double lowerBound()
        Double computeLoss(const size_t sz,
                           const Double* const XData,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil
        Double computeProxPenalty() nogil

    cdef cppclass _SparseMISO "solvers::SparseMISO":
        _SparseMISO(size_t dim, size_t n, Double alpha, Double lmbda, string loss, bool computeLB)
        void startDecay()
        void decay(Double mult)
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()
        Double lowerBound()
        Double computeLoss(const size_t sz,
                           const size_t nnz,
                           const int32_t* const Xindptr,
                           const int32_t* const Xindices,
                           const Double* const Xvalues,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil

cdef class MISO:
    cdef _MISO* solver

    def __cinit__(self, size_t dim, size_t n, Double alpha=1.0,
                  Double lmbda=0.1, string loss="logistic", bool compute_lb=False,
                  string prox="none", Double prox_weight=0., bool average=False):
        self.solver = new _MISO(dim, n, alpha, lmbda, loss, compute_lb, prox, prox_weight, average)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property nexamples:
        def __get__(self):
            return self.solver.nexamples()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def lower_bound(self):
        return self.solver.lowerBound()

    def compute_loss(self, Double[:,::1] X not None, Double[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()

    def iterate(self,
                Double[:,::1] X not None,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        iterateBlock[_MISO](deref(self.solver),
                            X.shape[0],
                            &X[0,0],
                            &y[0],
                            &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        iterateBlockIndexed[_MISO](deref(self.solver),
                                   X.shape[0],
                                   &X[0,0],
                                   &y[0],
                                   idx.shape[0],
                                   &idx[0])


cdef class SparseMISO:
    cdef _SparseMISO* solver

    def __cinit__(self, size_t dim, size_t n, Double alpha=1.0,
                  Double lmbda=0.1, string loss="logistic", bool compute_lb=False):
        self.solver = new _SparseMISO(dim, n, alpha, lmbda, loss, compute_lb)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property nexamples:
        def __get__(self):
            return self.solver.nexamples()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def lower_bound(self):
        return self.solver.lowerBound()

    def init(self, X):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        initFromX[_SparseMISO](deref(self.solver), X.shape[0], X.nnz, &indptr[0],
                               &indices[0], &values[0])

    def set_q(self, Double[::1] q not None):
        setQ[_SparseMISO](deref(self.solver), q.shape[0], &q[0])

    def compute_loss(self, X, Double[::1] y not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        cdef Double loss
        with nogil:
            loss = self.solver.computeLoss(n, nnz, &indptr[0],
                                           &indices[0], &values[0], &y[0])
        return loss

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def iterate(self, X,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        assert X.has_sorted_indices, "The Sparse MISO implementation requires sorted indices." \
                "Use X.sort_indices() to sort them."
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        with nogil:
            iterateBlockSparse[_SparseMISO](deref(self.solver),
                                        n,
                                        nnz,
                                        &indptr[0],
                                        &indices[0],
                                        &values[0],
                                        &y[0],
                                        &idx[0])

    def iterate_indexed(self, X,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indices = X.indices
        cdef int32_t[:] indptr = X.indptr
        cdef size_t n = X.shape[0], nnz = X.nnz, blockSize = idx.shape[0]
        with nogil:
            iterateBlockIndexedSparse[_SparseMISO](deref(self.solver),
                                               n, nnz,
                                               &indptr[0],
                                               &indices[0],
                                               &values[0],
                                               &y[0],
                                               blockSize,
                                               &idx[0])


cdef class MISOOneVsRest:
    cdef OneVsRest[_MISO]* solver

    def __cinit__(self, size_t nclasses, size_t dim, size_t n, Double alpha=1.0,
                  Double lmbda=0.1, string loss="logistic", bool compute_lb=False,
                  string prox="none", Double prox_weight=0., bool average=False):
        self.solver = new OneVsRest[_MISO](nclasses, dim, n, alpha, lmbda, loss,
                                           compute_lb, prox, prox_weight, average)

    def __dealloc__(self):
        del self.solver

    def start_decay(self):
        self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        self.solver.decay(multiplier)

    def iterate(self,
                Double[:,::1] X not None,
                int32_t[::1] y not None,
                int64_t[::1] idx not None):
        self.solver.iterateBlock(X.shape[0], &X[0,0], &y[0], &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        int32_t[::1] y not None,
                        int64_t[::1] idx not None):
        self.solver.iterateBlockIndexed(
                X.shape[0], &X[0,0], &y[0], idx.shape[0], &idx[0])

    def predict(self, Double[:,::1] X not None):
        preds = np.empty(X.shape[0], dtype=np.int32)
        cdef int32_t[:] out = preds
        self.solver.predict(out.shape[0], &out[0], &X[0,0])
        return preds

    def compute_loss(self, Double[:,::1] X not None, int32_t[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()


cdef extern from "solvers/SAGA.h" namespace "solvers":
    cdef cppclass _SAGA "solvers::SAGA":
        _SAGA(size_t dim, size_t n, Double lr, Double lmbda,
              string loss, string prox, Double proxWeight)
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()
        Double computeLoss(const size_t sz,
                           const Double* const XData,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil
        Double computeProxPenalty()

    cdef cppclass _SparseSAGA "solvers::SparseSAGA":
        _SparseSAGA(size_t dim, size_t n, Double lr, Double lmbda, string loss)
        size_t t()
        size_t nfeatures()
        size_t nexamples()
        Double* wdata()
        Double computeLoss(const size_t sz,
                           const size_t nnz,
                           const int32_t* const Xindptr,
                           const int32_t* const Xindices,
                           const Double* const Xvalues,
                           const Double* const yData) nogil
        Double computeSquaredNorm() nogil


cdef class SAGA:
    cdef _SAGA* solver

    def __cinit__(self, size_t dim, size_t n, Double lr=1.0,
                  Double lmbda=0.1, string loss="logistic",
                  string prox="none", Double prox_weight=0.):
        self.solver = new _SAGA(dim, n, lr, lmbda, loss, prox, prox_weight)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property nexamples:
        def __get__(self):
            return self.solver.nexamples()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def start_decay(self):
        pass  # self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        pass  # self.solver.decay(multiplier)

    def compute_loss(self, Double[:,::1] X not None, Double[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()

    def iterate(self,
                Double[:,::1] X not None,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        iterateBlock[_SAGA](deref(self.solver),
                            X.shape[0],
                            &X[0,0],
                            &y[0],
                            &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        iterateBlockIndexed[_SAGA](deref(self.solver),
                                   X.shape[0],
                                   &X[0,0],
                                   &y[0],
                                   idx.shape[0],
                                   &idx[0])


cdef class SparseSAGA:
    cdef _SparseSAGA* solver

    def __cinit__(self, size_t dim, size_t n, Double lr=1.0,
                  Double lmbda=0.1, string loss="logistic"):
        self.solver = new _SparseSAGA(dim, n, lr, lmbda, loss)

    def __dealloc__(self):
        del self.solver

    property nfeatures:
        def __get__(self):
            return self.solver.nfeatures()

    property nexamples:
        def __get__(self):
            return self.solver.nexamples()

    property w:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = self.nfeatures
            cdef np.ndarray[Double, ndim=1] arr = \
                np.PyArray_SimpleNewFromData(1, shape, npDOUBLE,
                                             self.solver.wdata())
            Py_INCREF(self)
            PyArray_SetBaseObject(arr, <PyObject*>self)
            return arr

    def init(self, X):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        return initFromX[_SparseSAGA](deref(self.solver), X.shape[0], X.nnz, &indptr[0],
                                      &indices[0], &values[0])

    def start_decay(self):
        pass  # self.solver.startDecay()

    def decay(self, Double multiplier=0.5):
        pass  # self.solver.decay(multiplier)

    def compute_loss(self, X, Double[::1] y not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        cdef Double loss
        with nogil:
            loss = self.solver.computeLoss(n, nnz, &indptr[0],
                                           &indices[0], &values[0], &y[0])
        return loss

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def iterate(self, X,
                Double[::1] y not None,
                int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        assert X.has_sorted_indices, "The Sparse SAGA implementation requires sorted indices." \
                "Use X.sort_indices() to sort them."
        cdef Double[:] values = X.data
        cdef int32_t[:] indptr = X.indptr
        cdef int32_t[:] indices = X.indices
        cdef size_t n = X.shape[0], nnz = X.nnz
        with nogil:
            iterateBlockSparse[_SparseSAGA](deref(self.solver),
                                        n, nnz,
                                        &indptr[0],
                                        &indices[0],
                                        &values[0],
                                        &y[0],
                                        &idx[0])

    def iterate_indexed(self, X,
                        Double[::1] y not None,
                        int64_t[::1] idx not None):
        assert isinstance(X, sp.csr_matrix)
        cdef Double[:] values = X.data
        cdef int32_t[:] indices = X.indices
        cdef int32_t[:] indptr = X.indptr
        cdef size_t n = X.shape[0], nnz = X.nnz, blockSize = idx.shape[0]
        with nogil:
            iterateBlockIndexedSparse[_SparseSAGA](deref(self.solver),
                                               n, nnz,
                                               &indptr[0],
                                               &indices[0],
                                               &values[0],
                                               &y[0],
                                               blockSize,
                                               &idx[0])
cdef class SAGAOneVsRest:
    cdef OneVsRest[_SAGA]* solver

    def __cinit__(self, size_t nclasses, size_t dim, size_t n, Double lr=1.0,
                  Double lmbda=0.1, string loss="logistic",
                  string prox="none", Double prox_weight=0.):
        self.solver = new OneVsRest[_SAGA](nclasses, dim, n, lr, lmbda, loss, prox, prox_weight)

    def __dealloc__(self):
        del self.solver

    def iterate(self,
                Double[:,::1] X not None,
                int32_t[::1] y not None,
                int64_t[::1] idx not None):
        self.solver.iterateBlock(X.shape[0], &X[0,0], &y[0], &idx[0])

    def iterate_indexed(self,
                        Double[:,::1] X not None,
                        int32_t[::1] y not None,
                        int64_t[::1] idx not None):
        self.solver.iterateBlockIndexed(
                X.shape[0], &X[0,0], &y[0], idx.shape[0], &idx[0])

    def predict(self, Double[:,::1] X not None):
        preds = np.empty(X.shape[0], dtype=np.int32)
        cdef int32_t[:] out = preds
        self.solver.predict(out.shape[0], &out[0], &X[0,0])
        return preds

    def compute_loss(self, Double[:,::1] X not None, int32_t[::1] y not None):
        return self.solver.computeLoss(X.shape[0], &X[0,0], &y[0])

    def compute_squared_norm(self):
        cdef Double norm
        with nogil:
            norm = self.solver.computeSquaredNorm()
        return norm

    def compute_prox_penalty(self):
        return self.solver.computeProxPenalty()
