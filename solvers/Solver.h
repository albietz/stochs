
#pragma once

#include <iostream>

#include "common.h"
#include "Loss.h"
#include "Prox.h"

namespace solvers {

class Solver {
 public:
  Solver(const size_t nfeatures,
         const std::string& loss,
         const std::string& prox = "none",
         const Double proxWeight = 0)
    : nfeatures_(nfeatures),
      w_(Vector::Zero(nfeatures_)),
      loss_(loss),
      prox_(prox),
      proxWeight_(proxWeight) {
  }

  virtual ~Solver() {
  }

  // delete copy constructors
  Solver(const Solver&) = delete;
  Solver& operator=(const Solver&) = delete;
  Solver(Solver&&) = default;
  Solver& operator=(Solver&&) = default;

  size_t nfeatures() const {
    return nfeatures_;
  }

  const std::string& loss() const {
    return loss_;
  }

  virtual Vector& w() {
    return w_;
  }

  virtual const Vector& w() const {
    return w_;
  }

  Double* wdata() {
    return w().data();
  }

  Double computeSquaredNorm() const {
    return w().squaredNorm();
  }

  Double computeProxPenalty() const {
    return Prox::computePenalty(w(), prox_);
  }

  // for dense data
  void predict(const size_t dataSize,
               Double* const outPreds,
               const Double* const XData) const {
    const MatrixMap Xmap(XData, dataSize, nfeatures_);
    Eigen::Map<Vector> preds(outPreds, dataSize);
    preds = Xmap * w();
  }

  Double computeLoss(const size_t dataSize,
                     const Double* const XData,
                     const Double* const yData) const {
    Vector preds(dataSize);
    predict(dataSize, preds.data(), XData);
    return computeLossImpl(preds, yData);
  }

  template <typename SolverT>
  static void iterateBlock(SolverT& solver,
                           const size_t blockSize,
                           const Double* const XData,
                           const Double* const yData,
                           const int64_t* const idxData) {
    const MatrixMap Xmap(XData, blockSize, solver.nfeatures());
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(i), yData[i], idxData[i]);
    }
  }

  template <typename SolverT>
  static void iterateBlockIndexed(SolverT& solver,
                                  const size_t dataSize,
                                  const Double* const XData,
                                  const Double* const yData,
                                  const size_t blockSize,
                                  const int64_t* const idxData) {
    const MatrixMap Xmap(XData, dataSize, solver.nfeatures());
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(idxData[i]), yData[idxData[i]], idxData[i]);
    }
  }

  template <typename SolverT>
  static void setQ(SolverT& solver, const size_t n, const Double* const qData) {
    const VectorMap qMap(qData, n);
    solver.setQ(qMap);
  }

  // for sparse data
  void predict(const size_t dataSize,
               Double* const outPreds,
               const size_t nnz,
               const int32_t* const Xindptr,
               const int32_t* const Xindices,
               const Double* const Xvalues) const {
    const SpMatrixMap Xmap(dataSize, nfeatures(), nnz,
                           Xindptr, Xindices, Xvalues);
    Eigen::Map<Vector> preds(outPreds, dataSize);
    preds = Xmap * w();
  }

  Double computeLoss(const size_t dataSize,
                     const size_t nnz,
                     const int32_t* const Xindptr,
                     const int32_t* const Xindices,
                     const Double* const Xvalues,
                     const Double* const yData) const {
    Vector preds(dataSize);
    predict(dataSize, preds.data(), nnz, Xindptr, Xindices, Xvalues);
    return computeLossImpl(preds, yData);
  }

  template <typename SolverT>
  static void iterateBlock(SolverT& solver,
                           const size_t blockSize, // rows
                           const size_t nnz,
                           const int32_t* const Xindptr,
                           const int32_t* const Xindices,
                           const Double* const Xvalues,
                           const Double* const yData,
                           const int64_t* const idxData) {
    const SpMatrixMap Xmap(blockSize, solver.nfeatures(), nnz,
                           Xindptr, Xindices, Xvalues);
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(i), yData[i], idxData[i]);
    }
  }

  template <typename SolverT>
  static void iterateBlockIndexed(SolverT& solver,
                                  const size_t dataSize,
                                  const size_t nnz,
                                  const int32_t* const Xindptr,
                                  const int32_t* const Xindices,
                                  const Double* const Xvalues,
                                  const Double* const yData,
                                  const size_t blockSize,
                                  const int64_t* const idxData) {
    const SpMatrixMap Xmap(dataSize, solver.nfeatures(), nnz,
                           Xindptr, Xindices, Xvalues);
    for (size_t i = 0; i < blockSize; ++i) {
      solver.iterate(Xmap.row(idxData[i]), yData[idxData[i]], idxData[i]);
    }
  }

  template <typename SolverT>
  static void initFromX(SolverT& solver,
                        const size_t dataSize,
                        const size_t nnz,
                        const int32_t* const Xindptr,
                        const int32_t* const Xindices,
                        const Double* const Xvalues) {
    const SpMatrixMap Xmap(
        dataSize, solver.nfeatures(), nnz, Xindptr, Xindices, Xvalues);
    solver.initFromX(Xmap);
  }

  template <typename SolverT>
  static void initQ(SolverT& solver,
                    const size_t dataSize,
                    const size_t nnz,
                    const int32_t* const Xindptr,
                    const int32_t* const Xindices,
                    const Double* const Xvalues) {
    const SpMatrixMap Xmap(
        dataSize, solver.nfeatures(), nnz, Xindptr, Xindices, Xvalues);
    solver.initQ(Xmap);
  }

 private:
  Double computeLossImpl(const Vector& preds,
                         const Double* const yData) const {
    const size_t dataSize = preds.size();
    Double loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < dataSize; ++i) {
      loss += Loss::computeLoss(loss_, preds(i), yData[i]);
    }

    return loss / preds.size();
  }

  const size_t nfeatures_;

 protected:
  mutable Vector w_;

  const std::string loss_;

  const std::string prox_;

  const Double proxWeight_;
};

template <typename SolverT>
class OneVsRest {
 public:
  template <typename... Args>
  OneVsRest(const size_t nclasses, Args&&... args) : nclasses_(nclasses) {
    solvers_.reserve(nclasses_);
    for (size_t i = 0; i < nclasses_; ++i) {
      solvers_.emplace_back(std::forward<Args>(args)...);
    }
  }

  size_t nclasses() const {
    return nclasses_;
  }

  void startDecay() {
    for (auto& solver : solvers_) {
      solver.startDecay();
    }
  }

  void decay(const Double multiplier = 0.5) {
    for (auto& solver : solvers_) {
      solver.decay(multiplier);
    }
  }

  void iterateBlock(const size_t blockSize,
                    const Double* const XData,
                    const int32_t* const yData,
                    const int64_t* const idxData) {
    const MatrixMap Xmap(XData, blockSize, solvers_.front().nfeatures());
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      for (size_t i = 0; i < blockSize; ++i) {
        solvers_[c].iterate(
            Xmap.row(i),
            static_cast<Double>(yData[i] == static_cast<int32_t>(c)),
            idxData[i]);
      }
    }
  }

  void iterateBlockIndexed(const size_t dataSize,
                           const Double* const XData,
                           const int32_t* const yData,
                           const size_t blockSize,
                           const int64_t* const idxData) {
    const MatrixMap Xmap(XData, dataSize, solvers_.front().nfeatures());
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      for (size_t i = 0; i < blockSize; ++i) {
        solvers_[c].iterate(
            Xmap.row(idxData[i]),
            static_cast<Double>(yData[idxData[i]] == static_cast<int32_t>(c)),
            idxData[i]);
      }
    }
  }

  template <typename... Args>
  void predict(const size_t dataSize,
               int32_t* const out,
               Args... Xargs) const {
    Matrix preds(nclasses_, dataSize);
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      solvers_[c].predict(dataSize, preds.row(c).data(), Xargs...);
    }

#pragma omp parallel for
    for (size_t i = 0; i < dataSize; ++i) {
      out[i] = 0;
      Double m = preds(0, i);
      for (size_t c = 1; c < nclasses_; ++c) {
        if (preds(c, i) > m) {
          m = preds(c, i);
          out[i] = c;
        }
      }
    }
  }

  Double computeLoss(const size_t dataSize,
                     const Double* const XData,
                     const int32_t* const yData) const {
    Matrix preds(nclasses_, dataSize);
#pragma omp parallel for
    for (size_t c = 0; c < nclasses_; ++c) {
      solvers_[c].predict(dataSize, preds.row(c).data(), XData);
    }
    return computeLossImpl(preds, yData);
  }

  Double computeSquaredNorm() const {
    Double res = 0;
#pragma omp parallel for reduction(+:res)
    for (size_t c = 0; c < nclasses_; ++c) {
      res += solvers_[c].w().squaredNorm();
    }

    return res;
  }

  Double computeProxPenalty() const {
    Double res = 0;
#pragma omp parallel for reduction(+:res)
    for (size_t c = 0; c < nclasses_; ++c) {
      res += solvers_[c].computeProxPenalty();
    }

    return res;
  }

 private:
  Double computeLossImpl(const Matrix& preds,
                         const int32_t* const yData) const {
    const size_t dataSize = preds.cols();
    Double loss = 0;
#pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < dataSize; ++i) {
      Double l = 0;
      for (size_t c = 0; c < nclasses_; ++c) {
        l += Loss::computeLoss(
            solvers_[0].loss(), preds(c, i),
            static_cast<Double>(yData[i] == static_cast<int32_t>(c)));
      }
      loss += l;
    }

    return loss / dataSize;
  }

  const size_t nclasses_;

  std::vector<SolverT> solvers_;
};
}
