
#pragma once

#include <string>

#include "common.h"
#include "Solver.h"

namespace solvers {

class MISOBase : public Solver {
 public:
  MISOBase(const size_t nfeatures,
           const size_t nexamples,
           const Double alpha,
           const Double lambda,
           const std::string& loss,
           const bool computeLB,
           const std::string& prox = "none",
           const Double proxWeight = 0,
           const bool average = false);

  template <typename Derived>
  void setQ(const Eigen::MatrixBase<Derived>& q) {
    q_ = q;
  }

  Vector& w() {
    if (average_ && decay_) {
      return wavg_;
    } else {
      return w_;
    }
  }

  const Vector& w() const {
    if (average_ && decay_) {
      return wavg_;
    } else {
      return w_;
    }
  }

  void startDecay();

  void decay(const Double multiplier = 0.5);

  size_t nexamples() const {
    return n_;
  }

  size_t t() const {
    return t_;
  }

  virtual Double lowerBound() const = 0;

 protected:
  Double getStepSize() const {
    return decay_ ?
      std::min<Double>(alpha_, 2 * static_cast<Double>(n_) / (t_ - t0_ + gamma_)) : alpha_;
  }

  Double getWeight(const size_t idx) const {
    return q_.size() == 0 ? 1.0 : 1.0 / (q_(idx) * n_);
  }

  const size_t n_; // number of examples/clusters in the dataset

  Double alpha_; // step size

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;

  size_t gamma_; // offset for decaying stepsize C / (gamma + t - t0)

  Vector q_; // weights for non-uniform sampling

  bool computeLB_; // whether to compute lower bounds

  Vector c_; // for computing lower bounds

  bool average_; // whether to do iterate averaging (from when we start decaying)

  Vector wavg_; // averaged iterate
};

class MISO : public MISOBase {
 public:
  MISO(const size_t nfeatures,
       const size_t nexamples,
       const Double alpha,
       const Double lambda,
       const std::string& loss,
       const bool computeLB,
       const std::string& prox,
       const Double proxWeight,
       const bool average);

  template <typename Derived>
  void initQ(const Eigen::MatrixBase<Derived>& X);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

  Double lowerBound() const;

 private:
  Matrix z_;

  Vector grad_;

  Vector zi_;

  Vector zbar_; // 1/n sum_i z_i: store separately for using prox
};

// Naive Eigen-based sparse implementation that doesn't require the same
// sparsity pattern every time the same idx appears (and hence doesn't need
// special initialization for the Z matrix).
class SparseMISONaive : public MISOBase {
 public:
  SparseMISONaive(const size_t nfeatures,
                  const size_t nexamples,
                  const Double alpha,
                  const Double lambda,
                  const std::string& loss,
                  const bool computeLB);

  template <typename Derived>
  void initFromX(const Eigen::SparseMatrixBase<Derived>& X);

  template <typename Derived>
  void iterate(const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

  Double lowerBound() const;

 private:
  SpMatrix z_;

  SpVector grad_;

  SpVector zi_;
};

// optimized sparse implementation which requires that examples have the same
// sparsity pattern for any fixed idx as that given during initialization.
// Initialization with a full data matrix using initFromX is required.
// Indices in each CSR row vector need to be sorted.
class SparseMISO : public MISOBase {
 public:
  SparseMISO(const size_t nfeatures,
             const size_t nexamples,
             const Double alpha,
             const Double lambda,
             const std::string& loss,
             const bool computeLB);

  template <typename Derived>
  void initFromX(const Eigen::SparseMatrixBase<Derived>& X);

  template <typename Derived>
  void initQ(const Eigen::SparseMatrixBase<Derived>& X);

  template <typename Derived>
  void iterate(const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

  Double lowerBound() const;

 private:
  SpMatrix z_;

  Vector grad_;

  SpVector ziOld_;
};
}

#include "MISO-inl.h"
