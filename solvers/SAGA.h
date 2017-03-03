
#pragma once

#include <string>
#include <vector>

#include "common.h"
#include "Solver.h"

namespace solvers {

class SAGABase : public Solver {
 public:
  SAGABase(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss,
           const std::string& prox = "none",
           const Double proxWeight = 0);

  size_t nexamples() const {
    return n_;
  }

  size_t t() const {
    return t_;
  }

 protected:
  const size_t n_; // number of examples/clusters in the dataset

  Double lr_; // step size

  const Double lambda_;

  size_t t_; // iteration
};

class SAGA : public SAGABase {
 public:
  SAGA(const size_t nfeatures,
       const size_t nexamples,
       const Double lr,
       const Double lambda,
       const std::string& loss,
       const std::string& prox,
       const Double proxWeight);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

 private:
  Matrix g_; // matrix of last gradients

  Vector grad_;

  Vector gbar_; // average gradient
};

class SparseSAGA : public SAGABase {
 public:
  SparseSAGA(const size_t nfeatures,
             const size_t nexamples,
             const Double lr,
             const Double lambda,
             const std::string& loss);

  template <typename Derived>
  void initFromX(const Eigen::SparseMatrixBase<Derived>& X);

  Vector& w() {
    updateW();
    return w_;
  }

  const Vector& w() const {
    updateW();
    return w_;
  }

  template <typename Derived>
  void iterate(const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

 private:
  void updateW() const {
    // NB: normally the solver isn't declared as const so this should be ok
    const_cast<SparseSAGA*>(this)->updateAllCoords(t_ - 1);
    w_ = s_ * ws_;
  }

  void updateCoord(const size_t coord, const size_t t);

  template <typename Derived>
  void updateCoords(const Eigen::SparseMatrixBase<Derived>& x, const size_t t);

  void updateAllCoords(const size_t t);

  SpMatrix g_; // gradient table

  Vector grad_; // current gradient in dense form

  SpVector gradDiff_; // gradient difference

  Vector gbar_; // average of stored gradients

  Vector ws_; // w_ / s_

  Double s_; // prod_t (1 - gamma lambda)

  // last iteration at which a coord was updated
  std::vector<size_t> lastUpdate_;

  // geometric series factors, used for lazy updates
  std::vector<Double> updateFactor_;

  static size_t maxDt_; // maximum lag supported
};
}

#include "SAGA-inl.h"
