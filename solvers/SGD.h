
#pragma once

#include <string>

#include "common.h"
#include "Solver.h"

namespace solvers {

class SGDBase : public Solver {
 public:
  SGDBase(const size_t nfeatures,
          const Double lr,
          const Double lambda,
          const std::string& loss,
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

  void decay(const double multiplier = 0.5) {
    lr_ *= multiplier;
  }

  size_t t() const {
    return t_;
  }

 protected:
  Double getStepSize() const {
    return decay_ ? 2.0 / (lambda_ * (t_ - t0_ + gamma_)) : lr_;
  }

  Double getWeight(const size_t idx) const {
    return q_.size() == 0 ? 1.0 : 1.0 / (q_(idx) * q_.size());
  }

  Double lr_;

  const Double lambda_;

  bool decay_;

  size_t t_; // iteration

  size_t t0_;

  size_t gamma_;

  Vector q_; // weights for non-uniform sampling

  Vector wavg_;

  bool average_;
};

class SGD : public SGDBase {
 public:
  SGD(const size_t nfeatures,
      const Double lr,
      const Double lambda,
      const std::string& loss,
      const std::string& prox,
      const Double proxWeight,
      const bool average);

  template <typename Derived>
  void iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

 private:
  Vector grad_;
};

class SparseSGD : public SGDBase {
 public:
  SparseSGD(const size_t nfeatures,
            const Double lr,
            const Double lambda,
            const std::string& loss);

  Vector& w() {
    updateW();
    return SGDBase::w();
  }

  const Vector& w() const {
    updateW();
    return SGDBase::w();
  }

  template <typename Derived>
  void iterate(const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
               const Double y,
               const size_t idx);

 private:
  void updateW() const {
    w_ = s_ * ws_;
  }

  Vector ws_; // w_ / s_

  Double s_; // prod_t (1 - gamma_t lambda)

  SpVector grad_;
};
}

#include "SGD-inl.h"
