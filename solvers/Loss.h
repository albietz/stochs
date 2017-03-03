
#pragma once

#include <random>
#include <string>
#include <type_traits>

#include "common.h"

namespace solvers {

namespace detail {

template <typename Derived, typename VecT>
using EBase = typename std::conditional<
  std::is_base_of<Eigen::MatrixBase<VecT>, VecT>::value,
  Eigen::MatrixBase<Derived>,
  Eigen::SparseMatrixBase<Derived>>::type;

inline void makeZero(SpVector& g, const size_t sz) {
  g.setZero();
}

inline void makeZero(Vector& g, const size_t sz) {
  g = Vector::Zero(sz);
}
}

class Loss {
 public:
  static Double computeLoss(const std::string& loss,
                            const Double pred,
                            const Double y);

  static Double computeGradient(const std::string& loss,
                                const Double pred,
                                const Double y);

  template <typename VectorT, typename Derived>
  static void computeGradient(VectorT& g,
                              const std::string& loss,
                              // x is a row vector
                              const detail::EBase<Derived, VectorT>& x,
                              const Double pred,
                              const Double y);

  static void setGradSigma(const Double gradSigma) {
    LOG(INFO) << "setting gradient std dev to " << gradSigma;
    gradSigma_ = gradSigma;
  }

  static Double gradSigma() {
    return gradSigma_;
  }

 private:
  static std::mt19937 gen_;

  static Double gradSigma_;
};
}

#include "Loss-inl.h"
