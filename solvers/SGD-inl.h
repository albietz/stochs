
#include "Loss.h"
#include "Prox.h"

namespace solvers {

template <typename Derived>
void SGD::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
                  const Double y,
                  const size_t idx) {
  const Double stepSize = getStepSize();
  const Double weight = getWeight(idx);

  Loss::computeGradient<Vector, Derived>(grad_, loss_, x, x * w_, y);

  // SGD update
  w_ = w_ - weight * stepSize * (grad_ + lambda_ * w_);

  Prox::applyProx(w_, prox_, stepSize * proxWeight_);

  if (average_ && decay_) {
    const Double avgWeight =
        2.0 * (gamma_ + t_ - t0_) / ((t_ - t0_ + 1) * (2 * gamma_ + t_ - t0_));
    wavg_ = (1 - avgWeight) * wavg_ + avgWeight * w_;
  }
  ++t_;
}

template <typename Derived>
void SparseSGD::iterate(
    const Eigen::SparseMatrixBase<Derived>& x, // x is a row vector
    const Double y,
    const size_t idx) {
  const Double stepSize = getStepSize();
  const Double weight = getWeight(idx);

  Loss::computeGradient<SpVector, Derived>(
      grad_, loss_, x, s_ * static_cast<Double>(x * ws_), y);

  s_ *= (1 - weight * stepSize * lambda_);
  ws_ -= (weight * stepSize / s_) * grad_;

  ++t_;

  // for numerical stability
  if (s_ < 1e-9) {
    LOG(INFO) << "resetting ws and s";
    ws_ = s_ * ws_;
    s_ = 1.0;
  }
}
}
