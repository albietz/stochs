
#include "Loss.h"
#include "Prox.h"
#include "Util.h"

namespace solvers {

template <typename Derived>
void SAGA::iterate(const Eigen::MatrixBase<Derived>& x, // x is a row vector
                   const Double y,
                   const size_t idx) {
  const Double pred = x * w_;

  Loss::computeGradient<Vector, Derived>(grad_, loss_, x, pred, y);
  grad_ += lambda_ * w_;

  w_ = w_ - lr_ * (grad_ - g_.row(idx).transpose() + gbar_);

  gbar_ += 1.0 / n_ * (grad_ - g_.row(idx).transpose());

  g_.row(idx) = grad_.transpose();

  Prox::applyProx(w_, prox_, lr_ * proxWeight_);

  ++t_;
}

template <typename Derived>
void SparseSAGA::initFromX(const Eigen::SparseMatrixBase<Derived>& X) {
  g_ = 0.0 * X;
}

inline void SparseSAGA::updateCoord(const size_t coord, const size_t t) {
  const size_t dt = t - lastUpdate_[coord];
  lastUpdate_[coord] = t;
  ws_(coord) -= updateFactor_[dt] * lr_ * gbar_(coord) / s_;
}

inline void SparseSAGA::updateAllCoords(const size_t t) {
  for (size_t i = 0; i < nfeatures(); ++i) {
    updateCoord(i, t);
  }
}

template <typename Derived>
void SparseSAGA::updateCoords(const Eigen::SparseMatrixBase<Derived>& x,
                              const size_t t) {
  const auto& Xblock = x.derived();
  const auto& X = Xblock.nestedExpression();
  Util::iterateRowIndices(
      [this, t](const size_t coord) { updateCoord(coord, t); }, X,
      Xblock.startRow());
}

template <typename Derived>
void SparseSAGA::iterate(const Eigen::SparseMatrixBase<Derived>& x,
                         const Double y,
                         const size_t idx) {
  if (t_ % maxDt_ == 0) {
    // update everything when lags could exceed the max
    updateAllCoords(t_ - 1);
  } else {
    // just-in-time update of ws_
    updateCoords(x.derived(), t_ - 1);
  }

  const Double pred = s_ * static_cast<Double>(x * ws_);

  gradDiff_ = -g_.row(idx); // gradient will be added below

  // [hacky] retrieve original sparse matrix from the view x
  // assumes that x was passed as a block of a sparse matrix
  const auto& Xblock = x.derived();
  const auto& X = Xblock.nestedExpression();
  // get dense vector maps of non-zero values
  RowVectorMap xMap = Util::denseRowMap<RowVector>(X, Xblock.startRow());
  auto gMap = Util::denseRowMap<Vector>(g_, idx);
  auto gDiffMap = Util::denseMap<Vector>(gradDiff_);

  if (xMap.size() != gMap.size() || gDiffMap.size() != gMap.size()) {
    LOG_EVERY_N(ERROR, 1000) << "size mismatch in sparse value arrays! ("
                             << xMap.size() << ", " << gMap.size() << ", "
                             << gDiffMap.size() << ") Did you call init?";
    return;
  }

  grad_.resize(xMap.size());
  Loss::computeGradient<Vector, RowVectorMap>(grad_, loss_, xMap, pred, y);
  gDiffMap += grad_; // delta = grad - grad_old

  s_ *= (1.0 - lr_ * lambda_);
  ws_ -= (lr_ / s_) * gradDiff_; // sparse update
  updateCoords(x, t_); // lazy dense update
  gbar_ += (1.0 / n_) * gradDiff_;

  gMap = grad_;
  ++t_;

  // for numerical stability
  if (s_ < 1e-9) {
    LOG(INFO) << "resetting ws and s";
    ws_ = s_ * ws_;
    s_ = 1.0;
  }
}
}
