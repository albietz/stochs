
#include "MISO.h"

namespace solvers {

MISOBase::MISOBase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double alpha,
                   const Double lambda,
                   const std::string& loss,
                   const bool computeLB,
                   const std::string& prox,
                   const Double proxWeight,
                   const bool average)
  : Solver(nfeatures, loss, prox, proxWeight),
    n_(nexamples),
    alpha_(alpha),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1),
    q_(0),
    computeLB_(computeLB),
    average_(average) {
  if (computeLB_) {
    c_ = Vector::Zero(n_);
  }
  if (average_) {
    wavg_ = Vector::Zero(nfeatures);
  }
}

void MISOBase::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2 * static_cast<Double>(n_) / alpha_) + 1;

  if (average_) {
    wavg_ = w_;
  }
}

void MISOBase::decay(const Double multiplier) {
  alpha_ *= multiplier;
}

MISO::MISO(const size_t nfeatures,
           const size_t nexamples,
           const Double alpha,
           const Double lambda,
           const std::string& loss,
           const bool computeLB,
           const std::string& prox,
           const Double proxWeight,
           const bool average)
  : MISOBase(nfeatures,
             nexamples,
             alpha,
             lambda,
             loss,
             computeLB,
             prox,
             proxWeight,
             average),
    z_(Matrix::Zero(nexamples, nfeatures)),
    grad_(nfeatures),
    zi_(nfeatures),
    zbar_(Vector::Zero(nfeatures)) {
}

Double MISO::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    LOG(ERROR) << "computeLB is false!";
    return 0;
  }
}

SparseMISONaive::SparseMISONaive(const size_t nfeatures,
                                 const size_t nexamples,
                                 const Double alpha,
                                 const Double lambda,
                                 const std::string& loss,
                                 const bool computeLB)
  : MISOBase(nfeatures, nexamples, alpha, lambda, loss, computeLB),
    z_(nexamples, nfeatures),
    grad_(nfeatures),
    zi_(nfeatures) {
}

Double SparseMISONaive::lowerBound() const {
  if (computeLB_) {
    return (c_ - lambda_ * z_ * w_).mean() + 0.5 * lambda_ * w_.squaredNorm();
  } else {
    LOG(ERROR) << "computeLB is false!";
    return 0;
  }
}

SparseMISO::SparseMISO(const size_t nfeatures,
                       const size_t nexamples,
                       const Double alpha,
                       const Double lambda,
                       const std::string& loss,
                       const bool computeLB)
  : MISOBase(nfeatures, nexamples, alpha, lambda, loss, computeLB),
    z_(nexamples, nfeatures) {
}

Double SparseMISO::lowerBound() const {
  LOG(ERROR) << "lowerBound() not implemented";
  return 0;
}
}
