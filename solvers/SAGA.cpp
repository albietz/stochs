
#include "SAGA.h"

namespace solvers {

SAGABase::SAGABase(const size_t nfeatures,
                   const size_t nexamples,
                   const Double lr,
                   const Double lambda,
                   const std::string& loss,
                   const std::string& prox,
                   const Double proxWeight)
  : Solver(nfeatures, loss, prox, proxWeight),
    n_(nexamples),
    lr_(lr),
    lambda_(lambda),
    t_(1) {
}

SAGA::SAGA(const size_t nfeatures,
           const size_t nexamples,
           const Double lr,
           const Double lambda,
           const std::string& loss,
           const std::string& prox,
           const Double proxWeight)
  : SAGABase(nfeatures, nexamples, lr, lambda, loss, prox, proxWeight),
    g_(Matrix::Zero(nexamples, nfeatures)),
    grad_(nfeatures),
    gbar_(Vector::Zero(nfeatures)) {
}

size_t SparseSAGA::maxDt_ = 20000;

SparseSAGA::SparseSAGA(const size_t nfeatures,
                       const size_t nexamples,
                       const Double lr,
                       const Double lambda,
                       const std::string& loss)
  : SAGABase(nfeatures, nexamples, lr, lambda, loss),
    g_(nexamples, nfeatures),
    gbar_(Vector::Zero(nfeatures)),
    ws_(Vector::Zero(nfeatures)),
    s_(1.0),
    lastUpdate_(nfeatures) {
  updateFactor_.reserve(maxDt_);
  updateFactor_.push_back(0.0);
  Double factor = 1.0;
  for (size_t i = 0; i < maxDt_; ++i) {
    const Double f = updateFactor_.back() + factor;
    updateFactor_.push_back(f);
    factor *= (1.0 - lr * lambda);
  }
}
}
