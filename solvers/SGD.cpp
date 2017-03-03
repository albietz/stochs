
#include "SGD.h"

namespace solvers {

SGDBase::SGDBase(const size_t nfeatures,
                 const Double lr,
                 const Double lambda,
                 const std::string& loss,
                 const std::string& prox,
                 const Double proxWeight,
                 const bool average)
  : Solver(nfeatures, loss, prox, proxWeight),
    lr_(lr),
    lambda_(lambda),
    decay_(false),
    t_(1),
    t0_(1),
    average_(average) {
  if (average_) {
    wavg_ = Vector::Zero(nfeatures);
  }
}

void SGDBase::startDecay() {
  decay_ = true;
  t0_ = t_;
  gamma_ = static_cast<size_t>(2.0 / (lambda_ * lr_)) + 1;

  if (average_) {
    wavg_ = w_;
  }
}

SGD::SGD(const size_t nfeatures,
         const Double lr,
         const Double lambda,
         const std::string& loss,
         const std::string& prox,
         const Double proxWeight,
         const bool average)
  : SGDBase(nfeatures, lr, lambda, loss, prox, proxWeight, average),
    grad_(nfeatures) {
}

SparseSGD::SparseSGD(const size_t nfeatures,
                     const Double lr,
                     const Double lambda,
                     const std::string& loss)
  : SGDBase(nfeatures, lr, lambda, loss),
    ws_(Vector::Zero(nfeatures)),
    s_(1.0),
    grad_(nfeatures) {
}
}
