
#include "Loss.h"

namespace solvers {

std::mt19937 Loss::gen_ = std::mt19937(std::random_device()());

Double Loss::gradSigma_ = -1;

Double Loss::computeLoss(const std::string& loss,
                         const Double pred,
                         const Double y) {
  if (loss == "l2") {
    const auto err = pred - y;
    return 0.5 * err * err;
  } else if (loss == "logistic") {
    const auto sigm = 1.0 / (1 + std::exp(-pred));
    return -y * std::log(sigm) - (1 - y) * std::log(1 - sigm);
  } else if (loss == "squared_hinge") {
    const Double s = y > 0 ? pred : -pred;
    const Double hinge = std::max(0.0, 1.0 - s);
    return 0.5 * hinge * hinge;
  } else {
    LOG(ERROR) << "loss not supported: " << loss;
    return 0;
  }
}
}
