
namespace solvers {

inline Double Loss::computeGradient(const std::string& loss,
                                    const Double pred,
                                    const Double y) {
  if (loss == "l2") {
    return pred - y;
  } else if (loss == "logistic") {
    const auto sigm = 1.0 / (1 + std::exp(-pred));
    return sigm - y;
  } else if (loss == "squared_hinge") {
    const Double s = y > 0 ? pred : -pred;
    if (s > 1) {
      return 0;
    } else {
      return (y > 0 ? -1 : 1) * (1.0 - s);
    }
  } else {
    LOG_EVERY_N(ERROR, 1000) << "loss not supported: " << loss;
    return 0;
  }
}

template <typename VectorT, typename Derived>
void Loss::computeGradient(VectorT& g,
                           const std::string& loss,
                           const detail::EBase<Derived, VectorT>& x,
                           const Double pred,
                           const Double y) {
  // gradient of phi
  const Double grad = computeGradient(loss, pred, y);
  if (grad == 0) {
    detail::makeZero(g, x.size());
  } else {
    g = grad * x.transpose();
  }

  if (gradSigma_ > 0) {
    std::normal_distribution<Double> distr(0.0, gradSigma_);
    g = g.unaryExpr([&distr](Double val) { return val + distr(gen_); });
  }
}
}
