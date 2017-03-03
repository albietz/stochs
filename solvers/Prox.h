
#pragma once

#include <random>
#include <string>
#include <type_traits>

#include "common.h"

namespace solvers {

class Prox {
 public:
  static Double computePenalty(const Vector& w, const std::string& prox) {
    if (prox == "none") {
      return 0;
    } else if (prox == "l1") {
      return w.lpNorm<1>();
    } else {
      LOG_EVERY_N(ERROR, 1000) << "prox not supported: " << prox;
      return 0;
    }
  }

  static void applyProx(Vector& w, const std::string& prox, const Double step) {
    if (prox == "none") {
      return;
    } else if (prox == "l1") {
      w = w.unaryExpr([step](Double val) {
        if (val > step) {
          return val - step;
        } else if (val < -step) {
          return val + step;
        } else {
          return static_cast<Double>(0);
        }
      });
    } else {
      LOG_EVERY_N(ERROR, 1000) << "prox not supported: " << prox;
    }
  }
};
}
