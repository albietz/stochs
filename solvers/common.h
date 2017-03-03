
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <glog/logging.h>
#include <iostream>

namespace solvers {

#if USE_FLOAT
using Double = float;
#else
using Double = double;
#endif

using Matrix =
  Eigen::Matrix<Double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<const Matrix>;

using Vector = Eigen::Matrix<Double, Eigen::Dynamic, 1>;
using VectorMap = Eigen::Map<const Vector>;
using RowVector = Eigen::Matrix<Double, 1, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorMap = Eigen::Map<const RowVector>;

using IdxVector = Eigen::Matrix<int64_t, Eigen::Dynamic, 1>;
using IdxVectorMap = Eigen::Map<const IdxVector>;

using SpMatrix = Eigen::SparseMatrix<Double, Eigen::RowMajor>;
using SpMatrixMap = Eigen::Map<const SpMatrix>;
using SpVector = Eigen::SparseVector<Double>;

/** subtract row mean from each row in data matrix */
inline void center(Double* const XData,
                   const size_t rows,
                   const size_t cols) {
#pragma omp parallel for
  for (size_t r = 0; r < rows; ++r) {
    Double sum = 0;
    for (size_t c = 0; c < cols; ++c) {
      sum += XData[r*cols + c];
    }
    sum /= cols;
    for (size_t c = 0; c < cols; ++c) {
      XData[r*cols + c] -= sum;
    }
  }
}

/** L2 normalize each row in data matrix */
inline void normalize(Double* const XData,
                      const size_t rows,
                      const size_t cols) {
#pragma omp parallel for
  for (size_t r = 0; r < rows; ++r) {
    Double sum = 0;
    for (size_t c = 0; c < cols; ++c) {
      Double x = XData[r*cols + c];
      sum += x * x;
    }
    sum = std::sqrt(sum);
    if (sum > 0) {
      for (size_t c = 0; c < cols; ++c) {
        XData[r*cols + c] /= sum;
      }
    }
  }
}

/** L2 normalize, sparse */
inline void normalize(const size_t rows,
                      const size_t cols,
                      const size_t nnz,
                      const int32_t* const indptr,
                      const int32_t* const indices,
                      Double* const values) {
#pragma omp parallel for
  for (size_t r = 0; r < rows; ++r) {
    Double sum = 0;
    for (int32_t idx = indptr[r]; idx < indptr[r + 1]; ++idx) {
      Double x = values[idx];
      sum += x * x;
    }
    sum = std::sqrt(sum);
    if (sum > 0) {
      for (int32_t idx = indptr[r]; idx < indptr[r + 1]; ++idx) {
        values[idx] /= sum;
      }
    }
  }
}
}
