#ifndef STRASSEN_OPENMP_H
#define STRASSEN_OPENMP_H

#include "strassen_op.h"
#include <omp.h>

class StrassenOpenMP : public IStrassenOp {
  public:
    Matrix apply_strassen(const Matrix &A, const Matrix &B) override;

  private:
    const int THRESHOLD = 32;

    vector<Matrix> divide_mat(const Matrix &A);
    Matrix padding(const Matrix &A);
    Matrix implement_strassen(const Matrix &A, const Matrix &B);
    Matrix mat_add(const Matrix &A, const Matrix &B);
    Matrix mat_sub(const Matrix &A, const Matrix &B);
    Matrix mat_mul_naive(const Matrix &A, const Matrix &B);
};

#endif