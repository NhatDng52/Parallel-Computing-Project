#ifndef STRASSEN_HYBRID_H
#define STRASSEN_HYBRID_H

#include "strassen_op.h"
#include <mpi.h>
#include <omp.h>

class StrassenHybrid : public IStrassenOp {
  public:
    Matrix apply_strassen(const Matrix &A, const Matrix &B) override;

  private:
    const int THRESHOLD = 32;

    Matrix padding(const Matrix &A);
    Matrix implement_strassen(const Matrix &A, const Matrix &B, bool use_mpi);

    // Helper functions for matrix operations (parallelized with OpenMP)
    Matrix mat_add(const Matrix &A, const Matrix &B);
    Matrix mat_sub(const Matrix &A, const Matrix &B);
    Matrix mat_mul_naive(const Matrix &A, const Matrix &B);

    // Helper functions for matrix manipulation
    vector<Matrix> separate_mat(const Matrix &A);
    Matrix combine_mat(const Matrix &C11, const Matrix &C12, const Matrix &C21, const Matrix &C22);
};

#endif // STRASSEN_HYBRID_H
