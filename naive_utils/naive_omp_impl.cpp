#include "naive_omp.h"

Matrix NaiveOmp::matrix_mult(const Matrix &A, const Matrix &B) {
    size_t n = A.size();
    size_t m = A[0].size();
    size_t p = B[0].size();
    Matrix result(n, vector<double>(p, 0.0));

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}