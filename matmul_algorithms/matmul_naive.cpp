#include "matmul_naive.h"
#include "naive_utils/naive_omp.h"
#include "naive_utils/naive_mpi.h"
#include "naive_utils/naive_hybrid.h"

using namespace std;
vector<vector<double>> matrix_mult_naive(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    vector<vector<double>> result(A.size(), vector<double>(B[0].size(), 0.0));
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < B[0].size(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < A[0].size(); ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
   return result;
}
vector<vector<double>> matrix_mult_naive_OpenMP(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    return NaiveOmp::matrix_mult(A, B);
}
vector<vector<double>> matrix_mult_naive_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    return NaiveMpi::matrix_mult(A, B);
}

vector<vector<double>> matrix_mult_naive_Hybrid(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    return NaiveHybrid::matrix_mult(A, B);
}