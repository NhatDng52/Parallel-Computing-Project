#include "matmul_naive.h"
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
    /*
        Implement code 
    */
   return vector<vector<double>>{};
}
vector<vector<double>> matrix_mult_naive_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    /*
        Implement code 
    */
   return vector<vector<double>>{};
}

vector<vector<double>> matrix_mult_naive_Hybrid(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    /*
        Implement code 
    */
   return vector<vector<double>>{};
}