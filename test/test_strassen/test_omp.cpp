#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib> 
#include <ctime>   
#include <algorithm>
#include <cmath>  
#include "strassen_utils/strassen_open_mp.cpp"
#include "utils.cpp"

using namespace std;

Matrix ground_truth_add(const Matrix& A, const Matrix& B) {
    int N = A.size();
    Matrix C(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix ground_truth_sub(const Matrix& A, const Matrix& B) {
    int N = A.size();
    Matrix C(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

Matrix ground_truth_mul(const Matrix& A, const Matrix& B) {
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        throw std::runtime_error("Lỗi kích thước ma trận: Số cột của A không bằng số hàng của B.");
    }

    int rows = A.size();       
    int cols = B[0].size();    
    int middleDim = A[0].size();

    Matrix C(rows, std::vector<double>(cols, 0.0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < middleDim; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

int main() {
    srand(static_cast<unsigned>(time(0)));
    IStrassenOp* op = new StrassenOpenMP();

    // Matrix A = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
    // Matrix B = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");
    
    // Matrix A = read_matrix_from_csv("./test/test_case/matrix_3x4.csv");
    // Matrix B = read_matrix_from_csv("./test/test_case/matrix_4x2.csv");

    Matrix A = read_matrix_from_csv("./test/test_case/medium_negative_matrix_256x256.csv");
    Matrix B = read_matrix_from_csv("./test/test_case/medium_positive_matrix_256x256.csv");
    
    cout << "\nTesting Mul Strassen...\n";
    auto C_strassen_op = op->apply_strassen(A, B);
    auto C_strassen_expected = ground_truth_mul(A, B);

    print_matrix(C_strassen_op);
    cout << "-----------------------------------------\n";
    print_matrix(C_strassen_expected);

    cout << "Strassen Mul OK? " 
        << (compare_matrices(C_strassen_op, C_strassen_expected) ? "YES" : "NO") 
        << "\n";

    return 0;
}

