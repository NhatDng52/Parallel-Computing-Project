#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib> 
#include <ctime>   
#include <chrono>
#include <future>
#include <csignal>
#include "../strassen_utils/strassen_op.h"
#include "../strassen_utils/strassen_open_mpi.cpp"
#include "../../utils.h"

using namespace std;
using namespace std::chrono;

const double MAX_TEST_TIME_SECONDS = 10.0;

template<typename Func>
bool run_with_timeout(Func func, double timeout_seconds, const string& test_name) {
    auto future = std::async(std::launch::async, func);
    
    if (future.wait_for(std::chrono::duration<double>(timeout_seconds)) == std::future_status::timeout) {
        cout << test_name << " TIMEOUT: Test exceeded " << timeout_seconds << " seconds and was terminated.\n";
        return false;
    }
    
    return future.get();
}

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
    IStrassenOp *op = new StrassenOpenMPI();
    int total_tests = 0;
    int passed_tests = 0;

    cout << "TEST 1: Negative 256x256 matrix from CSV\n";
    cout << "-----------------------------------------\n";
    
    Matrix A1 = read_matrix_from_csv("./test/test_case/medium_negative_matrix_256x256.csv");
    Matrix B1 = read_matrix_from_csv("./test/test_case/medium_positive_matrix_256x256.csv");
    
    cout << "Master process starting computation...\n";
    auto start1 = high_resolution_clock::now();
    
    bool test1_completed = run_with_timeout([&]() -> bool {
        auto C_strassen1 = op->apply_strassen(A1, B1);
        auto C_expected1 = ground_truth_mul(A1, B1);
        return compare_matrices(C_strassen1, C_expected1);
    }, MAX_TEST_TIME_SECONDS, "TEST 1");
    
    auto end1 = high_resolution_clock::now();
    duration<double> elapsed1 = end1 - start1;
    
    cout << "Execution time: " << fixed << setprecision(3) << elapsed1.count() << " seconds\n";
    
    if (elapsed1.count() > MAX_TEST_TIME_SECONDS) {
        cout << "TEST 1 FAILED: Execution time exceeded " << MAX_TEST_TIME_SECONDS << " seconds.\n";
    } else if (test1_completed) {
        cout << "TEST 1 PASSED: Strassen MPI result matches ground truth multiplication.\n";
        passed_tests++;
    } else {
        cout << "TEST 1 FAILED: Test timed out or result mismatch.\n";
    }

    total_tests++;
    
    return 0;
}
