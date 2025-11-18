#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib> 
#include <ctime>   
#include <chrono>
#include <future>
#include <csignal>
#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mpi.cpp"
#include "utils.h"

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

void testcase(int &total_tests, int &passed_tests, IStrassenOp* op, const string& test_name, const Matrix& A, const Matrix& B) {
    cout << test_name << "\n";
    cout << "-----------------------------------------\n";

    auto start = high_resolution_clock::now();

    bool test_completed = run_with_timeout([&]() -> bool {
        auto C_strassen = op->apply_strassen(A, B);
        auto C_expected = ground_truth_mul(A, B);
        return compare_matrices(C_strassen, C_expected);
    }, MAX_TEST_TIME_SECONDS, test_name);

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Execution time: " << fixed << setprecision(3) << elapsed.count() << " seconds\n";

    if (elapsed.count() > MAX_TEST_TIME_SECONDS) {
        cout << test_name << " FAILED: Execution time exceeded " << MAX_TEST_TIME_SECONDS << " seconds.\n";
    } else if (test_completed) {
        cout << test_name << " PASSED: Strassen MPI result matches ground truth multiplication.\n";
        passed_tests++;
    } else {
        cout << test_name << " FAILED: Test timed out or result mismatch.\n";
    }

    total_tests++;
}

void main_test(IStrassenOp* op) {
    int total_tests = 0;
    int passed_tests = 0;

    // TestCase 1
    Matrix A1 = read_matrix_from_csv("./test/test_case/medium_negative_matrix_256x256.csv");
    Matrix B1 = read_matrix_from_csv("./test/test_case/medium_positive_matrix_256x256.csv");
    testcase(total_tests, passed_tests, op, "TestCase 1: 256x256 Matrix Multiplication", A1, B1);

    // TestCase 2
    Matrix A2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
    Matrix B2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");
    testcase(total_tests, passed_tests, op, "TestCase 2: 256x256 Random Matrix Multiplication", A2, B2);

    // TestCase 3
    Matrix A3 = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
    Matrix C3 = read_matrix_from_csv("./test/test_case/medium_random_matrix_C_256x256.csv");
    testcase(total_tests, passed_tests, op, "TestCase 3: 512x512 Random Matrix Multiplication", A3, C3);
}

int main() {
    MPI_Init(nullptr, nullptr);
    
    int world_rank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    IStrassenOp *op = new StrassenOpenMPI();

    if (world_rank == 0) {
        main_test(op);
        op->cleanup();
    } else {
        op->apply_strassen(Matrix(), Matrix());
    }

    delete op;
    MPI_Finalize();
    return 0;
}

