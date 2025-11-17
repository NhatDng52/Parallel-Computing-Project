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

int main(int argc, char** argv) {
    
    MPI_Init(&argc, &argv);
    
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    IStrassenOp* op = new StrassenOpenMPI();
    
    if (rank == 0) {
        srand(static_cast<unsigned>(time(0)));
        
        cout << "========================================\n";
        cout << "   Testing Strassen MPI Implementation  \n";
        cout << "   Running with " << world_size << " processes\n";
        cout << "========================================\n\n";
        
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

        cout << "TEST 2: Random: 256\n";
        cout << "-----------------------------------------\n";
        
        Matrix A2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
        Matrix B2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");


        cout << "Master process starting computation...\n";
        auto start2 = high_resolution_clock::now();
        
        bool test2_completed = run_with_timeout([&]() -> bool {
            auto C_strassen2 = op->apply_strassen(A2, B2);
            auto C_expected2 = ground_truth_mul(A2, B2);
            return compare_matrices(C_strassen2, C_expected2);
        }, MAX_TEST_TIME_SECONDS, "TEST 2");
        
        auto end2 = high_resolution_clock::now();
        duration<double> elapsed2 = end2 - start2;
        
        cout << "Execution time: " << fixed << setprecision(3) << elapsed2.count() << " seconds\n";

        if (elapsed2.count() > MAX_TEST_TIME_SECONDS) {
            cout << "TEST 2 FAILED: Execution time exceeded " << MAX_TEST_TIME_SECONDS << " seconds.\n";
        } else if (test2_completed) {
            cout << "TEST 2 PASSED: Strassen MPI result matches expected multiplication.\n";
            passed_tests++;
        } else {
            cout << "TEST 2 FAILED: Test timed out or result mismatch.\n";
        }
        total_tests++;
        
        cout << "TEST 3: Medium Test: 256x256 with Negative Values\n";
        Matrix C2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_C_256x256.csv");
        
        auto start3 = high_resolution_clock::now();
        
        bool test3_completed = run_with_timeout([&]() -> bool {
            auto C_strassen3 = op->apply_strassen(A2, C2);
            auto C_expected3 = ground_truth_mul(A2, C2);
            return compare_matrices(C_strassen3, C_expected3);
        }, MAX_TEST_TIME_SECONDS, "TEST 3");
        
        auto end3 = high_resolution_clock::now();
        duration<double> elapsed3 = end3 - start3;
        
        cout << "Execution time: " << fixed << setprecision(3) << elapsed3.count() << " seconds\n";

        if (elapsed3.count() > MAX_TEST_TIME_SECONDS) {
            cout << "TEST 3 FAILED: Execution time exceeded " << MAX_TEST_TIME_SECONDS << " seconds.\n";
        } else if (test3_completed) {
            cout << "TEST 3 PASSED: Strassen MPI result matches expected multiplication.\n";
            passed_tests++;
        } else {
            cout << "TEST 3 FAILED: Test timed out or result mismatch.\n";
        }

        total_tests++;

        cout << "TEST 4:\n";
        auto start4 = high_resolution_clock::now();
        
        bool test4_completed = run_with_timeout([&]() -> bool {
            auto C_strassen4 = op->apply_strassen(B2, C2);
            auto C_expected4 = ground_truth_mul(B2, C2);
            return compare_matrices(C_strassen4, C_expected4);
        }, MAX_TEST_TIME_SECONDS, "TEST 4");
        
        auto end4 = high_resolution_clock::now();
        duration<double> elapsed4 = end4 - start4;
        
        cout << "Execution time: " << fixed << setprecision(3) << elapsed4.count() << " seconds\n";

        if (elapsed4.count() > MAX_TEST_TIME_SECONDS) {
            cout << "TEST 4 FAILED: Execution time exceeded " << MAX_TEST_TIME_SECONDS << " seconds.\n";
        } else if (test4_completed) {
            cout << "TEST 4 PASSED: Strassen MPI result matches expected multiplication.\n";
            passed_tests++;
        } else {
            cout << "TEST 4 FAILED: Test timed out or result mismatch.\n";
        }

        total_tests++;


        
        cout << "========================================\n";
        cout << "   Test Summary                         \n";
        cout << "========================================\n";
        cout << "Total tests: " << total_tests << "\n";
        cout << "Passed: " << passed_tests << "\n";
        cout << "Failed: " << (total_tests - passed_tests) << "\n";
        cout << "Success rate: " << fixed << setprecision(1) 
             << (100.0 * passed_tests / total_tests) << "%\n";
        cout << "========================================\n";
    } else {
        op->apply_strassen(Matrix(), Matrix());
    }
    
    delete op;
    MPI_Finalize();
    
    return 0;
}
