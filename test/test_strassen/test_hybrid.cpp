#include <iostream>
#include <vector>
#include <chrono>
#include <mpi.h>
#include <omp.h>
#include "../../matmul_algorithms/matmul_strassen.h"
#include "../../utils.h"

using namespace std;

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        cout << "=== Hybrid Strassen (MPI + OpenMP) Test ===" << endl;
        cout << "MPI Processes: " << size << endl;
        cout << "OpenMP Threads per process: " << omp_get_max_threads() << endl;
        cout << "===========================================" << endl << endl;
    }

    // Test 1: Small matrix (8x8)
    // All ranks must participate in MPI collective operations
    vector<vector<double>> A1(8, vector<double>(8));
    vector<vector<double>> B1(8, vector<double>(8));

    // Initialize with simple values (all ranks)
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            A1[i][j] = static_cast<double>(i + j);
            B1[i][j] = static_cast<double>(i - j);
        }
    }

    if (rank == 0) {
        cout << "Test 1: Small matrix (8x8)" << endl;
    }

    auto start1 = chrono::high_resolution_clock::now();
    vector<vector<double>> C1 = matrix_mult_strassen_Hybrid(A1, B1);  // All ranks call this!
    auto end1 = chrono::high_resolution_clock::now();

    if (rank == 0) {
        chrono::duration<double, std::milli> duration = end1 - start1;
        cout << "  Time: " << duration.count() << " ms" << endl;
        cout << "  Result dimension: " << C1.size() << "x" << C1[0].size() << endl;
        cout << "  Sample result C[0][0]: " << C1[0][0] << endl;
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 2: Medium matrix (256x256)
    // All ranks load and process matrices
    if (rank == 0) {
        cout << "Test 2: Medium matrix (256x256)" << endl;
    }

    try {
        // All ranks read the CSV files
        vector<vector<double>> A2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
        vector<vector<double>> B2 = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");

        auto start2 = chrono::high_resolution_clock::now();
        vector<vector<double>> C2 = matrix_mult_strassen_Hybrid(A2, B2);  // All ranks call this!
        auto end2 = chrono::high_resolution_clock::now();

        if (rank == 0) {
            chrono::duration<double, std::milli> duration = end2 - start2;
            cout << "  Time: " << duration.count() << " ms" << endl;
            cout << "  Result dimension: " << C2.size() << "x" << C2[0].size() << endl;
            cout << endl;
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "  Error: " << e.what() << endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 3: Correctness test with identity matrix
    if (rank == 0) {
        cout << "Test 3: Correctness test with identity matrix" << endl;
    }

    try {
        // All ranks read the CSV files
        vector<vector<double>> A3 = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
        vector<vector<double>> I3 = read_matrix_from_csv("./test/test_case/medium_identity_matrix_256x256.csv");

        auto start3 = chrono::high_resolution_clock::now();
        vector<vector<double>> C3 = matrix_mult_strassen_Hybrid(A3, I3);  // All ranks call this!
        auto end3 = chrono::high_resolution_clock::now();

        if (rank == 0) {
            chrono::duration<double, std::milli> duration = end3 - start3;
            bool correct = compare_matrices(A3, C3);
            cout << "  Time: " << duration.count() << " ms" << endl;
            cout << "  A * I == A: " << (correct ? "PASS" : "FAIL") << endl;
            cout << endl;
        }
    } catch (const exception& e) {
        if (rank == 0) {
            cerr << "  Error: " << e.what() << endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Test 4: Performance comparison with different matrix sizes
    if (rank == 0) {
        cout << "Test 4: Performance benchmark" << endl;
    }

    vector<int> sizes = {64, 128, 256, 512};

    // All ranks iterate through sizes
    for (int mat_size : sizes) {
        // All ranks create and initialize matrices
        vector<vector<double>> A4(static_cast<size_t>(mat_size), vector<double>(static_cast<size_t>(mat_size)));
        vector<vector<double>> B4(static_cast<size_t>(mat_size), vector<double>(static_cast<size_t>(mat_size)));

        // Initialize matrices (all ranks)
        for (int i = 0; i < mat_size; i++) {
            for (int j = 0; j < mat_size; j++) {
                A4[static_cast<size_t>(i)][static_cast<size_t>(j)] = static_cast<double>(rand()) / RAND_MAX;
                B4[static_cast<size_t>(i)][static_cast<size_t>(j)] = static_cast<double>(rand()) / RAND_MAX;
            }
        }

        auto start4 = chrono::high_resolution_clock::now();
        vector<vector<double>> C4 = matrix_mult_strassen_Hybrid(A4, B4);  // All ranks call this!
        auto end4 = chrono::high_resolution_clock::now();

        if (rank == 0) {
            chrono::duration<double, std::milli> duration = end4 - start4;
            cout << "  Matrix " << mat_size << "x" << mat_size << ": " << duration.count() << " ms" << endl;
        }
    }

    if (rank == 0) {
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "=== All tests completed ===" << endl;
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
