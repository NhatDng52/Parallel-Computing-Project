#include "test/correctness_test.h"
#include "test/performance_test.h"
#include "test/scalability_test.h"
#include "utils.h"
#include "mpi.h"
#include <iostream>
#include <vector>
#include "matmul_algorithms/matmul_naive.h"

using namespace std;
using Matrix = vector<vector<double>>;

void run_worker_shadow(int num_calls) {
    for(int i = 0; i < num_calls; ++i) {
        matrix_mult_naive_Hybrid({}, {});
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        test_correctness(matrix_mult_naive_Hybrid);
    } else {
        run_worker_shadow(14);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0) {
        cout << "=== RUNNING PERFORMANCE TEST ===" << endl;
        test_performance(matrix_mult_naive_Hybrid);
    } else {
        run_worker_shadow(3);
    }

    MPI_Finalize();
    return 0;
}