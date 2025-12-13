#include "matmul_algorithms/matmul_strassen.h"
#include "test/correctness_test.h"
#include "test/performance_test.h"
#include "test/scalability_test.h"
#include <mpi.h>

int MPI_RANK;

void run_worker_shadow(int num_calls) {
    for(int i = 0; i < num_calls; ++i) {
        matrix_mult_strassen_OpenMPI({}, {});
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);

    if (MPI_RANK == 0) {
        test_correctness(matrix_mult_strassen_OpenMPI);
    } else {
        run_worker_shadow(14);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (MPI_RANK == 0) {
        test_performance(matrix_mult_strassen_OpenMPI);
    } else {
        run_worker_shadow(3);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}