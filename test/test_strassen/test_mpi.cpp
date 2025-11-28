#include "matmul_algorithms/matmul_strassen.h"
#include "test/correctness_test.h"
#include "test/performance_test.h"
#include "test/scalability_test.h"
#include <mpi.h>

int MPI_RANK;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &MPI_RANK);

    if (MPI_RANK == 0) {
        test_1(matrix_mult_strassen_OpenMPI);
    } else {
        matrix_mult_strassen_OpenMPI({}, {});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPI_RANK == 0) {
        test_2(matrix_mult_strassen_OpenMPI);
    } else {
        matrix_mult_strassen_OpenMPI({}, {});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPI_RANK == 0) {
        test_3(matrix_mult_strassen_OpenMPI);
    } else {
        matrix_mult_strassen_OpenMPI({}, {});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPI_RANK == 0) {
        test_performance_1(matrix_mult_strassen_OpenMPI);
    } else {
        matrix_mult_strassen_OpenMPI({}, {});
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (MPI_RANK == 0) {
        test_performance_2(matrix_mult_strassen_OpenMPI);
    } else {
        matrix_mult_strassen_OpenMPI({}, {});
    }
    
    // MPI_Barrier(MPI_COMM_WORLD);

    // if (MPI_RANK == 0) {
    //     test_performance_3(matrix_mult_strassen_OpenMPI);
    // } else {
    //     matrix_mult_strassen_OpenMPI({}, {});
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}