#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mpi.h"
#include "utils.h"
#include "correctness_test.h"
#include "performance_test.h"
#include "scalability_test.h"
#include <mpi.h>
#include <iostream>

using namespace std;

vector<vector<double>> strassen_openmpi_wrapper(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    static StrassenOpenMPI op;
    return op.apply_strassen(A, B);
}

int main() {
    MPI_Init(nullptr, nullptr);
    
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        try {
            test_correctness(strassen_openmpi_wrapper);
            test_performance(strassen_openmpi_wrapper);
            test_scalability_OpenMPI(strassen_openmpi_wrapper);
            
            StrassenOpenMPI op;
            op.cleanup();
        } catch (const exception& e) {
            cerr << "\nError: " << e.what() << endl;
            MPI_Finalize();
            return 1;
        }
    } else {
        StrassenOpenMPI op;
        op.apply_strassen(Matrix(), Matrix());
    }
    
    MPI_Finalize();
    return 0;
}
