#include "strassen_op.h"
#include <mpi.h>
#include "utils.h"

class BetterStrassenOpenMPI : public IStrassenOp {
    private:
        MPI_Comm create_dynamic_comm(const string &worker_exec, int num_workers, const string &hostfile) {
            MPI_Comm inter_comm;
            MPI_Info info;
            MPI_Info_create(&info);
            MPI_Info_set(info, "hostfile", hostfile.c_str());

            MPI_Comm_spawn(worker_exec.c_str(),
                           MPI_ARGV_NULL,
                           num_workers,
                           info,
                           0,
                           MPI_COMM_SELF,
                           &inter_comm,
                           MPI_ERRCODES_IGNORE);

            MPI_Info_free(&info);
            return inter_comm;
        }

    public:
        Matrix apply_strassen(const Matrix &A, const Matrix &B, bool isWorker=false) override {

            if (isWorker) {
                return Matrix();
            } else {
                int initialized;
                MPI_Initialized(&initialized);
                bool should_finalize = false;

                if (!initialized) {
                    cout << "Initializing MPI environment inside BetterStrassenOpenMPI.\n";
                    MPI_Init(nullptr, nullptr);
                    should_finalize = true;
                }

                static MPI_Comm intercomm = MPI_COMM_NULL;

                static bool comm_created = false;

                if (!comm_created) {
                    intercomm = create_dynamic_comm("./strassen_worker_mpi", 4, "hostfile/hostfile.txt");
                    comm_created = true;
                    cout << "Created MPI inter-communicator with workers.\n";
                }

                // TODO: Implement the Strassen algorithm using MPI with the created intercomm

                if (should_finalize) {
                    MPI_Finalize();
                }

                return Matrix();
            }
        }
};