#include "strassen_op.h"
#include <mpi.h>
#include "utils.h"

class MPIUtils {
    public:
        static Matrix mat_add(const Matrix &A, const Matrix &B) {
            if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_add strassen_op_omp");
            }

            int rows = A.size();
            int cols = A[0].size();

            Matrix C(rows, vector<double>(cols));

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    C[i][j] = A[i][j] + B[i][j];
                }
            }

            return C;
        }
        
        static Matrix mat_sub(const Matrix &A, const Matrix &B) {
            if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_sub strassen_op_omp");
            }

            int rows = A.size();
            int cols = A[0].size();

            Matrix C(rows, vector<double>(cols));

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    C[i][j] = A[i][j] - B[i][j];
                }
            }

            return C;
        }
        
        static Matrix mat_mul_naive(const Matrix &A, const Matrix &B) {
            // A(n x m) x B(m x k) -> C(n, k)

            if (A.empty() || B.empty() || A[0].size() != B.size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_mul strassen_op_omp");
            }

            int rows = A.size();
            int cols = B[0].size();
            int middleCos = A[0].size();

            Matrix C(rows, vector<double> (cols));

            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    for (int k = 0; k < middleCos; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            return C;
        }

        static int next_power_of_two(int n) {
            
            if (n <= 0) return 1;

            if ((n & (n - 1)) == 0) return n;
            
            int power = 1;
            
            while (power < n) {
                power <<= 1;
            }
            
            return power;
        }

        static vector<Matrix> divide_mat(const Matrix &A) {
            int mat_size = A.size();

            if (mat_size == 1) {
                return {A};
            } else {
                int sub_size = mat_size / 2;
                Matrix A11(sub_size, vector<double>(sub_size));
                Matrix A12(sub_size, vector<double>(sub_size));
                Matrix A21(sub_size, vector<double>(sub_size));
                Matrix A22(sub_size, vector<double>(sub_size));

                for (int i = 0; i < sub_size; i++) {
                    for (int j = 0; j < sub_size; j++) {
                        A11[i][j] = A[i][j];
                        A12[i][j] = A[i][j+sub_size];
                        A21[i][j] = A[i+sub_size][j];
                        A22[i][j] = A[i+sub_size][j+sub_size];
                    }
                }

                return {A11, A12, A21, A22};
            }
        }

        static Matrix padding(const Matrix &A, int new_size) {
            
            int old_rows = A.size();
            int old_cols = (old_rows > 0) ? A[0].size() : 0;
            Matrix A_padded(new_size, vector<double>(new_size, 0.0));

            for(int i=0; i<old_rows; ++i) {
                for(int j=0; j<old_cols; ++j) {
                    A_padded[i][j] = A[i][j];
                }
            }
            return A_padded;
        }

        static Matrix remove_padding(const Matrix &A, int original_rows, int original_cols) {
            Matrix A_trimmed(original_rows, vector<double>(original_cols));
            for(int i=0; i<original_rows; ++i) {
                for(int j=0; j<original_cols; ++j) {
                    A_trimmed[i][j] = A[i][j];
                }
            }
            return A_trimmed;
        }
};


class StrassenOpenMPI : public IStrassenOp {
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

        void master_work(MPI_Comm intercomm, const Matrix &A, const Matrix &B, Matrix &C) {
            (void)intercomm;
            (void)A;
            (void)B;
            (void)C;
        }

        void worker_work(MPI_Comm intercomm) {
            (void)intercomm;
        }

    public:
        Matrix apply_strassen(const Matrix &A, const Matrix &B, bool isWorker=false) override {
            int initialized;
            MPI_Initialized(&initialized);
            bool should_finalize = false;

            if (!initialized) {
                MPI_Init(nullptr, nullptr);
                should_finalize = true;
            }

            Matrix result = Matrix();

            if (isWorker) {
                worker_work(MPI_COMM_WORLD);
            } else {
                static MPI_Comm intercomm = MPI_COMM_NULL;

                static bool comm_created = false;

                if (!comm_created) {
                    intercomm = create_dynamic_comm("./strassen_worker_mpi", 4, "hostfile/hostfile.txt");
                    comm_created = true;
                    
                }

                master_work(intercomm, A, B, result);
            }

            if (should_finalize) {
                MPI_Finalize();
            }

            return result;
        }
};