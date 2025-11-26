#include "strassen_op.h"
#include <mpi.h>
#include <queue>
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

        static vector<double> flatten(const Matrix &A) {
            int rows = A.size();
            if (rows == 0) return vector<double>();
            int cols = A[0].size();
            vector<double> flat(rows * cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    flat[i * cols + j] = A[i][j];
                }
            }
            return flat;
        }

        static Matrix unflatten(const vector<double> &flat, int rows, int cols) {
            Matrix A(rows, vector<double>(cols));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    A[i][j] = flat[i * cols + j];
                }
            }
            return A;
        }
    };

struct Task {
    Matrix A;
    Matrix B;
    int task_id;
};

class StrassenOpenMPI : public IStrassenOp {
    private:
        const int THRESHOLD = 64;
        MPI_Comm global_comm = MPI_COMM_WORLD;
        int world_rank;
        int world_size;

        void get_mpi_info() {
            MPI_Comm_rank(global_comm, &world_rank);
            MPI_Comm_size(global_comm, &world_size);
        }

        vector<Task> distribute_matrix(const Matrix &A, const Matrix &B) {
            vector<Matrix> As = MPIUtils::divide_mat(A);
            vector<Matrix> Bs = MPIUtils::divide_mat(B);
            Matrix A11=As[0], A12=As[1], A21=As[2], A22=As[3];
            Matrix B11=Bs[0], B12=Bs[1], B21=Bs[2], B22=Bs[3];

            Matrix S1 = MPIUtils::mat_add(A11, A22); // M1
            Matrix S2 = MPIUtils::mat_add(B11, B22); // M1
            Matrix S3 = MPIUtils::mat_add(A21, A22); // M2
            Matrix S4 = MPIUtils::mat_sub(B12, B22); // M3
            Matrix S5 = MPIUtils::mat_sub(B21, B11); // M4
            Matrix S6 = MPIUtils::mat_add(A11, A12); // M5
            Matrix S7 = MPIUtils::mat_sub(A21, A11); // M6
            Matrix S8 = MPIUtils::mat_add(B11, B12); // M6
            Matrix S9 = MPIUtils::mat_sub(A12, A22); // M7
            Matrix S10 = MPIUtils::mat_add(B21, B22); // M7

            return {
                {S1, S2, 0}, // M1
                {S3, B11, 1}, // M2
                {A11, S4, 2}, // M3
                {A22, S5, 3}, // M4
                {S6, B22, 4}, // M5
                {S7, S8, 5}, // M6
                {S9, S10, 6}  // M7
            };
        }

        Matrix strassen_recursive(const Matrix &A, const Matrix &B) {
            int n = A.size();

            if (n <= THRESHOLD) {
                return MPIUtils::mat_mul_naive(A, B);
            }

            vector<Matrix> As = MPIUtils::divide_mat(A);
            vector<Matrix> Bs = MPIUtils::divide_mat(B);
            Matrix A11=As[0], A12=As[1], A21=As[2], A22=As[3];
            Matrix B11=Bs[0], B12=Bs[1], B21=Bs[2], B22=Bs[3];

            Matrix S1 = MPIUtils::mat_add(A11, A22); // M1
            Matrix S2 = MPIUtils::mat_add(B11, B22); // M1
            Matrix S3 = MPIUtils::mat_add(A21, A22); // M2
            Matrix S4 = MPIUtils::mat_sub(B12, B22); // M3
            Matrix S5 = MPIUtils::mat_sub(B21, B11); // M4
            Matrix S6 = MPIUtils::mat_add(A11, A12); // M5
            Matrix S7 = MPIUtils::mat_sub(A21, A11); // M6
            Matrix S8 = MPIUtils::mat_add(B11, B12); // M6
            Matrix S9 = MPIUtils::mat_sub(A12, A22); // M7
            Matrix S10 = MPIUtils::mat_add(B21, B22); // M7

            Matrix M1 = strassen_recursive(S1, S2);
            Matrix M2 = strassen_recursive(S3, B11);
            Matrix M3 = strassen_recursive(A11, S4);
            Matrix M4 = strassen_recursive(A22, S5);
            Matrix M5 = strassen_recursive(S6, B22);
            Matrix M6 = strassen_recursive(S7, S8);
            Matrix M7 = strassen_recursive(S9, S10);

            Matrix C11 = MPIUtils::mat_add(MPIUtils::mat_sub(MPIUtils::mat_add(M1, M4), M5), M7);
            Matrix C12 = MPIUtils::mat_add(M3, M5);
            Matrix C21 = MPIUtils::mat_add(M2, M4);
            Matrix C22 = MPIUtils::mat_add(MPIUtils::mat_sub(MPIUtils::mat_add(M1, M3), M2), M6);

            int new_size = C11.size();
            Matrix C(n, vector<double>(n));
            for (int i = 0; i < new_size; i++) {
                for (int j = 0; j < new_size; j++) {
                    C[i][j] = C11[i][j];
                    C[i][j + new_size] = C12[i][j];
                    C[i + new_size][j] = C21[i][j];
                    C[i + new_size][j + new_size] = C22[i][j];
                }
            }
            return C;
        }

        Matrix master_process(const Matrix &A, const Matrix &B) {
            vector<Task> tasks = distribute_matrix(A, B);
            vector<Matrix> M(7);
            
            int num_workers = world_size - 1;
            
            for (size_t i = 1; i < tasks.size() && i <= (size_t)num_workers; ++i) {
                vector<double> flat_A = MPIUtils::flatten(tasks[i].A);
                vector<double> flat_B = MPIUtils::flatten(tasks[i].B);

                int rows_A = tasks[i].A.size();
                int cols_A = tasks[i].A[0].size();
                int rows_B = tasks[i].B.size();
                int cols_B = tasks[i].B[0].size();

                MPI_Send(&rows_A, 1, MPI_INT, i, 0, global_comm);
                MPI_Send(&cols_A, 1, MPI_INT, i, 0, global_comm);
                MPI_Send(&rows_B, 1, MPI_INT, i, 0, global_comm);
                MPI_Send(&cols_B, 1, MPI_INT, i, 0, global_comm);
                MPI_Send(flat_A.data(), rows_A * cols_A, MPI_DOUBLE, i, 0, global_comm);
                MPI_Send(flat_B.data(), rows_B * cols_B, MPI_DOUBLE, i, 0, global_comm);
            }

            M[0] = strassen_recursive(tasks[0].A, tasks[0].B);
            for (size_t i = num_workers + 1; i < tasks.size(); ++i) {
                M[i] = strassen_recursive(tasks[i].A, tasks[i].B);
            }
            
            for (size_t i = 1; i < tasks.size() && i <= (size_t)num_workers; ++i) {
                int rows_M, cols_M;
                MPI_Recv(&rows_M, 1, MPI_INT, i, 0, global_comm, MPI_STATUS_IGNORE);
                MPI_Recv(&cols_M, 1, MPI_INT, i, 0, global_comm, MPI_STATUS_IGNORE);
                vector<double> flat_M(rows_M * cols_M);
                MPI_Recv(flat_M.data(), rows_M * cols_M, MPI_DOUBLE, i, 0, global_comm, MPI_STATUS_IGNORE);
                M[i] = MPIUtils::unflatten(flat_M, rows_M, cols_M);
            }

            Matrix C11 = MPIUtils::mat_add(MPIUtils::mat_sub(MPIUtils::mat_add(M[0], M[3]), M[4]), M[6]);
            Matrix C12 = MPIUtils::mat_add(M[2], M[4]);
            Matrix C21 = MPIUtils::mat_add(M[1], M[3]);
            Matrix C22 = MPIUtils::mat_add(MPIUtils::mat_sub(MPIUtils::mat_add(M[0], M[2]), M[1]), M[5]);

            int new_size = C11.size();
            Matrix C(A.size(), vector<double>(B[0].size()));
            for (int i = 0; i < new_size; i++) {
                for (int j = 0; j < new_size; j++) {
                    C[i][j] = C11[i][j];
                    C[i][j + new_size] = C12[i][j];
                    C[i + new_size][j] = C21[i][j];
                    C[i + new_size][j + new_size] = C22[i][j];
                }
            }

            return C;
        }

        void worker_process() {
            while (true) {
                int rows_A, cols_A, rows_B, cols_B;
                MPI_Recv(&rows_A, 1, MPI_INT, 0, 0, global_comm, MPI_STATUS_IGNORE);
                
                if (rows_A == -1) {
                    break;
                }
                
                MPI_Recv(&cols_A, 1, MPI_INT, 0, 0, global_comm, MPI_STATUS_IGNORE);
                MPI_Recv(&rows_B, 1, MPI_INT, 0, 0, global_comm, MPI_STATUS_IGNORE);
                MPI_Recv(&cols_B, 1, MPI_INT, 0, 0, global_comm, MPI_STATUS_IGNORE);

                vector<double> flat_A(rows_A * cols_A);
                vector<double> flat_B(rows_B * cols_B);
                MPI_Recv(flat_A.data(), rows_A * cols_A, MPI_DOUBLE, 0, 0, global_comm, MPI_STATUS_IGNORE);
                MPI_Recv(flat_B.data(), rows_B * cols_B, MPI_DOUBLE, 0, 0, global_comm, MPI_STATUS_IGNORE);

                Matrix A = MPIUtils::unflatten(flat_A, rows_A, cols_A);
                Matrix B = MPIUtils::unflatten(flat_B, rows_B, cols_B);

                Matrix M = strassen_recursive(A, B);

                vector<double> flat_M = MPIUtils::flatten(M);
                int rows_M = M.size();
                int cols_M = M[0].size();

                MPI_Send(&rows_M, 1, MPI_INT, 0, 0, global_comm);
                MPI_Send(&cols_M, 1, MPI_INT, 0, 0, global_comm);
                MPI_Send(flat_M.data(), rows_M * cols_M, MPI_DOUBLE, 0, 0, global_comm);
            }
        }

        Matrix mpi_strassen(const Matrix &A, const Matrix &B) {
            if (A.empty() || B.empty()) {
                if (world_rank != 0) {
                    worker_process();
                }
                return Matrix();
            }
            
            int n = max({A.size(), A[0].size(), B.size(), B[0].size()});
            int m = MPIUtils::next_power_of_two(n);

            Matrix A_padded = MPIUtils::padding(A, m);
            Matrix B_padded = MPIUtils::padding(B, m);
            Matrix C_padded;

            if (world_rank == 0) {
                C_padded = master_process(A_padded, B_padded);
            } else {
                worker_process();
            }

            Matrix C = MPIUtils::remove_padding(C_padded, A.size(), B[0].size());
            return C;
        }

    public:
        Matrix apply_strassen(const Matrix &A, const Matrix &B) override {
            int initialized;
            MPI_Initialized(&initialized);
            
            bool should_finalize = false;

            if (!initialized) {
                MPI_Init(nullptr, nullptr);
                should_finalize = true;
            }

            get_mpi_info();
            
            Matrix result = mpi_strassen(A, B);
            
            if (should_finalize) {
                MPI_Finalize();
            }
            
            return result;
        }
        
        void cleanup() override {
            get_mpi_info();
            if (world_rank == 0) {
                int num_workers = world_size - 1;
                for (int i = 1; i <= num_workers; ++i) {
                    int terminate = -1;
                    MPI_Send(&terminate, 1, MPI_INT, i, 0, global_comm);
                }
            }
        }
};