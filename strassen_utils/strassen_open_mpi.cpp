#include "strassen_op.h"
#include "matmul_naive.h"
#include <mpi.h>

struct Task {
    Matrix A;
    Matrix B;
    int task_id;
};

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

static Matrix remove_padding(const Matrix &A, int original_rows, int original_cols) {
    Matrix A_trimmed(original_rows, vector<double>(original_cols));
    for(int i = 0; i < original_rows; ++i) {
        for(int j = 0; j < original_cols; ++j) {
            A_trimmed[i][j] = A[i][j];
        }
    }
    return A_trimmed;
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
            vector<Matrix> As = separate_mat(A);
            vector<Matrix> Bs = separate_mat(B);
            Matrix A11 = As[0], A12 = As[1], A21 = As[2], A22 = As[3];
            Matrix B11 = Bs[0], B12 = Bs[1], B21 = Bs[2], B22 = Bs[3];

            Matrix S1 = mat_add(A11, A22);
            Matrix S2 = mat_add(B11, B22);
            Matrix S3 = mat_add(A21, A22);
            Matrix S4 = mat_sub(B12, B22);
            Matrix S5 = mat_sub(B21, B11);
            Matrix S6 = mat_add(A11, A12);
            Matrix S7 = mat_sub(A21, A11);
            Matrix S8 = mat_add(B11, B12);
            Matrix S9 = mat_sub(A12, A22);
            Matrix S10 = mat_add(B21, B22);

            return {
                {S1, S2, 0},
                {S3, B11, 1},
                {A11, S4, 2},
                {A22, S5, 3},
                {S6, B22, 4},
                {S7, S8, 5},
                {S9, S10, 6}
            };
        }

        Matrix strassen_recursive(const Matrix &A, const Matrix &B) {
            int n = A.size();

            if (n <= THRESHOLD) {
                return matrix_mult_naive(A, B);
            }

            vector<Matrix> As = separate_mat(A);
            vector<Matrix> Bs = separate_mat(B);
            Matrix A11 = As[0], A12 = As[1], A21 = As[2], A22 = As[3];
            Matrix B11 = Bs[0], B12 = Bs[1], B21 = Bs[2], B22 = Bs[3];

            Matrix S1 = mat_add(A11, A22);
            Matrix S2 = mat_add(B11, B22);
            Matrix S3 = mat_add(A21, A22);
            Matrix S4 = mat_sub(B12, B22);
            Matrix S5 = mat_sub(B21, B11);
            Matrix S6 = mat_add(A11, A12);
            Matrix S7 = mat_sub(A21, A11);
            Matrix S8 = mat_add(B11, B12);
            Matrix S9 = mat_sub(A12, A22);
            Matrix S10 = mat_add(B21, B22);

            Matrix M1 = strassen_recursive(S1, S2);
            Matrix M2 = strassen_recursive(S3, B11);
            Matrix M3 = strassen_recursive(A11, S4);
            Matrix M4 = strassen_recursive(A22, S5);
            Matrix M5 = strassen_recursive(S6, B22);
            Matrix M6 = strassen_recursive(S7, S8);
            Matrix M7 = strassen_recursive(S9, S10);

            Matrix C11 = mat_add(mat_sub(mat_add(M1, M4), M5), M7);
            Matrix C12 = mat_add(M3, M5);
            Matrix C21 = mat_add(M2, M4);
            Matrix C22 = mat_add(mat_sub(mat_add(M1, M3), M2), M6);

            return combine_mat(C11, C12, C21, C22);
        }

        Matrix master_process(const Matrix &A, const Matrix &B) {
            vector<Task> tasks = distribute_matrix(A, B);
            vector<Matrix> M(7);
            
            int num_workers = world_size - 1;
            
            for (size_t i = 1; i < tasks.size() && i <= (size_t)num_workers; ++i) {
                vector<double> flat_A = flatten(tasks[i].A);
                vector<double> flat_B = flatten(tasks[i].B);

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
                M[i] = unflatten(flat_M, rows_M, cols_M);
            }

            Matrix C11 = mat_add(mat_sub(mat_add(M[0], M[3]), M[4]), M[6]);
            Matrix C12 = mat_add(M[2], M[4]);
            Matrix C21 = mat_add(M[1], M[3]);
            Matrix C22 = mat_add(mat_sub(mat_add(M[0], M[2]), M[1]), M[5]);

            return combine_mat(C11, C12, C21, C22);
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

                Matrix A = unflatten(flat_A, rows_A, cols_A);
                Matrix B = unflatten(flat_B, rows_B, cols_B);

                Matrix M = strassen_recursive(A, B);

                vector<double> flat_M = flatten(M);
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
            int m = next_power_of_two(n);

            Matrix A_padded = padding(A);
            Matrix B_padded = padding(B);
            Matrix C_padded;

            if (world_rank == 0) {
                C_padded = master_process(A_padded, B_padded);
            } else {
                worker_process();
            }

            Matrix C = remove_padding(C_padded, A.size(), B[0].size());
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