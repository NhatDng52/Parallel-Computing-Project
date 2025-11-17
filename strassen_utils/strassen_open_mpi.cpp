#include "strassen_op.h"
#include <mpi.h>
#include <queue>
#include <vector>
#include <time.h>

class StrassenOpenMPI : public IStrassenOp {

private:
    const int THRESHOLD = 64;
    
    struct Task {
        int task_id;
        Matrix A;
        Matrix B;
    };
    
    struct WorkerState {
        int rank;
        bool busy;
        int current_task_id;
        MPI_Request send_req_A;
        MPI_Request send_req_B;
        MPI_Request recv_req;
        vector<double> recv_buffer;
        int recv_dims[2];
        MPI_Request recv_dims_req;
    };

    vector<double> flatten(const Matrix &A) {
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

    Matrix unflatten(const vector<double> &flat, int rows, int cols) {
        Matrix A(rows, vector<double>(cols));
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                A[i][j] = flat[i * cols + j];
            }
        }
        return A;
    }

    void send_matrix_async(const Matrix &A, int dest, int tag, MPI_Comm comm, 
                          vector<double> &flat_buffer, MPI_Request &req) {
        int rows = A.size();
        int cols = (rows > 0) ? A[0].size() : 0;
        int dims[2] = {rows, cols};
        
        MPI_Send(dims, 2, MPI_INT, dest, tag, comm);

        if (rows > 0 && cols > 0) {
            flat_buffer = flatten(A);
            MPI_Isend(flat_buffer.data(), flat_buffer.size(), MPI_DOUBLE, dest, tag + 1, comm, &req);
        }
    }

    void send_matrix(const Matrix &A, int dest, int tag, MPI_Comm comm) {
        int rows = A.size();
        int cols = (rows > 0) ? A[0].size() : 0;
        int dims[2] = {rows, cols};
        
        MPI_Send(dims, 2, MPI_INT, dest, tag, comm);

        if (rows > 0 && cols > 0) {
            vector<double> flat = flatten(A);
            MPI_Send(flat.data(), flat.size(), MPI_DOUBLE, dest, tag + 1, comm);
        }
    }

    int next_power_of_two(int n) {
        if (n <= 0) return 1;
        if ((n & (n - 1)) == 0) return n;
        int power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }

    Matrix recv_matrix(int source, int tag, MPI_Comm comm) {
        int dims[2];
        MPI_Status status;
        
        MPI_Recv(dims, 2, MPI_INT, source, tag, comm, &status);
        
        int rows = dims[0];
        int cols = dims[1];

        if (rows == 0 || cols == 0) {
            return Matrix();
        }

        vector<double> flat(rows * cols);
        MPI_Recv(flat.data(), flat.size(), MPI_DOUBLE, source, tag + 1, comm, &status);
        
        return unflatten(flat, rows, cols);
    }
    
    void start_recv_matrix_async(int source, int tag, MPI_Comm comm,
                                 int* dims_buffer, MPI_Request &dims_req) {
        MPI_Irecv(dims_buffer, 2, MPI_INT, source, tag, comm, &dims_req);
    }
    
    Matrix complete_recv_matrix_async(int source, int tag, MPI_Comm comm,
                                      int* dims_buffer, MPI_Request &dims_req,
                                      vector<double> &data_buffer) {
        MPI_Status status;
        MPI_Wait(&dims_req, &status);
        
        int rows = dims_buffer[0];
        int cols = dims_buffer[1];
        
        if (rows == 0 || cols == 0) {
            return Matrix();
        }
        
        data_buffer.resize(rows * cols);
        MPI_Recv(data_buffer.data(), data_buffer.size(), MPI_DOUBLE, source, tag + 1, comm, &status);
        
        return unflatten(data_buffer, rows, cols);
    }

    vector<Matrix> split_matrix(const Matrix &A) {
        int n = A.size();
        int mid = n / 2;
        Matrix A11(mid, vector<double>(mid));
        Matrix A12(mid, vector<double>(mid));
        Matrix A21(mid, vector<double>(mid));
        Matrix A22(mid, vector<double>(mid));

        for(int i=0; i<mid; ++i) {
            for(int j=0; j<mid; ++j) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j+mid];
                A21[i][j] = A[i+mid][j];
                A22[i][j] = A[i+mid][j+mid];
            }
        }
        return {A11, A12, A21, A22};
    }

    Matrix mat_add(const Matrix &A, const Matrix &B) {
        int n = A.size();
        Matrix C(n, vector<double>(n));
        for(int i=0; i<n; ++i) {
            for(int j=0; j<n; ++j) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    Matrix mat_sub(const Matrix &A, const Matrix &B) {
        int n = A.size();
        Matrix C(n, vector<double>(n));
        for(int i=0; i<n; ++i) {
            for(int j=0; j<n; ++j) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    Matrix mat_mul_naive(const Matrix &A, const Matrix &B) {
        int n = A.size();
        int m = A[0].size();
        int p = B[0].size();

        Matrix C(n, vector<double>(p, 0.0));
        for(int i=0; i<n; ++i) {
            for(int j=0; j<p; ++j) {
                for(int k=0; k<m; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    Matrix padding(const Matrix &A, int new_size) {
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
    
    Matrix remove_padding(const Matrix &A, int original_rows, int original_cols) {
        Matrix A_trimmed(original_rows, vector<double>(original_cols));
        for(int i=0; i<original_rows; ++i) {
            for(int j=0; j<original_cols; ++j) {
                A_trimmed[i][j] = A[i][j];
            }
        }
        return A_trimmed;
    }

    Matrix strassen_sequential(const Matrix &A, const Matrix &B) {
        int n = A.size();
        int m = A[0].size();
        int p = B[0].size();
        
        if (n <= THRESHOLD) {
            return mat_mul_naive(A, B);
        }
        
        int max_dim = max({n, m, p});
        int new_size = next_power_of_two(max_dim);
        
        Matrix A_padded = padding(A, new_size);
        Matrix B_padded = padding(B, new_size);
        
        vector<Matrix> As = split_matrix(A_padded);
        vector<Matrix> Bs = split_matrix(B_padded);
        Matrix A11=As[0], A12=As[1], A21=As[2], A22=As[3];
        Matrix B11=Bs[0], B12=Bs[1], B21=Bs[2], B22=Bs[3];
        
        Matrix S1 = mat_sub(B12, B22);
        Matrix S2 = mat_add(A11, A12);
        Matrix S3 = mat_add(A21, A22);
        Matrix S4 = mat_sub(B21, B11);
        Matrix S5 = mat_add(A11, A22);
        Matrix S6 = mat_add(B11, B22);
        Matrix S7 = mat_sub(A12, A22);
        Matrix S8 = mat_add(B21, B22);
        Matrix S9 = mat_sub(A11, A21);
        Matrix S10 = mat_add(B11, B12);
        
        Matrix M1 = strassen_sequential(A11, S1);
        Matrix M2 = strassen_sequential(S2, B22);
        Matrix M3 = strassen_sequential(S3, B11);
        Matrix M4 = strassen_sequential(A22, S4);
        Matrix M5 = strassen_sequential(S5, S6);
        Matrix M6 = strassen_sequential(S7, S8);
        Matrix M7 = strassen_sequential(S9, S10);
        
        Matrix C11 = mat_add(mat_sub(mat_add(M5, M4), M2), M6);
        Matrix C12 = mat_add(M1, M2);
        Matrix C21 = mat_add(M3, M4);
        Matrix C22 = mat_add(mat_sub(mat_add(M5, M1), M3), M7);
        
        int mid = new_size / 2;
        Matrix C_padded(new_size, vector<double>(new_size));
        for(int i=0; i<mid; ++i) {
            for(int j=0; j<mid; ++j) {
                C_padded[i][j] = C11[i][j];
                C_padded[i][j+mid] = C12[i][j];
                C_padded[i+mid][j] = C21[i][j];
                C_padded[i+mid][j+mid] = C22[i][j];
            }
        }
        
        return remove_padding(C_padded, n, p);
    }

    Matrix master_distribute(const Matrix &A, const Matrix &B, int world_size) {
        
        vector<Matrix> As = split_matrix(A);
        vector<Matrix> Bs = split_matrix(B);
        Matrix A11=As[0], A12=As[1], A21=As[2], A22=As[3];
        Matrix B11=Bs[0], B12=Bs[1], B21=Bs[2], B22=Bs[3];

        Matrix S1 = mat_sub(B12, B22);
        Matrix S2 = mat_add(A11, A12);
        Matrix S3 = mat_add(A21, A22);
        Matrix S4 = mat_sub(B21, B11);
        Matrix S5 = mat_add(A11, A22);
        Matrix S6 = mat_add(B11, B22);
        Matrix S7 = mat_sub(A12, A22);
        Matrix S8 = mat_add(B21, B22);
        Matrix S9 = mat_sub(A11, A21);
        Matrix S10 = mat_add(B11, B12);

        vector<Task> tasks(7);
        tasks[0] = {0, A11, S1};
        tasks[1] = {1, S2, B22};  
        tasks[2] = {2, S3, B11};  
        tasks[3] = {3, A22, S4};  
        tasks[4] = {4, S5, S6};   
        tasks[5] = {5, S7, S8};   
        tasks[6] = {6, S9, S10};  
        
        vector<Matrix> M_results(7);
        int num_workers = world_size - 1;
        
        if (num_workers == 0) {
            for (int i = 0; i < 7; ++i) {
                M_results[i] = strassen_sequential(tasks[i].A, tasks[i].B);
            }
        } else {
            queue<Task> task_queue;
            for (auto &task : tasks) {
                task_queue.push(task);
            }
            
            vector<WorkerState> workers(num_workers);
            for (int i = 0; i < num_workers; ++i) {
                workers[i].rank = i + 1;
                workers[i].busy = false;
                workers[i].current_task_id = -1;
            }
            
            vector<vector<double>> send_buffers_A(num_workers);
            vector<vector<double>> send_buffers_B(num_workers);
            
            int tasks_completed = 0;
            
            while (tasks_completed < 7) {
                // Assign tasks to idle workers
                for (int i = 0; i < num_workers && !task_queue.empty(); ++i) {
                    if (!workers[i].busy) {
                        Task task = task_queue.front();
                        task_queue.pop();
                        
                        workers[i].busy = true;
                        workers[i].current_task_id = task.task_id;
                        
                        send_matrix(task.A, workers[i].rank, 10, MPI_COMM_WORLD);
                        send_matrix(task.B, workers[i].rank, 20, MPI_COMM_WORLD);
                        
                        start_recv_matrix_async(workers[i].rank, 30, MPI_COMM_WORLD,
                                              workers[i].recv_dims, workers[i].recv_dims_req);
                    }
                }
                
                // Check for completed tasks
                bool any_completed = false;
                for (int i = 0; i < num_workers; ++i) {
                    if (workers[i].busy) {
                        int flag;
                        MPI_Status status;
                        MPI_Test(&workers[i].recv_dims_req, &flag, &status);
                        
                        if (flag) {
                            Matrix result = complete_recv_matrix_async(
                                workers[i].rank, 30, MPI_COMM_WORLD,
                                workers[i].recv_dims, workers[i].recv_dims_req,
                                workers[i].recv_buffer
                            );
                            
                            M_results[workers[i].current_task_id] = result;
                            workers[i].busy = false;
                            tasks_completed++;
                            any_completed = true;
                        }
                    }
                }
                
                // Small sleep to prevent busy waiting
                if (!any_completed && tasks_completed < 7) {
                    struct timespec ts;
                    ts.tv_sec = 0;
                    ts.tv_nsec = 1000000; // 1ms
                    nanosleep(&ts, NULL);
                }
            }
            
            // Send termination signal to all workers
            for (int i = 1; i < world_size; ++i) {
                send_matrix(Matrix(), i, 10, MPI_COMM_WORLD);
            }
        }

        Matrix M1=M_results[0], M2=M_results[1], M3=M_results[2], M4=M_results[3], 
               M5=M_results[4], M6=M_results[5], M7=M_results[6];

        Matrix C11 = mat_add(mat_sub(mat_add(M5, M4), M2), M6);
        Matrix C12 = mat_add(M1, M2);
        Matrix C21 = mat_add(M3, M4);
        Matrix C22 = mat_add(mat_sub(mat_add(M5, M1), M3), M7);

        int mid = A.size() / 2;
        Matrix C(A.size(), vector<double>(A.size()));
        for(int i=0; i<mid; ++i) {
            for(int j=0; j<mid; ++j) {
                C[i][j] = C11[i][j];
                C[i][j+mid] = C12[i][j];
                C[i+mid][j] = C21[i][j];
                C[i+mid][j+mid] = C22[i][j];
            }
        }
        return C;
    }

    void worker_compute() {
        // Process tasks until receiving termination signal
        while (true) {
            Matrix A_con = recv_matrix(0, 10, MPI_COMM_WORLD);
            
            if (A_con.empty()) {
                // Termination signal - exit this batch
                break;
            }
            
            Matrix B_con = recv_matrix(0, 20, MPI_COMM_WORLD);
            Matrix M_result = strassen_sequential(A_con, B_con);
            send_matrix(M_result, 0, 30, MPI_COMM_WORLD);
        }
    }

    Matrix strassen_implementation(const Matrix &A, const Matrix &B) {
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        Matrix C_final;

        if (rank == 0) {
            int original_rows = A.size();
            int original_cols = (B.empty() ? 0 : B[0].size());
            
            int max_size_to_pad = max({A.size(), (A.empty() ? 0 : A[0].size()), B.size(), (B.empty() ? 0 : B[0].size())});

            if (max_size_to_pad <= THRESHOLD) {
                return mat_mul_naive(A, B);
            }

            max_size_to_pad = next_power_of_two(max_size_to_pad);
            Matrix A_padded = padding(A, max_size_to_pad);
            Matrix B_padded = padding(B, max_size_to_pad);

            Matrix C_padded = master_distribute(A_padded, B_padded, world_size);
            
            C_final = remove_padding(C_padded, original_rows, original_cols);

        } else {
            // Worker process: process one task at a time
            worker_compute();
        } 

        return C_final;
    }

public:
    Matrix apply_strassen(const Matrix &A, const Matrix &B) {
        
        int initialized;
        MPI_Initialized(&initialized);
        
        bool should_finalize = false;

        if (!initialized) {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
            should_finalize = true;
        }
        
        Matrix result = strassen_implementation(A, B);
        
        if (should_finalize) {
            MPI_Finalize();
        }
        
        return result;
    }
};