#include "strassen_open_mpi.h"
#include "matmul_algorithms/matmul_naive.h"
#include <algorithm>

int WORLD_RANK, WORLD_SIZE;
const int THRESHOLD = 64;

vector<double> StrassenOpenMPI::flatten(const Matrix &A) {
    int n = A.size();
    vector<double> flat;
    flat.reserve(n * n);
    for (const auto &row : A) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
}

Matrix StrassenOpenMPI::unflatten(const vector<double> &flat, int n) {
    Matrix A(n, vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = flat[i * n + j];
        }
    }
    return A;
}

void StrassenOpenMPI::send_matrix(const Matrix &A, int dest_rank, int tag) {
    vector<double> flat = flatten(A);
    MPI_Send(flat.data(), flat.size(), MPI_DOUBLE, dest_rank, tag, MPI_COMM_WORLD);
}

Matrix StrassenOpenMPI::recv_matrix(int src_rank, int n, int tag) {
    vector<double> flat(n * n);
    MPI_Recv(flat.data(), n * n, MPI_DOUBLE, src_rank, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return unflatten(flat, n);
}

Matrix StrassenOpenMPI::mat_add(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix StrassenOpenMPI::mat_sub(const Matrix &A, const Matrix &B) {
    int n = A.size();
    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

vector<Matrix> StrassenOpenMPI::divide_mat(const Matrix &A) {
    int n = A.size();
    int mid = n / 2;
    Matrix A11(mid, vector<double>(mid));
    Matrix A12(mid, vector<double>(mid));
    Matrix A21(mid, vector<double>(mid));
    Matrix A22(mid, vector<double>(mid));

    for (int i = 0; i < mid; i++) {
        for (int j = 0; j < mid; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + mid];
            A21[i][j] = A[i + mid][j];
            A22[i][j] = A[i + mid][j + mid];
        }
    }
    return {A11, A12, A21, A22};
}

Matrix StrassenOpenMPI::padding(const Matrix &A) {
    int n = A.size();
    int m = A[0].size();
    int size = 1;
    while (size < std::max(n, m)) size <<= 1;

    if (size == n && size == m) return A;

    Matrix A_pad(size, vector<double>(size, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A_pad[i][j] = A[i][j];
    return A_pad;
}

Matrix StrassenOpenMPI::remove_padding(const Matrix &A, int original_rows, int original_cols) {
    Matrix res(original_rows, vector<double>(original_cols));
    for(int i=0; i<original_rows; i++)
        for(int j=0; j<original_cols; j++)
            res[i][j] = A[i][j];
    return res;
}

Matrix StrassenOpenMPI::local_compute(const Matrix &A, const Matrix &B) {
    if (A.empty() || B.empty()) return {};
    return matrix_mult_naive(A, B);
}

Matrix StrassenOpenMPI::apply_strassen(const Matrix &A, const Matrix &B) {
    MPI_Comm_rank(MPI_COMM_WORLD, &WORLD_RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &WORLD_SIZE);

    int n_dim = 0;
    if (WORLD_RANK == 0) {
        n_dim = A.empty() ? 0 : std::max((int)A.size(), (int)A[0].size());
    }
    
    MPI_Bcast(&n_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n_dim <= THRESHOLD) {
        if (WORLD_RANK == 0) {
            return matrix_mult_naive(A, B);
        } else {
            return {};
        }
    }

    if (WORLD_RANK == 0) {
        Matrix A_pad = padding(A);
        Matrix B_pad = padding(B);
        
        Matrix C_pad = implement_strassen(A_pad, B_pad);
        
        return remove_padding(C_pad, A.size(), B[0].size());
    } else {
        implement_strassen({}, {});
        return {};
    }
}

Matrix StrassenOpenMPI::implement_strassen(const Matrix &A, const Matrix &B) {
    int local_n = 0;

    if (WORLD_RANK == 0) {
        int n = A.size();
        local_n = n / 2;
        MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        auto pA = divide_mat(A);
        auto pB = divide_mat(B);
        Matrix A11=pA[0], A12=pA[1], A21=pA[2], A22=pA[3];
        Matrix B11=pB[0], B12=pB[1], B21=pB[2], B22=pB[3];

        vector<Matrix> lhs = { mat_add(A11, A22), mat_add(A21, A22), A11, A22, mat_add(A11, A12), mat_sub(A21, A11), mat_sub(A12, A22) };
        vector<Matrix> rhs = { mat_add(B11, B22), B11, mat_sub(B12, B22), mat_sub(B21, B11), B22, mat_add(B11, B12), mat_add(B21, B22) };

        int num_workers = WORLD_SIZE - 1; 
        if (num_workers == 0) return matrix_mult_naive(A, B); 

        vector<vector<double>> send_buffers_lhs(7);
        vector<vector<double>> send_buffers_rhs(7);
        vector<MPI_Request> requests; 
        requests.reserve(14);

        for (int i = 0; i < 7; i++) {
            int dest = (i % num_workers) + 1;
            
            send_buffers_lhs[i] = flatten(lhs[i]);
            MPI_Request req_l;
            MPI_Isend(send_buffers_lhs[i].data(), send_buffers_lhs[i].size(), MPI_DOUBLE, 
                      dest, i * 2, MPI_COMM_WORLD, &req_l);
            requests.push_back(req_l);

            send_buffers_rhs[i] = flatten(rhs[i]);
            MPI_Request req_r;
            MPI_Isend(send_buffers_rhs[i].data(), send_buffers_rhs[i].size(), MPI_DOUBLE, 
                      dest, i * 2 + 1, MPI_COMM_WORLD, &req_r);
            requests.push_back(req_r);
        }

        vector<Matrix> M(7);
        for (int i = 0; i < 7; i++) {
            int src = (i % num_workers) + 1;
            M[i] = recv_matrix(src, local_n, 100 + i);
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        Matrix C11 = mat_add(mat_sub(mat_add(M[0], M[3]), M[4]), M[6]);
        Matrix C12 = mat_add(M[2], M[4]);
        Matrix C21 = mat_add(M[1], M[3]);
        Matrix C22 = mat_add(mat_sub(mat_add(M[0], M[2]), M[1]), M[5]);

        Matrix C(n, vector<double>(n));
        for(int i=0; i<local_n; i++) {
            for(int j=0; j<local_n; j++) {
                C[i][j] = C11[i][j];
                C[i][j+local_n] = C12[i][j];
                C[i+local_n][j] = C21[i][j];
                C[i+local_n][j+local_n] = C22[i][j];
            }
        }
        return C;

    } else {
        MPI_Bcast(&local_n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        int num_workers = WORLD_SIZE - 1;
        vector<int> my_tasks;
        for(int i=0; i<7; i++) {
            if ((i % num_workers) + 1 == WORLD_RANK) my_tasks.push_back(i);
        }

        for (int task_id : my_tasks) {
            Matrix subA = recv_matrix(0, local_n, task_id * 2);
            Matrix subB = recv_matrix(0, local_n, task_id * 2 + 1);
            Matrix subC = local_compute(subA, subB);
            send_matrix(subC, 0, 100 + task_id);
        }
        return {};
    }
}