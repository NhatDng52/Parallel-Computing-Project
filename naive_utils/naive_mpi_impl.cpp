#include "naive_mpi.h"
#include <vector>
#include <iostream>

using namespace std;

vector<double> flatten(const Matrix& mat) {
    if (mat.empty()) return {};
    int rows = mat.size();
    int cols = mat[0].size();
    vector<double> flat(rows * cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}

Matrix unflatten(const vector<double>& flat, int rows, int cols) {
    Matrix mat(rows, vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = flat[i * cols + j];
        }
    }
    return mat;
}

Matrix NaiveMpi::matrix_mult(const Matrix &A, const Matrix &B) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 0;
    int M = 0;
    int K = 0;

    vector<double> flat_A; 
    vector<double> flat_B;

    if (world_rank == 0) {
        if (A.empty() || B.empty()) {
            N = M = K = 0;
        } else {
            N = A.size();
            M = A[0].size();
            K = B[0].size();
            
            if (M != B.size()) {
                cerr << "Matrix dimensions do not match for multiplication." << endl;
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            flat_A = flatten(A);
            flat_B = flatten(B);
        }
    }

    // --- BƯỚC 2: BROADCAST KÍCH THƯỚC ---
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- BƯỚC 3: BROADCAST MA TRẬN B ---
    if (world_rank != 0) {
        flat_B.resize(M * K);
    }

    MPI_Bcast(flat_B.data(), M * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // --- BƯỚC 4: CHIA A (SCATTERV) ---
    vector<int> sendcounts(world_size);
    vector<int> displs(world_size);
    
    int rows_per_proc = N / world_size;
    int remainder = N % world_size;
    int offset = 0;

    for (int i = 0; i < world_size; ++i) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        sendcounts[i] = rows * M;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    // Cấp phát bộ nhớ nhận A cục bộ
    int my_count = sendcounts[world_rank];
    int my_rows = my_count / M;
    vector<double> local_A(my_count);

    // Scatter từ mảng đã duỗi flat_A
    MPI_Scatterv(flat_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // --- BƯỚC 5: TÍNH TOÁN CỤC BỘ ---
    vector<double> local_C(my_rows * K, 0.0);

    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                double a_val = local_A[i * M + k];
                double b_val = flat_B[k * K + j]; 
                sum += a_val * b_val;
            }
            local_C[i * K + j] = sum;
        }
    }

    // --- BƯỚC 6: GOM KẾT QUẢ (GATHERV) ---
    vector<double> flat_C;
    if (world_rank == 0) {
        flat_C.resize(N * K);
    }

    vector<int> recvcounts(world_size);
    vector<int> rdispls(world_size);
    offset = 0;
    for (int i = 0; i < world_size; ++i) {
        int rows = rows_per_proc + (i < remainder ? 1 : 0);
        recvcounts[i] = rows * K;
        rdispls[i] = offset;
        offset += recvcounts[i];
    }

    MPI_Gatherv(local_C.data(), recvcounts[world_rank], MPI_DOUBLE,
                flat_C.data(), recvcounts.data(), rdispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    Matrix C;
    if (world_rank == 0) {
        C = unflatten(flat_C, N, K);
    }

    return C;
}