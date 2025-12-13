#include "naive_hybrid.h"

using namespace std;

vector<double> NaiveHybrid::flatten(const Matrix& mat) {
    if (mat.empty()) return {};
    int rows = mat.size();
    int cols = mat[0].size();
    vector<double> flat(rows * cols);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            flat[i * cols + j] = mat[i][j];
        }
    }
    return flat;
}

Matrix NaiveHybrid::unflatten(const vector<double>& flat, int rows, int cols) {
    Matrix mat(rows, vector<double>(cols));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = flat[i * cols + j];
        }
    }
    return mat;
}

Matrix NaiveHybrid::matrix_mult(const Matrix &A, const Matrix &B) {
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 0, M = 0, K = 0;
    vector<double> flat_A; 
    vector<double> flat_B;

    // --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU Ở RANK 0 ---
    if (world_rank == 0) {
        if (!A.empty() && !B.empty()) {
            N = A.size();
            M = A[0].size();
            K = B[0].size();
            
            if (M != B.size()) {
                cerr << "Error: Matrix dimensions mismatch!" << endl;
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            flat_A = NaiveHybrid::flatten(A);
            flat_B = NaiveHybrid::flatten(B);
        }
    }

    // --- BƯỚC 2: BROADCAST KÍCH THƯỚC ---
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Nếu kích thước = 0, dừng ngay để tránh lỗi
    if (N == 0 || M == 0 || K == 0) return {};

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

    int my_count = sendcounts[world_rank];
    int my_rows = my_count / M;
    vector<double> local_A(my_count);

    MPI_Scatterv(flat_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 local_A.data(), my_count, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // --- BƯỚC 5: TÍNH TOÁN CỤC BỘ (HYBRID PART) ---
    
    vector<double> local_C(my_rows * K, 0.0);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < my_rows; ++i) {
        for (int j = 0; j < K; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += local_A[i * M + k] * flat_B[k * K + j]; 
            }
            local_C[i * K + j] = sum;
        }
    }

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
        C = NaiveHybrid::unflatten(flat_C, N, K);
    }

    return C;
}