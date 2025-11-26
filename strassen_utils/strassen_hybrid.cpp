#include "strassen_hybrid.h"
#include <algorithm>

Matrix StrassenHybrid::padding(const Matrix &A) {
    int n = static_cast<int>(A.size());
    if (n == 0) return {};
    int m = static_cast<int>(A[0].size());

    int size = 1;
    while (size < std::max(n, m)) size <<= 1;

    if (size == n && size == m) return A;

    Matrix A_pad(static_cast<size_t>(size), vector<double>(static_cast<size_t>(size), 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            A_pad[static_cast<size_t>(i)][static_cast<size_t>(j)] = A[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return A_pad;
}

Matrix StrassenHybrid::mat_add(const Matrix &A, const Matrix &B) {
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(A[0].size());

    Matrix C(static_cast<size_t>(rows), vector<double>(static_cast<size_t>(cols)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                A[static_cast<size_t>(i)][static_cast<size_t>(j)] +
                B[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return C;
}

Matrix StrassenHybrid::mat_sub(const Matrix &A, const Matrix &B) {
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(A[0].size());

    Matrix C(static_cast<size_t>(rows), vector<double>(static_cast<size_t>(cols)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] =
                A[static_cast<size_t>(i)][static_cast<size_t>(j)] -
                B[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return C;
}

Matrix StrassenHybrid::mat_mul_naive(const Matrix &A, const Matrix &B) {
    int rows = static_cast<int>(A.size());
    int cols = static_cast<int>(B[0].size());
    int middle = static_cast<int>(A[0].size());

    Matrix C(static_cast<size_t>(rows), vector<double>(static_cast<size_t>(cols), 0.0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < middle; k++) {
                sum += A[static_cast<size_t>(i)][static_cast<size_t>(k)] *
                       B[static_cast<size_t>(k)][static_cast<size_t>(j)];
            }
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] = sum;
        }
    }

    return C;
}

vector<Matrix> StrassenHybrid::separate_mat(const Matrix &A) {
    int mat_size = static_cast<int>(A.size());
    if (mat_size == 1) {
        return {A};
    }
    int sub_size = mat_size / 2;
    Matrix A11(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
    Matrix A12(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
    Matrix A21(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
    Matrix A22(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            A11[static_cast<size_t>(i)][static_cast<size_t>(j)] = A[static_cast<size_t>(i)][static_cast<size_t>(j)];
            A12[static_cast<size_t>(i)][static_cast<size_t>(j)] = A[static_cast<size_t>(i)][static_cast<size_t>(j + sub_size)];
            A21[static_cast<size_t>(i)][static_cast<size_t>(j)] = A[static_cast<size_t>(i + sub_size)][static_cast<size_t>(j)];
            A22[static_cast<size_t>(i)][static_cast<size_t>(j)] = A[static_cast<size_t>(i + sub_size)][static_cast<size_t>(j + sub_size)];
        }
    }
    return {A11, A12, A21, A22};
}

Matrix StrassenHybrid::combine_mat(const Matrix &C11, const Matrix &C12, const Matrix &C21, const Matrix &C22) {
    int sub_size = static_cast<int>(C11.size());
    int mat_size = sub_size * 2;
    Matrix C(static_cast<size_t>(mat_size), vector<double>(static_cast<size_t>(mat_size)));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] = C11[static_cast<size_t>(i)][static_cast<size_t>(j)];
            C[static_cast<size_t>(i)][static_cast<size_t>(j + sub_size)] = C12[static_cast<size_t>(i)][static_cast<size_t>(j)];
            C[static_cast<size_t>(i + sub_size)][static_cast<size_t>(j)] = C21[static_cast<size_t>(i)][static_cast<size_t>(j)];
            C[static_cast<size_t>(i + sub_size)][static_cast<size_t>(j + sub_size)] = C22[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }
    return C;
}

Matrix StrassenHybrid::implement_strassen(const Matrix &A, const Matrix &B, bool use_mpi) {
    int mat_size = static_cast<int>(A.size());

    // Base case: use naive multiplication for small matrices
    if (mat_size <= THRESHOLD) {
        return mat_mul_naive(A, B);
    }

    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Separate matrices into quadrants
    vector<Matrix> sub_mat_a = separate_mat(A);
    vector<Matrix> sub_mat_b = separate_mat(B);

    Matrix &A11 = sub_mat_a[0];
    Matrix &A12 = sub_mat_a[1];
    Matrix &A21 = sub_mat_a[2];
    Matrix &A22 = sub_mat_a[3];

    Matrix &B11 = sub_mat_b[0];
    Matrix &B12 = sub_mat_b[1];
    Matrix &B21 = sub_mat_b[2];
    Matrix &B22 = sub_mat_b[3];

    // Initialize M matrices
    Matrix M1, M2, M3, M4, M5, M6, M7;

    // Strategy: Use MPI to distribute M1-M7 computations ONLY at top level
    // All recursive calls use OpenMP to avoid nested MPI collectives
    if (use_mpi && size >= 7) {
        // If we have 7 or more processes, assign one M to each of first 7 processes
        // All recursive calls use OpenMP only (use_mpi = false)
        if (rank == 0) {
            M1 = implement_strassen(mat_add(A11, A22), mat_add(B11, B22), false);
        } else if (rank == 1) {
            M2 = implement_strassen(mat_add(A21, A22), B11, false);
        } else if (rank == 2) {
            M3 = implement_strassen(A11, mat_sub(B12, B22), false);
        } else if (rank == 3) {
            M4 = implement_strassen(A22, mat_sub(B21, B11), false);
        } else if (rank == 4) {
            M5 = implement_strassen(mat_add(A11, A12), B22, false);
        } else if (rank == 5) {
            M6 = implement_strassen(mat_sub(A21, A11), mat_add(B11, B12), false);
        } else if (rank == 6) {
            M7 = implement_strassen(mat_sub(A12, A22), mat_add(B21, B22), false);
        }

        // Broadcast results to all processes
        int sub_size = mat_size / 2;
        int total_elements = sub_size * sub_size;

        // Flatten and broadcast M1-M7
        vector<double> M_flat(static_cast<size_t>(total_elements), 0.0);

        // M1
        if (rank == 0) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M1[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        M1.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M1[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M2
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 1) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M2[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 1, MPI_COMM_WORLD);
        M2.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M2[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M3
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 2) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M3[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 2, MPI_COMM_WORLD);
        M3.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M3[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M4
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 3) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M4[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 3, MPI_COMM_WORLD);
        M4.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M4[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M5
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 4) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M5[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 4, MPI_COMM_WORLD);
        M5.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M5[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M6
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 5) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M6[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 5, MPI_COMM_WORLD);
        M6.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M6[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

        // M7
        std::fill(M_flat.begin(), M_flat.end(), 0.0);
        if (rank == 6) {
            for (int i = 0; i < sub_size; i++)
                for (int j = 0; j < sub_size; j++)
                    M_flat[static_cast<size_t>(i * sub_size + j)] = M7[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
        MPI_Bcast(M_flat.data(), total_elements, MPI_DOUBLE, 6, MPI_COMM_WORLD);
        M7.resize(static_cast<size_t>(sub_size), vector<double>(static_cast<size_t>(sub_size)));
        for (int i = 0; i < sub_size; i++)
            for (int j = 0; j < sub_size; j++)
                M7[static_cast<size_t>(i)][static_cast<size_t>(j)] = M_flat[static_cast<size_t>(i * sub_size + j)];

    } else {
        // Fewer processes or recursive call: use OpenMP tasks within each MPI process
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                #pragma omp task shared(M1)
                M1 = implement_strassen(mat_add(A11, A22), mat_add(B11, B22), false);

                #pragma omp task shared(M2)
                M2 = implement_strassen(mat_add(A21, A22), B11, false);

                #pragma omp task shared(M3)
                M3 = implement_strassen(A11, mat_sub(B12, B22), false);

                #pragma omp task shared(M4)
                M4 = implement_strassen(A22, mat_sub(B21, B11), false);

                #pragma omp task shared(M5)
                M5 = implement_strassen(mat_add(A11, A12), B22, false);

                #pragma omp task shared(M6)
                M6 = implement_strassen(mat_sub(A21, A11), mat_add(B11, B12), false);

                #pragma omp task shared(M7)
                M7 = implement_strassen(mat_sub(A12, A22), mat_add(B21, B22), false);

                #pragma omp taskwait
            }
        }
    }

    // Combine results using OpenMP for parallelization
    Matrix C11 = mat_add(mat_sub(mat_add(M1, M4), M5), M7);
    Matrix C12 = mat_add(M3, M5);
    Matrix C21 = mat_add(M2, M4);
    Matrix C22 = mat_add(mat_add(mat_sub(M1, M2), M3), M6);

    Matrix C = combine_mat(C11, C12, C21, C22);

    return C;
}

Matrix StrassenHybrid::apply_strassen(const Matrix &A, const Matrix &B) {
    int n = static_cast<int>(A.size());
    if (n == 0) return {};
    int m = static_cast<int>(B[0].size());
    int k = static_cast<int>(A[0].size());

    if (k != static_cast<int>(B.size())) {
        throw std::runtime_error("Matrix dimension mismatch");
    }

    // Padding
    Matrix A2 = padding(A);
    Matrix B2 = padding(B);

    // Recursive Strassen on padded matrices (use_mpi = true for top level)
    Matrix C2 = implement_strassen(A2, B2, true);

    // Remove padding
    Matrix C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(m)));
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            C[static_cast<size_t>(i)][static_cast<size_t>(j)] = C2[static_cast<size_t>(i)][static_cast<size_t>(j)];
        }
    }

    return C;
}
