#include "strassen_open_mp.h"
#include "matmul_algorithms/matmul_naive.h"

// OpenMP-parallelized utility functions
namespace {
    Matrix divide_mat_parallel(const Matrix &A) {
        int mat_size = A.size();
        if (mat_size == 1) return A;
        
        int sub_size = mat_size / 2;
        Matrix A11(sub_size, vector<double>(sub_size));
        Matrix A12(sub_size, vector<double>(sub_size));
        Matrix A21(sub_size, vector<double>(sub_size));
        Matrix A22(sub_size, vector<double>(sub_size));

        #pragma omp parallel for collapse(2) if(sub_size > 64)
        for (int i = 0; i < sub_size; i++) {
            for (int j = 0; j < sub_size; j++) {
                A11[i][j] = A[i][j];
                A12[i][j] = A[i][j + sub_size];
                A21[i][j] = A[i + sub_size][j];
                A22[i][j] = A[i + sub_size][j + sub_size];
            }
        }
        return A11; // Note: This should return vector, will be fixed
    }

    Matrix mat_add_parallel(const Matrix &A, const Matrix &B) {
        int rows = A.size();
        int cols = A[0].size();
        Matrix C(rows, vector<double>(cols));

        #pragma omp parallel for collapse(2) if(rows * cols > 1024)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
        return C;
    }

    Matrix mat_sub_parallel(const Matrix &A, const Matrix &B) {
        int rows = A.size();
        int cols = A[0].size();
        Matrix C(rows, vector<double>(cols));

        #pragma omp parallel for collapse(2) if(rows * cols > 1024)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    Matrix combine_mat_parallel(const Matrix &C11, const Matrix &C12, 
                                 const Matrix &C21, const Matrix &C22) {
        int sub_size = C11.size();
        int mat_size = sub_size * 2;
        Matrix C(mat_size, vector<double>(mat_size));

        #pragma omp parallel for collapse(2) if(sub_size > 64)
        for (int i = 0; i < sub_size; i++) {
            for (int j = 0; j < sub_size; j++) {
                C[i][j] = C11[i][j];
                C[i][j + sub_size] = C12[i][j];
                C[i + sub_size][j] = C21[i][j];
                C[i + sub_size][j + sub_size] = C22[i][j];
            }
        }
        return C;
    }
}

// Private methods
vector<Matrix> StrassenOpenMP::divide_mat(const Matrix &A) {
    return separate_mat(A);  // Use existing utility function
}

Matrix StrassenOpenMP::padding(const Matrix &A) {
    return ::padding(A);  // Use existing utility function
}

Matrix StrassenOpenMP::mat_add(const Matrix &A, const Matrix &B) {
    return mat_add_parallel(A, B);
}

Matrix StrassenOpenMP::mat_sub(const Matrix &A, const Matrix &B) {
    return mat_sub_parallel(A, B);
}

Matrix StrassenOpenMP::mat_mul_naive(const Matrix &A, const Matrix &B) {
    return matrix_mult_naive(A, B);  // Use existing naive implementation
}

Matrix StrassenOpenMP::implement_strassen(const Matrix &A, const Matrix &B) {
    int mat_size = A.size();

    if (mat_size <= THRESHOLD) {
        return mat_mul_naive(A, B);
    }

    vector<Matrix> sub_mat_a = divide_mat(A);
    vector<Matrix> sub_mat_b = divide_mat(B);

    const Matrix &A11 = sub_mat_a[0], &A12 = sub_mat_a[1];
    const Matrix &A21 = sub_mat_a[2], &A22 = sub_mat_a[3];
    const Matrix &B11 = sub_mat_b[0], &B12 = sub_mat_b[1];
    const Matrix &B21 = sub_mat_b[2], &B22 = sub_mat_b[3];

    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task shared(M1)
            M1 = implement_strassen(mat_add(A11, A22), mat_add(B11, B22));

            #pragma omp task shared(M2)
            M2 = implement_strassen(mat_add(A21, A22), B11);

            #pragma omp task shared(M3)
            M3 = implement_strassen(A11, mat_sub(B12, B22));

            #pragma omp task shared(M4)
            M4 = implement_strassen(A22, mat_sub(B21, B11));

            #pragma omp task shared(M5)
            M5 = implement_strassen(mat_add(A11, A12), B22);

            #pragma omp task shared(M6)
            M6 = implement_strassen(mat_sub(A21, A11), mat_add(B11, B12));

            #pragma omp task shared(M7)
            M7 = implement_strassen(mat_sub(A12, A22), mat_add(B21, B22));

            #pragma omp taskwait
        }
    }

    Matrix C11 = mat_add(mat_sub(mat_add(M1, M4), M5), M7);
    Matrix C12 = mat_add(M3, M5);
    Matrix C21 = mat_add(M2, M4);
    Matrix C22 = mat_add(mat_add(mat_sub(M1, M2), M3), M6);

    return combine_mat_parallel(C11, C12, C21, C22);
}

// Public method
Matrix StrassenOpenMP::apply_strassen(const Matrix &A, const Matrix &B) {
    int rows = A.size();
    int cols = B[0].size();
    int middleCols = A[0].size();
    int middleRows = B.size();

    if (A.empty() || B.empty() || middleCols != middleRows) {
        throw runtime_error("Matrix dimension mismatch in StrassenOpenMP::apply_strassen");
    }

    Matrix A_pad = padding(A);
    Matrix B_pad = padding(B);
    Matrix C_pad = implement_strassen(A_pad, B_pad);

    // Extract result with original dimensions
    Matrix C(rows, vector<double>(cols, 0.0));
    
    #pragma omp parallel for collapse(2) if(rows * cols > 1024)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = C_pad[i][j];
        }
    }

    return C;
}