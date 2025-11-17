#include "strassen_op.h"
#include "matmul_naive.h"
#include <algorithm>

Matrix padding(const Matrix &A) {
    int n = A.size();
    if (n == 0) return {};
    int m = A[0].size();

    int size = 1;
    while (size < max(n, m)) size <<= 1;

    if (size == n && size == m) return A;

    Matrix A_pad(size, std::vector<double>(size, 0.0));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            A_pad[i][j] = A[i][j];

    return A_pad;
}

Matrix mat_add(const Matrix &A, const Matrix &B) {
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

Matrix mat_sub(const Matrix &A, const Matrix &B) {
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

vector<Matrix> separate_mat(const Matrix &A) {
    int mat_size = A.size();
    if (mat_size == 1) {
        return {A};
    }
    int sub_size = mat_size / 2;
    Matrix A11(sub_size, std::vector<double>(sub_size));
    Matrix A12(sub_size, std::vector<double>(sub_size));
    Matrix A21(sub_size, std::vector<double>(sub_size));
    Matrix A22(sub_size, std::vector<double>(sub_size));

    for (int i = 0; i < sub_size; i++) {
        for (int j = 0; j < sub_size; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + sub_size];
            A21[i][j] = A[i + sub_size][j];
            A22[i][j] = A[i + sub_size][j + sub_size];
        }
    }
    return {A11, A12, A21, A22};
}


Matrix combine_mat(const Matrix &C11, const Matrix &C12, const Matrix &C21, const Matrix &C22) {
    int sub_size = C11.size();
    int mat_size = sub_size * 2;
    Matrix C(mat_size, std::vector<double>(mat_size));
    
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

Matrix strassen_recursive(const Matrix &A, const Matrix &B) {
    const int THRESHOLD = 32; // Ngưỡng chuyển sang nhân thuần
    int mat_size = A.size();

    // SỬA: Điều kiện dừng dùng THRESHOLD và mat_mul_naive
    if (mat_size <= THRESHOLD) {
        return matrix_mult_naive(A, B);
    } 

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

    // M1 to M7 calculations (recursive calls)
    Matrix M1 = strassen_recursive(mat_add(A11, A22), mat_add(B11, B22));
    Matrix M2 = strassen_recursive(mat_add(A21, A22), B11);
    Matrix M3 = strassen_recursive(A11, mat_sub(B12, B22));
    Matrix M4 = strassen_recursive(A22, mat_sub(B21, B11));
    Matrix M5 = strassen_recursive(mat_add(A11, A12), B22);
    Matrix M6 = strassen_recursive(mat_sub(A21, A11), mat_add(B11, B12));
    Matrix M7 = strassen_recursive(mat_sub(A12, A22), mat_add(B21, B22));

    // C11, C12, C21, C22 calculations
    Matrix C11 = mat_add(mat_sub(mat_add(M1, M4), M5), M7);
    Matrix C12 = mat_add(M3, M5);
    Matrix C21 = mat_add(M2, M4);
    // SỬA LỖI LOGIC C22: C22 = M1 - M2 + M3 + M6
    Matrix C22 = mat_add(mat_add(mat_sub(M1, M2), M3), M6);

    // Combine sub-matrices into result matrix C
    Matrix C = combine_mat(C11, C12, C21, C22);

    return C;
}



