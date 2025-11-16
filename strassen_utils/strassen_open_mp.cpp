#include "strassen_op.h"
#include <omp.h>

class StrassenOpenMPI : public IStrassenOp {
    private:
        const int THRESHOLD = 1;

        vector<Matrix> divide_mat(const Matrix &A) {
            int mat_size = A.size();

            if (mat_size == 1) {
                return {A};
            } else {
                int sub_size = mat_size / 2;
                Matrix A11(sub_size, vector<double>(sub_size));
                Matrix A12(sub_size, vector<double>(sub_size));
                Matrix A21(sub_size, vector<double>(sub_size));
                Matrix A22(sub_size, vector<double>(sub_size));

                #pragma omp parallel for collapse(2)
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

        Matrix padding(const Matrix &A) {
            int n = A.size();          
            if (n == 0) return {};
            int m = A[0].size();

            int size = 1;
            while (size < std::max(n, m)) size *= 2;

            if (size == n && size == m) return A;

            Matrix A_pad(size, vector<double>(size, 0.0));
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < m; ++j)
                    A_pad[i][j] = A[i][j];

            return A_pad;
        }

        Matrix implement_strassen(const Matrix &A, const Matrix &B) {
            int mat_size = A.size();

            if (mat_size <= THRESHOLD) {
                return mat_mul_naive(A, B);
            } else {
                int n = A.size();

                vector<Matrix> sub_mat_a = divide_mat(A);
                vector<Matrix> sub_mat_b = divide_mat(B);

                Matrix A11 = sub_mat_a[0], A12 = sub_mat_a[1], A21 = sub_mat_a[2], A22 = sub_mat_a[3];
                Matrix B11 = sub_mat_b[0], B12 = sub_mat_b[1], B21 = sub_mat_b[2], B22 = sub_mat_b[3];

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

                Matrix C(n, vector<double>(n));
                int k = n / 2;

                #pragma omp parallel for collapse(2)
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        C[i][j] = C11[i][j];
                        C[i][j + k] = C12[i][j];
                        C[i + k][j] = C21[i][j];
                        C[i + k][j + k] = C22[i][j]; 
                    }
                }

                return C;
            }
        }

        Matrix mat_add(const Matrix &A, const Matrix &B) {
            // Pre condition: Check matrix size
            if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_add strassen_op_omp");
            }
            
            int rows = A.size();
            int cols = A[0].size();

            Matrix C(rows, vector<double>(cols));

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    C[i][j] = A[i][j] + B[i][j];
                }
            }

            return C;
        }

        Matrix mat_sub(const Matrix &A, const Matrix &B) {
            
            if (A.empty() || B.empty() || A.size() != B.size() || A[0].size() != B[0].size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_sub strassen_op_omp");
            }
            
            int rows = A.size();
            int cols = A[0].size();

            Matrix C(rows, vector<double>(cols));

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    C[i][j] = A[i][j] - B[i][j];
                }
            }

            return C;
        }
        
        Matrix mat_mul_naive(const Matrix &A, const Matrix &B) {
            
            // A(n x m) x B(m x k) -> C(n, k)

            if (A.empty() || B.empty() || A[0].size() != B.size()) {
                throw runtime_error("Lỗi ma trận, trong phép toán mat_mul strassen_op_omp");
            }

            int rows = A.size();
            int cols = B[0].size();
            int middleCos = A[0].size();

            Matrix C(rows, vector<double> (cols));

            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    for (int k = 0; k < middleCos; k++) {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }

            return C;
        }

    public:

        Matrix apply_strassen(const Matrix &A, const Matrix &B) {
            int rows = A.size();
            int cols = B[0].size();
            int middleCos = A[0].size();
            int middle_B_size = B.size();

            if (A.empty() || B.empty() || middleCos != middle_B_size) {
                throw runtime_error("Lỗi ma trận, trong phép toán apply_strassen strassen_op_omp");
            }
            Matrix A_pad = padding(A);
            Matrix B_pad = padding(B);

            Matrix C_pad = implement_strassen(A_pad, B_pad);

            Matrix C(rows, std::vector<double>(cols, 0.0));
            
            #pragma omp parallel for collapse(2)
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    C[i][j] = C_pad[i][j];

            return C;
            
        }
};