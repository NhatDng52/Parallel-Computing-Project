#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib> 
#include <ctime>   
#include <algorithm>
#include <cmath>  
#include "../strassen_utils/strassen_open_mp.cpp"
#include "../utils.cpp"

using namespace std;

/**
 * ================================================================
 * Phần 2: Các hàm "Ground Truth" (Naive) để tạo kết quả mong đợi
 * ================================================================
 */

/**
 * @brief Tạo ma trận N x N với giá trị ngẫu nhiên (0-9)
 */
Matrix createRandomMatrix(int N) {
    Matrix mat(N, vector<double>(N));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            mat[i][j] = rand() % 10;
        }
    }
    return mat;
}

/**
 * @brief Tính A + B (phiên bản chuẩn)
 */
Matrix ground_truth_add(const Matrix& A, const Matrix& B) {
    int N = A.size();
    Matrix C(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

/**
 * @brief Tính A - B (phiên bản chuẩn)
 */
Matrix ground_truth_sub(const Matrix& A, const Matrix& B) {
    int N = A.size();
    Matrix C(N, vector<double>(N));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

/**
 * @brief Thực hiện phép nhân ma trận A * B một cách chuẩn xác (ground truth).
 * Hàm này hỗ trợ ma trận không vuông: A(n x m) * B(m x k) -> C(n x k).
 * * @param A Ma trận bên trái (n x m).
 * @param B Ma trận bên phải (m x k).
 * @return Ma trận kết quả C (n x k).
 */
Matrix ground_truth_mul(const Matrix& A, const Matrix& B) {
    // A(n x m) x B(m x k) -> C(n x k)
    
    // 1. Kiểm tra điều kiện nhân ma trận: số cột của A phải bằng số hàng của B.
    if (A.empty() || B.empty() || A[0].size() != B.size()) {
        // Hoặc trả về một ma trận rỗng hoặc ném ngoại lệ
        throw std::runtime_error("Lỗi kích thước ma trận: Số cột của A không bằng số hàng của B.");
    }

    // 2. Xác định kích thước ma trận kết quả C
    int rows = A.size();       // n
    int cols = B[0].size();    // k
    int middleDim = A[0].size(); // m (kích thước trung gian)
    
    // 3. Khởi tạo ma trận kết quả C(n x k) với tất cả phần tử là 0.0
    Matrix C(rows, std::vector<double>(cols, 0.0));

    // 4. Thực hiện phép nhân ma trận cổ điển (i, j, k)
    for (int i = 0; i < rows; ++i) { // Duyệt qua hàng của A (rows = n)
        for (int j = 0; j < cols; ++j) { // Duyệt qua cột của B (cols = k)
            // Tính C[i][j]
            for (int k = 0; k < middleDim; ++k) { // Duyệt qua kích thước trung gian (middleDim = m)
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    // const int N = 400; 

    // cout << "Đang tạo test case cho ma trận " << N << "x" << N << "...\n";

    IStrassenOp* op = new StrassenOpenMPI();

    // Matrix A = createRandomMatrix(N);
    // Matrix B = createRandomMatrix(N);

    // cout << "\nTesting Mul Strassen...\n";
    // auto C_strassen_op = op->apply_strassen(A, B);
    // auto C_strassen_expected = ground_truth_mul(A, B);

    // cout << "Strassen Mul OK? " 
    //     << (compare_matrices(C_strassen_op, C_strassen_expected) ? "YES" : "NO") 
    //     << "\n";

    // Test 02 Test Correctness with Big Matrix
    Matrix A = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
    Matrix B = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");
    
    // Matrix A = read_matrix_from_csv("./test/test_case/matrix_3x4.csv");
    // Matrix B = read_matrix_from_csv("./test/test_case/matrix_4x2.csv");

    cout << "\nTesting Mul Strassen...\n";
    auto C_strassen_op = op->apply_strassen(A, B);
    auto C_strassen_expected = ground_truth_mul(A, B);

    print_matrix(C_strassen_op);
    cout << "-----------------------------------------\n";
    print_matrix(C_strassen_expected);

    cout << "Strassen Mul OK? " 
        << (compare_matrices(C_strassen_op, C_strassen_expected) ? "YES" : "NO") 
        << "\n";

    return 0;
}

