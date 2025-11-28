#include "matmul_strassen.h"
#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mp.h"
#include "strassen_utils/strassen_hybrid.h"
#include "strassen_utils/strassen_open_mpi.h"

using namespace std;

vector<vector<double>> matrix_mult_strassen(const vector<vector<double>> &A, const vector<vector<double>> &B) {
      int n = static_cast<int>(A.size());
      int m = static_cast<int>(B[0].size());
      int k = static_cast<int>(A[0].size());

      if (k != static_cast<int>(B.size()))
            throw std::runtime_error("dimension mismatch");

      // padding
      vector<vector<double>> A2 = padding(A);
      vector<vector<double>> B2 = padding(B);

      // recursive Strassen on padded matrices
      vector<vector<double>> C2 = strassen_recursive(A2, B2);

      // remove padding
      vector<vector<double>> C(static_cast<size_t>(n), vector<double>(static_cast<size_t>(m)));
      for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                  C[static_cast<size_t>(i)][static_cast<size_t>(j)] = C2[static_cast<size_t>(i)][static_cast<size_t>(j)];

      return C;
}

vector<vector<double>> matrix_mult_strassen_OpenMP(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    IStrassenOp * strassenOpenMP = new StrassenOpenMP();
    vector<vector<double>> result = strassenOpenMP->apply_strassen(A, B);
    delete strassenOpenMP;
    return result;
}

vector<vector<double>> matrix_mult_strassen_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) {
      IStrassenOp * strassenOpenMPI = new StrassenOpenMPI();
      vector<vector<double>> result = strassenOpenMPI->apply_strassen(A, B);
      delete strassenOpenMPI;
      return result;
}

vector<vector<double>> matrix_mult_strassen_Hybrid(const vector<vector<double>> &A, const vector<vector<double>> &B) {
      IStrassenOp *strassenHybrid = new StrassenHybrid();
      vector<vector<double>> result = strassenHybrid->apply_strassen(A, B);
      delete strassenHybrid;
      return result;
}