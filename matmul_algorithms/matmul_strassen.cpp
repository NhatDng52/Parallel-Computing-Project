#include "matmul_strassen.h"
#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mp.cpp"
#include "strassen_utils/strassen_open_mpi.cpp"

using namespace std;
vector<vector<double>> matrix_mult_strassen(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int k = A[0].size();


    if(k != B.size()) throw "dimension mismatch";


    // padding  
    vector<vector<double>> A2 = padding(A);
    vector<vector<double>> B2 = padding(B);


    // recursive Strassen on padded matrices
    vector<vector<double>> C2 = strassen_recursive(A2, B2);


    // remove padding
    vector<vector<double>> C(n, vector<double>(m));
    for(int i = 0; i < n; i++)
    for(int j = 0; j < m; j++)
    C[i][j] = C2[i][j];


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