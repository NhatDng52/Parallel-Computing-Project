#include "matmul_strassen.h"
#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mp.cpp"

using namespace std;
vector<vector<double>> matrix_mult_strassen(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    /*
        Implement code 
    */
   return vector<vector<double>>{};
}

vector<vector<double>> matrix_mult_strassen_OpenMP(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    IStrassenOp * strassenOpenMP = new StrassenOpenMP();
    vector<vector<double>> result = strassenOpenMP->apply_strassen(A, B);
    delete strassenOpenMP;
    return result;
}

vector<vector<double>> matrix_mult_strassen_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    /*
        Implement code 
    */
   return vector<vector<double>>{};
}