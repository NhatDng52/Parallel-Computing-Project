#include <vector>

using namespace std;

vector<vector<double>> matrix_mult_strassen(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_strassen_OpenMP(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_strassen_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_strassen_Hybrid(const vector<vector<double>>& A, const vector<vector<double>>& B) ;