#include <vector>

using namespace std;

vector<vector<double>> matrix_mult_naive(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_naive_OpenMP(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_naive_OpenMPI(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
vector<vector<double>> matrix_mult_naive_Hybrid(const vector<vector<double>>& A, const vector<vector<double>>& B) ;
