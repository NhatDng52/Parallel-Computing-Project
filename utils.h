


#include<vector>
#include<fstream>
#include<sstream>
#include<string>
#include<stdexcept>
#include<iostream>
using namespace std;

vector<vector<double>> read_matrix_from_csv(const std::string& filepath);

bool compare_matrices(const vector<vector<double>>& A, const vector<vector<double>>& B);

void print_matrix(const std::vector<std::vector<double>>& m);