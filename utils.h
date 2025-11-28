

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

vector<vector<double>> read_matrix_from_csv(const std::string &filepath);

bool compare_matrices(const vector<vector<double>> &A, const vector<vector<double>> &B);

void print_matrix(const std::vector<std::vector<double>> &m);

void write_matrix_to_csv(const std::string &filepath, const std::vector<std::vector<double>> &matrix);