
#include "utils.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <iostream>
#include <iomanip>

const double EPSILON = 1e-9;

std::vector<std::vector<double>> read_matrix_from_csv(const std::string& filepath) {
    
    std::vector<std::vector<double>> matrix;
    
    
    std::ifstream file(filepath);
    
    
    if (!file.is_open()) {
        throw std::runtime_error("can not open file : " + filepath);
    }
    
    std::string line;
    
    while (std::getline(file, line)) {
        
        std::stringstream ss(line);
        std::string cell;
        
        
        std::vector<double> row;
        
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (const std::exception& e) {
                std::cerr << "convert error " << cell 
                          << "' in file" << filepath << std::endl;
                row.push_back(0.0); 
            }
        }
        
        if (!row.empty()) {
            matrix.push_back(row);
        }
    }
    
    file.close();
    return matrix;
}
bool compare_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    
    if (A.size() != B.size()) {
        return false;
    }
    
    for (size_t i = 0; i < A.size(); ++i) {
        if (A[i].size() != B[i].size()) {
            return false;
        }
        for (size_t j = 0; j < A[i].size(); ++j) {
            if (abs(A[i][j] - B[i][j]) > EPSILON) {
                cout << "Mismatch at (" << i << ", " << j << "): "
                     << A[i][j] << " vs " << B[i][j] << "\n";
                
                // print_matrix(A);
                // print_matrix(B);

                return false;
            }
        }
    }
    
    return true;
}

void print_matrix(const std::vector<std::vector<double>>& m) {
    // for (const auto& row : m) {
    //     for (size_t j = 0; j < row.size(); ++j) {
    //         if (j) std::cout << ' ';
    //         std::cout << row[j];
    //     }
    //     std::cout << '\n';
    // }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cout << setw(10) << fixed << setprecision(2) << m[i][j] << " ";
        }
        cout << "\n";
    }
    cout << "----------------------\n";
}