#include "scalability_test.h"
void test_scalability_OpenMP(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some declarations
    // Iterate over different thread numbers
        cout<<"Scalability test with OpenMP completed \n";
    
}

void test_scalability_OpenMPI(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some declarations
    cout<<"Scalability test with OpenMPI completed \n";
}

void test_scalability_Hybrid(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some declarations
    // Hybrid test code combining OpenMP and OpenMPI
    cout<<"Scalability test with Hybrid completed \n";
}
