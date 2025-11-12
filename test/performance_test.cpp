#include "performance_test.h"

void test_performance(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some decalrations
        // Iterate over different matrix sizes
        auto start = chrono::high_resolution_clock::now();
        // Body code + output results each case 
        auto end = chrono::high_resolution_clock::now();
        
        cout<<"Performance test completed \n";
    
}