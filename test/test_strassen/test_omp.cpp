#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_open_mp.h"
#include "utils.h"
#include "correctness_test.h"
#include "performance_test.h"
#include "scalability_test.h"
#include <iostream>

using namespace std;

vector<vector<double>> strassen_openmp_wrapper(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    static StrassenOpenMP op;
    return op.apply_strassen(A, B);
}

int main() {
    
    try {
        test_correctness(strassen_openmp_wrapper);
        test_performance(strassen_openmp_wrapper);
        test_scalability_OpenMP(strassen_openmp_wrapper);
        
    } catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
