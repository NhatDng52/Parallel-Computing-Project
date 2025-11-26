#include "strassen_utils/strassen_op.h"
#include "strassen_utils/strassen_openmp.h"
#include "utils.h"
#include "correctness_test.h"
#include "performance_test.h"
#include "scalability_test.h"
#include <iostream>
#include <iomanip>
#include <omp.h>

using namespace std;

vector<vector<double>> strassen_openmp_wrapper(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    static StrassenOpenMP op;
    return op.apply_strassen(A, B);
}

void run_correctness_tests() {
    cout << "\n========================================" << endl;
    cout << "  CORRECTNESS TESTS - Strassen OpenMP  " << endl;
    cout << "========================================\n" << endl;
    
    test_correctness(strassen_openmp_wrapper);
    
    cout << "\n✓ All correctness tests passed!\n" << endl;
}

void run_performance_tests() {
    cout << "\n========================================" << endl;
    cout << "  PERFORMANCE TESTS - Strassen OpenMP  " << endl;
    cout << "========================================\n" << endl;
    
    test_performance(strassen_openmp_wrapper);
    
    cout << "\n✓ Performance tests completed!\n" << endl;
}

// Scalability tests for Strassen OpenMP
void run_scalability_tests() {
    cout << "\n========================================" << endl;
    cout << "  SCALABILITY TESTS - Strassen OpenMP  " << endl;
    cout << "========================================\n" << endl;
    
    test_scalability_OpenMP(strassen_openmp_wrapper);
    
    cout << "\n✓ Scalability tests completed!\n" << endl;
}

// Main test runner
int main(int argc, char* argv[]) {
    cout << "\n╔════════════════════════════════════════╗" << endl;
    cout << "║  Strassen Algorithm - OpenMP Testing  ║" << endl;
    cout << "╚════════════════════════════════════════╝\n" << endl;
    
    // Show OpenMP configuration
    #pragma omp parallel
    {
        #pragma omp single
        {
            cout << "OpenMP Configuration:" << endl;
            cout << "  Max threads available: " << omp_get_max_threads() << endl;
            cout << "  Number of processors: " << omp_get_num_procs() << endl;
        }
    }
    
    // Parse command line arguments
    bool run_all = (argc == 1);
    bool run_correct = false;
    bool run_perf = false;
    bool run_scale = false;
    
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--correctness" || arg == "-c") run_correct = true;
        else if (arg == "--performance" || arg == "-p") run_perf = true;
        else if (arg == "--scalability" || arg == "-s") run_scale = true;
        else if (arg == "--all" || arg == "-a") run_all = true;
        else if (arg == "--help" || arg == "-h") {
            cout << "\nUsage: " << argv[0] << " [options]" << endl;
            cout << "Options:" << endl;
            cout << "  -c, --correctness    Run correctness tests" << endl;
            cout << "  -p, --performance    Run performance tests" << endl;
            cout << "  -s, --scalability    Run scalability tests" << endl;
            cout << "  -a, --all            Run all tests (default)" << endl;
            cout << "  -h, --help           Show this help message" << endl;
            return 0;
        }
    }
    
    // Run selected tests
    try {
        if (run_all || run_correct) {
            run_correctness_tests();
        }
        
        if (run_all || run_perf) {
            run_performance_tests();
        }
        
        if (run_all || run_scale) {
            run_scalability_tests();
        }
        
        cout << "\n╔════════════════════════════════════════╗" << endl;
        cout << "║     All Tests Completed Successfully  ║" << endl;
        cout << "╚════════════════════════════════════════╝\n" << endl;
        
    } catch (const exception& e) {
        cerr << "\n❌ ERROR: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
