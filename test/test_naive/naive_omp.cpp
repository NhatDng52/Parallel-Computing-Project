#include "matmul_algorithms/matmul_naive.h"
#include "test/correctness_test.h"
#include "test/performance_test.h"
#include "test/scalability_test.h"

using namespace std;

int main() {
    try {
        test_correctness(matrix_mult_naive_OpenMP);
        test_performance(matrix_mult_naive_OpenMP);
        test_scalability_OpenMP(matrix_mult_naive_OpenMP);
    } catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }

    return 0;
}