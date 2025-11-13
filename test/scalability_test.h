// scalability : Check how well the algorithm scales with increasing computing resources
/***
 * Sequence of tests with different size
***/
// Because We use OpenMP and OpenMPI, so we need 3 specific tests to adjust to these libraries
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <mpi.h>
using namespace std;

void test_scalability_OpenMP(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_scalability_OpenMPI(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_scalability_Hybrid(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;