// performance : Check how efficiently the algorithm runs as input size increases
/***
 * Sequence of tests with different matrix sizes and input distributions, aim to render the big O complexity
 * ***/


#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
using namespace std;

void test_performance(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&) );
