// performance : Check how efficiently the algorithm runs as input size increases
/***
 * Several test to measure execution time and memory usage 
 * ***/


#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include"../utils.h"
using namespace std;

void test_performance(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&) );
void test_performance_1(vector<vector<double>> (*matrix_mult)(const vector<vector<double>>&, const vector<vector<double>>&));
void test_performance_2(vector<vector<double>> (*matrix_mult)(const vector<vector<double>>&, const vector<vector<double>>&));
void test_performance_3(vector<vector<double>> (*matrix_mult)(const vector<vector<double>>&, const vector<vector<double>>&));
