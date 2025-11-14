// correctness : Check if the algorithm produces the correct output for a given input
/***
 * Test 1: Small input test with zeroes, negative numbers, and positive numbers 
 * Test 2: Small non-square matrix test
 * Test 3(longest): Mathematical properties test (associativity, distributivity, etc.)
 *     Include : 
 *        * (A + B) + C = A + (B + C)  // this is not tested in this assignment, since our job just matrix multiplication
 *        * (A * B) * C = A * (B * C)
 *        * A * (B + C) = A * B + A * C //this is same as above
 *        * A + 0 = A                   // this is same as above
 *        * A * I = I * A = A
 *        * A * 0 = 0 * A = 0
 *        * A * B not= B * A
 * ****/

#include <iostream>
#include <vector>
#include <numeric>
#include"../utils.h"
using namespace std;

void test_1(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_2(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_3(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_4(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;
void test_5(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;

void test_correctness(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) ;