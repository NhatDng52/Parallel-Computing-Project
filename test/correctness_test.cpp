#include "correctness_test.h"

void test_correctness(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    test_1(matrix_mult);
    test_2(matrix_mult);
    test_3(matrix_mult);
    test_4(matrix_mult);
    test_5(matrix_mult);
    cout<<"Correctness tests completed \n";
}

void test_1(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Small input test with zeroes, negative numbers, and positive numbers 
    cout<<"Test 1 completed \n";
}
void test_2(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Large input test with zeroes, negative numbers, and positive numbers
    cout<<"Test 2 completed \n";
}   
void test_3(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Small non-square matrix test
    cout<<"Test 3 completed \n";
}
void test_4(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Large non-square matrix test
    cout<<"Test 4 completed \n";
}
void test_5(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Mathematical properties test (associativity, distributivity, etc.)
    
    cout<<"Test 5 completed \n";
}