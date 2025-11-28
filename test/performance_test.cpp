#include "performance_test.h"

void test_performance(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Some decalrations
        // 3 case unit , medium , large
        double total_time[3] = {0.0, 0.0, 0.0};
        for(int i = 0; i < 3; i++) {
            vector<vector<double>> A;
            vector<vector<double>> B;
            if(i == 0) {
                A = read_matrix_from_csv("./test/test_case/unit_negative_matrix_1x1.csv");
                B = read_matrix_from_csv("./test/test_case/unit_zero_matrix_1x1.csv");
            }
            else if(i == 1) {
                A = read_matrix_from_csv("./test/test_case/medium_positive_matrix_256x256.csv");
                B = read_matrix_from_csv("./test/test_case/medium_zero_matrix_256x256.csv");
            }
            else {
                A = read_matrix_from_csv("./test/test_case/large_zero_matrix_2048x2048.csv");
                B = read_matrix_from_csv("./test/test_case/large_zero_matrix_2048x2048.csv");
            }
        auto start = chrono::high_resolution_clock::now();
        auto result = matrix_mult(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, std::milli> duration = end - start;
        total_time[i] += duration.count();
        }
        cout << "Performance Results (in milliseconds):\n";
        cout << "1. Unit Test (1x1): " << total_time[0] << " ms\n";
        cout << "2. Medium Test (256x256): " << total_time[1] << " ms\n";
        cout << "3. Large Test (2048x2048): " << total_time[2] << " ms\n";
        cout<<"Performance test completed \n";
    
}