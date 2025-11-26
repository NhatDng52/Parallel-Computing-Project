#include "correctness_test.h"
#include <iostream>
using namespace std;
void test_correctness(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    test_1(matrix_mult);
    test_2(matrix_mult);
    test_3(matrix_mult);
    cout<<"Correctness tests completed \n";
}

void test_1(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // test CSV files are located in test/test_case relative to project root
    vector<vector<double>> A = read_matrix_from_csv("./test/test_case/unit_positive_matrix_1x1.csv");
    vector<vector<double>> B = read_matrix_from_csv("./test/test_case/unit_zero_matrix_1x1.csv");
    vector<vector<double>> C = read_matrix_from_csv("./test/test_case/unit_negative_matrix_1x1.csv");
    try
    {
        vector<vector<double>> AA = matrix_mult(A, A);
        vector<vector<double>> AB = matrix_mult(A, B);
        vector<vector<double>> AC = matrix_mult(A, C);
        
        if(!compare_matrices(AA, A)) {
            cout << "Test 1 Failed: A * A != A" << endl;
        }
        if(!compare_matrices(AB, B)) {
            cout << "Test 1 Failed: A * B != B" << endl;
        }
        if(!compare_matrices(AC, C)) {
            cout << "Test 1 Failed: A * C != C" << endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    cout<<"Test 1 completed \n";
}
void test_2(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    // Small non-square matrix test
    vector<vector<double>> A = read_matrix_from_csv("./test/test_case/matrix_3x4.csv");
    vector<vector<double>> B = read_matrix_from_csv("./test/test_case/matrix_4x2.csv");
    vector<vector<double>> expected = read_matrix_from_csv("./test/test_case/result_correctness_2.csv");
    try
    {
        vector<vector<double>> result = matrix_mult(A, B);
        
        if(!compare_matrices(result, expected)) {
            cout << "Test 2 Failed: A * B != expected" << endl;
            cout << "Expected:" << endl;
            print_matrix(expected);
            cout << "Got:" << endl;
            print_matrix(result);
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    cout<<"Test 2 completed \n";
}   
void test_3(vector<std::vector<double>> (*matrix_mult)(const vector<std::vector<double>>&, const vector<std::vector<double>>&)) {
    //3B 
    vector<vector<double>> A = read_matrix_from_csv("./test/test_case/medium_random_matrix_A_256x256.csv");
    vector<vector<double>> B = read_matrix_from_csv("./test/test_case/medium_random_matrix_B_256x256.csv");
    vector<vector<double>> C = read_matrix_from_csv("./test/test_case/medium_random_matrix_C_256x256.csv");
    try
    {
        vector<vector<double>> AB = matrix_mult(A, B);
        vector<vector<double>> BC = matrix_mult(B, C);
        vector<vector<double>> ABC_1 = matrix_mult(AB, C);
        vector<vector<double>> ABC_2 = matrix_mult(A, BC);
        

        if(!compare_matrices(ABC_1, ABC_2)) {
            write_matrix_to_csv("./test/results/3B_ABC_1.csv", ABC_1);
            write_matrix_to_csv("./test/results/3B_ABC_2.csv", ABC_2);
            cout << "Test 3 Failed: (A * B) * C != A * (B * C)" << endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //3E 
    vector<vector<double>> I = read_matrix_from_csv("./test/test_case/medium_identity_matrix_256x256.csv");
    vector<vector<double>> AI = matrix_mult(A, I);
    vector<vector<double>> IA = matrix_mult(I, A);
        
    try
    {
        if(!compare_matrices(AI, A)) {
            write_matrix_to_csv("./test/results/3E_AI.csv", AI);
            cout << "Test 3 Failed: A * I != A" << endl;
        }
        if(!compare_matrices(IA, A)) {
            write_matrix_to_csv("./test/results/3E_IA.csv", IA);
            cout << "Test 3 Failed: I * A != A" << endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //3F 
    vector<vector<double>> zero = read_matrix_from_csv("./test/test_case/medium_zero_matrix_256x256.csv");
    vector<vector<double>> Azero = matrix_mult(A, zero);
    vector<vector<double>> zeroA = matrix_mult(zero, A);
    try
    {
        if(!compare_matrices(Azero, zero)) {
            write_matrix_to_csv("./test/results/3F_Azero.csv", Azero);
            cout << "Test 3 Failed: A * 0 != 0" << endl;
        }
        if(!compare_matrices(zeroA, zero)) {
            write_matrix_to_csv("./test/results/3F_zeroA.csv", zeroA);
            cout << "Test 3 Failed: 0 * A != 0" << endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }

    //3G 
    vector<vector<double>> BA = matrix_mult(B, A);
    vector<vector<double>> AB = matrix_mult(A, B);
    try
    {
        if(compare_matrices(AB, BA)) {
            write_matrix_to_csv("./test/results/3G_AB.csv", AB);
            write_matrix_to_csv("./test/results/3G_BA.csv", BA);
            cout << "Test 3 Failed: A * B == B * A" << endl;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }


    cout<<"Test 3 completed \n";
}
