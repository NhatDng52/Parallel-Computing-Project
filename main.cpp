// do 3 test on 6 algorithm  
// 3 test include : correctness , performance , scalability
// put results in folder named "results"

#include "main.h"

int main () {
    
    // Correctness tests
    test_correctness(matrix_mult_naive);

    cout << "All tests completed." << endl;
    return 0;
}