// do 3 test on 6 algorithm  
// 3 test include : correctness , performance , scalability
// put results in folder named "results"

#include "main.h"

int main () {
    
    // Correctness tests
    test_correctness(matrix_mult_strassen);
    test_performance(matrix_mult_strassen);

    cout << "All tests completed." << endl;
    return 0;
}