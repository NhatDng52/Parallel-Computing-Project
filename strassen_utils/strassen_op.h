#include<vector>
#include<iostream>

using namespace std;
using Matrix = vector<vector<double>>;

class IStrassenOp {
    public:
        virtual Matrix apply_strassen(const Matrix &A, const Matrix &B) = 0;
};