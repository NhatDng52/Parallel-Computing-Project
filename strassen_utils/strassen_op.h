#ifndef STRASSEN_OP_H
#define STRASSEN_OP_H

#include<vector>
#include<iostream>
#include <stdexcept>
#include <algorithm>

using namespace std;
using Matrix = vector<vector<double>>;

class IStrassenOp {
    public:
        virtual Matrix apply_strassen(const Matrix &A, const Matrix &B, bool isWorker=false) = 0;
        virtual ~IStrassenOp() = default;
};

#endif // STRASSEN_OP_H