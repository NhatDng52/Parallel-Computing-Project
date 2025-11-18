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
        virtual Matrix apply_strassen(const Matrix &A, const Matrix &B) = 0;
        virtual void cleanup() {}
        virtual ~IStrassenOp() = default;
};


//Naive strassen utils
Matrix strassen_recursive(const Matrix &A, const Matrix &B);
Matrix padding(const Matrix &A);
vector<Matrix> separate_mat(const Matrix &A);
Matrix combine_mat(const Matrix &C11, const Matrix &C12, const Matrix &C21, const Matrix &C22);
Matrix mat_add(const Matrix &A, const Matrix &B);
Matrix mat_sub(const Matrix &A, const Matrix &B);

#endif