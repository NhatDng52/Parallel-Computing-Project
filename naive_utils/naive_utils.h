#ifndef NAIVE_UTILS_H
#define NAIVE_UTILS_H

#include <vector>
#include <iostream>

using namespace std;

using Matrix = vector<vector<double>>;

class NaiveUtils {
    public:
        static Matrix matrix_mult(const Matrix &A, const Matrix &B);
        virtual ~NaiveUtils() = default;

};

#endif // NAIVE_UTILS_H