#pragma once
#include "naive_utils.h"
#include "mpi.h"
#include <omp.h>

class NaiveHybrid : public NaiveUtils {
    private:
        NaiveHybrid() = default;
        static vector<double> flatten(const Matrix& mat);
        static Matrix unflatten(const vector<double>& flat, int rows, int cols);
    public:
        static Matrix matrix_mult(const Matrix &A, const Matrix &B);
        virtual ~NaiveHybrid() = default;
};