#pragma once
#include "naive_utils.h"
#include "mpi.h"

class NaiveMpi : public NaiveUtils {
    public:
        static Matrix matrix_mult(const Matrix &A, const Matrix &B);
        virtual ~NaiveMpi() = default;
};