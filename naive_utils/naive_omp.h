#include "naive_utils.h"
#include <omp.h>

class NaiveOmp : public NaiveUtils {
    public:
        static Matrix matrix_mult(const Matrix &A, const Matrix &B);
        virtual ~NaiveOmp() = default;
};