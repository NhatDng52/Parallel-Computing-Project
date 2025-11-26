#ifndef STRASSEN_OPEN_MPI_H
#define STRASSEN_OPEN_MPI_H

#include "strassen_op.h"

class StrassenOpenMPI : public IStrassenOp {
    public:
        Matrix apply_strassen(const Matrix &A, const Matrix &B) override;
        void cleanup() override;
};

#endif
