#include "strassen_op.h"
#include <mpi.h>
#include "utils.h"

class BetterStrassenOpenMPI : public IStrassenOp {
    public:
        Matrix apply_strassen(const Matrix &A, const Matrix &B) override {
        }
}