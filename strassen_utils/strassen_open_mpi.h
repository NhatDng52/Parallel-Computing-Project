#ifndef STRASSEN_OPEN_MPI_H
#define STRASSEN_OPEN_MPI_H

#include <vector>
#include <mpi.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "strassen_utils/strassen_op.h"

using namespace std;
using Matrix = vector<vector<double>>;

class StrassenOpenMPI : public IStrassenOp {
public:
    Matrix apply_strassen(const Matrix &A, const Matrix &B) override;

private:
    static Matrix implement_strassen(const Matrix &A, const Matrix &B);
    static vector<Matrix> divide_mat(const Matrix &A);
    static Matrix padding(const Matrix &A);
    static Matrix remove_padding(const Matrix &A, int original_rows, int original_cols);
    
    static Matrix mat_add(const Matrix &A, const Matrix &B);
    static Matrix mat_sub(const Matrix &A, const Matrix &B);
    static Matrix local_compute(const Matrix &A, const Matrix &B);

    static vector<double> flatten(const Matrix &A);
    static Matrix unflatten(const vector<double> &flat, int n);
    static void send_matrix(const Matrix &A, int dest_rank, int tag);
    static Matrix recv_matrix(int src_rank, int n, int tag);
};

#endif