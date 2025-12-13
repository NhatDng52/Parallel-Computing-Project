#include "matmul_naive.h"
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const double* A, const double* B, double* C, 
                               int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rowsA && col < colsB) {
        double sum = 0.0;
        for (int k = 0; k < colsA; ++k) {
            sum += A[row * colsA + k] * B[k * colsB + col];
        }
        C[row * colsB + col] = sum;
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Host function to perform matrix multiplication on GPU
std::vector<std::vector<double>> matrix_mult_naive_GPU(
    const std::vector<std::vector<double>>& A, 
    const std::vector<std::vector<double>>& B) {
    
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();
    
    // Check dimensions
    if (colsA != rowsB) {
        std::cerr << "Matrix dimensions mismatch!" << std::endl;
        return std::vector<std::vector<double>>();
    }
    
    // Flatten input matrices
    std::vector<double> A_flat(rowsA * colsA);
    std::vector<double> B_flat(rowsB * colsB);
    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A_flat[i * colsA + j] = A[i][j];
        }
    }
    
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B_flat[i * colsB + j] = B[i][j];
        }
    }
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    size_t sizeA = rowsA * colsA * sizeof(double);
    size_t sizeB = rowsB * colsB * sizeof(double);
    size_t sizeC = rowsA * colsB * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, A_flat.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B_flat.data(), sizeB, cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x,
                  (rowsA + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, rowsA, colsA, colsB);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for kernel to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    std::vector<double> C_flat(rowsA * colsB);
    CUDA_CHECK(cudaMemcpy(C_flat.data(), d_C, sizeC, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // Convert result back to 2D vector
    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB));
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            result[i][j] = C_flat[i * colsB + j];
        }
    }
    
    return result;
}
