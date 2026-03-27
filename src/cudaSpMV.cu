#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "my_time_lib.h"
#include "spmv_formats.h"

/**
 * CUDA Kernel: SpMV CSR Scalar implementation
 * Each thread handles exactly one row of the matrix.
 * It computes: y[i] = A[i,:] * x
 */
__global__ void spmv_csr_kernel(int M, const int* row_ptr, const int* col_idx, const float* values, const float* x, float* y) {
    // Map thread to row index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary condition to avoid out-of-bounds access
    if (i < M) {
        float sum = 0.0f;
        // Loop through non-zero elements of the current row
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        // Write result to the output vector
        y[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 2) {
        printf("Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    // --- 1. HOST PREPARATION (CPU) ---
    CSRMatrix A;
    // Load matrix from file and convert to CSR format
    load_matrix_market_to_csr(argv[1], &A);

    Vector x, y;
    x.size = A.N; x.data = (float*)malloc(x.size * sizeof(float));
    y.size = A.M; y.data = (float*)calloc(y.size, sizeof(float));
    
    // Initialize input vector x with dummy values (1.0f)
    for (int i = 0; i < x.size; i++) x.data[i] = 1.0f;

    // --- 2. DEVICE ALLOCATION (GPU) ---
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    // Allocate global memory on the GPU for matrix and vectors
    cudaMalloc(&d_row_ptr, (A.M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, A.nnz * sizeof(int));
    cudaMalloc(&d_values, A.nnz * sizeof(float));
    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, y.size * sizeof(float));

    // --- 3. HOST TO DEVICE DATA TRANSFER ---
    // Transfer matrix data and vector x to the GPU memory
    // Transfers are kept outside the timer to measure pure kernel performance
    cudaMemcpy(d_row_ptr, A.row_ptr, (A.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A.col_idx, A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A.values, A.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data, x.size * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. KERNEL EXECUTION CONFIGURATION ---
    // Define block size and calculate grid size based on number of rows
    int blockSize = 256;
    int gridSize = (A.M + blockSize - 1) / blockSize;
    int num_iterations = 100;

    // --- 5. KERNEL BENCHMARK ---
    TIMER_DEF(timer);
    printf("Matrix: %s | NNZ: %d\n", argv[1], A.nnz);
    printf("Running GPU benchmark with %d iterations...\n", num_iterations);

    TIMER_START(timer);
    // Execute multiple iterations to amortize overhead and startup latency
    for (int i = 0; i < num_iterations; i++) {
        spmv_csr_kernel<<<gridSize, blockSize>>>(A.M, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    }
    // Synchronize to ensure all kernel executions are finished before stopping timer
    cudaDeviceSynchronize(); 
    TIMER_STOP(timer);

    // --- 6. RESULTS AND STATISTICS ---
    double total_time = TIMER_ELAPSED(timer) / 1e6; // Convert microseconds to seconds
    double avg_time = total_time / num_iterations;

    printf("GPU Total Time (%d runs): %lfs\n", num_iterations, total_time);
    printf("GPU Average Time per run:  %lfs\n", avg_time);

    // Copy the result vector y back to CPU memory for validation
    cudaMemcpy(y.data, d_y, y.size * sizeof(float), cudaMemcpyDeviceToHost);

    // --- 7. CLEANUP AND MEMORY DEALLOCATION ---
    // Free GPU resources
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    // Free CPU resources
    free(A.row_ptr); free(A.col_idx); free(A.values); free(x.data); free(y.data);

    return 0;
}