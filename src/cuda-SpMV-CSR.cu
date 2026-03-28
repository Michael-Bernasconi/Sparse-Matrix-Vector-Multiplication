#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
    void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);
}

/**
 * Standard CUDA error checking macro.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

/**
 * CUDA Kernel for CSR SpMV.
 * Mapping: 1 Thread = 1 Row.
 * This avoids atomic operations but can lead to "load imbalance" if rows 
 * have very different numbers of non-zero elements.
 */
__global__ void spmv_csr_kernel(int M, const int *row_ptr, const int *col_idx, const float *vals, const float *x, float *y) {
    // Calculate the row index this thread is responsible for
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        // Get the start and end indices for this specific row
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        // Loop through the non-zeros of this row
        for (int i = row_start; i < row_end; i++) {
            // __ldg() bypasses the standard cache for the Read-Only Data Cache,
            // which is highly effective for the irregular access pattern of x[col_idx[i]]
            sum += __ldg(&vals[i]) * __ldg(&x[col_idx[i]]);
        }
        // Each thread writes the result to its own assigned row in y
        y[row] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    CSRMatrix A;
    load_matrix_market_to_csr(argv[1], &A);
    int M = A.M; int nnz = A.nnz;

    // Initialize vector x on host
    float *h_x = (float*)malloc(A.N * sizeof(float));
    for (int i = 0; i < A.N; i++) h_x[i] = 1.0f;

    // Device pointers
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    // 1. Allocate GPU memory for CSR components and vectors
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, A.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // 2. Transfer matrix and vector data to the GPU
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, A.N * sizeof(float), cudaMemcpyHostToDevice));

    // Define execution configuration
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    // Events for high-precision timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup: Priming the GPU and checking for kernel launch errors
    spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    cudaDeviceSynchronize();

    // Benchmark loop: 500 iterations
    int iterations = 500;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Force a single element download to ensure the CPU waits for completion 
    // and to verify the kernel actually did something
    float check;
    cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    // Timing calculation
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_time = (ms / 1000.0) / iterations;

    printf("\n--- GPU CSR Benchmark ---\n");
    printf("Avg Time: %e s | GFLOPS: %f (Check: %f)\n", avg_time, (2.0 * nnz) / (avg_time * 1e9), check);

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    free(h_x);
    
    return 0;
}