#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

/**
 * CUDA CSR kernel: each thread calculates one full row of the result vector y.
 */
__global__ void spmv_csr_kernel(int M, const int *row_ptr, const int *col_idx, 
                                const float *vals, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];
        
        for (int i = row_start; i < row_end; i++) {
            sum += __ldg(&vals[i]) * __ldg(&x[col_idx[i]]);
        }
        y[row] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    CSRMatrix A;
    load_matrix_market_to_csr(argv[1], &A);
    int M = A.M; 
    int nnz = A.nnz;

    // Initialize host vector x
    float *h_x = (float*)malloc(A.N * sizeof(float));
    fill_random_vector(h_x, A.N);

    // Device pointers
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    // Memory Allocation
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, A.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // Data Transfer
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, A.N * sizeof(float), cudaMemcpyHostToDevice));

    // Execution Configuration
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // CSR kernel doesn't strictly need memset if y is fully overwritten, 
        // but we keep it for consistency with COO and CPU benchmarks.
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_time_s = (ms / 1000.0) / BENCHMARK_ITERATIONS;
    double gflops = (2.0 * nnz) / (avg_time_s * 1e9);
    double bw = calculate_bandwidth(M, A.N, nnz, avg_time_s, "CSR");

    printf("\n--- GPU CSR Benchmark ---\n");
    printf("Matrix: %s (%d x %d, nnz: %d)\n", argv[1], M, A.N, nnz);
    printf("Avg Time: %e s | GFLOPS: %.4f | BW: %.4f GB/s\n", avg_time_s, gflops, bw);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    // Note: matrix A arrays are freed in matrix_utils loader or main depending on your style
    
    return 0;
}