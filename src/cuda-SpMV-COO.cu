#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
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
 * CUDA Kernel for COO SpMV.
 * Mapping: Each thread processes one non-zero element.
 * AtomicAdd is required because multiple threads might update the same row in y.
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols, 
                                const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nnz) {
        // __ldg() leverages the Read-Only Data Cache for irregular access patterns
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Load matrix directly in COO format
    COOMatrix mat;
    load_matrix_market_to_coo(argv[1], &mat);

    int nnz = mat.nnz;
    int M = mat.M;

    // Initialize host vector x using the common utility (Seed 42)
    float *h_x = (float*)malloc(mat.N * sizeof(float));
    fill_random_vector(h_x, mat.N);

    // Device pointers
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    // 1. Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // 2. Data Transfer (Host to Device)
    CUDA_CHECK(cudaMemcpy(d_rows, mat.rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, mat.cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, mat.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.N * sizeof(float), cudaMemcpyHostToDevice));

    // Execution Configuration
    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARM-UP PHASE ---
    // Stabilizes GPU clocks and primes caches
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Performance Metrics Calculation
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_time_s = (ms / 1000.0) / BENCHMARK_ITERATIONS;
    double gflops = (2.0 * nnz) / (avg_time_s * 1e9);
    double bw = calculate_bandwidth(M, mat.N, nnz, avg_time_s, "COO");

    // Verification check (first element of result)
    float check;
    CUDA_CHECK(cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n--- GPU COO Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, mat.N, nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", check);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    free(h_x);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}