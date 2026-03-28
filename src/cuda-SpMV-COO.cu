#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
    void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);
}

/**
 * Macro for CUDA error checking.
 * Wraps CUDA API calls and prints the error string if a failure occurs.
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
 * Each thread handles one non-zero entry.
 * __ldg() is used to hint the compiler to use the Read-Only Data Cache for better performance.
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols, const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        // atomicAdd is necessary because multiple threads may update the same y[row] index
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    CSRMatrix temp;
    load_matrix_market_to_csr(argv[1], &temp);
    int M = temp.M; int nnz = temp.nnz;

    // Host-side decompression: Convert CSR row pointers to COO row indices
    int *h_rows = (int*)malloc(nnz * sizeof(int));
    for (int i = 0; i < M; i++) {
        for (int j = temp.row_ptr[i]; j < temp.row_ptr[i + 1]; j++) h_rows[j] = i;
    }

    // Initialize input vector x on host
    float *h_x = (float*)malloc(temp.N * sizeof(float));
    for (int i = 0; i < temp.N; i++) h_x[i] = 1.0f;

    // Device (GPU) pointers
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    // 1. Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, temp.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // 2. Transfer data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_rows, h_rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, temp.col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, temp.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, temp.N * sizeof(float), cudaMemcpyHostToDevice));

    // Calculate grid size (number of blocks) based on a 256-thread block size
    int gridSize = (nnz + 255) / 256;

    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup run to prime the GPU
    cudaMemset(d_y, 0, M * sizeof(float));
    spmv_coo_kernel<<<gridSize, 256>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    cudaDeviceSynchronize();

    // Main benchmarking loop
    int iterations = 500;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        // NOTE: In a real application, you must reset d_y to zero here.
        // It's commented out here to measure the raw kernel execution speed.
        spmv_coo_kernel<<<gridSize, 256>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copy a single result back for a quick sanity check
    float check;
    cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate elapsed time and performance
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double avg_time = (ms / 1000.0) / iterations; // average time in seconds

    printf("\n--- GPU COO Benchmark ---\n");
    printf("Avg Time: %e s | GFLOPS: %f (Check: %f)\n", avg_time, (2.0 * nnz) / (avg_time * 1e9), check);

    // Cleanup GPU and Host resources
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_rows); cudaFree(d_cols); cudaFree(d_vals); cudaFree(d_x); cudaFree(d_y);
    free(h_rows); free(h_x);
    
    return 0;
}