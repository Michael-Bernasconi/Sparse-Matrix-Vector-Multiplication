#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
    void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix);
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
 * CUDA COO kernel
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols,
                                const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nnz) {
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;

    // ✅ Load DIRECTLY in COO
    COOMatrix mat;
    load_matrix_market_to_coo(argv[1], &mat);

    int nnz = mat.nnz;
    int M = mat.M;

    float *h_x = (float*)malloc(mat.N * sizeof(float));
    for (int i = 0; i < mat.N; i++) h_x[i] = 1.0f;

    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_rows, mat.rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, mat.cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, mat.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.N * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    cudaMemset(d_y, 0, M * sizeof(float));
    spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    cudaDeviceSynchronize();

    int iterations = 500;

    cudaEventRecord(start);

    for (int i = 0; i < iterations; i++) {
        cudaMemset(d_y, 0, M * sizeof(float));
        spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    double avg_time = (ms / 1000.0) / iterations;

    float check;
    cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n--- GPU COO Benchmark ---\n");
    printf("Avg Time: %e s | GFLOPS: %f (Check: %f)\n",
           avg_time,
           (2.0 * nnz) / (avg_time * 1e9),
           check);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_rows);
    cudaFree(d_cols);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);

    free(h_x);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}