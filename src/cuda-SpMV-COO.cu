#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h> // Added for memset

extern "C" {
    #include "spmv_formats.h"
}

/**
 * Sequential version to provide the "Gold Standard" reference for results validation.
 * Runs on the CPU.
 */
void spmv_coo_sequential(const COOMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
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
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols, 
                                const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nnz) {
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char **argv) {
    double global_start = omp_get_wtime(); // TTS start measurement
    
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Load sparse matrix into host memory COO format
    COOMatrix mat;
    load_matrix_market_to_coo(argv[1], &mat);
    int nnz = mat.nnz;
    int M = mat.M;

    // Allocate and initialize input vector x on the Host
    float *h_x = (float*)malloc(mat.N * sizeof(float));
    fill_random_vector(h_x, mat.N);

    // --- 1. REFERENCE GENERATION ---
    // Allocate host vectors for validation
    float *h_y_ref = (float*)malloc(M * sizeof(float));
    float *h_y_gpu = (float*)malloc(M * sizeof(float)); // Buffer to copy back GPU results
    
    // Compute the sequential result on CPU as ground truth
    memset(h_y_ref, 0, M * sizeof(float));
    spmv_coo_sequential(&mat, h_x, h_y_ref);

    // Device (GPU) pointers
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    // --- 2. DEVICE MEMORY ALLOCATION ---
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // --- 3. DATA TRANSFER (HOST TO DEVICE) ---
    CUDA_CHECK(cudaMemcpy(d_rows, mat.rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, mat.cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, mat.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 4. EXECUTION CONFIGURATION ---
    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARM-UP PHASE ---
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

    // --- PERFORMANCE CALCULATIONS ---
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_time_s = (ms / 1000.0) / BENCHMARK_ITERATIONS;
    
    double gflops = calculate_gflops(nnz, avg_time_s);
    double bw = calculate_bandwidth(M, mat.N, nnz, avg_time_s, "COO");
    double tts = calculate_tts(global_start);

    // --- VALIDATION PHASE ---
    // Copy the full result back from GPU to Host for verification
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    // Rigorous element-by-element validation against the CPU reference
    validate_results(h_y_ref, h_y_gpu, M);

    // Display formatted results
    printf("\n--- GPU COO Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, mat.N, nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("TTS     : %.4f s\n", tts); 
    printf("Check   : %f (First element of y)\n", h_y_gpu[0]);

    // --- CLEANUP ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rows));
    CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    free(h_x);
    free(h_y_ref);
    free(h_y_gpu);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}