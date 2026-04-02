#include <cuda_runtime.h>
#include <cusparse.h>
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
 * Standard cuSPARSE error checking macro.
 * Specifically checks the return values of cuSPARSE library functions.
 */
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            printf("cuSPARSE Error at %s:%d - code %d\n", __FILE__, __LINE__, status); \
            exit(1); \
        } \
    } while (0)

int main(int argc, char **argv) {
    // Command line argument check
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Load matrix into host memory directly in CSR format
    CSRMatrix A;
    load_matrix_market_to_csr(argv[1], &A);
    int M = A.M; 
    int nnz = A.nnz;

    // Allocate and initialize the input vector x on the Host
    float *h_x = (float*)malloc(A.N * sizeof(float));
    fill_random_vector(h_x, A.N);

    // Device (GPU) pointers for matrix and vectors
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    // --- 1. DEVICE MEMORY ALLOCATION ---
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, A.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // --- 2. DATA TRANSFER (HOST TO DEVICE) ---
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, A.N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 3. CUSPARSE SETUP AND INITIALIZATION ---
    // Initialize the cuSPARSE library handle
    cusparseHandle_t handle = NULL;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Create descriptors for the sparse matrix (A) and dense vectors (x, y)
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // Create sparse matrix A descriptor in CSR format
    // We specify 32-bit integers (CUSPARSE_INDEX_32I) and 32-bit floats (CUDA_R_32F)
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, A.N, nnz,
                                      d_row_ptr, d_col_idx, d_vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, A.N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F));

    // Set SpMV constants: y = alpha * A * x + beta * y
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Determine the required buffer size for the cuSPARSE SpMV algorithm
    size_t bufferSize = 0;
    void *dBuffer = NULL;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
    // Allocate the external buffer required by cuSPARSE
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Initialize CUDA events for high-resolution timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // --- PERFORMANCE CALCULATIONS ---
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_time_s = (ms / 1000.0) / BENCHMARK_ITERATIONS;

    // Verification: copy first element back to Host to ensure correct computation
    float check;
    CUDA_CHECK(cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate throughput and bandwidth using common utilities
    double gflops = calculate_gflops(nnz, avg_time_s);
    double bw = calculate_bandwidth(M, A.N, nnz, avg_time_s, "CSR");

    // Display formatted results
    printf("\n--- cuSPARSE BASELINE Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, A.N, nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", check);

    // --- CLEANUP ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // Destroy cuSPARSE descriptors and handle
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    // Free GPU memory
    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    
    return 0;
}