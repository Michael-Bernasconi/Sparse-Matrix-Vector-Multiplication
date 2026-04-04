#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

extern "C" {
    #include "spmv_formats.h"
    #include "my_time_lib.h" // Added to use arithmetic_mean and sigma_fn_sol
}

/**
 * Sequential version to provide the "Gold Standard" reference for results validation.
 * Runs on the CPU using CSR format.
 */
void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        int row_start = mat->row_ptr[i];
        int row_end   = mat->row_ptr[i+1];
        for (int j = row_start; j < row_end; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
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
 * Standard cuSPARSE error checking macro.
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
    double global_start = omp_get_wtime(); //start measure TTS
    
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

    // --- 1. REFERENCE GENERATION ---
    float *h_y_ref = (float*)malloc(M * sizeof(float));
    float *h_y_gpu = (float*)malloc(M * sizeof(float)); 
    
    // Compute the sequential result on CPU as ground truth
    memset(h_y_ref, 0, M * sizeof(float));
    spmv_csr_sequential(&A, h_x, h_y_ref);

    // Device (GPU) pointers
    int *d_row_ptr, *d_col_idx;
    float *d_vals, *d_x, *d_y;

    // --- 2. DEVICE MEMORY ALLOCATION ---
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, A.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // --- 3. DATA TRANSFER (HOST TO DEVICE) ---
    CUDA_CHECK(cudaMemcpy(d_row_ptr, A.row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, A.col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, A.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, A.N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 4. CUSPARSE SETUP AND INITIALIZATION ---
    cusparseHandle_t handle = NULL;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    CUSPARSE_CHECK(cusparseCreateCsr(&matA, M, A.N, nnz,
                                      d_row_ptr, d_col_idx, d_vals,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, A.N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, M, d_y, CUDA_R_32F));

    float alpha = 1.0f;
    float beta  = 0.0f;

    size_t bufferSize = 0;
    void *dBuffer = NULL;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

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
    // Allocate array to store the execution time of each individual iteration
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    if (!iter_times) {
        fprintf(stderr, "Critical: Memory allocation failed for iter_times array\n");
        return 1;
    }

    // Accurate timing measurement recording every single iteration on the GPU
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Start recording time
        CUDA_CHECK(cudaEventRecord(start));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        // Stop recording time
        CUDA_CHECK(cudaEventRecord(stop));
        
        // Wait for the GPU to finish this specific iteration
        CUDA_CHECK(cudaEventSynchronize(stop));

        // Calculate elapsed time in milliseconds and convert to seconds
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        iter_times[i] = (double)ms / 1000.0;
    }

    // --- PERFORMANCE CALCULATIONS ---
    // Calculate the arithmetic mean and standard deviation (variability) of the iteration times
    double avg_time_s = arithmetic_mean(iter_times, BENCHMARK_ITERATIONS);
    double std_dev_s  = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // --- VALIDATION PHASE ---
    // Transfer the result from GPU back to Host
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    // Verify results against CPU reference
    validate_results(h_y_ref, h_y_gpu, M);

    double gflops = calculate_gflops(nnz, avg_time_s);
    double bw = calculate_bandwidth(M, A.N, nnz, avg_time_s, "CSR");
    double tts = calculate_tts(global_start);

    printf("\n--- cuSPARSE BASELINE Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, A.N, nnz);
    printf("Avg Time: %e s ", avg_time_s);
    printf("Std Dev Time(± %e s)\n", std_dev_s);    
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("TTS     : %.4f s\n", tts); 
    printf("Check   : %f (First element of y)\n", h_y_gpu[0]);

    // --- CLEANUP ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    
    // Free all host allocated memory to prevent memory leaks
    free(iter_times);
    free(h_x);
    free(h_y_ref);
    free(h_y_gpu);
    free(A.row_ptr);
    free(A.col_idx);
    free(A.values);
    
    return 0;
}