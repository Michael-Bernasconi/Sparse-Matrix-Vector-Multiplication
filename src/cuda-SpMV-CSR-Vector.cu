#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h> // Added for memset

extern "C"
{
#include "spmv_formats.h"
#include "my_time_lib.h" // Added to use arithmetic_mean and sigma_fn_sol
}

/**
 * Sequential version to provide the "Gold Standard" reference for results validation.
 * Runs on the CPU using CSR format.
 */
void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y)
{
    for (int i = 0; i < mat->M; i++)
    {
        float sum = 0.0f;
        int row_start = mat->row_ptr[i];
        int row_end = mat->row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++)
        {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

/**
 * Standard CUDA error checking macro.
 */
#define CUDA_CHECK(call)                                                                       \
    do                                                                                         \
    {                                                                                          \
        cudaError_t err = call;                                                                \
        if (err != cudaSuccess)                                                                \
        {                                                                                      \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

/**
 * OPTIMIZED CUDA CSR kernel: CSR-Vector implementation.
 */
__global__ void spmv_csr_vector_kernel(int M, const int *row_ptr, const int *col_idx,
                                       const float *vals, const float *x, float *y)
{
    extern __shared__ float sdata[];

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / 32;
    int lane_id = threadIdx.x % 32;

    int row = warp_id;

    if (row < M)
    {
        float sum = 0.0f;
        int row_start = row_ptr[row];
        int row_end = row_ptr[row + 1];

        for (int i = row_start + lane_id; i < row_end; i += 32)
        {
            sum += __ldg(&vals[i]) * __ldg(&x[col_idx[i]]);
        }

        sdata[threadIdx.x] = sum;
        __syncwarp();

        // Tree-based reduction in shared memory
        if (lane_id < 16)
            sdata[threadIdx.x] += sdata[threadIdx.x + 16];
        __syncwarp();
        if (lane_id < 8)
            sdata[threadIdx.x] += sdata[threadIdx.x + 8];
        __syncwarp();
        if (lane_id < 4)
            sdata[threadIdx.x] += sdata[threadIdx.x + 4];
        __syncwarp();
        if (lane_id < 2)
            sdata[threadIdx.x] += sdata[threadIdx.x + 2];
        __syncwarp();
        if (lane_id < 1)
            sdata[threadIdx.x] += sdata[threadIdx.x + 1];
        __syncwarp();

        if (lane_id == 0)
        {
            // Store the result directly (no atomic operations needed for CSR)
            y[row] = sdata[threadIdx.x];
        }
    }
}

int main(int argc, char **argv)
{
    double global_start = omp_get_wtime(); // Start measurement for Time-to-Solution (TTS)

    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    CSRMatrix A;
    load_matrix_market_to_csr(argv[1], &A);
    int M = A.M;
    int nnz = A.nnz;

    // Allocate and initialize the input vector x on the Host
    float *h_x = (float *)malloc(A.N * sizeof(float));
    fill_random_vector(h_x, A.N);

    // --- 1. REFERENCE GENERATION ---
    float *h_y_ref = (float *)malloc(M * sizeof(float));
    float *h_y_gpu = (float *)malloc(M * sizeof(float));

    memset(h_y_ref, 0, M * sizeof(float));
    spmv_csr_sequential(&A, h_x, h_y_ref);

    // Device pointers
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

    // --- 4. EXECUTION CONFIGURATION ---
    int blockSize = 256;
    int warpsPerBlock = blockSize / 32;
    int gridSize = (M + warpsPerBlock - 1) / warpsPerBlock;
    size_t sharedMemSize = blockSize * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        spmv_csr_vector_kernel<<<gridSize, blockSize, sharedMemSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    // Allocate array to store the execution time of each individual iteration
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    if (!iter_times)
    {
        fprintf(stderr, "Critical: Memory allocation failed for iter_times array\n");
        return 1;
    }

    // Accurate timing measurement recording every single iteration on the GPU
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        // Start recording time
        CUDA_CHECK(cudaEventRecord(start));
        spmv_csr_vector_kernel<<<gridSize, blockSize, sharedMemSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
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
    double std_dev_s = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // --- VALIDATION PHASE ---
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));
    validate_results(h_y_ref, h_y_gpu, M);

    double gflops = calculate_gflops(nnz, avg_time_s);
    double bw = calculate_bandwidth(M, A.N, nnz, avg_time_s, "CSR");
    double tts = calculate_tts(global_start);

    // Display formatted results including variability
    printf("\n--- GPU CSR-VECTOR (OPTIMIZED) Benchmark ---\n");
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