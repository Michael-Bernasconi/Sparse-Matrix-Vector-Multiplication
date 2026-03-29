#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
}

/**
 * Standard CUDA error checking macro.
 * Checks the return value of CUDA API calls and prints the error 
 * string along with the file and line number if a failure occurs.
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
 * CUDA CSR kernel: row-parallel implementation.
 * Thread Mapping: Each thread is responsible for calculating one full row 
 * of the result vector 'y'.
 * * Logic: 
 * The thread fetches its assigned row index, then loops through the 
 * non-zero elements of that row using the row_ptr array to find the 
 * start and end boundaries in values and col_idx.
 */
__global__ void spmv_csr_kernel(int M, const int *row_ptr, const int *col_idx, 
                                const float *vals, const float *x, float *y) {
    // Determine the global row index for this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: ensure the thread is within the number of rows (M)
    if (row < M) {
        float sum = 0.0f;
        // Fetch row boundaries from the row pointer array
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];
        
        // Accumulate products for all non-zero elements in this row
        for (int i = row_start; i < row_end; i++) {
            // __ldg() hint for the compiler to use the Read-Only Data Cache
            sum += __ldg(&vals[i]) * __ldg(&x[col_idx[i]]);
        }
        // Write the final result directly to the output vector
        // No atomic operations needed as each thread writes to a unique index
        y[row] = sum;
    }
}

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
    // Note: row_ptr size is M+1 to account for the end-of-row boundary
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

    // --- 3. EXECUTION CONFIGURATION ---
    // Grid size is based on the number of rows (M) since each thread processes a row
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;

    // Initialize CUDA events for high-resolution timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    // Execute multiple runs to stabilize hardware and prime caches
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    // Measure time only for the compute kernel and output reset
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_csr_kernel<<<gridSize, blockSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
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
    printf("\n--- GPU CSR Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, A.N, nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", check);

    // --- CLEANUP ---
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    free(h_x);
    // Note: matrix arrays (A.row_ptr, etc.) are freed during Matrix structure destruction if needed
    
    return 0;
}