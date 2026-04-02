#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {
    #include "spmv_formats.h"
}

/**
 * Standard CUDA error checking macro.
 * Intercepts CUDA API return values and terminates execution on failure
 * with a descriptive error message and line number.
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
 * Thread Mapping: Each thread processes exactly one non-zero element from the matrix.
 * * * AtomicAdd: Necessary because multiple non-zero elements might belong to the same 
 * row, causing multiple threads to attempt updating the same index in vector 'y' 
 * simultaneously (Race Condition).
 * * __ldg(): Instructs the compiler to use the Read-Only Data Cache, which is highly 
 * efficient for the irregular access patterns typically found in sparse operations.
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols, 
                                const float *vals, const float *x, float *y) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check to prevent illegal memory access if nnz is not a multiple of blockSize
    if (i < nnz) {
        // Atomic addition to global memory ensures correct accumulation of partial products
        // __ldg() hint for the compiler to use the Read-Only Data Cache
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char **argv) {
    // Verify command line arguments
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

    // Device (GPU) pointers
    int *d_rows, *d_cols;
    float *d_vals, *d_x, *d_y;

    // --- 1. DEVICE MEMORY ALLOCATION ---
    // Allocate space for matrix arrays and vectors on the GPU VRAM
    CUDA_CHECK(cudaMalloc(&d_rows, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, mat.N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float)));

    // --- 2. DATA TRANSFER (HOST TO DEVICE) ---
    // Copy matrix data and input vector from CPU RAM to GPU VRAM
    CUDA_CHECK(cudaMemcpy(d_rows, mat.rows, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, mat.cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, mat.values, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, mat.N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 3. EXECUTION CONFIGURATION ---
    // I use a fixed block size of 256 threads, a common "sweet spot" for modern GPUs.
    // The number of blocks is calculated to cover all non-zero elements (nnz).
    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;

    // Setup CUDA events for high-precision timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARM-UP PHASE ---
    // Run the kernel several times to stabilize GPU clock speeds and prime caches.
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Reset output vector to zero before each calculation
        spmv_coo_kernel<<<gridSize, blockSize>>>(nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }
    // Synchronize to ensure warm-up is complete before starting the timer
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    // Measure only the kernel and memset execution time
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
    // Calculate average execution time in seconds
    double avg_time_s = (ms / 1000.0) / BENCHMARK_ITERATIONS;
    
    // Throughput (GFLOPS) and Effective Bandwidth (GB/s)
    double gflops = calculate_gflops(nnz, avg_time_s);
    double bw = calculate_bandwidth(M, mat.N, nnz, avg_time_s, "COO");

    // Copy the first element of the result back to Host for verification purposes
    float check;
    CUDA_CHECK(cudaMemcpy(&check, d_y, sizeof(float), cudaMemcpyDeviceToHost));

    // Display formatted results
    printf("\n--- GPU COO Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, mat.N, nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", check);

    // --- CLEANUP ---
    // Release all GPU and CPU resources
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