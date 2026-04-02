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
 * OPTIMIZED CUDA CSR kernel: CSR-Vector implementation.
 * Thread Mapping: 1 Warp (32 threads) is responsible for calculating one full row.
 * * * Optimizations Applied:
 * 1. Load Balancing: Instead of 1 thread handling a full row (which causes severe divergence 
 * if rows have vastly different lengths), 32 threads share the workload of a single row.
 * 2. Memory Coalescing: The 32 threads read adjacent elements in memory simultaneously (i += 32), 
 * maximizing global memory bandwidth utilization.
 * 3. Shared Memory: Used to perform a fast parallel reduction of the 32 partial sums into a 
 * single final result for the row, avoiding slow global memory atomic operations.
 */
__global__ void spmv_csr_vector_kernel(int M, const int *row_ptr, const int *col_idx, 
                                       const float *vals, const float *x, float *y) {
    // Dynamically allocated shared memory. Size is defined during kernel launch.
    // We need 1 float for every thread in the block to store partial sums.
    extern __shared__ float sdata[];

    // Calculate global thread IDs and Warp IDs
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id   = thread_id / 32;   // 1 Warp = 32 threads. Global ID of the warp.
    int lane_id   = threadIdx.x % 32; // ID of the thread within its warp (0 to 31).

    // In CSR-Vector, each Warp processes exactly one row.
    int row = warp_id;
    
    // Boundary check: ensure the warp is assigned to a valid row (M)
    if (row < M) {
        float sum = 0.0f;
        
        // Fetch row boundaries from the row pointer array
        int row_start = row_ptr[row];
        int row_end   = row_ptr[row + 1];
        
        // All 32 threads in the warp iterate over the non-zero elements of this row.
        // Stride is 32, guaranteeing perfectly coalesced reads from global memory.
        for (int i = row_start + lane_id; i < row_end; i += 32) {
            // __ldg() hint for the compiler to use the Read-Only Data Cache
            sum += __ldg(&vals[i]) * __ldg(&x[col_idx[i]]);
        }
        
        // --- PARALLEL REDUCTION IN SHARED MEMORY ---
        // 1. Each thread writes its partial sum to the shared memory array.
        sdata[threadIdx.x] = sum;
        __syncwarp(); // Synchronize all threads in the warp to ensure memory visibility

        // 2. Tree-based reduction: fold the 32 values into 1.
        // Threads add the value of the thread offset by 16, then 8, 4, 2, 1.
        if (lane_id < 16) sdata[threadIdx.x] += sdata[threadIdx.x + 16]; __syncwarp();
        if (lane_id < 8)  sdata[threadIdx.x] += sdata[threadIdx.x + 8];  __syncwarp();
        if (lane_id < 4)  sdata[threadIdx.x] += sdata[threadIdx.x + 4];  __syncwarp();
        if (lane_id < 2)  sdata[threadIdx.x] += sdata[threadIdx.x + 2];  __syncwarp();
        if (lane_id < 1)  sdata[threadIdx.x] += sdata[threadIdx.x + 1];  __syncwarp();

        // 3. The first thread of the warp (lane_id 0) now holds the final sum.
        // It writes the result directly to the output vector.
        if (lane_id == 0) {
            y[row] = sdata[threadIdx.x];
        }
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

    // --- 3. OPTIMIZED EXECUTION CONFIGURATION ---
    int blockSize = 256; 
    // Calculate how many warps are in a single block (256 threads / 32 = 8 warps)
    int warpsPerBlock = blockSize / 32;
    
    // Grid size is now based on WARPS, not threads, because 1 Warp processes 1 Row (M).
    int gridSize = (M + warpsPerBlock - 1) / warpsPerBlock;

    // Calculate Shared Memory Size dynamically. We need 1 float for each thread in the block.
    size_t sharedMemSize = blockSize * sizeof(float);

    // Initialize CUDA events for high-resolution timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    // Execute multiple runs to stabilize hardware and prime caches
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Note the addition of 'sharedMemSize' as the 3rd kernel launch parameter
        spmv_csr_vector_kernel<<<gridSize, blockSize, sharedMemSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- BENCHMARK PHASE ---
    // Measure time only for the compute kernel and output reset
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        spmv_csr_vector_kernel<<<gridSize, blockSize, sharedMemSize>>>(M, d_row_ptr, d_col_idx, d_vals, d_x, d_y);
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
    printf("\n--- GPU CSR-VECTOR (OPTIMIZED) Benchmark ---\n");
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
    
    return 0;
}