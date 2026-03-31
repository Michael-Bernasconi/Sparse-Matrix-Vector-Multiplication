#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"
#include "my_time_lib.h"
#include <omp.h>

/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using CSR format.
 * Computes the operation: y = A * x
 * * CSR Logic:
 * Each row 'i' is processed independently. The row_ptr array provides the 
 * start and end indices for non-zero elements belonging to row 'i' within 
 * the values and col_idx arrays.
 */
void spmv_csr_cpu(const CSRMatrix *mat, const float *x, float *y) {
    #pragma omp parallel for //multi-core
    // Iterate over each row of the matrix
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        
        // Boundaries of the current row in the packed arrays
        int row_start = mat->row_ptr[i];
        int row_end   = mat->row_ptr[i+1];

        // Dot product between the sparse row and the dense vector x
        for (int j = row_start; j < row_end; j++) {
            // values[j] is the non-zero element, col_idx[j] is its column position
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        // Store the result in the output vector y
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
    // Basic command line argument validation
    if (argc < 2) { 
        printf("Usage: %s <matrix.mtx>\n", argv[0]); 
        return 1; 
    }
    
    // Structure to hold the sparse matrix in Compressed Sparse Row format
    CSRMatrix mat;
    // Utility function to parse Matrix Market file and convert it to CSR
    load_matrix_market_to_csr(argv[1], &mat);

    // Memory allocation for the input vector (x) and output vector (y)
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Critical: Memory allocation failed\n");
        return 1;
    }

    // Populate vector x with random values to simulate real computation
    fill_random_vector(x, mat.N);

    // --- WARM-UP PHASE ---
    // Perform initial iterations to:
    // 1. Prime the CPU caches (L1, L2, L3)
    // 2. Allow the OS/CPU to reach maximum clock frequency (Turbo Boost)
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memset(y, 0, mat.M * sizeof(float));
        spmv_csr_cpu(&mat, x, y);
    }

    // --- BENCHMARK PHASE ---
    // Measure only the kernel execution time over multiple iterations
    TIMER_DEF(0);
    TIMER_START(0);

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Reset y to ensure we are not accumulating results across iterations
        memset(y, 0, mat.M * sizeof(float));
        spmv_csr_cpu(&mat, x, y);
    }

    TIMER_STOP(0);

    // --- PERFORMANCE ANALYSIS ---
    // Convert elapsed microseconds to average seconds per iteration
    double avg_time_s = (TIMER_ELAPSED(0) / 1e6) / BENCHMARK_ITERATIONS;
    
    // GFLOPS = (2 * NNZ) / (Time * 1e9)
    double gflops = calculate_gflops(mat.nnz, avg_time_s);
    
    // Effective Bandwidth = Bytes Accessed / Time
    double bw = calculate_bandwidth(mat.M, mat.N, mat.nnz, avg_time_s, "CSR");

    // Formatted output for reporting
    printf("\n--- CPU CSR Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], mat.M, mat.N, mat.nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", y[0]);

    // --- CLEANUP ---
    // Free allocated memory to avoid leaks
    free(x); 
    free(y); 
    free(mat.row_ptr); 
    free(mat.col_idx); 
    free(mat.values);
    
    return 0;
}