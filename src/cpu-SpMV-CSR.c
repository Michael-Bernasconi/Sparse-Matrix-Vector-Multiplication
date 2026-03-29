#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"
#include "my_time_lib.h"

/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using CSR format.
 * Computes y = A * x, where A is in Compressed Sparse Row (CSR) format.
 */
void spmv_csr_cpu(const CSRMatrix *mat, const float *x, float *y) {
    // Loop over each row of the matrix
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        // Access non-zero elements of the current row using row_ptr
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i+1]; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { 
        printf("Usage: %s <matrix.mtx>\n", argv[0]); 
        return 1; 
    }
    
    // Load matrix directly into CSR format
    CSRMatrix mat;
    load_matrix_market_to_csr(argv[1], &mat);

    // Allocate memory for vectors
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));
    
    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize vector x using the common utility function
    fill_random_vector(x, mat.N);

    // --- WARM-UP PHASE ---
    // Execute multiple times to prime CPU caches and stabilize clocks
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memset(y, 0, mat.M * sizeof(float));
        spmv_csr_cpu(&mat, x, y);
    }

    // --- BENCHMARK PHASE ---
    TIMER_DEF(0);
    TIMER_START(0);

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Reset output vector every iteration
        memset(y, 0, mat.M * sizeof(float));
        spmv_csr_cpu(&mat, x, y);
    }

    TIMER_STOP(0);

    // Performance calculations
    double avg_time_s = (TIMER_ELAPSED(0) / 1e6) / BENCHMARK_ITERATIONS;
    double gflops = calculate_gflops(mat.nnz, avg_time_s);
    double bw = calculate_bandwidth(mat.M, mat.N, mat.nnz, avg_time_s, "CSR");

    printf("\n--- CPU CSR Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], mat.M, mat.N, mat.nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);

    
    // Cleanup
    free(x); 
    free(y); 
    free(mat.row_ptr); 
    free(mat.col_idx); 
    free(mat.values);
    
    return 0;
}