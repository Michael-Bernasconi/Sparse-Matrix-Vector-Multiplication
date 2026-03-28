#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
        // row_ptr[i] to row_ptr[i+1] defines the range of non-zero elements in row i
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i+1]; j++) {
            // Standard dot product: value * corresponding x element
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        // Write the final row result to the output vector
        y[i] = sum;
    }
}

int main(int argc, char **argv) {
    // Check for correct command line usage
    if (argc < 2) { 
        printf("Usage: %s <matrix.mtx>\n", argv[0]); 
        return 1; 
    }
    
    // Load the matrix directly into CSR format
    CSRMatrix mat;
    load_matrix_market_to_csr(argv[1], &mat);

    // Memory allocation for input vector x and result vector y
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));
    
    // Initialize x with random values between 0.0 and 1.0
    for(int i = 0; i < mat.N; i++) x[i] = (float)rand() / RAND_MAX;

    // WARM-UP: Execute once to prime the CPU caches
    spmv_csr_cpu(&mat, x, y);

    // Benchmarking loop: 100 iterations for statistical stability
    TIMER_DEF(0);
    TIMER_START(0);
    int iter = 100;
    for(int i = 0; i < iter; i++) {
        spmv_csr_cpu(&mat, x, y);
    }
    TIMER_STOP(0);

    // Calculate performance metrics
    double avg_s = (TIMER_ELAPSED(0) / 1e6) / iter; // Average execution time (seconds)
    
    // GFLOPS: Billion Floating Point Operations per Second
    double gflops = (2.0 * mat.nnz) / (avg_s * 1e9);
    
    /**
     * Bandwidth (BW) calculation in GB/s:
     * Estimates the total data moved: 
     * - Indices: row_ptr (M+1 ints) + col_idx (nnz ints)
     * - Data: values (nnz floats) + vector x (N floats) + vector y (M floats)
     */
    double total_bytes = (sizeof(int) * (mat.M + 1 + mat.nnz) + 
                          sizeof(float) * (mat.nnz + mat.N + mat.M));
    double bw = total_bytes / (avg_s * 1e9);

    printf("CPU CSR -> Time: %fs, GFLOPS: %f, BW: %f GB/s\n", avg_s, gflops, bw);

    // Cleanup: free allocated memory
    free(x); 
    free(y); 
    free(mat.row_ptr); 
    free(mat.col_idx); 
    free(mat.values);
    
    return 0;
}