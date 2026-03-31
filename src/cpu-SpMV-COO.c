#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"
#include "my_time_lib.h"
#include <omp.h>


/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using COO format.
 * Computes: y = A * x
 * * COO Logic:
 * Unlike CSR, the Coordinate format stores each non-zero element as a triplet 
 * (row, column, value). The algorithm iterates through all non-zero elements (nnz)
 * and updates the corresponding row in the output vector 'y'.
 */
void spmv_coo_cpu(const COOMatrix *mat, const float *x, float *y) {
    // Iterate through all non-zero elements
    #pragma omp parallel for    //multi-core
    for (int i = 0; i < mat->nnz; i++) {
        // mat->rows[i]: destination row index
        // mat->cols[i]: index for the input vector x
        // mat->values[i]: the non-zero value
        #pragma omp atomic //to avoid RACE CONDITIONS
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

int main(int argc, char **argv) {
    // Validate command line arguments
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Structure for Coordinate (COO) sparse matrix format
    COOMatrix mat;
    // Load matrix from Matrix Market file (.mtx) into the COO structure
    load_matrix_market_to_coo(argv[1], &mat);

    // Allocate memory for the input dense vector x and output vector y
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));

    if (!x || !y) {
        fprintf(stderr, "Critical: Memory allocation failed\n");
        return 1;
    }

    // Initialize the dense vector x with random values
    fill_random_vector(x, mat.N);

    // --- WARM-UP PHASE ---
    // Multiple runs to stabilize CPU thermal state and pre-load caches
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        // Ensure y is zeroed before each run for consistency
        memset(y, 0, mat.M * sizeof(float));
        spmv_coo_cpu(&mat, x, y);
    }

    // --- BENCHMARK PHASE ---
    // Accurate timing measurement over several iterations
    TIMER_DEF(0);
    TIMER_START(0);

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Reset output vector every iteration to simulate a fresh operation
        memset(y, 0, mat.M * sizeof(float));
        spmv_coo_cpu(&mat, x, y);
    }

    TIMER_STOP(0);

    // --- PERFORMANCE CALCULATIONS ---
    // Average execution time in seconds
    double avg_time_s = (TIMER_ELAPSED(0) / 1e6) / BENCHMARK_ITERATIONS;
    
    // Calculate performance metrics (GFLOPS and Effective Bandwidth)
    double gflops = calculate_gflops(mat.nnz, avg_time_s);
    double bw = calculate_bandwidth(mat.M, mat.N, mat.nnz, avg_time_s, "COO");

    // Print results in the extended reporting format
    printf("\n--- CPU COO Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], mat.M, mat.N, mat.nnz);
    printf("Avg Time: %e s\n", avg_time_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("Check   : %f (First element of y)\n", y[0]);

    // --- CLEANUP ---
    // Free allocated memory to prevent leaks
    free(x);
    free(y);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}