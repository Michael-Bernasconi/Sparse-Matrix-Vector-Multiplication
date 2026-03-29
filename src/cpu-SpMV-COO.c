#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"
#include "my_time_lib.h"

/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using COO format.
 * Computes y = A * x, where A is in Coordinate (COO) format.
 * This implementation includes a manual reset of the output vector y.
 */
void spmv_coo_cpu(const COOMatrix *mat, const float *x, float *y) {
    // Perform SpMV: y[row] += value * x[col]
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Load matrix directly in COO format
    COOMatrix mat;
    load_matrix_market_to_coo(argv[1], &mat);

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
    // Execute multiple times to stabilize CPU frequency and prime caches
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        memset(y, 0, mat.M * sizeof(float));
        spmv_coo_cpu(&mat, x, y);
    }

    // --- BENCHMARK PHASE ---
    TIMER_DEF(0);
    TIMER_START(0);

    for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
        // Reset output vector every iteration to ensure consistent work
        memset(y, 0, mat.M * sizeof(float));
        spmv_coo_cpu(&mat, x, y);
    }

    TIMER_STOP(0);

    // Performance calculations using utility functions
    double avg_time_s = (TIMER_ELAPSED(0) / 1e6) / BENCHMARK_ITERATIONS;
    double gflops = calculate_gflops(mat.nnz, avg_time_s);
    double bw = calculate_bandwidth(mat.M, mat.N, mat.nnz, avg_time_s, "COO");

    printf("\n--- CPU COO Benchmark ---\n");
    printf("Matrix: %s (%d x %d, nnz: %d)\n", argv[1], mat.M, mat.N, mat.nnz);
    printf("Avg Time: %e s | GFLOPS: %.4f | BW: %.4f GB/s\n", avg_time_s, gflops, bw);

    // Cleanup
    free(x);
    free(y);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}