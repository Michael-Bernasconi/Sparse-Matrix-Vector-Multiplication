#include <stdio.h>
#include <stdlib.h>
#include "spmv_formats.h"
#include "my_time_lib.h"

/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using COO format.
 * Computes y = A * x, where A is in Coordinate (COO) format.
 */
void spmv_coo_cpu(const COOMatrix *mat, const float *x, float *y) {
    // Initialize output vector
    for (int i = 0; i < mat->M; i++) {
        y[i] = 0.0f;
    }

    // Perform SpMV
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

int main(int argc, char **argv) {
    // Check input
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        return 1;
    }

    // Load matrix directly in COO format
    COOMatrix mat;
    load_matrix_market_to_coo(argv[1], &mat);

    // Allocate vectors
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));

    if (!x || !y) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize x
    for (int i = 0; i < mat.N; i++) {
        x[i] = 1.0f;
    }

    // Warm-up
    spmv_coo_cpu(&mat, x, y);

    // Benchmark
    TIMER_DEF(0);
    TIMER_START(0);

    for (int i = 0; i < 100; i++) {
        spmv_coo_cpu(&mat, x, y);
    }

    TIMER_STOP(0);

    // Compute average time
    double avg_s = (TIMER_ELAPSED(0) / 1e6) / 100.0;

    // GFLOPS calculation
    double gflops = (2.0 * mat.nnz) / (avg_s * 1e9);

    printf("CPU COO -> Time: %fs, GFLOPS: %f\n", avg_s, gflops);

    // Cleanup
    free(x);
    free(y);
    free(mat.rows);
    free(mat.cols);
    free(mat.values);

    return 0;
}