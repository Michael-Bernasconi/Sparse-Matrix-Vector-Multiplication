#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"
#include "my_time_lib.h"
#include <omp.h>

// Sequential version to provide the "Gold Standard" reference for results validation
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
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using CSR format.
 * Computes the operation: y = A * x
 * * CSR Logic:
 * Each row 'i' is processed independently. The row_ptr array provides the
 * start and end indices for non-zero elements belonging to row 'i' within
 * the values and col_idx arrays.
 */
void spmv_csr_cpu(const CSRMatrix *mat, const float *x, float *y)
{
#pragma omp parallel for // Enable multi-core parallelization
    // Iterate over each row of the matrix
    for (int i = 0; i < mat->M; i++)
    {
        float sum = 0.0f;

        // Boundaries of the current row in the packed arrays
        int row_start = mat->row_ptr[i];
        int row_end = mat->row_ptr[i + 1];
        // There is no need for omp atomic since each thread is assigned a unique row index i.
        // This prevents write conflicts and ensures there are no race conditions on the output vector y.
        // Dot product between the sparse row and the dense vector x
        for (int j = row_start; j < row_end; j++)
        {
            // values[j] is the non-zero element, col_idx[j] is its column position
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        // Store the result in the output vector y (direct assignment, no += needed)
        y[i] = sum;
    }
}

int main(int argc, char **argv)
{
    double global_start = omp_get_wtime(); // Capture the start time immediately for Time-to-Solution (TTS)

    // Basic command line argument validation
    if (argc < 2)
    {
        printf("Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    // Structure to hold the sparse matrix in Compressed Sparse Row format
    CSRMatrix mat;
    // Utility function to parse Matrix Market file and convert it to CSR structure
    load_matrix_market_to_csr(argv[1], &mat);

    // Memory allocation for the input vector (x) and output vector (y)
    float *x = (float *)malloc(mat.N * sizeof(float));
    float *y = (float *)malloc(mat.M * sizeof(float));
    // Allocate memory for the reference vector to verify numerical correctness
    float *y_ref = (float *)malloc(mat.M * sizeof(float));

    if (!x || !y || !y_ref)
    {
        fprintf(stderr, "Critical: Memory allocation failed\n");
        return 1;
    }

    // Initialize the dense vector x with random values (fixed seed for reproducibility)
    fill_random_vector(x, mat.N);

    // --- REFERENCE GENERATION ---
    // Compute the sequential result once to use as a ground truth for validation
    memset(y_ref, 0, mat.M * sizeof(float));
    spmv_csr_sequential(&mat, x, y_ref);

    // --- WARM-UP PHASE ---
    // Perform initial iterations to:
    // 1. Prime the CPU caches (L1, L2, L3)
    // 2. Allow the OS/CPU to reach maximum clock frequency (Turbo Boost)
    for (int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        spmv_csr_cpu(&mat, x, y);
    }

    // --- BENCHMARK PHASE ---
    // Allocate array to store the execution time of each individual iteration
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    if (!iter_times)
    {
        fprintf(stderr, "Critical: Memory allocation failed for iter_times array\n");
        return 1;
    }

    TIMER_DEF(0);
    // Accurate timing measurement recording every single iteration
    for (int i = 0; i < BENCHMARK_ITERATIONS; i++)
    {
        TIMER_START(0);
        spmv_csr_cpu(&mat, x, y);
        TIMER_STOP(0);

        // Store the elapsed time for this specific iteration in seconds
        iter_times[i] = TIMER_ELAPSED(0) / 1e6;
    }

    // --- VALIDATION PHASE ---
    // Rigorous comparison between the parallel result and the sequential reference
    validate_results(y_ref, y, mat.M);

    // --- PERFORMANCE CALCULATIONS ---
    // Calculate the arithmetic mean and standard deviation (variability) of the iteration times
    double avg_time_s = arithmetic_mean(iter_times, BENCHMARK_ITERATIONS);
    double std_dev_s = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // Calculate performance metrics (GFLOPS, Effective Bandwidth and TTS)
    double gflops = calculate_gflops(mat.nnz, avg_time_s);
    double bw = calculate_bandwidth(mat.M, mat.N, mat.nnz, avg_time_s, "CSR");
    double tts = calculate_tts(global_start);

    // Formatted output for reporting
    printf("\n--- CPU CSR Benchmark ---\n");
    printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], mat.M, mat.N, mat.nnz);
    printf("Avg Time: %e s ", avg_time_s);
    printf("Std Dev Time(± %e s)\n", std_dev_s);
    printf("GFLOPS  : %.4f\n", gflops);
    printf("BW      : %.4f GB/s\n", bw);
    printf("TTS     : %.4f s\n", tts);
    printf("Check   : %f (First element of y)\n", y[0]);

    // --- CLEANUP ---
    // Free all allocated memory to prevent memory leaks
    free(iter_times);
    free(x);
    free(y);
    free(y_ref);
    free(mat.row_ptr);
    free(mat.col_idx);
    free(mat.values);

    return 0;
}