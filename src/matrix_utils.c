#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "spmv_formats.h"
#include <omp.h>

/**
 * Reproducibility: Fills a vector with random float values using a fixed seed.
 * This ensures that the input vector x is identical across CPU and GPU tests.
 */
void fill_random_vector(float *vec, int n) {
    srand(42); // Fixed seed for scientific reproducibility
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * METRIC 1: FLOPS (Floating Point Operations Per Second)
 * Measures the raw computational throughput.
 * SpMV performs 2 operations (1 multiply, 1 add) per non-zero element.
 */
double calculate_gflops(int nnz, double avg_time_s) {
    if (avg_time_s <= 0) return 0;
    double total_flops = 2.0 * (double)nnz;
    return total_flops / (avg_time_s * 1e9);
}

/**
 * METRIC 2: EFFECTIVE BANDWIDTH
 * Measures the data transfer rate between memory and processors.
 * Calculation accounts for:
 * - Matrix structure (indices and pointers)
 * - Matrix values
 * - Input vector x and output vector y
 */
double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char* format) {
    if (avg_time_s <= 0) return 0;
    size_t bytes = 0;

    if (strcmp(format, "CSR") == 0) {
        // CSR: row_ptr (M+1 ints), col_idx (nnz ints), values (nnz floats) 
        // + x vector (N floats) + y vector (M floats)
        bytes = (sizeof(int) * (M + 1 + nnz)) + (sizeof(float) * (nnz + N + M));
    } else { 
        // COO: rows (nnz ints), cols (nnz ints), values (nnz floats) 
        // + x vector (N floats) + y vector (M floats)
        bytes = (sizeof(int) * (2 * nnz)) + (sizeof(float) * (nnz + N + M));
    }

    return (double)bytes / (avg_time_s * 1e9);
}

/**
 * METRIC 3: TIME TO SOLUTION
 * Calculates the total Time-to-Solution (TTS).
 * TTS measures the "wall-clock" time from the beginning of the program 
 * (including I/O and memory allocation) until the completion of the task.
 * start_time = The timestamp recorded at the very start of main().
 * return The elapsed time in seconds.
 */
double calculate_tts(double start_time) {
    // Returns current wall-clock time minus the initial timestamp
    return omp_get_wtime() - start_time;
}

/**
 * Loads a Matrix Market file (.mtx) and converts it to CSR format.
 */
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); exit(1); }

    char line[1024];
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    int rows, cols, nnz;
    sscanf(line, "%d %d %d", &rows, &cols, &nnz);
    matrix->M = rows; matrix->N = cols; matrix->nnz = nnz;

    int *coo_rows = malloc(nnz * sizeof(int));
    int *coo_cols = malloc(nnz * sizeof(int));
    float *coo_vals = malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++) {
        double val;
        fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &val);
        coo_rows[i] -= 1; // Convert to 0-based indexing
        coo_cols[i] -= 1;
        coo_vals[i] = (float)val;
    }
    fclose(f);

    matrix->row_ptr = calloc((rows + 1), sizeof(int));
    matrix->col_idx = malloc(nnz * sizeof(int));
    matrix->values = malloc(nnz * sizeof(float));

    // Histogram for row_ptr
    for (int i = 0; i < nnz; i++) matrix->row_ptr[coo_rows[i] + 1]++;
    // Prefix sum
    for (int i = 0; i < rows; i++) matrix->row_ptr[i + 1] += matrix->row_ptr[i];

    int *temp_ptr = malloc(rows * sizeof(int));
    memcpy(temp_ptr, matrix->row_ptr, rows * sizeof(int));
    for (int i = 0; i < nnz; i++) {
        int r = coo_rows[i];
        int dest = temp_ptr[r]++;
        matrix->col_idx[dest] = coo_cols[i];
        matrix->values[dest] = coo_vals[i];
    }

    free(coo_rows); free(coo_cols); free(coo_vals); free(temp_ptr);
}

/**
 * Loads a Matrix Market file (.mtx) directly into COO format.
 */
void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix) {
    FILE *f = fopen(filename, "r");
    if (!f) { fprintf(stderr, "Error opening %s\n", filename); exit(1); }

    char line[1024];
    while (fgets(line, sizeof(line), f) && line[0] == '%');
    sscanf(line, "%d %d %d", &matrix->M, &matrix->N, &matrix->nnz);

    matrix->rows = malloc(matrix->nnz * sizeof(int));
    matrix->cols = malloc(matrix->nnz * sizeof(int));
    matrix->values = malloc(matrix->nnz * sizeof(float));

    for (int i = 0; i < matrix->nnz; i++) {
        double val;
        fscanf(f, "%d %d %lf", &matrix->rows[i], &matrix->cols[i], &val);
        matrix->rows[i] -= 1; // 0-based indexing
        matrix->cols[i] -= 1;
        matrix->values[i] = (float)val;
    }
    fclose(f);
}