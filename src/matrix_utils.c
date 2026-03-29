#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "spmv_formats.h"

void fill_random_vector(float *vec, int n) {
    srand(42); // Fixed seed for reproducibility across CPU and GPU
    for (int i = 0; i < n; i++) vec[i] = (float)rand() / RAND_MAX;
}

double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char* format) {
    size_t bytes = 0;
    if (strcmp(format, "CSR") == 0) {
        bytes = (sizeof(int) * (M + 1 + nnz)) + (sizeof(float) * (nnz + N + M));
    } else { // COO
        bytes = (sizeof(int) * (2 * nnz)) + (sizeof(float) * (nnz + N + M));
    }
    return (double)bytes / (avg_time_s * 1e9);
}

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
        coo_rows[i] -= 1; coo_cols[i] -= 1;
        coo_vals[i] = (float)val;
    }
    fclose(f);

    matrix->row_ptr = calloc((rows + 1), sizeof(int));
    matrix->col_idx = malloc(nnz * sizeof(int));
    matrix->values = malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++) matrix->row_ptr[coo_rows[i] + 1]++;
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
        matrix->rows[i] -= 1; matrix->cols[i] -= 1;
        matrix->values[i] = (float)val;
    }
    fclose(f);
}