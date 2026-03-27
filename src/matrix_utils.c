#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"

void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        exit(1);
    }

    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    int rows, cols, nnz;
    sscanf(line, "%d %d %d", &rows, &cols, &nnz);
    
    matrix->M = rows;
    matrix->N = cols;
    matrix->nnz = nnz;

    int *coo_rows = (int *)malloc(nnz * sizeof(int));
    int *coo_cols = (int *)malloc(nnz * sizeof(int));
    float *coo_vals = (float *)malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++) {
        double val; 
        fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &val);
        coo_rows[i] -= 1; 
        coo_cols[i] -= 1;
        coo_vals[i] = (float)val;
    }
    fclose(f);

    matrix->row_ptr = (int *)calloc((rows + 1), sizeof(int));
    matrix->col_idx = (int *)malloc(nnz * sizeof(int));
    matrix->values = (float *)malloc(nnz * sizeof(float));

    for (int i = 0; i < nnz; i++) matrix->row_ptr[coo_rows[i] + 1]++;
    for (int i = 0; i < rows; i++) matrix->row_ptr[i + 1] += matrix->row_ptr[i];

    int *temp_ptr = (int *)malloc(rows * sizeof(int));
    memcpy(temp_ptr, matrix->row_ptr, rows * sizeof(int));
    
    for (int i = 0; i < nnz; i++) {
        int r = coo_rows[i];
        int dest = temp_ptr[r]++;
        matrix->col_idx[dest] = coo_cols[i];
        matrix->values[dest] = coo_vals[i];
    }

    free(coo_rows); free(coo_cols); free(coo_vals); free(temp_ptr);
}