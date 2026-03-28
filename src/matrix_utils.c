#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "spmv_formats.h"

/**
 * Loads a sparse matrix from a Matrix Market (.mtx) file and converts it to CSR format.
 * Handles the 1-based indexing of .mtx files by converting to 0-based C indexing.
 */
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Error: could not open file %s\n", filename);
        exit(1);
    }

    char line[1024];
    // Skip the header comments (lines starting with '%')
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // Read dimensions: number of rows, columns, and non-zero elements (nnz)
    int rows, cols, nnz;
    sscanf(line, "%d %d %d", &rows, &cols, &nnz);
    
    matrix->M = rows;
    matrix->N = cols;
    matrix->nnz = nnz;

    // Temporary storage for Coordinate (COO) data
    int *coo_rows = (int *)malloc(nnz * sizeof(int));
    int *coo_cols = (int *)malloc(nnz * sizeof(int));
    float *coo_vals = (float *)malloc(nnz * sizeof(float));

    // Pass 1: Read all non-zero entries from the file
    for (int i = 0; i < nnz; i++) {
        double val; 
        fscanf(f, "%d %d %lf", &coo_rows[i], &coo_cols[i], &val);
        // Adjust for 1-based indexing used in Matrix Market files
        coo_rows[i] -= 1; 
        coo_cols[i] -= 1;
        coo_vals[i] = (float)val;
    }
    fclose(f);

    // Allocate final CSR arrays
    matrix->row_ptr = (int *)calloc((rows + 1), sizeof(int));
    matrix->col_idx = (int *)malloc(nnz * sizeof(int));
    matrix->values = (float *)malloc(nnz * sizeof(float));

    // Pass 2a: Count elements per row to build the histogram (row_ptr)
    for (int i = 0; i < nnz; i++) matrix->row_ptr[coo_rows[i] + 1]++;
    
    // Pass 2b: Cumulative sum to transform counts into actual row offsets
    for (int i = 0; i < rows; i++) matrix->row_ptr[i + 1] += matrix->row_ptr[i];

    // Pass 2c: Map the COO entries to their specific locations in CSR arrays
    // Use temp_ptr to keep track of the current insertion point for each row
    int *temp_ptr = (int *)malloc(rows * sizeof(int));
    memcpy(temp_ptr, matrix->row_ptr, rows * sizeof(int));
    
    for (int i = 0; i < nnz; i++) {
        int r = coo_rows[i];
        int dest = temp_ptr[r]++; // Move insertion point forward as we fill
        matrix->col_idx[dest] = coo_cols[i];
        matrix->values[dest] = coo_vals[i];
    }

    // Cleanup temporary memory
    free(coo_rows); 
    free(coo_cols); 
    free(coo_vals); 
    free(temp_ptr);
}