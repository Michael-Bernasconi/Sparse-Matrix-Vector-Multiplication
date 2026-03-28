#ifndef SPMV_FORMATS_H
#define SPMV_FORMATS_H

/**
 * Compressed Sparse Row (CSR) Matrix Format.
 * Highly efficient for row-based access and matrix-vector multiplication.
 */
typedef struct {
    int M, N;       // Dimensions of the matrix (M rows, N columns)
    int nnz;        // Number of Non-Zero elements
    int *row_ptr;   // Row pointers: array of size M+1 (offsets into col_idx/values)
    int *col_idx;   // Column indices of non-zero elements: array of size nnz
    float *values;  // Actual values of non-zero elements: array of size nnz
} CSRMatrix;

/**
 * Coordinate (COO) Matrix Format.
 * Simple format often used as an intermediate step for loading data.
 */
typedef struct {
    int M, N;       // Dimensions of the matrix (M rows, N columns)
    int nnz;        // Number of Non-Zero elements
    int *rows;      // Row index for every non-zero element: array of size nnz
    int *cols;      // Column index for every non-zero element: array of size nnz
    float *values;  // Actual values of non-zero elements: array of size nnz
} COOMatrix;

/* --- C Linkage Wrapper for C++ Compatibility --- */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Loads a matrix from a Matrix Market (.mtx) file and converts it to CSR format.
 * @param filename Path to the .mtx file.
 * @param matrix   Pointer to a CSRMatrix struct to be populated.
 */
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);

#ifdef __cplusplus
}
#endif

#endif