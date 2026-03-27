#ifndef SPMV_FORMATS_H
#define SPMV_FORMATS_H

/**
 * Structure for Compressed Sparse Row (CSR) matrix representation.
 * The CSR format represents a sparse matrix by storing only non-zero elements.
 * It is highly efficient for row-wise operations and the standard for SpMV.
 */
typedef struct {
    int M;           /* Number of rows in the matrix */
    int N;           /* Number of columns in the matrix */
    int nnz;         /* Number of non-zero (NNZ) elements */
    
    int *row_ptr;    /* Row pointers: array of size (M + 1). 
                        row_ptr[i] stores the index in col_idx and values where row i starts. */
    
    int *col_idx;    /* Column indices: array of size (nnz). 
                        Stores the column index for each non-zero element. */
    
    float *values;   /* Non-zero values: array of size (nnz). 
                        Stores the numerical value of each non-zero element. */
} CSRMatrix;

/*
 * Structure for a dense vector.
 * Used for the input vector x and the output vector y in the SpMV operation (y = Ax).
 */
typedef struct {
    int size;        /* Number of elements in the vector */
    float *data;     /* Pointer to the array of floating-point values */
} Vector;

// In fondo a include/spmv_formats.h
#ifdef __cplusplus
extern "C" {
#endif

void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);

#ifdef __cplusplus
}
#endif
#endif