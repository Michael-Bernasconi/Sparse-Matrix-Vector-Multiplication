#include <stdio.h>
#include <stdlib.h>
#include "spmv_formats.h"
#include "my_time_lib.h"

/**
 * CPU Implementation of Sparse Matrix-Vector Multiplication (SpMV) using COO format.
 * Computes y = A * x, where A is in Coordinate (COO) format.
 */
void spmv_coo_cpu(const COOMatrix *mat, const float *x, float *y) {
    // 1. Initialize the output vector y with zeros
    for (int i = 0; i < mat->M; i++) y[i] = 0.0f;
    
    // 2. Iterate through all Non-Zero (nnz) elements
    for (int i = 0; i < mat->nnz; i++) {
        // Accumulate the product: y[row] += value * x[column]
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

int main(int argc, char **argv) {
    // Check if a matrix file path was provided
    if (argc < 2) return 1;

    // Load matrix from file into CSR format temporarily
    CSRMatrix temp;
    load_matrix_market_to_csr(argv[1], &temp);
    
    /**
     * Conversion CSR -> COO
     * CSR stores row offsets, but COO needs the explicit row index for every element.
     */
    COOMatrix mat = {temp.M, temp.N, temp.nnz};
    mat.rows = (int*)malloc(mat.nnz * sizeof(int));
    mat.cols = temp.col_idx;  // Share pointer with temp CSR
    mat.values = temp.values; // Share pointer with temp CSR
    
    // Decompress row_ptr from CSR into explicit row indices for COO
    for(int i = 0; i < temp.M; i++) {
        for(int j = temp.row_ptr[i]; j < temp.row_ptr[i+1]; j++) {
            mat.rows[j] = i;
        }
    }

    // Allocate memory for the input vector x and result vector y
    float *x = (float*)malloc(mat.N * sizeof(float));
    float *y = (float*)malloc(mat.M * sizeof(float));
    
    // Initialize input vector x with 1.0f for testing
    for(int i = 0; i < mat.N; i++) x[i] = 1.0f;

    // Warm-up run: Ensures the CPU cache is primed and libraries are loaded
    spmv_coo_cpu(&mat, x, y);

    // Benchmarking: Run the operation 100 times to get a stable average
    TIMER_DEF(0); 
    TIMER_START(0);
    for(int i = 0; i < 100; i++) {
        spmv_coo_cpu(&mat, x, y);
    }
    TIMER_STOP(0);

    // Calculate performance metrics
    double avg_s = (TIMER_ELAPSED(0) / 1e6) / 100.0; // Average time in seconds
    // GFLOPS = (2 operations per non-zero: 1 multiply + 1 add) / (time * 10^9)
    printf("CPU COO -> Time: %fs, GFLOPS: %f\n", avg_s, (2.0 * mat.nnz) / (avg_s * 1e9));

    // Resource Cleanup
    free(x); 
    free(y); 
    free(mat.rows); 
    free(temp.row_ptr); 
    free(temp.col_idx); 
    free(temp.values);
    
    return 0;
}