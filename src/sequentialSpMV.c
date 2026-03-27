#include <stdio.h>
#include <stdlib.h>   
#include <string.h>    
#include <unistd.h>    
#include "my_time_lib.h"
#include "spmv_formats.h"


/**
 * @brief Sequential CSR SpMV: y = A * x
 */
void spmv_csr_cpu(const CSRMatrix *A, const Vector *x, Vector *y) {
    for (int i = 0; i < A->M; i++) {
        float sum = 0.0f;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
            sum += A->values[j] * x->data[A->col_idx[j]];
        }
        y->data[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    CSRMatrix A;
    printf("Loading matrix: %s...\n", argv[1]);
    load_matrix_market_to_csr(argv[1], &A); 

    printf("Matrix details: %d rows, %d cols, %d nnz\n", A.M, A.N, A.nnz);

    Vector x, y;
    x.size = A.N;
    x.data = (float *)malloc(x.size * sizeof(float));
    for (int i = 0; i < x.size; i++) x.data[i] = 1.0f;

    y.size = A.M;
    y.data = (float *)calloc(y.size, sizeof(float));

    TIMER_DEF(timer);
    TIMER_START(timer);
    spmv_csr_cpu(&A, &x, &y);
    TIMER_STOP(timer);
    
    printf("CPU ");
    TIMER_PRINT(timer);

    free(A.row_ptr); free(A.col_idx); free(A.values);
    free(x.data); free(y.data);
    return 0;
}