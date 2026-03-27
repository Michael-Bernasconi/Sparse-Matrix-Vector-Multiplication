#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "my_time_lib.h"
#include "spmv_formats.h"

// Kernel CUDA 
__global__ void spmv_csr_kernel(int M, const int* row_ptr, const int* col_idx, const float* values, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M) {
        float sum = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            sum += values[j] * x[col_idx[j]];
        }
        y[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <matrix_market_file>\n", argv[0]);
        return 1;
    }

    CSRMatrix A;
    load_matrix_market_to_csr(argv[1], &A);

    Vector x, y;
    x.size = A.N; x.data = (float*)malloc(x.size * sizeof(float));
    y.size = A.M; y.data = (float*)calloc(y.size, sizeof(float));
    for (int i = 0; i < x.size; i++) x.data[i] = 1.0f;

    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ptr, (A.M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, A.nnz * sizeof(int));
    cudaMalloc(&d_values, A.nnz * sizeof(float));
    cudaMalloc(&d_x, x.size * sizeof(float));
    cudaMalloc(&d_y, y.size * sizeof(float));

    cudaMemcpy(d_row_ptr, A.row_ptr, (A.M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, A.col_idx, A.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, A.values, A.nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data, x.size * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (A.M + blockSize - 1) / blockSize;

    TIMER_DEF(timer);
    TIMER_START(timer);

    spmv_csr_kernel<<<gridSize, blockSize>>>(A.M, d_row_ptr, d_col_idx, d_values, d_x, d_y);
    cudaDeviceSynchronize(); 
    
    TIMER_STOP(timer);
    printf("GPU ");
    TIMER_PRINT(timer);

    cudaMemcpy(y.data, d_y, y.size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_values); cudaFree(d_x); cudaFree(d_y);
    free(A.row_ptr); free(A.col_idx); free(A.values); free(x.data); free(y.data);

    return 0;
}