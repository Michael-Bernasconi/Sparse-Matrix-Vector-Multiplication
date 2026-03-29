#ifndef SPMV_FORMATS_H
#define SPMV_FORMATS_H

// Common Benchmark Configuration
#define WARMUP_ITERATIONS 20
#define BENCHMARK_ITERATIONS 500

typedef struct {
    int M, N, nnz;
    int *row_ptr, *col_idx;
    float *values;
} CSRMatrix;

typedef struct {
    int M, N, nnz;
    int *rows, *cols;
    float *values;
} COOMatrix;

#ifdef __cplusplus
extern "C" {
#endif

// Loader functions
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);
void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix);

// Utility functions for benchmarking
void fill_random_vector(float *vec, int n);
double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char* format);

#ifdef __cplusplus
}
#endif

#endif