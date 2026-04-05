#ifndef SPMV_FORMATS_H
#define SPMV_FORMATS_H

// --- BENCHMARK CONFIGURATION ---
// We use #ifndef to allow the Makefile to override these values.
// If no value is passed during compilation, it falls back to defaults (20 and 500).

#ifndef WARMUP_ITERATIONS
#define WARMUP_ITERATIONS 20 // Initial runs to stabilize hardware and prime caches
#endif

#ifndef BENCHMARK_ITERATIONS
#define BENCHMARK_ITERATIONS 500 // Number of iterations for timing measurements
#endif

// --- DATA STRUCTURES ---

/**
 * Compressed Sparse Row (CSR) format.
 * Efficient for row-wise access patterns.
 */
typedef struct
{
    int M, N, nnz;          // Rows, Columns, and Number of Non-Zero elements
    int *row_ptr, *col_idx; // Row pointers (size M+1) and Column indices (size nnz)
    float *values;          // Non-zero values (size nnz)
} CSRMatrix;

/**
 * Coordinate (COO) format.
 * Simple format using (row, column, value) triplets for each non-zero.
 */
typedef struct
{
    int M, N, nnz;    // Rows, Columns, and Number of Non-Zero elements
    int *rows, *cols; // Row indices and Column indices (both size nnz)
    float *values;    // Non-zero values (size nnz)
} COOMatrix;

#ifdef __cplusplus
extern "C"
{
#endif

    // --- UTILITY FUNCTIONS ---

    /**
     * Matrix Loading Functions
     * Read a .mtx file and populate the respective sparse structures.
     */
    void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);
    void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix);

    /**
     * Performance Calculation Functions
     * Calculate throughput (GFLOPS) and memory bandwidth (GB/s).
     * Calculate time to solution (TTS).
     */
    double calculate_gflops(int nnz, double avg_time_s);
    double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char *format);
    double calculate_tts(double start_time);
    /**
     * Vector Initialization
     * Fills a vector with random floating-point values for SpMV testing.
     */
    void fill_random_vector(float *vec, int n);

    /**
     * Rigorous validation between two result vectors.
     * Returns 1 if valid, 0 if fail.
     */
    void validate_results(const float *h_y_ref, const float *h_y_test, int M);

#ifdef __cplusplus
}
#endif

#endif // SPMV_FORMATS_H