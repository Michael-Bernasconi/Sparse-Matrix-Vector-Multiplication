#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "spmv_formats.h"
#include <omp.h>
#include <math.h>

/**
 * Reproducibility: Fills a vector with random float values using a fixed seed.
 * This ensures that the input vector x is identical across CPU and GPU tests.
 */
void fill_random_vector(float *vec, int n)
{
    srand(42); // Fixed seed for scientific reproducibility
    for (int i = 0; i < n; i++)
    {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

/**
 * METRIC 1: FLOPS (Floating Point Operations Per Second)
 * Measures the raw computational throughput.
 * SpMV performs 2 operations (1 multiply, 1 add) per non-zero element.
 */
double calculate_gflops(int nnz, double avg_time_s)
{
    if (avg_time_s <= 0)
        return 0;
    double total_flops = 2.0 * (double)nnz;
    return total_flops / (avg_time_s * 1e9);
}

/**
 * METRIC 2: EFFECTIVE BANDWIDTH
 * Measures the data transfer rate between memory and processors.
 * Calculation accounts for:
 * - Matrix structure (indices and pointers)
 * - Matrix values
 * - Input vector x and output vector y
 */
double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char *format)
{
    if (avg_time_s <= 0)
        return 0;
    size_t bytes = 0;
    if (strcmp(format, "CSR") == 0)
    {
        // CSR: row_ptr (M+1 ints), col_idx (nnz ints), values (nnz floats)
        // + x vector (N floats) + y vector (M floats)
        bytes = (sizeof(int) * (M + 1 + nnz)) + (sizeof(float) * (nnz + N + M));
    }
    else
    {
        // COO: rows (nnz ints), cols (nnz ints), values (nnz floats)
        // + x vector (N floats) + y vector (M floats)
        bytes = (sizeof(int) * (2 * nnz)) + (sizeof(float) * (nnz + N + M));
    }
    return (double)bytes / (avg_time_s * 1e9);
}

/**
 * METRIC 3: TIME TO SOLUTION
 * Calculates the total Time-to-Solution (TTS).
 * TTS measures the "wall-clock" time from the beginning of the program
 * (including I/O and memory allocation) until the completion of the task.
 * start_time = The timestamp recorded at the very start of main().
 * return The elapsed time in seconds.
 */
double calculate_tts(double start_time)
{
    return omp_get_wtime() - start_time;       // Returns current wall-clock time minus the initial timestamp
}

/**
 * Validates the SpMV results by comparing the test vector against a reference vector.
 * This function uses both Absolute and Relative Error validation.
 * In parallel architectures (GPU, cuSPARSE, OpenMP), floating-point operations
 * are non-associative; changing the summation order leads to different rounding.
 * For large values, an absolute epsilon is too strict, while for
 * values near zero, a pure relative error would cause division by zero.
 * ref =  The "Gold Standard" result (sequential CPU CSR).
 * test = The result to be validated (GPU or optimized kernel).
 * n  =   The number of elements in the vectors (M rows).
 */
void validate_results(const float *ref, const float *test, int n)
{
    int errors = 0;
    const float rel_tolerance = 1e-3f; // 0.1% relative tolerance
    const float abs_tolerance = 1e-4f; // Absolute tolerance guard for values near zero
    for (int i = 0; i < n; i++)
    {
        float diff = fabsf(ref[i] - test[i]);
        float abs_ref = fabsf(ref[i]);

        // Calculate the relative error.
        // Adding 1e-7f safely prevents division by zero for extremely small or zero values.
        float rel_diff = diff / (abs_ref + 1e-7f);

        // Validation check: it fails ONLY IF the difference exceeds BOTH absolute AND relative tolerances.
        if (diff > abs_tolerance && rel_diff > rel_tolerance)
        {
            if (errors < 5)
            {
                printf("Validation Error at index %d: Ref %f, Test %f (Abs Diff: %e, Rel Diff: %e)\n",
                       i, ref[i], test[i], diff, rel_diff);
            }
            errors++;
        }
    }
    if (errors == 0)
    {
        // Log dynamically reports the exact variables used, avoiding hardcoded text mismatch.
        printf(">>> VALIDATION PASSED: All %d elements are within the tolerance (abs: %g, rel: %g).\n",
               n, abs_tolerance, rel_tolerance);
    }
    else
    {
        printf(">>> VALIDATION FAILED: %d total errors found out of %d elements.\n", errors, n);
    }
}

/**
 * Loads a Matrix Market file (.mtx) and converts it to CSR format.
 * This version handles both 'general' and 'symmetric' storage types.
 */
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix) {
    FILE *f = fopen(filename, "r");
    if (!f) { 
        fprintf(stderr, "Error: Could not open file %s\n", filename); 
        exit(1); 
    }

    char line[1024];
    int is_symmetric = 0;

    // 1. Parse the header to detect storage type (general vs symmetric)
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "symmetric")) {
            is_symmetric = 1;
        }
    }

    // 2. Skip comment lines (starting with %)
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    // 3. Read matrix dimensions and number of non-zero elements (nnz)
    int rows, cols, nnz_in_file;
    sscanf(line, "%d %d %d", &rows, &cols, &nnz_in_file);

    // Temporary COO storage allocation
    // If symmetric, we might need up to twice the file's nnz (excluding diagonal)
    int max_nnz = is_symmetric ? nnz_in_file * 2 : nnz_in_file;
    int *coo_rows = (int *)malloc(max_nnz * sizeof(int));
    int *coo_cols = (int *)malloc(max_nnz * sizeof(int));
    float *coo_vals = (float *)malloc(max_nnz * sizeof(float));

    int actual_nnz = 0;
    for (int i = 0; i < nnz_in_file; i++) {
        int r, c;
        float v;
        if (fscanf(f, "%d %d %f", &r, &c, &v) != 3) break;
        
        // Convert from 1-based (Matrix Market) to 0-based indexing
        r--; c--; 

        coo_rows[actual_nnz] = r;
        coo_cols[actual_nnz] = c;
        coo_vals[actual_nnz] = v;
        actual_nnz++;

        // If the matrix is symmetric and the element is off-diagonal,
        // add the mirrored element (c, r, v)
        if (is_symmetric && r != c) {
            coo_rows[actual_nnz] = c;
            coo_cols[actual_nnz] = r;
            coo_vals[actual_nnz] = v;
            actual_nnz++;
        }
    }
    fclose(f);

    // 4. Initialize CSR structure with the actual total number of non-zeros
    matrix->M = rows;
    matrix->N = cols;
    matrix->nnz = actual_nnz;
    matrix->row_ptr = (int *)calloc(rows + 1, sizeof(int));
    matrix->col_idx = (int *)malloc(actual_nnz * sizeof(int));
    matrix->values = (float *)malloc(actual_nnz * sizeof(float));

    // Histogram of non-zeros per row
    for (int i = 0; i < actual_nnz; i++)
        matrix->row_ptr[coo_rows[i] + 1]++;

    // Prefix sum to calculate row_ptr offsets
    for (int i = 0; i < rows; i++)
        matrix->row_ptr[i + 1] += matrix->row_ptr[i];

    // Populate CSR arrays using a temporary pointer to track insertions per row
    int *temp_ptr = (int *)malloc(rows * sizeof(int));
    memcpy(temp_ptr, matrix->row_ptr, rows * sizeof(int));
    for (int i = 0; i < actual_nnz; i++) {
        int r = coo_rows[i];
        int dest = temp_ptr[r]++;
        matrix->col_idx[dest] = coo_cols[i];
        matrix->values[dest] = coo_vals[i];
    }

    // Free temporary COO buffers and auxiliary pointer
    free(coo_rows); 
    free(coo_cols); 
    free(coo_vals); 
    free(temp_ptr);
}
/**
 * Loads a Matrix Market file (.mtx) into COO format.
 * Handles both 'general' and 'symmetric' types.
 */
void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }

    char line[1024];
    int is_symmetric = 0;

    // 1. Detect symmetry from the first header line
    // Usa fgets per leggere la riga di testo
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "symmetric")) {
            is_symmetric = 1;
        }
    }

    // 2. Skip comments
    while (fgets(line, sizeof(line), f) && line[0] == '%');

    // 3. Read dimensions and non-zeros in file
    int M, N, nnz_in_file;
    sscanf(line, "%d %d %d", &M, &N, &nnz_in_file);

    // Initial allocation: for symmetric, we might need up to 2*nnz
    int max_nnz = is_symmetric ? nnz_in_file * 2 : nnz_in_file;
    matrix->rows = (int *)malloc(max_nnz * sizeof(int));
    matrix->cols = (int *)malloc(max_nnz * sizeof(int));
    matrix->values = (float *)malloc(max_nnz * sizeof(float));

    int actual_nnz = 0;
    for (int i = 0; i < nnz_in_file; i++)
    {
        int r, c;
        float val; // Valore in float
        
        // Usa fscanf per leggere i numeri
        if (fscanf(f, "%d %d %f", &r, &c, &val) != 3) break;
        
        r--; c--; // Convert to 0-based indexing

        matrix->rows[actual_nnz] = r;
        matrix->cols[actual_nnz] = c;
        matrix->values[actual_nnz] = val; // Nessun cast necessario
        actual_nnz++;

        // If symmetric and off-diagonal, add the (c, r) pair
        if (is_symmetric && r != c)
        {
            matrix->rows[actual_nnz] = c;
            matrix->cols[actual_nnz] = r;
            matrix->values[actual_nnz] = val; // Nessun cast necessario
            actual_nnz++;
        }
    }

    // 4. Update the structure with the final dimensions and actual NNZ
    matrix->M = M;
    matrix->N = N;
    matrix->nnz = actual_nnz;

    if (actual_nnz < max_nnz) {
        matrix->rows = (int *)realloc(matrix->rows, actual_nnz * sizeof(int));
        matrix->cols = (int *)realloc(matrix->cols, actual_nnz * sizeof(int));
        matrix->values = (float *)realloc(matrix->values, actual_nnz * sizeof(float));
    }

    fclose(f);
}
