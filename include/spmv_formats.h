#ifndef SPMV_FORMATS_H
#define SPMV_FORMATS_H

// --- CONFIG BENCHMARK ---
// Usiamo #ifndef: se il valore non viene passato dal Makefile, 
// allora usa i valori di default (20 e 500).

#ifndef WARMUP_ITERATIONS
    #define WARMUP_ITERATIONS 20
#endif

#ifndef BENCHMARK_ITERATIONS
    #define BENCHMARK_ITERATIONS 500
#endif

// --- STRUTTURE DATI ---

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

// --- FUNZIONI UTILI ---

// Caricamento matrici
void load_matrix_market_to_csr(const char *filename, CSRMatrix *matrix);
void load_matrix_market_to_coo(const char *filename, COOMatrix *matrix);

// Calcoli di performance
double calculate_gflops(int nnz, double avg_time_s);
double calculate_bandwidth(int M, int N, int nnz, double avg_time_s, const char* format);

// Inizializzazione vettori
void fill_random_vector(float *vec, int n);

#ifdef __cplusplus
}
#endif

#endif