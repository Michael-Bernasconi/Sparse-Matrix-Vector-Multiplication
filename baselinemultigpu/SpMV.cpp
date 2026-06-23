#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "include/mmio.h"

void coo_to_csr(int M, int nz, int *I, int *J, double *val,
                int **row_ptr_out, int **col_ind_out, float **values_out) {
    int *row_ptr = (int *)calloc(M + 1, sizeof(int));
    int *col_ind = (int *)malloc(nz * sizeof(int));
    float *values = (float*)malloc(nz * sizeof(float));

    for (int i = 0; i < nz; i++) {
        row_ptr[I[i] + 1]++;
    }
    for (int i = 0; i < M; i++) {
        row_ptr[i + 1] += row_ptr[i];
    }

    int *temp_row_ptr = (int *)malloc((M + 1) * sizeof(int));
    for (int i = 0; i <= M; i++) {
        temp_row_ptr[i] = row_ptr[i];
    }

    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = temp_row_ptr[row];
        col_ind[dest] = J[i];
        values[dest] = (float)val[i];
        temp_row_ptr[row]++;
    }

    free(temp_row_ptr);
    *row_ptr_out = row_ptr;
    *col_ind_out = col_ind;
    *values_out = values;
}

// Sequential Host Baseline (Executed inside MPI Rank 0 context)
void spmv_csr(int rows, const int *row_ptr, const int *col_ind, 
              const float *values, const float *x, float *y) {
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
            y[i] += values[j] * x[col_ind[j]];
        }
    }
}

int main(int argc, char *argv[])
{
    int myid, ntask;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &ntask);

    // Only Rank 0 parses and executes the sequential benchmark
    if (myid == 0) {
        int ret_code;
        MM_typecode matcode;
        FILE *f;
        int M, N, nz;
        int i, *I, *J;
        double *val;

        if (argc < 2) {
            fprintf(stderr, "Usage: %s [matrix-market-filename]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if ((f = fopen(argv[1], "r")) == NULL) {
            fprintf(stderr, "Error opening file: %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (mm_read_banner(f, &matcode) != 0) {
            printf("Could not process Matrix Market banner.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) {
            printf("Sorry, this application does not support complex matrices.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0) {
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        I = (int *) malloc(nz * sizeof(int));
        J = (int *) malloc(nz * sizeof(int));
        val = (double *) malloc(nz * sizeof(double));

        for (i = 0; i < nz; i++) {
            fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
            I[i]--;  
            J[i]--;
        }
        fclose(f);

        int *row_ptr, *col_ind;
        float *values, *x, *y;

        x = (float*)malloc(sizeof(float) * N);
        y = (float*)malloc(sizeof(float) * M);
        for (int i = 0; i < N; i++) x[i] = 1.0f;

        coo_to_csr(M, nz, I, J, val, &row_ptr, &col_ind, &values);

        // --- BENCHMARK EXECUTION CAMPAIGN ---
        int runs = 10;
        double start_time = MPI_Wtime();
        
        for (int r = 0; r < runs; r++) {
            spmv_csr(M, row_ptr, col_ind, values, x, y);
        }
        
        double end_time = MPI_Wtime();
        double avg_time = (end_time - start_time) / runs;

        // Metrics Calculation
        double gflops = (2.0 * nz) / (avg_time * 1e9);
        // Bytes read: row_ptr (M+1)*4 + col_ind (nz)*4 + values (nz)*4 + x vectors (nz)*4 approx.
        // Bytes written: y vector (M)*4
        double bytes_accessed = ((M + 1) * sizeof(int)) + (nz * sizeof(int)) + (nz * sizeof(float)) + (nz * sizeof(float)) + (M * sizeof(float));
        double bw = bytes_accessed / (avg_time * 1e9);

        // Output matching your standard logger layout perfectly
        printf("\n--- PROF BASELINE (Sequential Emulation) ---\n");
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, nz);
        printf("Avg Time: %e s\n", avg_time);
        printf("GFLOPS  : %.4f\n", gflops);
        printf("BW      : %.4f GB/s\n", bw);

        // Free resources
        free(I); free(J); free(val);
        free(row_ptr); free(col_ind); free(values);
        free(x); free(y);
    }

    MPI_Finalize();
    return 0;
}
