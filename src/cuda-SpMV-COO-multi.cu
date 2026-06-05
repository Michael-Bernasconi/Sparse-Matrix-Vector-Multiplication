#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <omp.h>

extern "C" {
    #include "spmv_formats.h"
    #include "my_time_lib.h"
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

void spmv_coo_sequential(const COOMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

// COO Kernel: Threads can write to anywhere in the vector y, requiring atomicAdd
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols, const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&y[rows[i]], vals[i] * x[cols[i]]);
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix.mtx>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int M, N, global_nnz;
    COOMatrix mat;
    float *h_x = NULL;
    float *h_y_ref = NULL;

    double global_start = get_time();

    if (rank == 0) {
        load_mtx_coo(argv[1], &mat);
        M = mat.M; N = mat.N; global_nnz = mat.nnz;
        h_x = (float*)malloc(N * sizeof(float));
        h_y_ref = (float*)calloc(M, sizeof(float));
        fill_random_vector(h_x, N);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) h_x = (float*)malloc(N * sizeof(float));
    MPI_Bcast(h_x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- DAY 2: Modulo 1D Partitioning Setup (Filtered on row index) ---
    int local_M = M / size + (rank < M % size ? 1 : 0);
    int local_nnz = 0;

    int *send_counts_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    float *flat_values = NULL;
    int *flat_rows = NULL;
    int *flat_cols = NULL;
    int *rank_nnz = (rank == 0) ? (int*)calloc(size, sizeof(int)) : NULL;

    if (rank == 0) {
        for (int i = 0; i < global_nnz; i++) {
            int target_rank = mat.rows[i] % size;
            rank_nnz[target_rank]++;
        }

        float **rank_values_bufs = (float**)malloc(size * sizeof(float*));
        int **rank_rows_bufs = (int**)malloc(size * sizeof(int*));
        int **rank_cols_bufs = (int**)malloc(size * sizeof(int*));

        for (int r = 0; r < size; r++) {
            rank_values_bufs[r] = (float*)malloc(rank_nnz[r] * sizeof(float));
            rank_rows_bufs[r] = (int*)malloc(rank_nnz[r] * sizeof(int));
            rank_cols_bufs[r] = (int*)malloc(rank_nnz[r] * sizeof(int));
        }

        int *rank_curr_nnz = (int*)calloc(size, sizeof(int));
        for (int i = 0; i < global_nnz; i++) {
            int r = mat.rows[i] % size;
            int idx = rank_curr_nnz[r]++;
            rank_values_bufs[r][idx] = mat.values[i];
            rank_rows_bufs[r][idx] = mat.rows[i]; 
            rank_cols_bufs[r][idx] = mat.cols[i];
        }

        int total_nnz_alloc = 0;
        for (int r = 0; r < size; r++) {
            send_counts_nnz[r] = rank_nnz[r];
            displs_nnz[r] = (r == 0) ? 0 : displs_nnz[r - 1] + send_counts_nnz[r - 1];
            total_nnz_alloc += send_counts_nnz[r];
        }

        flat_values = (float*)malloc(total_nnz_alloc * sizeof(float));
        flat_rows = (int*)malloc(total_nnz_alloc * sizeof(int));
        flat_cols = (int*)malloc(total_nnz_alloc * sizeof(int));

        for (int r = 0; r < size; r++) {
            memcpy(flat_values + displs_nnz[r], rank_values_bufs[r], rank_nnz[r] * sizeof(float));
            memcpy(flat_rows + displs_nnz[r], rank_rows_bufs[r], rank_nnz[r] * sizeof(int));
            memcpy(flat_cols + displs_nnz[r], rank_cols_bufs[r], rank_nnz[r] * sizeof(int));
            free(rank_values_bufs[r]); free(rank_rows_bufs[r]); free(rank_cols_bufs[r]);
        }
        free(rank_values_bufs); free(rank_rows_bufs); free(rank_cols_bufs);
        free(rank_curr_nnz);
    }

    MPI_Scatter(rank_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_rows = (int*)malloc(local_nnz * sizeof(int));
    int *local_cols = (int*)malloc(local_nnz * sizeof(int));
    float *local_values = (float*)malloc(local_nnz * sizeof(float));

    MPI_Scatterv(flat_rows, send_counts_nnz, displs_nnz, MPI_INT, local_rows, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_cols, send_counts_nnz, displs_nnz, MPI_INT, local_cols, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_values, send_counts_nnz, displs_nnz, MPI_FLOAT, local_values, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    int *d_rows, *d_cols;
    float *d_values, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_rows, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    // IMPORTANT: COO output array is sized M because scattered NNZs can land on any row
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_rows, local_rows, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(0)); // Context init
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, local_values, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int num_iterations = 100;
    double start_time = get_time();

    int block_size = 256;
    int grid_size = (local_nnz + block_size - 1) / block_size;

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_coo_kernel<<<grid_size, block_size>>>(local_nnz, d_rows, d_cols, d_values, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double end_time = get_time();
    double avg_time_s = (end_time - start_time) / num_iterations;

    double max_avg_time_s;
    MPI_Reduce(&avg_time_s, &max_avg_time_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    float *h_local_y = (float*)malloc(M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    
    // =========================================================================
    // --- DAY 3: GATHER FOR COO (Special Case) ---
    // Note: Since COO kernel writes directly to the correct global row index using
    // atomicAdd, the local result is already "un-shuffled" but sparse. 
    // We simply sum the overlapping partial sparse vectors with MPI_Reduce.
    // =========================================================================
    MPI_Reduce(h_local_y, h_global_y_gpu, M, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Task 2: Testing Phase
        memset(h_y_ref, 0, M * sizeof(float));
        spmv_coo_sequential(&mat, h_x, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU COO ( %d GPUs - Modulo 1D ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, global_nnz);
        printf("Avg Time: %e s\n", max_avg_time_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(global_nnz, max_avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, global_nnz, max_avg_time_s, "COO"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref);
        free(flat_rows); free(flat_cols); free(flat_values); free(rank_nnz);
        free(send_counts_nnz); free(displs_nnz);
    }

    CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(local_rows); free(local_cols); free(local_values); free(h_local_y); free(h_x);

    MPI_Finalize();
    return 0;
}