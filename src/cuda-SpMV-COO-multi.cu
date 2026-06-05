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

    double global_start = omp_get_wtime();

    if (rank == 0) {
        load_matrix_market_to_coo(argv[1], &mat);
        M = mat.M; N = mat.N; global_nnz = mat.nnz;
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocazione locale del vettore x. Ogni rank allocherà l'intero N per comodità di
    // indicizzazione (solo indici own e ghost saranno valorizzati). Inizializzato a 0.
    float *h_x = (float*)calloc(N, sizeof(float)); 
    float *h_x_full = NULL;
    float *h_y_ref = NULL;

    if (rank == 0) {
        h_x_full = (float*)malloc(N * sizeof(float));
        h_y_ref = (float*)calloc(M, sizeof(float));
        fill_random_vector(h_x_full, N);

        // Distribuzione degli elementi "own" di X con Modulo 1D
        for (int r = 1; r < size; r++) {
            int count_r = N / size + (r < N % size ? 1 : 0);
            if (count_r > 0) {
                float *buf = (float*)malloc(count_r * sizeof(float));
                int idx = 0;
                for (int i = 0; i < N; i++) {
                    if (i % size == r) buf[idx++] = h_x_full[i];
                }
                MPI_Send(buf, count_r, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
                free(buf);
            }
        }
        // Il Rank 0 copia i propri elementi
        for (int i = 0; i < N; i++) {
            if (i % size == 0) h_x[i] = h_x_full[i];
        }
    } else {
        int count_my = N / size + (rank < N % size ? 1 : 0);
        if (count_my > 0) {
            float *buf = (float*)malloc(count_my * sizeof(float));
            MPI_Recv(buf, count_my, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int idx = 0;
            for (int i = 0; i < N; i++) {
                if (i % size == rank) h_x[i] = buf[idx++];
            }
            free(buf);
        }
    }

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

    // =========================================================================
    // --- DAY 4 & 5: GHOST ENTRIES IDENTIFICATION E SCAMBIO VALORI ---
    // =========================================================================
    int *ghost_cols = (int*)malloc(local_nnz * sizeof(int));
    int local_ghost_count = 0;
    
    int *send_to_rank_counts = (int*)calloc(size, sizeof(int));
    int *recv_from_rank_counts = (int*)calloc(size, sizeof(int));

    // Identificazione necessità ghost
    for (int i = 0; i < local_nnz; i++) {
        int col = local_cols[i];
        int owner_rank = col % size; 
        if (owner_rank != rank) {
            int gia_presente = 0;
            for (int j = 0; j < local_ghost_count; j++) {
                if (ghost_cols[j] == col) {
                    gia_presente = 1;
                    break;
                }
            }
            if (!gia_presente) {
                ghost_cols[local_ghost_count++] = col;
                recv_from_rank_counts[owner_rank]++; 
            }
        }
    }

    MPI_Alltoall(recv_from_rank_counts, 1, MPI_INT, send_to_rank_counts, 1, MPI_INT, MPI_COMM_WORLD);

    // Preparazione Array Indici
    int **recv_indices = (int**)malloc(size * sizeof(int*));
    int *recv_idx_pos = (int*)calloc(size, sizeof(int));
    for (int r = 0; r < size; r++) recv_indices[r] = (int*)malloc(recv_from_rank_counts[r] * sizeof(int));
    
    for (int i = 0; i < local_ghost_count; i++) {
        int col = ghost_cols[i];
        int owner_rank = col % size;
        recv_indices[owner_rank][recv_idx_pos[owner_rank]++] = col;
    }
    free(recv_idx_pos);

    int **send_indices = (int**)malloc(size * sizeof(int*));
    for (int r = 0; r < size; r++) send_indices[r] = (int*)malloc(send_to_rank_counts[r] * sizeof(int));

    MPI_Request *reqs = (MPI_Request*)malloc(2 * size * sizeof(MPI_Request));
    int req_count = 0;

    // Scambio Indici
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (recv_from_rank_counts[r] > 0) MPI_Isend(recv_indices[r], recv_from_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
            if (send_to_rank_counts[r] > 0) MPI_Irecv(send_indices[r], send_to_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Scambio Valori
    float **send_values = (float**)malloc(size * sizeof(float*));
    float **recv_values = (float**)malloc(size * sizeof(float*));
    for (int r = 0; r < size; r++) {
        send_values[r] = (float*)malloc(send_to_rank_counts[r] * sizeof(float));
        recv_values[r] = (float*)malloc(recv_from_rank_counts[r] * sizeof(float));
    }

    req_count = 0;
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (send_to_rank_counts[r] > 0) {
                for (int k = 0; k < send_to_rank_counts[r]; k++) {
                    send_values[r][k] = h_x[send_indices[r][k]];
                }
                MPI_Isend(send_values[r], send_to_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            if (recv_from_rank_counts[r] > 0) {
                MPI_Irecv(recv_values[r], recv_from_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Integrazione valori ricevuti in X
    for (int r = 0; r < size; r++) {
        if (r != rank && recv_from_rank_counts[r] > 0) {
            for (int k = 0; k < recv_from_rank_counts[r]; k++) {
                h_x[recv_indices[r][k]] = recv_values[r][k];
            }
        }
    }

    if (rank == 0) {
        printf("\n=== [DAY 5 DIAGNOSTIC - COO] ===\n");
        printf("I valori Ghost di X sono stati scambiati con successo tramite p2p.\n");
        printf("=================================\n\n");
    }

    for (int r = 0; r < size; r++) {
        free(recv_indices[r]); free(send_indices[r]);
        free(send_values[r]); free(recv_values[r]);
    }
    free(recv_indices); free(send_indices); free(send_values); free(recv_values); free(reqs);
    free(send_to_rank_counts); free(recv_from_rank_counts); free(ghost_cols);
    // =========================================================================

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    int *d_rows, *d_cols;
    float *d_values, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_rows, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_rows, local_rows, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, local_values, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int num_iterations = 100;
    double start_time = omp_get_wtime();

    int block_size = 256;
    int grid_size = (local_nnz + block_size - 1) / block_size;

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_coo_kernel<<<grid_size, block_size>>>(local_nnz, d_rows, d_cols, d_values, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double end_time = omp_get_wtime();
    double avg_time_s = (end_time - start_time) / num_iterations;

    double max_avg_time_s;
    MPI_Reduce(&avg_time_s, &max_avg_time_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================================================================
    // --- GIORNO 6: REFACTORING IN MPI "GPU-AWARE" ---
    // Eliminiamo h_local_y e passiamo direttamente i puntatori device d_y e d_global_y_gpu.
    // =========================================================================
    float *d_global_y_gpu = NULL;
    if (rank == 0) {
        CUDA_CHECK(cudaMalloc(&d_global_y_gpu, M * sizeof(float)));
    }
    
    // Eseguiamo la riduzione direttamente sulla GPU sfruttando CUDA-Aware MPI
    MPI_Reduce(d_y, d_global_y_gpu, M, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(h_global_y_gpu, d_global_y_gpu, M * sizeof(float), cudaMemcpyDeviceToHost));
    }
    // =========================================================================

    if (rank == 0) {
        memset(h_y_ref, 0, M * sizeof(float));
        spmv_coo_sequential(&mat, h_x_full, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU COO ( %d GPUs - Modulo 1D - GPU-Aware ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, global_nnz);
        printf("Avg Time: %e s\n", max_avg_time_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(global_nnz, max_avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, global_nnz, max_avg_time_s, "COO"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(h_x_full);
        free(flat_rows); free(flat_cols); free(flat_values); free(rank_nnz);
        free(send_counts_nnz); free(displs_nnz);
        CUDA_CHECK(cudaFree(d_global_y_gpu));
    }

    CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(local_rows); free(local_cols); free(local_values); free(h_x);

    MPI_Finalize();
    return 0;
}