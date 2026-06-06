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

// --- KERNEL PER GHOST PACKING/UNPACKING SU GPU ---
__global__ void pack_ghost_kernel(const float* d_x, float* d_send_values, const int* d_send_indices, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        d_send_values[i] = d_x[d_send_indices[i]];
    }
}

__global__ void unpack_ghost_kernel(float* d_x, const float* d_recv_values, const int* d_recv_indices, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        d_x[d_recv_indices[i]] = d_recv_values[i];
    }
}

void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

__global__ void spmv_csr_kernel(int num_rows, const int* d_row_ptr, const int* d_col_ind, const float* d_values, const float* d_x, float* d_y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_rows) {
        float sum = 0.0f;
        int row_start = d_row_ptr[i];
        int row_end = d_row_ptr[i+1];
        for (int j = row_start; j < row_end; j++) {
            sum += d_values[j] * d_x[d_col_ind[j]];
        }
        d_y[i] = sum;
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

    int M, N, nnz;
    CSRMatrix A;

    double global_start = omp_get_wtime();

    if (rank == 0) {
        load_matrix_market_to_csr(argv[1], &A);
        M = A.M; N = A.N; nnz = A.nnz;
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *h_x = (float*)calloc(N, sizeof(float));
    float *h_x_full = NULL;
    float *h_y_ref = NULL;

    if (rank == 0) {
        h_x_full = (float*)malloc(N * sizeof(float));
        h_y_ref = (float*)calloc(M, sizeof(float));
        fill_random_vector(h_x_full, N);

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

    int local_M = M / size + (rank < M % size ? 1 : 0);
    int local_nnz = 0;

    int *send_counts_rows = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_rows = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *send_counts_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    float *flat_values = NULL;
    int *flat_col_idx = NULL;
    int *flat_row_ptr = NULL;
    int *rank_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    if (rank == 0) {
        int *rank_M = (int*)malloc(size * sizeof(int));
        int **rank_row_ptr_bufs = (int**)malloc(size * sizeof(int*));
        float **rank_values_bufs = (float**)malloc(size * sizeof(float*));
        int **rank_col_idx_bufs = (int**)malloc(size * sizeof(int*));

        for (int r = 0; r < size; r++) {
            rank_M[r] = M / size + (r < M % size ? 1 : 0);
            rank_row_ptr_bufs[r] = (int*)malloc((rank_M[r] + 1) * sizeof(int));
            rank_row_ptr_bufs[r][0] = 0;
            rank_nnz[r] = 0;
        }

        for (int i = 0; i < M; i++) {
            int target_rank = i % size;
            rank_nnz[target_rank] += (A.row_ptr[i + 1] - A.row_ptr[i]);
        }

        for (int r = 0; r < size; r++) {
            rank_values_bufs[r] = (float*)malloc(rank_nnz[r] * sizeof(float));
            rank_col_idx_bufs[r] = (int*)malloc(rank_nnz[r] * sizeof(int));
        }

        int *rank_curr_row = (int*)calloc(size, sizeof(int));
        int *rank_curr_nnz = (int*)calloc(size, sizeof(int));

        for (int i = 0; i < M; i++) {
            int r = i % size;
            int start = A.row_ptr[i];
            int end = A.row_ptr[i + 1];
            
            for (int j = start; j < end; j++) {
                int idx = rank_curr_nnz[r]++;
                rank_values_bufs[r][idx] = A.values[j];
                rank_col_idx_bufs[r][idx] = A.col_idx[j];
            }
            int row_idx = ++rank_curr_row[r];
            rank_row_ptr_bufs[r][row_idx] = rank_curr_nnz[r];
        }

        int total_rows_alloc = 0;
        int total_nnz_alloc = 0;
        for (int r = 0; r < size; r++) {
            send_counts_rows[r] = rank_M[r] + 1;
            displs_rows[r] = (r == 0) ? 0 : displs_rows[r - 1] + send_counts_rows[r - 1];
            send_counts_nnz[r] = rank_nnz[r];
            displs_nnz[r] = (r == 0) ? 0 : displs_nnz[r - 1] + send_counts_nnz[r - 1];
            total_rows_alloc += send_counts_rows[r];
            total_nnz_alloc += send_counts_nnz[r];
        }

        flat_row_ptr = (int*)malloc(total_rows_alloc * sizeof(int));
        flat_values = (float*)malloc(total_nnz_alloc * sizeof(float));
        flat_col_idx = (int*)malloc(total_nnz_alloc * sizeof(int));

        for (int r = 0; r < size; r++) {
            memcpy(flat_row_ptr + displs_rows[r], rank_row_ptr_bufs[r], (rank_M[r] + 1) * sizeof(int));
            memcpy(flat_values + displs_nnz[r], rank_values_bufs[r], rank_nnz[r] * sizeof(float));
            memcpy(flat_col_idx + displs_nnz[r], rank_col_idx_bufs[r], rank_nnz[r] * sizeof(int));
            free(rank_row_ptr_bufs[r]); free(rank_values_bufs[r]); free(rank_col_idx_bufs[r]);
        }
        free(rank_M); free(rank_row_ptr_bufs); free(rank_values_bufs); free(rank_col_idx_bufs);
        free(rank_curr_row); free(rank_curr_nnz);
    }

    MPI_Scatter(rank_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_row_ptr = (int*)malloc((local_M + 1) * sizeof(int));
    float *local_values = (float*)malloc(local_nnz * sizeof(float));
    int *local_col_idx = (int*)malloc(local_nnz * sizeof(int));

    MPI_Scatterv(flat_row_ptr, send_counts_rows, displs_rows, MPI_INT, local_row_ptr, local_M + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_values, send_counts_nnz, displs_nnz, MPI_FLOAT, local_values, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_col_idx, send_counts_nnz, displs_nnz, MPI_INT, local_col_idx, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    int min_nnz, max_nnz, sum_nnz;
    MPI_Reduce(&local_nnz, &min_nnz, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &max_nnz, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &sum_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    float avg_nnz = (float)sum_nnz / size;

    // =========================================================================
    // --- SETUP DEVICE (SPOSTATO PRIMA DELLO SCAMBIO GHOST) ---
    // =========================================================================
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    // Creazione eventi CUDA per isolare i tempi
    cudaEvent_t start_comm, stop_comm, start_comp, stop_comp;
    CUDA_CHECK(cudaEventCreate(&start_comm));
    CUDA_CHECK(cudaEventCreate(&stop_comm));
    CUDA_CHECK(cudaEventCreate(&start_comp));
    CUDA_CHECK(cudaEventCreate(&stop_comp));

    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // --- DAY 4 & 5: GHOST ENTRIES IDENTIFICATION (SU CPU) ---
    int *ghost_cols = (int*)malloc(local_nnz * sizeof(int));
    int local_ghost_count = 0;
    
    int *send_to_rank_counts = (int*)calloc(size, sizeof(int));
    int *recv_from_rank_counts = (int*)calloc(size, sizeof(int));

    for (int i = 0; i < local_nnz; i++) {
        int col = local_col_idx[i];
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

    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (recv_from_rank_counts[r] > 0) MPI_Isend(recv_indices[r], recv_from_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
            if (send_to_rank_counts[r] > 0) MPI_Irecv(send_indices[r], send_to_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // =========================================================================
    // --- DAY 6: GHOST EXCHANGE VALUES (100% GPU-AWARE MPI) ---
    // =========================================================================
    float **d_send_values = (float**)malloc(size * sizeof(float*));
    float **d_recv_values = (float**)malloc(size * sizeof(float*));
    int **d_send_indices = (int**)malloc(size * sizeof(int*));
    int **d_recv_indices = (int**)malloc(size * sizeof(int*));

    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (send_to_rank_counts[r] > 0) {
                CUDA_CHECK(cudaMalloc(&d_send_values[r], send_to_rank_counts[r] * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_send_indices[r], send_to_rank_counts[r] * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_send_indices[r], send_indices[r], send_to_rank_counts[r] * sizeof(int), cudaMemcpyHostToDevice));
            }
            if (recv_from_rank_counts[r] > 0) {
                CUDA_CHECK(cudaMalloc(&d_recv_values[r], recv_from_rank_counts[r] * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_recv_indices[r], recv_from_rank_counts[r] * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(d_recv_indices[r], recv_indices[r], recv_from_rank_counts[r] * sizeof(int), cudaMemcpyHostToDevice));
            }
        }
    }

    // START TIMER COMUNICAZIONE (Include packing su GPU, MPI p2p e unpacking)
    CUDA_CHECK(cudaEventRecord(start_comm));

    // FASE 1: Packing dei valori da inviare (eseguito in GPU)
    for (int r = 0; r < size; r++) {
        if (r != rank && send_to_rank_counts[r] > 0) {
            int blocks = (send_to_rank_counts[r] + 255) / 256;
            pack_ghost_kernel<<<blocks, 256>>>(d_x, d_send_values[r], d_send_indices[r], send_to_rank_counts[r]);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Sincronizza prima dell'invio MPI

    // FASE 2: Scambio MPI Isend/Irecv passando PUNTATORI DEVICE
    req_count = 0;
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (send_to_rank_counts[r] > 0) {
                MPI_Isend(d_send_values[r], send_to_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            if (recv_from_rank_counts[r] > 0) {
                MPI_Irecv(d_recv_values[r], recv_from_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // FASE 3: Unpacking dei valori ricevuti (eseguito in GPU)
    for (int r = 0; r < size; r++) {
        if (r != rank && recv_from_rank_counts[r] > 0) {
            int blocks = (recv_from_rank_counts[r] + 255) / 256;
            unpack_ghost_kernel<<<blocks, 256>>>(d_x, d_recv_values[r], d_recv_indices[r], recv_from_rank_counts[r]);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Assicura che d_x sia pronto per la SpMV

    // STOP TIMER COMUNICAZIONE
    CUDA_CHECK(cudaEventRecord(stop_comm));
    CUDA_CHECK(cudaEventSynchronize(stop_comm));
    float ms_comm = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_comm, start_comm, stop_comm));

    // --- STAMPA VOLUME DI COMUNICAZIONE PER RANK ---
    int total_elements_sent = 0;
    int total_elements_recv = 0;
    for (int r = 0; r < size; r++) {
        total_elements_sent += send_to_rank_counts[r];
        total_elements_recv += recv_from_rank_counts[r];
    }
    
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("[Rank %d] Volume di comunicazione Ghost: %d elementi inviati (%zu byte), %d elementi ricevuti (%zu byte)\n", 
                   rank, 
                   total_elements_sent, total_elements_sent * sizeof(float), 
                   total_elements_recv, total_elements_recv * sizeof(float));
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    // ---------------------------------------------------------------

    if (rank == 0) {
        printf("\n=== [DAY 6/7 DIAGNOSTIC - CSR SCALAR] ===\n");
        printf("I valori Ghost di X sono stati scambiati con successo tramite MPI GPU-Aware.\n");
        printf("===========================================\n\n");
    }

    // Pulizia delle strutture Host e Device temporanee per i Ghost
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (send_to_rank_counts[r] > 0) {
                CUDA_CHECK(cudaFree(d_send_values[r]));
                CUDA_CHECK(cudaFree(d_send_indices[r]));
            }
            if (recv_from_rank_counts[r] > 0) {
                CUDA_CHECK(cudaFree(d_recv_values[r]));
                CUDA_CHECK(cudaFree(d_recv_indices[r]));
            }
        }
        free(recv_indices[r]); free(send_indices[r]);
    }
    free(recv_indices); free(send_indices); free(reqs);
    free(d_send_values); free(d_recv_values); free(d_send_indices); free(d_recv_indices);
    free(send_to_rank_counts); free(recv_from_rank_counts); free(ghost_cols);

    // =========================================================================
    // --- COMPLETAMENTO SETUP DEVICE ---
    // =========================================================================
    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_y;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (local_M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, local_M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, local_row_ptr, (local_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, local_col_idx, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, local_values, local_nnz * sizeof(float), cudaMemcpyHostToDevice));

    int num_iterations = 100;
    double start_time = omp_get_wtime();

    int block_size = 256;
    int grid_size = (local_M + block_size - 1) / block_size;

    // START TIMER COMPUTAZIONE
    CUDA_CHECK(cudaEventRecord(start_comp));

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemset(d_y, 0, local_M * sizeof(float)));
        spmv_csr_kernel<<<grid_size, block_size>>>(local_M, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // STOP TIMER COMPUTAZIONE
    CUDA_CHECK(cudaEventRecord(stop_comp));
    CUDA_CHECK(cudaEventSynchronize(stop_comp));
    float ms_comp = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_comp, start_comp, stop_comp));

    double end_time = omp_get_wtime();
    double avg_time_s = (end_time - start_time) / num_iterations;

    double max_avg_time_s;
    MPI_Reduce(&avg_time_s, &max_avg_time_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Riduzione dei tempi specifici per Comm e Comp ricavati dagli eventi CUDA
    double local_comm_s = ms_comm / 1000.0;
    double local_comp_s = (ms_comp / 1000.0) / num_iterations; // scalato per numero iterazioni
    double max_comm_s, max_comp_s;
    MPI_Reduce(&local_comm_s, &max_comm_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp_s, &max_comp_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================================================================
    // --- GATHER FINALE GPU-AWARE MPI ---
    // =========================================================================
    int *recv_counts = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *recv_displs = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    float *d_gather_buf = NULL;
    float *gather_buf = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recv_counts[i] = M / size + (i < M % size ? 1 : 0);
            recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
        }
        CUDA_CHECK(cudaMalloc(&d_gather_buf, M * sizeof(float)));
    }

    MPI_Gatherv(d_y, local_M, MPI_FLOAT, d_gather_buf, recv_counts, recv_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(gather_buf, d_gather_buf, M * sizeof(float), cudaMemcpyDeviceToHost));

        int *rank_offset = (int*)calloc(size, sizeof(int));
        for (int i = 0; i < M; i++) {
            int r = i % size;
            int buf_pos = recv_displs[r] + rank_offset[r]++;
            h_global_y_gpu[i] = gather_buf[buf_pos];
        }
        free(rank_offset);

        spmv_csr_sequential(&A, h_x_full, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU CSR SCALAR ( %d GPUs - Modulo 1D - GPU-Aware ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, nnz);
        printf("Load Bal: NNZ Min: %d | NNZ Avg: %.2f | NNZ Max: %d\n", min_nnz, avg_nnz, max_nnz);
        printf("Avg Time: %e s\n", max_avg_time_s);
        printf("  ├─ Comm Time: %e s (Ghost Exchange Setup & Comm)\n", max_comm_s);
        printf("  └─ Comp Time: %e s (Pure Kernel Computation)\n", max_comp_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(nnz, max_avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, nnz, max_avg_time_s, "CSR"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(gather_buf); free(h_x_full);
        free(recv_counts); free(recv_displs);
        free(flat_row_ptr); free(flat_values); free(flat_col_idx); free(rank_nnz);
        free(send_counts_rows); free(displs_rows); free(send_counts_nnz); free(displs_nnz);
        CUDA_CHECK(cudaFree(d_gather_buf));
    }

    // Pulizia eventi
    CUDA_CHECK(cudaEventDestroy(start_comm));
    CUDA_CHECK(cudaEventDestroy(stop_comm));
    CUDA_CHECK(cudaEventDestroy(start_comp));
    CUDA_CHECK(cudaEventDestroy(stop_comp));

    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(local_row_ptr); free(local_values); free(local_col_idx); free(h_x);

    MPI_Finalize();
    return 0;
}