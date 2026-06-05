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

// Reference CPU function for validation
void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

// Standard CSR Scalar Kernel
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
    float *h_x = NULL;
    float *h_y_ref = NULL;

    double global_start = get_time();

    if (rank == 0) {
        load_mtx_csr(argv[1], &A); 
        M = A.M; N = A.N; nnz = A.nnz;
        
        h_x = (float*)malloc(N * sizeof(float));
        h_y_ref = (float*)calloc(M, sizeof(float));
        fill_random_vector(h_x, N);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        h_x = (float*)malloc(N * sizeof(float));
    }
    MPI_Bcast(h_x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- DAY 2: Modulo 1D Partitioning Setup ---
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

        // Loop rows with Modulo logic
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

        // Packing local interleaved data
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

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    int *d_row_ptr, *d_col_idx;
    float *d_values, *d_x, *d_y;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (local_M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, local_M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_row_ptr, local_row_ptr, (local_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, local_col_idx, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, local_values, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int num_iterations = 100;
    double start_time = get_time();

    int block_size = 256;
    int grid_size = (local_M + block_size - 1) / block_size;

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemset(d_y, 0, local_M * sizeof(float)));
        spmv_csr_kernel<<<grid_size, block_size>>>(local_M, d_row_ptr, d_col_idx, d_values, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    double end_time = get_time();
    double avg_time_s = (end_time - start_time) / num_iterations;

    double max_avg_time_s;
    MPI_Reduce(&avg_time_s, &max_avg_time_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    float *h_local_y = (float*)malloc(local_M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_y, d_y, local_M * sizeof(float), cudaMemcpyDeviceToHost));

    // =========================================================================
    // --- DAY 3: GATHER & UN-SHUFFLING THE INTERLEAVED RESULT ---
    // Task 1: Reconstruct the global vector from the modulo partitioning
    // =========================================================================
    int *recv_counts = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *recv_displs = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    float *gather_buf = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recv_counts[i] = M / size + (i < M % size ? 1 : 0);
            recv_displs[i] = (i == 0) ? 0 : recv_displs[i - 1] + recv_counts[i - 1];
        }
    }

    // Step 1: Gather the dense local arrays into one big intermediate buffer
    MPI_Gatherv(h_local_y, local_M, MPI_FLOAT, gather_buf, recv_counts, recv_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Step 2: Un-shuffle the gather_buf by distributing back to i % size positions
        int *rank_offset = (int*)calloc(size, sizeof(int));
        for (int i = 0; i < M; i++) {
            int r = i % size;
            int buf_pos = recv_displs[r] + rank_offset[r]++;
            h_global_y_gpu[i] = gather_buf[buf_pos];
        }
        free(rank_offset);

        // Task 2: Validation Testing
        spmv_csr_sequential(&A, h_x, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU CSR SCALAR ( %d GPUs - Modulo 1D ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, nnz);
        printf("Avg Time: %e s\n", max_avg_time_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(nnz, max_avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, nnz, max_avg_time_s, "CSR"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(gather_buf);
        free(recv_counts); free(recv_displs);
        free(flat_row_ptr); free(flat_values); free(flat_col_idx); free(rank_nnz);
        free(send_counts_rows); free(displs_rows); free(send_counts_nnz); free(displs_nnz);
    }

    CUDA_CHECK(cudaFree(d_row_ptr)); CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(local_row_ptr); free(local_values); free(local_col_idx); free(h_local_y); free(h_x);

    MPI_Finalize();
    return 0;
}