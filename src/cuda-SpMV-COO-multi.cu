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

/**
 * @brief Macro for runtime validation of CUDA API calls.
 * Aborts the entire MPI execution context if a CUDA error is detected.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// ---------------------------------------------------------
// GHOST EXCHANGE SUPPORT KERNELS
// ---------------------------------------------------------

/**
 * @brief Packs required vector elements into a contiguous buffer before transmission.
 * @param d_x Vector containing localized values.
 * @param d_indices Map of required remote indices.
 * @param d_values Destination buffer for outbound ghost elements.
 * @param count Number of elements to pack.
 */
__global__ void pack_ghost_kernel(const float *d_x, const int *d_indices, float *d_values, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        d_values[i] = d_x[d_indices[i]];
    }
}

/**
 * @brief Unpacks received ghost vector elements into their designated positions in the local vector.
 * @param d_values Source buffer containing received ghost values.
 * @param d_indices Map of incoming index destinations.
 * @param d_x Target local vector to update.
 * @param count Number of elements to unpack.
 */
__global__ void unpack_ghost_kernel(const float *d_values, const int *d_indices, float *d_x, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        d_x[d_indices[i]] = d_values[i];
    }
}

/**
 * @brief Sequential Reference SpMV using the COO format for host-side validation.
 */
void spmv_coo_sequential(const COOMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

/**
 * @brief CUDA Parallel SpMV Kernel utilizing Coordinate (COO) format.
 * Employs atomic operations to safely handle potential race conditions on rows.
 */
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

    // --- CUDA Device Initialization and Selection ---
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    // Assigns GPUs using a round-robin strategy based on the local MPI rank
    CUDA_CHECK(cudaSetDevice(rank % device_count));

    // Instantiation of CUDA Events for high-precision, asynchronous timeline tracking
    cudaEvent_t start_comm, stop_comm, start_comp, stop_comp;
    CUDA_CHECK(cudaEventCreate(&start_comm));
    CUDA_CHECK(cudaEventCreate(&stop_comm));
    CUDA_CHECK(cudaEventCreate(&start_comp));
    CUDA_CHECK(cudaEventCreate(&stop_comp));

    int M, N, global_nnz;
    COOMatrix mat;

    double global_start = omp_get_wtime();

    // Rank 0 is responsible for reading the matrix file from storage
    if (rank == 0) {
        load_matrix_market_to_coo(argv[1], &mat);
        M = mat.M; N = mat.N; global_nnz = mat.nnz;
    }

    // Broadcast global dimensions to all processing ranks
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local segments of input vector X
    float *h_x = (float*)calloc(N, sizeof(float)); 
    float *h_x_full = NULL;
    float *h_y_ref = NULL;

    // --- 1D Round-Robin Distribution of Input Vector X ---
    if (rank == 0) {
        h_x_full = (float*)malloc(N * sizeof(float));
        h_y_ref = (float*)calloc(M, sizeof(float));
        fill_random_vector(h_x_full, N);

        // Serialize and distribute components to peer processes
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
        // Self-assign Rank 0 elements
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

    // --- 1D Matrix Partitioning Scheme Setup (Modulo-based Row Distribution) ---
    int local_M = M / size + (rank < M % size ? 1 : 0);
    int local_nnz = 0;

    int *send_counts_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    float *flat_values = NULL;
    int *flat_rows = NULL;
    int *flat_cols = NULL;
    int *rank_nnz = (rank == 0) ? (int*)calloc(size, sizeof(int)) : NULL;

    if (rank == 0) {
        // Evaluate workload mapping by querying row destinations via modulo indexing
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
            rank_rows_bufs[r][idx] = mat.rows[i] / size; // Scale to local coordinate system
            rank_cols_bufs[r][idx] = mat.cols[i];
        }

        int total_nnz_alloc = 0;
        for (int r = 0; r < size; r++) {
            send_counts_nnz[r] = rank_nnz[r];
            displs_nnz[r] = (r == 0) ? 0 : displs_nnz[r - 1] + send_counts_nnz[r - 1];
            total_nnz_alloc += send_counts_nnz[r];
        }

        // Flatten data buffers for unified MPI collective execution
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

    // Distribute structural element metadata sizes to all ranks
    MPI_Scatter(rank_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_rows = (int*)malloc(local_nnz * sizeof(int));
    int *local_cols = (int*)malloc(local_nnz * sizeof(int));
    float *local_values = (float*)malloc(local_nnz * sizeof(float));

    // Scatter partitioned sparse matrix matrices arrays across the network
    MPI_Scatterv(flat_rows, send_counts_nnz, displs_nnz, MPI_INT, local_rows, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_cols, send_counts_nnz, displs_nnz, MPI_INT, local_cols, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(flat_values, send_counts_nnz, displs_nnz, MPI_FLOAT, local_values, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int min_nnz, max_nnz, sum_nnz;
    MPI_Reduce(&local_nnz, &min_nnz, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &max_nnz, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_nnz, &sum_nnz, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    float avg_nnz = (float)sum_nnz / size;

    // --- Allocate Input Vector X on Device to Support Early Ghost Extraction ---
    float *d_x;
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // =========================================================================
    // --- GHOST VECTOR EXCHANGE VIA GPU-AWARE MPI POINT-TO-POINT ---
    // =========================================================================
    int *ghost_cols = (int*)malloc(local_nnz * sizeof(int));
    int local_ghost_count = 0;
    
    int *send_to_rank_counts = (int*)calloc(size, sizeof(int));
    int *recv_from_rank_counts = (int*)calloc(size, sizeof(int));

    // Identify non-local vector components (ghost indices) required by this rank
    for (int i = 0; i < local_nnz; i++) {
        int col = local_cols[i];
        int owner_rank = col % size; 
        if (owner_rank != rank) {
            int gia_presente = 0;
            for (int j = 0; j < local_ghost_count; j++) {
                if (ghost_cols[j] == col) {
                    gia_presente = 1; break;
                }
            }
            if (!gia_presente) {
                ghost_cols[local_ghost_count++] = col;
                recv_from_rank_counts[owner_rank]++; 
            }
        }
    }

    // Interchange structural data shapes to synchronize traffic requirements
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

    // Exchange internal layout metadata indices via non-blocking operations
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (recv_from_rank_counts[r] > 0) MPI_Isend(recv_indices[r], recv_from_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
            if (send_to_rank_counts[r] > 0) MPI_Irecv(send_indices[r], send_to_rank_counts[r], MPI_INT, r, 100, MPI_COMM_WORLD, &reqs[req_count++]);
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Device Pointer allocations dedicated to GPU-Aware MPI data streaming
    float **d_send_values = (float**)malloc(size * sizeof(float*));
    float **d_recv_values = (float**)malloc(size * sizeof(float*));
    int **d_send_indices = (int**)malloc(size * sizeof(int*));
    int **d_recv_indices = (int**)malloc(size * sizeof(int*));

    // START COMMUNICATION TIMER (Encompasses device setups, packing, MPI transfers, and unpacking)
    CUDA_CHECK(cudaEventRecord(start_comm));

    for (int r = 0; r < size; r++) {
        if (send_to_rank_counts[r] > 0) {
            CUDA_CHECK(cudaMalloc(&d_send_values[r], send_to_rank_counts[r] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_send_indices[r], send_to_rank_counts[r] * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_send_indices[r], send_indices[r], send_to_rank_counts[r] * sizeof(int), cudaMemcpyHostToDevice));
            
            // Execute on-device vector packing from d_x via spatial indexing maps
            int threads = 256;
            int blocks = (send_to_rank_counts[r] + threads - 1) / threads;
            pack_ghost_kernel<<<blocks, threads>>>(d_x, d_send_indices[r], d_send_values[r], send_to_rank_counts[r]);
        }
        if (recv_from_rank_counts[r] > 0) {
            CUDA_CHECK(cudaMalloc(&d_recv_values[r], recv_from_rank_counts[r] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_recv_indices[r], recv_from_rank_counts[r] * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_recv_indices[r], recv_indices[r], recv_from_rank_counts[r] * sizeof(int), cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize()); // Validate packing completion prior to network injection

    req_count = 0;
    for (int r = 0; r < size; r++) {
        if (r != rank) {
            if (send_to_rank_counts[r] > 0) {
                // Pass DEVICE addresses directly to GPU-Aware MPI library handles
                MPI_Isend(d_send_values[r], send_to_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
            if (recv_from_rank_counts[r] > 0) {
                // Pass DEVICE addresses directly to GPU-Aware MPI library handles
                MPI_Irecv(d_recv_values[r], recv_from_rank_counts[r], MPI_FLOAT, r, 200, MPI_COMM_WORLD, &reqs[req_count++]);
            }
        }
    }
    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    // Unpack received ghost allocations directly into global vector array targets on device
    for (int r = 0; r < size; r++) {
        if (r != rank && recv_from_rank_counts[r] > 0) {
            int threads = 256;
            int blocks = (recv_from_rank_counts[r] + threads - 1) / threads;
            unpack_ghost_kernel<<<blocks, threads>>>(d_recv_values[r], d_recv_indices[r], d_x, recv_from_rank_counts[r]);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // STOP COMMUNICATION TIMER
    CUDA_CHECK(cudaEventRecord(stop_comm));
    CUDA_CHECK(cudaEventSynchronize(stop_comm));
    float ms_comm = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_comm, start_comm, stop_comm));

    int total_elements_sent = 0;
    int total_elements_recv = 0;
    for (int r = 0; r < size; r++) {
        total_elements_sent += send_to_rank_counts[r];
        total_elements_recv += recv_from_rank_counts[r];
    }
    
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            printf("[Rank %d] Volume of comunication Ghost: %d send elements (%zu byte), %d receive elements (%zu byte)\n", 
                   rank, 
                   total_elements_sent, total_elements_sent * sizeof(float), 
                   total_elements_recv, total_elements_recv * sizeof(float));
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("\n=== [DIAGNOSTIC - COO] ===\n");
        printf("Ghost vector components interchanged successfully using P2P GPU-AWARE MPI.\n");
        printf("=================================\n\n");
    }

    // Clean up auxiliary ghost-exchange data structures
    for (int r = 0; r < size; r++) {
        free(recv_indices[r]); free(send_indices[r]);
        if (send_to_rank_counts[r] > 0) {
            CUDA_CHECK(cudaFree(d_send_values[r]));
            CUDA_CHECK(cudaFree(d_send_indices[r]));
        }
        if (recv_from_rank_counts[r] > 0) {
            CUDA_CHECK(cudaFree(d_recv_values[r]));
            CUDA_CHECK(cudaFree(d_recv_indices[r]));
        }
    }
    free(recv_indices); free(send_indices); free(reqs);
    free(d_send_values); free(d_recv_values); free(d_send_indices); free(d_recv_indices);
    free(send_to_rank_counts); free(recv_from_rank_counts); free(ghost_cols);
    // =========================================================================

    int *d_rows, *d_cols;
    float *d_values, *d_y;

    CUDA_CHECK(cudaMalloc(&d_rows, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, local_M * sizeof(float))); 

    CUDA_CHECK(cudaMemcpy(d_rows, local_rows, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, local_cols, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, local_values, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    // NOTE: d_x is already allocated and synchronized above

    int num_iterations = 100;
    double start_time = omp_get_wtime();

    int block_size = 256;
    int grid_size = (local_nnz + block_size - 1) / block_size;

    // START COMPUTATION KERNEL TIMER
    CUDA_CHECK(cudaEventRecord(start_comp));

    for (int iter = 0; iter < num_iterations; iter++) {
        CUDA_CHECK(cudaMemset(d_y, 0, local_M * sizeof(float)));
        spmv_coo_kernel<<<grid_size, block_size>>>(local_nnz, d_rows, d_cols, d_values, d_x, d_y);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // STOP COMPUTATION KERNEL TIMER
    CUDA_CHECK(cudaEventRecord(stop_comp));
    CUDA_CHECK(cudaEventSynchronize(stop_comp));
    float ms_comp = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_comp, start_comp, stop_comp));

    double end_time = omp_get_wtime();
    double avg_time_s = (end_time - start_time) / num_iterations;

    double max_avg_time_s;
    MPI_Reduce(&avg_time_s, &max_avg_time_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Compute metric scales across multi-GPU network topologies via reductions
    double local_comm_s = ms_comm / 1000.0;
    double local_comp_s = (ms_comp / 1000.0) / num_iterations; // Normalized per iteration loop
    double max_comm_s, max_comp_s;
    MPI_Reduce(&local_comm_s, &max_comm_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp_s, &max_comp_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // =========================================================================
    // --- RESULT CONSOLIDATION VIA GPU-AWARE MPI_GATHERV ---
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
    
    // Perform data reduction passing raw DEVICE pointers directly into the MPI context
    MPI_Gatherv(d_y, local_M, MPI_FLOAT, d_gather_buf, recv_counts, recv_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        CUDA_CHECK(cudaMemcpy(gather_buf, d_gather_buf, M * sizeof(float), cudaMemcpyDeviceToHost));

        // Unshuffle the 1D modulo block interleaved patterns into linear continuous rows
        int *rank_offset = (int*)calloc(size, sizeof(int));
        for (int i = 0; i < M; i++) {
            int r = i % size;
            int buf_pos = recv_displs[r] + rank_offset[r]++;
            h_global_y_gpu[i] = gather_buf[buf_pos];
        }
        free(rank_offset);

        // Verification and diagnostic reporting pipelines
        memset(h_y_ref, 0, M * sizeof(float));
        spmv_coo_sequential(&mat, h_x_full, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU COO ( %d GPUs - Modulo 1D - GPU-Aware ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, global_nnz);
        printf("Load Bal: NNZ Min: %d | NNZ Avg: %.2f | NNZ Max: %d\n", min_nnz, avg_nnz, max_nnz);
        printf("Avg Time: %e s\n", max_avg_time_s);
        printf("  ├─ Comm Time: %e s (Ghost Exchange Setup & Comm)\n", max_comm_s);
        printf("  └─ Comp Time: %e s (Pure Kernel Computation)\n", max_comp_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(global_nnz, max_avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, global_nnz, max_avg_time_s, "COO"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(h_x_full);
        free(flat_rows); free(flat_cols); free(flat_values); free(rank_nnz);
        free(send_counts_nnz); free(displs_nnz);
        free(recv_counts); free(recv_displs); free(gather_buf);
        CUDA_CHECK(cudaFree(d_gather_buf));
    }

    // Explicit destruction of performance analysis structures
    CUDA_CHECK(cudaEventDestroy(start_comm));
    CUDA_CHECK(cudaEventDestroy(stop_comm));
    CUDA_CHECK(cudaEventDestroy(start_comp));
    CUDA_CHECK(cudaEventDestroy(stop_comp));

    // Cleanup memory spaces allocated on devices and hosts
    CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_cols));
    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(local_rows); free(local_cols); free(local_values); free(h_x);

    MPI_Finalize();
    return 0;
}