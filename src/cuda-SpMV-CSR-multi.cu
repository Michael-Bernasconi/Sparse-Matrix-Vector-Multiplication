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

// Standard Sequential SpMV for validation (CPU Gold Standard)
void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

// CUDA Kernel with Read-Only Cache hint (__ldg)
__global__ void spmv_csr_kernel(int num_rows, const int* d_row_ptr, const int* d_col_ind, const float* d_values, const float* d_x, float* d_y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        float sum = 0.0f;
        int row_start = d_row_ptr[row];
        int row_end   = d_row_ptr[row + 1];
        for (int i = row_start; i < row_end; i++) {
            sum += __ldg(&d_values[i]) * __ldg(&d_x[d_col_ind[i]]);
        }
        d_y[row] = sum;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double global_start = omp_get_wtime(); // Measure TTS

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix.mtx>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    // 1. GPU Binding
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    CSRMatrix A;
    int M, N, nnz;
    float *h_x = NULL;
    float *h_y_ref = NULL;

    // 2. Load and Data Distribution
    if (rank == 0) {
        load_matrix_market_to_csr(argv[1], &A);
        M = A.M; N = A.N; nnz = A.nnz;
        h_y_ref = (float *)malloc(M * sizeof(float));
    }
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h_x = (float*)malloc(N * sizeof(float));
    if (rank == 0) fill_random_vector(h_x, N);
    MPI_Bcast(h_x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Domain Decomposition
    int local_M = M / size;
    int r_start = rank * local_M;
    if (rank == size - 1) local_M = M - r_start;

    int local_nnz;
    int *send_counts_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    if (rank == 0) {
        for(int i=0; i<size; i++) {
            int start = i * (M/size);
            int end = (i == size-1) ? M : (i+1)*(M/size);
            send_counts_nnz[i] = A.row_ptr[end] - A.row_ptr[start];
            displs_nnz[i] = A.row_ptr[start];
        }
    }
    MPI_Scatter(send_counts_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    float *h_local_val = (float*)malloc(local_nnz * sizeof(float));
    int *h_local_col = (int*)malloc(local_nnz * sizeof(int));
    int *h_local_ptr = (int*)malloc((local_M + 1) * sizeof(int));

    MPI_Scatterv(rank == 0 ? A.values : NULL, send_counts_nnz, displs_nnz, MPI_FLOAT, h_local_val, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.col_idx : NULL, send_counts_nnz, displs_nnz, MPI_INT, h_local_col, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    // Row Pointer Normalization
    if (rank == 0) {
        for(int i=1; i<size; i++) {
            int start = i * (M/size);
            int count = (i == size-1) ? M - start : M/size;
            int offset = A.row_ptr[start];
            int *tmp = (int*)malloc((count+1)*sizeof(int));
            for(int j=0; j<=count; j++) tmp[j] = A.row_ptr[start+j] - offset;
            MPI_Send(tmp, count+1, MPI_INT, i, 0, MPI_COMM_WORLD);
            free(tmp);
        }
        for(int j=0; j<=local_M; j++) h_local_ptr[j] = A.row_ptr[j];
    } else {
        MPI_Recv(h_local_ptr, local_M + 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // 3. GPU Setup
    int *d_ptr, *d_col; float *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_ptr, (local_M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, local_M * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ptr, h_local_ptr, (local_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_local_col, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_local_val, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (local_M + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP ---
    for(int i=0; i<WARMUP_ITERATIONS; i++) {
        spmv_csr_kernel<<<blocks, threads>>>(local_M, d_ptr, d_col, d_val, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // --- BENCHMARK ---
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    for(int i=0; i<BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        spmv_csr_kernel<<<blocks, threads>>>(local_M, d_ptr, d_col, d_val, d_x, d_y);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        iter_times[i] = (double)ms / 1000.0;
    }

    double avg_time_s = arithmetic_mean(iter_times, BENCHMARK_ITERATIONS);
    double std_dev_s = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // 4. Gather and Validation
    float *h_local_y = (float*)malloc(local_M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_y, d_y, local_M * sizeof(float), cudaMemcpyDeviceToHost));

    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    int *recv_counts = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *recv_displs = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    if (rank == 0) {
        for(int i=0; i<size; i++) {
            recv_counts[i] = (i == size-1) ? M - i*(M/size) : M/size;
            recv_displs[i] = i*(M/size);
        }
    }
    MPI_Gatherv(h_local_y, local_M, MPI_FLOAT, h_global_y_gpu, recv_counts, recv_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Validation logic
        spmv_csr_sequential(&A, h_x, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        double gflops = calculate_gflops(nnz, avg_time_s);
        double bw = calculate_bandwidth(M, N, nnz, avg_time_s, "CSR");
        double tts = calculate_tts(global_start);

        printf("\n--- MULTI-GPU CSR Benchmark (%d GPUs) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, nnz);
        printf("Avg Time: %e s (± %e s)\n", avg_time_s, std_dev_s);
        printf("GFLOPS  : %.4f\n", gflops);
        printf("BW      : %.4f GB/s\n", bw);
        printf("TTS     : %.4f s\n", tts);
        
        free(h_global_y_gpu); free(h_y_ref); free(recv_counts); free(recv_displs);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_ptr); cudaFree(d_col); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    free(h_local_val); free(h_local_col); free(h_local_ptr); free(h_x); free(h_local_y); free(iter_times);
    if (rank == 0) { free(A.row_ptr); free(A.col_idx); free(A.values); }
    MPI_Finalize();
    return 0;
}