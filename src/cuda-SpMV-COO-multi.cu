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
 * Standard CUDA error checking macro.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

/**
 * Sequential CPU version for validation.
 */
void spmv_coo_sequential(const COOMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->nnz; i++) {
        y[mat->rows[i]] += mat->values[i] * x[mat->cols[i]];
    }
}

/**
 * COO Kernel: Each thread processes one non-zero element.
 * atomicAdd is used because multiple threads might update the same row in y.
 */
__global__ void spmv_coo_kernel(int nnz, const int *rows, const int *cols,
                                const float *vals, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        atomicAdd(&y[rows[i]], __ldg(&vals[i]) * __ldg(&x[cols[i]]));
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double global_start = omp_get_wtime();

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix.mtx>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    // 1. DEVICE BINDING
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    COOMatrix mat;
    int M, N, global_nnz;
    float *h_x = NULL;
    float *h_y_ref = NULL;

    // 2. DATA LOADING AND DISTRIBUTION (Rank 0)
    if (rank == 0) {
        load_matrix_market_to_coo(argv[1], &mat);
        M = mat.M; N = mat.N; global_nnz = mat.nnz;
        h_y_ref = (float *)malloc(M * sizeof(float));
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&global_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    h_x = (float*)malloc(N * sizeof(float));
    if (rank == 0) fill_random_vector(h_x, N);
    MPI_Bcast(h_x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // DECOMPOSITION: Split NNZ instead of Rows
    int local_nnz = global_nnz / size;
    int nnz_start = rank * local_nnz;
    if (rank == size - 1) local_nnz = global_nnz - nnz_start;

    // Buffers for local NNZ data
    int *h_local_rows = (int*)malloc(local_nnz * sizeof(int));
    int *h_local_cols = (int*)malloc(local_nnz * sizeof(int));
    float *h_local_vals = (float*)malloc(local_nnz * sizeof(float));

    // Scatter the COO arrays
    int *send_counts = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    if (rank == 0) {
        for(int i=0; i<size; i++) {
            send_counts[i] = (i == size-1) ? global_nnz - i*(global_nnz/size) : global_nnz/size;
            displs[i] = i*(global_nnz/size);
        }
    }

    MPI_Scatterv(rank == 0 ? mat.rows : NULL, send_counts, displs, MPI_INT, h_local_rows, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? mat.cols : NULL, send_counts, displs, MPI_INT, h_local_cols, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? mat.values : NULL, send_counts, displs, MPI_FLOAT, h_local_vals, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 3. GPU SETUP
    int *d_rows, *d_cols; float *d_vals, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_rows, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cols, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, M * sizeof(float))); // Note: y is size M because any local NNZ can hit any row

    CUDA_CHECK(cudaMemcpy(d_rows, h_local_rows, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cols, h_local_cols, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, h_local_vals, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (local_nnz + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP ---
    for(int i=0; i<WARMUP_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        spmv_coo_kernel<<<gridSize, blockSize>>>(local_nnz, d_rows, d_cols, d_vals, d_x, d_y);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // --- BENCHMARK ---
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    for(int i=0; i<BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        spmv_coo_kernel<<<gridSize, blockSize>>>(local_nnz, d_rows, d_cols, d_vals, d_x, d_y);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        iter_times[i] = (double)ms / 1000.0;
    }

    double avg_time_s = arithmetic_mean(iter_times, BENCHMARK_ITERATIONS);
    double std_dev_s = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // 4. REDUCE RESULTS (Each GPU has partial sums for the WHOLE vector y)
    float *h_local_y = (float*)malloc(M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost));

    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    
    // We use MPI_Reduce with MPI_SUM because each GPU computed a partial y
    MPI_Reduce(h_local_y, h_global_y_gpu, M, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        memset(h_y_ref, 0, M * sizeof(float));
        spmv_coo_sequential(&mat, h_x, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU COO ( %d GPUs ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, global_nnz);
        printf("Avg Time: %e s (± %e s)\n", avg_time_s, std_dev_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(global_nnz, avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, global_nnz, avg_time_s, "COO"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(send_counts); free(displs);
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_rows)); CUDA_CHECK(cudaFree(d_cols)); CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(h_local_rows); free(h_local_cols); free(h_local_vals); free(h_x); free(h_local_y); free(iter_times);
    if (rank == 0) { free(mat.rows); free(mat.cols); free(mat.values); }
    MPI_Finalize();
    return 0;
}