#include <cuda_runtime.h>
#include <cusparse.h>
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
 * Standard cuSPARSE error checking macro.
 */
#define CUSPARSE_CHECK(call) \
    do { \
        cusparseStatus_t status = call; \
        if (status != CUSPARSE_STATUS_SUCCESS) { \
            printf("cuSPARSE Error at %s:%d - code %d\n", __FILE__, __LINE__, status); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

/**
 * Sequential CPU version used as the "Gold Standard" for result validation.
 */
void spmv_csr_sequential(const CSRMatrix *mat, const float *x, float *y) {
    for (int i = 0; i < mat->M; i++) {
        float sum = 0.0f;
        for (int j = mat->row_ptr[i]; j < mat->row_ptr[i + 1]; j++) {
            sum += mat->values[j] * x[mat->col_idx[j]];
        }
        y[i] = sum;
    }
}

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Record the start time for the Time-to-Solution (TTS) metric
    double global_start = omp_get_wtime();

    if (argc < 2) {
        if (rank == 0) printf("Usage: %s <matrix.mtx>\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    // 1. DEVICE BINDING
    // Assign each MPI rank to a specific GPU on the node
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(rank % dev_count));

    CSRMatrix A;
    int M, N, nnz;
    float *h_x = NULL;
    float *h_y_ref = NULL;

    // 2. DATA LOADING AND DISTRIBUTION (Handled by Rank 0)
    if (rank == 0) {
        load_matrix_market_to_csr(argv[1], &A);
        M = A.M; N = A.N; nnz = A.nnz;
        h_y_ref = (float *)malloc(M * sizeof(float));
    }

    // Broadcast global matrix dimensions to all ranks
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Distribute the input vector X (all-to-all broadcast)
    h_x = (float*)malloc(N * sizeof(float));
    if (rank == 0) fill_random_vector(h_x, N);
    MPI_Bcast(h_x, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // DOMAIN DECOMPOSITION: Partition rows across available GPUs
    int local_M = M / size;
    int r_start = rank * local_M;
    if (rank == size - 1) local_M = M - r_start; // Handle remainder rows in the last rank

    int local_nnz;
    int *send_counts_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *displs_nnz = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    // Rank 0 calculates the number of non-zero elements (NNZ) per partition
    if (rank == 0) {
        for(int i=0; i<size; i++) {
            int start = i * (M/size);
            int end = (i == size-1) ? M : (i+1)*(M/size);
            send_counts_nnz[i] = A.row_ptr[end] - A.row_ptr[start];
            displs_nnz[i] = A.row_ptr[start];
        }
    }
    MPI_Scatter(send_counts_nnz, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate local host buffers for the matrix partition
    float *h_local_val = (float*)malloc(local_nnz * sizeof(float));
    int *h_local_col = (int*)malloc(local_nnz * sizeof(int));
    int *h_local_ptr = (int*)malloc((local_M + 1) * sizeof(int));

    // Scatter the matrix values and column indices
    MPI_Scatterv(rank == 0 ? A.values : NULL, send_counts_nnz, displs_nnz, MPI_FLOAT, h_local_val, local_nnz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? A.col_idx : NULL, send_counts_nnz, displs_nnz, MPI_INT, h_local_col, local_nnz, MPI_INT, 0, MPI_COMM_WORLD);

    // ROW POINTER NORMALIZATION: Adjust pointers to be relative to the local partition (start at 0)
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

    // 3. GPU MEMORY ALLOCATION AND cuSPARSE SETUP
    int *d_ptr, *d_col; float *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_ptr, (local_M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col, local_nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val, local_nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, local_M * sizeof(float)));

    // Copy local data from Host to Device
    CUDA_CHECK(cudaMemcpy(d_ptr, h_local_ptr, (local_M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_local_col, local_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_val, h_local_val, local_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuSPARSE handle and descriptors
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, local_M, N, local_nnz, d_ptr, d_col, d_val, 
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, local_M, d_y, CUDA_R_32F));

    // Set SpMV parameters (y = alpha * A * x + beta * y)
    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    
    // Request required buffer size for cuSPARSE SpMV
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- WARMUP PHASE ---
    for(int i=0; i<WARMUP_ITERATIONS; i++) {
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize all ranks before benchmarking

    // --- BENCHMARK PHASE ---
    double *iter_times = (double *)malloc(BENCHMARK_ITERATIONS * sizeof(double));
    for(int i=0; i<BENCHMARK_ITERATIONS; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        CUSPARSE_CHECK(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        iter_times[i] = (double)ms / 1000.0;
    }

    // Compute average execution time and standard deviation
    double avg_time_s = arithmetic_mean(iter_times, BENCHMARK_ITERATIONS);
    double std_dev_s = sigma_fn_sol(iter_times, avg_time_s, BENCHMARK_ITERATIONS);

    // 4. GATHER RESULTS AND VALIDATION
    float *h_local_y = (float*)malloc(local_M * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_local_y, d_y, local_M * sizeof(float), cudaMemcpyDeviceToHost));

    float *h_global_y_gpu = (rank == 0) ? (float*)malloc(M * sizeof(float)) : NULL;
    int *recv_counts = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;
    int *recv_displs = (rank == 0) ? (int*)malloc(size * sizeof(int)) : NULL;

    // Rank 0 prepares displacements for Gatherv
    if (rank == 0) {
        for(int i=0; i<size; i++) {
            recv_counts[i] = (i == size-1) ? M - i*(M/size) : M/size;
            recv_displs[i] = i*(M/size);
        }
    }

    // Collect result partitions from all GPUs into the final global vector on Rank 0
    MPI_Gatherv(h_local_y, local_M, MPI_FLOAT, h_global_y_gpu, recv_counts, recv_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Performance reporting and validation on Rank 0
    if (rank == 0) {
        spmv_csr_sequential(&A, h_x, h_y_ref);
        validate_results(h_y_ref, h_global_y_gpu, M);

        printf("\n--- MULTI-GPU cuSPARSE ( %d GPUs ) ---\n", size);
        printf("Matrix  : %s (%d x %d, nnz: %d)\n", argv[1], M, N, nnz);
        printf("Avg Time: %e s (± %e s)\n", avg_time_s, std_dev_s);
        printf("GFLOPS  : %.4f\n", calculate_gflops(nnz, avg_time_s));
        printf("BW      : %.4f GB/s\n", calculate_bandwidth(M, N, nnz, avg_time_s, "CSR"));
        printf("TTS     : %.4f s\n", calculate_tts(global_start));

        free(h_global_y_gpu); free(h_y_ref); free(recv_counts); free(recv_displs);
    }

    // CLEANUP: Free resources
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_ptr)); CUDA_CHECK(cudaFree(d_col)); CUDA_CHECK(cudaFree(d_val)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    free(h_local_val); free(h_local_col); free(h_local_ptr); free(h_x); free(h_local_y); free(iter_times);
    if (rank == 0) { free(A.row_ptr); free(A.col_idx); free(A.values); }
    
    // Shut down MPI
    MPI_Finalize();
    return 0;
}