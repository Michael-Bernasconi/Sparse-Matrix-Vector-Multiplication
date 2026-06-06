#!/bin/bash

# ==============================================================================
# ENVIRONMENT MODULES LOADING
# ==============================================================================
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

# ==============================================================================
# INPUT ARGUMENTS & CONFIGURATION
# ==============================================================================
MATRIX_PATH=$1
RUN_FOLDER=$2
NUM_GPUS=$3  
BIN_DIR="./bin"

MATRIX_NAME=$(basename "$MATRIX_PATH")
LOG_FILE="$RUN_FOLDER/PERF_${NUM_GPUS}GPU_${MATRIX_NAME}.log"

# Elenco degli eseguibili da testare (incluso il baseline del prof)
executables=(
    "cuda-SpMV-CSR-multi" 
    "cuda-SpMV-COO-multi" 
    "cuda-SpMV-CSR-Vector-multi" 
    "cuda-SpMV-cuSparse-multi"
    "prof-SpMV-baseline"
)

# ==============================================================================
# BENCHMARK EXECUTION CAMPAIGN
# ==============================================================================
echo "--- Benchmarking: $MATRIX_NAME with $NUM_GPUS GPUs ---"

for exe in "${executables[@]}"; do
    if [ -f "$BIN_DIR/$exe" ]; then
        echo "Running $exe on $NUM_GPUS GPUs..."
        echo -e "\n[$exe - $NUM_GPUS GPUs]" >> "$LOG_FILE"
        
        # Esecuzione MPI con i flag MCA per disattivare il memory pinning problematico
        mpirun --oversubscribe -np $NUM_GPUS \
            --mca mpi_common_cuda_register_memory 0 \
            --mca btl_openib_allow_cuda_cuda_reg_mem 0 \
            --mca btl_smcuda_use_cuda_ipc 0 \
            "./$BIN_DIR/$exe" "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
    else
        echo "Warning: Executable $BIN_DIR/$exe not found, skipping."
    fi
done