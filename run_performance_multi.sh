#!/bin/bash
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX_PATH=$1
RUN_FOLDER=$2
NUM_GPUS=$3  
BIN_DIR="./bin"

MATRIX_NAME=$(basename "$MATRIX_PATH")
LOG_FILE="$RUN_FOLDER/PERF_${NUM_GPUS}GPU_${MATRIX_NAME}.log"

executables=("cuda-SpMV-CSR-multi" "cuda-SpMV-COO-multi" "cuda-SpMV-CSR-Vector-multi" "cuda-SpMV-cuSparse-multi")

echo "--- Benchmarking: $MATRIX_NAME with $NUM_GPUS GPUs ---"
for exe in "${executables[@]}"; do
    if [ -f "$BIN_DIR/$exe" ]; then
        echo "Running $exe on $NUM_GPUS GPUs..."
        echo -e "\n[$exe - $NUM_GPUS GPUs]" >> "$LOG_FILE"
        
        mpirun -np $NUM_GPUS ./$BIN_DIR/$exe "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
    fi
done