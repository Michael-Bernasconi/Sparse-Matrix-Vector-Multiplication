#!/bin/bash
# Script for performance benchmarking (No Profiling)
module load CUDA/11.8.0

BIN_DIR="./bin"
# Get the log directory from the second argument, fallback to a default if empty
LOG_DIR=${2:-"./results/single_matrices"}
mkdir -p "$LOG_DIR"

MATRIX_PATH=$1
MATRIX_NAME=$(basename "$MATRIX_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/PERF_${MATRIX_NAME}_${TIMESTAMP}.log"

echo "--- PERFORMANCE TESTING: $MATRIX_NAME ---"
echo "Start time: $(date)" > "$LOG_FILE"

# List of executables to test
executables=("cpu-SpMV-CSR" "cuda-SpMV-CSR" "cuda-SpMV-COO" "cuda-SpMV-CSR-Vector" "cuda-SpMV-cuSparse")

for exe in "${executables[@]}"; do
    if [ -f "$BIN_DIR/$exe" ]; then
        echo "--> Testing $exe (Standard iterations)..."
        echo -e "\n[$exe]" >> "$LOG_FILE"
        $BIN_DIR/$exe "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
        echo "-----------------------" >> "$LOG_FILE"
    else
        echo "Warning: $exe not found in $BIN_DIR"
    fi
done
