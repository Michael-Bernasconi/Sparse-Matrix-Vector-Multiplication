#!/bin/bash
# Script for performance benchmarking (No Profiling)
module load CUDA/11.8.0

BIN_DIR="./bin"
LOG_DIR="./results/single_matrices"
mkdir -p "$LOG_DIR"

MATRIX_PATH=$1
MATRIX_NAME=$(basename "$MATRIX_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/PERF_${MATRIX_NAME}_${TIMESTAMP}.log"

echo "--- PERFORMANCE TESTING: $MATRIX_NAME ---"
echo "Start time: $(date)" > "$LOG_FILE"

# All executables including GPU versions
executables=("cpu-SpMV-CSR" "cuda-SpMV-CSR" "cuda-SpMV-COO" "cuda-SpMV-CSR-Vector" "cuda-SpMV-cuSPARSE")

for exe in "${executables[@]}"; do
    if [ -f "$BIN_DIR/$exe" ]; then
        echo "--> Testing $exe..."
        echo -e "\n[$exe]" >> "$LOG_FILE"
        
        # Standard execution without overhead
        $BIN_DIR/$exe "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
        
        echo "-----------------------" >> "$LOG_FILE"
    else
        echo "Warning: $exe not found"
    fi
done