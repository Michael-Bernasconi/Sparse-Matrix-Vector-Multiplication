#!/bin/bash
module load CUDA/11.8.0

BIN_DIR="./bin"
LOG_DIR="./results/single_matrices"
mkdir -p "$LOG_DIR"

MATRIX_PATH=$1
MATRIX_NAME=$(basename "$MATRIX_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/report_${MATRIX_NAME}_${TIMESTAMP}.log"

echo "--- PROCESSING MATRIX: $MATRIX_NAME ---"
echo "Start time: $(date)" > "$LOG_FILE"

executables=("cpu-SpMV-CSR" "cuda-SpMV-CSR" "cuda-SpMV-COO" "cuda-SpMV-CSR-Vector" "cuda-SpMV-cuSPARSE")

for exe in "${executables[@]}"; do
    if [ -f "$BIN_DIR/$exe" ]; then
        echo "--> Testing $exe..."
        echo -e "\n[$exe]" >> "$LOG_FILE"
        
        for i in {1..1}; do
            echo "    Run $i/1..."
            echo "--- Run $i ---" >> "$LOG_FILE"
            
            if [ "$exe" == "cpu-SpMV-CSR" ]; then
                echo "    [Profiling CPU Cache...]"
                # Valgrind con tool Cachegrind:
                valgrind --tool=cachegrind --cache-sim=yes $BIN_DIR/$exe "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
            else
                $BIN_DIR/$exe "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
            fi
            
            echo "-----------------------" >> "$LOG_FILE"
        done
    else
        echo "Warning: $exe not found"
    fi
done

echo "Matrix $MATRIX_NAME completed."