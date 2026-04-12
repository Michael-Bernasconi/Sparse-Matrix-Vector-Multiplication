#!/bin/bash
# Script for CPU Cache profiling using perf (Hardware Counters)
BIN_DIR="./bin"
LOG_DIR="./results/single_matrices"
mkdir -p "$LOG_DIR"

MATRIX_PATH=$1
MATRIX_NAME=$(basename "$MATRIX_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/CACHE_${MATRIX_NAME}_${TIMESTAMP}.log"

echo "--- CACHE PROFILING: $MATRIX_NAME ---"
echo "Start time: $(date)" > "$LOG_FILE"

# We use the lite version (1 iteration) which is sufficient
EXE="cpu-SpMV-CSR-lite"

if [ -f "$BIN_DIR/$EXE" ]; then
    echo "--> Profiling $EXE with perf..."
    echo -e "\n[Cache Profiling - $EXE]" >> "$LOG_FILE"
    
    perf stat -e cache-references,cache-misses "$BIN_DIR/$EXE" "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
    
    echo "-----------------------" >> "$LOG_FILE"
else
    echo "Error: $EXE not found. Check compilation in master script."
fi