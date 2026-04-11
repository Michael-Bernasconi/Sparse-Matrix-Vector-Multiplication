#!/bin/bash
# Script for CPU Cache profiling using Valgrind
# Uses the 'lite' version (1 iteration) to avoid Slurm timeouts

BIN_DIR="./bin"
LOG_DIR="./results/single_matrices"
mkdir -p "$LOG_DIR"

MATRIX_PATH=$1
MATRIX_NAME=$(basename "$MATRIX_PATH")
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/CACHE_${MATRIX_NAME}_${TIMESTAMP}.log"

echo "--- CACHE PROFILING: $MATRIX_NAME ---"
echo "Start time: $(date)" > "$LOG_FILE"

# Use the lite version specifically compiled for profiling
EXE="cpu-SpMV-CSR-lite"

if [ -f "$BIN_DIR/$EXE" ]; then
    echo "--> Profiling $EXE with Cachegrind (Fast Mode: 1 iteration)..."
    echo -e "\n[Cache Profiling - $EXE]" >> "$LOG_FILE"
    
    # Run Valgrind
    valgrind --tool=cachegrind --cache-sim=yes "$BIN_DIR/$EXE" "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
    
    echo "-----------------------" >> "$LOG_FILE"
else
    echo "Error: $EXE not found. Ensure the main script compiled the lite version."
fi