#!/bin/bash
# Script for CPU Cache profiling using Valgrind
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
    echo "--> Profiling $EXE with Cachegrind (1 iteration)..."
    echo -e "\n[Cache Profiling - $EXE]" >> "$LOG_FILE"
    valgrind --tool=cachegrind --cache-sim=yes "$BIN_DIR/$EXE" "$MATRIX_PATH" >> "$LOG_FILE" 2>&1
    echo "-----------------------" >> "$LOG_FILE"
else
    echo "Error: $EXE not found. Check compilation in master script."
fi