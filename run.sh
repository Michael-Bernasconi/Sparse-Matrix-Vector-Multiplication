#!/bin/bash

# Configuration (Percorsi Locali Relativi)
BIN_DIR="./bin"
MATRIX_DIR="./data"
LOG_DIR="./results/benchmark/flops-bw"
LOG_FILE="$LOG_DIR/report_flops-bw-cpu-gpu.log"

# Create the results directory if it doesn't exist
mkdir -p "$LOG_DIR"

# 1. Clean and Compile
echo "--- Starting Compilation (with -O3 optimizations) ---"
make clean && make
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi
echo "--- Compilation Successful ---"
echo ""

# Check if matrix directory exists
if [ ! -d "$MATRIX_DIR" ]; then
    echo "Error: Directory $MATRIX_DIR not found. (Assicurati di mettere i file .mtx in una cartella chiamata 'data')"
    exit 1
fi

# Prepare Log File
echo "Benchmark Execution - $(date)" > "$LOG_FILE"
echo "------------------------------------------------" >> "$LOG_FILE"

# 2. Iterate through every .mtx file in the folder
for matrix in "$MATRIX_DIR"/*.mtx; do
    # Controllo di sicurezza se la cartella è vuota
    [ -e "$matrix" ] || { echo "No .mtx files found in $MATRIX_DIR"; exit 1; }

    matrix_name=$(basename "$matrix")
    
    echo "=========================================================="
    echo " PROCESSING MATRIX: $matrix_name"
    echo "=========================================================="
    echo "Matrix: $matrix_name" >> "$LOG_FILE"

    # List of executables to run (AGGIORNATA CON I DUE NUOVI FILE)
    executables=(
        "cpu-SpMV-CSR" 
        "cpu-SpMV-COO" 
        "cuda-SpMV-CSR" 
        "cuda-SpMV-COO"
        "cuda-SpMV-CSR-Vector"
        "cuda-SpMV-cuSPARSE"
    )

    for exe in "${executables[@]}"; do
        if [ -f "$BIN_DIR/$exe" ]; then
            echo "--> Running $exe..."
            echo "[$exe]" >> "$LOG_FILE"
            
            # Execute and capture output
            output=$($BIN_DIR/$exe "$matrix" 2>&1)
            
            if [ $? -eq 0 ]; then
                echo "$output"
                echo "$output" >> "$LOG_FILE"
            else
                echo "RUNTIME ERROR: $exe failed on $matrix_name"
                echo "RUNTIME ERROR" >> "$LOG_FILE"
            fi
            echo "----------------------------------------------------------" >> "$LOG_FILE"
            echo "" >> "$LOG_FILE"
        else
            echo "Warning: Executable $exe not found in $BIN_DIR"
        fi
    done
    echo ""
done

echo "Benchmarks finished. Detailed logs saved locally at:"
echo "$LOG_FILE"