#!/bin/bash

# Configuration 
BIN_DIR="./bin"
MATRIX_DIR="./data"
LOG_DIR="./results/benchmark/flops-bw-tts"
NUM_RUNS=5

# Create the results directory if it doesn't exist
mkdir -p "$LOG_DIR"

# 1. Clean and Compile (Executed ONLY ONCE)
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
    echo "Error: Directory $MATRIX_DIR not found. (Please insert .mtx files in the data folder)"
    exit 1
fi

# 2. Main Loop: Execute the entire benchmark 5 times
for (( run=1; run<=NUM_RUNS; run++ )); do
    echo "=========================================================="
    echo " STARTING EXECUTION $run OF $NUM_RUNS"
    echo "=========================================================="

    # Generate a unique timestamp for this specific run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/report_flops-bw-tts-cpu-gpu_run${run}_${TIMESTAMP}.log"

    # Prepare Log File for this run
    echo "Benchmark Execution $run of $NUM_RUNS - $(date)" > "$LOG_FILE"
    echo "------------------------------------------------" >> "$LOG_FILE"

    # Iterate through every .mtx file in the folder
    for matrix in "$MATRIX_DIR"/*.mtx; do
        # Check if the file exists to avoid edge cases with empty folders
        [ -e "$matrix" ] || { echo "No .mtx files found in $MATRIX_DIR"; exit 1; }

        matrix_name=$(basename "$matrix")
        
        echo " PROCESSING MATRIX: $matrix_name"
        echo "Matrix: $matrix_name" >> "$LOG_FILE"

        # List of executables to run 
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

    echo ">>> Execution $run completed. Detailed log saved locally at: $LOG_FILE"
    echo ""
    
    # 2-second pause to guarantee a different timestamp for the next run
    sleep 2 
done

echo "All $NUM_RUNS benchmark executions have finished successfully!"