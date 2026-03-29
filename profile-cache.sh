#!/bin/bash

# Configuration
MATRIX_DIR="/mnt/c/Users/micha/Documents/GitHub/Sparse-Matrix-Vector-Multiplication/data"
MATRIX="$MATRIX_DIR/FEM_3D_thermal1.mtx" # Use this as a sample
PROF_DIR="./results/benchmark/profile-cache"
BIN_DIR="./bin"

mkdir -p "$PROF_DIR"

echo "--- 1. Special Compilation for Profiling (1 iteration) ---"
make clean
make profile_build ITER="-DBENCHMARK_ITERATIONS=1 -DWARMUP_ITERATIONS=0"

echo "--- 2. CPU Cache Analysis (Valgrind) ---"
for format in "CSR" "COO"; do
    EXE="cpu-SpMV-$format"
    echo "Analyzing $EXE..."
    # Running Valgrind to simulate cache hierarchy
    valgrind --tool=cachegrind --cachegrind-out-file="$PROF_DIR/cache_$format.out" \
             $BIN_DIR/$EXE "$MATRIX" > /dev/null 2>&1
    
    # Converting raw data into a readable report
    cg_annotate "$PROF_DIR/cache_$format.out" > "$PROF_DIR/report_cpu_$format.txt"
done

echo "--- 3. GPU Cache Analysis (Nsight Compute) ---"
for format in "CSR" "COO"; do
    EXE="cuda-SpMV-$format"
    echo "Analyzing $EXE..."
    # Extracting L1/Texture and L2 Hit Rates
    ncu --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct \
        --target-processes all --log-file "$PROF_DIR/report_gpu_$format.txt" \
        $BIN_DIR/$EXE "$MATRIX" > /dev/null 2>&1
done

echo "--- FINISHED! ---"
echo "Results are located in: $PROF_DIR"
