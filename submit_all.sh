#!/bin/bash
# Main submission script to manage 5 runs and folder organization

MATRIX_DIR="./data"
BASE_LOG_DIR="./results/single_matrices"

# if you want execute with only 1 file (./submit_all.sh data/mio.mtx),
# otherwise you can execute all
if [ -n "$1" ]; then
    SELECTED_MATRICES=("$1")
    echo "Mode: Execute only one matrix: $1"
else
    SELECTED_MATRICES=("$MATRIX_DIR"/*.mtx)
    echo "Mode: Execute all (all matrix in $MATRIX_DIR)"
fi

echo "Starting compilation..."

# 1. Clean previous builds
make clean

# 2. Compile "Lite" version for Cache Profiling (1 iteration)
echo "Compiling Lite version..."
make profile_build ITER="-DBENCHMARK_ITERATIONS=1 -DWARMUP_ITERATIONS=0"
if [ -f "bin/cpu-SpMV-CSR" ]; then
    mv bin/cpu-SpMV-CSR bin/cpu-SpMV-CSR-lite
else
    echo "Error: Lite compilation failed!"
    exit 1
fi

# 3. Compile Standard versions for Performance
echo "Compiling Standard versions..."
rm -f obj/cpu-SpMV-CSR.o
make

# --- START OF 5 RUNS LOOP -----
for run_id in {1..5}; do
    RUN_FOLDER="${BASE_LOG_DIR}/run_${run_id}"
    mkdir -p "$RUN_FOLDER"
    
    echo "======================================================="
    echo " STARTING RUN ${run_id} / 5 "
    echo "======================================================="

    # Iteriamo sull'array definito all'inizio (singolo file o tutti)
    for matrix in "${SELECTED_MATRICES[@]}"; do
        [ -e "$matrix" ] || continue
        m_name=$(basename "$matrix")
        
        echo "-------------------------------------------------------"
        echo "PROCESSING MATRIX: $m_name (Run $run_id)"
        echo "-------------------------------------------------------"

        # JOB 1: PERFORMANCE
        echo "Submitting PERFORMANCE job..."
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
             --partition=edu-short -w edu01 --account=gpu.computing26 \
             ./run_performance.sh "$matrix" "$RUN_FOLDER"

        # JOB 2: CACHE
        echo "Submitting CACHE job..."
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 \
             --partition=edu-short -w edu01 --account=gpu.computing26 \
             ./run_cache.sh "$matrix" "$RUN_FOLDER"
    done
done

echo "All jobs submitted. Check $BASE_LOG_DIR for logs."