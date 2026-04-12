#!/bin/bash
# Main submission script to manage 5 runs and folder organization

MATRIX_DIR="./data"
BASE_LOG_DIR="./results/single_matrices"

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
# Force recompile of CPU source with standard flags (removing the "Lite" object)
echo "Compiling Standard versions..."
rm -f obj/cpu-SpMV-CSR.o
make

# --- START OF 5 RUNS LOOP ---
for run_id in {1..5}; do
    # Define and create the specific folder for this run
    RUN_FOLDER="${BASE_LOG_DIR}/run_${run_id}"
    mkdir -p "$RUN_FOLDER"
    
    echo "======================================================="
    echo " STARTING RUN ${run_id} / 5 "
    echo "======================================================="

    for matrix in "$MATRIX_DIR"/*.mtx; do
        [ -e "$matrix" ] || continue
        m_name=$(basename "$matrix")
        
        echo "-------------------------------------------------------"
        echo "PROCESSING MATRIX: $m_name (Run $run_id)"
        echo "-------------------------------------------------------"

        # JOB 1: PERFORMANCE (Passing RUN_FOLDER as second argument)
        echo "Submitting PERFORMANCE job..."
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
             --partition=edu-short -w edu01 --account=gpu.computing26 \
             ./run_performance.sh "$matrix" "$RUN_FOLDER"

        # JOB 2: CACHE (Passing RUN_FOLDER as second argument)
        echo "Submitting CACHE job..."
        srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 \
             --partition=edu-short -w edu01 --account=gpu.computing26 \
             ./run_cache.sh "$matrix" "$RUN_FOLDER"

        echo "Finished $m_name for Run $run_id."
    done
done

echo "All 5 runs completed successfully."
