#!/bin/bash
MATRIX_DIR="./data"

echo "Starting compilation..."

# 1. Compile standard versions for Performance (uses header defaults: 20/500)
make clean && make
if [ $? -ne 0 ]; then
    echo "Error: Standard compilation failed!"
    exit 1
fi

# 2. Compile "Lite" version for Cache Profiling (1 iteration, 0 warmup)
# We rename the binary to cpu-SpMV-CSR-lite to keep both versions
make profile_build ITER="-DBENCHMARK_ITERATIONS=1 -DWARMUP_ITERATIONS=0"
mv bin/cpu-SpMV-CSR bin/cpu-SpMV-CSR-lite

for matrix in "$MATRIX_DIR"/*.mtx; do
    [ -e "$matrix" ] || continue
    m_name=$(basename "$matrix")
    
    echo "-------------------------------------------------------"
    echo "PROCESSING MATRIX: $m_name"
    echo "-------------------------------------------------------"

    # --- JOB 1: PERFORMANCE (Uses standard 500-iter binaries) ---
    echo "Submitting PERFORMANCE job..."
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
         --partition=edu-short -w edu01 --account=gpu.computing26 \
         ./run_performance.sh "$matrix"

    # --- JOB 2: CACHE (Uses the 1-iter lite binary) ---
    echo "Submitting CACHE job..."
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 \
         --partition=edu-short -w edu01 --account=gpu.computing26 \
         ./run_cache.sh "$matrix"

    echo "Finished $m_name. Proceeding to the next one..."
    echo ""
done

echo "All matrices have been processed."