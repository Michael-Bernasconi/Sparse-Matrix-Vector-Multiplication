#!/bin/bash
MATRIX_DIR="./data"

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
# We do NOT run 'make clean' here, but we must force a recompile of the CPU part
# because the object files currently have the "Lite" flags.
echo "Compiling Standard versions..."
rm -f obj/cpu-SpMV-CSR.o  # Force recompile of CPU source with standard flags
make

for matrix in "$MATRIX_DIR"/*.mtx; do
    [ -e "$matrix" ] || continue
    m_name=$(basename "$matrix")
    
    echo "-------------------------------------------------------"
    echo "PROCESSING MATRIX: $m_name"
    echo "-------------------------------------------------------"

    # --- JOB 1: PERFORMANCE ---
    echo "Submitting PERFORMANCE job..."
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 \
         --partition=edu-short -w edu01 --account=gpu.computing26 \
         ./run_performance.sh "$matrix"

    # --- JOB 2: CACHE ---
    echo "Submitting CACHE job..."
    srun --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:0 \
         --partition=edu-short -w edu01 --account=gpu.computing26 \
         ./run_cache.sh "$matrix"

    echo "Finished $m_name."
    echo ""
done

echo "All tasks completed."