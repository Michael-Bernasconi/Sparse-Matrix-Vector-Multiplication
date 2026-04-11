#!/bin/bash
MATRIX_DIR="./data"

echo "Starting compilation..."
make clean && make
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

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

    echo "Finished $m_name. Proceeding to the next one..."
    echo ""
done

echo "All matrices have been processed."