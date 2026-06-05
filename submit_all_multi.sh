#!/bin/bash
MATRIX_DIR="./data"
BASE_LOG_DIR="./results/multi_gpu"

make clean
make

SELECTED_MATRICES=("$MATRIX_DIR"/*.mtx)
GPU_CONFIGS=(1 2 4)

for run_id in {1..5}; do
    for gpus in "${GPU_CONFIGS[@]}"; do
        RUN_FOLDER="${BASE_LOG_DIR}/run_${run_id}/${gpus}gpu"
        mkdir -p "$RUN_FOLDER"

        echo "=== RUN ${run_id} - CONFIG: ${gpus} GPUs ==="

        for matrix in "${SELECTED_MATRICES[@]}"; do
            [ -e "$matrix" ] || continue
            m_name=$(basename "$matrix")

            echo "Submitting: $m_name on $gpus GPUs"

            
            srun --nodes=1 --ntasks=1 --cpus-per-task=$gpus --gres=gpu:$gpus \
                 --partition=edu-short -w edu01 --account=gpu.computing26 \
                 --time=00:05:00 \
                 ./run_performance_multi.sh "$matrix" "$RUN_FOLDER" "$gpus"
        done
    done
done