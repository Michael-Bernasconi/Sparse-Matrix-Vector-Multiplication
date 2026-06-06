#!/bin/bash
SYNTH_DIR="./data-synt"
BASE_LOG_DIR="./results/weak_scaling"

make clean
make

GPU_CONFIGS=(1 2 4)

for run_id in {1..5}; do
    echo "=========================================="
    echo "=== WEAK SCALING EXPERIMENT - RUN ${run_id} ==="
    echo "=========================================="

    for gpus in "${GPU_CONFIGS[@]}"; do
        RUN_FOLDER="${BASE_LOG_DIR}/run_${run_id}/${gpus}gpu"
        mkdir -p "$RUN_FOLDER"

        echo "--- Active Configuration: ${gpus} GPU(s) ---"

        # Seleziona unicamente i set sintetici dedicati a questa scala di risorse
        MATRICES=("$SYNTH_DIR"/*_${gpus}gpu.mtx)

        for matrix in "${MATRICES[@]}"; do
            [ -e "$matrix" ] || continue
            m_name=$(basename "$matrix")

            echo "Submitting Weak Scaling workload: $m_name on ${gpus} GPU(s)"

            srun --nodes=1 --ntasks=1 --cpus-per-task=$gpus --gres=gpu:$gpus \
                 --partition=edu-short -w edu01 --account=gpu.computing26 \
                 --time=00:05:00 \
                 ./run_performance_multi.sh "$matrix" "$RUN_FOLDER" "$gpus"
        done
    done
done
echo "All Weak Scaling benchmark sequences completed!"