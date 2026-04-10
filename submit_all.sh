#!/bin/bash
# Master script to submit Slurm jobs for each matrix in the data directory

MATRIX_DIR="./data"

echo "Starting compilation..."
# Clean previous builds and compile everything
make clean && make
if [ $? -ne 0 ]; then
    echo "Error: Compilation failed!"
    exit 1
fi

# Iterate through all .mtx files in the data folder
for matrix in "$MATRIX_DIR"/*.mtx; do
    # Skip if no .mtx files are found
    [ -e "$matrix" ] || continue
    
    m_name=$(basename "$matrix")
    
    echo "Submitting job for: $m_name..."

    # Launch srun for each matrix individually to reset the time limit
    srun --nodes=1 \
         --ntasks=1 \
         --cpus-per-task=1 \
         --gres=gpu:1 \
         -w edu02 \
         --partition=edu-short \
         --account=gpu.computing26 \
         ./run_single.sh "$matrix"

    echo "Finished $m_name. Proceeding to the next one..."
    
    # Brief pause to allow the scheduler to update
    sleep 2
done

echo "All matrices have been processed."
