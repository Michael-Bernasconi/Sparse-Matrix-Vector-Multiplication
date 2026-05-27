#!/bin/bash
#SBATCH --job-name=spmv_vector_multi
#SBATCH --nodes=1
#SBATCH --ntasks=2                  
#SBATCH --cpus-per-task=1           
#SBATCH --gres=gpu:2           
#SBATCH --partition=edu-short     
#SBATCH -w edu01
#SBATCH --account=gpu.computing26   
#SBATCH --time=00:05:00               
#SBATCH --output=vector_multi_gpu_res.out

# Carica i moduli necessari
module purge
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

echo "--- Compilazione CSR-Vector Multi-GPU ---"
make clean
make

echo -e "\n--- Esecuzione su 2 GPU (Nodo edu01) ---"
# Lanciamo il nuovo eseguibile CSR-Vector
mpirun -np 2 ./bin/cuda-SpMV-CSR-Vector-multi ./data/ASIC_680ks.mtx