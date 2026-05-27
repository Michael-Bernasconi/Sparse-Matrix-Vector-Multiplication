#!/bin/bash
#SBATCH --job-name=spmv_coo_multi
#SBATCH --nodes=1
#SBATCH --ntasks=2                  
#SBATCH --cpus-per-task=1           
#SBATCH --gres=gpu:2           
#SBATCH --partition=edu-short     
#SBATCH -w edu01
#SBATCH --account=gpu.computing26   
#SBATCH --time=00:05:00               
#SBATCH --output=coo_multi_gpu_res.out

# Load required modules
module purge
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

echo "--- Compilazione COO Multi-GPU ---"
# Assicurati di aver salvato il codice come src/cuda-SpMV-coo-multi.cu
make clean
make

echo -e "\n--- Esecuzione su 2 GPU (Nodo edu01) ---"
# Lancio del binario COO distribuito su 2 GPU
mpirun -np 2 ./bin/cuda-SpMV-coo-multi ./data/ASIC_680ks.mtx