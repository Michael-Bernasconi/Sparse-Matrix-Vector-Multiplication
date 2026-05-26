#!/bin/bash
#SBATCH --job-name=spmv_multi_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2                  
#SBATCH --cpus-per-task=1           
#SBATCH --gres=gpu:2           
#SBATCH --partition=edu-short     
#SBATCH -w edu01
#SBATCH --account=gpu.computing26   
#SBATCH --time=00:05:00               # Aumentato a 10 min per sicurezza
#SBATCH --output=multi_gpu_res.out

# Carica i moduli necessari
module purge
module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

echo "--- Compilazione ---"
make clean
make

echo -e "\n--- Esecuzione su 2 GPU ---"
# Eseguiamo il file specifico
mpirun -np 2 ./bin/cuda-SpMV-multi-gpu ./data/ASIC_680ks.mtx