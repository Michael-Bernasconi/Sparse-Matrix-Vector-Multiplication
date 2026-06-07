# Sparse Matrix-Vector Multiplication (SpMV)

**Course:** GPU-Computing-2026

**Author:** Michael Bernasconi ([michael.bernasconi@studenti.unitn.it](mailto:michael.bernasconi@studenti.unitn.it)) - Student ID: 267681

---

## Description

This project implements and analyzes the performance of algorithms for Sparse Matrix-Vector Multiplication (SpMV) in a heterogeneous environment (CPU and GPU). The main goal is to study the impact of different sparse matrix storage formats (such as CSR and COO) on architectural performance, evaluating the efficiency of custom-developed CUDA kernels compared to standard libraries.

---

## Provided Implementations

The project contains several versions of the SpMV operation:

* **CPU SpMV CSR**: Parallel implementation on the host using OpenMP.
* **GPU SpMV COO**: Multi-GPU CUDA kernel utilizing the Coordinate Format.
* **GPU SpMV CSR**: Multi-GPU Native CUDA kernel based on Compressed Sparse Row.
* **GPU SpMV CSR-Vector**: Multi-GPU CUDA kernel optimized to assign one warp per row, maximizing memory coalescence.
* **GPU cuSPARSE Baseline**: Multi-GPU reference implementation using the NVIDIA cuSPARSE library.

---

## Measured Metrics

The performance analysis records the following metrics:

1. **GFLOPS (Giga Floating-Point Operations Per Second)** – computational throughput.
2. **Effective Bandwidth (GB/s)** – utilized memory bandwidth.
3. **Kernel-Time** – actual execution time of the SpMV kernel.
4. **Cache Metrics** – cache hits and misses (D1, LL) collected via Cachegrind.

---

## Target Hardware (UniTN Cluster)

Benchmarks were designed for the University cluster (**edu01 node**, **edu-short partition**).

| Feature              | Host (CPU)                        | Device (GPU)                    |
| -------------------- | --------------------------------- | ------------------------------- |
| **Model**            | Intel(R) Xeon(R) Silver 4309Y     | NVIDIA A30                      |
| **Architecture**     | Ice Lake (x86_64)                 | Ampere (Compute Capability 8.0) |
| **Cores / SMs**      | 16 Cores / 32 Threads (2 Sockets) | 56 SMs (3584 CUDA Cores)        |
| **Clock Frequency**  | 2.80 GHz (Base) / 3.60 GHz (Max)  | 1.44 GHz (Boost)                |
| **FP32 Performance** | 1.84 TFLOPS                       | 10.3 TFLOPS                     |
| **Memory Bandwidth** | 102.4 GB/s                        | 933.1 GB/s (HBM2)               |
| **Global Memory**    | N/A                               | 24 GB                           |
| **L1 Cache**         | 768 KiB Data / 512 KiB Inst.      | 192 KiB per SM                  |
| **L2 Cache**         | 20 MiB                            | 24 MiB                          |
| **L3 Cache**         | 24 MiB                            | N/A                             |
| **Shared Memory**    | N/A                               | 48 KiB (up to 100 KiB per SM)   |
| **Thread Limits**    | 2 Threads/Core                    | 1024 Threads/Block              |
| **Warp Size**        | N/A                               | 32                              |

---

## Software Environment and Dependencies

### Cluster Modules

```bash
CUDA/12.3.2
OpenMpi/4.1.5-CUDA-12.3.2
```

### Compilers

* `gcc` with OpenMP support (`-fopenmp`)
* `nvcc` targeting architecture `sm_80`

### Profiling Tools

* Valgrind
* Cachegrind

---

## Repository Structure

```text
.
├── baselinemultigpu/             # Reference Multi-GPU baseline implementations
│   ├── include/                  # Baseline matrix parsing and utility headers
│   └── testmtx/                  # Small example matrices
├── bin/                          # Generated executables
├── data/                         # Real SuiteSparse datasets (.mtx)
├── data-synt/                    # Synthetic datasets for Weak Scaling
├── deviceQuery/                  # CPU/GPU environment logs
├── doc/                          # Technical report and LaTeX material
├── generate_matrices.py          # Synthetic matrix generator
├── include/                      # Project headers
├── makefile                      # Build configuration
├── paper/                        # Reference literature
├── README.md                     # Project documentation
├── results/                      # Analysis scripts, CSVs and plots
├── src/                          # Source code (.c, .cu)
├── run_performance_multi.sh      # Main execution script
├── submit_all_multi.sh           # Strong Scaling batch launcher
└── submit_weak_scaling.sh        # Weak Scaling batch launcher
```

---

# Reproducing Benchmarks on the Cluster

## Phase 1: Access and Preparation

### 1. Connect to the University VPN

Use Global Protect:

```text
vpn-mfa.icts.unitn.it
```

### 2. Access the cluster

```bash
ssh username@baldo.disi.unitn.it
```

### 3. Clone the repository

```bash
git clone -b Deliverable2 https://github.com/Michael-Bernasconi/Sparse-Matrix-Vector-Multiplication.git

cd Sparse-Matrix-Vector-Multiplication
```

---

## Phase 2: Real Dataset Download and Setup

Create and populate the `data/` directory with matrices from SuiteSparse.

### Download datasets

```bash
mkdir -p data
cd data

wget https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_320ks.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_680ks.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Rucci/Rucci1.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/GHS_indef/boyd2.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Williams/webbase-1M.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Pajek/patents_main.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk17.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/DNVS/m_t1.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/HB/bcsstk16.tar.gz

wget https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_100ks.tar.gz

```

### Extract archives

```bash
for f in *.tar.gz; do
    tar -xzf "$f"
done

mv */*.mtx .
rm *.tar.gz
rm -rf */
```

---

## Phase 2.3: Synthetic Dataset Generation (Weak Scaling Targets)

Large synthetic workloads are generated locally and therefore are not stored in the repository.

Return to the project root and execute:

```bash
cd ~/Sparse-Matrix-Vector-Multiplication

python3 generate_matrices.py
```

The script creates benchmark datasets inside:

```text
data-synt/
```

Two workload families are generated:

### Balanced Matrices    (da cambiare)

* Uniform row distribution
* Constant non-zero density
* Inspired by the structural behavior of `rajat31`

### Imbalanced Matrices  (da cambiare)

* Heavy-tail distribution
* Approximately 5% of rows contain 80% of total non-zeros
* Inspired by the irregularity observed in `FullChip`

Datasets are produced for experiments targeting:

* 1 GPU
* 2 GPUs
* 4 GPUs

allowing Weak Scaling studies where problem size grows proportionally to the number of devices.

---

## Phase 3: Compilation and Execution

Move to the project root:

```bash
cd ~/Sparse-Matrix-Vector-Multiplication
```

### Load the environment

```bash
module purge

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
```

### Grant execution permissions

```bash
chmod +x run_performance_multi.sh
chmod +x submit_all_multi.sh
chmod +x submit_weak_scaling.sh
```

### Build the project

```bash
make clean
make
```

---

## Running Benchmark Campaigns

### Strong Scaling (Real Matrices)

Evaluate performance on SuiteSparse datasets using:

```bash
./submit_all_multi.sh
```

The campaign executes experiments across:

* 1 GPU
* 2 GPUs
* 4 GPUs

while keeping the problem size fixed.

### Weak Scaling (Synthetic Matrices)

Evaluate resource-to-problem growth behavior using:

```bash
./submit_weak_scaling.sh
```

The campaign pairs:

| GPUs | Problem Size |
| ---- | ------------ |
| 1    | Base         |
| 2    | 2× Base      |
| 4    | 4× Base      |

for both balanced and imbalanced synthetic datasets.

---

## Graphs and Tables Generation

After all SLURM jobs complete, move to the analysis directory:

```bash
cd results
```

### Aggregate benchmark logs

```bash
python3 analyze-result.py
```

This produces structured CSV files containing:

* GFLOPS
* Bandwidth
* Kernel Time
* Cache Statistics

### Generate plots

```bash
python3 gflops-bw-tts-kerneltime-cache.py
```

The generated artifacts are stored in:

```text
results/
├── plots/
└── tables/
```

---

## Output Artifacts

### Plots

Located in:

```text
results/plots/
```

Including:

* GFLOPS comparisons
* Effective bandwidth comparisons
* Kernel execution times
* Scaling efficiency
* Cache behavior visualizations

### Tables

Located in:

```text
results/tables/
```

Containing aggregated benchmark statistics and summary metrics.

---

