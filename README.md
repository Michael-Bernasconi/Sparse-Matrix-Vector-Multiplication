# Distributed Multi-GPU Sparse Matrix-Vector Multiplication (SpMV) via 1D Cyclic Partitioning and GPU-Aware Communication
**Course:** GPU-Computing-2026

**Author:** Michael Bernasconi ([michael.bernasconi@studenti.unitn.it](mailto:michael.bernasconi@studenti.unitn.it)) - Student ID: 267681

---

## Project Overview & Architectural Methodology

This repository provides a high-performance, distributed multi-GPU framework for **Sparse Matrix-Vector Multiplication ($y = A \times x$)**, engineered specifically for heterogeneous cluster environments.

The original project codebase lacked multi-GPU support and relied on a simple 1D contiguous row-block layout, which would lead to significant load imbalances when handling highly irregular datasets in parallel. 

To overcome this, this module transitions the architecture into a fully distributed environment, implementing a **1D cyclic row-partitioning scheme**:

$$\text{owner}(i) = i \pmod{\text{size}}$$

*where $\text{size}$ represents the total number of MPI ranks.*

This structural change guarantees a uniform distribution of non-zero elements across all distributed hardware accelerators using standard MPI and CUDA paradigms.
The application architecture was evolved from scratch adhering strictly to **Foster’s Four-Stage Parallel Design Methodology**:

1. **Partitioning:** Finely grained domain decomposition is performed on the rows of the sparse matrix $A$. Rank 0 streams the Matrix Market file and interleaves entries across cluster ranks following a modular distribution.
2. **Communication (Ghost Entries Discovery):** Rather than blindly broadcasting the entire multiplier vector $x$ via expensive global collectives, a specialized metadata pre-pass scans local column indices to isolate non-local vector elements requested by local non-zero coordinates. These are packed into optimized **Ghost Receive and Send Descriptors**.
3. **Agglomeration:** Assigned rows are compacted locally into native sparse formats (CSR, COO, or specialized variants), translating global coordinates into relative, zero-overhead localized index spaces.
4. **Mapping:** Each independent MPI process is pinned to a dedicated hardware device (**NVIDIA A30 GPU**), orchestrating localized execution patterns inside dedicated, non-interfering CUDA streams.

---

## Provided Implementations & Kernel Design

The repository incorporates several low-overhead parallel execution paths for comparison:

* **GPU SpMV COO:** Multi-GPU CUDA kernel utilizing coordinate list tracking, well-suited for extremely irregular matrices.
* **GPU SpMV CSR:** Multi-GPU native CUDA kernel mapping standard Compressed Sparse Row structures.
* **GPU SpMV CSR-Vector:** An optimized, warp-coalesced variant assigning one 32-thread Warp per matrix row. This effectively neutralizes control-flow divergence and ensures strict memory coalescence across high-bandwidth memory interfaces.
* **GPU cuSPARSE Baseline:** A high-throughput vendor reference implementation wrapped around NVIDIA’s proprietary `cuSPARSE` multi-GPU streaming library.
* **Baseline Multi-GPU:** A provided unoptimized reference layout utilizing global collective vector broadcasts (`MPI_Bcast`), used as a baseline to highlight the scaling advantages of sparse point-to-point transfers.

---

## High-Performance GPU-Aware Interconnect Overlay

To maximize data ingestion throughput, the communication backbone completely bypasses traditional host-side staging (`cudaMemcpy` to RAM). The pipeline utilizes a **GPU-Aware MPI Overlay** that passes device pointers directly into non-blocking communication primitives (`MPI_Irecv` / `MPI_Isend`).

---

## Measured Metrics

The performance analysis infrastructure records and plots the following metrics:

1. **GFLOPS (Giga Floating-Point Operations Per Second)** – Raw computational throughput.
2. **Effective Bandwidth (GB/s)** – Utilized memory bandwidth across device high-bandwidth interfaces (HBM2).
3. **Execution Time (TTS & Kernel-Time)** – Total time to solution (including MPI synchronization) contrasted against pure execution time of the underlying CUDA kernels.
4. **Communication Overhead Breakdown** – Microsecond-level isolation of pure GPU computing steps vs. MPI communication buffer packaging and transport.
5. **Speedup & Parallel Efficiency** – Rigorous scaling assessments across Strong Scaling regimes (fixed matrix sizes on 1, 2, and 4 GPUs) and Weak Scaling workloads (proportional data scaling per device).

---

## Target Hardware (UniTN Cluster)

Benchmarks were designed for the University cluster (**edu01 node**, **edu-short partition**).

| Feature | Host (CPU) | Device (GPU) |
| --- | --- | --- |
| **Model** | Intel(R) Xeon(R) Silver 4309Y | NVIDIA A30 |
| **Architecture** | Ice Lake (x86_64) | Ampere (Compute Capability 8.0) |
| **Cores / SMs** | 16 Cores / 32 Threads (2 Sockets) | 56 SMs (3584 CUDA Cores) |
| **Clock Frequency** | 2.80 GHz (Base) / 3.60 GHz (Max) | 1.44 GHz (Boost) |
| **FP32 Performance** | 1.84 TFLOPS | 10.3 TFLOPS |
| **Memory Bandwidth** | 102.4 GB/s | 933.1 GB/s (HBM2) |
| **Global Memory** | N/A | 24 GB |
| **L1 Cache** | 768 KiB Data / 512 KiB Inst. | 192 KiB per SM |
| **L2 Cache** | 20 MiB | 24 MiB |
| **L3 Cache** | 24 MiB | N/A |
| **Shared Memory** | N/A | 48 KiB (up to 100 KiB per SM) |
| **Thread Limits** | 2 Threads/Core | 1024 Threads/Block |
| **Warp Size** | N/A | 32 |

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

---

## Repository Structure

```text
├── baselinemultigpu/             # Reference Multi-GPU baseline implementations
├── bin/                          # Generated executables (COO, CSR, CSR-Vector, cuSparse)
├── data/                         # Real SuiteSparse datasets (.mtx) [To be downloaded]
├── data-synt/                    # Synthetic datasets for Weak Scaling [To be generated]
├── deviceQuery/                  # CPU/GPU environment logs
├── doc/                          # Technical report and LaTeX material
├── generate_matrices.py          # Synthetic matrix generator script
├── include/                      # Project headers (CUDA timers, formats, etc.)
├── makefile                      # Build configuration
├── obj/                          # Compiled object files
├── README.md                     # Project documentation
├── results/                      # Analysis output and plot generation
│   ├── analyze-result-strong.py  # Log parser for Strong Scaling
│   ├── analyze-result-weak.py    # Log parser for Weak Scaling
│   ├── plots.py                  # Script to generate final PDF charts
│   ├── multi_gpu_analysis_*.csv  # Aggregated CSV reports
│   ├── plots/                    # Generated PDF plots (Speedup, GFLOPS, Breakdown, etc.)
│   └── tables/                   # Generated CSV summary tables
├── src/                          # Source code (.c, .cu)
├── run_performance_multi.sh      # Main execution script wrapper
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

Large synthetic workloads are generated locally to match the multi-GPU environment and therefore are not stored directly in the repository.

Return to the project root and execute:

```bash
cd ~/Sparse-Matrix-Vector-Multiplication

python3 generate_matrices.py

```

The script creates benchmark datasets inside:

```text
data-synt/

```

Two workload families are generated to evaluate the system under different sparsity profiles:

### 1. ASIC-like Synthetic Matrices (`synth_asic`)

* Synthetic structures inspired by the standard `ASIC` matrix patterns.
* Designed to test how the algorithms scale over moderately structured/balanced non-zero distribution profiles.

### 2. Boyd2-like Synthetic Matrices (`synth_boyd2`)

* Synthetic structures inspired by the topology of the `boyd2` dataset.
* Provides a different structural challenge for the distributed MPI algorithms, allowing performance observation over varying densities.

Datasets are dynamically produced for experiments targeting:

* **1 GPU** (Base Size)
* **2 GPUs** (2x Problem Size)
* **4 GPUs** (4x Problem Size)

This permits rigorous Weak Scaling studies where the global problem size grows proportionally to the number of computing devices.

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
| --- | --- |
| 1 | Base |
| 2 | 2× Base |
| 4 | 4× Base |

for both the `synth_asic` and `synth_boyd2` dataset families.

---

## Graphs and Tables Generation

After all SLURM jobs complete, move to the results analysis directory:

```bash
cd results

```

### Aggregate benchmark logs
You must delete these logs for the analysis such that are sub-matrix and are not important for the statistics:

```bash
cd multi_gpu
cd run_1
cd 1gpu
rm -rf PERF_1GPU_boyd2_b.mtx.log
rm -rf PERF_1GPU_patents_main_appyear.mtx.log
rm -rf PERF_1GPU_patents_main_cat.mtx.log
rm -rf PERF_1GPU_patents_main_class.mtx.log
rm -rf PERF_1GPU_patents_main_date.mtx.log
rm -rf PERF_1GPU_patents_main_subcat.mtx.log
rm -rf PERF_1GPU_patents_main_year.mtx.log

cd ..

cd 2gpu
rm -rf PERF_2GPU_boyd2_b.mtx.log
rm -rf PERF_2GPU_patents_main_appyear.mtx.log
rm -rf PERF_2GPU_patents_main_cat.mtx.log
rm -rf PERF_2GPU_patents_main_class.mtx.log
rm -rf PERF_2GPU_patents_main_date.mtx.log
rm -rf PERF_2GPU_patents_main_subcat.mtx.log
rm -rf PERF_2GPU_patents_main_year.mtx.log


cd ..

cd 4gpu
rm -rf PERF_4GPU_boyd2_b.mtx.log
rm -rf PERF_4GPU_patents_main_appyear.mtx.log
rm -rf PERF_4GPU_patents_main_cat.mtx.log
rm -rf PERF_4GPU_patents_main_class.mtx.log
rm -rf PERF_4GPU_patents_main_date.mtx.log
rm -rf PERF_4GPU_patents_main_subcat.mtx.log
rm -rf PERF_4GPU_patents_main_year.mtx.log

cd ..		
cd ..		
cd ..
```

Parse the output logs of both strong and weak scaling runs:

```bash
python3 analyze-result-strong.py

python3 analyze-result-weak.py

```

This produces structured CSV files (e.g., `multi_gpu_analysis_strong_report.csv`) containing all extracted metrics.

### Generate plots

Run the plotting script to build the visual artifacts:

```bash
python3 plots.py

```

---

## Output Artifacts

### Plots

Located in:

```text
results/plots/

```

Includes files like:

* `strong_individual_gflops.pdf`
* `strong_individual_speedup.pdf`
* `strong_comm_comp_breakdown.pdf`
* `strong_individual_efficiency.pdf`
* `strong_individual_time.pdf`
* `weak_individual_synthetic.pdf`

### Tables

Located in:

```text
results/tables/

```

Containing aggregated benchmark statistics, breakdown summaries, and scalability metrics.