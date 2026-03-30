import pandas as pd
import re

raw_log = """
Benchmark Execution - Mon Mar 30 01:09:17 PM CEST 2026
Matrix: cavity12.mtx
[cpu-SpMV-CSR]
Avg Time: 4.245800e-05 s | GFLOPS: 3.5973 | BW: 15.1233 GB/s
[cpu-SpMV-COO]
Avg Time: 4.517400e-05 s | GFLOPS: 3.3810 | BW: 20.7460 GB/s
[cuda-SpMV-CSR]
GFLOPS  : 22.9116
BW      : 96.3218 GB/s
[cuda-SpMV-COO]
GFLOPS  : 36.3614
BW      : 223.1143 GB/s
Matrix: FEM_3D_thermal1.mtx
[cpu-SpMV-CSR]
Avg Time: 2.372740e-04 s | GFLOPS: 3.6307 | BW: 15.4272 GB/s
[cpu-SpMV-COO]
Avg Time: 2.575320e-04 s | GFLOPS: 3.3451 | BW: 20.6263 GB/s
[cuda-SpMV-CSR]
GFLOPS  : 155.8520
BW      : 662.2251 GB/s
[cuda-SpMV-COO]
GFLOPS  : 109.4004
BW      : 674.5672 GB/s
Matrix: scircuit.mtx
[cpu-SpMV-CSR]
Avg Time: 1.003230e-03 s | GFLOPS: 1.9117 | BW: 9.6922 GB/s
[cpu-SpMV-COO]
Avg Time: 6.822580e-04 s | GFLOPS: 2.8111 | BW: 18.8715 GB/s
[cuda-SpMV-CSR]
GFLOPS  : 116.3161
BW      : 589.7140 GB/s
[cuda-SpMV-COO]
GFLOPS  : 143.4749
BW      : 963.1878 GB/s
Matrix: tols4000.mtx
[cpu-SpMV-CSR]
Avg Time: 4.982000e-06 s | GFLOPS: 3.5263 | BW: 23.7407 GB/s
[cpu-SpMV-COO]
Avg Time: 5.328000e-06 s | GFLOPS: 3.2973 | BW: 25.7898 GB/s
[cuda-SpMV-CSR]
GFLOPS  : 2.6073
BW      : 17.5538 GB/s
[cuda-SpMV-COO]
GFLOPS  : 4.9757
BW      : 38.9175 GB/s
"""

def extract_and_save():
    #  find all possibile matrix
    matrix_blocks = re.split(r'Matrix:\s+', raw_log)[1:]
    data_list = []

    for block in matrix_blocks:
        # Extract name of matrix
        m_name = re.search(r'([^\s]+)', block).group(1).replace('./data/', '')
        
        # to find the configuration [device-SpMV-format]
        configs = re.findall(r'\[(cpu|cuda)-SpMV-(CSR|COO)\](.*?)(?=\[|Matrix:|$)', block, re.DOTALL)
        
        for device, fmt, content in configs:
            label = f"{'GPU' if device == 'cuda' else 'CPU'}-{fmt}"
            
            # Extract GFLOPS and BW
            gflops = re.search(r'GFLOPS\s*[:|]\s*([\d.]+)', content).group(1)
            bw = re.search(r'BW\s*[:|]\s*([\d.]+)', content).group(1)
            
            data_list.append({
                'Matrix': m_name,
                'Config': label,
                'GFLOPS': float(gflops),
                'BW': float(bw)
            })

    df = pd.DataFrame(data_list)
    cols = ['CPU-CSR', 'CPU-COO', 'GPU-CSR', 'GPU-COO']

    # Create CSV GFLOPS
    df_gflops = df.pivot(index='Matrix', columns='Config', values='GFLOPS')[cols]
    df_gflops.to_csv('benchmark_gflops.csv')
    
    # Create CSV BW
    df_bw = df.pivot(index='Matrix', columns='Config', values='BW')[cols]
    df_bw.to_csv('benchmark_bandwidth.csv')

    print("FATTO! I file 'benchmark_gflops.csv' e 'benchmark_bandwidth.csv' sono stati creati correttamente.")

if __name__ == "__main__":
    extract_and_save()