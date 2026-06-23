import os
import re
import csv
from collections import defaultdict

# Find the absolute directory where THIS script resides (e.g., /home/michael/.../results)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the bulletproof absolute path to the weak scaling logs
BASE_DIR = os.path.join(SCRIPT_DIR, "multi_gpu", "weak_scaling", "run_1")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "multi_gpu_analysis_weak_report.csv")

# Configuration of expected GPUs
GPU_CONFIGS = [1, 2, 4]

# Line-by-line Regex patterns
kernel_header_re = re.compile(r"^\[(.*?) - (\d+) GPUs?\]")
avg_time_re = re.compile(r"Avg Time:\s+([0-9\.e+-]+)\s+s")
comm_time_re = re.compile(r"Comm Time:\s+([0-9\.e+-]+)\s+s")
comp_time_re = re.compile(r"Comp Time:\s+([0-9\.e+-]+)\s+s")
gflops_re = re.compile(r"GFLOPS\s+:\s+([0-9\.e+-]+)")
bw_re = re.compile(r"BW\s+:\s+([0-9\.e+-]+)\s+GB/s")

def analyze():
    # Structure: data[matrix_family][kernel][gpus] = { 'Time': 0.0, 'Comm': 0.0, 'Comp': 0.0, 'GFLOPS': 0.0, 'BW': 0.0 }
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    if not os.path.exists(BASE_DIR):
        print(f"Error: Absolute directory not found: {BASE_DIR}")
        return

    print(f"Starting weak scaling log analysis in: {BASE_DIR}...")

    # Recursive scan of folders 1gpu, 2gpu, 4gpu under weak_scaling
    for gpu_folder in os.listdir(BASE_DIR):
        if not gpu_folder.endswith("gpu"):
            continue
        
        gpus_num = int(gpu_folder.replace("gpu", ""))
        gpu_path = os.path.join(BASE_DIR, gpu_folder)
        
        if not os.path.isdir(gpu_path):
            continue

        for filename in os.listdir(gpu_path):
            if not filename.startswith("PERF_") or not filename.endswith(".log"):
                continue
            
            # Clean the raw matrix name from the log file name
            # e.g., "PERF_2GPU_synth_asic_2gpu.mtx.log" -> "synth_asic_2gpu"
            raw_matrix_name = filename.replace(f"PERF_{gpus_num}GPU_", "").replace(".mtx.log", "").replace(".log", "")
            raw_matrix_name = raw_matrix_name.replace("prof-SpMV-baseline", "SpMV-baseline")
            
            # Extract the Matrix Family name for weak scaling grouping
            # e.g., "synth_asic_2gpu" -> "synth_asic", "synth_boyd2_4gpu" -> "synth_boyd2"
            matrix_family = re.sub(r'_\d+gpu$', '', raw_matrix_name)

            file_path = os.path.join(gpu_path, filename)
            current_kernel = None
            
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    
                    # Intercept the beginning of a kernel block (e.g., [cuda-SpMV-CSR-multi - 2 GPUs])
                    header_match = kernel_header_re.match(line)
                    if header_match:
                        current_kernel = header_match.group(1).strip()
                        current_kernel = current_kernel.replace("prof-SpMV-baseline", "SpMV-baseline")
                        continue
                    
                    # If inside a valid kernel block, extract metrics line by line
                    if current_kernel:
                        time_match = avg_time_re.search(line)
                        comm_match = comm_time_re.search(line)
                        comp_match = comp_time_re.search(line)
                        gflops_match = gflops_re.search(line)
                        bw_match = bw_re.search(line)
                        
                        if time_match:
                            data[matrix_family][current_kernel][gpus_num]['Time'] = float(time_match.group(1))
                        if comm_match:
                            data[matrix_family][current_kernel][gpus_num]['Comm'] = float(comm_match.group(1))
                        if comp_match:
                            data[matrix_family][current_kernel][gpus_num]['Comp'] = float(comp_match.group(1))
                        if gflops_match:
                            data[matrix_family][current_kernel][gpus_num]['GFLOPS'] = float(gflops_match.group(1))
                        if bw_match:
                            data[matrix_family][current_kernel][gpus_num]['BW'] = float(bw_match.group(1))

    # Controlled writing into the Weak Scaling CSV file
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Matrix Family", "Kernel", "GPUs", 
            "Total Time (s)", "Comm Time (s)", "Comp Time (s)", 
            "GFLOPS", "BW (GB/s)", 
            "Weak Speedup", "Weak Efficiency"
        ])

        for family in sorted(data.keys()):
            for kernel in sorted(data[family].keys()):
                
                # Weak Scaling Reference: baseline is 1 GPU time on the 1 GPU problem size
                t_1gpu = data[family][kernel].get(1, {}).get('Time', None)
                
                for gpus in GPU_CONFIGS:
                    metrics = data[family][kernel].get(gpus, {})
                    
                    if not metrics:
                        continue
                    
                    total_t = metrics.get('Time', 0.0)
                    comm_t = metrics.get('Comm', 0.0)
                    
                    # Force communication time to 0.0 on 1 GPU configuration
                    if gpus == 1:
                        comm_t = 0.0
                        
                    comp_t = metrics.get('Comp', total_t)
                    gflops = metrics.get('GFLOPS', 0.0)
                    bw = metrics.get('BW', 0.0)
                    
                    # Weak Scaling Formulas Calculation:
                    # Weak Speedup = T1 / Tn
                    # Weak Efficiency = Weak Speedup (Ideal value is 1.0)
                    if t_1gpu and total_t > 0:
                        weak_speedup = t_1gpu / total_t
                        weak_efficiency = weak_speedup
                    else:
                        weak_speedup = 1.0 if total_t > 0 else 0.0
                        weak_efficiency = 1.0 if total_t > 0 else 0.0

                    writer.writerow([
                        family, 
                        kernel, 
                        gpus, 
                        f"{total_t:.6e}", 
                        f"{comm_t:.6e}", 
                        f"{comp_t:.6e}", 
                        round(gflops, 4), 
                        round(bw, 4), 
                        round(weak_speedup, 2), 
                        round(weak_efficiency, 2)
                    ])

    print(f"\nWeak Scaling Analysis completed successfully!")
    print(f"Generated CSV File: {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze()