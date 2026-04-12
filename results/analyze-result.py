import os
import re
import csv
import statistics
from collections import defaultdict

# Setup paths
BASE_DIR = "./results/single_matrices"
RUNS = ["run_1", "run_2", "run_3", "run_4", "run_5"]
OUTPUT_CSV = "1_deliverable_analysis_report.csv"

# Regex patterns to extract data
exe_name_re = re.compile(r"\[(.*?)\]")
avg_time_re = re.compile(r"Avg Time:\s+([0-9\.e+-]+)\s+s")
gflops_re = re.compile(r"GFLOPS\s+:\s+([0-9\.]+)")
bw_re = re.compile(r"BW\s+:\s+([0-9\.]+)\s+GB/s")
tts_re = re.compile(r"TTS\s+:\s+([0-9\.]+)\s+s") 

# Cache patterns
d1_miss_re = re.compile(r"D1\s+miss rate:\s+([0-9\.]+)\%")
ll_miss_re = re.compile(r"LL\s+miss rate:\s+([0-9\.]+)\%")

def analyze():
    # Structure: data[matrix][config][metric] -> list of values
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    print(f"Starting analysis of {len(RUNS)} runs...")

    for run in RUNS:
        run_path = os.path.join(BASE_DIR, run)
        if not os.path.exists(run_path):
            print(f"Warning: Folder {run} not found, skipping.")
            continue

        for filename in os.listdir(run_path):
            file_path = os.path.join(run_path, filename)
            
            # Identify Matrix Name
            parts = filename.split('_')
            if len(parts) < 2: continue
            matrix_name = parts[1]

            with open(file_path, 'r') as f:
                content = f.read()

                if filename.startswith("PERF_"):
                    sections = content.split('-----------------------')
                    for section in sections:
                        exe_match = exe_name_re.search(section)
                        if exe_match:
                            config = exe_match.group(1)
                            
                            time_val = avg_time_re.search(section)
                            gflops_val = gflops_re.search(section)
                            bw_val = bw_re.search(section)
                            tts_val = tts_re.search(section) # <-- Parsing TTS

                            if time_val: data[matrix_name][config]['Time'].append(float(time_val.group(1)))
                            if gflops_val: data[matrix_name][config]['GFLOPS'].append(float(gflops_val.group(1)))
                            if bw_val: data[matrix_name][config]['BW'].append(float(bw_val.group(1)))
                            if tts_val: data[matrix_name][config]['TTS'].append(float(tts_val.group(1)))

                elif filename.startswith("CACHE_"):
                    config = "cpu-SpMV-CSR-lite"
                    d1_match = d1_miss_re.search(content)
                    ll_match = ll_miss_re.search(content)
                    
                    if d1_match: data[matrix_name][config]['D1_Miss%'].append(float(d1_match.group(1)))
                    if ll_match: data[matrix_name][config]['LL_Miss%'].append(float(ll_match.group(1)))

    # Write Results to CSV
    with open(OUTPUT_CSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "Matrix", "Config", "Avg Time (s)", "StdDev Time", "Avg GFLOPS", "Avg BW (GB/s)", "Avg TTS (s)", "D1 Miss %", "LL Miss %", "Runs Found"
        ])

        for matrix in sorted(data.keys()):
            for config in sorted(data[matrix].keys()):
                metrics = data[matrix][config]
                
                # Performance metrics
                times = metrics['Time']
                avg_t = statistics.mean(times) if times else 0
                stdev_t = statistics.stdev(times) if len(times) > 1 else 0
                
                avg_gflops = statistics.mean(metrics['GFLOPS']) if metrics['GFLOPS'] else 0
                avg_bw = statistics.mean(metrics['BW']) if metrics['BW'] else 0
                avg_tts = statistics.mean(metrics['TTS']) if metrics['TTS'] else 0 # <-- Media TTS
                
                # Cache metrics
                avg_d1 = statistics.mean(metrics['D1_Miss%']) if metrics['D1_Miss%'] else "N/A"
                avg_ll = statistics.mean(metrics['LL_Miss%']) if metrics['LL_Miss%'] else "N/A"
                
                run_count = len(times) if times else len(metrics['D1_Miss%'])

                writer.writerow([
                    matrix, config, f"{avg_t:.6e}", f"{stdev_t:.6e}", 
                    round(avg_gflops, 4), round(avg_bw, 4), round(avg_tts, 4), avg_d1, avg_ll, run_count
                ])

    print(f"Analysis complete! Report saved in: {OUTPUT_CSV}")

if __name__ == "__main__":
    analyze()