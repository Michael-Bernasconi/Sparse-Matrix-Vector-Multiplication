import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

def parse_log(file_path):
    """
    Parses the SpMV benchmark log file and extracts metrics for each matrix and algorithm.
    """
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as f:
        content = f.read()

    matrices_data = {}
    # Split the content by Matrix name
    matrix_blocks = re.split(r'Matrix: ', content)[1:]

    for block in matrix_blocks:
        lines = block.split('\n')
        matrix_name = lines[0].strip()
        if not matrix_name: continue
        matrices_data[matrix_name] = {}

        # Split by benchmark section (identified by dashed lines)
        bench_blocks = re.split(r'-{10,}', block)
        
        for bench in bench_blocks:
            match_type = re.search(r'\[(.*?)\]', bench)
            if not match_type: continue
            bench_type = match_type.group(1)

            # Extract metrics using regex
            gflops = re.search(r'GFLOPS\s*:\s*([\d.]+)', bench)
            bw = re.search(r'BW\s*:\s*([\d.]+)', bench)
            tts = re.search(r'(?:TTS|Time-to-Solution)\s*:\s*([\d.]+)', bench)

            if gflops and bw and tts:
                matrices_data[matrix_name][bench_type] = {
                    'GFLOPS': float(gflops.group(1)),
                    'BW': float(bw.group(1)),
                    'TTS': float(tts.group(1))
                }
    return matrices_data

def plot_metrics_unified(data):
    """
    Generates unified bar charts for BW, GFLOPS, and TTS (Log Scale).
    """
    matrix_names = list(data.keys())
    metrics = ['BW', 'GFLOPS', 'TTS']
    
    # Strictly ordered benchmark types for the x-axis groups
    bench_types = [
        'cpu-SpMV-COO', 
        'cpu-SpMV-CSR', 
        'cuda-SpMV-COO', 
        'cuda-SpMV-CSR', 
        'cuda-SpMV-CSR-Vector', 
        'cuda-SpMV-cuSPARSE'
    ]
    
    # Professional Color Palette (Blues for CPU, Greens for GPU, Purple/Orange for Optimized/Vendor)
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#9467bd', '#ff7f0e']
    
    for metric in metrics:
        # High-resolution figure setup
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.suptitle(f'SpMV Unified Performance Analysis: {metric}', fontsize=22, fontweight='bold', y=0.98)

        x = np.arange(len(matrix_names))
        width = 0.13
        
        for j, b_type in enumerate(bench_types):
            values = [data[m][b_type][metric] if b_type in data[m] else 0 for m in matrix_names]
            offset = (j - len(bench_types)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=b_type, color=colors[j], edgecolor='black', linewidth=0.5)
            
            # Formatting labels on top of bars
            if metric == 'TTS':
                # Use scientific notation for very small values, standard for others
                labels = [f'{v:.1e}' if (0 < v < 0.01) else f'{v:.2f}' if v > 0 else '' for v in values]
                ax.bar_label(bars, labels=labels, padding=3, fontsize=8, rotation=90)
            else:
                ax.bar_label(bars, padding=3, fmt='%.2f', fontsize=8, rotation=90)

        # Axis labeling
        ylabel = metric
        if metric == 'BW': 
            ylabel += " (GB/s)"
        elif metric == 'GFLOPS':
            ylabel += " (GigaFLOPS)"
        elif metric == 'TTS': 
            ylabel += " (Seconds - Log Scale)"
            # Apply Logarithmic Scale to TTS to handle the massive range between CPU and GPU
            ax.set_yscale('log', nonpositive='clip')
        
        ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        
        # Matrix filenames labels (smaller font for better fit)
        ax.set_xticklabels(matrix_names, rotation=30, ha='right', fontsize=9)
        
        # Grid and Ticks precision for Linear Scales
        if metric != 'TTS':
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=20))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        
        # Horizontal legend placed above the plot
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), fontsize='small', ncol=6, frameon=True, shadow=True)
        
        # Visual styling
        ax.grid(axis='y', which='major', linestyle='--', alpha=0.6)
        ax.grid(axis='y', which='minor', linestyle=':', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Exporting high-quality PNG
        filename = f"SpMV_{metric}_Analysis_Report.png"
        plt.savefig(filename, dpi=300)
        print(f"Report saved: {filename}")
        plt.show()

# --- UPDATE THIS PATH TO YOUR LOG FILE ---
log_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\benchmark\flops-bw-tts\report_flops-bw-tts-cpu-gpu.log'

if os.path.exists(log_path):
    results = parse_log(log_path)
    if results:
        plot_metrics_unified(results)
    else:
        print("Error: No data found in the log file.")
else:
    print(f"Error: Log file not found at {log_path}")