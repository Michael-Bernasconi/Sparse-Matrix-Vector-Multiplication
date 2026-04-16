import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Path Settings
csv_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\tables\1_deliverable_analysis_report.csv'
output_dir = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\plots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Data Loading
df = pd.read_csv(csv_path)

# 3. Paper-style Visual Settings
vibrant_colors = ["#ff4649", "#fffc35", "#76ff36", "#2ea8ff", "#7A2EFF"]

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False, log_lower_limit=1e-4):
    """
    Final optimized version:
    - TTS starts at 10^-1
    - Kernel Time starts at 10^-4
    - Clean decimal labels (no scientific notation like 'e-04' on bars)
    - Full chart box with black borders
    """
    plt.figure(figsize=(12, 6.0)) 
    
    # Light background grid
    sns.set_style("whitegrid", {'grid.color': '#f0f0f0'})
    
    ax = sns.barplot(
        data=data, 
        x='Matrix', 
        y=metric, 
        hue='Config', 
        palette=vibrant_colors, 
        edgecolor="#333333", 
        linewidth=0.7,
        width=0.8,
        gap=0.05 
    )
    
    # TITLE AND AXIS LABELS
    plt.title(title, fontsize=14, fontweight='bold', color="#000000", pad=55) 
    plt.xlabel('Sparse Matrix Name', fontsize=11, fontweight='bold', labelpad=10)
    plt.ylabel(ylabel, fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=25, ha='right', fontsize=10, color="#000000")
    plt.yticks(fontsize=10, color="#000000")
    
    # LEGEND AND "ALGORITHMS" TEXT
    plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.14), 
        ncol=5,
        frameon=False,
        fontsize=9,
        columnspacing=1.5
    )
    
    # AXIS LIMITS AND LOG SCALE
    if log_scale:
        ax.set_yscale("log")
        curr_ylim = ax.get_ylim()
        # Apply specific lower limit (10^-1 for TTS, 10^-4 for Kernel)
        ax.set_ylim(log_lower_limit, curr_ylim[1] * 8) 
    else:
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, max(curr_ylim[1], 1) * 1.25)

    # MAINTAIN BORDERS (FULL BOX)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#333333')
        spine.set_linewidth(0.8)

    # LABELS ABOVE BARS (DECIMAL LOGIC)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            # Fixed decimal formatting to avoid messy scientific notation on bars
            if val < 0.001:
                label = f'{val:.4f}' 
            elif val < 1:
                label = f'{val:.3f}'
            else:
                label = f'{val:.2f}'
            
            if log_scale:
                # Logarithmic offset to prevent text overlapping the bar
                y_pos = val * 1.15 
            else:
                y_pos = val + (ax.get_ylim()[1] * 0.02)

            ax.text(
                p.get_x() + p.get_width() / 2., 
                y_pos, 
                label,
                ha='center', va='bottom', fontsize=8, color='#000000', 
                rotation=90, fontweight='medium'
            )

    # Final margin adjustments
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.08, right=0.98)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- EXECUTION ---
df_perf = df[df['Config'] != 'cpu-SpMV-CSR-lite'].copy()

# 1. Memory Bandwidth
save_spmv_plot_minimal(df_perf[df_perf['Avg BW (GB/s)'] >= 0], 'Avg BW (GB/s)', 
                      'Memory Bandwidth Performance', '1_del_bandwidth.png', 'Bandwidth (GB/s)')

# 2. Computational Throughput
save_spmv_plot_minimal(df_perf[df_perf['Avg GFLOPS'] >= 0], 'Avg GFLOPS', 
                      'Computational Throughput', '1_del_gflops.png', 'Throughput (GFLOPS)')

# 3. Total Time To Solution (Log scale starting from 10^-1 / 0.1)
save_spmv_plot_minimal(df_perf[df_perf['Avg TTS (s)'] >= 0], 'Avg TTS (s)', 
                      'Total Time To Solution (TTS)', '1_del_tts.png', 'Time (Seconds)', 
                      log_scale=True, log_lower_limit=1e-1)

# 4. Kernel Execution Time (Log scale starting from 10^-4 / 0.0001)
save_spmv_plot_minimal(df_perf[df_perf['Avg Time (s)'] >= 0], 'Avg Time (s)', 
                      'Kernel Execution Time (Pure Compute)', '1_del_kernel_time.png', 'Time (Seconds)', 
                      log_scale=True, log_lower_limit=1e-4)

print(f"Analysis complete. All plots saved successfully in: {output_dir}")