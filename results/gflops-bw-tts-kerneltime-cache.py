import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

# 1. Path Settings
csv_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\tables\1_deliverable_analysis_report.csv'
output_dir = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\plots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Data Loading and Cleaning
df = pd.read_csv(csv_path)
# Remove file extension for cleaner axis labels
df['Matrix'] = df['Matrix'].str.replace('.mtx', '', regex=False)

# 3. Paper-style Visual Settings
vibrant_colors = ["#ff4649", "#fffc35", "#76ff36", "#2ea8ff", "#7A2EFF"]

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False, log_lower_limit=0.1, specific_ticks=None, is_ms=False):
    """
    BALANCED VERSION: 
    Spaced-out matrix names for readability, but tight X-axis title to eliminate excess white space.
    """
    plot_data = data.copy()
    if is_ms:
        plot_data[metric] = plot_data[metric] * 1000

    plt.figure(figsize=(12, 6.2)) 
    sns.set_style("whitegrid", {'grid.color': '#f0f0f0'})
    
    ax = sns.barplot(
        data=plot_data, x='Matrix', y=metric, hue='Config', 
        palette=vibrant_colors, edgecolor="#333333", linewidth=0.7, width=0.8, gap=0.05 
    )
    
    # TITLES AND LABELS
    plt.title(title, fontsize=14, fontweight='bold', color="#000000", pad=45) 
    plt.ylabel(ylabel, fontsize=11, fontweight='bold')
    
    # --- SPACE OPTIMIZATION ---
    # Low labelpad (15) brings "Sparse Matrix Name" closer to the matrix labels
    plt.xlabel('Sparse Matrix Name', fontsize=11, fontweight='bold', labelpad=15) 
    
    # Keep pad at 20 to maintain distance between matrix names and the bars
    plt.xticks(rotation=30, ha='right', va='top', rotation_mode='anchor', fontsize=10, color="#000000")
    ax.tick_params(axis='x', which='major', length=0, pad=20) 
    
    # Maintain the long pointer line (-0.08) for a clean professional look
    unique_matrices = plot_data['Matrix'].unique()
    for i in range(len(unique_matrices)):
        ax.vlines(x=i, ymin=-0.08, ymax=0, transform=ax.get_xaxis_transform(), 
                  color='black', linewidth=1.5, clip_on=False)
    # ------------------------------

    # LOG SCALE MANAGEMENT
    if log_scale:
        ax.set_yscale("log")
        if specific_ticks:
            ax.set_yticks(specific_ticks)
            ax.get_yaxis().set_major_formatter(ticker.ScalarFormatter())
            ax.set_ylim(specific_ticks[0], specific_ticks[-1] * 3) 
    else:
        ax.set_ylim(0, plot_data[metric].max() * 1.3)

    # SPINES
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.0)

    # VALUE LABELS ON TOP OF BARS
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            label = f'{val:.4f}' if val < 0.01 else (f'{val:.3f}' if val < 1 else f'{val:.2f}')
            y_pos = val * 1.12 if log_scale else val + (ax.get_ylim()[1] * 0.02)
            ax.text(p.get_x() + p.get_width() / 2., y_pos, label,
                ha='center', va='bottom', fontsize=8, color='#000000', rotation=90)

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=9)
    
    # Adjust bottom margin to 0.28 to tighten the figure without clipping labels
    plt.subplots_adjust(top=0.85, bottom=0.28, left=0.08, right=0.98)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- EXECUTION ---
df_perf = df[df['Config'] != 'cpu-SpMV-CSR-lite'].copy()

# 1. Bandwidth Plot
save_spmv_plot_minimal(df_perf, 'Avg BW (GB/s)', 
                       'Memory Bandwidth Performance', '1_del_bandwidth.png', 'Bandwidth (GB/s)', 
                       log_scale=True, specific_ticks=[0.1, 1, 10, 100, 1000])

# 2. GFLOPS Plot
save_spmv_plot_minimal(df_perf, 'Avg GFLOPS', 
                       'Computational Throughput', '1_del_gflops.png', 'Throughput (GFLOPS)', 
                       log_scale=True, specific_ticks=[0.1, 1, 10, 100])

# 3. TTS Plot
save_spmv_plot_minimal(df_perf, 'Avg TTS (s)', 
                       'Total Time To Solution (TTS)', '1_del_tts.png', 'Time (Seconds)', 
                       log_scale=True, specific_ticks=[0.1, 1, 10, 100])

# 4. Kernel Time Plot
save_spmv_plot_minimal(df_perf, 'Avg Time (s)', 
                       'Kernel Execution Time', '1_del_kernel_time.png', 'Time (ms)', 
                       log_scale=True, is_ms=True, specific_ticks=[0.01, 0.1, 1, 10, 100, 1000])

print("Done! Plots generated with proper name spacing and minimized bottom white space.")