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

# 3. Vibrant Colors
vibrant_colors = ["#ff4649", "#fffc35", "#76ff36", "#2ea8ff", "#7A2EFF"]

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False):
    plt.figure(figsize=(20, 10))
    sns.set_style("white", {'axes.grid': True, 'grid.color': '#f8f9fa', 'grid.linestyle': '--'})
    
    ax = sns.barplot(
        data=data, 
        x='Matrix', 
        y=metric, 
        hue='Config', 
        palette=vibrant_colors, 
        edgecolor="#666666", 
        linewidth=1.0,
        width=0.8
    )
    
    plt.title(title, fontsize=24, fontweight='bold', pad=60, color="#333333") 
    plt.xlabel('Matrix Name', fontsize=16, labelpad=15, color="#555555")
    plt.ylabel(ylabel, fontsize=16, labelpad=15, color="#555555")
    plt.xticks(rotation=45, ha='right', fontsize=14, color="#333333")
    plt.yticks(fontsize=14, color="#333333")
    
    plt.legend(
        title=None, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        frameon=False,
        fontsize=14
    )
    
    if log_scale:
        ax.set_yscale("log")
    else:
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, max(curr_ylim[1], 1) * 1.15)

    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            if val < 0.001: label = f'{val:.2e}'
            elif val < 1: label = f'{val:.3f}'
            else: label = f'{val:.2f}'
            
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = y_range * 0.01 
            
            ax.text(
                p.get_x() + p.get_width() / 2., 
                val + offset, 
                label,
                ha='center', 
                va='bottom', 
                fontsize=11, 
                color='#333333', 
                rotation=90
            )

    sns.despine(left=False, bottom=False, top=True, right=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- 1, 2, 3. PERFORMANCES: ---
df_perf = df[df['Config'] != 'cpu-SpMV-CSR-lite'].copy()

save_spmv_plot_minimal(df_perf[df_perf['Avg BW (GB/s)'] >= 0], 'Avg BW (GB/s)', 
                      'Memory Bandwidth Performance', '1_del_bandwidth.png', 'GB/s')

save_spmv_plot_minimal(df_perf[df_perf['Avg GFLOPS'] >= 0], 'Avg GFLOPS', 
                      'Computational Throughput', '1_del_gflops.png', 'GFLOPS')

save_spmv_plot_minimal(df_perf[df_perf['Avg TTS (s)'] >= 0], 'Avg TTS (s)', 
                      'Time To Solution (TTS)', '1_del_tts.png', 'Seconds')


# --- 4. CACHE MISS RATE: only lite ---
all_matrices = df['Matrix'].unique()

df_cache = df[df['Config'] == 'cpu-SpMV-CSR-lite'].copy()
df_cache['Config'] = 'cpu-SpMV-CSR'
df_cache['D1 Miss %'] = pd.to_numeric(df_cache['D1 Miss %'], errors='coerce').fillna(0)

plt.figure(figsize=(16, 8))
sns.set_style("white", {'axes.grid': True, 'grid.color': '#f8f9fa', 'grid.linestyle': '--'})

# order
ax = sns.barplot(data=df_cache, x='Matrix', y='D1 Miss %', color='#2ea8ff', 
                 edgecolor="#666666", linewidth=1.0, width=0.5, order=all_matrices)

plt.title('CPU L1 Cache Miss Rate', fontsize=24, fontweight='bold', pad=30, color="#333333")
plt.ylabel('Percentage (%)', fontsize=16, color="#555555")
plt.xticks(rotation=45, ha='right', fontsize=14, color="#333333")
plt.yticks(fontsize=14, color="#333333")

ax.set_ylim(0, max(ax.get_ylim()[1], 0.3) * 1.1)

for p in ax.patches:
    val = p.get_height()
    if val > 0:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        ax.text(
            p.get_x() + p.get_width() / 2., 
            val + (y_range * 0.01), 
            f'{val:.2f}%', 
            ha='center', va='bottom', fontsize=12, color='#333333', rotation=0
        )

sns.despine(left=False, bottom=False, top=True, right=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '1_del_cache_cpu_miss.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved in: {output_dir}")