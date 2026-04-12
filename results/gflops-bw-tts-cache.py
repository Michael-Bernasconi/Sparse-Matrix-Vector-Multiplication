import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Path Settings
csv_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\plots-table\1_deliverable_analysis_report.csv'
output_dir = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\plots-table'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Data Loading & Cleaning
df = pd.read_csv(csv_path)
df['Config'] = df['Config'].replace('cpu-SpMV-CSR-lite', 'cpu-SpMV-CSR')

# 3. Light Professional Colors (Red, Yellow, Green, Blue, Pink - Soft versions)
light_colors = ["#ff7675", "#ffeaa7", "#55efc4", "#74b9ff", "#fd98c0"]

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False):
    # Setup ultra-clean white style
    plt.figure(figsize=(20, 10))
    sns.set_style("white", {'axes.grid': True, 'grid.color': '#f2f2f2', 'axes.edgecolor': '#404040'})
    
    # Create the grouped bar chart
    ax = sns.barplot(
        data=data, 
        x='Matrix', 
        y=metric, 
        hue='Config', 
        palette=light_colors, 
        edgecolor="#404040", 
        linewidth=0.8
    )
    
    # Titles and Labels
    plt.title(title, fontsize=26, fontweight='bold', pad=30)
    plt.xlabel('Matrix Name', fontsize=18, labelpad=15)
    plt.ylabel(ylabel, fontsize=18, labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    
    # LEGEND ON ONE ROW (Horizontal)
    plt.legend(
        title=None, 
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.15), # Moves it below the X-axis
        ncol=5,                      # Forces implementations on one row
        frameon=False,               # No box around legend
        fontsize=14
    )
    
    if log_scale:
        ax.set_yscale("log")

    # --- MINIMAL VERTICAL DATA LABELS (No boxes, no shadows) ---
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            if val < 0.001: label = f'{val:.2e}'
            elif val < 1: label = f'{val:.3f}'
            else: label = f'{val:.2f}'
            
            # Simple text without background box
            ax.text(
                p.get_x() + p.get_width() / 2., 
                val + (val * 0.05 if not log_scale else val * 0.2), # Dynamic spacing
                label,
                ha='center', 
                va='bottom', 
                fontsize=11, 
                fontweight='bold', 
                color='#333333',
                rotation=90
            )

    # Despine for a modern look (remove top and right borders)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Exported: {filename}")

# --- 1. BANDWIDTH ---
df_bw = df[df['Avg BW (GB/s)'] > 0].copy()
save_spmv_plot_minimal(df_bw, 'Avg BW (GB/s)', 'Memory Bandwidth Performance', 
                      'bandwidth_minimal.png', 'GB/s')

# --- 2. GFLOPS ---
df_gflops = df[df['Avg GFLOPS'] > 0].copy()
save_spmv_plot_minimal(df_gflops, 'Avg GFLOPS', 'Computational Throughput', 
                      'gflops_minimal.png', 'GFLOPS')

# --- 3. EXECUTION TIME (TTS) ---
df_tts = df[df['Avg TTS (s)'] > 0].copy()
save_spmv_plot_minimal(df_tts, 'Avg TTS (s)', 'Time To Solution (TTS)', 
                      'tts_minimal.png', 'Seconds', log_scale=True)

# --- 4. CACHE MISS RATE ---
df_cache = df[df['D1 Miss %'] != "N/A"].copy()
df_cache['D1 Miss %'] = pd.to_numeric(df_cache['D1 Miss %'])

plt.figure(figsize=(16, 8))
sns.set_style("white", {'axes.grid': True, 'grid.color': '#f2f2f2'})
ax = sns.barplot(data=df_cache, x='Matrix', y='D1 Miss %', color='#b2bec3', edgecolor="#404040")

plt.title('CPU L1 Cache Miss Rate', fontsize=24, fontweight='bold', pad=25)
plt.ylabel('Percentage (%)', fontsize=16)
plt.xticks(rotation=45, ha='right')

for p in ax.patches:
    ax.text(
        p.get_x() + p.get_width() / 2., 
        p.get_height() + 0.01, 
        f'{p.get_height():.2f}%', 
        ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=90
    )

sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cache_miss_minimal.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nMinimalist plots saved in: {output_dir}")