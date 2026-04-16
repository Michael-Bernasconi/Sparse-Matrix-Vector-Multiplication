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

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False):
    """
    Utility function to create and save standardized bar plots for SpMV performance.
    """
    plt.figure(figsize=(10, 5.0)) 
    
    sns.set_style("white", {'axes.grid': True, 'grid.color': '#eeeeee', 'grid.linestyle': '-'})
    
    ax = sns.barplot(
        data=data, 
        x='Matrix', 
        y=metric, 
        hue='Config', 
        palette=vibrant_colors, 
        edgecolor="#333333", 
        linewidth=0.6,
        width=0.85
    )
    
    # TITOLO: Alzato e centrato
    plt.title(title, fontsize=12, fontweight='bold', color="#000000", pad=45) 
    
    plt.xlabel('Matrix Name', fontsize=10, color="#000000")
    plt.ylabel(ylabel, fontsize=10, color="#000000")
    
    plt.xticks(rotation=20, ha='right', fontsize=9, color="#000000")
    plt.yticks(fontsize=9, color="#000000")
    
    # LEGENDA: Allineata orizzontalmente senza titolo interno
    leg = plt.legend(
        loc='upper center', 
        bbox_to_anchor=(0.55, 1.15), 
        ncol=5,
        frameon=False,
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.3
    )
    
    # SCRITTA "Algorithms:": Posizionata correttamente a sinistra
    plt.gcf().text(0.12, 0.88, 'Algorithms:', fontsize=9, fontweight='bold', color="#000000")
    
    if log_scale:
        ax.set_yscale("log")
        curr_ylim = ax.get_ylim()
        ax.set_ylim(curr_ylim[0], curr_ylim[1] * 6) 
    else:
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, max(curr_ylim[1], 1) * 1.15)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#000000')
        spine.set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8, color='#000000', left=True, bottom=True)

    # Label sopra le barre (rotazione 90°)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            label = f'{val:.1e}' if val < 0.01 else (f'{val:.1f}' if val >= 1 else f'{val:.3f}')
            
            if log_scale:
                y_pos = val * 1.12
            else:
                y_pos = val + (ax.get_ylim()[1] * 0.01)

            ax.text(
                p.get_x() + p.get_width() / 2., 
                y_pos, 
                label,
                ha='center', va='bottom', fontsize=7, color='#000000', rotation=90
            )

    # Margini per evitare tagli e sovrapposizioni
    plt.subplots_adjust(top=0.82, bottom=0.15)
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- 4. PERFORMANCE PLOTS GENERATION ---
df_perf = df[df['Config'] != 'cpu-SpMV-CSR-lite'].copy()

save_spmv_plot_minimal(df_perf[df_perf['Avg BW (GB/s)'] >= 0], 'Avg BW (GB/s)', 
                      'Memory Bandwidth Performance', '1_del_bandwidth.png', 'GB/s')

save_spmv_plot_minimal(df_perf[df_perf['Avg GFLOPS'] >= 0], 'Avg GFLOPS', 
                      'Computational Throughput', '1_del_gflops.png', 'GFLOPS')

save_spmv_plot_minimal(df_perf[df_perf['Avg TTS (s)'] >= 0], 'Avg TTS (s)', 
                      'Time To Solution (TTS)', '1_del_tts.png', 'Seconds', log_scale=True)

save_spmv_plot_minimal(df_perf[df_perf['Avg Time (s)'] >= 0], 'Avg Time (s)', 
                      'Kernel Execution Time (Pure Compute)', '1_del_kernel_time.png', 'Seconds', log_scale=True)

# --- 5. CACHE MISS VISUAL TABLE GENERATION ---
all_matrices = df['Matrix'].unique()
df_cache = df[df['Config'] == 'cpu-SpMV-CSR-lite'].copy()
cache_table_data = pd.DataFrame({'Matrix Name': all_matrices})
merge_data = df_cache[['Matrix', 'D1 Miss %', 'LL Miss %']].rename(columns={'Matrix': 'Matrix Name'})
cache_df = pd.merge(cache_table_data, merge_data, on='Matrix Name', how='left')

def format_cache_val(val):
    try:
        if pd.isna(val) or val == 'N/A': return "-"
        return f"{float(val):.1f}%"
    except:
        return "-"

cache_df['L1 Data Miss'] = cache_df['D1 Miss %'].apply(format_cache_val)
cache_df['LLC Miss'] = cache_df['LL Miss %'].apply(format_cache_val)
final_table_df = cache_df[['Matrix Name', 'L1 Data Miss', 'LLC Miss']]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axis('tight')
ax.axis('off')
header_color = vibrant_colors[0] 

table = ax.table(
    cellText=final_table_df.values, 
    colLabels=final_table_df.columns, 
    cellLoc='center', 
    loc='center',
    colColours=[header_color] * 3
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8) 

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#333333')
    cell.set_linewidth(0.6)
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_text_props(color='#000000')
        if cell.get_text().get_text() == "-":
            cell.set_text_props(color='#999999')

plt.title('CPU Cache Metrics Analysis (cpu-SpMV-CSR)', fontsize=12, fontweight='bold')
plt.savefig(os.path.join(output_dir, '1_del_cache_table.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Analysis complete. All plots saved in: {output_dir}")