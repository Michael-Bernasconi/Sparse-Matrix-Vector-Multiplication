import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Path Settings
# Define the source path for the analysis report and the destination for the generated plots
csv_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\tables\1_deliverable_analysis_report.csv'
output_dir = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\plots'

# Ensure the output directory exists; if not, create it
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. Data Loading
# Read the CSV file containing the SpMV performance metrics
df = pd.read_csv(csv_path)

# 3. Paper-style Visual Settings
# Define a professional color palette used in scientific publications
vibrant_colors = ["#ff4649", "#fffc35", "#76ff36", "#2ea8ff", "#7A2EFF"]

def save_spmv_plot_minimal(data, metric, title, filename, ylabel, log_scale=False):
    """
    Utility function to create and save standardized bar plots for SpMV performance.
    """
    plt.figure(figsize=(10, 3.5)) 
    
    # Apply a clean white style with soft grid lines
    sns.set_style("white", {'axes.grid': True, 'grid.color': '#eeeeee', 'grid.linestyle': '-'})
    
    # Create the bar plot comparing different configurations across matrices
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
    
    # Set titles and labels with specific font sizes and padding
    plt.title(title, fontsize=12, fontweight='bold', pad=35, color="#000000") 
    plt.xlabel('Matrix Name', fontsize=10, color="#000000")
    plt.ylabel(ylabel, fontsize=10, color="#000000")
    
    # Format ticks for better readability
    plt.xticks(rotation=20, ha='right', fontsize=9, color="#000000")
    plt.yticks(fontsize=9, color="#000000")
    
    # Configure the legend to appear above the plot area (scientific paper style)
    plt.legend(
        title='Algorithms', 
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.3), 
        ncol=5,
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        columnspacing=1.2,
        handletextpad=0.5
    )
    
    # Apply log scale if requested, otherwise expand Y-axis to prevent label clipping
    if log_scale:
        ax.set_yscale("log")
    else:
        curr_ylim = ax.get_ylim()
        ax.set_ylim(0, max(curr_ylim[1], 1) * 1.3)

    # Enhance spine visibility for a "boxed" appearance
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#000000')
        spine.set_linewidth(0.8)

    ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.8, color='#000000', left=True, bottom=True)

    # Add numeric labels on top of each bar (vertical rotation for clarity)
    for p in ax.patches:
        val = p.get_height()
        if val > 0:
            label = f'{val:.1f}' if val >= 1 else f'{val:.2f}'
            ax.text(
                p.get_x() + p.get_width() / 2., 
                val + (ax.get_ylim()[1] * 0.02), 
                label,
                ha='center', va='bottom', fontsize=7, color='#000000', rotation=90
            )

    # Save the final figure with high DPI
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- 4. PERFORMANCE PLOTS GENERATION ---
# Filter out the "lite" configuration used for cache analysis to keep performance plots clean
df_perf = df[df['Config'] != 'cpu-SpMV-CSR-lite'].copy()

# Plot 1: Memory Bandwidth (GB/s)
save_spmv_plot_minimal(df_perf[df_perf['Avg BW (GB/s)'] >= 0], 'Avg BW (GB/s)', 
                      'Memory Bandwidth Performance', '1_del_bandwidth.png', 'GB/s')

# Plot 2: Throughput (GFLOPS)
save_spmv_plot_minimal(df_perf[df_perf['Avg GFLOPS'] >= 0], 'Avg GFLOPS', 
                      'Computational Throughput', '1_del_gflops.png', 'GFLOPS')

# Plot 3: Time To Solution (Seconds)
save_spmv_plot_minimal(df_perf[df_perf['Avg TTS (s)'] >= 0], 'Avg TTS (s)', 
                      'Time To Solution (TTS)', '1_del_tts.png', 'Seconds')

# --- 5. CACHE MISS VISUAL TABLE GENERATION (ENGLISH & PLOT STYLE) ---
# 1. Identify all unique matrices from the original report
all_matrices = df['Matrix'].unique()

# 2. Extract cache data (lite configuration from Cachegrind)
df_cache = df[df['Config'] == 'cpu-SpMV-CSR-lite'].copy()

# 3. Build the full table including matrices with missing data (Timeouts)
cache_table_data = pd.DataFrame({'Matrix Name': all_matrices})
merge_data = df_cache[['Matrix', 'D1 Miss %', 'LL Miss %']].rename(columns={'Matrix': 'Matrix Name'})
cache_df = pd.merge(cache_table_data, merge_data, on='Matrix Name', how='left')

# 4. English Labeling & Formatting
def format_cache_val(val):
    try:
        if pd.isna(val) or val == 'N/A': return "-"
        return f"{float(val):.1f}%"
    except:
        return "-"

cache_df['L1 Data Miss'] = cache_df['D1 Miss %'].apply(format_cache_val)
cache_df['LLC Miss'] = cache_df['LL Miss %'].apply(format_cache_val)

# Select only the required columns (Algorithm removed)
final_table_df = cache_df[['Matrix Name', 'L1 Data Miss', 'LLC Miss']]

# --- GRAPHICAL TABLE CREATION (MATPLOTLIB) ---
fig, ax = plt.subplots(figsize=(8, 5)) # Slightly narrower as we have fewer columns
ax.axis('tight')
ax.axis('off')

# Use the first color of your palette for the header
header_color = vibrant_colors[0] 

# Create the table
table = ax.table(
    cellText=final_table_df.values, 
    colLabels=final_table_df.columns, 
    cellLoc='center', 
    loc='center',
    colColours=[header_color] * 3 # Adjusted for 3 columns
)

# Apply Styling
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

# Title positioned closer to the table (pad reduced from 30 to 10)
plt.title('CPU Cache Metrics Analysis (cpu-SpMV-CSR)', fontsize=12, fontweight='bold')

# Save the table as an image
plt.savefig(os.path.join(output_dir, '1_del_cache_table.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nCache table image saved in: {output_dir}")
print(f"\nAnalysis complete. Paper-style plots saved in: {output_dir}")