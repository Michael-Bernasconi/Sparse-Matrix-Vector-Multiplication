import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.ticker as ticker

# Suppress minor matplotlib layout warnings for a clean CLI output
warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================================
# PATH CONFIGURATION
# =========================================================================
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

STRONG_CSV = os.path.join(RESULTS_DIR, "multi_gpu_analysis_strong_report.csv")
WEAK_CSV = os.path.join(RESULTS_DIR, "multi_gpu_analysis_weak_report.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# =========================================================================
# GLOBAL GRAPHICAL STYLING (Optimized for Ultra-Compact Page Budget)
# =========================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 7.5,
    'axes.labelsize': 8.0,
    'axes.titlesize': 8.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'grid.linewidth': 0.3,
    'grid.alpha': 0.4,
    'legend.fontsize': 7.0,
    'figure.titlesize': 9.5
})

def clean_kernel_name(name):
    """Simplifies kernel names for cleaner plot labels."""
    cleaned = name.replace('cuda-SpMV-', '').replace('-multi', '')
    if cleaned == 'prof-SpMV-baseline':
        return 'baseline'
    return cleaned

# Distinctive color palettes for different scaling environments
STRONG_COLORS = {1: '#ff3636', 2: '#20e1ff', 4: '#deff20'}
WEAK_COLORS = {1: '#13ff2b', 2: '#894aff', 4: '#ff23a4'}

# =========================================================================
# CORE STRONG SCALING PLOT GENERATOR
# =========================================================================
def generate_essential_strong_plots(df, metric_column, y_label, title, filename, is_log=False, reference_line=None):
    """Generates strong scaling bar charts (Time, GFLOPS, Speedup, Efficiency)."""
    df = df.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    kernels = sorted(df['Kernel_Clean'].unique())
    matrices = sorted(df['Matrix'].unique())
    
    # Shared X-axis layout to save vertical space
    fig, axs = plt.subplots(len(kernels), 1, figsize=(6.0, 4.8), sharex=True)
    if len(kernels) == 1:
        axs = [axs]

    x = np.arange(len(matrices))
    width = 0.24 

    for idx, kernel in enumerate(kernels):
        ax = axs[idx]
        df_k = df[df['Kernel_Clean'] == kernel]
        
        y_1 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 1)][metric_column].mean() for m in matrices]
        y_2 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 2)][metric_column].mean() for m in matrices]
        y_4 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 4)][metric_column].mean() for m in matrices]

        y_1 = [0 if np.isnan(v) else v for v in y_1]
        y_2 = [0 if np.isnan(v) else v for v in y_2]
        y_4 = [0 if np.isnan(v) else v for v in y_4]

        ax.bar(x - width, y_1, width, label='1 GPU', color=STRONG_COLORS[1], edgecolor='black', linewidth=0.2)
        ax.bar(x, y_2, width, label='2 GPUs', color=STRONG_COLORS[2], edgecolor='black', linewidth=0.2)
        ax.bar(x + width, y_4, width, label='4 GPUs', color=STRONG_COLORS[4], edgecolor='black', linewidth=0.2)

        # Plot ideal parallel efficiency line if provided
        if reference_line is not None:
            ax.axhline(reference_line, color='red', linestyle='--', linewidth=0.6, alpha=0.7)

        # Inline title to save space
        ax.text(0.015, 0.78, f"Kernel: {kernel}", transform=ax.transAxes, 
                fontweight='bold', fontsize=7.5, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
        ax.set_ylabel(y_label, fontsize=7.0)
        
        if is_log:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=8))
            ax.grid(True, which='both', linestyle=':', linewidth=0.2, alpha=0.5)
        else:
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6, prune='both'))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axs[-1].set_xlabel("Matrix File Name (.mtx)", labelpad=2)
    axs[-1].set_xticks(x)
    axs[-1].set_xticklabels(matrices, rotation=25, ha='right', fontsize=6.5)

    fig.suptitle(title, y=0.98, fontweight='bold', fontsize=9.5)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=True, handletextpad=0.3, columnspacing=1.0)
    
    plt.subplots_adjust(hspace=0.12) 
    fig.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# =========================================================================
# COMPUTATION VS COMMUNICATION BREAKDOWN PLOT GENERATOR
# =========================================================================
def generate_breakdown_plot(df):
    """Generates a stacked bar chart illustrating MPI vs CUDA overhead."""
    plt.clf()
    plt.close('all')
    
    df = df.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    # Filter for distributed runs
    df = df[df['GPUs'].isin([2, 4])]
    
    if 'Comp Time (s)' in df.columns and 'Comm Time (s)' in df.columns:
        grouped = df.groupby(['Kernel_Clean', 'GPUs'])[['Comp Time (s)', 'Comm Time (s)']].mean().reset_index()
        grouped['Total'] = grouped['Comp Time (s)'] + grouped['Comm Time (s)']
        grouped['Comp_Pct'] = (grouped['Comp Time (s)'] / grouped['Total']) * 100
        grouped['Comm_Pct'] = (grouped['Comm Time (s)'] / grouped['Total']) * 100
        
        labels = [f"{row['Kernel_Clean']} ({int(row['GPUs'])}G)" for _, row in grouped.iterrows()]
        x = np.arange(len(labels))
        
        # Compact canvas size
        fig, ax = plt.subplots(figsize=(5.8, 2.1))
        
        ax.bar(x, grouped['Comp_Pct'], width=0.38, label='Pure CUDA Compute %', color='#4a90e2', edgecolor='black', linewidth=0.3)
        ax.bar(x, grouped['Comm_Pct'], width=0.38, bottom=grouped['Comp_Pct'], label='MPI Ghost Comm %', color='#e056fd', edgecolor='black', linewidth=0.3)
        
        ax.set_ylabel('Percentage (%)', fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=6.5)
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
        ax.set_ylim(0, 100) 
        
        # Perfect geometric placement for legend and title
        handles, labels_leg = ax.get_legend_handles_labels()
        ax.legend(handles, labels_leg, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True, handletextpad=0.4)
        ax.set_title('Execution Cost Breakdown: Compute vs MPI Comm', fontweight='bold', fontsize=9.5, pad=24)
        
        fig.savefig(os.path.join(PLOTS_DIR, "strong_comm_comp_breakdown.pdf"), bbox_inches='tight', pad_inches=0.03)
        plt.close(fig)

# =========================================================================
# WEAK SCALING PROFILE PLOT GENERATOR
# =========================================================================
def generate_weak_plots(df):
    """Generates dual-column plots for weak scaling metrics."""
    df = df.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    kernels = sorted(df['Kernel_Clean'].unique())
    families = sorted(df['Matrix Family'].unique())
    
    fig, axs = plt.subplots(len(kernels), 2, figsize=(6.0, 4.5), sharex=True)
    if len(kernels) == 1:
        axs = np.array([axs])

    x = np.arange(len(families))
    width = 0.24

    for idx, kernel in enumerate(kernels):
        df_k = df[df['Kernel_Clean'] == kernel]
        
        ws_1 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 1)]['Weak Speedup'].mean() for f in families]
        ws_2 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 2)]['Weak Speedup'].mean() for f in families]
        ws_4 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 4)]['Weak Speedup'].mean() for f in families]
        
        we_1 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 1)]['Weak Efficiency'].mean() for f in families]
        we_2 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 2)]['Weak Efficiency'].mean() for f in families]
        we_4 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 4)]['Weak Efficiency'].mean() for f in families]

        # Weak Speedup Subplot
        ax1 = axs[idx, 0]
        ax1.bar(x - width, ws_1, width, label='1 GPU', color=WEAK_COLORS[1], edgecolor='black', linewidth=0.2)
        ax1.bar(x, ws_2, width, label='2 GPUs', color=WEAK_COLORS[2], edgecolor='black', linewidth=0.2)
        ax1.bar(x + width, ws_4, width, label='4 GPUs', color=WEAK_COLORS[4], edgecolor='black', linewidth=0.2)
        ax1.text(0.02, 0.76, f"Speedup: {kernel}", transform=ax1.transAxes, fontweight='bold', fontsize=7.0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
        ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

        # Weak Efficiency Subplot
        ax2 = axs[idx, 1]
        ax2.bar(x - width, we_1, width, label='1 GPU', color=WEAK_COLORS[1], edgecolor='black', linewidth=0.2)
        ax2.bar(x, we_2, width, label='2 GPUs', color=WEAK_COLORS[2], edgecolor='black', linewidth=0.2)
        ax2.bar(x + width, we_4, width, label='4 GPUs', color=WEAK_COLORS[4], edgecolor='black', linewidth=0.2)
        ax2.text(0.02, 0.76, f"Efficiency: {kernel}", transform=ax2.transAxes, fontweight='bold', fontsize=7.0, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.5))
        ax2.set_ylim(0, 1.25)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.6)
        ax2.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, prune='both'))
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axs[-1, 0].set_xlabel("Synthetic Family", labelpad=2)
    axs[-1, 1].set_xlabel("Synthetic Family", labelpad=2)
    axs[-1, 0].set_xticks(x)
    axs[-1, 0].set_xticklabels(families, rotation=15, fontsize=6.5)
    axs[-1, 1].set_xticks(x)
    axs[-1, 1].set_xticklabels(families, rotation=15, fontsize=6.5)

    fig.suptitle("Weak Scaling Profile per Synthetic Workload Family", y=0.98, fontweight='bold', fontsize=9.5)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=True, handletextpad=0.3, columnspacing=1.0)
    
    plt.subplots_adjust(hspace=0.12, wspace=0.15)
    fig.savefig(os.path.join(PLOTS_DIR, "weak_individual_synthetic.pdf"), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

# =========================================================================
# QUANTITATIVE SUMMARY STATISTICS EXPORTER
# =========================================================================
def generate_summary_tables_and_latex(df_strong):
    """Computes summary statistics and exports to CSV tables."""
    df = df_strong.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    
    perf_summary = df.groupby(['Kernel_Clean', 'GPUs'])[['Speedup', 'Efficiency']].mean().reset_index()
    table1 = perf_summary.pivot(index='Kernel_Clean', columns='GPUs', values=['Speedup', 'Efficiency'])
    table1.columns = [f"{v}_{c}G" for v, c in table1.columns]
    table1 = table1.round(4)
    table1.to_csv(os.path.join(TABLES_DIR, "strong_report_summary_efficiency_speedup.csv"))

    if 'Comp Time (s)' in df.columns and 'Comm Time (s)' in df.columns:
        df_multi = df[df['GPUs'].isin([2, 4])].copy()
        df_multi['Comp_Pct'] = (df_multi['Comp Time (s)'] / df_multi['Total Time (s)']) * 100
        df_multi['Comm_Pct'] = (df_multi['Comm Time (s)'] / df_multi['Total Time (s)']) * 100
        breakdown = df_multi.groupby(['Kernel_Clean', 'GPUs'])[['Comp_Pct', 'Comm_Pct']].mean().reset_index().round(2)
        breakdown.to_csv(os.path.join(TABLES_DIR, "strong_report_summary_comm_comp_breakdown.csv"), index=False)

# =========================================================================
# MAIN EXECUTIVE PIPELINE ORCHESTRATOR
# =========================================================================
def main():
    if os.path.exists(STRONG_CSV):
        df_strong = pd.read_csv(STRONG_CSV)
        print("[1/4] Rendering compressed strong scaling charts...")
        
        generate_essential_strong_plots(df_strong, 'Total Time (s)', 'Time (s)', 
                                        'Execution Time per SpMV (Real Matrices)', 
                                        'strong_individual_time.pdf', is_log=True)
        
        generate_essential_strong_plots(df_strong, 'GFLOPS', 'GFLOPS', 
                                        'Computational Throughput per Real Matrix', 
                                        'strong_individual_gflops.pdf', is_log=False)
        
        generate_essential_strong_plots(df_strong, 'Speedup', 'Speedup X', 
                                        'Strong Scaling Speedup Factor per Real Matrix', 
                                        'strong_individual_speedup.pdf', is_log=False)
        
        generate_essential_strong_plots(df_strong, 'Efficiency', 'Efficiency', 
                                        'Parallel Efficiency per Real Matrix', 
                                        'strong_individual_efficiency.pdf', is_log=False, reference_line=1.0)
        
        print("[2/4] Rendering computation vs communication breakdown chart...")
        generate_breakdown_plot(df_strong)
        
        print("[3/4] Extracting summary tables data...")
        generate_summary_tables_and_latex(df_strong)
    else:
        print(f"Error: Target configuration file '{STRONG_CSV}' not found.")

    if os.path.exists(WEAK_CSV):
        print("[4/4] Rendering synthetic weak scaling panels...")
        df_weak = pd.read_csv(WEAK_CSV)
        generate_weak_plots(df_weak)
    else:
        print(f"Error: Target configuration file '{WEAK_CSV}' not found.")
        
    print(f"\n[Execution Completed Successfully! All plots exported to '{PLOTS_DIR}'.]")

if __name__ == "__main__":
    main()