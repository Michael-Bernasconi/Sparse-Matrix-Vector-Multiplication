import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration real paths
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# SOLUZIONE ERRORE 2: I file CSV sono direttamente in RESULTS_DIR, non in TABLES_DIR
STRONG_CSV = os.path.join(RESULTS_DIR, "multi_gpu_analysis_strong_report.csv")
WEAK_CSV = os.path.join(RESULTS_DIR, "multi_gpu_analysis_weak_report.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Clean IEEE / ACM Style for required core plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'axes.titlesize': 10,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 11
})

def clean_kernel_name(name):
    cleaned = name.replace('cuda-SpMV-', '').replace('-multi', '')
    if cleaned == 'prof-SpMV-baseline':
        return 'baseline'
    return cleaned

# Custom color palette for STRONG SCALING
STRONG_COLORS = {
    1: '#ff3636',  # Red
    2: '#20e1ff',  # Blue
    4: '#deff20'   # Yellow
}

# Custom color palette for WEAK SCALING
WEAK_COLORS = {
    1: '#13ff2b',  # Green
    2: '#894aff',  # Purple
    4: '#ff23a4'   # Pink
}

# =========================================================================
# CORE STRONG SCALING PLOTS (Ultratight & Compressed Layout)
# =========================================================================
def generate_essential_strong_plots(df, metric_column, y_label, title, filename, is_log=False):
    df = df.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    kernels = sorted(df['Kernel_Clean'].unique())
    matrices = sorted(df['Matrix'].unique())
    
    fig, axs = plt.subplots(len(kernels), 1, figsize=(12, 2.3 * len(kernels)))
    if len(kernels) == 1:
        axs = [axs]

    x = np.arange(len(matrices))
    width = 0.25

    for idx, kernel in enumerate(kernels):
        ax = axs[idx]
        df_k = df[df['Kernel_Clean'] == kernel]
        
        y_1 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 1)][metric_column].mean() for m in matrices]
        y_2 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 2)][metric_column].mean() for m in matrices]
        y_4 = [df_k[(df_k['Matrix'] == m) & (df_k['GPUs'] == 4)][metric_column].mean() for m in matrices]

        y_1 = [0 if np.isnan(v) else v for v in y_1]
        y_2 = [0 if np.isnan(v) else v for v in y_2]
        y_4 = [0 if np.isnan(v) else v for v in y_4]

        ax.bar(x - width, y_1, width, label='1 GPU', color=STRONG_COLORS[1], edgecolor='black', linewidth=0.3)
        ax.bar(x, y_2, width, label='2 GPUs', color=STRONG_COLORS[2], edgecolor='black', linewidth=0.3)
        ax.bar(x + width, y_4, width, label='4 GPUs', color=STRONG_COLORS[4], edgecolor='black', linewidth=0.3)

        ax.set_title(f"Format / Kernel: {kernel}", pad=2, fontweight='bold')
        ax.set_ylabel(y_label)
        
        if is_log:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=15))
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=15))
            
            formatter = ticker.LogFormatterMathtext(labelOnlyBase=False)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(formatter)
            
            ax.grid(True, which='both', linestyle=':', linewidth=0.3, alpha=0.6)
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        ax.set_xlabel("Matrix File Name (.mtx)", labelpad=1)
        ax.set_xticks(x)
        ax.set_xticklabels(matrices, rotation=11, ha='right', fontsize=7.5)

    fig.suptitle(title, y=0.98, fontweight='bold', fontsize=11)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(PLOTS_DIR, filename), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# WEAK SCALING PLOTS (Ultratight, Ultra-Compressed Layout)
# =========================================================================
def generate_weak_plots(df):
    df = df.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    kernels = sorted(df['Kernel_Clean'].unique())
    families = sorted(df['Matrix Family'].unique())
    
    fig, axs = plt.subplots(len(kernels), 2, figsize=(10, 1.5 * len(kernels)), sharex=True)
    if len(kernels) == 1:
        axs = np.array([axs])

    x = np.arange(len(families))
    width = 0.25

    for idx, kernel in enumerate(kernels):
        df_k = df[df['Kernel_Clean'] == kernel]
        
        ws_1 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 1)]['Weak Speedup'].mean() for f in families]
        ws_2 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 2)]['Weak Speedup'].mean() for f in families]
        ws_4 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 4)]['Weak Speedup'].mean() for f in families]
        
        we_1 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 1)]['Weak Efficiency'].mean() for f in families]
        we_2 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 2)]['Weak Efficiency'].mean() for f in families]
        we_4 = [df_k[(df_k['Matrix Family'] == f) & (df_k['GPUs'] == 4)]['Weak Efficiency'].mean() for f in families]

        # Weak Speedup Column
        ax1 = axs[idx, 0]
        ax1.bar(x - width, ws_1, width, label='1 GPU', color=WEAK_COLORS[1], edgecolor='black', linewidth=0.2)
        ax1.bar(x, ws_2, width, label='2 GPUs', color=WEAK_COLORS[2], edgecolor='black', linewidth=0.2)
        ax1.bar(x + width, ws_4, width, label='4 GPUs', color=WEAK_COLORS[4], edgecolor='black', linewidth=0.2)
        ax1.set_title(f"Weak Speedup - {kernel}", fontweight='bold', pad=1, fontsize=9)
        ax1.set_ylabel("Speedup Factor")
        ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax1.set_xmargin(0.05)

        # Weak Efficiency Column
        ax2 = axs[idx, 1]
        ax2.bar(x - width, we_1, width, label='1 GPU', color=WEAK_COLORS[1], edgecolor='black', linewidth=0.2)
        ax2.bar(x, we_2, width, label='2 GPUs', color=WEAK_COLORS[2], edgecolor='black', linewidth=0.2)
        ax2.bar(x + width, we_4, width, label='4 GPUs', color=WEAK_COLORS[4], edgecolor='black', linewidth=0.2)
        ax2.set_title(f"Weak Efficiency - {kernel}", fontweight='bold', pad=1, fontsize=9)
        ax2.set_ylabel("Efficiency")
        ax2.set_ylim(0, 1.2)
        ax2.axhline(1.0, color='red', linestyle='--', alpha=0.5, linewidth=0.8)
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax2.set_xmargin(0.05)

    axs[-1, 0].set_xlabel("Synthetic Matrix Family", labelpad=2)
    axs[-1, 1].set_xlabel("Synthetic Matrix Family", labelpad=2)
    axs[-1, 0].set_xticks(x)
    axs[-1, 0].set_xticklabels(families, rotation=12, fontsize=7)
    axs[-1, 1].set_xticks(x)
    axs[-1, 1].set_xticklabels(families, rotation=12, fontsize=7)

    fig.suptitle("Weak Scaling Profile per Individual Synthetic Matrix Family", y=0.98, fontweight='bold', fontsize=10)
    
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(PLOTS_DIR, "weak_individual_synthetic.pdf"), bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# STRUCTURED SUMMARY TABLES AND LATEX GENERATOR
# =========================================================================
def generate_summary_tables_and_latex(df_strong):
    df = df_strong.copy()
    df['Kernel_Clean'] = df['Kernel'].apply(clean_kernel_name)
    
    print("\n" + "="*80)
    print(" GENERATING SUMMARY TABLES AND RIGID LATEX EXPORTS ")
    print("="*80)

    perf_summary = df.groupby(['Kernel_Clean', 'GPUs'])[['Speedup', 'Efficiency']].mean().reset_index()
    table1 = perf_summary.pivot(index='Kernel_Clean', columns='GPUs', values=['Speedup', 'Efficiency'])
    table1.columns = [f"{v}_{c}G" for v, c in table1.columns]
    table1 = table1.round(4)
    
    output_t1 = os.path.join(TABLES_DIR, "strong_report_summary_efficiency_speedup.csv")
    table1.to_csv(output_t1)
    
    print("\n--- LATEX CODE READY FOR TABLE 1 ---")
    print(r"""\begin{table}[h!]
\centering
\caption{Strong Scaling Analysis: Mean Speedup and Parallel Efficiency per Kernel across Real Matrices.}
\label{tab:efficiency_summary}
\begin{tabular}{lcccccc}
\hline
\textbf{Format / Kernel} & \textbf{Sp. 1G} & \textbf{Sp. 2G} & \textbf{Sp. 4G} & \textbf{Eff. 1G} & \textbf{Eff. 2G} & \textbf{Eff. 4G} \\ \hline""")
    for k in table1.index:
        r = table1.loc[k]
        print(f"{k} & {r['Speedup_1G']:.4f} & {r['Speedup_2G']:.4f} & {r['Speedup_4G']:.4f} & {r['Efficiency_1G']:.4f} & {r['Efficiency_2G']:.4f} & {r['Efficiency_4G']:.4f} \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")

    df_multi = df[df['GPUs'].isin([2, 4]) & (df['Kernel_Clean'] != 'baseline')].copy()
    df_multi['Comp_Pct'] = (df_multi['Comp Time (s)'] / df_multi['Total Time (s)']) * 100
    df_multi['Comm_Pct'] = (df_multi['Comm Time (s)'] / df_multi['Total Time (s)']) * 100
    
    breakdown = df_multi.groupby(['Kernel_Clean', 'GPUs'])[['Comp_Pct', 'Comm_Pct']].mean().reset_index()
    breakdown = breakdown.round(2)
    
    output_t2 = os.path.join(TABLES_DIR, "strong_report_summary_comm_comp_breakdown.csv")
    breakdown.to_csv(output_t2, index=False)
    
    print("\n--- LATEX CODE READY FOR TABLE 2 ---")
    print(r"""\begin{table}[h!]
\centering
\caption{Execution Cost Breakdown: Local CUDA Compute vs MPI Communication Overhead.}
\label{tab:breakdown_summary}
\begin{tabular}{lcci}
\hline
\textbf{Kernel} & \textbf{GPU Config} & \textbf{\% Pure CUDA Compute} & \textbf{\% MPI Ghost Comm} \\ \hline""")
    for idx, row in breakdown.iterrows():
        # SOLUZIONE ERRORE 1: Inserito '\\%' al posto di '\%' per prevenire il SyntaxWarning di Python 3.12+
        print(f"{row['Kernel_Clean']} & {int(row['GPUs'])}G & {row['Comp_Pct']:.2f}\\% & {row['Comm_Pct']:.2f}\\% \\\\")
    print(r"""\hline
\end{tabular}
\end{table}""")

    print("\n--- LATEX CODE READY FOR TABLE 3 ---")
    print(r"""\begin{table}[h!]
\centering
\caption{Theoretical Model of Local Memory Allocation Footprint per Single Hardware Rank (GPU).}
\label{tab:memory_footprint}
\begin{tabular}{lcccc}
\hline
\textbf{Hardware Setup} & \textbf{Matrix $A$ Allocation} & \textbf{Vector $x$ Allocation} & \textbf{MPI Ghost Cells} & \textbf{Relative Local Load} \\ \hline
1 GPU (Rank 0)   & 100\% of $nnz$ elements & 100\% of components & 0 Bytes (Disabled) & 100.0\% (Full Load) \\
2 GPUs (Rank 0-1) & $\sim$50\% of $nnz$ elements & $\sim$50\% of components & Interface Buffers  & $\sim$50.0\% + Buffer Overhead \\
4 GPUs (Rank 0-3) & $\sim$25\% of $nnz$ elements & $\sim$25\% of components & Extended Interface Buffers & $\sim$25.0\% + Buffer Overhead \\ \hline
\end{tabular}
\end{table}""")
    print("="*80 + "\n")

# =========================================================================
# MAIN ORCHESTRATOR
# =========================================================================
def main():
    if os.path.exists(STRONG_CSV):
        df_strong = pd.read_csv(STRONG_CSV)
        print("[1/3] Generating mandatory quantitative strong scaling plots...")
        generate_essential_strong_plots(df_strong, 'Total Time (s)', 'Time (seconds)', 
                                        'Execution Time per SpMV (Individual Real Matrices)', 
                                        'strong_individual_time.pdf', is_log=True)
        generate_essential_strong_plots(df_strong, 'GFLOPS', 'Throughput (GFLOPS)', 
                                        'Computational Throughput (GFLOPS) per Individual Real Matrix', 
                                        'strong_individual_gflops.pdf', is_log=False)
        print("[2/3] Processing summary tables and LaTeX code...")
        generate_summary_tables_and_latex(df_strong)
    else:
        print(f"Error: {STRONG_CSV} not found.")

    if os.path.exists(WEAK_CSV):
        print("[3/3] Weak scaling report detected! Generating synthetic family plots...")
        df_weak = pd.read_csv(WEAK_CSV)
        generate_weak_plots(df_weak)
    else:
        print(f"Error: {WEAK_CSV} not found.")
        
    print(f"\n[Execution Completed Successfully! Plots compressed and spacing optimized.]")

if __name__ == "__main__":
    main()