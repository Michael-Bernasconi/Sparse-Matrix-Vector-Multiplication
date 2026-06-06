import os
import random

def generate_balanced(output_dir, filename, N, d):
    """Generates a perfectly balanced synthetic matrix with random float values."""
    total_nnz = N * d
    filepath = os.path.join(output_dir, filename)
    print(f"Generating BALANCED: {filepath} ({N}x{N}, nnz={total_nnz})...")
    
    with open(filepath, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {total_nnz}\n")
        for i in range(1, N + 1):
            for idx in range(d):
                j = ((i - 1 + idx) % N) + 1
                val = round(random.uniform(0.1, 10.0), 2)
                f.write(f"{i} {j} {val}\n")

def generate_imbalanced(output_dir, filename, N, d_avg):
    """Generates an imbalanced synthetic matrix (high variance) with random float values."""
    total_nnz = N * d_avg
    filepath = os.path.join(output_dir, filename)
    print(f"Generating IMBALANCED: {filepath} ({N}x{N}, nnz={total_nnz})...")
    
    dense_rows_count = max(1, int(N * 0.05))
    dense_nnz_total = int(total_nnz * 0.80)
    nnz_per_dense_row = dense_nnz_total // dense_rows_count
    
    remaining_nnz = total_nnz - (nnz_per_dense_row * dense_rows_count)
    remaining_rows_count = N - dense_rows_count
    nnz_per_light_row = max(1, remaining_nnz // remaining_rows_count) if remaining_rows_count > 0 else 1
    
    actual_nnz = (dense_rows_count * nnz_per_dense_row) + (remaining_rows_count * nnz_per_light_row)
    
    with open(filepath, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {actual_nnz}\n")
        
        # Hyper-Dense Rows
        for i in range(1, dense_rows_count + 1):
            cols = random.sample(range(1, N + 1), min(N, nnz_per_dense_row))
            for j in sorted(cols):
                val = round(random.uniform(0.1, 10.0), 2)
                f.write(f"{i} {j} {val}\n")
                
        # Sparse Rows
        for i in range(dense_rows_count + 1, N + 1):
            cols = random.sample(range(1, N + 1), min(N, nnz_per_light_row))
            for j in sorted(cols):
                val = round(random.uniform(0.1, 10.0), 2)
                f.write(f"{i} {j} {val}\n")

if __name__ == "__main__":
    # Target directory setup
    OUTPUT_DIR = "./data-synt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- WEAK SCALING CONFIGURATION ----
    N_BASE = 200000 
    D_AVG = 16 

    # 1 GPU
    generate_balanced(OUTPUT_DIR, "synth_balanced_1gpu.mtx", N_BASE, D_AVG)
    generate_imbalanced(OUTPUT_DIR, "synth_imbalanced_1gpu.mtx", N_BASE, D_AVG)

    # 2 GPU
    generate_balanced(OUTPUT_DIR, "synth_balanced_2gpu.mtx", N_BASE * 2, D_AVG)
    generate_imbalanced(OUTPUT_DIR, "synth_imbalanced_2gpu.mtx", N_BASE * 2, D_AVG)

    # 4 GPU
    generate_balanced(OUTPUT_DIR, "synth_balanced_4gpu.mtx", N_BASE * 4, D_AVG)
    generate_imbalanced(OUTPUT_DIR, "synth_imbalanced_4gpu.mtx", N_BASE * 4, D_AVG)

    print("Synthetic dataset successfully stored in ./data-synt/")