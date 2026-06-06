import os
import random

def generate_like_asic(output_dir, filename, N, d_avg=3):
    """
    Generates a synthetic matrix mirroring ASIC_680ks:
    Highly sparse, asymmetric/irregular, but structurally uniform across rows 
    ensuring an excellent 1D load balancing.
    """
    filepath = os.path.join(output_dir, filename)
    print(f"Generating ASIC-like matrix: {filepath} ({N}x{N})...")
    
    # We enforce that every row has roughly d_avg non-zeros to maintain 1D load balance,
    # but columns are picked completely at random to guarantee structural irregularity.
    total_nnz = 0
    row_entries = []
    
    for i in range(1, N + 1):
        # Small variance per row (e.g., 2, 3 or 4 elements if d_avg=3)
        nnz_in_row = random.randint(max(1, d_avg - 1), d_avg + 1)
        # Random column indices to make it highly asymmetric and irregular
        cols = random.sample(range(1, N + 1), nnz_in_row)
        row_entries.append((i, sorted(cols)))
        total_nnz += nnz_in_row

    with open(filepath, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {total_nnz}\n")
        for row, cols in row_entries:
            for col in cols:
                val = round(random.uniform(0.1, 10.0), 2)
                f.write(f"{row} {col} {val}\n")

def generate_like_boyd2(output_dir, filename, N, d_avg=3):
    """
    Generates a synthetic matrix mirroring boyd2:
    Extreme load imbalance. A small fraction of rows are scattered across the matrix 
    and are hyper-dense, while the vast majority of rows are almost empty.
    """
    filepath = os.path.join(output_dir, filename)
    print(f"Generating BOYD2-like matrix: {filepath} ({N}x{N})...")
    
    # Calculate target NNZ based on the real boyd2 average density (~3.2)
    target_total_nnz = int(N * d_avg)
    
    # Extreme irregularity: 2% of total rows absorb ~75% of total NNZ
    dense_rows_count = max(1, int(N * 0.02))
    dense_nnz_total = int(target_total_nnz * 0.75)
    nnz_per_dense_row = max(5, dense_nnz_total // dense_rows_count)
    # Cap the density per row to avoid exceeding N columns
    nnz_per_dense_row = min(nnz_per_dense_row, N)
    
    # The remaining 98% of rows share the remaining 25% of non-zeros
    remaining_nnz = target_total_nnz - (dense_rows_count * nnz_per_dense_row)
    remaining_rows_count = N - dense_rows_count
    nnz_per_light_row = max(1, remaining_nnz // remaining_rows_count) if remaining_rows_count > 0 else 1

    # Randomly scatter the dense row indices throughout the whole matrix 
    # (instead of putting them all at the beginning, mimicking real structural noise)
    all_row_indices = list(range(1, N + 1))
    dense_row_indices = set(random.sample(all_row_indices, dense_rows_count))
    
    actual_nnz = 0
    row_entries = []
    
    for i in range(1, N + 1):
        if i in dense_row_indices:
            cols = random.sample(range(1, N + 1), nnz_per_dense_row)
            actual_nnz += nnz_per_dense_row
        else:
            cols = random.sample(range(1, N + 1), min(N, nnz_per_light_row))
            actual_nnz += len(cols)
        row_entries.append((i, sorted(cols)))

    with open(filepath, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{N} {N} {actual_nnz}\n")
        for row, cols in row_entries:
            for col in cols:
                val = round(random.uniform(0.1, 10.0), 2)
                f.write(f"{row} {col} {val}\n")

if __name__ == "__main__":
    OUTPUT_DIR = "./data-synt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- WEAK SCALING EXPERIMENTAL CONFIGURATION ----
    # Keeping the scaling factor requested by Deliverable 2 (N, 2N, 4N)
    N_BASE = 200000 
    D_AVG = 3  # Set to 3 to precisely capture the density of both ASIC (~2.48) and Boyd2 (~3.22)

    # 1 GPU Configurations
    generate_like_asic(OUTPUT_DIR, "synth_asic_1gpu.mtx", N_BASE, D_AVG)
    generate_like_boyd2(OUTPUT_DIR, "synth_boyd2_1gpu.mtx", N_BASE, D_AVG)

    # 2 GPUs Configurations (Matrix size doubles)
    generate_like_asic(OUTPUT_DIR, "synth_asic_2gpu.mtx", N_BASE * 2, D_AVG)
    generate_like_boyd2(OUTPUT_DIR, "synth_boyd2_2gpu.mtx", N_BASE * 2, D_AVG)

    # 4 GPUs Configurations (Matrix size quadrupled)
    generate_like_asic(OUTPUT_DIR, "synth_asic_4gpu.mtx", N_BASE * 4, D_AVG)
    generate_like_boyd2(OUTPUT_DIR, "synth_boyd2_4gpu.mtx", N_BASE * 4, D_AVG)

    print("\nSynthetic dataset resembling ASIC and BOYD2 successfully stored in ./data-synt/")