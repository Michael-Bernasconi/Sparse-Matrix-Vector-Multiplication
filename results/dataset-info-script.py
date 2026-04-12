import os
import csv
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

# --- Configuration ---
# Path to the directory containing .mtx files
log_file_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\data'
# Output CSV file name
csv_file_path = 'info_dataset_results.csv'

# Filter directory to list only Matrix Market (.mtx) files
files = [f for f in os.listdir(log_file_path) if f.endswith('.mtx')]

# Initialize CSV writing
with open(csv_file_path, mode='w', newline='') as csvfile:
    # Define the structure of the output CSV
    fieldnames = ['Matrix Name', 'Rows', 'Columns', 'NNZ', 'Average NNZ for rows', 'Dev Std Rows']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for file_name in files:
        full_path = os.path.join(log_file_path, file_name)
        print(f"Processing: {file_name}...")
        
        try:
            # 1. Load the matrix
            # mmread automatically handles coordinate summation for duplicate entries
            matrix = mmread(full_path)
            
            # 2. Convert to CSR (Compressed Sparse Row) format
            # This is necessary for efficient row-wise operations and statistical calculations
            if not isinstance(matrix, csr_matrix):
                matrix = matrix.tocsr()

            # 3. Data Cleaning: Remove explicit zeros
            # Some datasets include entries like (row, col, 0.0). 
            # We remove them to match the official NNZ (Non-Zero) counts on websites like SuiteSparse.
            matrix.eliminate_zeros()
            
            # 4. Extract Non-Zero counts per row
            # getnnz(axis=1) returns an array where each element is the number of NNZ in that row
            nnz_per_row = matrix.getnnz(axis=1)
            
            # 5. Statistical Calculations
            # Calculate the mean (Average NNZ per row)
            avg_nnz = np.mean(nnz_per_row)
            # Calculate the Standard Deviation (measuring row imbalance/sparsity distribution)
            std_nnz = np.std(nnz_per_row)
            
            # 6. Save results to CSV
            writer.writerow({
                'Matrix Name': file_name.replace('.mtx', ''),
                'Rows': matrix.shape[0],
                'Columns': matrix.shape[1],
                'NNZ': matrix.nnz, # Total count of structural non-zeros
                'Average NNZ for rows': round(avg_nnz, 2),
                'Dev Std Rows': round(std_nnz, 2)
            })
            
        except Exception as e:
            # Error handling for corrupted files or memory issues
            print(f"Error processing {file_name}: {e}")

print(f"Completed! Results saved to: {csv_file_path}")