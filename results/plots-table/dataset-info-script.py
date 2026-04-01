import os
import csv
import numpy as np
from scipy.io import mmread
from scipy.sparse import issparse, csr_matrix

# Input and output file paths
log_file_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\data'
csv_file_path = 'info_dataset_results.csv'

# List all .mtx files in the directory
files = [f for f in os.listdir(log_file_path) if f.endswith('.mtx')]

with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Matrix Name', 'Rows', 'Columns', 'NNZ', 'Average NNZ for rows', 'Dev Std Rows']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for file_name in files:
        full_path = os.path.join(log_file_path, file_name)
        print(f"Processing: {file_name}...")
        
        try:
            # Load the matrix from Matrix Market format
            matrix = mmread(full_path)
            
            # Ensure the matrix is in CSR format for fast row-wise access
            if not isinstance(matrix, csr_matrix):
                matrix = matrix.tocsr()

            # Get the number of Non-Zeros (NNZ) for each row
            nnz_per_row = matrix.getnnz(axis=1)
            
            # Calculate Mean and Standard Deviation
            avg_nnz = np.mean(nnz_per_row)
            std_nnz = np.std(nnz_per_row)
            
            writer.writerow({
                'Matrix Name': file_name.replace('.mtx', ''),
                'Rows': matrix.shape[0],
                'Columns': matrix.shape[1],
                'NNZ': matrix.nnz,
                'Average NNZ for rows': round(avg_nnz, 2),
                'Dev Std Rows': round(std_nnz, 2)
            })
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print(f"Completed! Results saved to: {csv_file_path}")