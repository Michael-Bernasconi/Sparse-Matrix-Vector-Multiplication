import re
import csv
import os

# Input and output file names
log_file_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\results\benchmark\flops-bw\report_flops-bw-cpu-gpu.log'
csv_file_path = 'benchmark_results.csv'

# Regular expressions to extract the exact data from the text
kernel_pattern = re.compile(r'^\[(.*?)\]')
matrix_info_pattern = re.compile(r'Matrix\s*:\s*.*\/(.*\.mtx)\s*\((\d+)\s*x\s*(\d+),\s*nnz:\s*(\d+)\)')
time_pattern = re.compile(r'Avg Time:\s*([0-9eE\.\-\+]+)\s*s')
gflops_pattern = re.compile(r'GFLOPS\s*:\s*([0-9\.]+)')
bw_pattern = re.compile(r'BW\s*:\s*([0-9\.]+)\s*GB/s')
check_pattern = re.compile(r'Check\s*:\s*([0-9\.\-]+)')

# List to save all extracted rows
parsed_data = []

# Temporary variable to keep track of the current block
current_record = {}

if not os.path.exists(log_file_path):
    print(f"Error: The file '{log_file_path}' was not found.")
    exit(1)

print(f"Reading file {log_file_path}...")

with open(log_file_path, 'r') as file:
    for line in file:
        line = line.strip()
        
        # 1. Search for the Kernel name (e.g., [cpu-SpMV-CSR])
        k_match = kernel_pattern.match(line)
        if k_match:
            current_record['Kernel'] = k_match.group(1)
            continue
            
        # 2. Search for Matrix info (Name, Rows, Cols, NNZ)
        m_match = matrix_info_pattern.search(line)
        if m_match:
            current_record['Matrix'] = m_match.group(1)
            current_record['Rows'] = int(m_match.group(2))
            current_record['Cols'] = int(m_match.group(3))
            current_record['NNZ'] = int(m_match.group(4))
            # Calculate the average non-zeros per row (great for your report!)
            current_record['NNZ_per_Row_Avg'] = round(current_record['NNZ'] / current_record['Rows'], 2)
            continue
            
        # 3. Search for the Average Time
        t_match = time_pattern.search(line)
        if t_match:
            current_record['Avg_Time_s'] = float(t_match.group(1))
            continue
            
        # 4. Search for GFLOPS
        g_match = gflops_pattern.search(line)
        if g_match:
            current_record['GFLOPS'] = float(g_match.group(1))
            continue
            
        # 5. Search for Bandwidth
        b_match = bw_pattern.search(line)
        if b_match:
            current_record['BW_GBs'] = float(b_match.group(1))
            continue
            
        # 6. Search for the Check value (End of block)
        c_match = check_pattern.search(line)
        if c_match:
            current_record['Check_Value'] = float(c_match.group(1))
            
            # Since "Check" is the last line of each useful block,
            # we can now save the entire record and clear the dictionary
            parsed_data.append(current_record.copy())
            current_record.clear()

# Write the data to the CSV file
if parsed_data:
    # Define the column order in the CSV
    fieldnames = [
        'Matrix', 'Rows', 'Cols', 'NNZ', 'NNZ_per_Row_Avg', 
        'Kernel', 'Avg_Time_s', 'GFLOPS', 'BW_GBs', 'Check_Value'
    ]
    
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in parsed_data:
            writer.writerow(row)
            
    print(f"Success! Extracted {len(parsed_data)} records.")
    print(f"Data has been saved to: {csv_file_path}")
else:
    print("No valid data found in the log. Check the text file format.")