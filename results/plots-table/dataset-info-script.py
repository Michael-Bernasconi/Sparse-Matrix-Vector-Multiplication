import os
import csv
import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix

# Percorso della cartella contenente i file .mtx
log_file_path = r'C:\Users\micha\Documents\GitHub\Sparse-Matrix-Vector-Multiplication\data'
csv_file_path = 'info_dataset_results.csv'

# Lista tutti i file .mtx nella directory
files = [f for f in os.listdir(log_file_path) if f.endswith('.mtx')]

with open(csv_file_path, mode='w', newline='') as csvfile:
    fieldnames = ['Matrix Name', 'Rows', 'Columns', 'NNZ', 'Average NNZ for rows', 'Dev Std Rows']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for file_name in files:
        full_path = os.path.join(log_file_path, file_name)
        print(f"Processing: {file_name}...")
        
        try:
            # 1. Carica la matrice (gestisce automaticamente la somma dei duplicati)
            matrix = mmread(full_path)
            
            # 2. Converti in CSR per manipolazione efficiente
            if not isinstance(matrix, csr_matrix):
                matrix = matrix.tocsr()

            # 3. PULIZIA: Rimuove gli zeri espliciti (fondamentale per far coincidere i NNZ col sito)
            matrix.eliminate_zeros()
            
            # 4. Ottieni il numero di Non-Zero effettivi per riga
            nnz_per_row = matrix.getnnz(axis=1)
            
            # 5. Calcola Media e Deviazione Standard
            # Usiamo matrix.nnz (che ora è pulito) per il valore totale
            avg_nnz = np.mean(nnz_per_row)
            std_nnz = np.std(nnz_per_row)
            
            writer.writerow({
                'Matrix Name': file_name.replace('.mtx', ''),
                'Rows': matrix.shape[0],
                'Columns': matrix.shape[1],
                'NNZ': matrix.nnz, # Questo valore ora corrisponderà al sito
                'Average NNZ for rows': round(avg_nnz, 2),
                'Dev Std Rows': round(std_nnz, 2)
            })
            
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

print(f"Completed! Results saved to: {csv_file_path}")