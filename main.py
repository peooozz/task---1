import pandas as pd
import numpy as np

print("✓ Pandas and NumPy are working!")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# Try to load your CSV
# Replace 'your_file.csv' with your actual filename
csv_file = 'Dataset.csv'  
try:
    df = pd.read_csv(csv_file)
    print(f"\n✓ Successfully loaded {csv_file}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn names:\n{df.columns.tolist()}")
except:
    print(f"\n✗ Could not find {csv_file}")
    print("Make sure the file is in the same folder!")