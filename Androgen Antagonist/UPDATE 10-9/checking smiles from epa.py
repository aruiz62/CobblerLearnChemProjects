import pandas as pd

epa = pd.read_csv("SMILES.csv")

print("Columns:")
print(epa.columns.tolist())

print("\nFirst 5 rows:")
print(epa.head())