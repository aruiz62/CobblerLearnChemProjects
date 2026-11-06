
import pandas as pd
from pathlib import Path

# Find the folder where this Python script is saved
folder = Path(__file__).resolve().parent

# Find all CSV files in this folder
csv_files = list(folder.glob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print(file.name)

# Load each CSV and display its information
for file in csv_files:
    df = pd.read_csv(file, low_memory=False)

    print("\n==============================")
    print("FILE:", file.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
