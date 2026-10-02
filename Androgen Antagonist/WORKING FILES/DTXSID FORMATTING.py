import pandas as pd

# Load the PubChem CSV file
file = "AR BLA ANTAGONIST VIABILITY.csv"

# Skip PubChem's extra information rows
df = pd.read_csv(file, skiprows=[1, 2, 3, 4])

# Show every column name
print("\nCOLUMN NAMES:")
for col in df.columns:
    print(col)

# Show the first 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())
