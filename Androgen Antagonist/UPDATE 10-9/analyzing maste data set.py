import pandas as pd

# Load the master dataset
df = pd.read_csv("AR_BLA_Master_Dataset.csv")

print("TOTAL CHEMICALS:", len(df))

# Find relevant assay columns
print("\nASSAY RESULT COLUMNS:")
for col in df.columns:
    if any(x in col.lower() for x in
           ["hit_call", "ac50", "viability"]):
        print(col)

# Check activity results
for col in df.columns:
    if "hit_call" in col.lower():
        print("\n", col)
        print(df[col].value_counts(dropna=False))

        print("Missing values:", df[col].isna().sum())
