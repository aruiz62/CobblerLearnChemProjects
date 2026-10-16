
import pandas as pd
from pathlib import Path

# Locate the master dataset
folder = Path(__file__).resolve().parent

df = pd.read_csv(
    folder / "AR_BLA_Master_Dataset.csv"
)

# Extract DTXSID numbers from the EPA URLs
df["chemical_id"] = (
    df["dtxsid"]
    .astype(str)
    .str.extract(r"(DTXSID\d+)", expand=False)
)

# Keep unique, valid chemical IDs
chemical_ids = (
    df["chemical_id"]
    .dropna()
    .drop_duplicates()
)

# Save identifiers as a text file
output = folder / "Chemical_IDs.txt"

chemical_ids.to_csv(
    output,
    index=False,
    header=False
)

print("Total chemicals:", len(df))
print("Unique valid DTXSIDs:", len(chemical_ids))
print("\nFirst 10 chemical IDs:")
print(chemical_ids.head(10).to_string(index=False))

print("\nChemical IDs saved successfully!")
