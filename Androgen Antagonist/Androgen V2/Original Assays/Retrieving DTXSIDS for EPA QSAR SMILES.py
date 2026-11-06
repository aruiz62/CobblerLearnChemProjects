
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

# Find our original antagonist assay
ratio_file = next(folder.glob("*Antagonist_ratio*.csv"))

# Load the dataset
df = pd.read_csv(ratio_file, low_memory=False)

# Extract chemical identifiers
ids = df[["DTXSID"]].copy()
ids["DTXSID"] = (
    ids["DTXSID"]
    .astype(str)
    .str.extract(r"(DTXSID\d+)", expand=False)
)

# Remove missing and duplicate identifiers
ids = ids.dropna().drop_duplicates()

# Save a list for EPA
output_file = folder / "DTXSID_for_EPA.csv"
ids.to_csv(output_file, index=False)

print("Original chemicals:", len(df))
print("Unique chemical identifiers:", len(ids))
print("Missing identifiers:", df["DTXSID"].isna().sum())
print("Duplicate identifiers:", df["DTXSID"].duplicated().sum())

print("\nSaved:", output_file.name)
