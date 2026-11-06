
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

# Find the downloaded EPA CSV
csv_files = list(folder.glob("*.csv"))

if len(csv_files) != 1:
    raise ValueError(
        f"Expected one CSV in this folder, found {len(csv_files)}. "
        "Keep only the downloaded EPA CSV here."
    )

df = pd.read_csv(csv_files[0], low_memory=False)

print("===== EPA SMILES DATA =====")
print("File:", csv_files[0].name)
print("Total rows:", len(df))

print("\nColumn names:")
for col in df.columns:
    print("-", col)

print("\nFirst 5 chemicals:")
print(df.head().to_string(index=False))

# Check chemical identifiers
id_col = next(
    (c for c in df.columns if c.strip().upper() == "DTXSID"),
    None
)

if id_col:
    ids = df[id_col].astype("string").str.extract(
        r"(DTXSID\d+)", expand=False
    )
    print("\nUnique DTXSIDs:", ids.nunique())
    print("Missing DTXSIDs:", ids.isna().sum())

# Check SMILES columns
smiles_cols = [
    c for c in df.columns
    if "SMILES" in c.upper()
]

print("\n===== SMILES CHECK =====")

for col in smiles_cols:
    values = df[col].astype("string").str.strip()
    missing = values.isna() | values.eq("")
    print(f"{col}:")
    print("  Present:", (~missing).sum())
    print("  Missing:", missing.sum())

if not smiles_cols:
    print("No SMILES columns found!")
