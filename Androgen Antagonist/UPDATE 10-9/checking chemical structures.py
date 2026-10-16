
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

df = pd.read_csv(
    folder / "AR_BLA_Master_Dataset.csv"
)

print("TOTAL CHEMICALS:", len(df))

print("\nCHEMICAL IDENTIFIERS AND STRUCTURES:")

keywords = [
    "smiles",
    "inchi",
    "cas",
    "dtxsid",
    "cid",
    "name",
    "structure"
]

for col in df.columns:
    if any(word in col.lower() for word in keywords):
        print(col, "| Missing:", df[col].isna().sum())

print("\nFIRST 5 ROWS:")
print(df.head(5).to_string())
