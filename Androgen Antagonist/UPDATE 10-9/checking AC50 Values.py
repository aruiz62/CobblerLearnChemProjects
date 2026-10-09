
import pandas as pd
from pathlib import Path

folder = Path(__file__).resolve().parent

df = pd.read_csv(folder / "AR_BLA_Master_Dataset.csv")

print("AC50 AND CONCENTRATION COLUMNS:")

for col in df.columns:
    if any(term in col.lower() for term in
           ["ac50", "acc", "potency", "concentration"]):
        print(col)
        print("Non-missing values:", df[col].notna().sum())
        print()
